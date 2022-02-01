// Copyright 2015 The ALMA Project Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

/// @file
/// Writes the distribution in files.

#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmakevars.hpp>
#include <constants.hpp>
#include <utilities.hpp>
#include <structures.hpp>
#include <bulk_hdf5.hpp>
#include <isotopic_scattering.hpp>
#include <io_utils.hpp>
#include <msgpack.hpp>
#include <geometry_2d.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>


///Some alias
using gridData = std::unordered_map<std::string,
        std::unique_ptr<alma::Gamma_grid>>;
using cellData = std::unordered_map<std::string,
        std::unique_ptr<alma::Crystal_structure>>;

/// Probability density function for a Gamma distribution.
///
/// @param[in] k - shape parameter
/// @param[in] theta - scale parameter
/// @param[in] x - point at which to evaluate the function
/// @return the value of the pdf
inline double gamma_pdf(double k, double theta, double x) {
    return boost::math::gamma_p_derivative(k, x / theta) / theta;
}
///To store geometric and input data:
///Parameters of input file
struct input_parameters {
    
    ///Geometry
    std::vector<alma::geometry_2d>
        system;
    ///Material data
    cellData system_cell;
    gridData system_grid;

    ///Vector of thickness
    std::vector<double>
        thicknesses;
    ///Where to print data
    std::vector<std::size_t> boxesID;
    std::vector<double>      dumptimes;    
    ///nbins
    std::size_t nbins;
};

///To read input file:
///This reads the input file
///@param[in] filename - input filename with path
///@param[in] world    - mpi communicator
///@return structure with input parameters
input_parameters 
process_input(std::string& filename,
              boost::mpi::communicator& world
){
    // Create empty property tree object
    boost::property_tree::ptree tree;

    // Parse XML input file into the tree
    boost::property_tree::read_xml(filename, tree);
    
    input_parameters inpars;
    ///Map to store thickness
    std::map<std::string,double> zmap;
    
    for (const auto& v : tree.get_child("beRTAMC2D")) {
        ///Reading geometry
        if (v.first == "geometry") {
            std::string gfname = 
                alma::parseXMLfield<std::string>(v, "file");
            inpars.system = 
                alma::read_geometry_XML(gfname);
        }
        if (v.first == "material") {
            std::string name = 
                alma::parseXMLfield<std::string>(v, "name");
            std::string hdf5file = 
                alma::parseXMLfield<std::string>(v, "database");
            double zsize = 
                alma::parseXMLfield<double>(v, "thickness");
            auto data = 
                alma::load_bulk_hdf5(hdf5file.c_str(), world);
            inpars.system_cell[name] = 
                std::move(std::get<1>(data));
            inpars.system_grid[name] = 
                std::move(std::get<3>(data));
            
            zmap[name] = zsize;
        }
        
        if (v.first == "spectral") {
            for (auto it = v.second.begin(); it != v.second.end(); it++) {
                if (it->first == "resolution") {
                    inpars.nbins =
                        alma::parseXMLfield<std::size_t>(*it, "ticks");
                }
                if (it->first == "location") {
                    inpars.boxesID.push_back(
                        alma::parseXMLfield<std::size_t>(*it, "bin"));
                }
                if (it->first == "time") {
                    inpars.dumptimes.push_back(
                        alma::parseXMLfield<double>(*it, "t"));
                }
            }
        }
        
    }
    
    if (inpars.system_cell.size()>1){
        std::cout << "This is currently not supported\n";
        world.abort(1);
    }
    
    /// Fill thicknesses
    for (std::size_t i=0;i<inpars.system.size();i++) {
        auto mat = inpars.system[i].material;
        inpars.thicknesses.push_back(
            zmap[mat]);
    }
    
    /// Order the times
    std::sort(inpars.dumptimes.begin(),inpars.dumptimes.end());
    
    return inpars;
}




class data {
public:
    double time;
    double Eeff;
    std::vector<double> temperatures;
    std::vector<Eigen::VectorXi> histograms;
    std::vector<std::size_t> nbands;
    std::vector<std::ofstream> outf;
    std::vector<std::ofstream> outfvx;
    std::vector<std::ofstream> outfvy;
    std::vector<std::ofstream> outfd;
    
    data(input_parameters& inpars){
        temperatures.resize(inpars.system.size());
        histograms.resize(temperatures.size());
        nbands.resize(temperatures.size(),0);
        outf.resize(nbands.size());
        outfvx.resize(nbands.size());
        outfvy.resize(nbands.size());
        outfd.resize(nbands.size());
        
        for (auto s : inpars.system) {
            
            auto id = s.get_id();
            
            if (std::find(inpars.boxesID.begin(),
                inpars.boxesID.end(),id)==inpars.boxesID.end())
                continue;
            
            outf[id].open("deltaT_omega_"+std::to_string(id)+".csv");
            outfvx[id].open("jx_omega_"+std::to_string(id)+".csv");
            outfvy[id].open("jy_omega_"+std::to_string(id)+".csv");
            outfd[id].open("fd_"+std::to_string(id)+".csv");
            
            auto mat = inpars.system[id].material;
            
            auto nqpoints = inpars.system_grid[mat]->nqpoints;
            
            auto nbands_   = inpars.system_grid[mat
                ]->get_spectrum_at_q(0).omega.size();
            
            histograms[id].resize(nbands_*nqpoints);
            histograms[id].setZero();
            nbands[id] = nbands_;
        }
    }
    
    ~data(){
        for (auto& o : outf) {
            if(o.is_open())
                o.close();
        }
        for (auto& o : outfvx) {
            if(o.is_open())
                o.close();
        }
        for (auto& o : outfvy) {
            if(o.is_open())
                o.close();
        }
        for (auto& o : outfd) {
            if(o.is_open())
                o.close();
        }
        
    }
    
  // this function is looks like de-serializer, taking an msgpack object
  // and extracting data from it to the current class fields
  void msgpack_unpack(msgpack::object o) {
    
    // check if received structure is an array
    if(o.type != msgpack::type::ARRAY) { throw msgpack::type_error(); }
    
    o.via.array.ptr[0].convert(time);
    o.via.array.ptr[1].convert(Eeff);
    
    std::cout << "time " << time << std::endl;
    std::cout << "Eeff " << Eeff << std::endl;
    
    // extract value of second array entry which is array itself:
    for (std::size_t i = 0; i < temperatures.size() ; i++) {
      o.via.array.ptr[2].via.array.ptr[i].convert((temperatures[i]));
    }
    for (std::size_t i = 0; i < temperatures.size(); i++) {
        if (nbands[i] == 0)
            continue;
        std::size_t nelements;
        o.via.array.ptr[3+2*i].convert(nelements);
        
        std::cout << "nelements " << nelements << std::endl;
        
        histograms[i].setZero();
        
        for (std::size_t j = 0; j < nelements/2; j++) {
            std::size_t imode;
            o.via.array.ptr[4+2*i].via.array.ptr[2*j].convert(imode);
            o.via.array.ptr[4+2*i].via.array.ptr[2*j+1].convert((histograms[i](imode)));
        }
    }
    
  }
  
  // destination of this function is unknown - i've never ran into scenary
  // what it was called. some explaination/documentation needed.
  template <typename MSGPACK_OBJECT>
  void msgpack_object(MSGPACK_OBJECT* o, msgpack::zone* z) const { 

  }
  
  /// This prints the distribution
  ///@param[in] inpar - input parameters 
  double print_distribution(input_parameters& inpars){
      static std::map<std::string,Eigen::ArrayXXd> sigmas;  //Broadening
      static std::map<std::string,Eigen::ArrayXXd> vx;
      static std::map<std::string,Eigen::ArrayXXd> vy;
      static std::map<std::string,Eigen::MatrixXd> qpoints1stBZ; //Q points in the first BZ
      static std::map<std::string,double> Cvtot;      //specific heat of each material [J/(nm**3 * K)]
      static std::map<std::string,Eigen::VectorXd> mesh;
      static std::vector<bool> first(temperatures.size(),true);

      /// If not in list ignore
      if (std::find_if(inpars.dumptimes.begin(),inpars.dumptimes.end(),[&](double &t_){
          return alma::almost_equal(time,t_);		
      })==inpars.dumptimes.end()){
          return -1.0;
      }
      
      /// Filling tables
      
      if (sigmas.empty()) {
      
        for (std::size_t ibox = 0; ibox < temperatures.size(); ibox++) {
            if (nbands[ibox]==0) {
                continue;
            }
            //std::cout << "Printing " << ibox << std::endl;
            auto mat = inpars.system[ibox].material;
            if (sigmas.count(mat) !=0)
                    continue;
            auto tbox = temperatures[ibox];
            sigmas[mat].resize(nbands[ibox],inpars.system_grid[mat]->nqpoints);
            vx[mat].resize(nbands[ibox],inpars.system_grid[mat]->nqpoints);
            vy[mat].resize(nbands[ibox],inpars.system_grid[mat]->nqpoints);
            qpoints1stBZ[mat].resize(inpars.system_grid[mat]->nqpoints,2);
            sigmas[mat].setZero();
            vx[mat].setZero();
            vy[mat].setZero();
            qpoints1stBZ[mat].setZero();
            double maxf = 0.;
            
            double prefactor =
                alma::constants::kB / (inpars.system_cell[mat]->V * inpars.thicknesses[ibox]
                /inpars.system_cell[mat]->lattvec(2,2)) / inpars.system_grid[mat]->nqpoints;
            
            for (std::size_t iq = 0; iq < inpars.system_grid[mat]->nqpoints; ++iq) {
                    auto spectrum = inpars.system_grid[mat]->get_spectrum_at_q(iq);
                    
                    auto q1BZ = inpars.system_cell[mat]->map_to_firstbz(
                        inpars.system_grid[mat]->get_q(iq));
                    
                    qpoints1stBZ[mat](iq,0) = q1BZ.row(0).mean();
                    qpoints1stBZ[mat](iq,1) = q1BZ.row(1).mean();
                    
                    
                    auto maxomega_mode =
                        spectrum.omega.maxCoeff();
                    if (maxomega_mode > maxf)
                        maxf = maxomega_mode;
                    
                    for (std::size_t im = 0; im < nbands[ibox]; ++im) {
                        auto omega_ = spectrum.omega[im];
                        
                        if (alma::almost_equal(omega_,0.))
                            continue;
                        
                        sigmas[mat](im, iq) = inpars.system_grid[
                            mat]->base_sigma(spectrum.vg.col(im));
                        ///Get group velocities in m/s
                        vx[mat](im, iq) = 1.0e+3 * spectrum.vg(0,im);
                        vy[mat](im, iq) = 1.0e+3 * spectrum.vg(1,im);
                    }
            }
            // And refine them by removing outliers.
            auto percent = alma::calc_percentiles_log(sigmas[mat]);
            double lbound = std::exp(percent[0] - 1.5 * (percent[1] - percent[0]));
            sigmas[mat] = (sigmas[mat] < lbound).select(lbound, sigmas[mat]);
            
            mesh[mat] = Eigen::VectorXd::LinSpaced(inpars.nbins,0.,maxf);
            mesh[mat] = (mesh[mat].array() + mesh[mat](1)/2.0).matrix();
            
            /// Filling volumetric heat capacity
            Cvtot[mat] = 0.;
            
            for (std::size_t iq = 0; iq < inpars.system_grid[mat]->nqpoints; ++iq) {
                auto spectrum = inpars.system_grid[mat]->get_spectrum_at_q(iq);
                for (std::size_t im = 0; im < nbands[ibox]; ++im) {
                    auto omega_ = spectrum.omega[im];
                    
                    if (alma::almost_equal(0.,omega_))
                        continue;
                    
                    Cvtot[mat] += alma::bose_einstein_kernel(omega_, tbox);
                }
            }
            Cvtot[mat] *= prefactor;
        }
      }
      
      
      
      for (std::size_t ibox = 0; ibox < temperatures.size(); ibox++) {
          
          ///If not in print list ignore
          if (std::find(inpars.boxesID.begin(),
              inpars.boxesID.end(),ibox)==inpars.boxesID.end())
              continue;
          
          auto mat = inpars.system[ibox].material;
          if (inpars.system[ibox].reservoir
              or inpars.system[ibox].reservoir)
            continue;
          
          Eigen::VectorXd d(mesh[mat]), dvx(mesh[mat]), dvy(mesh[mat]);
          Eigen::VectorXd dfd(inpars.system_grid[mat]->nqpoints); 
          
          d.setZero();
          dvx.setZero();
          dvy.setZero();
          dfd.setZero();
          
          auto vol  = inpars.system[ibox].get_area()*inpars.thicknesses[ibox];
          double Ebox = 0.;
          
          for (std::size_t imode = 0; imode < static_cast<std::size_t>(
                histograms[ibox].rows()); imode++){
              auto value = histograms[ibox](imode);
              if (value == 0) {
                  continue;
              }
              auto ib = imode % nbands[ibox];
              auto iq = imode / nbands[ibox];
              auto omega = inpars.system_grid[mat]->get_spectrum_at_q(iq).omega(ib);
              dfd(iq) += value/omega;
              auto sigma = sigmas[mat](ib,iq);
              Eigen::VectorXd v = inpars.system_grid[mat]->get_spectrum_at_q(iq).vg.col(ib).matrix();
              Ebox  += value;
              ///If gamma they do not contribute
              if (alma::almost_equal(omega,0) or alma::almost_equal(sigma,0))
                  continue;
              for (int ii = 0; ii < d.rows(); ii++ ) {
                  double k = omega * omega / sigma;
                  double theta = sigma / omega;
                  double valati = value*gamma_pdf(k,theta,mesh[mat](ii));
                  d(ii)   += valati;
                  dvx(ii) += valati * 1.0e+3 * v(0);
                  dvy(ii) += valati * 1.0e+3 * v(1);
              }
          }
          
          ///Recover the distribution function
          dfd   *=   Eeff*(inpars.system_cell[mat]->V * inpars.thicknesses[ibox]
                 / inpars.system_cell[mat]->lattvec(2,2)) * 
                inpars.system_grid[mat]->nqpoints / (1.0e+12*vol*alma::constants::hbar);
          dvx      *= Eeff * 1.0e+27/vol  ;
          dvy      *= Eeff * 1.0e+27/vol  ;
          d        *= Eeff/(vol*Cvtot[mat]);
          Ebox     *= Eeff/vol;
          
          std::cout <<"Energy density of box " << ibox << " is " << Ebox << std::endl;
          std::cout <<"Specific heat " << Cvtot[mat] * 1.0e+27 << " J/(m**3 * K)" << std::endl; 
          
          auto &myo   = outf[ibox];
          auto &myox  = outfvx[ibox];
          auto &myoy  = outfvy[ibox];
          auto &myofd = outfd[ibox];
          
          if (first[ibox]) {
              myo << -1 << ',';
              myox << -1 << ',';
              myoy << -1 << ',';
              for (int ii = 0; ii < d.rows(); ii++ ) {
                  myo << mesh[mat](ii);
                  myox << mesh[mat](ii);
                  myoy << mesh[mat](ii);
                  if (ii != d.rows()-1) {
                    myo  << ',';
                    myox << ',';  
                    myoy << ',';  
                  }
              }
              myo << std::endl;
              myox << std::endl;
              myoy << std::endl;
              
              ///Print coordinates
              
              myofd << "# ";
              for (int ii = 0; ii < dfd.rows();ii++) {
                 myofd << qpoints1stBZ[mat](ii,0);
                 if (ii != dfd.rows()-1) { 
		    myofd << ",";
                 }
              }
              myofd << std::endl;
              myofd << "# ";
              for (int ii = 0; ii < dfd.rows();ii++) {
                  myofd << qpoints1stBZ[mat](ii,1);
                  if (ii != dfd.rows()-1) {
                    myofd << ",";
                 }
              }
              myofd << std::endl;
              
              
              first[ibox] = false;
          }
          
          myo  << time << ',';
          myox << time << ',';
          myoy << time << ',';
          myofd << time << ',';
          for (int ii = 0; ii < d.rows(); ii++ ) {
            myo  << d(ii);
            myox << dvx(ii);
            myoy << dvy(ii);
              
            if (ii != d.rows()-1) {
                myo  << ',';
                myox << ',';  
                myoy << ',';  
            }
          }
          
          for (int ii = 0; ii < dfd.rows();ii++) {
            myofd << dfd(ii);
            if (ii != dfd.rows()-1) {
                myofd << ',';
            }
          } 
         
          myo   << std::endl;
          myox  << std::endl;
          myoy  << std::endl;
          myofd << std::endl;
          
      }
    
    return time;
  }
  
    
};



int main(int argc, char** argv) {
    boost::mpi::environment env;
    boost::mpi::communicator world;
    
    if (world.size()!=1) {
        std::cout << "Error: dist_reader cannot be " 
                     "run in more than one MPI-process"
                    << std::endl;
        world.abort(1);
    }
    
    // Reference temperature at which to compute
    // heat capacities and scattering rates.

    // Path to the HDF5 file.
    std::string h5filename;
    std::string inputfilename;
    std::string msgpackfilename;
    std::string reader;

    if (argc != 3) {
        std::cout << "USAGE: dist_reader <input.xml> <histogram.msgpack.bin> "
                  << std::endl;
        world.abort(1);
    }

    inputfilename = std::string(argv[1]);
    msgpackfilename = std::string(argv[2]);

    std::cout << "***********************************" << std::endl;
    std::cout << "This is ALMA/dist_reader version " << ALMA_VERSION_MAJOR << "."
              << ALMA_VERSION_MINOR << std::endl;
    std::cout << "***********************************" << std::endl;

    ///Reading geometry and other things
    std::cout << "Reading " << inputfilename << std::endl;
    
    auto inpars = process_input(inputfilename,world);
    
    ///Processing:
    std::cout << "Init data storing" << std::endl;
    data simdata(inpars);
    
    std::ifstream histdat;
    
    boost::filesystem::path msgpack_path{msgpackfilename};
    
    if (!(boost::filesystem::exists(msgpack_path))) {
        std::cout << "ERROR:" << std::endl;
        std::cout << "msgpack file " << msgpack_path << " does not exist."
                  << std::endl;
        world.abort(1);
    }
    
    histdat.open(msgpackfilename.c_str());
    
    std::string sizeline;
    ///Each 
    std::size_t istep = 0;
    while(true){
        std::getline(histdat,sizeline,'#');
        auto length_block = boost::lexical_cast<std::size_t>(sizeline);
        if (length_block == 0)
            break;
        
        std::vector<char> dataline(length_block);
        histdat.read(dataline.data(),length_block);
        std::cout << "*Read line " << istep << std::endl;
        msgpack::object_handle oh =
            msgpack::unpack(dataline.data(), dataline.size());
        simdata.msgpack_unpack(oh.get());
        double that_time = simdata.print_distribution(inpars);
        
        /// Stop if last time has been read
        if (alma::almost_equal(that_time,inpars.dumptimes.back())){
            break;
        }
        
        istep++;
    }
    
    histdat.close();
    
    
    return 0;
}
