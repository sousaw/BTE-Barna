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
/// Writes phonon properties associated with HDF5 file to a text file.

#include <iostream>
#include <fstream>
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
#include <bulk_properties.hpp>

int main(int argc, char** argv) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    // Reference temperature at which to compute
    // heat capacities and scattering rates.
    // Optional 2nd argument, defaults to 300K
    double Tambient = 300.0;

    // Path to the HDF5 file.
    std::string h5filename;

    if (argc < 2) {
        std::cout << "USAGE: phononinfo <inputfile.hdf5> <OPTIONAL: Tambient "
                     "(default 300K)>"
                  << std::endl;
        world.abort(1);
    }

    if (argc >= 2) {
        h5filename = std::string(argv[1]);
    }

    if (argc >= 3) {
        Tambient = atof(argv[2]);
        if (Tambient <= 0.0) {
            std::cout << "ERROR: Tambient must be positive (" << Tambient
                      << " provided)." << std::endl;
            world.abort(1);
        }
    }

    std::cout << "***********************************" << std::endl;
    std::cout << "This is ALMA/phononinfo version " << ALMA_VERSION_MAJOR << "."
              << ALMA_VERSION_MINOR << std::endl;
    std::cout << "***********************************" << std::endl;

    // Create name of outputfile
    boost::filesystem::path hdf5_path{h5filename};
    std::stringstream output_file_builder;
    output_file_builder << hdf5_path.stem().string() << "_" << Tambient
                        << "K.phononinfo";
    std::string output_file = output_file_builder.str();

    // Obtain phonon data
    std::cout << "Reading " << h5filename << std::endl;

    if (!(boost::filesystem::exists(hdf5_path))) {
        std::cout << "ERROR:" << std::endl;
        std::cout << "HDF5 file " << hdf5_path << " does not exist."
                  << std::endl;
        world.abort(1);
    }

    auto hdf5_data = alma::load_bulk_hdf5(hdf5_path.string().c_str(), world);
    auto poscar = std::move(std::get<1>(hdf5_data));
    auto syms = std::move(std::get<2>(hdf5_data));
    auto grid = std::move(std::get<3>(hdf5_data));
    auto processes = std::move(std::get<4>(hdf5_data));

    // RTA scattering rates at the specified temperature.
    std::cout << "Calculating scattering rates" << std::endl;
    Eigen::ArrayXXd w3(
        alma::calc_w0_threeph(*grid, *processes, Tambient, world));
    auto twoph_processes = alma::find_allowed_twoph(*grid, world);
    Eigen::ArrayXXd w2(
        alma::calc_w0_twoph(*poscar, *grid, twoph_processes, world));
    Eigen::ArrayXXd w0(w3 + w2);

    // Create output writer
    std::cout << "Writing data to " << output_file << std::endl;
    std::ofstream filewriter;
    filewriter.open(output_file);
    filewriter << "nq,nbranch,qa[-],qb[-],qc[-],omega[rad/s],C[J/"
                  "m^3-K],tau[s],vx[m/s],vy[m/s],vz[m/s]"
               << std::endl;

    int Nq = grid->nqpoints;
    int Nbranches = grid->get_spectrum_at_q(0).omega.size();

    const double prefactor =
        1e27 * alma::constants::kB / grid->nqpoints / poscar->V;

    Eigen::Matrix3d kappa_RTA = alma::calc_kappa(*poscar,*grid,*syms, w0, Tambient);
    Eigen::Matrix3d invkappa_RTA;
    invkappa_RTA.setZero();
    for (auto axis : {0,1,2}) {
        invkappa_RTA(axis,axis) = 1./kappa_RTA(axis,axis);
    }
    
    
    Eigen::Matrix3d lnum, lden;
    lnum.setZero();
    lden.setZero();
    double lxnum = 0., lynum = 0., lznum = 0.;
    double lxden = 0., lyden = 0., lzden = 0.;

    
    for (int nq = 0; nq < Nq; ++nq) {
        
        auto coords_ = poscar->map_to_firstbz(
            grid->get_q(nq));
        
        double hqx = 1.0e+9 * alma::constants::hbar * coords_.row(0).mean();
        double hqy = 1.0e+9 * alma::constants::hbar * coords_.row(1).mean();
        double hqz = 1.0e+9 * alma::constants::hbar * coords_.row(2).mean();
        
        Eigen::Matrix3d moment;

        moment.setZero();

        moment(0,0) = hqx;
        moment(1,1) = hqy;
        moment(2,2) = hqz;
                  
        
        auto sp = grid->get_spectrum_at_q(nq);

        std::array<int, 3> qcoords = grid->one_to_three(nq);
        double qa =
            static_cast<double>(qcoords[0]) / static_cast<double>(grid->na);
        double qb =
            static_cast<double>(qcoords[1]) / static_cast<double>(grid->nb);
        double qc =
            static_cast<double>(qcoords[2]) / static_cast<double>(grid->nc);

        for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
            // scattering rate
            double my_w0 = w0(nbranch, nq);

            // relaxation time [seconds]
            double tau0 = (my_w0 == 0.) ? 0. : (1e-12 / my_w0);

            // angular frequency [rad/s]
            double omega = 1e12 * sp.omega[nbranch];

            // volumetric heat capacity [J/m^3-K]
            double C = prefactor *
                       alma::bose_einstein_kernel(sp.omega[nbranch], Tambient);

            // group velocity vector [m/s]
            Eigen::Vector3d vg_vector = 1e3 * sp.vg.col(nbranch);
            
            Eigen::Matrix3d vmat;
            vmat.setZero();
            for (auto axis : {0,1,2}){
                vmat(axis,axis) = vg_vector(axis);
            }
            
            Eigen::Matrix3d g1;
            g1.setZero();
            g1 = - tau0 * tau0 *  alma::constants::kB *
                   alma::bose_einstein_kernel(sp.omega[nbranch], Tambient) *
                   invkappa_RTA * vmat * vmat;


            double g1_x = - tau0 * tau0 * alma::constants::kB *
                   alma::bose_einstein_kernel(sp.omega[nbranch], Tambient) * 
                   vg_vector(0) * vg_vector(0)/kappa_RTA(0,0);
            double g1_y = - tau0 * tau0 * alma::constants::kB *
                   alma::bose_einstein_kernel(sp.omega[nbranch], Tambient) * 
                   vg_vector(1) * vg_vector(1)/kappa_RTA(1,1);
            double g1_z = - tau0 * tau0 * alma::constants::kB *
                   alma::bose_einstein_kernel(sp.omega[nbranch], Tambient) * 
                   vg_vector(2) * vg_vector(2)/kappa_RTA(2,2);


            lxnum += hqx * vg_vector(0) * g1_x;
            lynum += hqy * vg_vector(1) * g1_y;
            lznum += hqz * vg_vector(2) * g1_z;

            lxden += hqx * vg_vector(0) * alma::constants::kB * alma::bose_einstein_kernel(sp.omega[nbranch], Tambient);
            lyden += hqx * vg_vector(0) * alma::constants::kB * alma::bose_einstein_kernel(sp.omega[nbranch], Tambient);
            lzden += hqx * vg_vector(0) * alma::constants::kB * alma::bose_einstein_kernel(sp.omega[nbranch], Tambient);

                   
            lnum += moment * vmat * g1;
            lden += moment * vmat * alma::constants::kB *
                   alma::bose_einstein_kernel(sp.omega[nbranch], Tambient);
            


              
            // write output
            filewriter << nq << "," << nbranch << "," << qa << "," << qb << ","
                       << qc << ",";
            filewriter << omega << "," << C << "," << tau0 << ",";
            filewriter << vg_vector(0) << "," << vg_vector(1) << ","
                       << vg_vector(2) << std::endl;
        }
    }

    filewriter.close();
    if (world.rank() == 0.) {
        std::cout << "#Small-grain-limit kappa [W/(m**2 * K)]\n";
        Eigen::Matrix3d ksg = alma::calc_kappa_sg(*poscar,*grid,Tambient);
        std::cout << ksg << std::endl;
        std::cout << "#l**2  assuming isotropic bands [m**2]" <<std::endl;
        std::cout << "#x " << -1./5 * kappa_RTA(0,0) * lxnum / lxden << std::endl;
        std::cout << "#y " << -1./5 * kappa_RTA(1,1) * lynum / lyden << std::endl;
        std::cout << "#z " << -1./5 * kappa_RTA(2,2) * lznum / lzden << std::endl;
    }
    
    std::cout << std::endl << "[DONE.]" << std::endl;

    return 0;
}
