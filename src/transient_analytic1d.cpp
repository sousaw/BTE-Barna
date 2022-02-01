// Copyright 2015-2022 The ALMA Project Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANLIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

/// @file
/// EXECUTABLE THAT COMPUTES TRANSIENT SINGLE-PULSE RESPONSES
/// SEMI-ANALYTICALLY.

#include <iostream>
#include <functional>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include <utilities.hpp>
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>
#include <isotopic_scattering.hpp>
#include <bulk_hdf5.hpp>
#include <analytic1d.hpp>
#include <io_utils.hpp>
#include <bulk_properties.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

// Vector of time values for spacecurves calculations
std::vector<double> spacecurves_timevalues;

// Helper function that reads timevalues from string
// {t1,t2,...,tN} and puts them into the timegrid vector

void processTimepoints(std::string inputlist) {
    std::size_t Nvalues =
        std::count(inputlist.begin(), inputlist.end(), ',') + 1;

    // strip brackets and replace comma's to spaces for easy number
    // extraction
    std::string buffer(inputlist.substr(1, inputlist.length() - 2));

    std::replace(buffer.begin(), buffer.end(), ',', ' ');

    std::stringstream inputreader(buffer);
    double value = -1.0;

    for (std::size_t n = 0; n < Nvalues; n++) {
        inputreader >> value;
        spacecurves_timevalues.emplace_back(value);
    }
}


int main(int argc, char** argv) {
    // set up MPI environment
    boost::mpi::environment env;
    boost::mpi::communicator world;

    if (world.size() > 1) {
        std::cout << "*** ERROR: transient_analytic1d does not run in parallel "
                     "mode. ***"
                  << std::endl;
        world.abort(1);
    }

    if (argc < 2) {
        std::cout
            << "USAGE: transient_analytic1d <inputfile.xml> <OPTIONAL:Tambient>"
            << std::endl;
        return 1;
    }

    // DEFINE VARIABLES
    // output
    std::string target_directory = "AUTO";
    // material data
    std::string h5_repository = ".";
    std::string mat_directory;
    std::string mat_base;
    int gridDensityA = -1;
    int gridDensityB = -1;
    int gridDensityC = -1;
    bool superlattice = false;
    std::string superlattice_UID = "NULL";
    // spacecurve calculations
    bool perform_spacecurves = false;
    double spacecurves_tmin = -1.0;
    double spacecurves_tmax = -1.0;
    bool spacecurves_autotimegrid = true;
    bool spacecurves_logsweep = false;
    int spacecurves_Nt = -1;
    int spacecurves_Nx = -1;
    // source transient calculations
    bool perform_ST = false;
    double ST_tmin = -1.0;
    double ST_tmax = -1.0;
    bool ST_logsweep = false;
    int ST_Nt = -1;
    // MSD calculations
    bool perform_MSD = false;
    double MSD_tmin = -1.0;
    double MSD_tmax = -1.0;
    bool MSD_logsweep = false;
    int MSD_Nt = -1;
    // transport vector
    Eigen::Vector3d uvector(0.0, 0.0, 0.0);

    double T = 300.0;

    if (argc == 3) {
        T = atof(argv[2]);
    }

    std::cout << "*********************************************" << std::endl;
    std::cout << "This is ALMA/transient_analytic1d version "
              << ALMA_VERSION_MAJOR << "." << ALMA_VERSION_MINOR << std::endl;
    std::cout << "*********************************************" << std::endl;

    // verify that input file exists.
    if (!boost::filesystem::exists(boost::filesystem::path{argv[1]})) {
        std::cout << "ERROR: input file " << argv[1] << " does not exist."
                  << std::endl;
        exit(1);
    }


    /////////////////////////
    /// PARSE INPUT FILE  ///
    /////////////////////////

    std::string xmlfile(argv[1]);
    std::cout << "PARSING " << xmlfile << " ..." << std::endl;

    // Create empty property tree object
    boost::property_tree::ptree tree;

    // Parse XML input file into the tree
    boost::property_tree::read_xml(xmlfile, tree);

    for (const auto& v : tree.get_child("analytic1d")) {
        if (v.first == "H5repository") {
            h5_repository =
                alma::parseXMLfield<std::string>(v, "root_directory");
        }

        if (v.first == "target") {
            target_directory = alma::parseXMLfield<std::string>(v, "directory");
        }

        if (v.first == "compound") {
            mat_directory = alma::parseXMLfield<std::string>(v, "directory");
            mat_base = alma::parseXMLfield<std::string>(v, "base");

            gridDensityA = alma::parseXMLfield<int>(v, "gridA");
            gridDensityB = alma::parseXMLfield<int>(v, "gridB");
            gridDensityC = alma::parseXMLfield<int>(v, "gridC");
        }

        if (v.first == "superlattice") {
            superlattice_UID = alma::parseXMLfield<std::string>(v, "UID");
        }

        if (v.first == "transportAxis") {
            double ux = alma::parseXMLfield<double>(v, "x");
            double uy = alma::parseXMLfield<double>(v, "y");
            double uz = alma::parseXMLfield<double>(v, "z");

            uvector << ux, uy, uz;
        }

        if (v.first == "spacecurves") {
            perform_spacecurves = true;
            spacecurves_Nx = alma::parseXMLfield<int>(v, "spacepoints");

            std::string timepoints =
                alma::parseXMLfield<std::string>(v, "timepoints");

            // If timepoints contains a list of values, insert them into the
            // timegrid.

            if (timepoints.find("{") != std::string::npos) {
                processTimepoints(timepoints);
                spacecurves_autotimegrid = false;
            }

            else { // Not a list of values, obtain parameters for lin/log time
                   // grid

                std::string spacecurves_sweepID =
                    alma::parseXMLfield<std::string>(v, "timesweep");

                if (spacecurves_sweepID.compare("log") == 0) {
                    spacecurves_logsweep = true;
                }

                spacecurves_tmin = alma::parseXMLfield<double>(v, "tstart");
                spacecurves_tmax = alma::parseXMLfield<double>(v, "tstop");
                spacecurves_Nt = alma::parseXMLfield<int>(v, "timepoints");
            }
        }

        if (v.first == "sourcetransient") {
            perform_ST = true;

            std::string ST_sweepID =
                alma::parseXMLfield<std::string>(v, "timesweep");

            if (ST_sweepID.compare("log") == 0) {
                ST_logsweep = true;
            }

            ST_tmin = alma::parseXMLfield<double>(v, "tstart");
            ST_tmax = alma::parseXMLfield<double>(v, "tstop");
            ST_Nt = alma::parseXMLfield<int>(v, "points");
        }

        if (v.first == "MSD") {
            perform_MSD = true;

            std::string MSD_sweepID =
                alma::parseXMLfield<std::string>(v, "timesweep");

            if (MSD_sweepID.compare("log") == 0) {
                MSD_logsweep = true;
            }

            MSD_tmin = alma::parseXMLfield<double>(v, "tstart");
            MSD_tmax = alma::parseXMLfield<double>(v, "tstop");
            MSD_Nt = alma::parseXMLfield<int>(v, "points");
        }
    } // end XML parsing

    // Ensure that provided information is within expected bounds

    bool badinput = false;

    if (gridDensityA < 1) {
        std::cout << "ERROR: provided gridA is " << gridDensityA << std::endl;
        std::cout << "Value must be at least 1." << std::endl;
        badinput = true;
    }

    if (gridDensityB < 1) {
        std::cout << "ERROR: provided gridB is " << gridDensityB << std::endl;
        std::cout << "Value must be at least 1." << std::endl;
        badinput = true;
    }

    if (gridDensityC < 1) {
        std::cout << "ERROR: provided gridC is " << gridDensityC << std::endl;
        std::cout << "Value must be at least 1." << std::endl;
        badinput = true;
    }

    if (perform_spacecurves) {
        if (spacecurves_autotimegrid) {
            if (spacecurves_tmin <= 0.0) {
                std::cout << "ERROR: provided start time for spacecurves is "
                          << spacecurves_tmin << std::endl;
                std::cout << "Value must be positive." << std::endl;
                badinput = true;
            }

            if ((spacecurves_tmax < spacecurves_tmin) ||
                (spacecurves_tmax <= 0.0)) {
                std::cout << "ERROR: provided end time for spacecurves is "
                          << spacecurves_tmax << std::endl;
                std::cout << "Value must be positive and >= start time."
                          << std::endl;
                badinput = true;
            }

            if (spacecurves_Nt <= 0) {
                std::cout
                    << "ERROR: requested number of spacecurves timepoints is "
                    << spacecurves_Nt << std::endl;
                std::cout << "Value must be at least 1." << std::endl;
                badinput = true;
            }
        }

        else {
            spacecurves_Nt = spacecurves_timevalues.size();

            for (int nt = 0; nt < spacecurves_Nt; nt++) {
                if (spacecurves_timevalues.at(nt) <= 0.0) {
                    std::cout << "ERROR: provided spacecurves timevalues list "
                                 "contains "
                              << spacecurves_timevalues.at(nt) << std::endl;
                    std::cout << "All timevalues must be positive."
                              << std::endl;
                    badinput = true;
                }
            }
        }

        if (spacecurves_Nx <= 0) {
            std::cout
                << "ERROR: requested number of spacecurves spacepoints is "
                << spacecurves_Nx << std::endl;
            std::cout << "Value must be at least 1." << std::endl;
            badinput = true;
        }
    }

    if (perform_ST) {
        if (ST_tmin <= 0.0) {
            std::cout << "ERROR: provided start time for sourcetransient is "
                      << ST_tmin << std::endl;
            std::cout << "Value must be positive." << std::endl;
            badinput = true;
        }

        if ((ST_tmax < ST_tmin) || (ST_tmax <= 0.0)) {
            std::cout << "ERROR: provided end time for sourcetransient is "
                      << ST_tmax << std::endl;
            std::cout << "Value must be positive and >= start time."
                      << std::endl;
            badinput = true;
        }

        if (ST_Nt <= 0) {
            std::cout << "ERROR: requested number of sourcetransient points is "
                      << ST_Nt << std::endl;
            std::cout << "Value must be at least 1." << std::endl;
            badinput = true;
        }
    }

    if (perform_MSD) {
        if (MSD_tmin <= 0.0) {
            std::cout << "ERROR: provided start time for MSD is " << MSD_tmin
                      << std::endl;
            std::cout << "Value must be positive." << std::endl;
            badinput = true;
        }

        if ((MSD_tmax < MSD_tmin) || (MSD_tmax <= 0.0)) {
            std::cout << "ERROR: provided end time for MSD is " << MSD_tmax
                      << std::endl;
            std::cout << "Value must be positive and >= start time."
                      << std::endl;
            badinput = true;
        }

        if (MSD_Nt <= 0) {
            std::cout << "ERROR: requested number of MSD points is " << MSD_Nt
                      << std::endl;
            std::cout << "Value must be at least 1." << std::endl;
            badinput = true;
        }
    }

    if (uvector.norm() < 1e-12) {
        std::cout << "ERROR: provided transport axis vector has zero norm."
                  << std::endl;
        badinput = true;
    }

    if (T <= 0) {
        std::cout << "ERROR: provided ambient temperature is " << T
                  << std::endl;
        std::cout << "Value must be postive." << std::endl;
        badinput = true;
    }

    if (badinput) {
        exit(1);
    }

    // Initialise file system and verify that directories actually exist
    auto launch_path = boost::filesystem::current_path();
    auto basedir = boost::filesystem::path(h5_repository);

    if (!(boost::filesystem::exists(boost::filesystem::path(basedir)))) {
        std::cout << "ERROR:" << std::endl;
        std::cout << "Repository directory " << basedir << " does not exist."
                  << std::endl;
        exit(1);
    }

    if (!(boost::filesystem::exists(
            boost::filesystem::path(basedir / mat_directory)))) {
        std::cout << "ERROR:" << std::endl;
        std::cout << "Material directory " << mat_directory
                  << " does not exist within the HDF5 repository." << std::endl;
        exit(1);
    }

    // Resolve name of HDF5 file
    std::stringstream h5namebuilder;
    h5namebuilder << mat_base << "_";
    h5namebuilder << gridDensityA << "_" << gridDensityB << "_" << gridDensityC
                  << ".h5";
    std::string h5filename = h5namebuilder.str();

    // obtain phonon data from HDF5 file
    auto hdf5_path = basedir / boost::filesystem::path(mat_directory) /
                     boost::filesystem::path(h5filename);

    if (!(boost::filesystem::exists(hdf5_path))) {
        std::cout << "ERROR:" << std::endl;
        std::cout << "H5 file " << h5filename
                  << " does not exist within the material directory."
                  << std::endl;
        exit(1);
    }

    std::cout << "Opening HDF5 file " << hdf5_path << std::endl;

    auto hdf5_data = alma::load_bulk_hdf5(hdf5_path.string().c_str(), world);
    auto description = std::get<0>(hdf5_data);
    auto poscar = std::move(std::get<1>(hdf5_data));
    auto syms = std::move(std::get<2>(hdf5_data));
    auto grid = std::move(std::get<3>(hdf5_data));
    auto processes = std::move(std::get<4>(hdf5_data));

    if (processes->size() == 0) {
        std::cout << "ERROR:" << std::endl;
        std::cout << "List of 3-phonon processes is missing in H5 file."
                  << std::endl;
        exit(1);
    }

    // Check if we are dealing with a superlattice.
    // If so, load the applicable scattering data.

    auto subgroups =
        alma::list_scattering_subgroups(hdf5_path.string().c_str(), world);

    int superlattice_count = 0;

    for (std::size_t ngroup = 0; ngroup < subgroups.size(); ngroup++) {
        if (subgroups.at(ngroup).find("superlattice") != std::string::npos) {
            superlattice_count++;
        }
    }

    superlattice_count /= 2;

    if (superlattice_count > 0) {
        superlattice = true;
    }

    Eigen::ArrayXXd w0_SLdisorder;
    Eigen::ArrayXXd w0_SLbarriers;

    if (superlattice) {
        // complain if there are multiple possibilities

        if ((superlattice_count > 1) && (superlattice_UID == "NULL")) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "H5 file contains scattering information for multiple "
                         "superlattices."
                      << std::endl;
            std::cout << "Must provide the superlattice UID via the "
                         "<superlattice> XML tag."
                      << std::endl;
            exit(1);
        }

        // if the user provided a UID, verify that corresponding data exists

        if (superlattice_UID != "NULL") {
            int UIDcount = 0;

            for (std::size_t ngroup = 0; ngroup < subgroups.size(); ngroup++) {
                if (subgroups.at(ngroup).find(superlattice_UID) !=
                    std::string::npos) {
                    UIDcount++;
                }
            }

            if (UIDcount != 2) {
                std::cout << "ERROR:" << std::endl;
                std::cout << "H5 file does not contain any superlattice data "
                             "with provided UID "
                          << superlattice_UID << "." << std::endl;
                exit(1);
            }
        }

        // load the scattering rates from the H5 file

        bool UIDmatch = true;

        for (std::size_t ngroup = 0; ngroup < subgroups.size(); ngroup++) {
            bool contains_SLdisorder =
                (subgroups.at(ngroup).find("superlattice") !=
                 std::string::npos) &&
                (subgroups.at(ngroup).find("disorder") != std::string::npos);

            bool contains_SLbarriers =
                (subgroups.at(ngroup).find("superlattice") !=
                 std::string::npos) &&
                (subgroups.at(ngroup).find("barriers") != std::string::npos);

            if (superlattice_count > 1) {
                UIDmatch = (subgroups.at(ngroup).find(superlattice_UID) !=
                            std::string::npos);
            }

            if (contains_SLdisorder && UIDmatch) {
                auto mysubgroup = alma::load_scattering_subgroup(
                    hdf5_path.string().c_str(), subgroups.at(ngroup), world);
                w0_SLdisorder = mysubgroup.w0;
            }

            if (contains_SLbarriers && UIDmatch) {
                auto mysubgroup = alma::load_scattering_subgroup(
                    hdf5_path.string().c_str(), subgroups.at(ngroup), world);
                w0_SLbarriers = mysubgroup.w0;
            }
        }
    }

    // calculate scattering rates at the chosen temperature
    Eigen::ArrayXXd w3(alma::calc_w0_threeph(*grid, *processes, T, world));
    Eigen::ArrayXXd w_elastic;

    if (superlattice) {
        w_elastic = w0_SLdisorder.array() + w0_SLbarriers.array();
    }
    else {
        auto twoph_processes = alma::find_allowed_twoph(*grid, world);
        w_elastic = alma::calc_w0_twoph(*poscar, *grid, twoph_processes, world);
    }

    Eigen::ArrayXXd w(w3 + w_elastic);

    // obtain some basic thermal properties
    alma::analytic1D::BasicProperties_calculator propCalc(
        poscar.get(), grid.get(), &w, T);
    propCalc.setDirection(uvector);
    double Dbulk = propCalc.getDiffusivity();
    double Cbulk = propCalc.getCapacity();
    double tau_dom = propCalc.getDominantRT();

    // warn user if spacecurves are being computed out of the validity range

    double tmin = 1e300;

    for (std::size_t ntime = 0; ntime < spacecurves_timevalues.size();
         ntime++) {
        if (spacecurves_timevalues.at(ntime) < tmin) {
            tmin = spacecurves_timevalues.at(ntime);
        }
    }

    if (tmin < tau_dom) {
        std::cout << "** WARNING: **" << std::endl;
        std::cout << "Computation of temperature profiles is only valid in"
                  << std::endl;
        std::cout << "weakly quasiballistic regimes (times >= "
                  << alma::engineer_format(tau_dom) << "s)" << std::endl;
        std::cout << "The following output may be unreliable:" << std::endl;

        for (std::size_t ntime = 0; ntime < spacecurves_timevalues.size();
             ntime++) {
            if (spacecurves_timevalues.at(ntime) < tau_dom) {
                std::cout << "spacecurve requested at time "
                          << alma::engineer_format(
                                 spacecurves_timevalues.at(ntime))
                          << "s" << std::endl;
            }
        }
    }

    // RESOLVE OUTPUT DIRECTORY

    // go to the launch directory
    boost::filesystem::current_path(launch_path);

    // resolve output directory if AUTO is selected
    if (target_directory.compare("AUTO") == 0) {
        target_directory = "output/transient_analytic1d";
    }

    // create output directory if it doesn't exist yet
    auto outputfolder = boost::filesystem::path(target_directory);

    if (!(boost::filesystem::exists(outputfolder))) {
        boost::filesystem::create_directories(outputfolder);
    }

    boost::filesystem::current_path(launch_path);

    std::cout << "All output will be written to directory " << target_directory
              << std::endl;
    std::cout << "under base directory " << launch_path << std::endl;

    // *** PERFORM CALCULATIONS ***

    // SPACE CURVES

    if (perform_spacecurves) {
        // initialise time grid
        Eigen::VectorXd spacecurves_timegrid;

        if (spacecurves_autotimegrid) {
            if (spacecurves_logsweep) {
                spacecurves_timegrid = alma::logSpace(
                    spacecurves_tmin, spacecurves_tmax, spacecurves_Nt);
            }
            else {
                spacecurves_timegrid.setLinSpaced(
                    spacecurves_Nt, spacecurves_tmin, spacecurves_tmax);
            }
        }

        else {
            spacecurves_timegrid.resize(spacecurves_Nt);

            for (int nt = 0; nt < spacecurves_Nt; nt++) {
                spacecurves_timegrid(nt) = spacecurves_timevalues.at(nt);
            }
        }

        // spacecurves output
        Eigen::MatrixXd spacecurves_output(spacecurves_Nx, 3);

        // Initialise calculator
        alma::analytic1D::SPR_calculator_RealSpace SPRcalc(
            poscar.get(), grid.get(), &w, T);
        SPRcalc.setDirection(uvector);
        SPRcalc.setLinGrid(-5.0, 5.0, spacecurves_Nx);
        SPRcalc.declareGridNormalised(true);

        // RUN CALCULATIONS

        for (int nt = 0; nt < spacecurves_Nt; nt++) {
            std::cout << std::endl
                      << "Computing single pulse response across space (curve "
                      << nt + 1 << " of " << spacecurves_Nt << ")" << std::endl;

            double t = spacecurves_timegrid(nt);

            SPRcalc.setTime(t);

            Eigen::VectorXd spacegrid = SPRcalc.getGrid();

            // BTE solution
            spacecurves_output.col(0) = spacegrid;
            spacecurves_output.col(1) = (1.0 / Cbulk) * SPRcalc.getSPR();

            // Fourier solution
            double prefactor =
                1.0 /
                (Cbulk * std::sqrt(4.0 * alma::constants::pi * Dbulk * t));
            spacecurves_output.col(2) =
                prefactor * Eigen::exp(-(1.0 / (4.0 * Dbulk * t)) *
                                       spacegrid.array().square());

            // write to file

            std::stringstream fileheader_builder;
            fileheader_builder
                << "x[m],T(x,t=" << alma::engineer_format(t)
                << "s)[K],T_Fourier(x,t=" << alma::engineer_format(t)
                << "s)[K]\n";
            std::string fileheader = fileheader_builder.str();

            std::stringstream filenamebuilder;
            filenamebuilder << mat_base << "_" << gridDensityA << "_"
                            << gridDensityB << "_" << gridDensityC;
            filenamebuilder << "_" << uvector(0) << "," << uvector(1) << ","
                            << uvector(2);
            filenamebuilder << "_" << T << "K";
            filenamebuilder << "_" << alma::engineer_format(t)
                            << "s.spacecurve";

            std::string spacecurves_file = filenamebuilder.str();

            std::cout << "Writing to file " << spacecurves_file << std::endl;

            alma::write_to_csv("./" + target_directory + "/" + spacecurves_file,
                               spacecurves_output,
                               ',',
                               false,
                               fileheader);
        }
    }

    // SOURCE TRANSIENT

    if (perform_ST) {
        std::cout << std::endl << "Computing source transient..." << std::endl;

        if (ST_tmin < tau_dom) {
            std::cout << "*** WARNING: ***" << std::endl;
            std::cout << "Computation of source transient is only valid in"
                      << std::endl;
            std::cout << "weakly quasiballistic regimes (times >= "
                      << alma::engineer_format(tau_dom) << "s)" << std::endl;
            std::cout << "The initial portion of the computed transient may be "
                         "unreliable."
                      << std::endl;
        }

        // ST output
        Eigen::MatrixXd ST_output(ST_Nt, 3);

        // Initialise calculator
        alma::analytic1D::SPR_calculator_RealSpace SPRcalc(
            poscar.get(), grid.get(), &w, T);
        SPRcalc.setDirection(uvector);

        Eigen::VectorXd ST_timegrid = alma::logSpace(ST_tmin, ST_tmax, ST_Nt);

        if (!ST_logsweep) {
            ST_timegrid.setLinSpaced(ST_Nt, ST_tmin, ST_tmax);
        }

        // perform source transient calculations
        Eigen::VectorXd ST_raw = SPRcalc.getSourceTransient(ST_timegrid);
        Eigen::VectorXd ST_Fourier =
            Eigen::pow(4.0 * alma::constants::pi * SPRcalc.getDiffusivity() *
                           ST_timegrid.array(),
                       -0.5);
        Eigen::VectorXd ST_norm = ST_raw.array() / ST_Fourier.array();

        // for output, convert computed energy density to temperature
        ST_output << ST_timegrid, (1.0 / Cbulk) * ST_raw.array(), ST_norm;

        // write to file

        std::string fileheader = "t[s],T(x=0)[K],T(x=0)/T_Fourier(x=0)[-]\n";

        std::stringstream filenamebuilder;
        filenamebuilder << mat_base << "_" << gridDensityA << "_"
                        << gridDensityB << "_" << gridDensityC;
        filenamebuilder << "_" << uvector(0) << "," << uvector(1) << ","
                        << uvector(2);
        filenamebuilder << "_" << T << "K";
        filenamebuilder << "_" << alma::engineer_format(ST_tmin) << "s_"
                        << alma::engineer_format(ST_tmax)
                        << "s.sourcetransient";

        std::string ST_file = filenamebuilder.str();

        std::cout << "Writing to file " << ST_file << std::endl;

        alma::write_to_csv("./" + target_directory + "/" + ST_file,
                           ST_output,
                           ',',
                           false,
                           fileheader);
    }

    // MSD

    if (perform_MSD) {
        std::cout << std::endl
                  << "Computing mean square displacement..." << std::endl;

        // MSD output
        Eigen::MatrixXd MSD_output(MSD_Nt, 3);

        // Initialise calculator
        alma::analytic1D::MSD_calculator_RealTime MSDcalc(
            poscar.get(), grid.get(), &w, T);
        MSDcalc.setDirection(uvector);

        if (MSD_logsweep) {
            MSDcalc.setLogGrid(MSD_tmin, MSD_tmax, MSD_Nt);
        }
        else {
            MSDcalc.setLinGrid(MSD_tmin, MSD_tmax, MSD_Nt);
        }

        Eigen::VectorXd MSD_timegrid = MSDcalc.getGrid();

        // perform MSD calculations
        MSDcalc.normaliseOutput(true);
        Eigen::VectorXd MSD_norm = MSDcalc.getMSD();
        Eigen::VectorXd MSD_raw =
            2.0 * Dbulk * MSD_timegrid.array() * MSD_norm.array();

        MSD_output << MSD_timegrid, MSD_raw, MSD_norm;

        // write to file

        std::string fileheader = "t[s],MSD[m^2],MSD/2Dt[-]\n";

        std::stringstream filenamebuilder;
        filenamebuilder << mat_base << "_" << gridDensityA << "_"
                        << gridDensityB << "_" << gridDensityC;
        filenamebuilder << "_" << uvector(0) << "," << uvector(1) << ","
                        << uvector(2);
        filenamebuilder << "_" << T << "K";
        filenamebuilder << "_" << alma::engineer_format(MSD_tmin) << "s_"
                        << alma::engineer_format(MSD_tmax) << "s.MSD";

        std::string MSD_file = filenamebuilder.str();

        std::cout << "Writing to file " << MSD_file << std::endl;

        alma::write_to_csv("./" + target_directory + "/" + MSD_file,
                           MSD_output,
                           ',',
                           false,
                           fileheader);
    }

    std::cout << std::endl << "[DONE.]" << std::endl;

    return 0;
}
