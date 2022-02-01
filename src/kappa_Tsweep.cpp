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
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the License.

/// @file
/// Computes conductivities versus temperature.
/// Material information and sweep settings provided via XML input.

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
#include <Eigen/Dense>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <beyondRTA.hpp>

int main(int argc, char** argv) {
    // set up MPI environment
    boost::mpi::environment env;
    boost::mpi::communicator world;

    if (world.size() > 1) {
        std::cout
            << "*** ERROR: kappa_Tsweep does not run in parallel mode. ***"
            << std::endl;
        world.abort(1);
    }

    if (argc < 2) {
        std::cout << "USAGE: kappa_Tsweep <inputfile.xml>" << std::endl;
        return 1;
    }

    else {
        // define variables
        std::string target_directory = "AUTO";
        std::string target_filename = "AUTO";
        std::string h5_repository = ".";
        std::string mat_directory;
        std::string mat_base;
        int gridDensityA = -1;
        int gridDensityB = -1;
        int gridDensityC = -1;
        bool logsweep = false;
        double Tmin = -1.0;
        double Tmax = -1.0;
        int NT = -1;
        bool projectConductivity = false;
        Eigen::Vector3d uvector(0.0, 0.0, 0.0);
        Eigen::Vector3d u_norm(0.0, 0.0, 0.0);
        bool fullBTE = false;
        bool fullBTE_iterative = true;
        bool outputCapacity = false;
        bool superlattice = false;
        std::string superlattice_UID = "NULL";

        std::cout << "*************************************" << std::endl;
        std::cout << "This is ALMA/kappa_Tsweep version " << ALMA_VERSION_MAJOR
                  << "." << ALMA_VERSION_MINOR << std::endl;
        std::cout << "*************************************" << std::endl;

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

        for (const auto& v : tree.get_child("Tsweep")) {
            if (v.first == "H5repository") {
                h5_repository =
                    alma::parseXMLfield<std::string>(v, "root_directory");
            }

            if (v.first == "target") {
                target_directory =
                    alma::parseXMLfield<std::string>(v, "directory");
                target_filename = alma::parseXMLfield<std::string>(v, "file");
            }

            if (v.first == "compound") {
                mat_directory =
                    alma::parseXMLfield<std::string>(v, "directory");
                mat_base = alma::parseXMLfield<std::string>(v, "base");

                gridDensityA = alma::parseXMLfield<int>(v, "gridA");
                gridDensityB = alma::parseXMLfield<int>(v, "gridB");
                gridDensityC = alma::parseXMLfield<int>(v, "gridC");
            }

            if (v.first == "superlattice") {
                superlattice_UID = alma::parseXMLfield<std::string>(v, "UID");
            }

            if (v.first == "sweep") {
                std::string sweepID =
                    alma::parseXMLfield<std::string>(v, "type");

                if (sweepID.compare("log") == 0) {
                    logsweep = true;
                }

                Tmin = alma::parseXMLfield<double>(v, "start");
                Tmax = alma::parseXMLfield<double>(v, "stop");
                NT = alma::parseXMLfield<int>(v, "points");
            }

            if (v.first == "fullBTE") {
                fullBTE = true;
                fullBTE_iterative = alma::parseXMLfield<bool>(v, "iterative");
            }

            if (v.first == "transportAxis") {
                projectConductivity = true;
                double ux = alma::parseXMLfield<double>(v, "x");
                double uy = alma::parseXMLfield<double>(v, "y");
                double uz = alma::parseXMLfield<double>(v, "z");

                uvector << ux, uy, uz;
            }

            if (v.first == "outputHeatCapacity") {
                outputCapacity = true;
            }
        } // end XML parsing

        // Ensure that provided information is within expected bounds

        bool badinput = false;

        if (gridDensityA < 1) {
            std::cout << "ERROR: provided gridA is " << gridDensityA
                      << std::endl;
            std::cout << "Value must be at least 1." << std::endl;
            badinput = true;
        }

        if (gridDensityB < 1) {
            std::cout << "ERROR: provided gridB is " << gridDensityB
                      << std::endl;
            std::cout << "Value must be at least 1." << std::endl;
            badinput = true;
        }

        if (gridDensityC < 1) {
            std::cout << "ERROR: provided gridC is " << gridDensityC
                      << std::endl;
            std::cout << "Value must be at least 1." << std::endl;
            badinput = true;
        }

        if (Tmin <= 0.0) {
            std::cout << "ERROR: provided start temperature is " << Tmin
                      << std::endl;
            std::cout << "Value must be positive." << std::endl;
            badinput = true;
        }

        if ((Tmax < Tmin) || (Tmax <= 0.0)) {
            std::cout << "ERROR: provided end temperature is " << Tmax
                      << std::endl;
            std::cout << "Value must be positive and >= start temperature."
                      << std::endl;
            badinput = true;
        }

        if (NT <= 0) {
            std::cout << "ERROR: provided number of temperature values is "
                      << NT << std::endl;
            std::cout << "Value must be at least 1." << std::endl;
            badinput = true;
        }

        if (projectConductivity && (uvector.norm() < 1e-12)) {
            std::cout << "ERROR: provided transport axis vector has zero norm."
                      << std::endl;
            badinput = true;
        }

        if (badinput) {
            world.abort(1);
        }

        // Initialise file system and verify that directories actually exist
        auto launch_path = boost::filesystem::current_path();
        auto basedir = boost::filesystem::path(h5_repository);

        if (!(boost::filesystem::exists(boost::filesystem::path(basedir)))) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "Repository directory " << basedir
                      << " does not exist." << std::endl;
            world.abort(1);
        }

        if (!(boost::filesystem::exists(
                boost::filesystem::path(basedir / mat_directory)))) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "Material directory " << mat_directory
                      << " does not exist within the HDF5 repository."
                      << std::endl;
            world.abort(1);
        }

        // Resolve name of HDF5 file
        std::stringstream h5namebuilder;
        h5namebuilder << mat_base << "_" << gridDensityA << "_" << gridDensityB
                      << "_" << gridDensityC << ".h5";
        std::string h5filename = h5namebuilder.str();

        // obtain phonon data from HDF5 file
        auto hdf5_path = basedir / boost::filesystem::path(mat_directory) /
                         boost::filesystem::path(h5filename);

        if (!(boost::filesystem::exists(hdf5_path))) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "H5 file " << h5filename
                      << " does not exist within the material directory."
                      << std::endl;
            world.abort(1);
        }

        std::cout << "Opening HDF5 file " << hdf5_path << std::endl;

        auto hdf5_data =
            alma::load_bulk_hdf5(hdf5_path.string().c_str(), world);
        auto description = std::get<0>(hdf5_data);
        auto poscar = std::move(std::get<1>(hdf5_data));
        auto syms = std::move(std::get<2>(hdf5_data));
        auto grid = std::move(std::get<3>(hdf5_data));
        auto processes = std::move(std::get<4>(hdf5_data));

        if (processes->size() == 0) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "List of 3-phonon processes is missing in H5 file."
                      << std::endl;
            world.abort(1);
        }

        // Check if we are dealing with a superlattice.
        // If so, load the applicable scattering data.

        auto subgroups =
            alma::list_scattering_subgroups(hdf5_path.string().c_str(), world);

        int superlattice_count = 0;

        for (std::size_t ngroup = 0; ngroup < subgroups.size(); ngroup++) {
            if (subgroups.at(ngroup).find("superlattice") !=
                std::string::npos) {
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
                std::cout << "H5 file contains scattering information for "
                             "multiple superlattices."
                          << std::endl;
                std::cout << "Must provide the superlattice UID via the "
                             "<superlattice> XML tag."
                          << std::endl;
                world.abort(1);
            }

            // if the user provided a UID, verify that corresponding data exists

            if (superlattice_UID != "NULL") {
                int UIDcount = 0;

                for (std::size_t ngroup = 0; ngroup < subgroups.size();
                     ngroup++) {
                    if (subgroups.at(ngroup).find(superlattice_UID) !=
                        std::string::npos) {
                        UIDcount++;
                    }
                }

                if (UIDcount != 2) {
                    std::cout << "ERROR:" << std::endl;
                    std::cout << "H5 file does not contain any superlattice "
                                 "data with provided UID "
                              << superlattice_UID << "." << std::endl;
                    world.abort(1);
                }
            }

            // load the scattering rates from the H5 file

            bool UIDmatch = true;

            for (std::size_t ngroup = 0; ngroup < subgroups.size(); ngroup++) {
                bool contains_SLdisorder =
                    (subgroups.at(ngroup).find("superlattice") !=
                     std::string::npos) &&
                    (subgroups.at(ngroup).find("disorder") !=
                     std::string::npos);

                bool contains_SLbarriers =
                    (subgroups.at(ngroup).find("superlattice") !=
                     std::string::npos) &&
                    (subgroups.at(ngroup).find("barriers") !=
                     std::string::npos);

                if (superlattice_count > 1) {
                    UIDmatch = (subgroups.at(ngroup).find(superlattice_UID) !=
                                std::string::npos);
                }

                if (contains_SLdisorder && UIDmatch) {
                    auto mysubgroup = alma::load_scattering_subgroup(
                        hdf5_path.string().c_str(),
                        subgroups.at(ngroup),
                        world);
                    w0_SLdisorder = mysubgroup.w0;
                }

                if (contains_SLbarriers && UIDmatch) {
                    auto mysubgroup = alma::load_scattering_subgroup(
                        hdf5_path.string().c_str(),
                        subgroups.at(ngroup),
                        world);
                    w0_SLbarriers = mysubgroup.w0;
                }
            }
        }

        // Build list of temperature values
        Eigen::VectorXd Tlist;

        if (logsweep) {
            Tlist = alma::logSpace(Tmin, Tmax, NT);
        }

        else {
            Tlist.setLinSpaced(NT, Tmin, Tmax);
        }

        // Create output writer
        std::stringstream outputbuffer;

        // Write file header

        if (projectConductivity) {
            outputbuffer << "Temp[K],kappaRTA<" << uvector(0) << ","
                         << uvector(1) << "," << uvector(2) << ">[W/m-K]";

            if (fullBTE) {
                outputbuffer << ",kappaBTE<" << uvector(0) << "," << uvector(1)
                             << "," << uvector(2) << ">[W/m-K]";
            }
        }

        else {
            if (!fullBTE) {
                outputbuffer << "Temp[K],kappaRTA_xx[W/m-K],kappaRTA_xy[W/"
                                "m-K],kappaRTA_xz[W/m-K],kappaRTA_yy[W/"
                                "m-K],kappaRTA_yz[W/m-K],kappaRTA_zz[W/m-K]";
            }
            else {
                outputbuffer << "Temp[K],kappaRTA_xx[W/m-K],kappaBTE_xx[W/"
                                "m-K],kappaRTA_xy[W/m-K],kappaBTE_xy[W/"
                                "m-K],kappaRTA_xz[W/m-K],kappaBTE_xz[W/"
                                "m-K],kappaRTA_yy[W/m-K],kappaBTE_yy[W/"
                                "m-K],kappaRTA_yz[W/m-K],kappaBTE_yz[W/"
                                "m-K],kappaRTA_zz[W/m-K]kappaBTE_zz[W/m-K],";
            }
        }

        if (outputCapacity) {
            outputbuffer << ",Cbulk[J/m^3-K]";
        }

        outputbuffer << std::endl;

        // RUN CALCULATIONS

        std::cout << "Running Tsweep for " << mat_base << std::endl;

        auto twoph_processes = alma::find_allowed_twoph(*grid, world);

        // temperature-independent scattering rates
        Eigen::ArrayXXd w_elastic;

        if (superlattice) {
            w_elastic = w0_SLbarriers.array() + w0_SLdisorder.array();
        }
        else {
            w_elastic =
                alma::calc_w0_twoph(*poscar, *grid, twoph_processes, world);
        }

        if (projectConductivity) {
            u_norm = uvector.array() / uvector.norm();
        }

        for (int nT = 0; nT < NT; nT++) {
            double T = Tlist(nT);

            std::cout << " Processing " << T << " K (temperature " << nT + 1
                      << " of " << NT << ")" << std::endl;

            Eigen::ArrayXXd w3(
                alma::calc_w0_threeph(*grid, *processes, T, world));
            Eigen::ArrayXXd w(w3 + w_elastic);

            Eigen::Matrix3d kappa_RTA = alma::calc_kappa(*poscar, *grid, w, T);
            Eigen::Matrix3d kappa_BTE;

            if (fullBTE) {
                kappa_BTE = alma::beyondRTA::calc_kappa(*poscar,
                                                        *grid,
                                                        *syms,
                                                        *processes,
                                                        twoph_processes,
                                                        w,
                                                        T,
                                                        fullBTE_iterative,
                                                        world);
            }

            if (projectConductivity) { // write conductivity along chosen
                                       // direction
                Eigen::MatrixXd kappa_u =
                    u_norm.transpose() * kappa_RTA.matrix() * u_norm;
                outputbuffer << T << "," << kappa_u(0, 0);

                if (fullBTE) {
                    Eigen::MatrixXd kappa_u =
                        u_norm.transpose() * kappa_BTE.matrix() * u_norm;
                    outputbuffer << "," << kappa_u(0, 0);
                }
            }

            else { // write tensor components
                if (!fullBTE) {
                    outputbuffer << T << "," << kappa_RTA(0, 0) << ","
                                 << kappa_RTA(0, 1) << ",";
                    outputbuffer << kappa_RTA(0, 2) << "," << kappa_RTA(1, 1)
                                 << ",";
                    outputbuffer << kappa_RTA(1, 2) << "," << kappa_RTA(2, 2);
                }

                else {
                    outputbuffer << T << "," << kappa_RTA(0, 0) << ","
                                 << kappa_BTE(0, 0) << ",";
                    outputbuffer << kappa_RTA(0, 1) << "," << kappa_BTE(0, 1)
                                 << ",";
                    outputbuffer << kappa_RTA(0, 2) << "," << kappa_BTE(0, 2)
                                 << ",";
                    outputbuffer << kappa_RTA(1, 1) << "," << kappa_BTE(1, 1)
                                 << ",";
                    outputbuffer << kappa_RTA(1, 2) << "," << kappa_BTE(1, 2)
                                 << ",";
                    outputbuffer << kappa_RTA(2, 2) << "," << kappa_BTE(2, 2);
                }
            }

            if (outputCapacity) {
                outputbuffer << "," << 1e27 * alma::calc_cv(*poscar, *grid, T);
            }

            outputbuffer << std::endl;
        } // END CALCULATIONS

        // WRITE FILE

        // go to the launch directory
        boost::filesystem::current_path(launch_path);

        // resolve output directory if AUTO is selected
        if (target_directory.compare("AUTO") == 0) {
            target_directory = "output/kappa_Tsweep";
        }

        // create output directory if it doesn't exist yet
        auto outputfolder = boost::filesystem::path(target_directory);

        if (!(boost::filesystem::exists(outputfolder))) {
            boost::filesystem::create_directories(outputfolder);
        }

        boost::filesystem::current_path(launch_path);

        // resolve file name if AUTO selected

        if (target_filename.compare("AUTO") == 0) {
            std::stringstream filenamebuilder;
            filenamebuilder << mat_base << "_" << gridDensityA << "_"
                            << gridDensityB << "_" << gridDensityC;

            if (projectConductivity) {
                filenamebuilder << "_" << uvector(0) << "," << uvector(1) << ","
                                << uvector(2);
            }

            filenamebuilder << "_" << Tmin << "_" << Tmax;

            filenamebuilder << ".Tsweep";
            target_filename = filenamebuilder.str();
        }

        // save file

        std::cout << std::endl;
        std::cout << "Writing to file " << target_filename << std::endl;
        std::cout << "in subdirectory " << target_directory << std::endl;
        std::cout << "under root directory " << launch_path << std::endl;

        std::ofstream outputwriter;
        outputwriter.open("./" + target_directory + "/" + target_filename);
        outputwriter << outputbuffer.str();
        outputwriter.close();

        std::cout << std::endl << "[DONE.]" << std::endl;

        return 0;
    }
}
