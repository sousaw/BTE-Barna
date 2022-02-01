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
/// EXECUTABLE THAT COMPUTES CUMULATIVE CONDUCTIVITY/CAPACITY CURVES.

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

int main(int argc, char** argv) {
    // set up MPI environment
    boost::mpi::environment env;
    boost::mpi::communicator world;

    if (world.size() > 1) {
        std::cout
            << "*** ERROR: cumulativecurves does not run in parallel mode. ***"
            << std::endl;
        world.abort(1);
    }

    if (argc < 2) {
        std::cout
            << "USAGE: cumulativecurves <inputfile.xml> <OPTIONAL:Tambient>"
            << std::endl;
        return 1;
    }

    else {
        // define variables
        std::string target_directory = "AUTO";
        std::string h5_repository = ".";
        std::string mat_directory;
        std::string mat_base;
        int gridDensityA = -1;
        int gridDensityB = -1;
        int gridDensityC = -1;
        bool superlattice = false;
        std::string superlattice_UID = "NULL";

        Eigen::Vector3d uvector(0.0, 0.0, 0.0);

        double T = 300.0;

        if (argc == 3) {
            T = atof(argv[2]);
        }

        bool output_kappa = false;
        bool output_Cv = false;
        bool output_l2 = false;
        bool resolvebyMFP = false;
        bool resolvebyProjMFP = false;
        bool resolvebyRT = false;
        bool resolvebyFreq = false;
        bool resolvebyAngFreq = false;
        bool resolvebyEnergy = false;

        int Npoints = 500;

        std::cout << "*****************************************" << std::endl;
        std::cout << "This is ALMA/cumulativecurves version "
                  << ALMA_VERSION_MAJOR << "." << ALMA_VERSION_MINOR
                  << std::endl;
        std::cout << "*****************************************" << std::endl;

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

        for (const auto& v : tree.get_child("cumulativecurves")) {
            if (v.first == "H5repository") {
                h5_repository =
                    alma::parseXMLfield<std::string>(v, "root_directory");
            }

            if (v.first == "target") {
                target_directory =
                    alma::parseXMLfield<std::string>(v, "directory");
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

            if (v.first == "transportAxis") {
                double ux = alma::parseXMLfield<double>(v, "x");
                double uy = alma::parseXMLfield<double>(v, "y");
                double uz = alma::parseXMLfield<double>(v, "z");

                uvector << ux, uy, uz;
            }

            if (v.first == "output") {
                output_kappa = alma::parseXMLfield<bool>(v, "conductivity");
                output_Cv = alma::parseXMLfield<bool>(v, "capacity");
                output_l2 = alma::parseXMLfield<bool>(v, "l2");
            }

            if (v.first == "resolveby") {
                resolvebyMFP = alma::parseXMLfield<bool>(v, "MFP");
                resolvebyProjMFP = alma::parseXMLfield<bool>(v, "projMFP");
                resolvebyRT = alma::parseXMLfield<bool>(v, "RT");
                resolvebyFreq = alma::parseXMLfield<bool>(v, "freq");
                resolvebyAngFreq = alma::parseXMLfield<bool>(v, "angfreq");
                resolvebyEnergy = alma::parseXMLfield<bool>(v, "energy");
            }

            if (v.first == "optionalsettings") {
                Npoints = alma::parseXMLfield<int>(v, "curvepoints");
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

        if (Npoints <= 0) {
            std::cout << "ERROR: requested number of curve points is "
                      << Npoints << std::endl;
            std::cout << "Value must be postive." << std::endl;
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
        h5namebuilder << mat_base << "_";
        h5namebuilder << gridDensityA << "_" << gridDensityB << "_"
                      << gridDensityC << ".h5";
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

        // calculate scattering rates at the chosen temperature
        Eigen::ArrayXXd w3(alma::calc_w0_threeph(*grid, *processes, T, world));
        Eigen::ArrayXXd w_elastic;

        if (superlattice) {
            w_elastic = w0_SLdisorder.array() + w0_SLbarriers.array();
        }
        else {
            auto twoph_processes = alma::find_allowed_twoph(*grid, world);
            w_elastic =
                alma::calc_w0_twoph(*poscar, *grid, twoph_processes, world);
        }

        Eigen::ArrayXXd w(w3 + w_elastic);

        // Create calculator and obtain basic material properties
        alma::analytic1D::BasicProperties_calculator propCalc(
            poscar.get(), grid.get(), &w, T);
        propCalc.setDirection(uvector);
        propCalc.setBulk();
        double kappa_bulk = propCalc.getConductivity();
        double C_bulk = propCalc.getCapacity();

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

        // Create base for file names
        std::stringstream filebasebuilder;
        filebasebuilder << mat_base << "_" << gridDensityA << "_"
                        << gridDensityB << "_" << gridDensityC;
        filebasebuilder << "_" << uvector(0) << "," << uvector(1) << ","
                        << uvector(2);
        filebasebuilder << "_" << T << "K";
        std::string filebase = filebasebuilder.str();

        std::cout << "Files will be written in directory " << target_directory
                  << std::endl;
        std::cout << "under base directory " << launch_path << std::endl;

        // RUN CALCULATIONS

        std::cout << "Computing curves..." << std::endl;

        Eigen::VectorXd dataX(Npoints);
        Eigen::VectorXd dataY(Npoints);

        Eigen::MatrixXd output(Npoints, 3);

        std::string filename;
        std::string fileheader;

        std::stringstream suffixbuilder;
        suffixbuilder << "<" << uvector(0) << "," << uvector(1) << ","
                      << uvector(2) << ">";
        std::string suffix = suffixbuilder.str();

        // curves versus MFP

        if (resolvebyMFP) {
            propCalc.resolveByMFP();
            propCalc.setAutoMFPbins(Npoints);
            dataX = propCalc.getMFPbins();

            if (output_kappa) {
                dataY = propCalc.getCumulativeConductivity();
                output.col(0) = dataX;
                output.col(1) = dataY;
                output.col(2) = dataY.array() / kappa_bulk;

                fileheader = "MFP[m],kappa_cumul" + suffix + "[W/m-K]," +
                             "kappa_cumul" + suffix + "/kappa_bulk" + suffix +
                             "[-]" + "\n";

                filename = filebase + ".kappacumul_MFP";
                std::cout << "Writing kappa_cumul(MFP) to file " << filename
                          << std::endl;

                filename = "./" + target_directory + "/" + filename;
                alma::write_to_csv(filename, output, ',', false, fileheader);
            }

            if (output_Cv) {
                dataY = propCalc.getCumulativeCapacity();
                output.col(0) = dataX;
                output.col(1) = dataY;
                output.col(2) = dataY.array() / C_bulk;

                fileheader = "MFP[m],C_cumul[J/m^3-K],C_cumul/C_bulk[-]\n";

                filename = filebase + ".Ccumul_MFP";
                std::cout << "Writing C_cumul(MFP) to file " << filename
                          << std::endl;

                filename = "./" + target_directory + "/" + filename;
                alma::write_to_csv(filename, output, ',', false, fileheader);
            }


            if (output_l2) {
                dataY = propCalc.getCumulativel2RTAiso();
                output.col(0) = dataX;
                output.col(1) = dataY;
                output.col(2) = dataY.array() / dataY.maxCoeff();

                fileheader = "MFP[m],l2_cumul[m^2],l2_cumul/l2_bulk[-]\n";

                filename = filebase + ".l2cumul_MFP";
                std::cout << "Writing l2_cumul(MFP) to file " << filename
                          << std::endl;

                filename = "./" + target_directory + "/" + filename;
                alma::write_to_csv(filename, output, ',', false, fileheader);
            }
        }

        // curves versus proj MFP

        if (resolvebyProjMFP) {
            propCalc.resolveByProjMFP();
            propCalc.setAutoProjMFPbins(Npoints);
            dataX = propCalc.getMFPbins();

            if (output_kappa) {
                dataY = propCalc.getCumulativeConductivity();
                output.col(0) = dataX;
                output.col(1) = dataY;
                output.col(2) = dataY.array() / kappa_bulk;

                fileheader = "MFP" + suffix + "[m],kappa_cumul" + suffix +
                             "[W/m-K]," + "kappa_cumul" + suffix +
                             "/kappa_bulk" + suffix + "[-]" + "\n";

                filename = filebase + ".kappacumul_projMFP";
                std::cout << "Writing kappa_cumul(projMFP) to file " << filename
                          << std::endl;

                filename = "./" + target_directory + "/" + filename;
                alma::write_to_csv(filename, output, ',', false, fileheader);
            }

            if (output_Cv) {
                dataY = propCalc.getCumulativeCapacity();
                output.col(0) = dataX;
                output.col(1) = dataY;
                output.col(2) = dataY.array() / C_bulk;

                fileheader =
                    "MFP" + suffix + "[m],C_cumul[J/m^3-K],C_cumul/C_bulk[-]\n";

                filename = filebase + ".Ccumul_projMFP";
                std::cout << "Writing C_cumul(projMFP) to file " << filename
                          << std::endl;

                filename = "./" + target_directory + "/" + filename;
                alma::write_to_csv(filename, output, ',', false, fileheader);
            }

            if (output_l2) {
                dataY = propCalc.getCumulativel2RTAiso();
                output.col(0) = dataX;
                output.col(1) = dataY;
                output.col(2) = dataY.array() / dataY.maxCoeff();

                fileheader =
                    "MFP" + suffix + "[m],l2_cumul[m^2],l2_cumul/l2_bulk[-]\n";

                filename = filebase + ".l2cumul_projMFP";
                std::cout << "Writing l2_cumul(projMFP) to file " << filename
                          << std::endl;

                filename = "./" + target_directory + "/" + filename;
                alma::write_to_csv(filename, output, ',', false, fileheader);
            }
        }

        // curves versus RT

        if (resolvebyRT) {
            propCalc.resolveByRT();
            propCalc.setAutoRTbins(Npoints);
            dataX = propCalc.getRTbins();

            if (output_kappa) {
                dataY = propCalc.getCumulativeConductivity();
                output.col(0) = dataX;
                output.col(1) = dataY;
                output.col(2) = dataY.array() / kappa_bulk;

                fileheader = "RT[s],kappa_cumul" + suffix + "[W/m-K]," +
                             "kappa_cumul" + suffix + "/kappa_bulk" + suffix +
                             "[-]" + "\n";

                filename = filebase + ".kappacumul_RT";
                std::cout << "Writing kappa_cumul(RT) to file " << filename
                          << std::endl;

                filename = "./" + target_directory + "/" + filename;
                alma::write_to_csv(filename, output, ',', false, fileheader);
            }

            if (output_Cv) {
                dataY = propCalc.getCumulativeCapacity();
                output.col(0) = dataX;
                output.col(1) = dataY;
                output.col(2) = dataY.array() / C_bulk;

                fileheader = "RT[s],C_cumul[J/m^3-K],C_cumul/C_bulk[-]\n";

                filename = filebase + ".Ccumul_RT";
                std::cout << "Writing C_cumul(RT) to file " << filename
                          << std::endl;

                filename = "./" + target_directory + "/" + filename;
                alma::write_to_csv(filename, output, ',', false, fileheader);
            }


            if (output_l2) {
                dataY = propCalc.getCumulativel2RTAiso();
                output.col(0) = dataX;
                output.col(1) = dataY;
                output.col(2) = dataY.array() / dataY.maxCoeff();

                fileheader = "RT[s],l2_cumul[m^2],l2_cumul/l2_bulk[-]\n";

                filename = filebase + ".l2cumul_RT";
                std::cout << "Writing l2_cumul(RT) to file " << filename
                          << std::endl;

                filename = "./" + target_directory + "/" + filename;
                alma::write_to_csv(filename, output, ',', false, fileheader);
            }
        }

        // curves versus frequency and related variables

        if (resolvebyFreq || resolvebyAngFreq || resolvebyEnergy) {
            propCalc.resolveByOmega();
            propCalc.setAutoOmegabins(Npoints);
            dataX = propCalc.getOmegabins();

            if (output_kappa) {
                dataY = propCalc.getCumulativeConductivity();
                output.col(1) = dataY;
                output.col(2) = dataY.array() / kappa_bulk;

                if (resolvebyFreq) {
                    output.col(0) =
                        dataX.array() / (1e12 * 2.0 * alma::constants::pi);

                    fileheader = "nu[THz],kappa_cumul" + suffix + "[W/m-K]," +
                                 "kappa_cumul" + suffix + "/kappa_bulk" +
                                 suffix + "[-]" + "\n";

                    filename = filebase + ".kappacumul_freq";
                    std::cout << "Writing kappa_cumul(freq) to file "
                              << filename << std::endl;

                    filename = "./" + target_directory + "/" + filename;
                    alma::write_to_csv(
                        filename, output, ',', false, fileheader);
                }

                if (resolvebyAngFreq) {
                    output.col(0) = dataX.array() / 1e12;

                    fileheader = "omega[rad/ps],kappa_cumul" + suffix +
                                 "[W/m-K]," + "kappa_cumul" + suffix +
                                 "/kappa_bulk" + suffix + "[-]" + "\n";

                    filename = filebase + ".kappacumul_angfreq";
                    std::cout << "Writing kappa_cumul(angfreq) to file "
                              << filename << std::endl;

                    filename = "./" + target_directory + "/" + filename;
                    alma::write_to_csv(
                        filename, output, ',', false, fileheader);
                }

                if (resolvebyEnergy) {
                    output.col(0) =
                        (1e3 * alma::constants::hbar / alma::constants::e) *
                        dataX.array();

                    fileheader = "E[meV],kappa_cumul" + suffix + "[W/m-K]," +
                                 "kappa_cumul" + suffix + "/kappa_bulk" +
                                 suffix + "[-]" + "\n";

                    filename = filebase + ".kappacumul_energy";
                    std::cout << "Writing kappa_cumul(energy) to file "
                              << filename << std::endl;

                    filename = "./" + target_directory + "/" + filename;
                    alma::write_to_csv(
                        filename, output, ',', false, fileheader);
                }
            }

            if (output_Cv) {
                dataY = propCalc.getCumulativeCapacity();
                output.col(1) = dataY;
                output.col(2) = dataY.array() / C_bulk;

                if (resolvebyFreq) {
                    output.col(0) =
                        dataX.array() / (1e12 * 2.0 * alma::constants::pi);

                    fileheader = "nu[THz],C_cumul[J/m^3-K],C_cumul/C_bulk[-]\n";

                    filename = filebase + ".Ccumul_freq";
                    std::cout << "Writing C_cumul(freq) to file " << filename
                              << std::endl;

                    filename = "./" + target_directory + "/" + filename;
                    alma::write_to_csv(
                        filename, output, ',', false, fileheader);
                }

                if (resolvebyAngFreq) {
                    output.col(0) = dataX.array() / 1e12;

                    fileheader =
                        "omega[rad/ps],C_cumul[J/m^3-K],C_cumul/C_bulk[-]\n";

                    filename = filebase + ".Ccumul_angfreq";
                    std::cout << "Writing C_cumul(angfreq) to file " << filename
                              << std::endl;

                    filename = "./" + target_directory + "/" + filename;
                    alma::write_to_csv(
                        filename, output, ',', false, fileheader);
                }

                if (resolvebyEnergy) {
                    output.col(0) =
                        (1e3 * alma::constants::hbar / alma::constants::e) *
                        dataX.array();

                    fileheader = "E[meV],C_cumul[J/m^3-K],C_cumul/C_bulk[-]\n";

                    filename = filebase + ".Ccumul_energy";
                    std::cout << "Writing C_cumul(energy) to file " << filename
                              << std::endl;

                    filename = "./" + target_directory + "/" + filename;
                    alma::write_to_csv(
                        filename, output, ',', false, fileheader);
                }
            }


            if (output_l2) {
                dataY = propCalc.getCumulativel2RTAiso();
                output.col(1) = dataY;
                output.col(2) = dataY.array() / dataY.maxCoeff();

                if (resolvebyFreq) {
                    output.col(0) =
                        dataX.array() / (1e12 * 2.0 * alma::constants::pi);

                    fileheader = "nu[THz],l2_cumul[m^2],l2_cumul/l2_bulk[-]\n";

                    filename = filebase + ".l2cumul_freq";
                    std::cout << "Writing l2_cumul(freq) to file " << filename
                              << std::endl;

                    filename = "./" + target_directory + "/" + filename;
                    alma::write_to_csv(
                        filename, output, ',', false, fileheader);
                }

                if (resolvebyAngFreq) {
                    output.col(0) = dataX.array() / 1e12;

                    fileheader =
                        "omega[rad/ps],l2_cumul[m^2],l2_cumul/l2_bulk[-]\n";

                    filename = filebase + ".l2cumul_angfreq";
                    std::cout << "Writing l2_cumul(angfreq) to file "
                              << filename << std::endl;

                    filename = "./" + target_directory + "/" + filename;
                    alma::write_to_csv(
                        filename, output, ',', false, fileheader);
                }

                if (resolvebyEnergy) {
                    output.col(0) =
                        (1e3 * alma::constants::hbar / alma::constants::e) *
                        dataX.array();

                    fileheader = "E[meV],l2_cumul[m^2],l2_cumul/l2_bulk[-]\n";

                    filename = filebase + ".l2cumul_energy";
                    std::cout << "Writing l2_cumul(energy) to file " << filename
                              << std::endl;

                    filename = "./" + target_directory + "/" + filename;
                    alma::write_to_csv(
                        filename, output, ',', false, fileheader);
                }
            }
        }

        // END CALCULATIONS

        std::cout << std::endl << "[DONE.]" << std::endl;

        return 0;
    }
}
