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
/// Builds HDF5 material files from XML input.

#include <iostream>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include <cmakevars.hpp>
#include <utilities.hpp>
#include <structures.hpp>
#include <vc.hpp>
#include <vasp_io.hpp>
#include <bulk_hdf5.hpp>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <Eigen/Dense>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/iostreams/device/null.hpp>
#include <io_utils.hpp>

boost::iostreams::stream_buffer<boost::iostreams::null_sink> null_buf{
    boost::iostreams::null_sink()};

// TYPES OF MATERIALS THE BUILDER CAN HANDLE //
enum materialTypes { singlecrystal, alloy, parametricalloy };

/////////// GLOBAL PLACEHOLDERS ////////////
// list of compound names
std::vector<std::string> materialBase;
// number of wavevector points along A-axis
int gridDensityA;
// number of wavevector points along B-axis
int gridDensityB;
// number of wavevector points along C-axis
int gridDensityC;
// Scalebroad
double scalebroad = 0.1;
// root directory
std::string materials_repository = ".";
// target directory
std::string target_directory = "AUTO";
// target filename
std::string target_filename = "AUTO";
// type of material
int materialType = -1;
// list of mixfractions for alloys
std::vector<double> mixFraction;
// list of mix values for parametric alloys
std::map<std::string, std::vector<double>> parametric_mixvalues;
std::map<std::string, std::vector<double>> parametric_allcombinations;
// list that stores the mix parameter associated
// with each compound in parametric alloys
std::vector<std::string> mixparameterName;
// overwrite H5 file if it already exists?
bool overwrite = false;
////////////////////////////////////////////

// Helper function to list combinations of parametric mix fractions
Eigen::MatrixXd matrixRepeater(const Eigen::Ref<Eigen::MatrixXd> sourcematrix,
                               const Eigen::Ref<Eigen::VectorXd> sourcevector) {
    int Nrows = sourcematrix.rows();
    int Ncols = sourcematrix.cols();
    int Nadditions = sourcevector.size();

    Eigen::MatrixXd result(Nrows * Nadditions, Ncols + 1);

    for (int n = 0; n < Nadditions; n++) {
        Eigen::VectorXd entries(Nrows);
        entries.setConstant(sourcevector(n));

        result.block(n * Nrows, 0, Nrows, Ncols) = sourcematrix;
        result.block(n * Nrows, Ncols, Nrows, 1) = entries;
    }

    return result;
}


// Helper function that constructs labeled map based
// on all possible parametric mixfraction combinations

std::map<std::string, std::vector<double>> parametric_combinations(
    std::map<std::string, std::vector<double>>& sourcemap) {
    if (sourcemap.size() <= 1) {
        return sourcemap;
    }

    else {
        std::map<std::string, std::vector<double>> result;

        // turn first map entry into Eigen::MatrixXd
        Eigen::MatrixXd initial_matrix;
        initial_matrix.resize(sourcemap.begin()->second.size(), 1);

        for (std::size_t n = 0; n < sourcemap.begin()->second.size(); n++) {
            initial_matrix(n, 0) = sourcemap.begin()->second.at(n);
        }

        Eigen::MatrixXd combination_matrix;
        Eigen::MatrixXd buffer_matrix = initial_matrix;

        std::map<std::string, std::vector<double>>::iterator iter =
            sourcemap.begin();
        iter++; // points to second map entry

        for (; iter != sourcemap.end(); iter++) {
            // convert the considered map list into Eigen::VectorXd
            Eigen::VectorXd buffer_vector(iter->second.size());

            for (std::size_t n = 0; n < iter->second.size(); n++) {
                buffer_vector(n) = iter->second.at(n);
            }

            combination_matrix = matrixRepeater(buffer_matrix, buffer_vector);
            buffer_matrix = combination_matrix;
        }

        // turn the obtained combination matrix into the desired map

        int ncol = 0;
        int Nrows = combination_matrix.rows();

        for (iter = sourcemap.begin(); iter != sourcemap.end(); iter++) {
            std::vector<double> valuelist;
            valuelist.reserve(Nrows);

            for (int nrow = 0; nrow < Nrows; nrow++) {
                valuelist.emplace_back(combination_matrix(nrow, ncol));
            }

            result[iter->first] = valuelist;
            ncol++;
        }

        return result;
    }
}

///////////////////// BUILDER FOR SINGLE CRYSTALS //////////////////

void singleCrystalBuilder(boost::mpi::communicator world) {
    std::string compound = materialBase.at(0);

    // Initialise file system and verify that directories actually exist
    auto launch_path = boost::filesystem::current_path();
    auto basedir = boost::filesystem::path(materials_repository);

    if (!(boost::filesystem::exists(boost::filesystem::path(basedir)))) {
        std::cout << "ERROR:" << std::endl;
        std::cout << "Repository directory " << basedir << " does not exist."
                  << std::endl;
        world.abort(1);
    }

    if (!(boost::filesystem::exists(
            boost::filesystem::path(basedir / compound)))) {
        std::cout << "ERROR:" << std::endl;
        std::cout << "Material directory " << compound
                  << " does not exist within the repository." << std::endl;
        world.abort(1);
    }

    // CREATE FILENAME AND DIRECTORY

    // go to the launch directory
    boost::filesystem::current_path(launch_path);

    if (target_directory.compare("AUTO") == 0) {
        target_directory = compound;
    }
    auto output_dir = boost::filesystem::path(target_directory);

    // create output directory if it doesn't exist yet

    auto outputfolder = boost::filesystem::path(output_dir);

    if (!(boost::filesystem::exists(outputfolder))) {
        boost::filesystem::create_directories(outputfolder);
    }

    // create filename if AUTO is selected

    std::string filename = target_filename;

    if (filename.compare("AUTO") == 0) {
        std::stringstream buffer;
        buffer << compound << "_" << gridDensityA << "_" << gridDensityB << "_"
               << gridDensityC << ".h5";
        filename = buffer.str();
    }

    auto h5_target_file = output_dir / boost::filesystem::path(filename);

    // IF OVERWRITE IS TURNED OFF, MAKE SURE TARGETED HDF5 FILE DOESN'T EXIST
    // YET.

    if (!overwrite) {
        if (boost::filesystem::exists(h5_target_file)) {
            std::cout << std::endl;
            std::cout
                << "INFO: Target HDF5 already exists, no computations needed."
                << std::endl;
            std::cout << "If you wish to force (re)creation of this compound,"
                      << std::endl;
            std::cout << "change one of the following in the XML input file:"
                      << std::endl;
            std::cout << " (1) Turn on overwrite option using <overwrite/>"
                      << std::endl;
            std::cout << " (2) Specify alternate target directory using "
                         "<target directory=\"[target_dir]\"/>"
                      << std::endl;
            std::cout << std::endl;
            world.abort(1);
        }
    }

    auto dir = basedir / boost::filesystem::path(compound);

    // RETRIEVE SUPERCELL PARAMETERS

    auto metadata_path = dir / boost::filesystem::path("_metadata");

    if (!boost::filesystem::exists(metadata_path)) {
        std::cout << "ERROR:" << std::endl;
        std::cout << "_metadata file for " << compound << " is missing."
                  << std::endl;
        world.abort(1);
    }

    std::ifstream metadatareader;
    metadatareader.open(metadata_path.c_str());

    std::string linereader;
    // read first line (compound info) and skip
    getline(metadatareader, linereader, '\n');
    // read second line (contains 2nd-order IFC supercell parameters)
    getline(metadatareader, linereader, '\n');
    metadatareader.close();
    std::size_t index = linereader.find("=");
    std::stringstream metadataextractor(linereader.substr(index + 1));
    int scA = -1;
    int scB = -1;
    int scC = -1;
    metadataextractor >> scA;
    metadataextractor >> scB;
    metadataextractor >> scC;

    // automatically determine if the compound is polar or not
    bool polar = true;

    auto born_path = dir / boost::filesystem::path("BORN");

    std::unique_ptr<alma::Dielectric_parameters> born;

    try {
        born = alma::load_BORN(born_path.string().c_str());
    }
    catch (const alma::value_error&) {
        polar = false;
    }

    std::cout << "***********************************" << std::endl;
    std::cout << "This is ALMA/VCAbuilder version " << ALMA_VERSION_MAJOR << "."
              << ALMA_VERSION_MINOR << std::endl;
    std::cout << "***********************************" << std::endl;

    std::cout << "Generating ";

    if (!polar) {
        std::cout << "non";
    }
    std::cout << "polar singlecrystal " << compound << std::endl;

    // IMPORT CRYSTAL INFORMATION

    std::cout << "Loading POSCAR" << std::endl;
    auto poscar_path = dir / boost::filesystem::path("POSCAR");
    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());

    std::cout << "Loading force constants 2nd order" << std::endl;
    auto ifc_path = dir / boost::filesystem::path("FORCE_CONSTANTS");
    auto ifcs = alma::load_FORCE_CONSTANTS(
        ifc_path.string().c_str(), *poscar, scA, scB, scC);

    std::cout << "Loading force constants 3rd order" << std::endl;
    auto thirdorder_path = dir / boost::filesystem::path("FORCE_CONSTANTS_3RD");
    auto thirdorder = alma::load_FORCE_CONSTANTS_3RD(
        thirdorder_path.string().c_str(), *poscar);

    std::cout << "Calculating symmetries" << std::endl;
    auto syms = alma::Symmetry_operations(*poscar);

    std::cout << "Generating ";
    std::cout << gridDensityA << "x" << gridDensityB << "x" << gridDensityC;
    std::cout << " wavevector grid" << std::endl;

    std::unique_ptr<alma::Gamma_grid> grid;

    if (polar) {
        grid = std::make_unique<alma::Gamma_grid>(*poscar,
                                                  syms,
                                                  *ifcs,
                                                  *born,
                                                  gridDensityA,
                                                  gridDensityB,
                                                  gridDensityC);
    }

    else {
        grid = std::make_unique<alma::Gamma_grid>(
            *poscar, syms, *ifcs, gridDensityA, gridDensityB, gridDensityC);
    }

    grid->enforce_asr();

    std::cout << "Finding 3-phonon processes" << std::endl;
    std::cout << "using scalebroad = " << scalebroad << std::endl;
    // find allowed phonon processes
    auto threeph_processes =
        alma::find_allowed_threeph(*grid, world, scalebroad);
    std::cout << "Computing matrix elements" << std::endl;

    std::size_t targetpercentage = 1;

    // Precompute the matrix elements.
    for (std::size_t i = 0; i < threeph_processes.size(); ++i) {
        while (100 * (i + 1) / threeph_processes.size() >= targetpercentage) {
            std::cout << "  " << targetpercentage << "% \t";
            std::cout.flush();

            if (targetpercentage % 10 == 0) {
                std::cout << std::endl;
            }
            targetpercentage++;
        }

        threeph_processes[i].compute_vp2(*poscar, *grid, *thirdorder);
    }

    // WRITE OUTPUT

    std::cout << "Writing to file " << filename << std::endl;
    std::cout << "in directory " << output_dir << std::endl;

    alma::save_bulk_hdf5(h5_target_file.string().c_str(),
                         filename,
                         *poscar,
                         syms,
                         *grid,
                         threeph_processes,
                         world);

    std::cout << std::endl << "[DONE.]" << std::endl;
}


int alloyBuilder() {
    boost::mpi::communicator world;

    // number of compounds
    int Nmat = materialBase.size();

    // Initialise file system and verify that directories actually exist
    auto launch_path = boost::filesystem::current_path();
    auto basedir = boost::filesystem::path(materials_repository);

    bool base_error = false;

    for (std::size_t n = 0; n < materialBase.size(); n++) {
        if (!(boost::filesystem::exists(
                boost::filesystem::path(basedir / materialBase.at(n))))) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "Base directory " << materialBase.at(n)
                      << " does not exist." << std::endl;
            base_error = true;
        }
    }

    if (base_error) {
        world.abort(1);
    }

    // CREATE FILENAME

    // gather all atomic elements present in the system

    std::vector<std::vector<std::string>> elementList;
    boost::filesystem::current_path(basedir);

    for (int nmat = 0; nmat < Nmat; nmat++) {
        auto compound_dir = boost::filesystem::path(materialBase.at(nmat));
        auto poscar_path = compound_dir / boost::filesystem::path("POSCAR");

        if (!boost::filesystem::exists(poscar_path)) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "POSCAR file for " << materialBase.at(nmat)
                      << " is missing." << std::endl;
            world.abort(1);
        }

        auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
        elementList.emplace_back(poscar->elements);
    }

    // determine multiplicity and mix fraction for each element
    std::map<std::string, int> element_count;
    std::map<std::string, double> element_mixfraction;

    for (int nmat = 0; nmat < Nmat; nmat++) {
        for (std::size_t nel = 0; nel < elementList.at(nmat).size(); nel++) {
            element_count[elementList.at(nmat).at(nel)]++;
            element_mixfraction[elementList.at(nmat).at(nel)] +=
                mixFraction.at(nmat);
        }
    }

    // create lookup table that keeps track which elements
    // have been added to the filename already
    std::map<std::string, bool> element_todo;

    // find maximum multiplicity and fill lookup table

    int mpc_max = -1;

    for (std::map<std::string, int>::iterator it = element_count.begin();
         it != element_count.end();
         it++) {
        if (it->second > mpc_max) {
            mpc_max = it->second;
        }
        element_todo[it->first] = true;
    }

    // Construct name from elements sorted by multiplicity (low to high).
    // If elements have the same multiplicity, use the order in which
    // they appear in the provided compounds.

    std::stringstream target_file_builder;
    std::stringstream target_directory_builder;

    for (int mpc = 1; mpc <= mpc_max; mpc++) {
        for (int nmat = 0; nmat < Nmat; nmat++) {
            for (std::size_t nel = 0; nel < elementList.at(nmat).size();
                 nel++) {
                std::string elem = elementList.at(nmat).at(nel);

                if ((element_count[elem] == mpc) && element_todo[elem]) {
                    target_directory_builder << elem;
                    target_file_builder << elem;

                    if (element_mixfraction[elem] < 1.0) {
                        target_file_builder << element_mixfraction[elem];
                    }
                    element_todo[elem] = false;
                }
            }
        }
    }

    if (target_directory.compare("AUTO") == 0) {
        target_directory = target_directory_builder.str();
    }

    target_file_builder << "_" << gridDensityA << "_" << gridDensityB << "_"
                        << gridDensityC << ".h5";
    target_filename = target_file_builder.str();

    // create directory for the alloy if it doesn't exist yet

    boost::filesystem::current_path(launch_path);
    auto targetfolder = boost::filesystem::path(target_directory);

    if (!(boost::filesystem::exists(targetfolder))) {
        boost::filesystem::create_directories(targetfolder);
    }

    auto alloy_dir = boost::filesystem::path(target_directory);
    auto h5_target_file = alloy_dir / boost::filesystem::path(target_filename);

    // AUTOMATICALLY DETERMINE POLARITY AND SUPERCELL INFO OF THE COMPOUNDS
    std::set<bool> polarity_checker;
    std::set<int> scA_checker;
    std::set<int> scB_checker;
    std::set<int> scC_checker;

    std::vector<std::unique_ptr<alma::Dielectric_parameters>> born_pointers;

    boost::filesystem::current_path(basedir);

    for (int nmat = 0; nmat < Nmat; nmat++) {
        // obtain polarity information

        bool polar = true;

        auto compound_dir = boost::filesystem::path(materialBase.at(nmat));
        auto born_path = compound_dir / boost::filesystem::path("BORN");

        std::unique_ptr<alma::Dielectric_parameters> born;

        try {
            born = alma::load_BORN(born_path.string().c_str());
        }
        catch (const alma::value_error&) {
            polar = false;
        }

        polarity_checker.insert(polar);

        if (polarity_checker.size() == 2) {
            std::cout << "ERROR: Polarity mismatch in alloy components:"
                      << std::endl;

            for (int nprevmat = 0; nprevmat < nmat; nprevmat++) {
                std::cout << "  Compound " << materialBase.at(nprevmat)
                          << " is ";

                if (polar) {
                    std::cout << "non";
                }
                std::cout << "polar" << std::endl;
            }
            std::cout << "  Compound " << materialBase.at(nmat) << " is ";

            if (!polar) {
                std::cout << "non";
            }
            std::cout << "polar" << std::endl;

            world.abort(1);
        }

        if (polar) {
            born_pointers.emplace_back(std::move(born));
        }

        // retrieve the supercell parameters

        int scA = -1;
        int scB = -1;
        int scC = -1;

        auto metadata_path =
            compound_dir / boost::filesystem::path("_metadata");

        if (!boost::filesystem::exists(metadata_path)) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "_metadata file for " << materialBase.at(nmat)
                      << " is missing." << std::endl;
            world.abort(1);
        }

        std::ifstream metadatareader;
        metadatareader.open(metadata_path.c_str());
        std::string linereader;
        getline(metadatareader, linereader, '\n');
        getline(metadatareader, linereader, '\n');
        metadatareader.close();
        std::size_t index = linereader.find("=");
        std::stringstream metadataextractor(linereader.substr(index + 1));
        metadataextractor >> scA;
        metadataextractor >> scB;
        metadataextractor >> scC;

        if (nmat == 0) {
            scA_checker.insert(scA);
            scB_checker.insert(scB);
            scC_checker.insert(scC);
        }

        if ((scA_checker.count(scA) == 0) || (scB_checker.count(scB) == 0) ||
            (scC_checker.count(scC) == 0)) {
            int scA0 = *(scA_checker.begin());
            int scB0 = *(scB_checker.begin());
            int scC0 = *(scC_checker.begin());

            std::cout
                << "ERROR: Mismatch in supercell sizes of force constant files:"
                << std::endl;

            for (int nprevmat = 0; nprevmat < nmat; nprevmat++) {
                std::cout << "  Compound " << materialBase.at(nprevmat);
                std::cout << " was computed with supercell ";
                std::cout << "(" << scA0 << "," << scB0 << "," << scC0 << ")"
                          << std::endl;
            }
            std::cout << "  Compound " << materialBase.at(nmat);
            std::cout << " was computed with supercell ";
            std::cout << "(" << scA << "," << scB << "," << scC << ")"
                      << std::endl;

            world.abort(1);
        }
    } // end polarity/supercell retrieval

    bool polar = *(polarity_checker.begin());
    int scA = *(scA_checker.begin());
    int scB = *(scB_checker.begin());
    int scC = *(scC_checker.begin());

    // Obtain mixing ratios in percent (used in virtual crystal operations
    // below)
    std::vector<double> ratios(Nmat);

    for (int nmat = 0; nmat < Nmat; nmat++) {
        ratios.at(nmat) = 100.0 * mixFraction.at(nmat);
    }

    std::cout << "***********************************" << std::endl;
    std::cout << "This is ALMA/VCAbuilder version " << ALMA_VERSION_MAJOR << "."
              << ALMA_VERSION_MINOR << std::endl;
    std::cout << "***********************************" << std::endl;


    std::size_t cutoff = target_filename.find("_");
    std::cout << "Generating ";

    if (!polar) {
        std::cout << "non";
    }
    std::cout << "polar alloy " << target_filename.substr(0, cutoff)
              << std::endl;

    // IF OVERWRITE IS TURNED OFF, MAKE SURE TARGETED HDF5 FILE DOESN'T EXIST
    // YET.

    boost::filesystem::current_path(launch_path);

    if (!overwrite) {
        if (boost::filesystem::exists(h5_target_file)) {
            std::cout << std::endl;
            std::cout
                << "INFO: Target HDF5 already exists, no computations needed."
                << std::endl;
            std::cout << "If you wish to force (re)creation of this compound,"
                      << std::endl;
            std::cout << "change one of the following in the XML input file:"
                      << std::endl;
            std::cout << " (1) Turn on overwrite option using <overwrite/>"
                      << std::endl;
            std::cout << " (2) Specify alternate target directory using "
                         "<target directory=\"[target_dir]\"/>"
                      << std::endl;
            std::cout << std::endl;
            world.abort(1);
        }
    }

    // IMPORT AND STORE CRYSTAL INFORMATION FOR EACH OF THE COMPOUNDS

    std::vector<std::unique_ptr<alma::Crystal_structure>> poscar_pointers;
    std::vector<std::unique_ptr<alma::Harmonic_ifcs>> IFC2_pointers;
    std::vector<std::unique_ptr<std::vector<alma::Thirdorder_ifcs>>>
        IFC3_pointers;

    boost::filesystem::current_path(basedir);

    for (int nmat = 0; nmat < Nmat; nmat++) {
        std::cout << "Loading crystal information for " << materialBase.at(nmat)
                  << std::endl;

        std::unique_ptr<alma::Crystal_structure> poscar;
        std::unique_ptr<alma::Harmonic_ifcs> IFC2;
        std::unique_ptr<std::vector<alma::Thirdorder_ifcs>> IFC3;

        auto compound_dir = boost::filesystem::path(materialBase.at(nmat));

        // load poscar

        auto poscar_path = compound_dir / boost::filesystem::path("POSCAR");

        if (!boost::filesystem::exists(poscar_path)) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "POSCAR file for " << materialBase.at(nmat)
                      << " is missing." << std::endl;
            world.abort(1);
        }

        poscar = alma::load_POSCAR(poscar_path.string().c_str());

        // load 2nd-order force constants

        auto ifc2_path =
            compound_dir / boost::filesystem::path("FORCE_CONSTANTS");

        if (!boost::filesystem::exists(ifc2_path)) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "FORCE_CONSTANTS file for " << materialBase.at(nmat)
                      << " is missing." << std::endl;
            world.abort(1);
        }

        IFC2 = alma::load_FORCE_CONSTANTS(
            ifc2_path.string().c_str(), *poscar, scA, scB, scC);

        // load 3rd-order force constants

        auto ifc3_path =
            compound_dir / boost::filesystem::path("FORCE_CONSTANTS_3RD");

        if (!boost::filesystem::exists(ifc3_path)) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "FORCE_CONSTANTS_3RD file for "
                      << materialBase.at(nmat) << " is missing." << std::endl;
            world.abort(1);
        }

        IFC3 =
            alma::load_FORCE_CONSTANTS_3RD(ifc3_path.string().c_str(), *poscar);

        // STORE INFORMATION
        // (Only do this now since poscar pointer will point to NULL after move)

        poscar_pointers.emplace_back(std::move(poscar));
        IFC2_pointers.emplace_back(std::move(IFC2));
        IFC3_pointers.emplace_back(std::move(IFC3));
    }

    // CONSTRUCT VIRTUAL CRYSTAL

    std::cout << "Performing virtual crystal operations" << std::endl;

    std::vector<alma::Crystal_structure> poscar_list;
    std::vector<alma::Harmonic_ifcs> IFC2_list;
    std::vector<std::vector<alma::Thirdorder_ifcs>> IFC3_list;
    std::vector<alma::Dielectric_parameters> born_list;

    for (int nmat = 0; nmat < Nmat; nmat++) {
        poscar_list.emplace_back(*poscar_pointers.at(nmat));
        IFC2_list.emplace_back(*IFC2_pointers.at(nmat));
        IFC3_list.emplace_back(*IFC3_pointers.at(nmat));

        if (polar) {
            born_list.emplace_back(*born_pointers.at(nmat));
        }
    }

    auto vc_poscar = alma::vc_mix_structures(poscar_list, ratios);
    auto vc_ifcs = alma::vc_mix_harmonic_ifcs(IFC2_list, ratios);
    auto vc_thirdorder = alma::vc_mix_thirdorder_ifcs(IFC3_list, ratios);

    std::unique_ptr<alma::Dielectric_parameters> vc_born;

    if (polar) {
        vc_born = alma::vc_mix_dielectric_parameters(born_list, ratios);
    }

    auto syms = alma::Symmetry_operations(*vc_poscar);

    std::cout << "Generating ";
    std::cout << gridDensityA << "x" << gridDensityB << "x" << gridDensityC;
    std::cout << " wavevector grid" << std::endl;

    std::unique_ptr<alma::Gamma_grid> grid;

    if (polar) {
        grid = std::make_unique<alma::Gamma_grid>(*vc_poscar,
                                                  syms,
                                                  *vc_ifcs,
                                                  *vc_born,
                                                  gridDensityA,
                                                  gridDensityB,
                                                  gridDensityC);
    }

    else {
        grid = std::make_unique<alma::Gamma_grid>(*vc_poscar,
                                                  syms,
                                                  *vc_ifcs,
                                                  gridDensityA,
                                                  gridDensityB,
                                                  gridDensityC);
    }

    grid->enforce_asr();

    std::cout << "Finding 3-phonon processes" << std::endl;
    std::cout << "using scalebroad = " << scalebroad << std::endl;

    // find allowed phonon processes
    auto threeph_processes =
        alma::find_allowed_threeph(*grid, world, scalebroad);

    std::cout << "Computing matrix elements" << std::endl;

    std::size_t targetpercentage = 1;

    // Precompute the matrix elements.
    for (std::size_t i = 0; i < threeph_processes.size(); ++i) {
        while (100 * (i + 1) / threeph_processes.size() >= targetpercentage) {
            std::cout << "  " << targetpercentage << "% \t";
            std::cout.flush();

            if (targetpercentage % 10 == 0) {
                std::cout << std::endl;
            }
            targetpercentage++;
        }

        threeph_processes[i].compute_vp2(*vc_poscar, *grid, *vc_thirdorder);
    }

    // WRITE OUTPUT

    boost::filesystem::current_path(launch_path);

    std::cout << "Writing to file " << target_filename << std::endl;
    std::cout << "in directory " << alloy_dir << std::endl;

    alma::save_bulk_hdf5(h5_target_file.string().c_str(),
                         target_filename,
                         *vc_poscar,
                         syms,
                         *grid,
                         threeph_processes,
                         world);

    return 0;
}


int main(int argc, char** argv) {
    boost::mpi::environment env{argc, argv};
    boost::mpi::communicator world;
    auto my_id = world.rank();

    // make sure we only write out messages from the master process

    if (my_id != 0) {
        std::cout.rdbuf(&null_buf);
    }

    if (argc < 2) {
        std::cout << "usage: VCAbuilder <inputfile.xml> <OPTIONAL:griddensity>"
                  << std::endl;
        return 1;
    }

    std::string xmlfile(argv[1]);

    bool override_gridDensity = false;
    int external_gridDensity = -1;

    if (argc == 3) {
        external_gridDensity = atoi(argv[2]);
        override_gridDensity = true;
    }

    /////////////////////////
    /// PARSE INPUT FILE  ///
    /////////////////////////

    std::cout << "PARSING " << xmlfile << " ..." << std::endl;

    // Create empty property tree object
    boost::property_tree::ptree tree;

    // Parse XML input file into the tree
    boost::property_tree::read_xml(xmlfile, tree);

    // Determine the type of material to be built

    try {
        tree.get_child("singlecrystal");
        materialType = materialTypes::singlecrystal;
    }
    catch (const boost::property_tree::ptree_bad_path&) {
    }

    try {
        tree.get_child("alloy");
        materialType = materialTypes::alloy;
    }
    catch (const boost::property_tree::ptree_bad_path&) {
    }

    try {
        tree.get_child("parametricalloy");
        materialType = materialTypes::parametricalloy;
    }
    catch (const boost::property_tree::ptree_bad_path&) {
    }

    // Define some error-catching variables
    int AUTOmix_counter = 0;

    // Traverse the tree and extract all information

    if (materialType == materialTypes::singlecrystal) {
        for (const auto& v : tree.get_child("singlecrystal")) {
            if (v.first == "materials_repository") {
                materials_repository =
                    alma::parseXMLfield<std::string>(v, "root_directory");
            }

            if (v.first == "gridDensity") {
                gridDensityA = alma::parseXMLfield<int>(v, "A");
                gridDensityB = alma::parseXMLfield<int>(v, "B");
                gridDensityC = alma::parseXMLfield<int>(v, "C");

                if (override_gridDensity) {
                    gridDensityA = external_gridDensity;
                    gridDensityB = external_gridDensity;
                    gridDensityC = external_gridDensity;
                }
            }

            if (v.first == "compound") {
                std::string base = alma::parseXMLfield<std::string>(v, "name");
                materialBase.emplace_back(base);
            }

            if (v.first == "target") {
                target_directory =
                    alma::parseXMLfield<std::string>(v, "directory");
            }

            if (v.first == "overwrite") {
                overwrite = true;
            }

            if (v.first == "broadening") {
                scalebroad = alma::parseXMLfield<double>(v, "scale_factor");
            }
        }
    } // end singlecrystal

    if (materialType == materialTypes::alloy) {
        for (const auto& v : tree.get_child("alloy")) {
            if (v.first == "materials_repository") {
                materials_repository =
                    alma::parseXMLfield<std::string>(v, "root_directory");
            }

            if (v.first == "gridDensity") {
                gridDensityA = alma::parseXMLfield<int>(v, "A");
                gridDensityB = alma::parseXMLfield<int>(v, "B");
                gridDensityC = alma::parseXMLfield<int>(v, "C");
            }

            if (v.first == "compound") {
                std::string base = alma::parseXMLfield<std::string>(v, "name");
                double mixfraction =
                    alma::parseXMLfield<double>(v, "mixfraction");

                materialBase.emplace_back(base);
                mixFraction.emplace_back(mixfraction);
            }

            if (v.first == "target") {
                target_directory =
                    alma::parseXMLfield<std::string>(v, "directory");
            }

            if (v.first == "overwrite") {
                overwrite = true;
            }

            if (v.first == "broadening") {
                scalebroad = alma::parseXMLfield<double>(v, "scale_factor");
            }
        }
    } // end alloy

    if (materialType == materialTypes::parametricalloy) {
        for (const auto& v : tree.get_child("parametricalloy")) {
            if (v.first == "mixparameter") {
                std::string param_name =
                    alma::parseXMLfield<std::string>(v, "name");
                double param_start = alma::parseXMLfield<double>(v, "start");
                double param_stop = alma::parseXMLfield<double>(v, "stop");
                double param_step = alma::parseXMLfield<double>(v, "step");

                std::vector<double> mixvalues;

                for (double mix = param_start; mix <= (1 + 1e-6) * param_stop;
                     mix += param_step) {
                    mixvalues.emplace_back(mix);
                }

                if (parametric_mixvalues.count(param_name) == 1) {
                    std::cout << "ERROR in parametric alloy:" << std::endl;
                    std::cout << "Duplicate definition of mixparameter named "
                              << param_name << std::endl;
                    world.abort(1);
                }

                parametric_mixvalues[param_name] = mixvalues;
            }

            if (v.first == "materials_repository") {
                materials_repository =
                    alma::parseXMLfield<std::string>(v, "root_directory");
            }

            if (v.first == "gridDensity") {
                gridDensityA = alma::parseXMLfield<int>(v, "A");
                gridDensityB = alma::parseXMLfield<int>(v, "B");
                gridDensityC = alma::parseXMLfield<int>(v, "C");
            }

            if (v.first == "compound") {
                std::string base = alma::parseXMLfield<std::string>(v, "name");
                std::string mix_name =
                    alma::parseXMLfield<std::string>(v, "mixfraction");

                if (mix_name.compare("AUTO") == 0) {
                    AUTOmix_counter++;
                }

                materialBase.emplace_back(base);
                mixparameterName.emplace_back(mix_name);
            }

            if (v.first == "target") {
                target_directory =
                    alma::parseXMLfield<std::string>(v, "directory");
            }

            if (v.first == "overwrite") {
                overwrite = true;
            }

            if (v.first == "broadening") {
                scalebroad = alma::parseXMLfield<double>(v, "scale_factor");
            }
        }
    } // end parametricalloy

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

    if (materialType == materialTypes::alloy) {
        double totalFraction = 0.0;

        for (std::size_t n = 0; n < materialBase.size(); n++) {
            totalFraction += mixFraction.at(n);

            if ((mixFraction.at(n) <= 0.0) || (mixFraction.at(n) >= 1.0)) {
                std::cout << "ERROR in compound " << materialBase.at(n) << " :"
                          << std::endl;
                std::cout << "Provided mixfraction is " << mixFraction.at(n)
                          << std::endl;
                std::cout << "Value must be between 0 and 1." << std::endl;
                badinput = true;
            }
        }

        if (std::abs(totalFraction - 1.0) > 1e-12) {
            std::cout << "ERROR: the total mixfraction is " << totalFraction
                      << std::endl;
            std::cout << "Mixfractions should add up to exactly 1."
                      << std::endl;
            badinput = true;
        }
    }

    if (materialType == materialTypes::parametricalloy) {
        if (AUTOmix_counter > 1) {
            std::cout << "ERROR in parametric alloy:" << std::endl;
            std::cout
                << "Alloy composition is underdetermined because two or more"
                << std::endl;
            std::cout << "compounds were declared to have AUTO mixfraction."
                      << std::endl;
            badinput = true;
        }
    }

    if (badinput) {
        world.abort(1);
    }

    ///////// LAUNCH THE APPROPRIATE BUILDER ////////

    if (materialType == materialTypes::singlecrystal) {
        singleCrystalBuilder(world);
    }

    if (materialType == materialTypes::alloy) {
        alloyBuilder();
    }

    if (materialType == materialTypes::parametricalloy) {
        // create map with all potential parameter combinations
        parametric_allcombinations =
            parametric_combinations(parametric_mixvalues);
        int Ncombinations = parametric_allcombinations.begin()->second.size();

        // traverse all combinations and
        // build each compound that has physically valid mixfractions

        std::map<std::string, double> mymixvalues;

        for (int ncomb = 0; ncomb < Ncombinations; ncomb++) {
            mymixvalues.clear();

            std::map<std::string, std::vector<double>>::iterator iter =
                parametric_allcombinations.begin();
            for (; iter != parametric_allcombinations.end(); iter++) {
                mymixvalues[iter->first] = iter->second.at(ncomb);
            }

            // scan over all compounds that make up this alloy

            mixFraction.clear();
            double mixFraction_sum = 0.0;

            for (std::size_t nmat = 0; nmat < materialBase.size(); nmat++) {
                std::string mixparam_name = mixparameterName.at(nmat);

                // check if the "name" actually represents a numeric value
                double numeric_value = -1.0;
                std::stringstream extractor(mixparam_name);
                extractor >> numeric_value;

                if (numeric_value > 0.0) {
                    mixFraction.emplace_back(numeric_value);
                    mixFraction_sum += numeric_value;
                }
                else {
                    if (mixparam_name.compare("AUTO") == 0) {
                        // actual value will be determined below, push back
                        // zero to correctly reserve entry for this compound
                        mixFraction.emplace_back(0.0);
                    }

                    else {
                        mixFraction.emplace_back(mymixvalues[mixparam_name]);
                        mixFraction_sum += mymixvalues[mixparam_name];
                    }
                }
            }

            // resolve remaining "AUTO" mixfraction if needed
            // and check if this is a valid compound

            bool valid_combination = true;

            for (std::size_t nmat = 0; nmat < materialBase.size(); nmat++) {
                if (mixparameterName.at(nmat).compare("AUTO") == 0) {
                    mixFraction.at(nmat) = 1.0 - mixFraction_sum;
                }

                if ((mixFraction.at(nmat) <= 1e-6) ||
                    (mixFraction.at(nmat) >= 1.0 - 1e-6)) {
                    valid_combination = false;
                }
            }

            if (valid_combination) {
                alloyBuilder();
            }
        } // done traversing parametric combinations
    }     // end parametric alloy

    return 0;
}
