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
/// Writes the propagator for a given material at given temperature and timestep
#define TBB_PREVIEW_GLOBAL_CONTROL true
#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include <cmakevars.hpp>
#include <constants.hpp>
#include <utilities.hpp>
#include <structures.hpp>
#include <bulk_hdf5.hpp>
#include <io_utils.hpp>
#include <collision_operator.hpp>
#include <vasp_io.hpp>
#include <tbb/global_control.h>
#include <chrono>

int main(int argc, char** argv) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    /// Record start time
    auto begin = std::chrono::high_resolution_clock::now();

    /// MPI info
    std::size_t master = 0;
    std::size_t my_id = world.rank();
    std::size_t nprocs = world.size();

    // Temperature at which calculate the propagator
    double Tambient = -42.0;
    // Time step for the propagator P = exp(dt*B) where B is the collision
    // operator in energy
    double dt = -1.0;

    std::size_t nthreadsTBB = 1;

    // Path to the HDF5 file.
    std::string h5filename, anhIFCfile;

    if (!(argc == 6 or argc == 5)) {
        std::cout << "USAGE 0: PropagatorBuilder <inputfile.hdf5>  "
                     "FORCE_CONSTANTS_3RD Temperature[Kelvin] time_step[ps] "
                     "nthreads[optional]"
                  << std::endl;
        world.abort(1);
    }

    else {
        h5filename = argv[1];
        anhIFCfile = argv[2];
        Tambient = atof(argv[3]);
        dt = atof(argv[4]);
        if (Tambient <= 0.0) {
            std::cerr << "ERROR: Tambient must be positive (" << Tambient
                      << " provided)." << std::endl;
            world.abort(1);
        }
        if (dt <= 0.0) {
            std::cerr << "ERROR: dt must be positive (" << dt << " provided)."
                      << std::endl;
            world.abort(1);
        }
    }

    if (argc == 6) {
        nthreadsTBB = boost::lexical_cast<std::size_t>(argv[5]);
        if (nthreadsTBB < 1) {
            std::cerr << "ERROR: nthreads must be positive (" << nthreadsTBB
                      << " provided)." << std::endl;
            world.abort(1);
        }
    }

    /// Init TBB threads with user defined number of threads or single one as
    /// default.
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                nthreadsTBB);

    if (my_id == master) {
        std::cout << "***********************************" << std::endl;
        std::cout << "This is ALMA/PropagatorBuilder version "
                  << ALMA_VERSION_MAJOR << "." << ALMA_VERSION_MINOR
                  << std::endl;
        std::cout << "***********************************" << std::endl;
        std::cout << "#input variables:\n"
                  << "-h5filename: " << h5filename << "\n-3rdIFC: " << anhIFCfile
                  << "\n-Tambient: " << Tambient << " K\n-dt: " << dt << " ps"
                  << std::endl;
        std::cout << "#MPI procs:\n" << nprocs << std::endl;
        std::cout << "#TBB processes\n" << nthreadsTBB << std::endl;
    }
    // Create name of outputfile
    boost::filesystem::path hdf5_path{h5filename};
    std::stringstream output_file_builder;
    output_file_builder << hdf5_path.stem().string() << "_" << Tambient << "K_"
                        << dt << "ps";
    std::string output_file = output_file_builder.str();

    // Obtain phonon data and anharmonic force constants
    if (my_id == master)
        std::cout << "Reading " << h5filename << std::endl;

    if (!(boost::filesystem::exists(hdf5_path))) {
        std::cout << "ERROR:" << std::endl;
        std::cout << "HDF5 file " << hdf5_path << " does not exist."
                  << std::endl;
        world.abort(1);
    }


    auto t1 = std::chrono::high_resolution_clock::now();

    auto hdf5_data = alma::load_bulk_hdf5(hdf5_path.string().c_str(), world);
    auto poscar = std::move(std::get<1>(hdf5_data));
    auto syms = std::move(std::get<2>(hdf5_data));
    auto grid = std::move(std::get<3>(hdf5_data));
    auto processes = std::move(std::get<4>(hdf5_data));
    
    processes.release(); 
   
    if (my_id == master)
        std::cout << "Reading " << anhIFCfile << std::endl;
    auto anhIFC = alma::load_FORCE_CONSTANTS_3RD(anhIFCfile.c_str(), *poscar);

    auto t2 = std::chrono::high_resolution_clock::now();

    if (my_id == master)
        std::cout << std::endl << "Generating scattering operator" << std::endl;

    /// We are building B matrix (liniarized collision operator in energy
    /// formulation) from almaBTE database and extending it to fullBZ using
    /// symmetry and permutation. B is a huge matrix (it can be dozens of GBs)
    auto B =
        get_collision_operator_dense(*grid, *poscar, *anhIFC, Tambient, world);
    world.barrier();
    alma::save_P(output_file + ".B.eigen.bin",B,world);    
    auto t3 = std::chrono::high_resolution_clock::now();

    if (my_id == master)
        std::cout << std::endl << "Getting propagator" << std::endl;
    /// Propagator calculation; this is done through krylov subspace
    Eigen::MatrixXd P;
    alma::build_P(B, P, dt, world);
    world.barrier();
    auto t4 = std::chrono::high_resolution_clock::now();

    if (my_id == master)
        std::cout << std::endl << "Storing propagator" << std::endl;
    alma::save_P(output_file + ".P.eigen.bin", P, world);

    world.barrier();

    auto end = std::chrono::high_resolution_clock::now();

    if (my_id == master) {
        std::cout << std::endl << "[DONE.]" << std::endl;
        std::cout << "###############\nGLOBAL TIMING\n###############\n";
        auto begin_c = std::chrono::system_clock::to_time_t(begin);
        std::cout << "#" << std::endl
                  << "Started at "
                  << std::put_time(std::localtime(&begin_c), "%c") << std::endl;
        auto end_c = std::chrono::system_clock::to_time_t(end);
        std::cout << "#" << std::endl
                  << "Ended at " << std::put_time(std::localtime(&end_c), "%c")
                  << std::endl;
        std::cout
            << "#" << std::endl
            << "Total elapsed time: "
            << (static_cast<std::chrono::duration<double>>(end - begin)).count()
            << " s\n";
        std::cout
            << "###############\nTIME CONSUMED BY EACH TASK\n###############\n";
        std::cout
            << "#HDF5 and 3rdIFC processing:         : "
            << (static_cast<std::chrono::duration<double>>(t2 - t1)).count()
            << " s\n";
        std::cout
            << "#Scattering operator generation      : "
            << (static_cast<std::chrono::duration<double>>(t3 - t2)).count()
            << " s\n";
        std::cout
            << "#Calculating propagator (matrix exp) : "
            << (static_cast<std::chrono::duration<double>>(t4 - t3)).count()
            << " s\n";
        std::cout
            << "#Storing propagator                  : "
            << (static_cast<std::chrono::duration<double>>(end - t4)).count()
            << " s\n";
    }
    world.barrier();
    return 0;
}
