// Copyright 2015-2020 The ALMA Project Developers
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
/// Test whether we can correctly communicate descriptions
/// of allowed three-phonon processes among MPI processes.

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>

TEST(proc_serialization_case, proc_serialization_test) {
    constexpr int test_tolerance(15);

    boost::mpi::environment env;

    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto si_dir = basedir / boost::filesystem::path("Si");
    auto poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto ifc_path = si_dir / boost::filesystem::path("FORCE_CONSTANTS");

    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto syms = alma::Symmetry_operations(*poscar);
    auto force_constants =
        alma::load_FORCE_CONSTANTS(ifc_path.string().c_str(), *poscar, 5, 5, 5);

    auto grid = std::make_unique<alma::Gamma_grid>(
        *poscar, syms, *force_constants, 12, 12, 12);

    grid->enforce_asr();

    boost::mpi::communicator world;
    auto my_id = world.rank();
    auto processes = alma::find_allowed_threeph(*grid, world, 0.1);

    decltype(processes) other;

    if (my_id == 0) {
        auto ntotal = processes.size();

        for (auto i = 1; i < world.size(); ++i) {
            other.clear();
            world.recv(i, 0, other);
            ntotal += other.size();
        }
        // Check that the total number of processes that we got is the
        // same as in processes_test.
        EXPECT_NEAR(79818u + 95900u, ntotal, test_tolerance);
    }
    else {
        world.send(0, 0, processes);
    }
}
