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
/// Test whether we can correctly detect allowed three-phonon
/// processes.

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>

TEST(processes_case, processes_test) {
    boost::mpi::environment env;

    constexpr auto test_tolerance(10);

    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto si_dir = basedir / boost::filesystem::path("Si");
    auto poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto ifc_path = si_dir / boost::filesystem::path("FORCE_CONSTANTS");

    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto syms = alma::Symmetry_operations(*poscar);
    auto force_constants =
        alma::load_FORCE_CONSTANTS(ifc_path.string().c_str(), *poscar, 5, 5, 5);

    // We first try a homogeneous grid, the simplest case.
    auto grid = std::make_unique<alma::Gamma_grid>(
        *poscar, syms, *force_constants, 12, 12, 12);
    grid->enforce_asr();

    boost::mpi::communicator world;
    auto my_id = world.rank();
    auto processes = alma::find_allowed_threeph(*grid, world, 0.1);
    auto my_nprocs = processes.size();

    decltype(my_nprocs) my_nemission = 0;

    for (decltype(my_nprocs) i = 0; i < my_nprocs; ++i)
        if (processes[i].type == alma::threeph_type::emission)
            ++my_nemission;

    std::vector<decltype(my_nprocs)> all_nprocs;
    boost::mpi::gather(world, my_nprocs, all_nprocs, 0);
    std::vector<decltype(my_nemission)> all_nemission;
    boost::mpi::gather(world, my_nemission, all_nemission, 0);

    if (my_id == 0) {
        auto nprocs = std::accumulate(all_nprocs.begin(), all_nprocs.end(), 0);
        auto nemission =
            std::accumulate(all_nemission.begin(), all_nemission.end(), 0);
        EXPECT_NEAR(79818u, nprocs - nemission, test_tolerance);
        EXPECT_NEAR(95900u, nemission, test_tolerance);
    }

    // And then do the same test for a inhomogeneous grid.
    grid = std::make_unique<alma::Gamma_grid>(
        *poscar, syms, *force_constants, 12, 12, 11);
    grid->enforce_asr();
    processes = alma::find_allowed_threeph(*grid, world, 0.1);
    my_nprocs = processes.size();
    my_nemission = 0;

    for (decltype(my_nprocs) i = 0; i < my_nprocs; ++i)
        if (processes[i].type == alma::threeph_type::emission)
            ++my_nemission;
    all_nprocs.clear();
    boost::mpi::gather(world, my_nprocs, all_nprocs, 0);
    all_nemission.clear();
    boost::mpi::gather(world, my_nemission, all_nemission, 0);

    if (my_id == 0) {
        auto nprocs = std::accumulate(all_nprocs.begin(), all_nprocs.end(), 0);
        auto nemission =
            std::accumulate(all_nemission.begin(), all_nemission.end(), 0);
        EXPECT_NEAR(475031u, nprocs - nemission, test_tolerance);
        EXPECT_NEAR(563722u, nemission, test_tolerance);
    }
}
