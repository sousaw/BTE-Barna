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
/// Test if the implementation of regular q-point grids works.

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>

TEST(qpoint_grid_case, qpoint_grid_test) {
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
        *poscar, syms, *force_constants, 5, 5, 5);

    grid->enforce_asr();
    EXPECT_EQ(125u, grid->nqpoints);
    EXPECT_EQ(10u, grid->get_nequivalences());
    auto atgamma = grid->get_spectrum_at_q(0);

    for (auto i = 0; i < 3; ++i)
        EXPECT_EQ(0., atgamma.omega(i));

    unsigned int reference[] = {1, 8, 8, 6, 24, 24, 12, 6, 12, 24};

    for (std::size_t i = 0; i < grid->get_nequivalences(); ++i)
        EXPECT_EQ(reference[i], grid->get_cardinal(i));

    grid = std::make_unique<alma::Gamma_grid>(
        *poscar, syms, *force_constants, 5, 5, 3);
    grid->enforce_asr();
    EXPECT_EQ(75u, grid->nqpoints);
    EXPECT_EQ(24u, grid->get_nequivalences());

    grid = std::make_unique<alma::Gamma_grid>(
        *poscar, syms, *force_constants, 8, 5, 3);
    grid->enforce_asr();
    EXPECT_EQ(120u, grid->nqpoints);
    EXPECT_EQ(61u, grid->get_nequivalences());
}
