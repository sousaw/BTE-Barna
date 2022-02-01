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
/// Test if the specific heat is computed correctly.

#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <bulk_properties.hpp>

TEST(cv_case, cv_test) {
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

    auto cv = alma::calc_cv(*poscar, *grid, 300.);
    EXPECT_TRUE(alma::almost_equal(1e27 * cv, 1.625834959e6));
}
