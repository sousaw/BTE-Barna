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
/// Test if the small-grain thermal conductivity is computed correctly.

#include <iostream>
#include <boost/filesystem.hpp>
#include "gtest/gtest.h"
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <bulk_properties.hpp>

TEST(kappa_sg_case, kappa_sg_test) {
    boost::mpi::environment test;

    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto data_dir = basedir / boost::filesystem::path("GaN_wurtzite");
    auto poscar_path = data_dir / boost::filesystem::path("POSCAR");
    auto ifc_path = data_dir / boost::filesystem::path("FORCE_CONSTANTS");
    auto born_path = data_dir / boost::filesystem::path("BORN");

    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto syms = alma::Symmetry_operations(*poscar);
    auto force_constants =
        alma::load_FORCE_CONSTANTS(ifc_path.string().c_str(), *poscar, 5, 5, 5);
    auto born = alma::load_BORN(born_path.string().c_str());

    auto grid = std::make_unique<alma::Gamma_grid>(
        *poscar, syms, *force_constants, *born, 12, 12, 12);

    grid->enforce_asr();

    Eigen::Matrix3d kappa_sg = alma::calc_kappa_sg(*poscar, *grid, 300.);

    EXPECT_TRUE(alma::almost_equal(1.75453, kappa_sg(0, 0), 1e-2));
    EXPECT_TRUE(alma::almost_equal(2.12192, kappa_sg(2, 2), 1e-2));
}
