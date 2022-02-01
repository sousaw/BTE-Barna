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
/// Test force constant loading.
///
/// The code in this file loads a FORCE_CONSTANTS file for Si and
/// performs some checks on the resulting variables.

#include <boost/filesystem.hpp>
#include "gtest/gtest.h"
#include <cmakevars.hpp>
#include <vasp_io.hpp>

TEST(force_constants_test_case, force_constants_test) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto si_dir = basedir / boost::filesystem::path("Si");
    auto poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto ifc_path = si_dir / boost::filesystem::path("FORCE_CONSTANTS");

    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());

    auto force_constants =
        alma::load_FORCE_CONSTANTS(ifc_path.string().c_str(), *poscar, 5, 5, 5);

    ASSERT_EQ(125u, force_constants->get_ncells());

    auto counter = 0;

    for (auto i = 0; i < 5; ++i) {
        for (auto j = 0; j < 5; ++j) {
            for (auto k = 0; k < 5; ++k) {
                EXPECT_EQ(i, force_constants->pos[counter][0]);
                EXPECT_EQ(j, force_constants->pos[counter][1]);
                EXPECT_EQ(k, force_constants->pos[counter][2]);
                ++counter;
            }
        }
    }

    Eigen::Matrix3d block00;
    block00 << 12.980912139744303, 0.000000000000000, -0.000000000000000,
        0.000000000000000, 12.980912139744287, -0.000000000000004,
        -0.000000000000000, -0.000000000000004, 12.980912139744298;
    Eigen::Matrix3d block01;
    block01 << -3.193492567290560, -2.066877540825653, -2.066877540825653,
        -2.066877540825654, -3.193492567290558, -2.066877540825654,
        -2.066877540825654, -2.066877540825652, -3.193492567290560;
    Eigen::MatrixXd reference(6, 6);
    reference << block00, block01, block01.transpose(), block00;

    auto block = force_constants->ifcs[0];

    for (auto i = 0; i < 6; i++)
        for (auto j = 0; j < 6; ++j)
            EXPECT_NEAR(reference(i, j), block(i, j), 1e-12);
}
