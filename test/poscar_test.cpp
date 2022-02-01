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
/// Test POSCAR loading.
///
/// The code in this file loads a POSCAR for Si and checks
/// the values of all variables.

#include <boost/filesystem.hpp>
#include "gtest/gtest.h"
#include <cmakevars.hpp>
#include <vasp_io.hpp>

TEST(poscar_test_case, poscar_test) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto si_dir = basedir / boost::filesystem::path("Si");
    auto poscar_path = si_dir / boost::filesystem::path("POSCAR");

    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());

    ASSERT_EQ(1u, poscar->get_nelements());
    EXPECT_EQ("Si", poscar->elements[0]);
    EXPECT_EQ("Si", poscar->get_element(0));
    EXPECT_EQ("Si", poscar->get_element(1));
    EXPECT_THROW(poscar->get_element(2), alma::value_error);
    ASSERT_EQ(2, poscar->numbers[0]);

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            if (i == j)
                EXPECT_DOUBLE_EQ(0., poscar->lattvec(i, j));
            else
                EXPECT_DOUBLE_EQ(0.2733556057883652, poscar->lattvec(i, j));
        }
    }

    for (auto i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(0., poscar->positions(i, 0));
        EXPECT_DOUBLE_EQ(0.25, poscar->positions(i, 1));
    }
}
