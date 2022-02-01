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
/// The code in this file loads a BORN file for InAs and
/// performs some checks on the resulting variables.

#include <boost/filesystem.hpp>
#include "gtest/gtest.h"
#include <cmakevars.hpp>
#include <vasp_io.hpp>

TEST(born_test_case, born_test) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto inas_dir = basedir / boost::filesystem::path("InAs");
    auto born_path = inas_dir / boost::filesystem::path("BORN");

    auto born = alma::load_BORN(born_path.string().c_str());

    ASSERT_EQ(2u, born->born.size());

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            if (i == j) {
                EXPECT_DOUBLE_EQ(19.643, born->epsilon(i, j));
                EXPECT_DOUBLE_EQ(2.6781, born->born[0](i, j));
                EXPECT_DOUBLE_EQ(-2.6781, born->born[1](i, j));
            }
            else {
                EXPECT_DOUBLE_EQ(0., born->epsilon(i, j));
                EXPECT_DOUBLE_EQ(0., born->born[0](i, j));
                EXPECT_DOUBLE_EQ(0., born->born[1](i, j));
            }
        }
    }
}
