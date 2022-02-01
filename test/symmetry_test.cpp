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
/// Test the wrapper around spglib.

#include <iostream>
#include <boost/filesystem.hpp>
#include "gtest/gtest.h"
#include <cmakevars.hpp>
#include <constants.hpp>
#include <vasp_io.hpp>
#include <symmetry.hpp>

TEST(symmetry_test, Si_test) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto si_dir = basedir / boost::filesystem::path("Si");
    auto poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto symops = alma::Symmetry_operations(*poscar);

    EXPECT_EQ(227, symops.get_spacegroup_number());
    EXPECT_EQ("Fd-3m", symops.get_spacegroup_symbol());
    EXPECT_EQ(48u, symops.get_nsym());
    EXPECT_EQ("bb", symops.get_wyckoff());
    auto equivalences = symops.get_equivalences();
    EXPECT_EQ(0, equivalences[0]);
    EXPECT_EQ(0, equivalences[1]);

    // Since operation 0 is always the identity,
    // it must leave any vector untouched.
    Eigen::Vector3d vector;
    vector.setRandom();
    auto transformed = symops.transform_v(vector, 0);
    auto ctransformed = symops.transform_v(vector, 0, true);
    auto rotated = symops.rotate_v(vector, 0);
    auto crotated = symops.rotate_v(vector, 0, true);

    for (auto i = 0; i < 3; ++i) {
        EXPECT_NEAR(vector(i), transformed(i), 1e-8);
        EXPECT_NEAR(vector(i), ctransformed(i), 1e-8);
        EXPECT_NEAR(vector(i), rotated(i), 1e-8);
        EXPECT_NEAR(vector(i), crotated(i), 1e-8);
    }
}

TEST(symmetry_test, InAs_test) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto inas_dir = basedir / boost::filesystem::path("InAs");
    auto poscar_path = inas_dir / boost::filesystem::path("POSCAR");
    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto symops = alma::Symmetry_operations(*poscar);

    EXPECT_EQ(216, symops.get_spacegroup_number());
    EXPECT_EQ("F-43m", symops.get_spacegroup_symbol());
    EXPECT_EQ(24u, symops.get_nsym());
    EXPECT_EQ("ad", symops.get_wyckoff());
    auto equivalences = symops.get_equivalences();
    EXPECT_EQ(0, equivalences[0]);
    EXPECT_EQ(1, equivalences[1]);

    Eigen::Vector3d vector;
    vector.setRandom();
    auto transformed = symops.transform_v(vector, 0);
    auto ctransformed = symops.transform_v(vector, 0, true);
    auto rotated = symops.rotate_v(vector, 0);
    auto crotated = symops.rotate_v(vector, 0, true);

    for (auto i = 0; i < 3; ++i) {
        EXPECT_NEAR(vector(i), transformed(i), 1e-8);
        EXPECT_NEAR(vector(i), ctransformed(i), 1e-8);
        EXPECT_NEAR(vector(i), rotated(i), 1e-8);
        EXPECT_NEAR(vector(i), crotated(i), 1e-8);
    }
}

TEST(symmetry_test, Bi2O3_test) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto bi2o3_dir = basedir / boost::filesystem::path("Bi2O3");
    auto poscar_path = bi2o3_dir / boost::filesystem::path("POSCAR");
    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto symops = alma::Symmetry_operations(*poscar);

    EXPECT_EQ(14, symops.get_spacegroup_number());
    EXPECT_EQ("P2_1/c", symops.get_spacegroup_symbol());
    EXPECT_EQ(4u, symops.get_nsym());

    for (auto& i : symops.get_wyckoff())
        EXPECT_EQ('e', i);
    constexpr int ref_equivalences[] = {1, 1, 1,  1,  5,  5,  5,  5,  9,  9,
                                        9, 9, 13, 13, 13, 13, 17, 17, 17, 17};
    auto equivalences = symops.get_equivalences();

    for (decltype(equivalences.size()) i = 0; i < equivalences.size(); ++i)
        EXPECT_EQ(ref_equivalences[i] - 1, equivalences[i]);

    Eigen::Vector3d vector;
    vector.setRandom();
    auto transformed = symops.transform_v(vector, 0);
    auto ctransformed = symops.transform_v(vector, 0, true);
    auto rotated = symops.rotate_v(vector, 0);
    auto crotated = symops.rotate_v(vector, 0, true);

    for (auto i = 0; i < 3; ++i) {
        EXPECT_NEAR(vector(i), transformed(i), 1e-8);
        EXPECT_NEAR(vector(i), ctransformed(i), 1e-8);
        EXPECT_NEAR(vector(i), rotated(i), 1e-8);
        EXPECT_NEAR(vector(i), crotated(i), 1e-8);
    }
}

TEST(symmetry_test, borophene_test) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto b_dir = basedir / boost::filesystem::path("B-Pmmn");
    auto poscar_path = b_dir / boost::filesystem::path("POSCAR");
    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto symops = alma::Symmetry_operations(*poscar);

    EXPECT_EQ(59, symops.get_spacegroup_number());
    EXPECT_EQ("Pmmn", symops.get_spacegroup_symbol());
    EXPECT_EQ(8u, symops.get_nsym());
    EXPECT_EQ("ffeeefef", symops.get_wyckoff());
    constexpr int ref_equivalences[] = {1, 1, 3, 3, 3, 1, 3, 1};
    auto equivalences = symops.get_equivalences();

    for (decltype(equivalences.size()) i = 0; i < equivalences.size(); ++i)
        EXPECT_EQ(ref_equivalences[i] - 1, equivalences[i]);

    Eigen::Vector3d vector;
    vector.setRandom();
    auto transformed = symops.transform_v(vector, 0);
    auto ctransformed = symops.transform_v(vector, 0, true);
    auto rotated = symops.rotate_v(vector, 0);
    auto crotated = symops.rotate_v(vector, 0, true);

    for (auto i = 0; i < 3; ++i) {
        EXPECT_NEAR(vector(i), transformed(i), 1e-8);
        EXPECT_NEAR(vector(i), ctransformed(i), 1e-8);
        EXPECT_NEAR(vector(i), rotated(i), 1e-8);
        EXPECT_NEAR(vector(i), crotated(i), 1e-8);
    }
}
