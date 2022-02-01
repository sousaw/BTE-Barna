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
/// Test third-order force constant loading.
///
/// The code in this file loads a FORCE_CONSTANTS_3RD file for InAs
/// performs some checks on the resulting variables.

#include <boost/filesystem.hpp>
#include "gtest/gtest.h"
#include <cmakevars.hpp>
#include <vasp_io.hpp>

TEST(force_constants_3rd_test_case, force_constants_3rd_test) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto inas_dir = basedir / boost::filesystem::path("InAs");
    auto poscar_path = inas_dir / boost::filesystem::path("POSCAR");
    auto thirdorder_path =
        inas_dir / boost::filesystem::path("FORCE_CONSTANTS_3RD");

    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());

    auto thirdorder = alma::load_FORCE_CONSTANTS_3RD(
        thirdorder_path.string().c_str(), *poscar);

    ASSERT_EQ(1166u, thirdorder->size());

    for (auto ic = 0; ic < 3; ++ic) {
        EXPECT_DOUBLE_EQ(0., (*thirdorder)[0].rj(ic));
        EXPECT_DOUBLE_EQ(0., (*thirdorder)[0].rk(ic));
    }

    Eigen::Vector3d refj;
    refj << -0.30295705, 0.00000000, -0.30295705;
    Eigen::Vector3d refk;
    refk << -0.60591410, -0.30295705, -0.30295705;
    EXPECT_TRUE(
        alma::almost_equal(0., ((*thirdorder)[214].rj - refj).squaredNorm()));
    EXPECT_TRUE(
        alma::almost_equal(0., ((*thirdorder)[214].rk - refk).squaredNorm()));
    Eigen::VectorXd refs(27);
    refs << 6.0718519833e-03, 5.1088225752e-03, 1.0477414223e-02,
        3.8876531988e-03, 7.4020846273e-03, 2.8672253747e-02, 4.1666128361e-03,
        7.2535400885e-03, 1.5462338623e-03, -1.2364367404e-02,
        -1.3537520400e-02, -1.9748797440e-02, -2.2194154852e-03,
        1.4308973003e-02, 9.7455787988e-03, -1.4755870697e-02,
        -2.1127704398e-02, -2.3147063565e-02, 1.3426817411e-02,
        2.2670728496e-02, 2.3696599694e-02, 5.7203750827e-03, -6.9215003094e-03,
        -4.4161463088e-04, 1.1937167984e-02, 1.3496922666e-02, 5.7718963875e-03;
    auto i = 0u;

    for (auto a1 = 0u; a1 < 3u; ++a1)
        for (auto a2 = 0u; a2 < 3u; ++a2)
            for (auto a3 = 0u; a3 < 3u; ++a3) {
                auto ifc1 = (*thirdorder)[214].ifc(a1, a2, a3);
                EXPECT_DOUBLE_EQ(refs(i), ifc1);
                ++i;
            }
}
