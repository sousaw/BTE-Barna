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
/// Test the dynamical matrix routines.
///
/// The code in this file computes the phonon spectra for two
/// example systems (one nonpolar and one polar) and performs a series
/// of checks on the results.

#include <iomanip>
#include <boost/filesystem.hpp>
#include "gtest/gtest.h"
#include <cmakevars.hpp>
#include <constants.hpp>
#include <vasp_io.hpp>
#include <dynamical_matrix.hpp>

class Dynamical_matrix_nonpolar_test : public ::testing::Test {
public:
    int ndof;
    std::unique_ptr<alma::Dynamical_matrix_builder> factory;
    Eigen::Vector3d qpoint;

    virtual void SetUp() {
        auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
        auto si_dir = basedir / boost::filesystem::path("Si");
        auto poscar_path = si_dir / boost::filesystem::path("POSCAR");
        auto ifc_path = si_dir / boost::filesystem::path("FORCE_CONSTANTS");

        auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
        auto syms = alma::Symmetry_operations(*poscar);
        auto force_constants = alma::load_FORCE_CONSTANTS(
            ifc_path.string().c_str(), *poscar, 5, 5, 5);

        ndof = 3 * poscar->get_natoms();
        factory = std::make_unique<alma::Dynamical_matrix_builder>(
            *poscar, syms, *force_constants);
    }
};

TEST_F(Dynamical_matrix_nonpolar_test, works_at_gamma) {
    qpoint << 0.0, 0.0, 0.0;
    auto results = factory->get_spectrum(qpoint);

    for (auto i = 0; i < 3; ++i) {
        EXPECT_NEAR(0., results->omega(i), 1e-5);
        EXPECT_NEAR(95.15970442, results->omega(i + 3), 1e-5);
    }

    for (auto i = 0; i < ndof; ++i)
        for (auto j = 0; j < 3; ++j)
            EXPECT_NEAR(0., results->vg(j, i), 1e-5);
}

TEST_F(Dynamical_matrix_nonpolar_test, works_at_q) {
    qpoint << -0.2298539036E+01, 0.2298539036E+01, 0.2298539036E+01;
    auto results = factory->get_spectrum(qpoint);

    Eigen::ArrayXd ref_omega(6);
    ref_omega << 16.03880578, 16.03880578, 35.06088004, 92.30529849,
        92.52999096, 92.52999096;

    for (auto i = 0; i < ndof; ++i)
        EXPECT_NEAR(ref_omega(i), results->omega(i), 1e-5);

    Eigen::MatrixXd ref_vg(3, 6);
    ref_vg <<  -1.7026783888,  -1.0799586891,  -4.5780886026,   1.0085585989,  0.34670088629,  0.77070183705,
                1.7026783888,   1.0799586891,   4.5780886026,  -1.0085585989, -0.34670088629, -0.77070183705,
                1.7026783888,   1.0799586891,   4.5780886026,  -1.0085585989, -0.34670088629, -0.77070183705;

    for (auto i = 0; i < ndof; ++i)
        for (auto j = 0; j < 3; ++j)
            EXPECT_NEAR(ref_vg(j, i), results->vg(j, i), 1e-5);
}

class Dynamical_matrix_polar_test : public ::testing::Test {
public:
    int ndof;
    std::unique_ptr<alma::Dynamical_matrix_builder> factory;
    Eigen::Vector3d qpoint;

    virtual void SetUp() {
        auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
        auto inas_dir = basedir / boost::filesystem::path("InAs");
        auto poscar_path = inas_dir / boost::filesystem::path("POSCAR");
        auto ifc_path = inas_dir / boost::filesystem::path("FORCE_CONSTANTS");
        auto born_path = inas_dir / boost::filesystem::path("BORN");

        auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
        auto syms = alma::Symmetry_operations(*poscar);
        auto force_constants = alma::load_FORCE_CONSTANTS(
            ifc_path.string().c_str(), *poscar, 5, 5, 5);
        auto born = alma::load_BORN(born_path.string().c_str());

        ndof = 3 * poscar->get_natoms();
        factory = std::make_unique<alma::Dynamical_matrix_builder>(
            *poscar, syms, *force_constants, *born);
    }
};

TEST_F(Dynamical_matrix_polar_test, works_at_gamma) {
    qpoint << 0.0, 0.0, 0.0;
    auto results = factory->get_spectrum(qpoint);

    for (auto i = 0; i < 3; ++i) {
        EXPECT_NEAR(0., results->omega(i), 1e-5);
        EXPECT_NEAR(36.26531356, results->omega(i + 3), 1e-5);
    }

    for (auto i = 0; i < ndof; ++i)
        for (auto j = 0; j < 3; ++j)
            EXPECT_NEAR(0., results->vg(j, i), 1e-5);
}

TEST_F(Dynamical_matrix_polar_test, works_at_q) {
    qpoint << -0.8641468743E+00, 0.8641468743E+00, 0.8641468743E+00;
    auto results = factory->get_spectrum(qpoint);

    Eigen::ArrayXd ref_omega(6);
    ref_omega << 2.633265014, 2.633265014, 6.11513921, 36.9738694, 36.9738694,
        39.751155;

    for (auto i = 0; i < ndof; ++i)
        EXPECT_NEAR(ref_omega(i), results->omega(i), 1e-5);
    

    Eigen::MatrixXd ref_vg(3, 6);
    ref_vg << -1.3737072589,  -0.52558711805,   -2.3133538587,  -0.48125054448,  -0.46888696123, -0.089719709388,
               1.3737072589,   0.52558711805,    2.3133538587,   0.48125054448,   0.46888696123,  0.089719709388,
               1.3737072589,   0.52558711805,    2.3133538587,   0.48125054448,   0.46888696123,  0.089719709388;

    for (auto i = 0; i < ndof; ++i)
        for (auto j = 0; j < 3; ++j)
            EXPECT_NEAR(ref_vg(j, i), results->vg(j, i), 1e-5);
}
