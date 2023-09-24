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
/// Test the beyondRTA module by computing the thermal
/// conductivity tensor of wurtzite GaN under the full BTE.

#include <iostream>
#include <iomanip>
#include <functional>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <utilities.hpp>
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>
#include <beyondRTA.hpp>
#include <shengbte_iter.hpp>

TEST(beyondRTA_case, beyondRTA_test) {
    boost::mpi::environment env;
    boost::mpi::communicator world;
    auto myrank = world.rank();

    // load information
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto data_dir = basedir / boost::filesystem::path("GaN_wurtzite");
    auto poscar_path = data_dir / boost::filesystem::path("POSCAR");
    auto ifc_path = data_dir / boost::filesystem::path("FORCE_CONSTANTS");
    auto born_path = data_dir / boost::filesystem::path("BORN");
    auto thirdorder_path =
        data_dir / boost::filesystem::path("FORCE_CONSTANTS_3RD");

    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto syms = alma::Symmetry_operations(*poscar);
    auto force_constants =
        alma::load_FORCE_CONSTANTS(ifc_path.string().c_str(), *poscar, 5, 5, 5);
    auto born = alma::load_BORN(born_path.string().c_str());
    auto thirdorder = alma::load_FORCE_CONSTANTS_3RD(
        thirdorder_path.string().c_str(), *poscar);

    auto grid = std::make_unique<alma::Gamma_grid>(
        *poscar, syms, *force_constants, *born, 8, 8, 8);
    grid->enforce_asr();

    double Tref = 300.0;

    // determine phonon processes and precompute matrix elements
    auto three_ph_procs = alma::find_allowed_threeph(*grid, world, 0.1);
    auto two_ph_procs = alma::find_allowed_twoph(*grid, world);

    for (std::size_t i = 0; i < three_ph_procs.size(); ++i) {
        three_ph_procs[i].compute_vp2(*poscar, *grid, *thirdorder);
    }

    // compute scattering rates
    Eigen::ArrayXXd w3 =
        alma::calc_w0_threeph(*grid, three_ph_procs, Tref, world);
    Eigen::ArrayXXd w2 =
        alma::calc_w0_twoph(*poscar, *grid, two_ph_procs, world);
    Eigen::ArrayXXd w0(w3 + w2);

    // target results
    double kapparef_xx = 179.53063355; 
    double kapparef_yy = 179.53063355; 
    double kapparef_zz = 181.66460879;

    // test beyondRTA conductivity from iterative Eigen solver
    Eigen::Matrix3d kappa1 = alma::beyondRTA::calc_kappa(*poscar,
                                                         *grid,
                                                         syms,
                                                         three_ph_procs,
                                                         two_ph_procs,
                                                         w0,
                                                         Tref,
                                                         true,
                                                         world);
    
    if (myrank == 0) {
        EXPECT_TRUE(alma::almost_equal(kapparef_xx, kappa1(0, 0), 1e-8, 1e-3));
        EXPECT_TRUE(alma::almost_equal(kapparef_yy, kappa1(1, 1), 1e-8, 1e-3));
        EXPECT_TRUE(alma::almost_equal(kapparef_zz, kappa1(2, 2), 1e-8, 1e-3));
        kappa1(0, 0) = kappa1(1, 1) = kappa1(2, 2) = 0.0;
        EXPECT_TRUE(alma::almost_equal(kappa1.norm(), 0., 1e-13, 0.));
    }

    // test beyondRTA conductivity from LU Eigen solver
    Eigen::Matrix3d kappa2 = alma::beyondRTA::calc_kappa(*poscar,
                                                         *grid,
                                                         syms,
                                                         three_ph_procs,
                                                         two_ph_procs,
                                                         w0,
                                                         Tref,
                                                         false,
                                                         world);
    if (myrank == 0) {
        EXPECT_TRUE(alma::almost_equal(kapparef_xx, kappa2(0, 0), 1e-8, 1e-3));
        EXPECT_TRUE(alma::almost_equal(kapparef_yy, kappa2(1, 1), 1e-8, 1e-3));
        EXPECT_TRUE(alma::almost_equal(kapparef_zz, kappa2(2, 2), 1e-8, 1e-3));
        kappa2(0, 0) = kappa2(1, 1) = kappa2(2, 2) = 0.0;
        EXPECT_TRUE(alma::almost_equal(kappa2.norm(), 0., 1e-13, 0.));
    }

    // test the ShengBTE-like iterator
    double kapparef_xx_sheng = 179.54555271;
    double kapparef_yy_sheng = 179.54555271;
    double kapparef_zz_sheng = 201.73525076;
    Eigen::Matrix3d kappa3 = alma::calc_shengbte_kappa(
        *poscar, *grid, syms, three_ph_procs, two_ph_procs, Tref, world);
    if (myrank == 0) {
        EXPECT_TRUE(
            alma::almost_equal(kapparef_xx_sheng, kappa3(0, 0), 1e-8, 1e-3));
        EXPECT_TRUE(
            alma::almost_equal(kapparef_yy_sheng, kappa3(1, 1), 1e-8, 1e-3));
        EXPECT_TRUE(
            alma::almost_equal(kapparef_zz_sheng, kappa3(2, 2), 1e-8, 1e-3));
    }
    world.barrier();
}
