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
/// Test the functionality of analytic1D::SPRcalculator_FourierLaplace.

#include <iostream>
#include <functional>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <utilities.hpp>
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>
#include <bulk_hdf5.hpp>
#include <analytic1d.hpp>
#include <Eigen/Dense>

TEST(analytic1D_SPRFL_case, bulk_test) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto si_dir = basedir / boost::filesystem::path("Si");
    auto hdf5_path = si_dir / boost::filesystem::path("Si_threeph.h5");
    boost::mpi::communicator world;
    auto my_id = world.rank();

    // obtain phonon data from HDF5 file
    auto hdf5_data = alma::load_bulk_hdf5(hdf5_path.string().c_str(), world);
    auto description = std::get<0>(hdf5_data);
    auto poscar = std::move(std::get<1>(hdf5_data));
    auto syms = std::move(std::get<2>(hdf5_data));
    auto grid = std::move(std::get<3>(hdf5_data));
    auto processes = std::move(std::get<4>(hdf5_data));

    double T = 300.0;

    Eigen::ArrayXXd w(alma::calc_w0_threeph(*grid, *processes, T, world));

    // create transport direction
    Eigen::Vector3d u100(1.0, 0.0, 0.0);

    // general variables
    int Nxi;
    int Nf;

    Eigen::MatrixXd Pref_RE;
    Eigen::MatrixXd Pref_IM;
    Eigen::MatrixXd REratio;
    Eigen::MatrixXd IMratio;

    if (my_id == 0) { // analytic1D code is not parallellised; avoid
                      // running duplicates.
        Nxi = 4;
        Nf = 3;
        alma::analytic1D::SPR_calculator_FourierLaplace SPRcalc(
            poscar.get(), grid.get(), &w, T);
        SPRcalc.setDirection(u100);
        SPRcalc.setLogSpatialGrid(1e3, 1e9, Nxi);
        SPRcalc.setLogTemporalGrid(1e5, 1e9, Nf);

        // Calculate P(xi,s) for Si infinite bulk
        Eigen::MatrixXcd P_substrate = SPRcalc.getSPR();

        Pref_RE.resize(Nxi, Nf);
        Pref_IM.resize(Nxi, Nf);

        Pref_RE << 2.302051716e-10, 2.299552693e-14, 1.276018754e-18,
            7.483290804e-07, 2.268148871e-10, 1.276301494e-14, 2.718484701e-10,
            2.717794584e-10, 7.677579078e-11, 2.082367203e-12, 2.082355442e-12,
            2.048562079e-12;

        Pref_IM << -1.591549449e-06, -1.591549482e-08, -1.591549436e-10,
            -5.246497215e-07, -1.591697413e-08, -1.591601609e-10,
            -4.331275346e-14, -4.330167249e-12, -1.231921228e-10,
            -1.477398876e-17, -1.47715184e-15, -1.078409621e-13;

        REratio = P_substrate.array().real() / Pref_RE.array();
        IMratio = P_substrate.array().imag() / Pref_IM.array();

        for (int nxi = 0; nxi < Nxi; nxi++) {
            for (int nf = 0; nf < Nf; nf++) {
                EXPECT_NEAR(REratio(nxi, nf), 1.0, 5e-3);
                EXPECT_NEAR(IMratio(nxi, nf), 1.0, 5e-3);
            }
        }
    }
}

int main(int argc, char** argv) {
    boost::mpi::environment env;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
