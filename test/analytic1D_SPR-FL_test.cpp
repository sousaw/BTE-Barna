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

        Pref_RE << 2.185565324999054e-10, 2.183408917400566e-14, 1.276057885195172e-18,
                   7.601233815678526e-07, 2.159512761962504e-10, 1.276508241597913e-14,
                   2.738248127245801e-10, 2.737542071757401e-10, 7.634808264993343e-11,
                   2.080929050761724e-12, 2.080919593922536e-12, 2.049213430047117e-12;


        Pref_IM << -1.591549445193995e-06, -1.591549475086651e-08, -1.591549436164946e-10,
                   -5.597667421340535e-07, -1.591666253836578e-08, -1.591601902485261e-10,
                   -4.399168551581259e-14, -4.398028919778141e-12, -1.235891987501402e-10,
                   -1.403268145172194e-17, -1.403081917278291e-15, -1.078760071601977e-13;

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
