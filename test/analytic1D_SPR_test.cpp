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
/// Test the functionality of SPR_calculator_RealSpace.

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

TEST(analytic1D_SPR_case, bulk_test) {
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

    if (my_id == 0) { // analytic1D code is not parallellised; avoid
                      // running duplicates.
        int Nx = 5;
        int Nt = 4;
        alma::analytic1D::SPR_calculator_RealSpace SPRcalc(
            poscar.get(), grid.get(), &w, T);
        SPRcalc.setDirection(u100);
        SPRcalc.setLinGrid(0.001, 5.0, Nx);
        SPRcalc.declareGridNormalised(true);
        SPRcalc.normaliseOutput(true);

        Eigen::VectorXd logt(Nt);
        logt.setLinSpaced(Nt, -8.0, -5.0);

        Eigen::MatrixXd Pxt(Nx, Nt);
        Eigen::MatrixXd Pxt_ref(Nx, Nt);
        Eigen::MatrixXd Pxt_ratio(Nx, Nt);

        // calculate results for Si infinite bulk

        for (int nt = 0; nt < Nt; nt++) {
            SPRcalc.setTime(std::pow(10.0, logt(nt)));
            Pxt.col(nt) = SPRcalc.getSPR();
        }

        Pxt_ref << 1.194984108, 1.067881885, 1.010958769, 1.000941424,
            0.3675134491, 0.4209313142, 0.4508768696, 0.4567400139,
            0.0291076858, 0.04290196176, 0.0444827355, 0.0440293595,
            0.00584946382, 0.003989173249, 0.00133382099, 0.0009296630073,
            0.002139785681, 0.0005312278416, 1.955415964e-05, 4.536475296e-06;

        Pxt_ratio = Pxt.array() / Pxt_ref.array();

        for (int nx = 0; nx < Nx; nx++) {
            for (int nt = 0; nt < Nt; nt++) {
                EXPECT_NEAR(Pxt_ratio(nx, nt), 1.0, 5e-3);
            }
        }
    }
}

TEST(analytic1D_SPR_case, SPR_resolvedbyMFP_test) {
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

    if (my_id == 0) { // analytic1D code is not parallellised; avoid
                      // running duplicates.
        int Nx = 5;
        int Nbins = 7;
        alma::analytic1D::SPR_calculator_RealSpace SPRcalc(
            poscar.get(), grid.get(), &w, T);
        SPRcalc.setDirection(u100);
        SPRcalc.setLinGrid(0.001, 5.0, Nx);
        SPRcalc.declareGridNormalised(true);
        SPRcalc.normaliseOutput(true);

        Eigen::MatrixXd Pxt(Nx, Nbins);
        Eigen::MatrixXd Pxt_ref(Nx, Nbins);
        Eigen::MatrixXd Pxt_ratio(Nx, Nbins);

        // Calculate MFP resolved results for Si infinite bulk at 10ns

        SPRcalc.setTime(10e-9);
        SPRcalc.setLogMFPbins(1e-9, 1e-6, Nbins);

        Pxt.resize(Nx, Nbins);
        Pxt = SPRcalc.resolveSPRbyMFP();

        Pxt_ref.resize(Nx, Nbins);
        Pxt_ref << 0.142719388, 0.2690127494, 0.1568190371, 0.1880156038,
            0.3582121282, 0.0582407846, 0.01767945528, 0.04386478924,
            0.08269394788, 0.04819840336, 0.05777558569, 0.1103347399,
            0.01833293241, 0.006719688234, 0.003471704101, 0.006545948849,
            0.003814456351, 0.004569434726, 0.008734089073, 0.001482552663,
            0.001165266074, 0.0006979144149, 0.001315810551, 0.0007667854801,
            0.000918385565, 0.001750947786, 0.0002895690084, 0.0003652182859,
            0.0002553215109, 0.0004813610013, 0.0002805160718, 0.0003359767712,
            0.0006403224761, 0.0001053370905, 0.0001458159539;

        Pxt_ratio = Pxt.array() / Pxt_ref.array();

        for (int nx = 0; nx < Nx; nx++) {
            for (int nbin = 0; nbin < Nbins; nbin++) {
                EXPECT_NEAR(Pxt_ratio(nx, nbin), 1.0, 5e-3);
            }
        }
    }
}

int main(int argc, char** argv) {
    boost::mpi::environment env;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
