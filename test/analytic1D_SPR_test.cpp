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

        Pxt_ref << 1.171354946155000000, 1.057926093573489,     1.009236414520531,     1.000740661149974,
                   0.377998688686548100, 0.4262744195674638,    0.4518696218703666,    0.456859971003077,
                   0.031076549568823980, 0.04286897118580943,   0.04442396081146781,   0.0440168096332547,
                   0.005340755273538810, 0.003692878565291178,  0.001260719040458833,  0.0009225089148600758,
                   0.001838063743397109, 0.0004547747145356594, 1.560169244693463e-05, 4.392897570509941e-06;
            
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
        Pxt_ref << 0.1467656160141119    , 0.2517955610176555    , 0.1607865427039096    , 0.1808380981628194    , 0.3565060322244741    , 0.05073240562316528   , 0.01994858697926052  ,
                   0.04733347659582447   , 0.0812165877638579    , 0.05185440165708814   , 0.05831035762941063   , 0.1152126715730229    , 0.01671589469714046   , 0.007717520447501565 ,
                   0.003888633801037517  , 0.006673209513393253  , 0.00425969705035071   , 0.0047870494080686    , 0.00946956209123719   , 0.001400900290319257  , 0.00122297520878993  ,
                   0.000668488322601446  , 0.001147106019471539  , 0.0007322509158005205 , 0.0008227247159242492 , 0.001623446393608108  , 0.0002350108645855888 , 0.0003358831850631388,
                   0.0002300911854584819 , 0.0003948205086973243 , 0.0002520380597888477 , 0.0002831819260704965 , 0.0005584858983273304 , 8.035395193669677e-05 , 0.0001307850272676149;

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
