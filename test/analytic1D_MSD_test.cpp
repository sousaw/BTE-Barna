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
/// Test the functionality of the MSD_calculators.

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

TEST(analytic1D_MSD_case, bulk_test) {
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
    int Nt;
    Eigen::VectorXd MSD;
    Eigen::VectorXd MSD_ref;
    Eigen::VectorXd MSD_ratio;

    if (my_id == 0) { // analytic1D code is not parallellised; avoid
                      // running duplicates.
        Nt = 7;

        alma::analytic1D::MSD_calculator_RealTime MSDcalc(
            poscar.get(), grid.get(), &w, T);
        MSDcalc.setDirection(u100);
        MSDcalc.setLogGrid(1e-12, 1e-6, Nt);
        MSDcalc.normaliseOutput(false);

        MSD = MSDcalc.getMSD();
        MSD_ref.resize(Nt);
        MSD_ref << 2.258270594e-18, 1.539701019e-16, 7.205645714e-15,
            1.351483594e-13, 1.753307494e-12, 1.811198571e-11, 1.81698549e-10;
        ;
        MSD_ratio = MSD.array() / MSD_ref.array();

        for (int nt = 0; nt < Nt; nt++) {
            EXPECT_TRUE(alma::almost_equal(MSD_ratio(nt), 1.0, 1e-4));
        }
    }
}

int main(int argc, char** argv) {
    boost::mpi::environment env;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
