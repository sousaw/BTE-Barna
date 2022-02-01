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
/// Test the functionality of the psi_calculator.

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

TEST(analytic1D_psi_case, bulk100_test) {
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

    if (my_id == 0) { // analytic1D code is not parallellised; avoid
                      // running duplicates.
        Nxi = 4;
        alma::analytic1D::psi_calculator psiCalc(
            poscar.get(), grid.get(), &w, T);
        psiCalc.setLogGrid(1e3, 1e9, Nxi);
        Eigen::VectorXd xigrid = psiCalc.getSpatialFrequencies();

        // Calculate psi functions in bulk Si for 3 directions
        psiCalc.normaliseOutput(true); // turn on optional rescaling
                                       // (output gets divided by
                                       // Dbulk*xi^2)

        psiCalc.setDirection(u100);
        Eigen::VectorXd psi100 = psiCalc.getPsi();

        Eigen::VectorXd psi100_ref(4);
        psi100_ref << 1.000585888, 0.9864015985, 0.4049977681, 0.005287157147;
        Eigen::VectorXd ratio100 = psi100.array() / psi100_ref.array();

        for (int nxi = 0; nxi < Nxi; nxi++) {
            EXPECT_NEAR(ratio100(nxi), 1.0, 5e-3);
        }
    }
}

TEST(analytic1D_psi_case, bulk110_test) {
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
    Eigen::Vector3d u110(1.0, 1.0, 0.0);

    // general variables
    int Nxi;

    if (my_id == 0) { // analytic1D code is not parallellised; avoid
                      // running duplicates.
        Nxi = 4;
        alma::analytic1D::psi_calculator psiCalc(
            poscar.get(), grid.get(), &w, T);
        psiCalc.setLogGrid(1e3, 1e9, Nxi);
        Eigen::VectorXd xigrid = psiCalc.getSpatialFrequencies();

        // Calculate psi functions in bulk Si for 3 directions
        psiCalc.normaliseOutput(true); // turn on optional rescaling
                                       // (output gets divided by
                                       // Dbulk*xi^2)

        psiCalc.setDirection(u110);
        Eigen::VectorXd psi110 = psiCalc.getPsi();

        Eigen::VectorXd psi110_ref(4);
        psi110_ref << 1.000585549, 0.9834471977, 0.3966867118, 0.005560183293;
        Eigen::VectorXd ratio110 = psi110.array() / psi110_ref.array();

        for (int nxi = 0; nxi < Nxi; nxi++) {
            EXPECT_NEAR(ratio110(nxi), 1.0, 5e-3);
        }
    }
}

TEST(analytic1D_psi_case, bulk111_test) {
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
    Eigen::Vector3d u111(1.0, 1.0, 1.0);

    // general variables
    int Nxi;

    if (my_id == 0) { // analytic1D code is not parallellised; avoid
                      // running duplicates.
        Nxi = 4;
        alma::analytic1D::psi_calculator psiCalc(
            poscar.get(), grid.get(), &w, T);
        psiCalc.setLogGrid(1e3, 1e9, Nxi);
        Eigen::VectorXd xigrid = psiCalc.getSpatialFrequencies();

        // Calculate psi functions in bulk Si for 3 directions
        psiCalc.normaliseOutput(true); // turn on optional rescaling
                                       // (output gets divided by
                                       // Dbulk*xi^2)

        psiCalc.setDirection(u111);
        Eigen::VectorXd psi111 = psiCalc.getPsi();

        Eigen::VectorXd psi111_ref(4);
        psi111_ref << 1.000585193, 0.981217029, 0.4013087323, 0.006630052249;
        Eigen::VectorXd ratio111 = psi111.array() / psi111_ref.array();

        for (int nxi = 0; nxi < Nxi; nxi++) {
            EXPECT_NEAR(ratio111(nxi), 1.0, 5e-3);
        }
    }
}

int main(int argc, char** argv) {
    boost::mpi::environment env;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
