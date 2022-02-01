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
/// Compute the thermal conductivity of Si under the RTA
/// after reading all the relevant data from an HDF5 file
/// and check the result.

#include <iostream>
#include <functional>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <utilities.hpp>
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>
#include <bulk_properties.hpp>
#include <bulk_hdf5.hpp>

TEST(read_hdf5_case, read_hdf5_test) {
    boost::mpi::environment env;

    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto si_dir = basedir / boost::filesystem::path("Si");
    auto hdf5_path = si_dir / boost::filesystem::path("Si_threeph.h5");
    boost::mpi::communicator world;
    auto my_id = world.rank();

    auto hdf5_data = alma::load_bulk_hdf5(hdf5_path.string().c_str(), world);
    auto description = std::get<0>(hdf5_data);
    auto poscar = std::move(std::get<1>(hdf5_data));
    auto syms = std::move(std::get<2>(hdf5_data));
    auto grid = std::move(std::get<3>(hdf5_data));
    auto processes = std::move(std::get<4>(hdf5_data));

    double reference[] = {262.594,
                          188.086,
                          147.672,
                          122.172,
                          104.504,
                          91.4766,
                          81.4379,
                          73.4452,
                          66.9194,
                          61.4835};

    std::vector<double> Ts;

    for (auto i = 0; i < 10; ++i)
        Ts.emplace_back(200. + 50. * i);

    auto pos = 0;

    for (auto T : Ts) {
        Eigen::ArrayXXd total_w0(
            alma::calc_w0_threeph(*grid, *processes, T, world));

        if (my_id == 0) {
            auto kappa = alma::calc_kappa(*poscar, *grid, total_w0, T);
            EXPECT_TRUE(
                alma::almost_equal(reference[pos], kappa(0, 0), 1e-6, 5e-3));
            EXPECT_TRUE(
                alma::almost_equal(reference[pos], kappa(1, 1), 1e-6, 5e-3));
            EXPECT_TRUE(
                alma::almost_equal(reference[pos], kappa(2, 2), 1e-6, 5e-3));
            ++pos;
        }
    }
}
