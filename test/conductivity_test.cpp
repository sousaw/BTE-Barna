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
/// Compute the thermal conductivity of Si under the RTA to test
/// that we can compute three-phonon scattering probabilities
/// correctly.

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

TEST(conductivity_case, conductivity_test) {
    boost::mpi::environment env;

    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto si_dir = basedir / boost::filesystem::path("Si");
    auto poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto ifc_path = si_dir / boost::filesystem::path("FORCE_CONSTANTS");
    auto thirdorder_path =
        si_dir / boost::filesystem::path("FORCE_CONSTANTS_3RD");

    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto syms = alma::Symmetry_operations(*poscar);
    auto force_constants =
        alma::load_FORCE_CONSTANTS(ifc_path.string().c_str(), *poscar, 5, 5, 5);
    auto thirdorder = alma::load_FORCE_CONSTANTS_3RD(
        thirdorder_path.string().c_str(), *poscar);

    auto grid = std::make_unique<alma::Gamma_grid>(
        *poscar, syms, *force_constants, 12, 12, 12);

    grid->enforce_asr();

    boost::mpi::communicator world;
    auto my_id = world.rank();
    auto processes = alma::find_allowed_threeph(*grid, world, 0.1);

    double reference[] = {249.217,
                          178.552,
                          140.199,
                          115.993,
                          99.2201,
                          86.8514,
                          77.3200,
                          69.7313,
                          63.5352,
                          58.3741};

    std::vector<double> Ts;

    for (auto i = 0; i < 10; ++i)
        Ts.emplace_back(200. + 50. * i);

    // Precompute the matrix elements.
    for (std::size_t i = 0; i < processes.size(); ++i)
        processes[i].compute_vp2(*poscar, *grid, *thirdorder);
    auto pos = 0;

    for (auto T : Ts) {
        Eigen::ArrayXXd total_w0(
            alma::calc_w0_threeph(*grid, processes, T, world));

        if (my_id == 0) {
            auto kappa = alma::calc_kappa(*poscar, *grid, syms, total_w0, T);
            EXPECT_TRUE(
                alma::almost_equal(reference[pos], kappa(0, 0), 1e-6, 5e-3));
            EXPECT_TRUE(
                alma::almost_equal(reference[pos], kappa(1, 1), 1e-6, 5e-3));
            EXPECT_TRUE(
                alma::almost_equal(reference[pos], kappa(2, 2), 1e-6, 5e-3));
            // Test the scalar calculation as well.
            Eigen::Vector3d direction{Eigen::Vector3d::Random()};
            auto kappa1d =
                alma::calc_kappa_1d(*poscar, *grid, total_w0, T, direction);
            direction.normalize();
            EXPECT_TRUE(alma::almost_equal(
                direction.dot(kappa * direction), kappa1d, 1e-8, 1e-3));
            kappa = syms.symmetrize_m<double>(kappa, true);
            kappa(0, 0) = kappa(1, 1) = kappa(2, 2) = 0.;
            EXPECT_TRUE(alma::almost_equal(kappa.norm(), 0., 1e-8, 0.));
            ++pos;
        }
    }
}
