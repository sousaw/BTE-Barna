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
/// Test whether we can correctly compute the phase space volume
/// of allowed three-phonon processes.

#include <iostream>
#include <functional>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>

TEST(p3_case, p3_test) {
    boost::mpi::environment env;

    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto si_dir = basedir / boost::filesystem::path("Si");
    auto poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto ifc_path = si_dir / boost::filesystem::path("FORCE_CONSTANTS");

    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto syms = alma::Symmetry_operations(*poscar);
    auto force_constants =
        alma::load_FORCE_CONSTANTS(ifc_path.string().c_str(), *poscar, 5, 5, 5);

    auto grid = std::make_unique<alma::Gamma_grid>(
        *poscar, syms, *force_constants, 12, 12, 12);

    grid->enforce_asr();
    auto nptk = grid->na * grid->nb * grid->nc;
    auto nmodes = grid->get_spectrum_at_q(0).omega.size();

    boost::mpi::communicator world;
    auto my_id = world.rank();
    auto processes = alma::find_allowed_threeph(*grid, world, 0.1);

    double my_P3_plus = 0.;
    double my_P3_minus = 0.;

    for (std::size_t i = 0; i < processes.size(); ++i) {
        auto g = processes[i].compute_gaussian();
        auto weights = grid->get_cardinal(processes[i].c);
        g *= weights;

        if (processes[i].type == alma::threeph_type::emission)
            my_P3_minus += g;
        else
            my_P3_plus += g;
    }

    double P3_minus = 0.;
    double P3_plus = 0.;
    boost::mpi::reduce(world, my_P3_minus, P3_minus, std::plus<double>(), 0);
    boost::mpi::reduce(world, my_P3_plus, P3_plus, std::plus<double>(), 0);
    double norm = nmodes * nmodes * nmodes * nptk * nptk;
    P3_minus /= norm;
    P3_plus /= norm;
    auto P3_total = 2. * (P3_plus + 0.5 * P3_minus) / 3.;

    if (my_id == 0) {
        EXPECT_TRUE(alma::almost_equal(P3_total, 0.002898838171, 1e-8, 1e-2));
        EXPECT_TRUE(alma::almost_equal(P3_plus, 0.002900801885, 1e-8, 1e-2));
        EXPECT_TRUE(alma::almost_equal(P3_minus, 0.002894910742, 1e-8, 1e-2));
    }
}
