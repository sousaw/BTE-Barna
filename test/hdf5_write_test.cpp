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
/// Test whether all information about bulk Si
/// can be written to an hdf5 file.

#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>
#include "gtest/gtest.h"
#include <bulk_hdf5.hpp>

TEST(hdf5_write_case, hdf5_write_test) {
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
    auto processes = alma::find_allowed_threeph(*grid, world, 0.1);

    // Precompute everything possible.
    for (auto& p : processes) {
        p.compute_gaussian();
        p.compute_vp2(*poscar, *grid, *thirdorder);
    }

    auto h5fn = si_dir / boost::filesystem::path("Si_threeph.h5");
    alma::save_bulk_hdf5(h5fn.string().c_str(),
                         "Si test",
                         *poscar,
                         syms,
                         *grid,
                         processes,
                         world);
}
