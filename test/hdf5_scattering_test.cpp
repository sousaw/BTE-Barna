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
/// Test whether the functions designed to deal with the "/scattering"
/// group in HDF5 files work correctly.

#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <vasp_io.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>
#include <bulk_hdf5.hpp>

TEST(hdf5_scattering_case, hdf5_scattering_test) {
    boost::mpi::environment env;

    // Load some data for Si.
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

    // Create a trivially small q point grid.
    auto grid = std::make_unique<alma::Gamma_grid>(
        *poscar, syms, *force_constants, 4, 4, 4);

    grid->enforce_asr();

    // Look for three-phonon processes.
    boost::mpi::communicator world;
    auto my_id = world.rank();
    auto processes = alma::find_allowed_threeph(*grid, world, 0.1);

    // Precompute everything possible.
    for (auto& p : processes) {
        p.compute_gaussian();
        p.compute_vp2(*poscar, *grid, *thirdorder);
    }

    // Create a temporary file name to write the information.
    auto h5fn = boost::filesystem::unique_path();
    alma::save_bulk_hdf5(h5fn.string().c_str(),
                         "Si test",
                         *poscar,
                         syms,
                         *grid,
                         processes,
                         world);

    // Get a list of scattering subgroups in the file (should be empty
    // at this point).
    auto subgroups =
        alma::list_scattering_subgroups(h5fn.string().c_str(), world);
    ASSERT_EQ(0u, subgroups.size());

    // Add a subgroup (containing random data) to the file.
    std::size_t nmodes = grid->get_spectrum_at_q(0).omega.size();
    std::size_t nqpoints = grid->nqpoints;
    Eigen::ArrayXXd w0{nmodes, nqpoints};

    if (my_id == 0)
        w0.setRandom(nmodes, nqpoints);
    boost::mpi::broadcast(world, w0.data(), w0.size(), 0);
    alma::Scattering_subgroup subgroup{
        "test_subgroup", false, "This is a test subgroup", w0};
    alma::write_scattering_subgroup(h5fn.string().c_str(), subgroup, world);

    // Try to add a second subgroup with the wrong shape, and test
    // that the code throws an exception.
    if (my_id == 0) {
        Eigen::ArrayXXd w0p{nmodes / 3, nqpoints};
        alma::Scattering_subgroup subgroup2{
            "test_subgroup", false, "This is a test subgroup", w0p};
        EXPECT_THROW(alma::write_scattering_subgroup(
                         h5fn.string().c_str(), subgroup2, world),
                     alma::value_error);
    }

    // Now there should be one subgroup.
    subgroups = alma::list_scattering_subgroups(h5fn.string().c_str(), world);
    ASSERT_EQ(1u, subgroups.size());

    // Test that everything has been written correctly and can be
    // loaded.
    auto loaded = alma::load_scattering_subgroup(
        h5fn.string().c_str(), "test_subgroup", world);

    EXPECT_EQ(loaded.name, subgroup.name);
    EXPECT_EQ(loaded.preserves_symmetry, subgroup.preserves_symmetry);
    EXPECT_EQ(loaded.description, subgroup.description);
    EXPECT_EQ(loaded.w0.rows(), subgroup.w0.rows());
    EXPECT_EQ(loaded.w0.cols(), subgroup.w0.cols());

    for (std::size_t i = 0; i < nmodes; ++i)
        for (std::size_t j = 0; j < nqpoints; ++j)
            EXPECT_TRUE(alma::almost_equal(loaded.w0(i, j), subgroup.w0(i, j)));
    EXPECT_EQ(loaded.attributes, subgroup.attributes);
    EXPECT_EQ(loaded.datasets, subgroup.datasets);
    EXPECT_EQ(loaded.groups, subgroup.groups);

    // Remove the temporary file.
    if (my_id == 0)
        boost::filesystem::remove(h5fn);
}
