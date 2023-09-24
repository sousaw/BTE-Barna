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
/// Test our implementation of isotopic scattering.

#include <iostream>
#include <functional>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <utilities.hpp>
#include <vasp_io.hpp>
#include <vc.hpp>
#include <qpoint_grid.hpp>
#include <isotopic_scattering.hpp>
#include <processes.hpp>
#include <bulk_properties.hpp>
#include <bulk_hdf5.hpp>

TEST(isotopic_scattering_case, si_isotopes_test) {
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
    auto threeph_processes = std::move(std::get<4>(hdf5_data));

    double ref_pure[] = {249.21709531,
                         178.55206587,
                         140.19919058,
                         115.99314240,
                         99.220090845,
                         86.851359212,
                         77.320001020,
                         69.731278562,
                         63.535233274,
                         58.374091685};
    double ref_natural[] = {218.05920065,
                            161.50922329,
                            129.20154822,
                            108.20226642,
                            93.366456496,
                            82.271732501,
                            73.629286860,
                            66.688382686,
                            60.980508354,
                            56.197143227};

    std::vector<double> Ts;

    for (auto i = 0; i < 10; ++i)
        Ts.emplace_back(200. + 50. * i);

    auto twoph_processes = alma::find_allowed_twoph(*grid, world);
    Eigen::ArrayXXd elastic_w0(
        alma::calc_w0_twoph(*poscar, *grid, twoph_processes, world));

    auto pos = 0;

    for (auto T : Ts) {
        Eigen::ArrayXXd anharmonic_w0(
            alma::calc_w0_threeph(*grid, *threeph_processes, T, world));
        Eigen::ArrayXXd total_w0(anharmonic_w0 + elastic_w0);

        if (my_id == 0) {
            auto kappa_pure =
                alma::calc_kappa(*poscar, *grid, *syms, anharmonic_w0, T);
            auto kappa_natural = alma::calc_kappa(*poscar, *grid, *syms, total_w0, T);
            EXPECT_NEAR(
                ref_pure[pos], kappa_pure(0, 0), 5e-3 * kappa_pure(0, 0));
            EXPECT_NEAR(
                ref_natural[pos], kappa_natural(0, 0), 5e-3 * kappa_pure(0, 0));
            ++pos;
        }
    }
}

TEST(isotopic_scattering_case, sige_test) {
    boost::mpi::environment env;

    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);

    auto si_dir = basedir / boost::filesystem::path("Si");
    auto si_poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto si_poscar = alma::load_POSCAR(si_poscar_path.string().c_str());
    auto si_ifc_path = si_dir / boost::filesystem::path("FORCE_CONSTANTS");
    auto si_ifcs = alma::load_FORCE_CONSTANTS(
        si_ifc_path.string().c_str(), *si_poscar, 5, 5, 5);
    auto si_thirdorder_path =
        si_dir / boost::filesystem::path("FORCE_CONSTANTS_3RD");
    auto si_thirdorder = alma::load_FORCE_CONSTANTS_3RD(
        si_thirdorder_path.string().c_str(), *si_poscar);

    auto ge_dir = basedir / boost::filesystem::path("Ge");
    auto ge_poscar_path = ge_dir / boost::filesystem::path("POSCAR");
    auto ge_poscar = alma::load_POSCAR(ge_poscar_path.string().c_str());
    auto ge_ifc_path = ge_dir / boost::filesystem::path("FORCE_CONSTANTS");
    auto ge_ifcs = alma::load_FORCE_CONSTANTS(
        ge_ifc_path.string().c_str(), *ge_poscar, 5, 5, 5);
    auto ge_thirdorder_path =
        ge_dir / boost::filesystem::path("FORCE_CONSTANTS_3RD");
    auto ge_thirdorder = alma::load_FORCE_CONSTANTS_3RD(
        ge_thirdorder_path.string().c_str(), *ge_poscar);

    std::vector<double> ratios({81.6, 18.4});
    auto vc_poscar = alma::vc_mix_structures({*si_poscar, *ge_poscar}, ratios);
    auto vc_ifcs = alma::vc_mix_harmonic_ifcs({*si_ifcs, *ge_ifcs}, ratios);
    auto vc_thirdorder =
        alma::vc_mix_thirdorder_ifcs({*si_thirdorder, *ge_thirdorder}, ratios);

    auto syms = alma::Symmetry_operations(*vc_poscar);
    auto grid = std::make_unique<alma::Gamma_grid>(
        *vc_poscar, syms, *vc_ifcs, 12, 12, 12);
    grid->enforce_asr();

    boost::mpi::communicator world;
    auto my_id = world.rank();

    auto twoph_processes = alma::find_allowed_twoph(*grid, world);
    Eigen::ArrayXXd elastic_w0(
        alma::calc_w0_twoph(*vc_poscar, *grid, twoph_processes, world));
    auto threeph_processes = alma::find_allowed_threeph(*grid, world, 0.1);

    // Note that these are not converged values. The Brillouin zone is
    // not adequately sampled by such a coarse grid. The values are only
    // intended to be quickly reproducible.
    double ref_pure[] = {203.17579466,
                         148.76624000,
                         118.20176149,
                         98.478721594,
                         84.611079886,
                         74.282735780,
                         66.267821497,
                         59.853941110,
                         54.597199986,
                         50.205722786};
    double ref_natural[] = {3.0066381765,
                            3.0151492643,
                            3.0050757338,
                            2.9864230091,
                            2.9636042392,
                            2.9387512037,
                            2.9129692273,
                            2.8868631463,
                            2.8607771916,
                            2.8349123902};

    std::vector<double> Ts;

    for (auto i = 0; i < 10; ++i)
        Ts.emplace_back(200. + 50. * i);

    // Precompute the matrix elements.
    for (std::size_t i = 0; i < threeph_processes.size(); ++i)
        threeph_processes[i].compute_vp2(*vc_poscar, *grid, *vc_thirdorder);
    auto pos = 0;

    for (auto T : Ts) {
        Eigen::ArrayXXd anharmonic_w0(
            alma::calc_w0_threeph(*grid, threeph_processes, T, world));
        Eigen::ArrayXXd total_w0(anharmonic_w0 + elastic_w0);

        if (my_id == 0) {
            auto kappa_pure =
                alma::calc_kappa(*vc_poscar, *grid, syms, anharmonic_w0, T);
            auto kappa_natural =
                alma::calc_kappa(*vc_poscar, *grid, syms, total_w0, T);
            EXPECT_NEAR(
                ref_pure[pos], kappa_pure(0, 0), 5e-3 * kappa_pure(0, 0));
            EXPECT_NEAR(ref_natural[pos],
                        kappa_natural(0, 0),
                        5e-3 * kappa_natural(0, 0));
            ++pos;
        }
    }
}

int main(int argc, char** argv) {
    boost::mpi::environment env;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
