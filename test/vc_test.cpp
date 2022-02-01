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
/// Test our implementation of the virtual crystal approximation.

#include <iostream>
#include <boost/filesystem.hpp>
#include "gtest/gtest.h"
#include <cmakevars.hpp>
#include <utilities.hpp>
#include <periodic_table.hpp>
#include <structures.hpp>
#include <vasp_io.hpp>
#include <vc.hpp>

TEST(vc_test_case, constructs_vc) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);

    auto si_dir = basedir / boost::filesystem::path("Si");
    auto si_poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto si_poscar = alma::load_POSCAR(si_poscar_path.string().c_str());
    auto ge_dir = basedir / boost::filesystem::path("Ge");
    auto ge_poscar_path = ge_dir / boost::filesystem::path("POSCAR");
    auto ge_poscar = alma::load_POSCAR(ge_poscar_path.string().c_str());

    auto vc = alma::vc_mix_structures({*si_poscar, *si_poscar}, {1.0, 1.0});

    EXPECT_TRUE(
        alma::almost_equal(0., (vc->lattvec - si_poscar->lattvec).norm()));
    EXPECT_EQ(1u, vc->get_nelements());
    EXPECT_EQ("Si:1.00000000", vc->elements[0]);

    vc = alma::vc_mix_structures({*si_poscar, *ge_poscar}, {1.0, 1.0});
    Eigen::Matrix3d expected(.5 * (si_poscar->lattvec + ge_poscar->lattvec));
    EXPECT_TRUE(alma::almost_equal(0., .5 * (vc->lattvec - expected).norm()));
    EXPECT_EQ(1u, vc->get_nelements());
    EXPECT_EQ("Ge:0.50000000;Si:0.50000000", vc->elements[0]);

    auto inas_dir = basedir / boost::filesystem::path("InAs");
    auto inas_poscar_path = inas_dir / boost::filesystem::path("POSCAR");
    auto inas_poscar = alma::load_POSCAR(inas_poscar_path.string().c_str());
    auto gaas_dir = basedir / boost::filesystem::path("GaAs");
    auto gaas_poscar_path = gaas_dir / boost::filesystem::path("POSCAR");
    auto gaas_poscar = alma::load_POSCAR(gaas_poscar_path.string().c_str());

    vc = alma::vc_mix_structures({*inas_poscar, *inas_poscar}, {1.0, 1.0});

    EXPECT_TRUE(
        alma::almost_equal(0., (vc->lattvec - inas_poscar->lattvec).norm()));
    EXPECT_EQ(2u, vc->get_nelements());
    EXPECT_EQ("In:1.00000000", vc->elements[0]);
    EXPECT_EQ("As:1.00000000", vc->elements[1]);

    vc = alma::vc_mix_structures({*inas_poscar, *gaas_poscar}, {1.0, 1.0});
    expected = .5 * (inas_poscar->lattvec + gaas_poscar->lattvec);
    EXPECT_TRUE(alma::almost_equal(0., .5 * (vc->lattvec - expected).norm()));
    EXPECT_EQ(2u, vc->get_nelements());
    EXPECT_EQ("Ga:0.50000000;In:0.50000000", vc->elements[0]);
    EXPECT_EQ("As:1.00000000", vc->elements[1]);
}

TEST(vc_test_case, mix_structures_throws) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);

    auto si_dir = basedir / boost::filesystem::path("Si");
    auto si_poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto si_poscar = alma::load_POSCAR(si_poscar_path.string().c_str());
    auto bi2o3_dir = basedir / boost::filesystem::path("Bi2O3");
    auto bi2o3_poscar_path = bi2o3_dir / boost::filesystem::path("POSCAR");
    auto bi2o3_poscar = alma::load_POSCAR(bi2o3_poscar_path.string().c_str());

    EXPECT_THROW(
        alma::vc_mix_structures({*si_poscar, *bi2o3_poscar}, {1.0, 1.0}),
        alma::value_error);
}

TEST(vc_test_case, mixes_dielectric_parameters) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto inas_dir = basedir / boost::filesystem::path("InAs");
    auto inas_born_path = inas_dir / boost::filesystem::path("BORN");
    auto inas_born = alma::load_BORN(inas_born_path.string().c_str());
    auto gaas_dir = basedir / boost::filesystem::path("GaAs");
    auto gaas_born_path = gaas_dir / boost::filesystem::path("BORN");
    auto gaas_born = alma::load_BORN(gaas_born_path.string().c_str());

    auto vc_born = alma::vc_mix_dielectric_parameters({*inas_born, *inas_born},
                                                      {1.0, 1.0});

    EXPECT_TRUE(
        alma::almost_equal(0., (vc_born->epsilon - inas_born->epsilon).norm()));
    EXPECT_EQ(2u, vc_born->born.size());

    for (std::size_t ib = 0; ib < vc_born->born.size(); ++ib)
        EXPECT_TRUE(alma::almost_equal(
            0., (vc_born->born[ib] - inas_born->born[ib]).norm()));

    vc_born = alma::vc_mix_dielectric_parameters({*gaas_born, *inas_born},
                                                 {1.0, 1.0});

    Eigen::Matrix3d refeps(.5 * (inas_born->epsilon + gaas_born->epsilon));
    EXPECT_TRUE(alma::almost_equal(0., (refeps - vc_born->epsilon).norm()));
    EXPECT_EQ(2u, vc_born->born.size());

    for (std::size_t ib = 0; ib < vc_born->born.size(); ++ib) {
        Eigen::Matrix3d refborn(.5 *
                                (inas_born->born[ib] + gaas_born->born[ib]));
        EXPECT_TRUE(
            alma::almost_equal(0., (vc_born->born[ib] - refborn).norm()));
    }
}

TEST(vc_test_case, mix_dielectric_parameters_throws) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto inas_dir = basedir / boost::filesystem::path("InAs");
    auto inas_born_path = inas_dir / boost::filesystem::path("BORN");
    auto inas_born = alma::load_BORN(inas_born_path.string().c_str());
    auto gaas_dir = basedir / boost::filesystem::path("GaAs");
    auto gaas_born_path = gaas_dir / boost::filesystem::path("BORN");
    auto gaas_born = alma::load_BORN(gaas_born_path.string().c_str());

    // Add more Born charges to make both objects incompatible.
    std::vector<Eigen::MatrixXd> born(gaas_born->born);

    for (auto b : gaas_born->born)
        born.emplace_back(b);
    gaas_born = std::make_unique<alma::Dielectric_parameters>(
        born, gaas_born->epsilon);

    EXPECT_THROW(alma::vc_mix_dielectric_parameters({*inas_born, *gaas_born},
                                                    {1.0, 1.0}),
                 alma::value_error);
}

TEST(vc_test_case, mixes_harmonic_ifcs) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);

    auto si_dir = basedir / boost::filesystem::path("Si");
    auto si_poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto si_ifc_path = si_dir / boost::filesystem::path("FORCE_CONSTANTS");
    auto si_poscar = alma::load_POSCAR(si_poscar_path.string().c_str());
    auto si_ifcs = alma::load_FORCE_CONSTANTS(
        si_ifc_path.string().c_str(), *si_poscar, 5, 5, 5);

    auto ge_dir = basedir / boost::filesystem::path("Ge");
    auto ge_poscar_path = ge_dir / boost::filesystem::path("POSCAR");
    auto ge_ifc_path = ge_dir / boost::filesystem::path("FORCE_CONSTANTS");
    auto ge_poscar = alma::load_POSCAR(ge_poscar_path.string().c_str());
    auto ge_ifcs = alma::load_FORCE_CONSTANTS(
        ge_ifc_path.string().c_str(), *ge_poscar, 5, 5, 5);

    auto vc_ifcs = alma::vc_mix_harmonic_ifcs({*si_ifcs, *si_ifcs}, {1.0, 1.0});

    EXPECT_EQ(si_ifcs->na, vc_ifcs->na);
    EXPECT_EQ(si_ifcs->nb, vc_ifcs->nb);
    EXPECT_EQ(si_ifcs->nc, vc_ifcs->nc);
    auto nblocks = si_ifcs->ifcs.size();
    EXPECT_EQ(nblocks, vc_ifcs->ifcs.size());

    for (decltype(nblocks) ib = 0; ib < nblocks; ++ib) {
        EXPECT_EQ(si_ifcs->pos[ib], vc_ifcs->pos[ib]);
        EXPECT_TRUE(alma::almost_equal(
            0., (si_ifcs->ifcs[ib] - vc_ifcs->ifcs[ib]).norm()));
    }

    vc_ifcs = alma::vc_mix_harmonic_ifcs({*si_ifcs, *ge_ifcs}, {1.0, 1.0});
    EXPECT_EQ(si_ifcs->na, vc_ifcs->na);
    EXPECT_EQ(si_ifcs->nb, vc_ifcs->nb);
    EXPECT_EQ(si_ifcs->nc, vc_ifcs->nc);
    EXPECT_EQ(nblocks, vc_ifcs->ifcs.size());

    for (decltype(nblocks) ib = 0; ib < nblocks; ++ib) {
        EXPECT_EQ(si_ifcs->pos[ib], vc_ifcs->pos[ib]);
        Eigen::MatrixXd reference{.5 * (si_ifcs->ifcs[ib] + ge_ifcs->ifcs[ib])};
        EXPECT_TRUE(
            alma::almost_equal(0., (vc_ifcs->ifcs[ib] - reference).norm()));
    }
}

TEST(vc_test_case, mix_harmonic_ifcs_throws) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto si_dir = basedir / boost::filesystem::path("Si");
    auto poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto ifc_path = si_dir / boost::filesystem::path("FORCE_CONSTANTS");
    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto force_constants =
        alma::load_FORCE_CONSTANTS(ifc_path.string().c_str(), *poscar, 5, 5, 5);
    auto na = force_constants->na;
    auto nb = force_constants->nb;
    auto nc = force_constants->nc;

    std::vector<alma::Triple_int> pos{force_constants->pos};
    std::vector<Eigen::MatrixXd> blocks;

    for (auto& b : force_constants->ifcs)
        blocks.emplace_back(b);
    // Add a further block to make mixing impossible.
    pos.emplace_back(alma::Triple_int({{666, 666, 666}}));
    blocks.emplace_back(blocks.back());
    EXPECT_THROW(
        alma::vc_mix_harmonic_ifcs(
            {*force_constants, alma::Harmonic_ifcs(pos, blocks, na, nb, nc)},
            {1.0, 1.0}),
        alma::value_error);
}

TEST(vc_test_case, mixes_thirdorder_ifcs) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);

    auto si_dir = basedir / boost::filesystem::path("Si");
    auto si_poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto si_thirdorder_path =
        si_dir / boost::filesystem::path("FORCE_CONSTANTS_3RD");
    auto si_poscar = alma::load_POSCAR(si_poscar_path.string().c_str());
    auto si_thirdorder = alma::load_FORCE_CONSTANTS_3RD(
        si_thirdorder_path.string().c_str(), *si_poscar);

    auto ge_dir = basedir / boost::filesystem::path("Si");
    auto ge_poscar_path = ge_dir / boost::filesystem::path("POSCAR");
    auto ge_thirdorder_path =
        ge_dir / boost::filesystem::path("FORCE_CONSTANTS_3RD");
    auto ge_poscar = alma::load_POSCAR(ge_poscar_path.string().c_str());
    auto ge_thirdorder = alma::load_FORCE_CONSTANTS_3RD(
        ge_thirdorder_path.string().c_str(), *ge_poscar);

    auto vc_thirdorder = alma::vc_mix_thirdorder_ifcs(
        {*si_thirdorder, *si_thirdorder}, {1.0, 1.0});
    auto nblocks = si_thirdorder->size();

    EXPECT_EQ(nblocks, vc_thirdorder->size());

    for (decltype(nblocks) ib = 0; ib < nblocks; ++ib) {
        EXPECT_EQ((*si_thirdorder)[ib].i, (*vc_thirdorder)[ib].i);
        EXPECT_EQ((*si_thirdorder)[ib].j, (*vc_thirdorder)[ib].j);
        EXPECT_EQ((*si_thirdorder)[ib].k, (*vc_thirdorder)[ib].k);
        EXPECT_TRUE(alma::almost_equal(
            0, ((*si_thirdorder)[ib].rj - (*vc_thirdorder)[ib].rj).norm()));
        EXPECT_TRUE(alma::almost_equal(
            0, ((*si_thirdorder)[ib].rk - (*vc_thirdorder)[ib].rk).norm()));

        for (std::size_t alpha = 0; alpha < 3; ++alpha)
            for (std::size_t beta = 0; beta < 3; ++beta)
                for (std::size_t gamma = 0; gamma < 3; ++gamma)
                    EXPECT_TRUE(alma::almost_equal(
                        (*si_thirdorder)[ib].ifc(alpha, beta, gamma),
                        (*vc_thirdorder)[ib].ifc(alpha, beta, gamma)));
    }

    vc_thirdorder = alma::vc_mix_thirdorder_ifcs(
        {*si_thirdorder, *ge_thirdorder}, {1.0, 1.0});
    EXPECT_EQ(nblocks, vc_thirdorder->size());

    for (decltype(nblocks) ib = 0; ib < nblocks; ++ib) {
        EXPECT_EQ((*si_thirdorder)[ib].i, (*vc_thirdorder)[ib].i);
        EXPECT_EQ((*si_thirdorder)[ib].j, (*vc_thirdorder)[ib].j);
        EXPECT_EQ((*si_thirdorder)[ib].k, (*vc_thirdorder)[ib].k);
        Eigen::Vector3d refj{
            .5 * ((*si_thirdorder)[ib].rj + (*ge_thirdorder)[ib].rj)};
        EXPECT_TRUE(
            alma::almost_equal(0, (refj - (*vc_thirdorder)[ib].rj).norm()));
        Eigen::Vector3d refk{
            .5 * ((*si_thirdorder)[ib].rk + (*ge_thirdorder)[ib].rk)};
        EXPECT_TRUE(
            alma::almost_equal(0, (refk - (*vc_thirdorder)[ib].rk).norm()));

        for (std::size_t alpha = 0; alpha < 3; ++alpha)
            for (std::size_t beta = 0; beta < 3; ++beta)
                for (std::size_t gamma = 0; gamma < 3; ++gamma) {
                    double ref =
                        .5 * ((*si_thirdorder)[ib].ifc(alpha, beta, gamma) +
                              (*ge_thirdorder)[ib].ifc(alpha, beta, gamma));
                    EXPECT_TRUE(alma::almost_equal(
                        ref, (*vc_thirdorder)[ib].ifc(alpha, beta, gamma)));
                }
    }
}

TEST(vc_test_case, mix_thirdorder_ifcs_throws) {
    auto basedir = boost::filesystem::path(TEST_RESOURCE_DIR);
    auto si_dir = basedir / boost::filesystem::path("Si");
    auto poscar_path = si_dir / boost::filesystem::path("POSCAR");
    auto thirdorder_path =
        si_dir / boost::filesystem::path("FORCE_CONSTANTS_3RD");
    auto poscar = alma::load_POSCAR(poscar_path.string().c_str());
    auto thirdorder = alma::load_FORCE_CONSTANTS_3RD(
        thirdorder_path.string().c_str(), *poscar);

    std::vector<alma::Thirdorder_ifcs> copy(*thirdorder);
    copy.push_back(copy.back());
    EXPECT_THROW(alma::vc_mix_thirdorder_ifcs({*thirdorder, copy}, {1.0, 1.0}),
                 alma::value_error);
}
