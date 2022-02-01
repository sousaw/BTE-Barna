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
/// Test the functionality of the BasicProperties_calculator.

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
#include <bulk_properties.hpp>
#include <analytic1d.hpp>

TEST(analytic1D_basicprop_case, bulk_test) {
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
        alma::analytic1D::BasicProperties_calculator propCalc(
            poscar.get(), grid.get(), &w, T);

        // test bulk conductivity
        propCalc.setDirection(u100);
        double kappa = propCalc.getConductivity();
        Eigen::MatrixXd kappamatrix = alma::calc_kappa(*poscar, *grid, w, T);
        Eigen::MatrixXd buffer =
            (u100.transpose()).matrix() * kappamatrix * u100.matrix();
        double kappa_ref = buffer(0);
        double ratio = kappa / kappa_ref;
        EXPECT_NEAR(ratio, 1.0, 5e-3);

        // test heat capacity
        double Cv = propCalc.getCapacity();
        double Cv_ref = 1e27 * alma::calc_cv(*poscar, *grid, T);

        ratio = Cv / Cv_ref;
        EXPECT_NEAR(ratio, 1.0, 5e-3);
    }
}

TEST(analytic1D_basicprop_case, kappacumul_MFP_test) {
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
        alma::analytic1D::BasicProperties_calculator propCalc(
            poscar.get(), grid.get(), &w, T);
        propCalc.setDirection(u100);

        // test cumulative conductivity curves
        int Nbins = 9;

        propCalc.setLogMFPbins(1e-9, 1e-5, Nbins);
        propCalc.setLogRTbins(1e-12, 1e-9, Nbins);

        propCalc.resolveByMFP();
        Eigen::VectorXd kappacumul_byMFP = propCalc.getCumulativeConductivity();
        Eigen::VectorXd kappacumul_ref1(Nbins);
        kappacumul_ref1 << 0.09085889702, 0.9403198652, 4.278638123,
            20.78872976, 63.55033952, 94.76717674, 98.89460337, 136.9234453,
            147.6715806;
        Eigen::VectorXd ratiovector =
            kappacumul_byMFP.array() / kappacumul_ref1.array();

        for (int nbin = 0; nbin < Nbins; nbin++) {
            EXPECT_NEAR(ratiovector(nbin), 1.0, 5e-3);
        }
    }
}

TEST(analytic1D_basicprop_case, kappacumul_RT_test) {
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
        alma::analytic1D::BasicProperties_calculator propCalc(
            poscar.get(), grid.get(), &w, T);
        propCalc.setDirection(u100);

        // test cumulative conductivity curves
        int Nbins = 9;

        propCalc.setLogMFPbins(1e-9, 1e-5, Nbins);
        propCalc.setLogRTbins(1e-12, 1e-9, Nbins);

        propCalc.resolveByRT();
        Eigen::VectorXd kappacumul_byRT = propCalc.getCumulativeConductivity();
        Eigen::VectorXd kappacumul_ref2(Nbins);
        kappacumul_ref2 << 0.4831467689, 3.143165535, 7.665954394, 32.35090534,
            75.82629437, 90.79953246, 98.89460337, 125.379803, 147.6715806;
        Eigen::VectorXd ratiovector =
            kappacumul_byRT.array() / kappacumul_ref2.array();

        for (int nbin = 0; nbin < Nbins; nbin++) {
            EXPECT_NEAR(ratiovector(nbin), 1.0, 5e-3);
        }
    }
}

TEST(analytic1D_basicprop_case, thinfilm_crossplane_test) {
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
        alma::analytic1D::BasicProperties_calculator propCalc(
            poscar.get(), grid.get(), &w, T);
        propCalc.setDirection(u100);

        // test thickness dependent conductivity

        Eigen::VectorXd kappa_crossplane(3);
        Eigen::VectorXd thicknesslist(3);
        thicknesslist << 5e-6, 500e-9, 50e-9;

        for (int nfilm = 0; nfilm < 3; nfilm++) {
            propCalc.setCrossPlaneFilm(thicknesslist(nfilm));
            kappa_crossplane(nfilm) = propCalc.getConductivity();
        }

        Eigen::VectorXd kappa_crossplane_ref(3);
        kappa_crossplane_ref << 121.9813832, 76.32486756, 26.74947278;

        Eigen::VectorXd ratiovector =
            kappa_crossplane.array() / kappa_crossplane_ref.array();

        for (int nfilm = 0; nfilm < 3; nfilm++) {
            EXPECT_NEAR(ratiovector(nfilm), 1.0, 5e-3);
        }
    }
}

int main(int argc, char** argv) {
    boost::mpi::environment env;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
