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
/// Verify that the transient_analytic1d executable works correctly.

#include <cstdlib>
#include <iostream>
#include <functional>
#include <fstream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include "gtest/gtest.h"
#include <io_utils.hpp>
#include <Eigen/Dense>
#include <cmakevars.hpp>
#include <utilities.hpp>
#include <where_is_transient_analytic1d.hpp>

// Randomly-named directory to store the output of the program.
std::string targetdir;

class G_Environment : public ::testing::Environment {
public:
    virtual ~G_Environment() {
    }
    virtual void SetUp() {
        std::string path_prefix(boost::filesystem::path(transient_analytic1d)
                                    .parent_path()
                                    .string());

        // BUILD XML INPUT FILE
        std::string xmlfilename =
            (boost::filesystem::path(path_prefix) /
             boost::filesystem::path("test_transient_analytic1d.xml"))
                .string();
        std::ofstream xmlwriter(xmlfilename);

        targetdir = (boost::filesystem::path("test_results") /
                     boost::filesystem::unique_path(
                         "transient_analytic1d_%%%%-%%%%-%%%%-%%%%"))
                        .string();

        xmlwriter << "<analytic1d>" << std::endl;
        xmlwriter << "  <H5repository root_directory=\"";
        auto repository_root = boost::filesystem::path(TEST_RESOURCE_DIR);
        xmlwriter << repository_root.string() << "\"/>" << std::endl;
        xmlwriter << "  <compound directory=\"Si\" base=\"Si\" gridA=\"12\" "
                     "gridB=\"12\" gridC=\"12\"/>"
                  << std::endl;
        xmlwriter << "  <transportAxis x=\"0\" y=\"0\" z=\"1\"/>" << std::endl;
        xmlwriter << "  <spacecurves spacepoints = \"20\" "
                     "timepoints=\"{10e-9,100e-9,1e-6,10e-6}\"/>"
                  << std::endl;
        xmlwriter << "  <sourcetransient timesweep=\"log\" tstart=\"1e-9\" "
                     "tstop=\"1e-4\" points=\"6\"/>"
                  << std::endl;
        xmlwriter << "  <MSD timesweep=\"log\" tstart=\"1e-12\" tstop=\"1e-6\" "
                     "points=\"7\"/>"
                  << std::endl;
        xmlwriter << "  <target directory=\"" << targetdir << "\"/>"
                  << std::endl;
        xmlwriter << "</analytic1d>" << std::endl;
        xmlwriter.close();

        // RUN EXECUTABLE
        std::string command = (boost::filesystem::path(path_prefix) /
                               boost::filesystem::path("transient_analytic1d"))
                                  .string() +
                              " " + xmlfilename;
        ASSERT_EQ(std::system(command.c_str()), 0);
    }
    virtual void TearDown() {
    }
};

TEST(spacecurve1_case, transient_analytic1d_test) {
    Eigen::MatrixXd data = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K_10ns.spacecurve"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref(20, 3);
    data_ref << -6.45011e-06, 0.000426481, 7.08852e-07, -5.77115e-06,
        0.000634314, 8.57598e-06, -5.09219e-06, 0.000969288, 7.86519e-05,
        -4.41323e-06, 0.00156867, 0.000546804, -3.73428e-06, 0.00291792,
        0.00288171, -3.05532e-06, 0.00701135, 0.0115124, -2.37636e-06,
        0.0210139, 0.0348642, -1.6974e-06, 0.0615179, 0.0800367, -1.01844e-06,
        0.139666, 0.139282, -3.3948e-07, 0.216203, 0.183738, 3.3948e-07,
        0.216203, 0.183738, 1.01844e-06, 0.139666, 0.139282, 1.6974e-06,
        0.0615179, 0.0800367, 2.37636e-06, 0.0210139, 0.0348642, 3.05532e-06,
        0.00701135, 0.0115124, 3.73428e-06, 0.00291792, 0.00288171, 4.41323e-06,
        0.00156867, 0.000546804, 5.09219e-06, 0.000969288, 7.86519e-05,
        5.77115e-06, 0.000634314, 8.57598e-06, 6.45011e-06, 0.000426481,
        7.08852e-07;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data.col(ncol) - data_ref.col(ncol)).norm() /
                         data_ref.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(spacecurve2_case, transient_analytic1d_test) {
    Eigen::MatrixXd data = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K_100ns.spacecurve"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref(20, 3);
    data_ref << -2.0397e-05, 3.06086e-05, 2.24159e-07, -1.825e-05, 7.15429e-05,
        2.71196e-06, -1.61029e-05, 0.000173304, 2.48719e-05, -1.39559e-05,
        0.00044078, 0.000172915, -1.18088e-05, 0.00119182, 0.000911278,
        -9.66176e-06, 0.00338704, 0.00364055, -7.5147e-06, 0.00943525, 0.011025,
        -5.36764e-06, 0.02296, 0.0253098, -3.22059e-06, 0.0438951, 0.0440449,
        -1.07353e-06, 0.0616253, 0.058103, 1.07353e-06, 0.0616253, 0.058103,
        3.22059e-06, 0.0438951, 0.0440449, 5.36764e-06, 0.02296, 0.0253098,
        7.5147e-06, 0.00943525, 0.011025, 9.66176e-06, 0.00338704, 0.00364055,
        1.18088e-05, 0.00119182, 0.000911278, 1.39559e-05, 0.00044078,
        0.000172915, 1.61029e-05, 0.000173304, 2.48719e-05, 1.825e-05,
        7.15429e-05, 2.71196e-06, 2.0397e-05, 3.06086e-05, 2.24159e-07;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data.col(ncol) - data_ref.col(ncol)).norm() /
                         data_ref.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(spacecurve3_case, transient_analytic1d_test) {
    Eigen::MatrixXd data = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K_1us.spacecurve"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref(20, 3);
    data_ref << -6.45011e-05, 3.41688e-07, 7.08852e-08, -5.77115e-05,
        2.22979e-06, 8.57598e-07, -5.09219e-05, 1.32471e-05, 7.86519e-06,
        -4.41323e-05, 6.95189e-05, 5.46804e-05, -3.73428e-05, 0.00031155,
        0.000288171, -3.05532e-05, 0.0011525, 0.00115124, -2.37636e-05,
        0.00341194, 0.00348642, -1.6974e-05, 0.00787771, 0.00800367,
        -1.01844e-05, 0.0139108, 0.0139282, -3.3948e-06, 0.018545, 0.0183738,
        3.3948e-06, 0.018545, 0.0183738, 1.01844e-05, 0.0139108, 0.0139282,
        1.6974e-05, 0.00787771, 0.00800367, 2.37636e-05, 0.00341194, 0.00348642,
        3.05532e-05, 0.0011525, 0.00115124, 3.73428e-05, 0.00031155,
        0.000288171, 4.41323e-05, 6.95189e-05, 5.46804e-05, 5.09219e-05,
        1.32471e-05, 7.86519e-06, 5.77115e-05, 2.22979e-06, 8.57598e-07,
        6.45011e-05, 3.41688e-07, 7.08852e-08;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data.col(ncol) - data_ref.col(ncol)).norm() /
                         data_ref.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(spacecurve4_case, transient_analytic1d_test) {
    Eigen::MatrixXd data = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K_10us.spacecurve"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref(20, 3);
    data_ref << -0.00020397, 2.70924e-08, 2.24159e-08, -0.0001825, 3.04971e-07,
        2.71196e-07, -0.000161029, 2.65309e-06, 2.48719e-06, -0.000139559,
        1.78312e-05, 1.72915e-05, -0.000118088, 9.21545e-05, 9.11278e-05,
        -9.66176e-05, 0.000364648, 0.000364055, -7.5147e-05, 0.00110069,
        0.0011025, -5.36764e-05, 0.00252709, 0.00253098, -3.22059e-05,
        0.00440339, 0.00440449, -1.07353e-05, 0.00581472, 0.0058103,
        1.07353e-05, 0.00581472, 0.0058103, 3.22059e-05, 0.00440339, 0.00440449,
        5.36764e-05, 0.00252709, 0.00253098, 7.5147e-05, 0.00110069, 0.0011025,
        9.66176e-05, 0.000364648, 0.000364055, 0.000118088, 9.21545e-05,
        9.11278e-05, 0.000139559, 1.78312e-05, 1.72915e-05, 0.000161029,
        2.65309e-06, 2.48719e-06, 0.0001825, 3.04971e-07, 2.71196e-07,
        0.00020397, 2.70924e-08, 2.24159e-08;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data.col(ncol) - data_ref.col(ncol)).norm() /
                         data_ref.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(sourcetransient_case, transient_analytic1d_test) {
    Eigen::MatrixXd data = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path(
             "Si_12_12_12_0,0,1_300K_1ns_100us.sourcetransient"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref(6, 3);
    data_ref << 1e-09, 0.842106, 1.40001, 1e-08, 0.228608, 1.20186, 1e-07,
        0.0643381, 1.06963, 1e-06, 0.0192265, 1.0108, 1e-05, 0.00602044, 1.0009,
        0.0001, 0.00190178, 0.999824;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data.col(ncol) - data_ref.col(ncol)).norm() /
                         data_ref.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(MSD_case, transient_analytic1d_test) {
    Eigen::MatrixXd data = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K_1ps_1us.MSD"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref(7, 3);
    data_ref << 1e-12, 2.02828e-18, 0.012188, 1e-11, 1.4445e-16, 0.0868009,
        1e-10, 6.72649e-15, 0.404198, 1e-09, 1.23714e-13, 0.743401, 1e-08,
        1.60598e-12, 0.96504, 1e-07, 1.65922e-11, 0.997033, 1e-06, 1.66454e-10,
        1.00023;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data.col(ncol) - data_ref.col(ncol)).norm() /
                         data_ref.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

int main(int argc, char** argv) {
    boost::mpi::environment env;

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new G_Environment);
    return RUN_ALL_TESTS();
}