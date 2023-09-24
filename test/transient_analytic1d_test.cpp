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
    data_ref << -6.30348e-06,  0.000375327,  7.25341e-07,
                -5.63996e-06,   0.00056464,  8.77547e-06,
                -4.97643e-06,  0.000888687,  8.04815e-05,
                -4.31291e-06,   0.00151793,  0.000559524,
                -3.64938e-06,   0.00303649,   0.00294875,
                -2.98586e-06,   0.00768101,    0.0117802,
                -2.32234e-06,    0.0229392,    0.0356752,
                -1.65881e-06,    0.0649848,    0.0818984,
                -9.95287e-07,     0.143006,     0.142522,
                -3.31762e-07,     0.217567,     0.188012,
                 3.31762e-07,     0.217567,     0.188012,
                 9.95287e-07,     0.143006,     0.142522,
                 1.65881e-06,    0.0649848,    0.0818984,
                 2.32234e-06,    0.0229392,    0.0356752,
                 2.98586e-06,   0.00768101,    0.0117802,
                 3.64938e-06,   0.00303649,   0.00294875,
                 4.31291e-06,   0.00151793,  0.000559524,
                 4.97643e-06,  0.000888687,  8.04815e-05,
                 5.63996e-06,   0.00056464,  8.77547e-06,
                 6.30348e-06,  0.000375327,  7.25341e-07;

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
    data_ref <<  -1.99334e-05,  2.87803e-05,  2.29373e-07,
                 -1.78351e-05,  6.78954e-05,  2.77505e-06,
                 -1.57369e-05,  0.000165351,  2.54505e-05,
                 -1.36386e-05,  0.000425641,  0.000176937,
                 -1.15404e-05,   0.00118221,  0.000932475,
                 -9.44212e-06,   0.00346731,   0.00372523,
                 -7.34387e-06,   0.00981381,    0.0112815,
                 -5.24562e-06,       0.0238,    0.0258986,
                 -3.14737e-06,    0.0449805,    0.0450694,
                 -1.04912e-06,    0.0626136,    0.0594546,
                  1.04912e-06,    0.0626136,    0.0594546,
                  3.14737e-06,    0.0449805,    0.0450694,
                  5.24562e-06,       0.0238,    0.0258986,
                  7.34387e-06,   0.00981381,    0.0112815,
                  9.44212e-06,   0.00346731,   0.00372523,
                  1.15404e-05,   0.00118221,  0.000932475,
                  1.36386e-05,  0.000425641,  0.000176937,
                  1.57369e-05,  0.000165351,  2.54505e-05,
                  1.78351e-05,  6.78954e-05,  2.77505e-06,
                  1.99334e-05,  2.87803e-05,  2.29373e-07;

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
    data_ref << -6.30348e-05,  3.11225e-07,  7.25341e-08,
                -5.63996e-05,  2.11732e-06,  8.77547e-07,
                -4.97643e-05,  1.29795e-05,  8.04815e-06,
                -4.31291e-05,  6.96374e-05,  5.59524e-05,
                -3.64938e-05,   0.00031647,  0.000294875,
                -2.98586e-05,   0.00117923,   0.00117802,
                -2.32234e-05,   0.00349925,   0.00356752,
                -1.65881e-05,   0.00807486,   0.00818984,
                -9.95287e-06,    0.0142366,    0.0142522,
                -3.31762e-06,    0.0189571,    0.0188012,
                3.31762e-06 ,   0.0189571 ,   0.0188012 ,
                9.95287e-06 ,   0.0142366 ,   0.0142522 ,
                1.65881e-05 ,  0.00807486 ,  0.00818984 ,
                2.32234e-05 ,  0.00349925 ,  0.00356752 ,
                2.98586e-05 ,  0.00117923 ,  0.00117802 ,
                3.64938e-05 ,  0.00031647 , 0.000294875 ,
                4.31291e-05 , 6.96374e-05 , 5.59524e-05 ,
                4.97643e-05 , 1.29795e-05 , 8.04815e-06 ,
                5.63996e-05 , 2.11732e-06 , 8.77547e-07 ,
                6.30348e-05 , 3.11225e-07 , 7.25341e-08 ;

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
    data_ref << -0.000199334,  2.71915e-08,  2.29373e-08,
                -0.000178351,  3.08476e-07,  2.77505e-07,
                -0.000157369,  2.69788e-06,  2.54505e-06,
                -0.000136386,  1.81935e-05,  1.76937e-05,
                -0.000115404,  9.42087e-05,  9.32475e-05,
                -9.44212e-05,  0.000373119,  0.000372523,
                -7.34387e-05,   0.00112658,   0.00112815,
                -5.24562e-05,   0.00258636,   0.00258986,
                -3.14737e-05,   0.00450589,   0.00450694,
                -1.04912e-05,   0.00594933,   0.00594546,
                1.04912e-05 ,  0.00594933 ,  0.00594546 ,
                3.14737e-05 ,  0.00450589 ,  0.00450694 ,
                5.24562e-05 ,  0.00258636 ,  0.00258986 ,
                7.34387e-05 ,  0.00112658 ,  0.00112815 ,
                9.44212e-05 , 0.000373119 , 0.000372523 ,
                0.000115404 , 9.42087e-05 , 9.32475e-05 ,
                0.000136386 , 1.81935e-05 , 1.76937e-05 ,
                0.000157369 , 2.69788e-06 , 2.54505e-06 ,
                0.000178351 , 3.08476e-07 , 2.77505e-07 ,
                0.000199334 , 2.71915e-08 , 2.29373e-08 ;

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
    data_ref << 1e-09 ,  0.845488 ,   1.37368 ,
                1e-08 ,  0.229538 ,   1.17932 ,
                1e-07 , 0.0652923 ,   1.06081 ,
                1e-06 , 0.0196505 ,    1.0096 ,
                1e-05 , 0.0061597 ,   1.00077 ,
                0.0001, 0.00194599,   0.999812;
    
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
    data_ref << 1e-12, 2.03663e-18,   0.0128142,
                1e-11, 1.45549e-16,   0.0915775,
                1e-10, 6.70528e-15,    0.421887,
                1e-09, 1.21524e-13,    0.764613,
                1e-08, 1.53809e-12,    0.967743,
                1e-07, 1.58507e-11,    0.997303,
                1e-06, 1.58977e-10,     1.00026;

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
