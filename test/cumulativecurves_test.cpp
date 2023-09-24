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
/// Verify that the cumulativecurves executable works correctly.

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
#include <where_is_cumulativecurves.hpp>

// Randomly-named directory to store the output of the program.
std::string targetdir;

class G_Environment : public ::testing::Environment {
public:
    virtual ~G_Environment() {
    }
    virtual void SetUp() {
        std::string path_prefix(
            boost::filesystem::path(cumulativecurves).parent_path().string());

        // BUILD XML INPUT FILE
        std::string xmlfilename =
            (boost::filesystem::path(path_prefix) /
             boost::filesystem::path("test_cumulativecurves.xml"))
                .string();
        std::ofstream xmlwriter(xmlfilename);

        targetdir = (boost::filesystem::path("test_results") /
                     boost::filesystem::unique_path(
                         "cumulativecurves_%%%%-%%%%-%%%%-%%%%"))
                        .string();

        xmlwriter << "<cumulativecurves>" << std::endl;
        xmlwriter << "  <H5repository root_directory=\"";
        auto repository_root = boost::filesystem::path(TEST_RESOURCE_DIR);
        xmlwriter << repository_root.string() << "\"/>" << std::endl;
        xmlwriter << "  <compound directory=\"Si\" base=\"Si\" gridA=\"12\" "
                     "gridB=\"12\" gridC=\"12\"/>"
                  << std::endl;
        xmlwriter << "  <transportAxis x=\"0\" y=\"0\" z=\"1\"/>" << std::endl;
        xmlwriter << "  <output conductivity=\"true\" l2=\"false\" capacity=\"true\"/>"
                  << std::endl;
        xmlwriter << "  <resolveby MFP=\"true\" projMFP=\"true\" RT=\"true\" "
                     "freq=\"true\" angfreq=\"true\" energy=\"true\"/>"
                  << std::endl;
        xmlwriter << "  <optionalsettings curvepoints=\"10\"/>" << std::endl;
        xmlwriter << "  <target directory=\"" << targetdir << "\"/>"
                  << std::endl;
        xmlwriter << "</cumulativecurves>" << std::endl;
        xmlwriter.close();


        // RUN EXECUTABLE
        std::string command = (boost::filesystem::path(path_prefix) /
                               boost::filesystem::path("cumulativecurves"))
                                  .string() +
                              " " + xmlfilename;
        ASSERT_EQ(std::system(command.c_str()), 0);
    }
    virtual void TearDown() {
    }
};

TEST(Ccumul_angfreq_case, cumulativecurves_test) {
    // 1) Ccumul(angfreq)

    Eigen::MatrixXd data1 = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.Ccumul_angfreq"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref1(10, 3);
    data_ref1 << 0, 3705.23, 0.00227897, 11.6237, 64843.5, 0.0398832, 23.2473,
        437952, 0.26937, 34.871, 673624, 0.414325, 46.4946, 734148, 0.451552,
        58.1183, 871010, 0.535731, 69.7419, 990916, 0.609482, 81.3656, 1125730,
        0.6924, 92.9892, 1618070, 0.995221, 104.613, 1625830, 1;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data1.col(ncol) - data_ref1.col(ncol)).norm() /
                         data_ref1.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(Ccumul_energy_case, cumulativecurves_test) {
    // 2) Ccumul(energy)

    Eigen::MatrixXd data2 = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.Ccumul_energy"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref2(10, 3);
    data_ref2 << 0, 3705.23, 0.00227897, 7.65083, 64843.5, 0.0398832, 15.3017,
        437952, 0.26937, 22.9525, 673624, 0.414325, 30.6033, 734148, 0.451552,
        38.2541, 871010, 0.535731, 45.905, 990916, 0.609482, 53.5558, 1125730,
        0.6924, 61.2066, 1618070, 0.995221, 68.8574, 1625830, 1;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data2.col(ncol) - data_ref2.col(ncol)).norm() /
                         data_ref2.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(Ccumul_freq_case, cumulativecurves_test) {
    // 3) Ccumul(freq)

    Eigen::MatrixXd data3 = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.Ccumul_freq"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref3(10, 3);
    data_ref3 << 0, 3705.23, 0.00227897, 1.84996, 64843.5, 0.0398832, 3.69992,
        437952, 0.26937, 5.54988, 673624, 0.414325, 7.39985, 734148, 0.451552,
        9.24981, 871010, 0.535731, 11.0998, 990916, 0.609482, 12.9497, 1125730,
        0.6924, 14.7997, 1618070, 0.995221, 16.6497, 1625830, 1;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data3.col(ncol) - data_ref3.col(ncol)).norm() /
                         data_ref3.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(Ccumul_MFP_case, cumulativecurves_test) {
    // 4) Ccumul(MFP)

    Eigen::MatrixXd data4 = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.Ccumul_MFP"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref4(10, 3);
    data_ref4 << 4.27462e-12  ,     13135 ,  0.0080789,
                 2.14295e-11  ,     13135 ,  0.0080789,
                 1.0743e-10   ,  28725.2  ,  0.017668 ,
                 5.38569e-10  ,    104736 ,    0.06442,
                 2.69995e-09  ,    561988 ,   0.345661,
                 1.35354e-08  ,    817803 ,   0.503005,
                 6.78554e-08  ,   1494750 ,   0.919373,
                 3.40172e-07  ,   1611490 ,   0.991174,
                 1.70535e-06  ,   1621560 ,    0.99737,
                 8.54925e-06  ,   1625830 ,          1;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data4.col(ncol) - data_ref4.col(ncol)).norm() /
                         data_ref4.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(Ccumul_projMFP_case, cumulativecurves_test) {
    // 5) Ccumul(projMFP)

    Eigen::MatrixXd data5 = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.Ccumul_projMFP"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref5(10, 3);
    data_ref5 << 2.46796e-12   ,   158354 ,  0.0973983,
                 1.23723e-11   ,   172406 ,   0.106042,
                 6.20249e-11   ,   206772 ,   0.127179,
                 3.10943e-10   ,   363141 ,   0.223356,
                 1.55882e-09   ,   708051 ,     0.4355,
                 7.81465e-09   ,  1004240 ,   0.617677,
                 3.91764e-08   ,  1511990 ,   0.929976,
                 1.96399e-07   ,  1608830 ,   0.989541,
                 9.84584e-07   ,  1620780 ,   0.996889,
                 4.93591e-06   ,  1625830 ,          1;
                 
    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data5.col(ncol) - data_ref5.col(ncol)).norm() /
                         data_ref5.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(Ccumul_RT_case, cumulativecurves_test) {
    // 6) Ccumul(RT)

    Eigen::MatrixXd data6 = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.Ccumul_RT"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref6(10, 3);
    data_ref6 << 7.57539e-16  ,   586.741, 0.000360886,
                 3.79769e-15  ,   586.741, 0.000360886,
                 1.90386e-14  ,   586.741, 0.000360886,
                 9.54439e-14  ,   586.741, 0.000360886,
                 4.78479e-13  ,   586.741, 0.000360886,
                 2.39871e-12  ,    688673,    0.423581,
                 1.20252e-11  ,   1196320,     0.73582,
                 6.02846e-11  ,   1596080,    0.981699,
                 3.02218e-10  ,   1620390,    0.996649,
                 1.51508e-09  ,   1625830,           1;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data6.col(ncol) - data_ref6.col(ncol)).norm() /
                         data_ref6.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(kappacumul_angfreq_case, cumulativecurves_test) {
    // 7) kappacumul(angfreq)

    Eigen::MatrixXd data7 = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.kappacumul_angfreq"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref7(10, 3);
    data_ref7 << 0        ,    16.5638, 0.128202, 
                 11.6237  ,    58.3036, 0.451261,
                 23.2473  ,    90.4127,  0.69978,
                  34.871  ,    113.657, 0.879689,
                 46.4946  ,    119.127, 0.922027,
                 58.1183  ,    120.921, 0.935911,
                 69.7419  ,    123.637, 0.956933,
                 81.3656  ,    128.501,  0.99458,
                 92.9892  ,    129.198, 0.999972,
                 104.613  ,    129.202,        1;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data7.col(ncol) - data_ref7.col(ncol)).norm() /
                         data_ref7.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(kappacumul_energy_case, cumulativecurves_test) {
    // 8) kappacumul(energy)

    Eigen::MatrixXd data8 = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.kappacumul_energy"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref8(10, 3);
    data_ref8 <<        0      , 16.5638,  0.128202,
                        7.65083, 58.3036,  0.451261,
                        15.3017, 90.4127,   0.69978,
                        22.9525, 113.657,  0.879689,
                        30.6033, 119.127,  0.922027,
                        38.2541, 120.921,  0.935911,
                         45.905, 123.637,  0.956933,
                        53.5558, 128.501,   0.99458,
                        61.2066, 129.198,  0.999972,
                        68.8574, 129.202,         1;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data8.col(ncol) - data_ref8.col(ncol)).norm() /
                         data_ref8.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(kappacumul_freq_case, cumulativecurves_test) {
    // 9) kappacumul(freq)

    Eigen::MatrixXd data9 = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.kappacumul_freq"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref9(10, 3);
    data_ref9 <<        0, 16.5638,  0.128202,
                  1.84996, 58.3036,  0.451261,
                  3.69992, 90.4127,   0.69978,
                  5.54988, 113.657,  0.879689,
                  7.39985, 119.127,  0.922027,
                  9.24981, 120.921,  0.935911,
                  11.0998, 123.637,  0.956933,
                  12.9497, 128.501,   0.99458,
                  14.7997, 129.198,  0.999972,
                  16.6497, 129.202,         1;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data9.col(ncol) - data_ref9.col(ncol)).norm() /
                         data_ref9.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(kappacumul_MFP_case, cumulativecurves_test) {
    // 10) kappacumul(MFP)

    Eigen::MatrixXd data10 = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.kappacumul_MFP"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref10(10, 3);
    data_ref10 <<  4.27462e-12, 4.58128e-64,  3.54584e-66,
                   2.14295e-11, 4.58128e-64,  3.54584e-66,
                    1.0743e-10, 0.000112176,  8.68222e-07,
                   5.38569e-10,   0.0120951,  9.36139e-05,
                   2.69995e-09,    0.717419,   0.00555272,
                   1.35354e-08,     4.46297,    0.0345427,
                   6.78554e-08,     48.4065,     0.374659,
                   3.40172e-07,     85.0899,     0.658582,
                   1.70535e-06,     105.265,     0.814732,
                   8.54925e-06,     129.202,            1;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data10.col(ncol) - data_ref10.col(ncol)).norm() /
                         data_ref10.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(kappacumul_projMFP_case, cumulativecurves_test) {
    // 11) kappacumul(projMFP)

    Eigen::MatrixXd data11 = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.kappacumul_projMFP"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref11(10, 3);
    data_ref11 <<  2.46796e-12, 1.33736e-09,  1.0351e-11,
                   1.23723e-11, 7.20638e-06, 5.57763e-08,
                   6.20249e-11, 0.000261831, 2.02653e-06,
                   3.10943e-10,   0.0190494, 0.000147439,
                   1.55882e-09,     0.49274,  0.00381373,
                   7.81465e-09,      4.3733,   0.0338487,
                   3.91764e-08,     40.2352,    0.311414,
                   1.96399e-07,     76.6948,    0.593606,
                   9.84584e-07,     99.3347,    0.768836,
                   4.93591e-06,     129.202,           1;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data11.col(ncol) - data_ref11.col(ncol)).norm() /
                         data_ref11.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

TEST(kappacumul_RT_case, cumulativecurves_test) {
    // 12) kappacumul(RT)

    Eigen::MatrixXd data12 = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300K.kappacumul_RT"))
            .string(),
        ',',
        1);

    Eigen::MatrixXd data_ref12(10, 3);
    data_ref12 << 7.57539e-16,           0,           0,
                  3.79769e-15,           0,           0,
                  1.90386e-14,           0,           0,
                  9.54439e-14,           0,           0,
                  4.78479e-13,           0,           0,
                  2.39871e-12,     3.22021,   0.0249239,
                  1.20252e-11,     29.7475,    0.230241,
                  6.02846e-11,     77.2637,    0.598009,
                  3.02218e-10,     106.708,    0.825902,
                  1.51508e-09,     129.202,           1;

    for (int ncol = 0; ncol < 3; ncol++) {
        double residue = (data12.col(ncol) - data_ref12.col(ncol)).norm() /
                         data_ref12.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-6, 1e-6));
    }
}

int main(int argc, char** argv) {
    boost::mpi::environment env;

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new G_Environment);
    return RUN_ALL_TESTS();
}
