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
        xmlwriter << "  <output conductivity=\"true\" capacity=\"true\"/>"
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
    data_ref4 << 4.5545e-12, 13135, 0.0080789, 2.28326e-11, 13135, 0.0080789,
        1.14464e-10, 28853, 0.0177466, 5.73831e-10, 113115, 0.0695734,
        2.87673e-09, 579996, 0.356737, 1.44216e-08, 832410, 0.511989,
        7.22982e-08, 1506910, 0.926855, 3.62445e-07, 1612650, 0.991888,
        1.81701e-06, 1621560, 0.99737, 9.109e-06, 1625830, 1;

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
    data_ref5 << 2.84801e-12, 158354, 0.0973983, 1.42776e-11, 174495, 0.107326,
        7.15766e-11, 216239, 0.133002, 3.58827e-10, 389066, 0.239302,
        1.79887e-09, 734011, 0.451467, 9.01808e-09, 1038850, 0.638966,
        4.52094e-08, 1535550, 0.944467, 2.26643e-07, 1613460, 0.992388,
        1.13621e-06, 1621940, 0.997606, 5.69603e-06, 1625830, 1;

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
    data_ref6 << 7.71133e-16, 586.741, 0.000360886, 3.86584e-15, 586.741,
        0.000360886, 1.93802e-14, 586.741, 0.000360886, 9.71567e-14, 586.741,
        0.000360886, 4.87065e-13, 3675.69, 0.0022608, 2.44175e-12, 681410,
        0.419114, 1.2241e-11, 1228180, 0.755416, 6.13664e-11, 1600710, 0.984548,
        3.07641e-10, 1620390, 0.996649, 1.54227e-09, 1625830, 1;

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
    data_ref7 << 0, 16.1218, 0.119171, 11.6237, 63.2654, 0.467655, 23.2473,
        95.7373, 0.707685, 34.871, 119.258, 0.881553, 46.4946, 124.879,
        0.923102, 58.1183, 126.626, 0.936014, 69.7419, 129.457, 0.956941,
        81.3656, 134.578, 0.994791, 92.9892, 135.279, 0.999974, 104.613,
        135.282, 1;

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
    data_ref8 << 0, 16.1218, 0.119171, 7.65083, 63.2654, 0.467655, 15.3017,
        95.7373, 0.707685, 22.9525, 119.258, 0.881553, 30.6033, 124.879,
        0.923102, 38.2541, 126.626, 0.936014, 45.905, 129.457, 0.956941,
        53.5558, 134.578, 0.994791, 61.2066, 135.279, 0.999974, 68.8574,
        135.282, 1;

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
    data_ref9 << 0, 16.1218, 0.119171, 1.84996, 63.2654, 0.467655, 3.69992,
        95.7373, 0.707685, 5.54988, 119.258, 0.881553, 7.39985, 124.879,
        0.923102, 9.24981, 126.626, 0.936014, 11.0998, 129.457, 0.956941,
        12.9497, 134.578, 0.994791, 14.7997, 135.279, 0.999974, 16.6497,
        135.282, 1;

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
    data_ref10 << 4.5545e-12, 1.55215e-64, 1.14734e-66, 2.28326e-11,
        1.55215e-64, 1.14734e-66, 1.14464e-10, 0.00011882, 8.78312e-07,
        5.73831e-10, 0.0149386, 0.000110426, 2.87673e-09, 0.810197, 0.00598893,
        1.44216e-08, 4.91075, 0.0363, 7.22982e-08, 51.715, 0.382275,
        3.62445e-07, 86.5084, 0.639466, 1.81701e-06, 110.245, 0.814925,
        9.109e-06, 135.282, 1;

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
    data_ref11 << 2.84801e-12, 1.34603e-09, 9.9498e-12, 1.42776e-11,
        9.56298e-06, 7.06891e-08, 7.15766e-11, 0.000438819, 3.24373e-06,
        3.58827e-10, 0.0262961, 0.00019438, 1.79887e-09, 0.599477, 0.00443131,
        9.01808e-09, 5.42483, 0.0401001, 4.52094e-08, 45.6114, 0.337157,
        2.26643e-07, 80.8497, 0.597637, 1.13621e-06, 105.037, 0.776426,
        5.69603e-06, 135.282, 1;

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
    data_ref12 << 7.71133e-16, 0, 0, 3.86584e-15, 0, 0, 1.93802e-14, 0, 0,
        9.71567e-14, 0, 0, 4.87065e-13, 0.00126045, 9.31718e-06, 2.44175e-12,
        3.1146, 0.023023, 1.2241e-11, 32.6506, 0.241351, 6.13664e-11, 81.2035,
        0.600252, 3.07641e-10, 113.3, 0.837505, 1.54227e-09, 135.282, 1;

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