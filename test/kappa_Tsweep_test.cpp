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
/// Verify that the kappa_Tsweep executable works correctly.

#include <cstdlib>
#include <iostream>
#include <functional>
#include <fstream>
#include <string>
#include <boost/filesystem.hpp>
#include "gtest/gtest.h"
#include <io_utils.hpp>
#include <Eigen/Dense>
#include <cmakevars.hpp>
#include <utilities.hpp>
#include <where_is_kappa_Tsweep.hpp>

// Randomly-named directory to store the output of the program.
std::string targetdir;

class G_Environment : public ::testing::Environment {
public:
    virtual ~G_Environment() {
    }
    virtual void SetUp() {
        std::string path_prefix(
            boost::filesystem::path(kappa_Tsweep).parent_path().string());

        // BUILD XML INPUT FILE
        std::string xmlfilename =
            (boost::filesystem::path(path_prefix) /
             boost::filesystem::path("test_kappa_Tsweep.xml"))
                .string();
        std::ofstream xmlwriter(xmlfilename);

        targetdir =
            (boost::filesystem::path("test_results") /
             boost::filesystem::unique_path("kappa_Tsweep_%%%%-%%%%-%%%%-%%%%"))
                .string();

        xmlwriter << "<Tsweep>" << std::endl;
        xmlwriter << "  <H5repository root_directory=\"";
        auto repository_root = boost::filesystem::path(TEST_RESOURCE_DIR);
        xmlwriter << repository_root.string() << "\"/>" << std::endl;
        xmlwriter << "  <compound directory=\"Si\" base=\"Si\" gridA=\"12\" "
                     "gridB=\"12\" gridC=\"12\"/>"
                  << std::endl;
        xmlwriter
            << "  <sweep type=\"lin\" start=\"300\" stop=\"500\" points=\"3\"/>"
            << std::endl;
        xmlwriter << "  <transportAxis x=\"0\" y=\"0\" z=\"1\"/>" << std::endl;
        xmlwriter << "  <fullBTE iterative=\"true\"/>" << std::endl;
        xmlwriter << "  <outputHeatCapacity/>" << std::endl;
        xmlwriter << "  <target directory=\"" << targetdir
                  << "\" file=\"AUTO\"/>" << std::endl;
        xmlwriter << "</Tsweep>" << std::endl;
        xmlwriter.close();

        // RUN EXECUTABLE
        std::string command = (boost::filesystem::path(path_prefix) /
                               boost::filesystem::path("kappa_Tsweep"))
                                  .string() +
                              " " + xmlfilename;
        ASSERT_EQ(std::system(command.c_str()), 0);
    }
    virtual void TearDown() {
    }
};

TEST(kappa_Tsweep_case, kappa_Tsweep_test) {
    Eigen::MatrixXd data = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("Si_12_12_12_0,0,1_300_500.Tsweep"))
            .string(),
        ',',
        1);

    std::cout.precision(10);
    Eigen::IOFormat fmt(Eigen::StreamPrecision,
                        Eigen::DontAlignCols,
                        ", ",
                        ", ",
                        "",
                        "",
                        "",
                        ";");
    std::cout << "data =" << std::endl << data.format(fmt) << std::endl;

    Eigen::MatrixXd data_ref(3, 4);
    data_ref << 300, 135.282, 141.295, 1625830, 400, 97.9002, 102.868, 1782430,
        500, 77.2684, 81.482, 1864400;

    for (int ncol = 0; ncol < 4; ncol++) {
        double residue = (data.col(ncol) - data_ref.col(ncol)).norm() /
                         data_ref.col(ncol).norm();
        EXPECT_TRUE(alma::almost_equal(residue, 0.0, 1e-3, 1e-3));
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new G_Environment);
    return RUN_ALL_TESTS();
}
