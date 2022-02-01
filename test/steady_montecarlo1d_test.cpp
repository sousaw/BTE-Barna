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
/// Verify that the steady_montecarlo1d executable works correctly.

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
#include <where_is_steady_montecarlo1d.hpp>

// Randomly-named directory to store the output of the program.
std::string targetdir;

class G_Environment : public ::testing::Environment {
public:
    virtual ~G_Environment() {
    }
    virtual void SetUp() {
        std::string path_prefix(boost::filesystem::path(steady_montecarlo1d)
                                    .parent_path()
                                    .string());

        // BUILD XML INPUT FILE
        std::string xmlfilename =
            (boost::filesystem::path(path_prefix) /
             boost::filesystem::path("test_steady_montecarlo1d.xml"))
                .string();
        std::ofstream xmlwriter(xmlfilename);

        targetdir = (boost::filesystem::path("test_results") /
                     boost::filesystem::unique_path(
                         "steady_montecarlo1d_%%%%-%%%%-%%%%-%%%%"))
                        .string();

        xmlwriter << "<materials>" << std::endl;
        xmlwriter << "  <H5repository root_directory=\"";
        auto repository_root = boost::filesystem::path(TEST_RESOURCE_DIR);
        xmlwriter << repository_root.string() << "\"/>" << std::endl;
        xmlwriter
            << "  <material label=\"Si\" directory=\"Si\" compound=\"Si\" "
               "gridA=\"12\" gridB=\"12\" gridC=\"12\"/>"
            << std::endl;
        xmlwriter
            << "  <material label=\"Ge\" directory=\"Ge\" compound=\"Ge\" "
               "gridA=\"12\" gridB=\"12\" gridC=\"12\"/>"
            << std::endl;
        xmlwriter << "</materials>" << std::endl;
        xmlwriter << "<layers>" << std::endl;
        xmlwriter << "  <layer label=\"top\" index=\"1\" material=\"Si\" "
                     "thickness=\"100\"/>"
                  << std::endl;
        xmlwriter << "  <layer label=\"middle\" index=\"2\" material=\"Ge\" "
                     "thickness=\"100\"/>"
                  << std::endl;
        xmlwriter << "  <layer label=\"bottom\" index=\"3\" material=\"Si\" "
                     "thickness=\"100\"/>"
                  << std::endl;
        xmlwriter << "</layers>" << std::endl;
        xmlwriter << "<simulation>" << std::endl;
        xmlwriter << "  <core deltaT=\"2.0\" particles=\"1e5\" bins=\"300\"/>"
                  << std::endl;
        xmlwriter << "  <transportAxis x=\"0\" y=\"0\" z=\"1\"/>" << std::endl;
        xmlwriter << "  <target directory=\"" << targetdir << "\"/>"
                  << std::endl;
        xmlwriter << "</simulation>" << std::endl;
        xmlwriter << "<spectralflux>" << std::endl;
        xmlwriter << "  <resolution frequencybins=\"200\"/>" << std::endl;
        xmlwriter << "  <locationrange start=\"50\" stop=\"250\" step=\"100\"/>"
                  << std::endl;
        xmlwriter << "</spectralflux>" << std::endl;
        xmlwriter.close();

        // RUN EXECUTABLE
        std::string command = (boost::filesystem::path(path_prefix) /
                               boost::filesystem::path("steady_montecarlo1d"))
                                  .string() +
                              " " + xmlfilename;
        ASSERT_EQ(std::system(command.c_str()), 0);
    }
    virtual void TearDown() {
    }
};

TEST(filecreation_case, steady_montecarlo1d_test) {
    EXPECT_TRUE(boost::filesystem::exists(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("basicproperties_300K.txt"))
            .string()));
    EXPECT_TRUE(boost::filesystem::exists(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("temperature_300K.csv"))
            .string()));
    EXPECT_TRUE(boost::filesystem::exists(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("spectralflux_surface_1_300K.csv"))
            .string()));
    EXPECT_TRUE(boost::filesystem::exists(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("spectralflux_surface_2_300K.csv"))
            .string()));
    EXPECT_TRUE(boost::filesystem::exists(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("spectralflux_surface_3_300K.csv"))
            .string()));
}

TEST(basicproperties_case, steady_montecarlo1d_test) {
    std::ifstream filereader;
    std::string line;
    std::istringstream lineprocessor;
    std::string dump;
    double value;

    filereader.open((boost::filesystem::path(targetdir) /
                     boost::filesystem::path("basicproperties_300K.txt"))
                        .string());

    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("LAYER 1 Si (100 nm)") != std::string::npos);
    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("LAYER 2 Ge (100 nm)") != std::string::npos);
    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("LAYER 3 Si (100 nm)") != std::string::npos);
    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("T_TOP 301 K") != std::string::npos);
    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("T_BOTTOM 299 K") != std::string::npos);
    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("T_REF 300 K") != std::string::npos);
    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("N_PARTICLES 100000") != std::string::npos);

    // verify total heat flux
    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("MW/m^2") != std::string::npos);
    lineprocessor.str(line);
    lineprocessor >> dump;
    lineprocessor >> value;
    EXPECT_TRUE(alma::almost_equal(value, 102, 5.0, 5e-2));

    // verify heat flux tolerance
    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("MW/m^2") != std::string::npos ||
                line.find("kW/m^2") != std::string::npos);
    lineprocessor.str(line);
    lineprocessor >> dump;
    lineprocessor >> value;
    if (line.find("MW/m^2") != std::string::npos) {
        EXPECT_TRUE(alma::almost_equal(value, 2.0, 2.0, 100e-2));
    }
    if (line.find("kW/m^2") != std::string::npos) {
        EXPECT_TRUE(alma::almost_equal(value, 999, 999, 100e-2));
    }

    // verify effective conductivity
    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("W/m-K") != std::string::npos);
    lineprocessor.str(line);
    lineprocessor >> dump;
    lineprocessor >> value;
    EXPECT_TRUE(alma::almost_equal(value, 15.4, 0.8, 5e-2));

    // verify effective conductivity tolerance
    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("W/m-K") != std::string::npos);
    lineprocessor.str(line);
    lineprocessor >> dump;
    lineprocessor >> value;
    EXPECT_TRUE(alma::almost_equal(value, 0.15, 0.15, 100e-2));

    // verify effective resistivity
    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("nK-m^2/W") != std::string::npos);
    lineprocessor.str(line);
    lineprocessor >> dump;
    lineprocessor >> value;
    EXPECT_TRUE(alma::almost_equal(value, 19.5, 1.0, 5e-2));

    // verify effective conductance
    std::getline(filereader, line, '\n');
    EXPECT_TRUE(line.find("MW/K-m^2") != std::string::npos);
    lineprocessor.str(line);
    lineprocessor >> dump;
    lineprocessor >> value;
    EXPECT_TRUE(alma::almost_equal(value, 51.3, 2.5, 5e-2));
}

TEST(temperatureprofile_case, steady_montecarlo1d_test) {
    Eigen::MatrixXd data =
        alma::read_from_csv((boost::filesystem::path(targetdir) /
                             boost::filesystem::path("temperature_300K.csv"))
                                .string(),
                            ',',
                            0);
    Eigen::MatrixXd data_ref(300, 2);
    data_ref << 0.5, 300.965, 1.5, 300.962, 2.5, 300.962, 3.5, 300.956, 4.5,
        300.953, 5.5, 300.953, 6.5, 300.954, 7.5, 300.947, 8.5, 300.945, 9.5,
        300.945, 10.5, 300.936, 11.5, 300.939, 12.5, 300.936, 13.5, 300.936,
        14.5, 300.936, 15.5, 300.939, 16.5, 300.933, 17.5, 300.927, 18.5,
        300.93, 19.5, 300.924, 20.5, 300.924, 21.5, 300.916, 22.5, 300.917,
        23.5, 300.922, 24.5, 300.921, 25.5, 300.919, 26.5, 300.917, 27.5,
        300.915, 28.5, 300.91, 29.5, 300.911, 30.5, 300.906, 31.5, 300.901,
        32.5, 300.905, 33.5, 300.9, 34.5, 300.896, 35.5, 300.9, 36.5, 300.899,
        37.5, 300.901, 38.5, 300.894, 39.5, 300.898, 40.5, 300.896, 41.5,
        300.891, 42.5, 300.886, 43.5, 300.887, 44.5, 300.882, 45.5, 300.884,
        46.5, 300.883, 47.5, 300.881, 48.5, 300.882, 49.5, 300.877, 50.5,
        300.876, 51.5, 300.879, 52.5, 300.874, 53.5, 300.873, 54.5, 300.876,
        55.5, 300.873, 56.5, 300.878, 57.5, 300.873, 58.5, 300.87, 59.5,
        300.866, 60.5, 300.871, 61.5, 300.87, 62.5, 300.868, 63.5, 300.863,
        64.5, 300.865, 65.5, 300.857, 66.5, 300.853, 67.5, 300.858, 68.5,
        300.855, 69.5, 300.858, 70.5, 300.855, 71.5, 300.85, 72.5, 300.848,
        73.5, 300.843, 74.5, 300.846, 75.5, 300.85, 76.5, 300.842, 77.5,
        300.843, 78.5, 300.836, 79.5, 300.84, 80.5, 300.835, 81.5, 300.836,
        82.5, 300.837, 83.5, 300.838, 84.5, 300.837, 85.5, 300.829, 86.5,
        300.831, 87.5, 300.827, 88.5, 300.834, 89.5, 300.833, 90.5, 300.826,
        91.5, 300.822, 92.5, 300.818, 93.5, 300.822, 94.5, 300.824, 95.5,
        300.822, 96.5, 300.822, 97.5, 300.822, 98.5, 300.812, 99.5, 300.812,
        100.5, 300.366, 101.5, 300.344, 102.5, 300.327, 103.5, 300.311, 104.5,
        300.298, 105.5, 300.292, 106.5, 300.283, 107.5, 300.273, 108.5, 300.263,
        109.5, 300.253, 110.5, 300.253, 111.5, 300.241, 112.5, 300.228, 113.5,
        300.224, 114.5, 300.218, 115.5, 300.208, 116.5, 300.202, 117.5, 300.187,
        118.5, 300.187, 119.5, 300.186, 120.5, 300.174, 121.5, 300.171, 122.5,
        300.164, 123.5, 300.158, 124.5, 300.158, 125.5, 300.151, 126.5, 300.145,
        127.5, 300.137, 128.5, 300.13, 129.5, 300.118, 130.5, 300.119, 131.5,
        300.114, 132.5, 300.109, 133.5, 300.105, 134.5, 300.101, 135.5, 300.099,
        136.5, 300.086, 137.5, 300.079, 138.5, 300.077, 139.5, 300.072, 140.5,
        300.068, 141.5, 300.071, 142.5, 300.062, 143.5, 300.058, 144.5, 300.053,
        145.5, 300.044, 146.5, 300.047, 147.5, 300.033, 148.5, 300.037, 149.5,
        300.022, 150.5, 300.018, 151.5, 300.01, 152.5, 300.005, 153.5, 299.996,
        154.5, 299.993, 155.5, 299.987, 156.5, 299.985, 157.5, 299.981, 158.5,
        299.971, 159.5, 299.973, 160.5, 299.964, 161.5, 299.958, 162.5, 299.954,
        163.5, 299.948, 164.5, 299.946, 165.5, 299.941, 166.5, 299.931, 167.5,
        299.926, 168.5, 299.929, 169.5, 299.918, 170.5, 299.918, 171.5, 299.908,
        172.5, 299.901, 173.5, 299.891, 174.5, 299.892, 175.5, 299.881, 176.5,
        299.878, 177.5, 299.869, 178.5, 299.868, 179.5, 299.864, 180.5, 299.856,
        181.5, 299.851, 182.5, 299.842, 183.5, 299.833, 184.5, 299.834, 185.5,
        299.821, 186.5, 299.815, 187.5, 299.807, 188.5, 299.804, 189.5, 299.79,
        190.5, 299.786, 191.5, 299.774, 192.5, 299.768, 193.5, 299.761, 194.5,
        299.754, 195.5, 299.739, 196.5, 299.73, 197.5, 299.713, 198.5, 299.696,
        199.5, 299.672, 200.5, 299.194, 201.5, 299.192, 202.5, 299.192, 203.5,
        299.192, 204.5, 299.188, 205.5, 299.186, 206.5, 299.186, 207.5, 299.182,
        208.5, 299.182, 209.5, 299.174, 210.5, 299.172, 211.5, 299.171, 212.5,
        299.171, 213.5, 299.175, 214.5, 299.167, 215.5, 299.168, 216.5, 299.164,
        217.5, 299.166, 218.5, 299.162, 219.5, 299.163, 220.5, 299.158, 221.5,
        299.164, 222.5, 299.158, 223.5, 299.156, 224.5, 299.161, 225.5, 299.155,
        226.5, 299.16, 227.5, 299.156, 228.5, 299.155, 229.5, 299.147, 230.5,
        299.149, 231.5, 299.155, 232.5, 299.148, 233.5, 299.145, 234.5, 299.146,
        235.5, 299.145, 236.5, 299.138, 237.5, 299.14, 238.5, 299.138, 239.5,
        299.14, 240.5, 299.135, 241.5, 299.129, 242.5, 299.134, 243.5, 299.134,
        244.5, 299.126, 245.5, 299.121, 246.5, 299.123, 247.5, 299.122, 248.5,
        299.123, 249.5, 299.125, 250.5, 299.115, 251.5, 299.12, 252.5, 299.117,
        253.5, 299.113, 254.5, 299.111, 255.5, 299.111, 256.5, 299.111, 257.5,
        299.114, 258.5, 299.113, 259.5, 299.11, 260.5, 299.106, 261.5, 299.11,
        262.5, 299.106, 263.5, 299.1, 264.5, 299.098, 265.5, 299.096, 266.5,
        299.096, 267.5, 299.089, 268.5, 299.09, 269.5, 299.086, 270.5, 299.089,
        271.5, 299.088, 272.5, 299.083, 273.5, 299.082, 274.5, 299.082, 275.5,
        299.083, 276.5, 299.08, 277.5, 299.082, 278.5, 299.077, 279.5, 299.076,
        280.5, 299.079, 281.5, 299.071, 282.5, 299.074, 283.5, 299.072, 284.5,
        299.065, 285.5, 299.064, 286.5, 299.064, 287.5, 299.065, 288.5, 299.061,
        289.5, 299.057, 290.5, 299.055, 291.5, 299.053, 292.5, 299.052, 293.5,
        299.046, 294.5, 299.046, 295.5, 299.041, 296.5, 299.04, 297.5, 299.043,
        298.5, 299.037, 299.5, 299.033;

    Eigen::VectorXd reldev_space =
        data.col(0).array() / data_ref.col(0).array() - 1.0;
    Eigen::VectorXd reldev_temp =
        (data.col(1).array() - data_ref.col(1).array()).abs() / (301.0 - 299.0);

    for (int n = 0; n < 300; n++) {
        EXPECT_TRUE(alma::almost_equal(reldev_space(n), 0.0, 1e-8, 1e-8));
        EXPECT_TRUE(alma::almost_equal(reldev_temp(n), 0.0, 5e-2, 5e-2));
    }
}

TEST(spectralflux1_case, steady_montecarlo1d_test) {
    Eigen::MatrixXd data = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("spectralflux_surface_1_300K.csv"))
            .string(),
        ',',
        1);

    // check that the integrated curve matches the overall heat flux

    Eigen::VectorXd integrand = 0.5 * (data.col(1).segment(0, 199).array() +
                                       data.col(1).segment(1, 199).array());
    Eigen::VectorXd delta_omega = data.col(0).segment(1, 199).array() -
                                  data.col(0).segment(0, 199).array();

    double integral = (integrand.array() * delta_omega.array()).sum();

    EXPECT_TRUE(alma::almost_equal(integral, 102.6e6, 5.2e6, 5e-2));
}

TEST(spectralflux2_case, steady_montecarlo1d_test) {
    Eigen::MatrixXd data = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("spectralflux_surface_2_300K.csv"))
            .string(),
        ',',
        1);

    // check that the integrated curve matches the overall heat flux

    Eigen::VectorXd integrand = 0.5 * (data.col(1).segment(0, 199).array() +
                                       data.col(1).segment(1, 199).array());
    Eigen::VectorXd delta_omega = data.col(0).segment(1, 199).array() -
                                  data.col(0).segment(0, 199).array();

    double integral = (integrand.array() * delta_omega.array()).sum();

    EXPECT_TRUE(alma::almost_equal(integral, 102.6e6, 5.2e6, 5e-2));

    // check that there is no heat flux at frequencies that are out-of-reach for
    // Ge

    for (int n = 0; n < 200; n++) {
        if (data(n, 0) > 61.0) {
            EXPECT_TRUE(std::abs(data(n, 1)) < 1e-6);
        }
    }
}

TEST(spectralflux3_case, steady_montecarlo1d_test) {
    Eigen::MatrixXd data = alma::read_from_csv(
        (boost::filesystem::path(targetdir) /
         boost::filesystem::path("spectralflux_surface_3_300K.csv"))
            .string(),
        ',',
        1);

    // check that the integrated curve matches the overall heat flux

    Eigen::VectorXd integrand = 0.5 * (data.col(1).segment(0, 199).array() +
                                       data.col(1).segment(1, 199).array());
    Eigen::VectorXd delta_omega = data.col(0).segment(1, 199).array() -
                                  data.col(0).segment(0, 199).array();

    double integral = (integrand.array() * delta_omega.array()).sum();

    EXPECT_TRUE(alma::almost_equal(integral, 102.6e6, 5.2e6, 5e-2));
}

int main(int argc, char** argv) {
    boost::mpi::environment env;

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new G_Environment);
    return RUN_ALL_TESTS();
}
