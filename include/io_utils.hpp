// Copyright 2015-2022 The ALMA Project Developers
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

#pragma once

/// @file
///
/// Helper code to write calculation results into external files.

#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <ctime>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>

namespace alma {
/// Save a Matrix of double precision numbers as a csv file.
///
/// @param[in] filename - target file location
/// @param[in] data - matrix with data to save
/// @param[in] colsep - column separation character

inline void write_to_csv(const std::string filename,
                         const Eigen::Ref<const Eigen::MatrixXd>& data,
                         char colsep,
                         bool append = false,
                         std::string headermessage = "") {
    std::ofstream csvwriter;

    if (append) {
        csvwriter.open(filename, std::ofstream::out | std::ofstream::app);
    }
    else {
        csvwriter.open(filename, std::ofstream::out);
    }

    if (csvwriter.fail()) {
        std::cout << "************** ERROR IN write_to_csv **************"
                  << std::endl;
        std::cout << "Unable to write to target file " << filename << std::endl;
        std::cout
            << "Verify existence/permissions of the chosen target directory."
            << std::endl;

        // create timestamp
        auto t = std::time(nullptr);
        auto localtime = std::localtime(&t);
        std::size_t size = 0;
        std::vector<char> buffer;

        do {
            size += 64;
            buffer.reserve(size);
        } while (std::strftime(
                     buffer.data(), size, "%Y%m%d_%H%M%S", localtime) == 0u);
        std::string timestamp(buffer.data());

        // create override filename
        // the filename will have a random component to avoid conflicts if
        // these redirections happen more than once a second
        std::string filename_override = "CSV_DUMP_" + timestamp + "_%%%%%%.csv";
        filename_override =
            boost::filesystem::unique_path(filename_override).string();

        std::cout << "Data will be redirected to the current work directory"
                  << std::endl;
        std::cout << "and written to file " << filename_override << std::endl;
        std::cout << "***************************************************"
                  << std::endl;

        csvwriter.close();
        csvwriter.open(filename_override,
                       std::ofstream::out | std::ofstream::app);
        csvwriter << "*** The following data was originally destined for file "
                  << filename << "***" << std::endl;
    }

    csvwriter << headermessage;

    int Nrows = data.rows();
    int Ncols = data.cols();

    for (int nrow = 0; nrow < Nrows; nrow++) {
        for (int ncol = 0; ncol < Ncols; ncol++) {
            csvwriter << data(nrow, ncol);

            if (ncol == (Ncols - 1)) {
                csvwriter << std::endl;
            }

            else {
                csvwriter << colsep;
            }
        }
    }

    csvwriter.close();
}

/// Read a csv file into a Matrix.
///
/// @param[in] filename - sourcefile location
/// @param[in] colsep - column separation character
/// @param[out] data matrix

inline Eigen::MatrixXd read_from_csv(const std::string filename,
                                     char colsep,
                                     std::size_t skipheaderlines = 0) {
    std::ifstream csvreader;
    csvreader.open(filename);
    if (csvreader.fail()) {
        std::cout << "Error in read_from_csv:" << std::endl;
        std::cout << "Unable to open " << filename << std::endl;

        exit(1);
    }

    std::string linereader;
    std::vector<double> databuffer;
    double datavalue;
    char dump;

    // read header lines

    for (std::size_t nskip = 0; nskip < skipheaderlines; nskip++) {
        getline(csvreader, linereader, '\n');
    }

    // GRAB DATA

    // get first data line
    getline(csvreader, linereader, '\n');

    std::size_t Ncols =
        std::count(linereader.begin(), linereader.end(), colsep) + 1;

    while (csvreader.good()) {
        std::stringstream extractor(linereader);

        for (std::size_t ncol = 0; ncol < Ncols; ncol++) {
            extractor >> datavalue;
            databuffer.emplace_back(datavalue);
            extractor >> dump;
        }

        getline(csvreader, linereader, '\n');
    }

    csvreader.close();

    // PUT DATA IN MATRIX FORM

    int Nrows = databuffer.size() / Ncols;

    Eigen::MatrixXd result(Nrows, Ncols);
    result.fill(0.0);

    for (std::size_t idx = 0; idx < databuffer.size(); idx++) {
        std::size_t nrow = idx / Ncols;
        std::size_t ncol = idx % Ncols;
        result(nrow, ncol) = databuffer.at(idx);
    }

    return result;
}

/// Extract a value from an XML tree.

template <class C>
C parseXMLfield(boost::property_tree::ptree::value_type const& v,
                std::string field) {
    C result;
    std::string field_as_attribute = "<xmlattr>." + field;

    try {
        result = v.second.get<C>(field);
    }
    catch (boost::property_tree::ptree_bad_path& bad_path_error) {
        try {
            result = v.second.get<C>(field_as_attribute);
        }
        catch (boost::property_tree::ptree_bad_path& bad_path_error) {
            std::cout << "XML error:" << std::endl;
            std::cout << "Mandatory attribute \"" << field
                      << "\" is missing in element \"" << v.first << "\""
                      << std::endl;

            exit(1);
        }
    }

    return result;
}


/// Check if an XML field is present.

template <class C>
bool probeXMLfield(boost::property_tree::ptree::value_type const& v,
                   std::string field) {
    bool result = false;
    std::string field_as_attribute = "<xmlattr>." + field;

    try {
        v.second.get<C>(field);
        result = true;
    }
    catch (boost::property_tree::ptree_bad_path& bad_path_error) {
        try {
            v.second.get<C>(field_as_attribute);
            result = true;
        }
        catch (boost::property_tree::ptree_bad_path& bad_path_error) {
            result = false;
        }
    }

    return result;
}
} // namespace alma
