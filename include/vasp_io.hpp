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
/// Routines to load files coming from the VASP + Phonopy ecosystem.

#include <Eigen/Dense>
#include <structures.hpp>

namespace alma {
/// Build a Crystal_structure object from a VASP POSCAR file.
///
/// @param[in] filename - path to the POSCAR file
/// @return a Crystal_structure object with the structure contained
/// in the file
std::unique_ptr<Crystal_structure> load_POSCAR(const char* filename);

/// Create a Harmonic_ifcs object from the information stored in a
/// Phonopy FORCE_CONSTANTS file.
///
/// @param[in] filename - path to the FORCE_CONSTANTS file
/// @param[in] cell - description of the unit cell
/// @param[in] na - number of unit cells along the first axis
/// @param[in] nb - number of unit cells along the second axis
/// @param[in] nc - number of unit cells along the third axis
/// @return a Harmonic_ifcs object with the structure contained in
/// the file.
std::unique_ptr<Harmonic_ifcs> load_FORCE_CONSTANTS(
    const char* filename,
    const Crystal_structure& cell,
    const int na,
    const int nb,
    const int nc);

/// Load the data in a FORCE_CONSTANTS file and return it as a
/// raw matrix.
///
/// @param[in] filename - path to the FORCE_CONSTANTS file
/// @return a matrix with the force constants in the file. Each row
/// or column corresponds to a degree of freedom, with atom indices
/// running slower than coordinates.
Eigen::ArrayXXd load_FORCE_CONSTANTS_raw(const char* filename);

/// Create a Dielectric_parameters object from a Phonopy BORN file.
///
/// Note that the first line is ignored and VASP's default units
/// are assumed.
/// @param[in] filename - path to the BORN file
/// @return a Dielectric_parameters object containing all the
/// information in the BORN file.
std::unique_ptr<Dielectric_parameters> load_BORN(const char* filename);

/// Create a std::vector of Thirdorder_ifcs objects from a
/// FORCE_CONSTANTS_3RD file.
///
/// The format of the file is described in the ShengBTE
/// documentation.
/// @param[in] filename - path to the BORN file
/// @param[in] cell - description of the unit cell
/// @return a std::vector of Thirdorder_ifcs objects containing all
/// the information in the file.
std::unique_ptr<std::vector<Thirdorder_ifcs>> load_FORCE_CONSTANTS_3RD(
    const char* filename,
    const Crystal_structure& cell);
} // namespace alma
