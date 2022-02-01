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
/// Code related to the virtual crystal approximation for alloys and
/// isotopic mixtures.

#include <structures.hpp>
#include <dos.hpp>

namespace alma {
/// Create an average structure out of several Crystal_structure
/// objects containing the same atomic sites.
///
/// The resulting structure will be based on the weighted average of
/// the lattice vectors of the components, and contain "virtual"
/// elements with the right compositions.
/// @param[in] components - input structures
/// @param[in] ratios - set of positive weights for the components
/// @return the weighted average of the input structures
std::unique_ptr<Crystal_structure> vc_mix_structures(
    const std::vector<Crystal_structure>& components,
    const std::vector<double>& ratios);

/// Create a set of average dielectric parameters from compatible
/// inputs.
///
/// @param[in] components - input dielectric parameters
/// @param[in] ratios - set of positive weights for the components
/// @return the weighted average of the input dielectric parameters
std::unique_ptr<Dielectric_parameters> vc_mix_dielectric_parameters(
    const std::vector<Dielectric_parameters>& components,
    const std::vector<double>& ratios);

/// Create a set of average second-order force constants from
/// compatible inputs.
///
/// @param[in] components - input harmonic force constants
/// @param[in] ratios - set of positive weights for the components
/// @return the weighted average of the input harmomic force
/// constants.
std::unique_ptr<Harmonic_ifcs> vc_mix_harmonic_ifcs(
    const std::vector<Harmonic_ifcs>& components,
    const std::vector<double>& ratios);

/// Create a set of average third-order force constants from
/// compatible inputs.
///
/// @param[in] components - input third-order force constants
/// @param[in] ratios - set of positive weights for the components
/// @return the weighted average of the input third-order force
/// constants.
std::unique_ptr<std::vector<Thirdorder_ifcs>> vc_mix_thirdorder_ifcs(
    const std::vector<std::vector<Thirdorder_ifcs>>& components,
    const std::vector<double>& ratios);
} // namespace alma
