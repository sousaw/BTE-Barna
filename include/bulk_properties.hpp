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
/// Code related to bulk properties such as the specific heat.

#include <constants.hpp>
#include <structures.hpp>
#include <qpoint_grid.hpp>

namespace alma {
/// Compute the specific heat at constant volume.
///
/// @param[in] poscar - a description of the unit cell
/// @param[in] grid - phonon spectrum on a regular grid
/// @param[in] T - temperature in K
/// @return  the volumetric specific heat
inline double calc_cv(const alma::Crystal_structure& poscar,
                      const alma::Gamma_grid& grid,
                      double T) {
    double nruter = 0.;
    double nqpoints = grid.nqpoints;
    double nmodes = grid.get_spectrum_at_q(0).omega.size();

    for (std::size_t iq = 0; iq < nqpoints; ++iq) {
        auto& spectrum = grid.get_spectrum_at_q(iq);

        for (std::size_t im = 0; im < nmodes; ++im)
            nruter += alma::bose_einstein_kernel(spectrum.omega(im), T);
    }
    nruter *= alma::constants::kB / nqpoints / poscar.V;
    return nruter;
}


/// Obtain the thermal conductivity in the relaxation
/// time approximation.
///
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] w - scattering rates for all modes in each of
/// the irreducible classes of q points in the grid.
/// @param[in] T - temperature in K
/// @return the thermal conductivity tensor in SI units
Eigen::MatrixXd calc_kappa(const alma::Crystal_structure& poscar,
                           const alma::Gamma_grid& grid,
                           const Eigen::Ref<const Eigen::ArrayXXd>& w,
                           double T);


/// Obtain l**2 hydrodinamic variable
/// for isotropic materials under RTA
///
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] w - scattering rates for all modes in each of
/// the irreducible classes of q points in the grid.
/// @param[in] T - temperature in K
/// @return l**2 in m**2 over three axis
Eigen::ArrayXd calc_l2_RTAisotropic(const alma::Crystal_structure& poscar,
                                    const alma::Gamma_grid& grid,
                                    const Eigen::Ref<const Eigen::ArrayXXd>& w,
                                    double T);

/// Obtain the small-grain thermal conductivity tensor.
///
/// The small-grain thermal conductivity is defined as the value of the
/// thermal conductivity tensor over the mean free path when the mean
// free
/// path is uniform across all modes.
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] T - temperature in K
/// @return the small thermal conductivity tensor [W / (m K nm)]
Eigen::MatrixXd calc_kappa_sg(const alma::Crystal_structure& poscar,
                              const alma::Gamma_grid& grid,
                              double T);

/// Obtain the thermal conductivity along a particular direction in the
/// relaxation time approximation.
///
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] w0 - scattering rates for all modes in each of
/// the irreducible classes of q points in the grid.
/// @param[in] T - temperature in K
/// @param[in] direction - 1D thermal transport direction in
/// Cartesian coordinates.
/// @return the thermal conductivity tensor in SI units
double calc_kappa_1d(const alma::Crystal_structure& poscar,
                     const alma::Gamma_grid& grid,
                     const Eigen::Ref<const Eigen::ArrayXXd>& w,
                     double T,
                     const Eigen::Ref<const Eigen::Vector3d>& direction);
} // namespace alma
