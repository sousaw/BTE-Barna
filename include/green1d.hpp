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
/// Code for dealing with Green's functions for 1 dimension in real
/// space.

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include <dynamical_matrix.hpp>
#include <qpoint_grid.hpp>
#include <vasp_io.hpp>

namespace alma {
/// Objects of this class enable the calling code to compute 1d
/// Green's functions along particular directions in a bulk
/// material.
class Green1d_factory {
public:
    /// Number of q points in the grid.
    const std::size_t nqpoints;
    /// Number of degrees of freedom in the unit cell.
    const std::size_t ndof;
    /// Basic constructor.
    ///
    /// @param[in] structure - description of the crystal lattice
    /// @param[in] q0 - starting point in reciprocal space, in
    /// Cartesian coordinates
    /// @param[in] normal - direction of the path in reciprocal
    /// space, expressed in direct coordinates (only integers are
    /// allowed)
    /// @param[in] nqpoints_ - number of divisions of the segment
    /// @param[in] builder - factory of dynamical matrices
    Green1d_factory(const Crystal_structure& structure,
                    const Eigen::Ref<const Eigen::VectorXd>& q0,
                    const Eigen::Ref<const Eigen::VectorXi>& normal,
                    const std::size_t nqpoints_,
                    const Dynamical_matrix_builder& builder);
    /// Obtain the Green's function for a particular angular frequency.
    ///
    /// @param[in] omega - angular frequency [rad / ps]
    /// @param[in] ncells - number of unit cells to include in the
    /// supercell
    /// @return the Green's function matrix
    Eigen::MatrixXcd build(double omega, std::size_t ncells) const;

    /// Obtain the scattering rate caused by a diagonal perturbation
    /// proportional to the frequency squared
    ///
    /// @param[in] q - wave number of the incident phonon, as a number
    /// between 0 and 2 * pi
    /// @param[in] omega - frequency of the incident phonon [rad / ps]
    /// @param[in] wfin - wave function of the incident phonon over a
    /// single unit cell
    /// @param[in] factors - coefficient of omega ** 2 for each atom
    /// @return a scattering rate in ps^{-1}
    double calc_w0_dmass(
        double q,
        double omega,
        const Eigen::Ref<const Eigen::VectorXcd>& wfin,
        const Eigen::Ref<const Eigen::VectorXd>& factors) const;

private:
    /// Projection of the origin over the direction of interest,
    /// in the interval [0, 2 * pi).
    const double offset;
    /// Wave numbers at which the spectrum is sampled, from 0 to 2 *
    /// pi.
    Eigen::ArrayXd qgrid;
    /// Angular frequencies squared for all modes [rad ** 2 / ps ** 2].
    ///
    /// The first index runs over branches, the second over q points.
    Eigen::ArrayXXd Egrid;
    /// Directional derivative of omega ** 2 along the direction of
    /// interest.
    ///
    /// The first index runs over branches, the second over q points.
    Eigen::ArrayXXd dEgrid;
    /// Wave functions (over a single unit cell) for all modes.
    ///
    /// The main vector index runs over q points.
    std::vector<Eigen::MatrixXcd> wfgrid;
    /// Extend all wave functions at a particular q point from a single
    /// unit
    /// cell to a supercell.
    ///
    /// @param[in] iq - index of the q point
    /// @param[in] ncells - supercell size
    inline Eigen::MatrixXcd get_superwfs(std::size_t iq,
                                         std::size_t ncells) const;

    /// Compute the integration weights of all q points and all
    /// branches for a given energy.
    ///
    /// @param[in] omega - angular frequency squared [rad ** 2 / ps ** 2]
    /// @return the weights as a complex vector
    Eigen::MatrixXcd compute_weights(double omega2) const;
};
} // namespace alma
