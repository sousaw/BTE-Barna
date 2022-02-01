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
/// Code implementing the treatment of binary superlattices as
/// a combination of an effective medium (virtual crystal) plus a
/// source of elastic scattering. See the following reference:
/// P. Chen, N. A. Katcho, J. P. Feser, Wu Li, M. Glaser, O. G.
/// Schmidt,
/// D. G. Cahill, N. Mingo & A. Rastelli
/// Physical Review Letter 111 (2013) 115901
/// https://dx.doi.org/10.1103/PhysRevLett.111.115901

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <pcg_random.hpp>
#pragma GCC diagnostic pop
#include <structures.hpp>
#include <vc.hpp>
#include <dynamical_matrix.hpp>

namespace alma {
/// POD class containing the information
/// about a binary superlattice.
class Superlattice_structure {
public:
    /// Direction normal to the superlattice layers in the crystal
    /// structure.
    const Eigen::Vector3i normal;
    /// Fraction of the second component in each layer.
    const Eigen::ArrayXd profile;
    /// Number of layers.
    const std::size_t nlayers;
    /// Average fraction of the second component in the
    /// superlattice.
    const double average;
    /// Structural description of the average virtual crystal.
    const std::unique_ptr<Crystal_structure> vc_struct;
    /// Basic constructor.
    ///
    /// @param[in] struct1 - structural description of the first
    /// component of the superlattice
    /// @param[in] struct2 - structural description of the second
    /// component of the superlattice
    /// @param[in] _normal - direction normal to the superlattice
    /// layers in the crystal structure
    /// @param[in] _profile - fraction of the second component in
    /// each layer
    Superlattice_structure(const Crystal_structure& struct1,
                           const Crystal_structure& struct2,
                           const Eigen::Ref<const Eigen::Vector3i>& _normal,
                           const Eigen::Ref<const Eigen::ArrayXd>& _profile)
        : normal(_normal), profile(_profile), nlayers(_profile.size()),
          average(_profile.sum() / profile.size()),
          vc_struct(
              vc_mix_structures({struct1, struct2}, {1. - average, average})) {
        if (struct1.is_alloy() || struct2.is_alloy())
            throw value_error("superlattices cannot be built "
                              "out of virtual crystals");

        if (_profile.size() == 0)
            throw value_error("empty profile");

        if ((_profile.minCoeff() < 0.) || (_profile.minCoeff() > 1.))
            throw value_error("all atomic rations must be in "
                              "[0, 1]");

        if (_profile.minCoeff() == _profile.maxCoeff())
            throw value_error("flat composition profile");
        this->fill_factors(struct1, struct2);
    }
    /// Compute the contribution to scattering from disorder in the
    /// effective medium.
    ///
    /// @param[in] grid - a regular grid containing Gamma
    /// @param[in] communicator - MPI communicator to use
    /// @param[in] scalebroad - factor modulating all the broadenings
    /// @return a set of scattering rates in 1 / ps
    Eigen::ArrayXXd calc_w0_medium(const alma::Gamma_grid& grid,
                                   const boost::mpi::communicator& comm,
                                   double scalebroad = 1.0) const;

    /// Compute the contribution to scattering from the supercell
    /// barriers.
    ///
    /// @param[in] grid - a regular grid containing Gamma
    /// @param[in] factory - object able to build D(q)
    /// @param[in] rng - random number generator to use
    /// @param[in] communicator - MPI communicator to use
    /// @param[in] nqline - number of q points in the 1D Brillouin
    /// zone used to build the Green's function.
    /// @return a set of scattering rates in 1 / ps
    Eigen::ArrayXXd calc_w0_barriers(const alma::Gamma_grid& grid,
                                     const Dynamical_matrix_builder& factory,
                                     randutils::mt19937_rng& rng,
                                     const boost::mpi::communicator& comm,
                                     std::size_t nqline) const;

private:
    /// Set of g factors for isotopic scattering in the
    /// superlattice.
    Eigen::MatrixXd gfactors;
    /// Coefficients of the perturbation matrix for the superlattice.
    Eigen::VectorXd vfactors;
    /// Fill the masses, gfactors and vfactors arrays with the
    /// appropriate values.
    ///
    /// @param[in] struct1 - structural description of the first
    /// component of the superlattice
    /// @param[in] struct2 - structural description of the second
    /// component of the superlattice
    void fill_factors(const Crystal_structure& struct1,
                      const Crystal_structure& struct2);
};
} // namespace alma
