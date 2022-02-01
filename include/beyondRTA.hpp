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
/// Code related to BTE calculations beyond the RTA.

#include <constants.hpp>
#include <structures.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>
#include <isotopic_scattering.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>

namespace alma {
namespace beyondRTA {
/// Obtain the thermal conductivity of a bulk system
/// under the full BTE (beyond Relaxation Time Approximation).
/// This is achieved by deterministic solution a linear system
/// for the irreducible q-points.
///
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] syms - symmetry operations object
/// @param[in] threeph_procs - list of 3-phonon processes
/// @param[in] twoph_procs - list of 2-phonon processes
/// @param[in] w0 - RTA scattering rates
/// @param[in] T - temperature in K
/// @param[in] iterative - use iterative Eigen solver for faster computation
/// @return the thermal conductivity tensor in SI units
Eigen::MatrixXd calc_kappa(
    const alma::Crystal_structure& poscar,
    const alma::Gamma_grid& grid,
    const alma::Symmetry_operations& syms,
    const std::vector<alma::Threeph_process>& threeph_procs,
    const std::vector<alma::Twoph_process>& twoph_procs,
    const Eigen::Ref<const Eigen::ArrayXXd>& w0,
    double T,
    bool iterative,
    boost::mpi::communicator& world);
} // namespace beyondRTA
} // namespace alma
