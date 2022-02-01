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
/// Code implementing the full BTE solution by the Omini-Sparavigna
/// iterative method, as implemented in ShengBTE.

#include <Eigen/Dense>
#include <processes.hpp>
#include <isotopic_scattering.hpp>

namespace alma {
/// Class implementing an iterative solution to the BTE following
/// the scheme devised by Omini and Sparavigna.
class ShengBTE_iterator {
private:
    /// Number of points in the grid.
    const std::size_t nqpoints;
    /// Number of irreducible q points.
    const std::size_t nirred;
    /// Number of branches.
    const std::size_t nbranches;
    /// Volume of the unit cell.
    const double V;
    /// Iteration number.
    std::size_t n;
    /// Angular frequencies for all points on the grid [rad / ps].
    Eigen::MatrixXd omega;
    /// Velocities for all points on the grid [nm / ps].
    Eigen::MatrixXd vg;
    /// Lifetimes for all points on the grid [nm / ps]
    Eigen::MatrixXd tau0;
    /// Value of the intermediate quantity F, measuring the distance to
    /// equilibrium, for the current iteration.
    Eigen::MatrixXd F;
    /// MPI communicator used to coordinate different processes.
    const boost::mpi::communicator& comm;

public:
    /// Basic constructor to initialize F.
    ///
    /// @param[in] poscar - description of the unit cell
    /// @param[in] grid - phonon spectrum on a regular q-point grid
    /// @param[in] syms - symmetry operations object
    /// @param[inout] threeph_procs - list of 3-phonon processes
    /// @param[inout] twoph_procs - list of 2-phonon processes
    /// @param[in] T - temperature in K
    /// @param[in] comm - communicator used to synchronize all processes
    ShengBTE_iterator(const Crystal_structure& poscar,
                      const Gamma_grid& grid,
                      const Symmetry_operations& syms,
                      std::vector<Threeph_process>& threeph_procs,
                      std::vector<Twoph_process>& twoph_procs,
                      double T,
                      const boost::mpi::communicator& comm_);


    /// Advance to the next iteration.
    ///
    /// @param[in] poscar - description of the unit cell
    /// @param[in] grid - phonon spectrum on a regular q-point grid
    /// @param[in] syms - symmetry operations object
    /// @param[inout] threeph_procs - list of 3-phonon processes
    /// @param[inout] twoph_procs - list of 2-phonon processes
    /// @param[in] T - temperature in K
    void next(const Crystal_structure& poscar,
              const Gamma_grid& grid,
              const Symmetry_operations& syms,
              std::vector<Threeph_process>& threeph_procs,
              std::vector<Twoph_process>& twoph_procs,
              double T);


    /// Get the current estimate of the thermal conductivity tensor.
    ///
    /// @param[in] T - temperature in K
    /// @return a 3x3 matrix with all the components of kappa
    /// [W / (m K)]
    Eigen::Matrix3d calc_current_kappa(double T) const;


    /// Get the current estimate of the contribution to the thermal conductivity
    /// tensor from a single branch.
    ///
    /// In this context, a branch is defined as the set of modes with the same
    /// index
    /// when energies are sorted in ascending order at each q point.
    /// @param[in] T - temperature in K
    /// @param[in] branch - branch index
    /// @return a 3x3 matrix with all the components of kappa
    /// [W / (m K)]
    Eigen::Matrix3d calc_current_kappa_branch(double T,
                                              std::size_t branch) const;


    /// Obtain the cumulative histogram of contributions to the thermal
    /// conductivity as a function of angular frequency.
    ///
    /// @param[in] T - temperature in K
    /// @param[ticks] - upper edges of the histogram bins
    /// @return a std::vector of thermal conductivity tensors with
    /// the cumulative histogram
    std::vector<Eigen::Matrix3d> calc_cumulative_kappa_omega(
        double T,
        Eigen::ArrayXd ticks);


    /// Obtain the cumulative histogram of contributions to the thermal
    /// conductivity as a function of pseudo-mean free path.
    ///
    /// @param[in] T - temperature in K
    /// @param[ticks] - upper edges of the histogram bins
    /// @return a std::vector of thermal conductivity tensors with
    /// the cumulative histogram
    std::vector<Eigen::Matrix3d> calc_cumulative_kappa_lambda(
        double T,
        Eigen::ArrayXd ticks);


    /// Obtain a set of pseudo-scattering rates from the current estimate of the
    /// solution.
    ///
    /// Note that these are not the inverse of any real relaxation time, since
    /// no such thing exist in the full linearized BTE formalism for phonons.
    /// Substituting these into the RTA expression for kappa will not yield
    /// the right thermal conductivity. Moreover, some elements can be negative.
    /// See the ShengBTE paper for details.
    /// @return an array of pseudo-scattering rates for each q point and each
    /// mode [ps ** (-1)]
    Eigen::ArrayXXd calc_w() const;


    /// Obtain a set of pseudo-mean free paths from the current estimate of the
    /// solution.
    ///
    /// Note that these are not real MFPs, since no such thing exist in the full
    /// linearized BTE formalism for phonons. Substituting these into the RTA
    /// expression for kappa will not yield the right thermal conductivity.
    /// Moreover, some elements can be negative.See the ShengBTE paper for
    /// details.
    /// @return an array of pseudo-scattering rates for each q point and each
    /// mode [ps ** (-1)]
    Eigen::ArrayXXd calc_lambda() const;
};


/// Compute the full thermal conductivity tensor use the
/// Omini-Sparavigna
/// iterative approach.
///
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] syms - symmetry operations object
/// @param[inout] threeph_procs - list of 3-phonon processes
/// @param[inout] twoph_procs - list of 2-phonon processes
/// @param[in] T - temperature in K
/// @param[in] comm - communicator used to synchronize all processes
/// @param[in] tolerance - maximum change in the norm betwwen iterations
/// used as the convergence criterion
/// @param[in] maxiter - maximum number of iterations before giving up
/// @return the thermal conductivity tensor in SI units
Eigen::MatrixXd calc_shengbte_kappa(
    const alma::Crystal_structure& poscar,
    const alma::Gamma_grid& grid,
    const alma::Symmetry_operations& syms,
    std::vector<alma::Threeph_process>& threeph_procs,
    std::vector<alma::Twoph_process>& twoph_procs,
    double T,
    const boost::mpi::communicator& comm,
    double tolerance = 1e-4,
    std::size_t maxiter = 1000);
} // namespace alma