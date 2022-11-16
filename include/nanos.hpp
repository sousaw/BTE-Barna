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
/// Code related to for simple nanostructures:
/// nanowires and nanoribbons.
/// This is done using approach developed by Li et. al.
/// see 10.1103/PhysRevB.85.195436 for theoretical
/// expressions. Scaling integrals are solved in
/// non-cartesian coordinates to make them easier
/// to work with

#include <unordered_map>
#include <constants.hpp>
#include <structures.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>
#include <isotopic_scattering.hpp>
#include <boost/mpi.hpp>
#if BOOST_VERSION >= 106700
#include <boost/container_hash/hash.hpp>
#else
#include <boost/functional/hash.hpp>
#endif
#include <Eigen/Dense>

/// Hash for pairs and arrays
namespace std {
template <typename T> struct hash<std::pair<T, T>> {
    std::size_t operator()(const std::pair<T, T>& key) const {
        hash<T> backend;
        std::size_t nruter = 0;
        boost::hash_combine(nruter, backend(key.first));
        boost::hash_combine(nruter, backend(key.second));
        return nruter;
    }
};
template <typename T, std::size_t S> struct hash<std::array<T, S>> {
    std::size_t operator()(const array<T, S>& key) const {
        hash<T> backend;
        std::size_t nruter = 0;

        for (auto& e : key)
            boost::hash_combine(nruter, backend(e));
        return nruter;
    }
};
} // namespace std

namespace alma {
namespace nanos {


/// Obtain the scaled lifetime for nanowires (see Eq. 8 of
/// 10.1103/PhysRevB.85.195436) the integral is
/// analitically solved in cilidrical coordinates
/// with z-axis being the transport axis
/// @param[in] tau0 - bulk lifetime
/// @param[in] uaxis - transport axis ( unitary vector )
/// @param[in] vel - phonon velocity
/// @param[in] R - nanowire radii
/// @return scaled tau
double scale_tau_nanowire(double tau0,
                          const Eigen::Ref<const Eigen::Vector3d>& uaxis,
                          const Eigen::Ref<const Eigen::Vector3d>& vel,
                          double R);

/// Obtain the scaled lifetime for nanoribbons(see Eq. 8 of
/// 10.1103/PhysRevB.85.195436) the integral is
/// analitically solved in xy-rotated cartesian
/// coordinates with new y-axis (y') aligned with
/// transport axis. The nanoribbon is then centered
/// and defined from x' = [-L/2,L/2]
/// NOTE: the nanoribbon is contained in xy plane
///       otherwise it will fail.
/// @param[in] tau0 - bulk lifetime
/// @param[in] uaxis - transport axis (unitary vector)
/// @param[in] vel - phonon velocity
/// @param[in] L - nanoribbon width
/// @return scaled tau

double scale_tau_nanoribbon(double tau0,
                            const Eigen::Ref<const Eigen::Vector3d>& uaxis,
                            const Eigen::Ref<const Eigen::Vector3d>& vel,
                            double L);

/// Obtain the thermal conductivity of a nanosystem
/// under RTA
///
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] w - RTA scattering rates
/// @param[in] uaxis - transport axis ( unitary vector )
/// @param[in] system_name - kind of nanosystem
/// @param[in] limiting_length - the limiting length of the nanostructure [nm]
/// @param[in] T - temperature in K
/// @return the RTA thermal conductivity over transpot axis
double calc_kappa_RTA(const alma::Crystal_structure& poscar,
                      const alma::Gamma_grid& grid,
                      const Eigen::Ref<const Eigen::ArrayXXd>& w,
                      const Eigen::Ref<const Eigen::Vector3d>& uaxis,
                      const std::string& system_name,
                      const double limiting_length,
                      double T);


/// Obtain the thermal conductivity of a nanowires system
/// under the full BTE (beyond Relaxation Time Approximation).
/// This is achieved by deterministic solution a linear system
/// over full BZ due to symmetry breaking caused by boundaries
/// (i.e.: scaling of tau can be different for bulk equivalent
///    qpoints. Consequently we cannot use some symmetry operations)
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] syms - symmetry operations object
/// @param[in] emission_processes   - list of 3-phonon emission processes
/// @param[in] absorption_processes - list of 3-phonon absorption processes
/// @param[in] isotopic_processes   - list of 2-phonon processes
/// @param[in] w0 - RTA scattering rates
/// @param[in] uaxis - transport axis ( unitary vector )
/// @param[in] system_name - kind of nanosystem
/// @param[in] limiting_length - the limiting length of the nanostructure [nm]
/// @param[in] T - temperature in K
/// @param[in] iterative - use iterative Eigen solver for faster computation
/// @param[in] world - mpi communicator
/// @return the thermal conductivity over transpot axis
double calc_kappa_nanos(
    const alma::Crystal_structure& poscar,
    const alma::Gamma_grid& grid,
    const alma::Symmetry_operations& syms,
    std::unordered_map<std::array<std::size_t, 3>, alma::Threeph_process>&
        emission_processes,
    std::unordered_map<std::array<std::size_t, 3>, alma::Threeph_process>&
        absorption_processes,
    std::unordered_map<std::pair<std::size_t, std::size_t>, double>&
        isotopic_processes,
    const Eigen::Ref<const Eigen::ArrayXXd>& w0,
    const Eigen::Ref<const Eigen::Vector3d>& uaxis,
    const std::string& system_name,
    const double limiting_length,
    double T,
    bool iterative,
    boost::mpi::communicator& world);


/// Obtain the thermal conductivity of a nanowires system
/// under the full BTE (beyond Relaxation Time Approximation).
/// This is achieved by deterministic solution a linear system
/// over full BZ. The elements are built from h5 (It is user responasibility
/// to ensure its symmetry)
/// @param[in] poscar - description of the unit cell
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] syms - symmetry operations object
/// @param[in] threeph_processes - list of 3-phonon  processes
/// @param[in] twoph_processes   - list of 2-phonon processes
/// @param[in] w0 - RTA scattering rates
/// @param[in] uaxis - transport axis ( unitary vector )
/// @param[in] system_name - kind of nanosystem
/// @param[in] limiting_length - the limiting length of the nanostructure [nm]
/// @param[in] T - temperature in K
/// @param[in] iterative - use iterative Eigen solver for faster computation
/// @param[in] world - mpi communicator
/// @return the thermal conductivity over transpot axis
double calc_kappa_nanos(
    const alma::Crystal_structure& poscar,
    const alma::Gamma_grid& grid,
    const alma::Symmetry_operations& syms,
    std::vector<alma::Threeph_process>& threeph_processes,
    std::vector<alma::Twoph_process>& twoph_processes,
    const Eigen::Ref<const Eigen::ArrayXXd>& w0,
    const Eigen::Ref<const Eigen::Vector3d>& uaxis,
    const std::string& system_name,
    const double limiting_length,
    double T,
    bool iterative,
    boost::mpi::communicator& world);


/// Get processes in full BZ
/// @param[in] grid - phonon spectrum on a regular q-point grid
/// @param[in] cell - description of the unit cell
/// @param[in] syms - symmetry operations object
/// @param[in,out] emission_processes   - list of 3-phonon emission processes
/// @param[in,out] absorption_processes - list of 3-phonon  processes
/// @param[in,out] isotopic_processes   - list of 2-phonon processes
/// @param[in] world - mpi communicator
/// @param[in] scalebroad_three - scale parameter for broadening in 3-ph
/// processes
void get_fullBZ_processes(
    const alma::Gamma_grid& grid,
    const alma::Crystal_structure& cell,
    std::string& anhIFCfile,
    std::unordered_map<std::array<std::size_t, 3>, alma::Threeph_process>&
        emission_processes,
    std::unordered_map<std::array<std::size_t, 3>, alma::Threeph_process>&
        absorption_processes,
    std::unordered_map<std::pair<std::size_t, std::size_t>, double>&
        isotopic_processes,
    boost::mpi::communicator& world,
    double scalebroad_three = 0.1);

} // namespace nanos
} // namespace alma
