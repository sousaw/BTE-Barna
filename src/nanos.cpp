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

/// @file
/// Implements full BTE calculations defined in nanos.hpp.

#include <iostream>
#include <map>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unordered_map>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/src/IterativeSolvers/Scaling.h>
#include <unsupported/Eigen/src/IterativeSolvers/GMRES.h>

#include <utilities.hpp>
#include <nanos.hpp>
#include <vasp_io.hpp>

/// TBB stuff
#include <mutex>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>


namespace alma {
namespace nanos {


double scale_tau_nanowire(double tau0,
                          const Eigen::Ref<const Eigen::Vector3d>& uaxis,
                          const Eigen::Ref<const Eigen::Vector3d>& vel,
                          double R) {
    /// Calculate the velocity vector projected onto a
    /// plane with normal vector equal to uaxis
    Eigen::Vector3d vrho_xyz = vel - vel.dot(uaxis) * uaxis;

    double vrho = vrho_xyz.norm();
    double mfp = vrho * tau0;

    if (alma::almost_equal(mfp, 0.)) {
        return tau0;
    }


    return tau0 *
           (1 - 2 / (R * R) * mfp * (mfp * (std::exp(-R / mfp) - 1) + R));
}

double scale_tau_nanoribbon(double tau0,
                            const Eigen::Ref<const Eigen::Vector3d>& uaxis,
                            const Eigen::Ref<const Eigen::Vector3d>& vel,
                            double L) {
    /// It is supposed that the z axis is the one discarded


    /// Change-of-base matrices to xy coordinates with
    /// y-axis equal to uaxis
    static Eigen::Matrix2d B = Eigen::Matrix2d::Zero();
    static Eigen::Matrix2d Binv;

    /// If not inited fill B and its inverse
    if (alma::almost_equal(B.sum(), 0.)) {
        if (!alma::almost_equal(uaxis(2), 0.)) {
            std::cout << "ERROR: Nanoribbons are defined in xy plane"
                      << std::endl;
            exit(1);
        }

        B << uaxis(1), uaxis(0), -uaxis(0), uaxis(1);
        Binv = B.inverse();
    }

    /// Calculate velocity in new coordinates
    Eigen::Vector2d v_cart, v_newcord;
    v_cart << vel(0), vel(1);
    v_newcord = Binv * v_cart;

    double mfp = std::abs(tau0 * v_newcord(0));

    if (alma::almost_equal(mfp, 0.)) {
        return tau0;
    }

    return tau0 * (mfp / L * (std::exp(-L / mfp) - 1.0) + 1.0);
}


double calc_kappa_RTA(const alma::Crystal_structure& poscar,
                      const alma::Gamma_grid& grid,
                      const Eigen::Ref<const Eigen::ArrayXXd>& w,
                      const Eigen::Ref<const Eigen::Vector3d>& uaxis,
                      const std::string& system_name,
                      const double limiting_length,
                      double T) {
    auto Nbranches =
        static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size());

    if ((static_cast<std::size_t>(w.rows()) != Nbranches) ||
        (static_cast<std::size_t>(w.cols()) != grid.nqpoints))
        throw alma::value_error("inconsistent dimensions");

    double kappa = 0.;

    // The Gamma point is ignored.
    for (decltype(Nbranches) iq = 1; iq < grid.nqpoints; iq++) {
        auto sp = grid.get_spectrum_at_q(iq);

        for (decltype(Nbranches) im = 0; im < Nbranches; im++) {
            double tau = (w(im, iq) == 0.) ? 0. : (1. / w(im, iq));

            Eigen::Vector3d vg = sp.vg.col(im);
            double vgproj = vg.dot(uaxis);


            /// Scale the lifetimes
            if (system_name == "nanowire") {
                tau = scale_tau_nanowire(tau, uaxis, vg, limiting_length);
            }
            else if (system_name == "nanoribbon") {
                tau = scale_tau_nanoribbon(tau, uaxis, vg, limiting_length);
            }
            else {
                std::cout << "ERROR: calc_k not recognized system_name"
                          << std::endl;
                exit(1);
            }


            kappa += alma::bose_einstein_kernel(sp.omega[im], T) * tau *
                     vgproj * vgproj;
        }
    }
    return (1e21 * alma::constants::kB / poscar.V / grid.nqpoints) * kappa;
}

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
    double scalebroad_three) {
    /// Get anharmonic IFC
    auto anhIFC = alma::load_FORCE_CONSTANTS_3RD(anhIFCfile.c_str(), cell);

    /// First we are getting sizes
    std::size_t nbands = grid.get_spectrum_at_q(0).omega.size();

    std::size_t my_id = world.rank();
    std::size_t nprocs = world.size();

    /// Create a unique set of Three_phonon processes from irreductible triplets
    /// including the symmetry:
    auto limits = alma::my_jobs(grid.get_nequivalences(), nprocs, my_id);
    std::array<int, 2> signs(
        {{static_cast<int>(alma::threeph_type::emission),
          static_cast<int>(alma::threeph_type::absorption)}});

    std::vector<alma::Threeph_process> myprocesses;

    tbb::concurrent_unordered_map<std::array<std::size_t, 3>,
                                  alma::Threeph_process,
                                  std::hash<std::array<std::size_t, 3>>>
        emission_processes_tbb;
    tbb::concurrent_unordered_map<std::array<std::size_t, 3>,
                                  alma::Threeph_process,
                                  std::hash<std::array<std::size_t, 3>>>
        absorption_processes_tbb;

    /// Calculating the three phonon processes for full BZ. The anhIFC are
    /// needed in case that due to symmetrization we are registering some
    /// transition that was not available in the already calculated elements. We
    /// are calculating them under the assumption that the matrix element is
    /// invariant to rotations, inversions and reciprocity but the smearing is
    /// not.
    for (auto ic = limits[0]; ic < limits[1]; ++ic) {
        if (my_id == 0)
            std::cout << "#Recalculating fullBZ triplets "
                      << static_cast<std::size_t>(ic - limits[0]) << " / "
                      << static_cast<std::size_t>(limits[1] - limits[0])
                      << std::endl;
        auto iq1 = grid.get_representative(ic);
        auto coords1 = grid.one_to_three(iq1);
        auto spectrum1 = grid.get_spectrum_at_q(iq1);

        tbb::parallel_for(
            tbb::blocked_range<std::size_t>(0, grid.nqpoints),
            [&](tbb::blocked_range<std::size_t> iq2range) {
                for (std::size_t iq2 = iq2range.begin(); iq2 < iq2range.end();
                     ++iq2) {
                    auto coords2 = grid.one_to_three(iq2);
                    auto spectrum2 = grid.get_spectrum_at_q(iq2);
                    decltype(coords2) coords3;

                    // Emission and absorption processes satisfy different
                    // conservation rules, with the second phonon in
                    // different sides of the equations.
                    for (auto s : signs) {
                        for (auto i = 0; i < 3; ++i)
                            coords3[i] = coords1[i] + s * coords2[i];
                        auto iq3 = grid.three_to_one(coords3);
                        auto spectrum3 = grid.get_spectrum_at_q(iq3);

                        auto eqqtrip =
                            grid.equivalent_qtriplets({iq1, iq2, iq3});

                        for (decltype(nbands) im1 = 0; im1 < nbands; ++im1) {
                            if (alma::almost_equal(spectrum1.omega(im1), 0.))
                                continue;

                            for (decltype(nbands) im2 = 0; im2 < nbands;
                                 ++im2) {
                                if (alma::almost_equal(spectrum2.omega(im2),
                                                       0.))
                                    continue;

                                for (decltype(nbands) im3 = 0; im3 < nbands;
                                     ++im3) {
                                    if (alma::almost_equal(spectrum3.omega(im3),
                                                           0.))
                                        continue;

                                    auto delta =
                                        std::fabs(spectrum1.omega(im1) +
                                                  s * spectrum2.omega(im2) -
                                                  spectrum3.omega(im3));

                                    std::vector<double> vpsXdelta; //, sigmas;
                                    std::vector<alma::Threeph_process> mythree;


                                    for (auto& qt : eqqtrip) {
                                        auto v1 = grid.get_spectrum_at_q(qt[0])
                                                      .vg.col(im1);
                                        auto v2 = grid.get_spectrum_at_q(qt[1])
                                                      .vg.col(im2);
                                        auto v3 = grid.get_spectrum_at_q(qt[2])
                                                      .vg.col(im3);


                                        if (static_cast<alma::threeph_type>(
                                                s) ==
                                            alma::threeph_type::absorption) {
                                            auto sigma =
                                                scalebroad_three *
                                                std::sqrt(
                                                    boost::math::pow<2>(
                                                        grid.base_sigma(v1)) +
                                                    boost::math::pow<2>(
                                                        grid.base_sigma(v2)) +
                                                    boost::math::pow<2>(
                                                        grid.base_sigma(v3)));
                                            //                                                 0.1 * grid.base_sigma(v2 - v3);
                                            alma::Threeph_process
                                                Gamma_of_triplet1(
                                                    ic,
                                                    std::array<std::size_t, 3>(
                                                        {{qt[0],
                                                          qt[1],
                                                          qt[2]}}),
                                                    std::array<std::size_t, 3>(
                                                        {{im1, im2, im3}}),
                                                    static_cast<
                                                        alma::threeph_type>(s),
                                                    delta,
                                                    sigma);
                                            mythree.push_back(
                                                Gamma_of_triplet1);
                                            if (delta <=
                                                alma::constants::nsigma *
                                                    sigma) {
                                                //                                         Gamma_of_triplet1.compute_vp2(cell,grid,anhIFC);
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet1
                                                        .compute_gaussian());
                                            }
                                            //                                             sigma =
                                            //                                                 0.1 * grid.base_sigma(v1 - v3);
                                            alma::Threeph_process
                                                Gamma_of_triplet2(
                                                    ic,
                                                    std::array<std::size_t, 3>(
                                                        {{qt[1],
                                                          qt[0],
                                                          qt[2]}}),
                                                    std::array<std::size_t, 3>(
                                                        {{im2, im1, im3}}),
                                                    static_cast<
                                                        alma::threeph_type>(s),
                                                    delta,
                                                    sigma);
                                            mythree.push_back(
                                                Gamma_of_triplet2);
                                            if (delta <=
                                                alma::constants::nsigma *
                                                    sigma) {
                                                //                                         Gamma_of_triplet2.compute_vp2(cell,grid,anhIFC);
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet2
                                                        .compute_gaussian());
                                            }
                                            //                                             sigma =
                                            //                                                 0.1 * grid.base_sigma(v1 - v2);
                                            alma::Threeph_process
                                                Gamma_of_triplet3(
                                                    ic,
                                                    std::array<std::size_t, 3>(
                                                        {{qt[2],
                                                          qt[0],
                                                          qt[1]}}),
                                                    std::array<std::size_t, 3>(
                                                        {{im3, im1, im2}}),
                                                    static_cast<
                                                        alma::threeph_type>(-s),
                                                    delta,
                                                    sigma);
                                            mythree.push_back(
                                                Gamma_of_triplet3);
                                            if (delta <=
                                                alma::constants::nsigma *
                                                    sigma) {
                                                //                                         Gamma_of_triplet3.compute_vp2(cell,grid,anhIFC);
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet3
                                                        .compute_gaussian());
                                            }
                                            /*                                            sigma
                                               = 0.1 * grid.base_sigma(v2 -
                                               v1)*/
                                            ;
                                            alma::Threeph_process
                                                Gamma_of_triplet4(
                                                    ic,
                                                    std::array<std::size_t, 3>(
                                                        {{qt[2],
                                                          qt[1],
                                                          qt[0]}}),
                                                    std::array<std::size_t, 3>(
                                                        {{im3, im2, im1}}),
                                                    static_cast<
                                                        alma::threeph_type>(-s),
                                                    delta,
                                                    sigma);
                                            mythree.push_back(
                                                Gamma_of_triplet4);
                                            if (delta <=
                                                alma::constants::nsigma *
                                                    sigma) {
                                                //                                         Gamma_of_triplet4.compute_vp2(cell,grid,anhIFC);
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet4
                                                        .compute_gaussian());
                                            }
                                        }
                                        else {
                                            auto sigma =
                                                1.0 *
                                                std::sqrt(
                                                    boost::math::pow<2>(
                                                        grid.base_sigma(v1)) +
                                                    boost::math::pow<2>(
                                                        grid.base_sigma(v2)) +
                                                    boost::math::pow<2>(
                                                        grid.base_sigma(v3)));
                                            //                                                 0.1 * grid.base_sigma(v2 - v3);
                                            alma::Threeph_process
                                                Gamma_of_triplet1(
                                                    ic,
                                                    std::array<std::size_t, 3>(
                                                        {{qt[0],
                                                          qt[1],
                                                          qt[2]}}),
                                                    std::array<std::size_t, 3>(
                                                        {{im1, im2, im3}}),
                                                    static_cast<
                                                        alma::threeph_type>(s),
                                                    delta,
                                                    sigma);
                                            mythree.push_back(
                                                Gamma_of_triplet1);
                                            if (delta <=
                                                alma::constants::nsigma *
                                                    sigma) {
                                                //                                         Gamma_of_triplet1.compute_vp2(cell,grid,anhIFC);
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet1
                                                        .compute_gaussian());
                                            }
                                            //                                             sigma =
                                            //                                                 0.1 * grid.base_sigma(v3 - v2);
                                            alma::Threeph_process
                                                Gamma_of_triplet2(
                                                    ic,
                                                    std::array<std::size_t, 3>(
                                                        {{qt[0],
                                                          qt[2],
                                                          qt[1]}}),
                                                    std::array<std::size_t, 3>(
                                                        {{im1, im3, im2}}),
                                                    static_cast<
                                                        alma::threeph_type>(s),
                                                    delta,
                                                    sigma);
                                            mythree.push_back(
                                                Gamma_of_triplet2);
                                            if (delta <=
                                                alma::constants::nsigma *
                                                    sigma) {
                                                //                                         Gamma_of_triplet2.compute_vp2(cell,grid,anhIFC);
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet2
                                                        .compute_gaussian());
                                            }
                                            //                                             sigma =
                                            //                                                 0.1 * grid.base_sigma(v2 - v1);
                                            alma::Threeph_process
                                                Gamma_of_triplet3(
                                                    ic,
                                                    std::array<std::size_t, 3>(
                                                        {{qt[2],
                                                          qt[1],
                                                          qt[0]}}),
                                                    std::array<std::size_t, 3>(
                                                        {{im3, im2, im1}}),
                                                    static_cast<
                                                        alma::threeph_type>(-s),
                                                    delta,
                                                    sigma);
                                            mythree.push_back(
                                                Gamma_of_triplet3);
                                            if (delta <=
                                                alma::constants::nsigma *
                                                    sigma) {
                                                //                                         Gamma_of_triplet3.compute_vp2(cell,grid,anhIFC);
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet3
                                                        .compute_gaussian());
                                            }
                                            //                                             sigma =
                                            //                                                 0.1 * grid.base_sigma(v3 - v1);
                                            alma::Threeph_process
                                                Gamma_of_triplet4(
                                                    ic,
                                                    std::array<std::size_t, 3>(
                                                        {{qt[1],
                                                          qt[2],
                                                          qt[0]}}),
                                                    std::array<std::size_t, 3>(
                                                        {{im2, im3, im1}}),
                                                    static_cast<
                                                        alma::threeph_type>(-s),
                                                    delta,
                                                    sigma);
                                            mythree.push_back(
                                                Gamma_of_triplet4);
                                            if (delta <=
                                                alma::constants::nsigma *
                                                    sigma) {
                                                // Gamma_of_triplet4.compute_vp2(cell,grid,anhIFC);
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet4
                                                        .compute_gaussian());
                                            }
                                        }
                                    }

                                    if (vpsXdelta.empty())
                                        continue;
                                    std::sort(mythree.begin(), mythree.end());
                                    double VP2 = mythree[0].compute_vp2(
                                        cell, grid, *anhIFC);

                                    double SymGamma = 0.;
                                    for (auto& vD : vpsXdelta) {
                                        SymGamma += VP2 * vD;
                                    }

                                    SymGamma /= 4 * eqqtrip.size();


                                    /// If not 0 all register in map, we account
                                    /// for a state decaying into two of same
                                    /// index
                                    if (!almost_equal(
                                            SymGamma, 0., 1.0e-12, 1.0e-9)) {
                                        for (auto& peq : mythree) {
                                            peq.set_vp2(SymGamma);
                                            std::array<std::size_t, 3>
                                                triplet_indexes;
                                            for (std::size_t idx_ = 0; idx_ < 3;
                                                 idx_++)
                                                triplet_indexes[idx_] =
                                                    peq.q[idx_] * nbands +
                                                    peq.alpha[idx_];
                                            if (peq.type == alma::threeph_type::
                                                                absorption) {
                                                if (triplet_indexes[0] ==
                                                    triplet_indexes[1]) {
                                                    double effective_vp2 =
                                                        2 * peq.get_vp2();
                                                    peq.set_vp2(effective_vp2);
                                                }

                                                if (absorption_processes_tbb
                                                        .count(
                                                            triplet_indexes) ==
                                                    0) {
                                                    absorption_processes_tbb
                                                        .emplace(std::make_pair(
                                                            triplet_indexes,
                                                            peq));
                                                }
                                            }
                                            else {
                                                // We account for those states
                                                // by multiplying by 2
                                                if (triplet_indexes[1] ==
                                                    triplet_indexes[2]) {
                                                    double effective_vp2 =
                                                        2 * peq.get_vp2();
                                                    peq.set_vp2(effective_vp2);
                                                }
                                                if (emission_processes_tbb
                                                        .count(
                                                            triplet_indexes) ==
                                                    0) {
                                                    emission_processes_tbb
                                                        .emplace(std::make_pair(
                                                            triplet_indexes,
                                                            peq));
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
    }

    /// Passing tbb map to stl map
    absorption_processes.insert(absorption_processes_tbb.begin(),
                                absorption_processes_tbb.end());
    emission_processes.insert(emission_processes_tbb.begin(),
                              emission_processes_tbb.end());


    /// We are now working on isotopic/mass-disoreder scattering
    if (my_id == 0)
        std::cout << "#Working on 2ph processes" << std::endl;

    auto twoph_processes = alma::find_allowed_twoph(grid, world, 1.0);

    double scalebroad_iso = 1.0;

    /// We need sigma full list to symmetrize:
    Eigen::ArrayXXd broadening_sigmas(nbands, grid.nqpoints);
    for (std::size_t iq = 0; iq < grid.nqpoints; ++iq) {
        for (std::size_t im = 0; im < nbands; ++im) {
            auto spectrum = grid.get_spectrum_at_q(iq);
            broadening_sigmas(im, iq) =
                scalebroad_iso * grid.base_sigma(spectrum.vg.col(im));
        }
    }
    // And refine them by removing outliers.
    auto percent = calc_percentiles_log(broadening_sigmas);
    double lbound = std::exp(percent[0] - 1.5 * (percent[1] - percent[0]));
    broadening_sigmas =
        (broadening_sigmas < lbound).select(lbound, broadening_sigmas);


    for (auto& twoph : twoph_processes) {
        auto b0 = twoph.alpha[0];
        auto b1 = twoph.alpha[1];

        auto sp0 = grid.get_spectrum_at_q(twoph.q[0]);
        auto sp1 = grid.get_spectrum_at_q(twoph.q[1]);

        auto delta = std::abs(sp0.omega(b0) - sp1.omega(b1));

        /// If some of them is Gamma-point and acoustic, ignore
        if (twoph.q[0] * nbands + b0 < 3 or twoph.q[1] * nbands + b1 < 3)
            continue;

        /// The scattering to itselt has no effect in scattering operator
        if (twoph.q[0] * nbands + b0 == twoph.q[1] * nbands + b1)
            continue;

        auto qpairs = grid.equivalent_qpairs({twoph.q[0], twoph.q[1]});

        std::vector<alma::Twoph_process> mytwo;

        double gamma = 0.;

        for (auto& qi : qpairs) {
            auto sigma = std::hypot(broadening_sigmas(b0, qi[0]),
                                    broadening_sigmas(b1, qi[1]));
            alma::Twoph_process t1(0,
                                   std::array<std::size_t, 2>({{qi[0], qi[1]}}),
                                   std::array<std::size_t, 2>({{b0, b1}}),
                                   delta,
                                   sigma);
            alma::Twoph_process t2(0,
                                   std::array<std::size_t, 2>({{qi[1], qi[0]}}),
                                   std::array<std::size_t, 2>({{b1, b0}}),
                                   delta,
                                   sigma);
            mytwo.push_back(t1);
            mytwo.push_back(t2);
            if (delta <= alma::constants::nsigma * sigma) {
                gamma += t1.compute_gamma(cell, grid);
                gamma += t2.compute_gamma(cell, grid);
            }
        }
        gamma /= 2 * qpairs.size();
        /// If some equivalent process is not null
        if (!almost_equal(gamma, 0., 1.0e-12, 1.0e-9)) {
            for (auto& peq : mytwo) {
                std::pair<std::size_t, std::size_t> pair_mode_ids =
                    std::make_pair(peq.q[0] * nbands + peq.alpha[0],
                                   peq.q[1] * nbands + peq.alpha[1]);
                if (isotopic_processes.count(pair_mode_ids) == 0)
                    isotopic_processes.emplace(
                        std::make_pair(pair_mode_ids, gamma));
            }
        }
    }
}

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
    boost::mpi::communicator& world) {
    // GATHER INFORMATION FOR FULL BTE SYSTEM AND DECLARE SOME VARIABLES

    int Ngridpoints = grid.nqpoints;

    int Nbranches = grid.get_spectrum_at_q(0).omega.size();
    int Ntot = Ngridpoints * Nbranches - 3;

    // Unknowns in the linear system
    Eigen::VectorXd H(Ntot);

    // stores heat capacities of the irreducible points
    Eigen::VectorXd C(Ntot);
    C.setConstant(0.0);

    // stores relaxation times of the irreducible points
    Eigen::VectorXd tau(Ntot);
    tau.setConstant(0.0);

    // stores phonon frequencies of the irreducible points
    Eigen::VectorXd omega(Ntot);
    omega.setConstant(0.0);

    // stores group velocity vectors of the irreducible points
    Eigen::MatrixXd vg(Ntot, 3);
    Eigen::VectorXd vgproj(Ntot);
    vg.fill(0.0);

    // GATHER PHONON PROPERTIES FOR IRREDUCIBLE Q-POINTS

    double C_factor = 1e27 * alma::constants::kB / Ngridpoints / poscar.V;

    // scan over all points in the grid

    for (int nq = 0; nq < Ngridpoints; nq++) {
        // spectrum at point

        auto spectrum = grid.get_spectrum_at_q(nq);


        for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
            int imode = nq * Nbranches + nbranch - 3;

            if (imode < 0)
                continue;

            // store heat capacity
            C(imode) = C_factor *
                       alma::bose_einstein_kernel(spectrum.omega[nbranch], T);


            // store group velocity vector and projected one
            Eigen::Vector3d myvg = spectrum.vg.col(nbranch);
            vg(imode, 0) = myvg(0);
            vg(imode, 1) = myvg(1);
            vg(imode, 2) = myvg(2);

            vgproj(imode) = myvg.dot(uaxis);

            // store relaxation time
            double mytau =
                (w0(nbranch, nq) == 0.) ? 0. : (1. / w0(nbranch, nq));

            if (system_name == "nanowire") {
                mytau = scale_tau_nanowire(mytau, uaxis, myvg, limiting_length);
            }
            else if (system_name == "nanoribbon") {
                mytau =
                    scale_tau_nanoribbon(mytau, uaxis, myvg, limiting_length);
            }
            else {
                std::cout
                    << "ERROR: calc_kappa_nanos not recognized system_name"
                    << std::endl;
                exit(1);
            }

            tau(imode) = mytau;

            // store phonon frequency
            omega(imode) = spectrum.omega(nbranch);
        }
    }

    typedef std::array<int, 2> idx_pair;

    std::unordered_map<idx_pair, double> A_elements;
    A_elements.reserve(std::ceil(0.15 * Ntot * Ntot));

    // CALCULATE NON-RTA CONDUCTIVITY BY SOLVING LINEAR SYSTEM

    // build the list of Gamma values for 2-phonon processes
    for (auto& process_block : isotopic_processes) {
        auto I = process_block.first.first;
        auto J = process_block.first.second;
        A_elements[{static_cast<int>(I - 3), static_cast<int>(J - 3)}] -=
            process_block.second * tau(I - 3);
    } // done scanning over all 2-phonon processes

    for (auto& process_block : absorption_processes) {
        auto process(process_block.second);
        auto I = process.q[0] * Nbranches + process.alpha[0];
        auto J = process.q[1] * Nbranches + process.alpha[1];
        auto K = process.q[2] * Nbranches + process.alpha[2];

        double Gamma = process.compute_gamma(grid, T, true);

        A_elements[{static_cast<int>(I - 3), static_cast<int>(J - 3)}] +=
            Gamma * tau(I - 3);
        A_elements[{static_cast<int>(I - 3), static_cast<int>(K - 3)}] -=
            Gamma * tau(I - 3);
    }


    for (auto& process_block : emission_processes) {
        auto process(process_block.second);
        auto I = process.q[0] * Nbranches + process.alpha[0];
        auto J = process.q[1] * Nbranches + process.alpha[1];
        auto K = process.q[2] * Nbranches + process.alpha[2];

        double Gamma = process.compute_gamma(grid, T, true);

        A_elements[{static_cast<int>(I - 3), static_cast<int>(J - 3)}] -=
            0.5 * Gamma * tau(I - 3);
        A_elements[{static_cast<int>(I - 3), static_cast<int>(K - 3)}] -=
            0.5 * Gamma * tau(I - 3);
    }


    // diagonal elements in the system
    for (int diag_idx = 0; diag_idx < Ntot; diag_idx++) {
        A_elements[{diag_idx, diag_idx}] += 1.0;
    }


    /// Get tripletList
    std::vector<Eigen::Triplet<double>> tripletList;
    for (auto& [key, val] : A_elements) {
        auto idx1 = key[0];
        auto idx2 = key[1];
        tripletList.push_back(Eigen::Triplet<double>(idx1, idx2, val));
    }

    // build system of equations "A*H = B"

    Eigen::SparseMatrix<double> A(Ntot, Ntot);

    // Fill sparse matrix
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    Eigen::VectorXd B(Ntot);

    // projected-components of omega*MFP_RTA over transport axis
    B = omega.array() * tau.array() * vgproj.array();

    // Transform to compressed format

    A.makeCompressed();

    std::cout << "#Linear system information\n";
    std::cout << "**A sparsity "
              << static_cast<double>(A.nonZeros()) / (Ntot * Ntot) << std::endl;
    std::cout.flush();

    // Scale system
    Eigen::IterScaling<Eigen::SparseMatrix<double>> scal;
    scal.computeRef(A);
    B = scal.LeftScaling().cwiseProduct(B);

    // SOLVE SYSTEM

    if (iterative) {
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
        // Eigen::GMRES<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);

        // RTA solution to be used as initial guess
        Eigen::VectorXd H_guess(Ntot);

        H_guess = omega.array() * tau.array() * vgproj.array();

        H_guess = scal.RightScaling().cwiseInverse().cwiseProduct(H_guess);

        H = solver.solveWithGuess(B, H_guess);

        std::cout << "# Iterative solver information:\n" << std::endl;
        std::cout << "# -iterations:     " << solver.iterations() << std::endl;
        std::cout << "# -estimated error: " << solver.error() << std::endl;
    }

    else {
        H = omega.array() * tau.array() * vgproj.array();
    }

    double rel_err = (A * H - B).norm() / B.norm();

    if (rel_err > 1e-3) {
        std::cout << "alma::beyondRTA::calc_kappa_nanos > WARNING:"
                  << std::endl;
        std::cout << "solution of linear system might be unstable."
                  << std::endl;
        std::cout << "Relative error metric = " << rel_err << std::endl;
    }

    // Scale back the solution
    H = scal.RightScaling().cwiseProduct(H);

    // PROCESS THE LINEAR SYSTEM SOLUTION

    // construct "generalised MFPs"
    Eigen::VectorXd MFP_nonRTA(Ntot);
    MFP_nonRTA = H.array() / omega.array();

    // fix potential NaN/Inf problems
    for (int n = 0; n < Ntot; n++) {
        if (omega(n) <= 0.0) {
            MFP_nonRTA(n) = 0.0;
        }
    }

    // CALCULATE NON-RTA CONDUCTIVITY

    double kappa_nano = 0.0;
    for (int nq = 0; nq < Ngridpoints; nq++) {
        for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
            int imode = nq * Nbranches + nbranch - 3;

            if (imode < 0)
                continue;

            kappa_nano += 1e-6 * C(imode) * MFP_nonRTA(imode) * vgproj(imode);
        }
    }

    // RETURN RESULT
    return kappa_nano;

} // end of calc_kappa_nanos




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
    boost::mpi::communicator& world) {
    // GATHER INFORMATION FOR FULL BTE SYSTEM AND DECLARE SOME VARIABLES

    int Ngridpoints = grid.nqpoints;

    int Nbranches = grid.get_spectrum_at_q(0).omega.size();
    int Ntot = Ngridpoints * Nbranches - 3;

    // Unknowns in the linear system
    Eigen::VectorXd H(Ntot);

    // stores heat capacities of the irreducible points
    Eigen::VectorXd C(Ntot);
    C.setConstant(0.0);

    // stores relaxation times of the irreducible points
    Eigen::VectorXd tau(Ntot);
    tau.setConstant(0.0);

    // stores phonon frequencies of the irreducible points
    Eigen::VectorXd omega(Ntot);
    omega.setConstant(0.0);

    // stores group velocity vectors of the irreducible points
    Eigen::MatrixXd vg(Ntot, 3);
    Eigen::VectorXd vgproj(Ntot);
    vg.fill(0.0);

    // GATHER PHONON PROPERTIES FOR IRREDUCIBLE Q-POINTS

    double C_factor = 1e27 * alma::constants::kB / Ngridpoints / poscar.V;

    // scan over all points in the grid

    for (int nq = 0; nq < Ngridpoints; nq++) {
        // spectrum at point

        auto spectrum = grid.get_spectrum_at_q(nq);


        for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
            int imode = nq * Nbranches + nbranch - 3;

            if (imode < 0)
                continue;

            // store heat capacity
            C(imode) = C_factor *
                       alma::bose_einstein_kernel(spectrum.omega[nbranch], T);


            // store group velocity vector and projected one
            Eigen::Vector3d myvg = spectrum.vg.col(nbranch);
            vg(imode, 0) = myvg(0);
            vg(imode, 1) = myvg(1);
            vg(imode, 2) = myvg(2);

            vgproj(imode) = myvg.dot(uaxis);

            // store relaxation time
            double mytau =
                (w0(nbranch, nq) == 0.) ? 0. : (1. / w0(nbranch, nq));

            if (system_name == "nanowire") {
                mytau = scale_tau_nanowire(mytau, uaxis, myvg, limiting_length);
            }
            else if (system_name == "nanoribbon") {
                mytau =
                    scale_tau_nanoribbon(mytau, uaxis, myvg, limiting_length);
            }
            else {
                std::cout
                    << "ERROR: calc_kappa_nanos not recognized system_name"
                    << std::endl;
                exit(1);
            }

            tau(imode) = mytau;

            // store phonon frequency
            omega(imode) = spectrum.omega(nbranch);
        }
    }

    typedef std::array<int, 2> idx_pair;

    std::unordered_map<idx_pair, double> A_elements;
    A_elements.reserve(std::ceil(0.15 * Ntot * Ntot));

    // CALCULATE NON-RTA CONDUCTIVITY BY SOLVING LINEAR SYSTEM





    // build the list of Gamma values for 2-phonon processes
    for (auto& process_block : isotopic_processes) {
        auto I = process_block.first.first;
        auto J = process_block.first.second;
        A_elements[{static_cast<int>(I - 3), static_cast<int>(J - 3)}] -=
            process_block.second * tau(I - 3);
    } // done scanning over all 2-phonon processes

    for (auto& process_block : absorption_processes) {
        auto process(process_block.second);
        auto I = process.q[0] * Nbranches + process.alpha[0];
        auto J = process.q[1] * Nbranches + process.alpha[1];
        auto K = process.q[2] * Nbranches + process.alpha[2];

        double Gamma = process.compute_gamma(grid, T, true);

        A_elements[{static_cast<int>(I - 3), static_cast<int>(J - 3)}] +=
            Gamma * tau(I - 3);
        A_elements[{static_cast<int>(I - 3), static_cast<int>(K - 3)}] -=
            Gamma * tau(I - 3);
    }


    for (auto& process_block : emission_processes) {
        auto process(process_block.second);
        auto I = process.q[0] * Nbranches + process.alpha[0];
        auto J = process.q[1] * Nbranches + process.alpha[1];
        auto K = process.q[2] * Nbranches + process.alpha[2];

        double Gamma = process.compute_gamma(grid, T, true);

        A_elements[{static_cast<int>(I - 3), static_cast<int>(J - 3)}] -=
            0.5 * Gamma * tau(I - 3);
        A_elements[{static_cast<int>(I - 3), static_cast<int>(K - 3)}] -=
            0.5 * Gamma * tau(I - 3);
    }


    // diagonal elements in the system
    for (int diag_idx = 0; diag_idx < Ntot; diag_idx++) {
        A_elements[{diag_idx, diag_idx}] += 1.0;
    }


    /// Get tripletList
    std::vector<Eigen::Triplet<double>> tripletList;
    for (auto& [key, val] : A_elements) {
        auto idx1 = key[0];
        auto idx2 = key[1];
        tripletList.push_back(Eigen::Triplet<double>(idx1, idx2, val));
    }

    // build system of equations "A*H = B"

    Eigen::SparseMatrix<double> A(Ntot, Ntot);

    // Fill sparse matrix
    A.setFromTriplets(tripletList.begin(), tripletList.end());

    Eigen::VectorXd B(Ntot);

    // projected-components of omega*MFP_RTA over transport axis
    B = omega.array() * tau.array() * vgproj.array();

    // Transform to compressed format

    A.makeCompressed();

    std::cout << "#Linear system information\n";
    std::cout << "**A sparsity "
              << static_cast<double>(A.nonZeros()) / (Ntot * Ntot) << std::endl;
    std::cout.flush();

    // Scale system
    Eigen::IterScaling<Eigen::SparseMatrix<double>> scal;
    scal.computeRef(A);
    B = scal.LeftScaling().cwiseProduct(B);

    // SOLVE SYSTEM

    if (iterative) {
        Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
        // Eigen::GMRES<Eigen::SparseMatrix<double>> solver;
        solver.compute(A);

        // RTA solution to be used as initial guess
        Eigen::VectorXd H_guess(Ntot);

        H_guess = omega.array() * tau.array() * vgproj.array();

        H_guess = scal.RightScaling().cwiseInverse().cwiseProduct(H_guess);

        H = solver.solveWithGuess(B, H_guess);

        std::cout << "# Iterative solver information:\n" << std::endl;
        std::cout << "# -iterations:     " << solver.iterations() << std::endl;
        std::cout << "# -estimated error: " << solver.error() << std::endl;
    }

    else {
        H = omega.array() * tau.array() * vgproj.array();
    }

    double rel_err = (A * H - B).norm() / B.norm();

    if (rel_err > 1e-3) {
        std::cout << "alma::beyondRTA::calc_kappa_nanos > WARNING:"
                  << std::endl;
        std::cout << "solution of linear system might be unstable."
                  << std::endl;
        std::cout << "Relative error metric = " << rel_err << std::endl;
    }

    // Scale back the solution
    H = scal.RightScaling().cwiseProduct(H);

    // PROCESS THE LINEAR SYSTEM SOLUTION

    // construct "generalised MFPs"
    Eigen::VectorXd MFP_nonRTA(Ntot);
    MFP_nonRTA = H.array() / omega.array();

    // fix potential NaN/Inf problems
    for (int n = 0; n < Ntot; n++) {
        if (omega(n) <= 0.0) {
            MFP_nonRTA(n) = 0.0;
        }
    }

    // CALCULATE NON-RTA CONDUCTIVITY

    double kappa_nano = 0.0;
    for (int nq = 0; nq < Ngridpoints; nq++) {
        for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
            int imode = nq * Nbranches + nbranch - 3;

            if (imode < 0)
                continue;

            kappa_nano += 1e-6 * C(imode) * MFP_nonRTA(imode) * vgproj(imode);
        }
    }

    // RETURN RESULT
    return kappa_nano;

} // end of calc_kappa_nanos



} // namespace nanos
} // end of namespace alma
