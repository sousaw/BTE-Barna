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
/// Definitions corresponding to collision_operator.hpp.

#include <collision_operator.hpp>
#include <constants.hpp>
#include <limits>
#include <iostream>
#include <fstream>
#include <unsupported/Eigen/MatrixFunctions>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>
#if BOOST_VERSION >= 106700
#include <boost/container_hash/hash.hpp>
#else
#include <boost/functional/hash.hpp>
#endif 
/// Threading through Intel TBB
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>
#include <mutex>

namespace alma {

Eigen::MatrixXd get_collision_operator_dense(
    const alma::Gamma_grid& grid,
    const alma::Crystal_structure& cell,
    std::vector<alma::Thirdorder_ifcs>& anhIFC,
    double Treference,
    boost::mpi::communicator& world) {
    /// First we are getting sizes
    std::size_t nqpoints = grid.nqpoints;
    std::size_t nbands = grid.get_spectrum_at_q(0).omega.size();
    std::size_t nmodes = nbands * nqpoints;

    std::size_t my_id = world.rank();
    std::size_t nprocs = world.size();

    /// Fill B matrix with 0
    if ((nmodes - 3) * (nmodes - 3) > std::numeric_limits<int>::max()) {
        if (my_id == 0) {
            std::cerr << "FATAL ERROR the number of matrix elements will not "
                         "fit in MPI buffer"
                      << std::endl;
            std::cerr << "max_buffer_elements = "
                      << std::numeric_limits<int>::max() << std::endl;
            std::cerr << "matrix_elements     = " << (nmodes - 3) * (nmodes - 3)
                      << std::endl;
        }
        world.barrier();
        world.abort(1);
    }

    Eigen::MatrixXcd cB = Eigen::MatrixXcd::Zero(
        nmodes - 3,
        nmodes - 3); /// We are not counting the Gamma-point acustic bands.

    /// We first calculate the equilibrium distribution for reference
    /// temperature:
    std::vector<double> neq(nmodes, 0.);

    for (std::size_t iq = 0; iq < nqpoints; iq++) {
        auto s = grid.get_spectrum_at_q(iq);
        for (std::size_t ib = 0; ib < nbands; ib++) {
            double omega = s.omega(ib);
            if (omega >= 1.0e-6) {
                neq[iq * nbands + ib] = alma::bose_einstein(omega, Treference);
            }
        }
    }


    /// Create a unique set of Three_phonon processes from irreductible triplets
    /// including the symmetry:
    auto limits = alma::my_jobs(grid.get_nequivalences(), nprocs, my_id);
    std::array<int, 2> signs(
        {{static_cast<int>(alma::threeph_type::emission),
          static_cast<int>(alma::threeph_type::absorption)}});

    std::vector<alma::Threeph_process> myprocesses;

    std::unordered_map<std::array<std::size_t, 3>, alma::Threeph_process>
        emission_processes;
    std::unordered_map<std::array<std::size_t, 3>, alma::Threeph_process>
        absorption_processes;


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
                                                1.0 *
                                                std::sqrt(
                                                    boost::math::pow<2>(
                                                        grid.base_sigma(v1)) +
                                                    boost::math::pow<2>(
                                                        grid.base_sigma(v2)) +
                                                    boost::math::pow<2>(
                                                        grid.base_sigma(v3)));
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
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet1
                                                        .compute_gaussian());
                                            }
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
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet2
                                                        .compute_gaussian());
                                            }
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
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet3
                                                        .compute_gaussian());
                                            }
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
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet1
                                                        .compute_gaussian());
                                            }
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
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet2
                                                        .compute_gaussian());
                                            }
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
                                                vpsXdelta.push_back(
                                                    Gamma_of_triplet3
                                                        .compute_gaussian());
                                            }
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
                                        cell, grid, anhIFC);

                                    double SymGamma = 0.;
                                    for (auto& vD : vpsXdelta) {
                                        SymGamma += VP2 * vD;
                                    }

                                    SymGamma /= 4 * eqqtrip.size();


                                    /// If not 0 all register in map, we account
                                    /// for a state decaying into two of same
                                    /// index
                                    if (!almost_equal(SymGamma,
                                        0.,1.0e-12,1.0e-9)) {
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
                                                if (absorption_processes.count(
                                                        triplet_indexes) == 0) {
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
                                                if (emission_processes_tbb.count(
                                                        triplet_indexes) == 0) {
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

    /// Passing tbb map to stl map for more classic stuff like MPI:
    /// It can be avoided if serialization of tbb container is achieved
    absorption_processes.insert(absorption_processes_tbb.begin(),
                                absorption_processes_tbb.end());
    emission_processes.insert(emission_processes_tbb.begin(),
                              emission_processes_tbb.end());

    /// We are now working on isotopic/mass-disoreder scattering
    if (my_id == 0)
        std::cout << "#Working on 2ph processes" << std::endl;

    std::unordered_map<std::pair<std::size_t, std::size_t>, double>
        isotopic_processes;

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
        if (!almost_equal(gamma,0.,1.0e-12,1.0e-9)) {
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


    /// Sync section
    if (nprocs > 1) {
        if (my_id == 0)
            std::cout << "#Element distribution along processes" << std::endl;
        /// Chunking map
        std::vector<std::unordered_map<std::array<std::size_t, 3>,
                                       alma::Threeph_process>>
            chunked_map_emission(nprocs);
        std::vector<std::unordered_map<std::array<std::size_t, 3>,
                                       alma::Threeph_process>>
            chunked_map_absorption(nprocs);
        std::vector<
            std::unordered_map<std::pair<std::size_t, std::size_t>, double>>
            chunked_map_isotopic(nprocs);
        for (auto& process_block : emission_processes) {
            auto id1 = process_block.first[0];
            /// Check where we want this
            auto mode_proc_id = alma::index_proc(id1, nmodes, nprocs);
            chunked_map_emission[mode_proc_id].insert(process_block);
        }
        for (auto& process_block : absorption_processes) {
            auto id1 = process_block.first[0];
            /// Check where we want this
            auto mode_proc_id = alma::index_proc(id1, nmodes, nprocs);
            chunked_map_absorption[mode_proc_id].insert(process_block);
        }
        for (auto& process_block : isotopic_processes) {
            auto id1 = process_block.first.first;
            /// Check where we want this
            auto mode_proc_id = alma::index_proc(id1, nmodes, nprocs);
            chunked_map_isotopic[mode_proc_id].insert(process_block);
        }


        /// Here we distributing the chunks (there is no real balance, but for a
        /// big mesh it is expected to be quite even)
        for (std::size_t iproc = 0; iproc < nprocs; iproc++) {
            std::vector<std::unordered_map<std::array<std::size_t, 3>,
                                           alma::Threeph_process>>
                collector_abs, collector_emi;
            std::vector<
                std::unordered_map<std::pair<std::size_t, std::size_t>, double>>
                collector_iso;
            /// Gathering the chunk of process in collector
            gather(world, chunked_map_absorption[iproc], collector_abs, iproc);
            gather(world, chunked_map_emission[iproc], collector_emi, iproc);
            gather(world, chunked_map_isotopic[iproc], collector_iso, iproc);
            /// If a am the collector put in my map
            if (my_id == iproc) {
                absorption_processes.clear();
                for (auto& map2insert : collector_abs) {
                    absorption_processes.insert(map2insert.begin(),
                                                map2insert.end());
                }
                emission_processes.clear();
                for (auto& map2insert : collector_emi) {
                    emission_processes.insert(map2insert.begin(),
                                              map2insert.end());
                }
                isotopic_processes.clear();
                for (auto& map2insert : collector_iso) {
                    isotopic_processes.insert(map2insert.begin(),
                                              map2insert.end());
                }
            }
            world.barrier();
        }

        if (my_id == 0)
            std::cout << "#Element distribution along processes=>DONE"
                      << std::endl;
    }
    std::size_t proc_nemi = emission_processes.size();
    std::size_t proc_nabs = absorption_processes.size();
    std::size_t proc_niso = isotopic_processes.size();
    std::size_t tnemi = 0;
    std::size_t tnabs = 0;
    std::size_t tniso = 0;

    boost::mpi::all_reduce(
        world, &proc_nemi, 1, &tnemi, std::plus<std::size_t>());
    boost::mpi::all_reduce(
        world, &proc_nabs, 1, &tnabs, std::plus<std::size_t>());
    boost::mpi::all_reduce(
        world, &proc_niso, 1, &tniso, std::plus<std::size_t>());

    if (my_id == 0) {
        std::cout << "#Nemi " << tnemi << std::endl;
        std::cout << "#Nabs " << tnabs << std::endl;
        std::cout << "#Niso " << tniso << std::endl;
    }
    std::size_t ntot = proc_nemi + proc_nabs + proc_niso;

    std::size_t iel = 0;


    for (auto& process_block : emission_processes) {
        auto process(process_block.second);
        auto q0 = process.q[0];
        auto q1 = process.q[1];
        auto q2 = process.q[2];
        auto b0 = process.alpha[0];
        auto b1 = process.alpha[1];
        auto b2 = process.alpha[2];
        double omega0 = grid.get_spectrum_at_q(q0).omega(b0);
        double omega1 = grid.get_spectrum_at_q(q1).omega(b1);
        double omega2 = grid.get_spectrum_at_q(q2).omega(b2);


        double factor = 1.0e6 * alma::constants::hbar * alma::constants::pi /
                        (4.0 * omega0 * omega1 * omega2) / nqpoints;

        double vpdelta = factor * process.get_vp2();

        if (process.type == alma::threeph_type::emission)
            vpdelta /= 2;

        if (process.type == alma::threeph_type::emission) {
            auto I = q0 * nbands + b0;
            auto J = q1 * nbands + b1;
            auto K = q2 * nbands + b2;
            /// Check if normal

            cB(I - 3, I - 3) = NeumaierSum(cB(I - 3, I - 3),
                                           -vpdelta * (neq[K] + neq[J] + 1.0));
            cB(I - 3, J - 3) =
                NeumaierSum(cB(I - 3, J - 3),
                            (omega0 / omega1) * vpdelta * (neq[K] - neq[I]));
            cB(I - 3, K - 3) =
                NeumaierSum(cB(I - 3, K - 3),
                            (omega0 / omega2) * vpdelta * (neq[J] - neq[I]));
        }
        else {
            std::cerr << "#Error building collision operator this process "
                         "should not be in this list"
                      << std::endl;
            world.abort(1);
        }

        iel++;
        if (my_id == 0) {
            std::cout << "#iel => " << iel << " / " << ntot << std::endl;
        }
    }


    for (auto& process_block : absorption_processes) {
        auto process(process_block.second);
        auto q0 = process.q[0];
        auto q1 = process.q[1];
        auto q2 = process.q[2];
        auto b0 = process.alpha[0];
        auto b1 = process.alpha[1];
        auto b2 = process.alpha[2];
        double omega0 = grid.get_spectrum_at_q(q0).omega(b0);
        double omega1 = grid.get_spectrum_at_q(q1).omega(b1);
        double omega2 = grid.get_spectrum_at_q(q2).omega(b2);

        double factor = 1.0e6 * alma::constants::hbar * alma::constants::pi /
                        (4.0 * omega0 * omega1 * omega2) / nqpoints;

        double vpdelta = factor * process.get_vp2();

        if (process.type == alma::threeph_type::absorption) {
            auto I = q0 * nbands + b0;
            auto J = q1 * nbands + b1;
            auto K = q2 * nbands + b2;

            cB(I - 3, I - 3) =
                NeumaierSum(cB(I - 3, I - 3), vpdelta * (neq[K] - neq[J]));
            cB(I - 3, J - 3) =
                NeumaierSum(cB(I - 3, J - 3),
                            omega0 / omega1 * vpdelta * (neq[K] - neq[I]));
            cB(I - 3, K - 3) = NeumaierSum(cB(I - 3, K - 3),
                                           omega0 / omega2 * vpdelta *
                                               (neq[J] + neq[I] + 1.0));
        }
        else {
            std::cerr << "#Error building collision operator this process "
                         "should not be in this list"
                      << std::endl;
            world.abort(1);
        }


        iel++;
        if (my_id == 0) {
            std::cout << "#iel => " << iel << " / " << ntot << std::endl;
        }
    }


    for (auto& process_block : isotopic_processes) {
        auto I = process_block.first.first;
        auto J = process_block.first.second;
        auto q0 = I / nbands;
        auto q1 = J / nbands;
        auto b0 = I % nbands;
        auto b1 = J % nbands;
        double omega0 = grid.get_spectrum_at_q(q0).omega(b0);
        double omega1 = grid.get_spectrum_at_q(q1).omega(b1);
        /// It goes with Umklapp as they do not conserve the momentum
        cB(I - 3, I - 3) = NeumaierSum(cB(I - 3, I - 3), -process_block.second);
        cB(I - 3, J - 3) = NeumaierSum(
            cB(I - 3, J - 3), (omega0 / omega1) * process_block.second);

        iel++;
        if (my_id == 0) {
            std::cout << "#iel => " << iel << " / " << ntot << std::endl;
        }
    }


    /// Now we collapse the matrix in root and broadcast to other procs:
    /// This is not RAM friendly (from memory point of view it will be cheaper
    /// to reduce by element or chunk of elements)
    Eigen::MatrixXd B(nmodes-3,nmodes-3);
    ///We collapse from procs and we 
    ///eval the Neumaier Summations
    if (nprocs == 1 ) {
        for (decltype(nmodes) J=0; J < nmodes-3; J++) {
            for (decltype(nmodes) I=0; I < nmodes-3; I++) {
                B(I,J) = 
                    evalNeumaierSum(cB(I,J));
            }
        }
    }
    else {
        for (decltype(nmodes) J = 0; J < nmodes - 3; J++) {
            for (decltype(nmodes) I = 0; I < nmodes - 3; I++) {
                std::vector<std::complex<double>> v;

                gather(world, cB(I, J), v, 0);

                /// Eval Neumaier
                if (my_id == 0) {
                    B(I, J) = combineNeumaierSums(v);
                }

                broadcast(world, B(I, J), 0);
            }
        }
    }
    
    
    ///Enforcing energy conservation
    for (std::size_t J = 0; J < nmodes - 3; J++) {
        /// Getting the mean of energy that we need to add back to enforce
        /// energy conservation column-major ordering makes those column
        /// operations efficient
        Eigen::VectorXd Bcol = B.col(J);
        
        auto deltaj_mean = -(NeumaierSum(Bcol.data(),
                         Bcol.rows())) / (nmodes - 3);
        /// We are now adding deltaj_mean to counter the defficit in
        /// conservation
        for (std::size_t I = 0; I < nmodes - 3; I++)
            B(I, J) += deltaj_mean;
        // Making sanity check:
        
        // Making sanity check:
        if (!alma::almost_equal(B.col(J).sum(),0.)) {
            std::cerr << "Error in enforcing energy"
                      << std::endl;
            world.abort(1);
        }
    }
    /// We return B, the user should take care and use std::move as it is
    /// cheaper than copying.

    return B;
}


void Arnoldi_algorithm(const Eigen::MatrixXd& A,
                       const Eigen::VectorXd& b,
                       int n,
                       Eigen::MatrixXd& B,
                       Eigen::MatrixXd& H,
                       double t) {
    // Init some variables
    int m = A.rows();
    H.resize(n, n);
    H.setZero();
    B.resize(m, n);
    B.setZero();
    Eigen::VectorXd q = b / b.norm(); // Normalize the input vector
    B.col(0) =
        q; // Use the normalized input vector as first Krylov subspace basis

    for (int k = 0; k < n; k++) {
        Eigen::VectorXd z = t * A * q;
        for (int j = 0; j < k;
             j++) { // We are substracting the projections from previous vectors
            H(j, k) = B.col(j).transpose() * z;
            z -= H(j, k) * B.col(j);
        }
        
	/// Break Arnoldi's, it is complete
        if (almost_equal(z.norm(),
            0.,1.0e-12,1.0e-9)) {
            break;
        }
        
        if (k + 1 < n) {
            H(k + 1, k) = z.norm();
            q = z / H(k + 1, k); // Getting new vector
            B.col(k + 1) = q;
        }
    }
}


propagator_H::propagator_H() {
}

propagator_H::propagator_H(Eigen::VectorXd& b) {
    std::size_t data_size = b.size();
    (this->signed_cumm).resize(data_size);

    this->pcol_abssum = b.array().abs().sum();

    double accumulator = 0.;
    for (std::size_t i = 0; i < data_size; i++) {
        accumulator += std::abs(b(i));
        (this->signed_cumm)(i) = (b(i) < 0.)
                                     ? -accumulator / (this->pcol_abssum)
                                     : accumulator / (this->pcol_abssum);
    }
}

propagator_H::propagator_H(const propagator_H& B) :
        signed_cumm(B.signed_cumm) , pcol_abssum(B.pcol_abssum) {
}

propagator_H::propagator_H(propagator_H&& B) : 
        signed_cumm(std::move(B.signed_cumm)) , 
        pcol_abssum(std::move(B.pcol_abssum)) {
}

propagator_H& propagator_H::operator=(const propagator_H& B) {
    this->signed_cumm = B.signed_cumm;
    this->pcol_abssum = B.pcol_abssum;
    return *this;
}

bool propagator_H::operator<(const propagator_H& B) const {
    return this->pcol_abssum < B.pcol_abssum;
};


std::pair<int, std::size_t> propagator_H::get(double R) const {
    auto im = this->lower_bound(R);

    if ((this->signed_cumm)(im) > 0.)
        return std::make_pair(1, im + 3);
    else
        return std::make_pair(-1, im + 3);

    std::cerr << "Error in selection" << std::endl;
    exit(EXIT_FAILURE);
    return std::make_pair(0, 0);
}

Eigen::VectorXd propagator_H::get_signed_cumm() const {
    return this->signed_cumm;
}

std::size_t propagator_H::byte_size() const {
    return sizeof(double) * (this->signed_cumm).size() +
           sizeof(Eigen::VectorXd) + sizeof(double);
}

std::size_t propagator_H::size() const {
    return (this->signed_cumm).size();
}

alma::propagator_H lirp(alma::propagator_H& H1,
                        alma::propagator_H& H2,
                        double T1,
                        double T2,
                        double Tx) {
    Eigen::VectorXd Pint =
        H1.signed_cumm +
        (Tx - T1) * (H2.signed_cumm - H1.signed_cumm) / (T2 - T1);
    double absint = H1.pcol_abssum +
                    (Tx - T1) * (H2.pcol_abssum - H1.pcol_abssum) / (T2 - T1);
    return alma::propagator_H(Pint, absint);
}

void save_P(std::string fname,
            Eigen::MatrixXd& data,
            boost::mpi::communicator& world) {
    std::size_t master = 0;
    std::size_t my_id = world.rank();
    std::size_t nprocs = world.size();
    if (nprocs > 1 and data.rows()!=data.cols()) {
        for (std::size_t iproc = 0; iproc < nprocs; iproc++) {
            if (my_id == iproc) {
                if (my_id == master) {
                    Eigen::MatrixXd::Index nmodes = data.rows();
                    auto flags =
                        std::ios::out | std::ios::binary | std::ios::trunc;
                    std::ofstream out((fname).c_str(), flags);
                    out.write((char*)(&nmodes), sizeof(Eigen::MatrixXd::Index));
                    out.write((char*)data.data(), data.size() * sizeof(double));
                    out.close();
                }
                else {
                    /// Other nodes open but not erase the info
                    auto flags =
                        std::ios::out | std::ios::binary | std::ios::app;
                    std::ofstream out((fname).c_str(), flags);
                    out.write((char*)data.data(), data.size() * sizeof(double));
                    out.close();
                }
            }
            world.barrier();
        }
    }
    else {
        Eigen::MatrixXd::Index nmodes = data.rows();
        auto flags = std::ios::out | std::ios::binary | std::ios::trunc;
        std::ofstream out((fname).c_str(), flags);
        out.write((char*)(&nmodes), sizeof(Eigen::MatrixXd::Index));
        out.write((char*)data.data(), data.size() * sizeof(double));
        out.close();
    }
}


void load_P(std::string fname,
            Eigen::MatrixXd& pmat,
            boost::mpi::communicator& world) {
    std::size_t master = 0;
    std::size_t my_id = world.rank();
    std::size_t nprocs = world.size();

    std::vector<Eigen::MatrixXd> pmat_procs(nprocs);

    /// Read binary to memory only in master
    Eigen::MatrixXd::Index nmodes = 0;

    /// It might be
    if (my_id == master) {
        auto flags = std::ios::in | std::ios::binary;
        std::ifstream in((fname).c_str(), flags);
        in.read((char*)(&nmodes), sizeof(Eigen::MatrixXd::Index));
        /// Scale propagator (not the most efficient way)
        if (nprocs > 1) {
            for (std::size_t iproc = 0; iproc < nprocs; iproc++) {
                auto limits = alma::my_jobs(nmodes, nprocs, iproc);
                // If node do not contain info simply ignore
                if (limits[1] - limits[0] == 0)
                    continue;
                auto ncols = limits[1] - limits[0];
                /// Master is special case
                pmat_procs[iproc].resize(nmodes, ncols);
                in.read((char*)pmat_procs[iproc].data(),
                        nmodes * ncols * sizeof(double));
            }
        }
        else {
            pmat.resize(nmodes, nmodes);
            in.read((char*)pmat.data(), nmodes * nmodes * sizeof(double));
        }
        in.close();
    }

    // If multiple procs scatter info
    if (nprocs > 1)
        scatter(world, pmat_procs, pmat, 0.);
}

void build_P(Eigen::MatrixXd& A,
             Eigen::MatrixXd& P,
             double t,
             boost::mpi::communicator& world,
             double eps) {
    /// Check for optimal Kryslov subspace
    int ksize = 2;

    while (true) {
        /// Check that the error per element is lower than the eps
        if (error_bound(A, ksize, t) / A.size() < eps) {
            break;
        }
        ksize++;
    }

    /// Now we will break the problem by MPI processes, taking each of them care
    /// of a column.
    auto limits = alma::my_jobs(A.cols(), world.size(), world.rank());

    auto proc_cols = limits[1] - limits[0];

    if (proc_cols != 0)
        P.resize(A.cols(), proc_cols);

    std::mutex coutmutex;
    std::size_t colcount = 0;
    /// TBB parallelism combined with boost mpi (use MPI for diferent nodes and
    /// TBB inside the same node)
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(limits[0], limits[1]),
        [&](tbb::blocked_range<std::size_t> colrange) {
            for (auto ii = colrange.begin(); ii < colrange.end(); ii++) {
                coutmutex.lock();
                if (world.rank()==0)
                	std::cout << "#Exp " << colcount << "/" << A.cols()
                          << std::endl;
                colcount++;
                coutmutex.unlock();
                Eigen::MatrixXd B, H;
                Eigen::VectorXd b(A.cols());
                b.setZero();
                b(ii) = 1.0;
                alma::Arnoldi_algorithm(A, b, ksize, B, H, t);
                P.col(ii - limits[0]) =
                    (B * (H.exp())).col(0); /// This is the Krylov approach to
                                            /// the exponential
            }
        });
}


void build_histogram(Eigen::MatrixXd& A,
                     std::unordered_map<std::size_t, propagator_H>& modes_histo,
                     double t,
                     boost::mpi::communicator& world,
                     double eps,
                     bool print_info) {
    /// Check for optimal Kryslov subspace
    int ksize = 2;

    while (true) {
        /// Check that the error per element is lower than the eps
        if (error_bound(A, ksize, t) / A.size() < eps) {
            break;
        }
        ksize++;
    }

    /// Now we will break the problem by MPI processes, taking each of them care
    /// of a column.
    auto limits = alma::my_jobs(A.cols(), world.size(), world.rank());

    std::size_t ram_size = 0, nel = 0;

    for (auto i = limits[0]; i < limits[1]; i++) {
        Eigen::MatrixXd B, H;
        Eigen::VectorXd b(A.cols());
        b.setZero();
        b(i) = 1.0;
        alma::Arnoldi_algorithm(A, b, ksize, B, H, t);
        b = (B * (H.exp()))
                .col(0); /// This is the Krylov approach to the exponential
        modes_histo[i + 3] = propagator_H(b);
        ram_size += modes_histo[i + 3].byte_size();
        nel += modes_histo[i + 3].size();
    }

    if (print_info) {
        std::cout << "the size in RAM " << ram_size << " bytes" << std::endl;
        std::cout << "non null elements count " << nel << " / " << A.size()
                  << std::endl;
        std::cout << "density " << (double)nel / (double)A.size() * 100.0
                  << " %" << std::endl;
    }
}


} // namespace alma
