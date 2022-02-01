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
/// Definitions corresponding to processes.hpp.

#include <cmath>
#include <complex>
#include <constants.hpp>
#include <periodic_table.hpp>
#include <utilities.hpp>
#include <processes.hpp>
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>

namespace alma {
std::vector<Threeph_process> find_allowed_threeph(
    const Gamma_grid& grid,
    const boost::mpi::communicator& communicator,
    double scalebroad) {
    std::array<int, 2> signs({{static_cast<int>(threeph_type::emission),
                               static_cast<int>(threeph_type::absorption)}});
    auto nprocs = communicator.size();
    auto my_id = communicator.rank();
    auto limits = my_jobs(grid.get_nequivalences(), nprocs, my_id);

    std::size_t nmodes = grid.get_spectrum_at_q(0).omega.size();

    // The nested for loops look unwieldy, but are in fact simple
    // to interpret. iq1 runs over an irreducible set of q points,
    // iq2 runs over all q points and iq3 is computed according to
    // the conservation of momentum. Then we loop over all modes at
    // these three q points to look for triplets that also satisfy
    // the conservation of energy.
    tbb::concurrent_vector<Threeph_process> concurrent_nruter;


    // for (auto ic = limits[0]; ic < limits[1]; ++ic) {
    tbb::parallel_for(
        tbb::blocked_range<std::size_t>(limits[0], limits[1]),
        [&](tbb::blocked_range<std::size_t> icrange) {
            for (auto ic = icrange.begin(); ic < icrange.end(); ++ic) {
                auto iq1 = grid.get_representative(ic);
                auto coords1 = grid.one_to_three(iq1);
                auto spectrum1 = grid.get_spectrum_at_q(iq1);

                for (std::size_t iq2 = 0; iq2 < grid.nqpoints; ++iq2) {
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

                        for (decltype(nmodes) im1 = 0; im1 < nmodes; ++im1) {
                            // Modes with omega exactly equal to zero
                            // are not taken into account. Hence, it is
                            // important to call enforce_asr() on the grid
                            // object before looking for three-phonon
                            // processes.
                            if (spectrum1.omega(im1) == 0.)
                                continue;

                            for (decltype(nmodes) im2 = 0; im2 < nmodes;
                                 ++im2) {
                                if (spectrum2.omega(im2) == 0.)
                                    continue;

                                for (decltype(nmodes) im3 = 0; im3 < nmodes;
                                     ++im3) {
                                    if (spectrum3.omega(im3) == 0.)
                                        continue;
                                    auto sigma =
                                        scalebroad *
                                        std::sqrt(
                                            boost::math::pow<2>(grid.base_sigma(
                                                spectrum1.vg.col(im1))) +
                                            boost::math::pow<2>(grid.base_sigma(
                                                spectrum2.vg.col(im2))) +
                                            boost::math::pow<2>(grid.base_sigma(
                                                spectrum3.vg.col(im3))));

                                    auto delta =
                                        std::fabs(spectrum1.omega(im1) +
                                                  s * spectrum2.omega(im2) -
                                                  spectrum3.omega(im3));

                                    if (delta <= constants::nsigma * sigma)
                                        concurrent_nruter.emplace_back(
                                            Threeph_process(
                                                ic,
                                                std::array<std::size_t, 3>(
                                                    {{iq1, iq2, iq3}}),
                                                std::array<std::size_t, 3>(
                                                    {{im1, im2, im3}}),
                                                static_cast<threeph_type>(s),
                                                delta,
                                                sigma));
                                }
                            }
                        }
                    }
                }
            }
        });

    /// Passing to stl version
    std::vector<Threeph_process> nruter;
    nruter.reserve(concurrent_nruter.size());
    std::copy(concurrent_nruter.begin(),
              concurrent_nruter.end(),
              std::back_inserter(nruter));

    return nruter;
}


double Threeph_process::compute_weighted_gaussian(const Gamma_grid& grid,
                                                  double T) {
    auto g = this->compute_gaussian();
    auto s = static_cast<int>(this->type);
    auto sp1 = grid.get_spectrum_at_q(this->q[0]);
    auto sp2 = grid.get_spectrum_at_q(this->q[1]);
    auto sp3 = grid.get_spectrum_at_q(this->q[2]);
    auto fBE2 = bose_einstein(sp2.omega[this->alpha[1]], T);
    auto fBE3 = bose_einstein(sp3.omega[this->alpha[2]], T);

    if (!this->gaussian_computed) {
        auto distr = boost::math::normal(0., this->sigma);
        this->gaussian = boost::math::pdf(distr, this->domega);
        this->gaussian_computed = true;
    }
    return (fBE2 - s * fBE3 + (1 - s) / 2) * g / sp1.omega[this->alpha[0]] /
           sp2.omega[this->alpha[1]] / sp3.omega[this->alpha[2]] /
           static_cast<double>(1 + (1 - s) / 2);
    ;
}


double Threeph_process::compute_vp2(
    const Crystal_structure& cell,
    const Gamma_grid& grid,
    const std::vector<Thirdorder_ifcs>& thirdorder) {
    // Prefactor used to convert the result to THz^4 / (kg nm^2)
    constexpr double unitfactor = 1e-6 * constants::e * constants::e /
                                  constants::amu / constants::amu /
                                  constants::amu;
    auto s = static_cast<int>(this->type);
    // Set up those variables that depend only on the q point.
    Eigen::VectorXd q2 = grid.get_q(this->q[1]);
    Eigen::VectorXd q3 = grid.get_q(this->q[2]);
    Eigen::VectorXcd wf1 =
        grid.get_spectrum_at_q(this->q[0]).wfs.col(this->alpha[0]);
    Eigen::VectorXcd wf2 =
        grid.get_spectrum_at_q(this->q[1]).wfs.col(this->alpha[1]);

    // This conditional conjugation takes care of the fact
    // that the second phonon appears on different sides of
    // the equations for emission and absorption processes.
    if (this->type == threeph_type::emission)
        wf2 = wf2.conjugate();
    Eigen::VectorXcd wf3 =
        grid.get_spectrum_at_q(this->q[2]).wfs.col(this->alpha[2]).conjugate();
    std::complex<double> vp = 0.;

    // Iterate over triplets to build the matrix element.
    for (auto& tri : thirdorder) {
        double massfactor = std::sqrt(
            cell.get_mass(tri.i) * cell.get_mass(tri.j) * cell.get_mass(tri.k));
        double arg2 = s * q2.dot(tri.rj);
        double arg3 = q3.dot(tri.rk);
        std::complex<double> prefactor =
            (std::cos(arg2) + constants::imud * std::sin(arg2)) *
            (std::cos(arg3) - constants::imud * std::sin(arg3)) / massfactor;
        std::complex<double> contr = 0.;

        for (auto rr = 0; rr < 3; ++rr)
            for (auto ss = 0; ss < 3; ++ss)
                for (auto tt = 0; tt < 3; ++tt)
                    contr += wf1(tt + 3 * tri.i) * wf2(ss + 3 * tri.j) *
                             wf3(rr + 3 * tri.k) * tri.ifc(tt, ss, rr);
        vp += prefactor * contr;
    }
    // And store only its modulus squared.
    this->vp2 = unitfactor * std::norm(vp);
    this->vp2_computed = true;
    return this->vp2;
}


double Threeph_process::compute_gamma(const Gamma_grid& grid,
                                      double T,
                                      bool compact) {
    auto s = static_cast<int>(this->type);
    auto gaussian = compact ? 1.0 : this->compute_gaussian();
    auto vp2 = this->get_vp2();
    auto sp1 = grid.get_spectrum_at_q(this->q[0]);
    auto sp2 = grid.get_spectrum_at_q(this->q[1]);
    auto sp3 = grid.get_spectrum_at_q(this->q[2]);
    auto fBE2 = bose_einstein(sp2.omega[this->alpha[1]], T);
    auto fBE3 = bose_einstein(sp3.omega[this->alpha[2]], T);
    const double prefactor = 1e6 * constants::hbar * constants::pi / 4.0;
    auto nruter = prefactor * (fBE2 - s * fBE3 + (1 - s) / 2) * gaussian * vp2 /
                  grid.nqpoints / sp1.omega[this->alpha[0]] /
                  sp2.omega[this->alpha[1]] / sp3.omega[this->alpha[2]];

    return nruter;
}

double Threeph_process::compute_gamma_reduced(const Gamma_grid& grid,
                                              double T) {
    auto gaussian = this->compute_gaussian();
    auto vp2 = this->get_vp2();
    auto sp1 = grid.get_spectrum_at_q(this->q[0]);
    auto sp2 = grid.get_spectrum_at_q(this->q[1]);
    auto sp3 = grid.get_spectrum_at_q(this->q[2]);
    const double prefactor = 1e6 * constants::hbar * constants::pi / 4.0;
    auto nruter = prefactor * gaussian * vp2 / grid.nqpoints /
                  sp1.omega[this->alpha[0]] / sp2.omega[this->alpha[1]] /
                  sp3.omega[this->alpha[2]];

    return nruter;
}

Eigen::ArrayXd Threeph_process::compute_collision(const Gamma_grid& grid,
                                                  const Eigen::ArrayXXd& n0) {
    auto s = static_cast<int>(this->type);
    auto gaussian = this->compute_gaussian();
    auto vp2 = this->get_vp2();
    auto sp1 = grid.get_spectrum_at_q(this->q[0]);
    auto sp2 = grid.get_spectrum_at_q(this->q[1]);
    auto sp3 = grid.get_spectrum_at_q(this->q[2]);
    double fBE1 = n0(this->alpha[0], this->q[0]);
    double fBE2 = n0(this->alpha[1], this->q[1]);
    double fBE3 = n0(this->alpha[2], this->q[2]);
    const double prefactor = 1e6 * constants::hbar * constants::pi / 4.0;
    Eigen::ArrayXd nruter(3);

    nruter.fill(prefactor * gaussian * vp2 * 2. / (3. - s) / grid.nqpoints /
                sp1.omega[this->alpha[0]] / sp2.omega[this->alpha[1]] /
                sp3.omega[this->alpha[2]]);
    // The expressions involved in this calculation are pretty
    // similar to the one in compute_gamma, but the BE distributions
    // involved are different in each case.
    nruter(0) *= s * (fBE3 - s * fBE2 + (1 + s) / 2);
    nruter(1) *= fBE3 - fBE1;
    nruter(2) *= fBE2 + s * fBE1 + (1 - s) / 2;
    return nruter;
}


Eigen::ArrayXXd calc_w0_threeph(const alma::Gamma_grid& grid,
                                std::vector<alma::Threeph_process>& processes,
                                double T,
                                const boost::mpi::communicator& comm) {
    auto nqpoints = grid.nqpoints;
    auto nmodes = grid.get_spectrum_at_q(0).omega.size();

    // Compute all contributions belonging to the current
    // MPI process.
    Eigen::ArrayXXd my_w0(nmodes, nqpoints);

    my_w0.fill(0.);

    for (std::size_t i = 0; i < processes.size(); ++i) {
        auto factor = 2. / (3. - static_cast<int>(processes[i].type));
        auto gamma = processes[i].compute_gamma(grid, T);
        my_w0(processes[i].alpha[0], processes[i].q[0]) += factor * gamma;
    }
    // Extend the calculation to all q points.
    auto nequivalences = grid.get_nequivalences();

    for (decltype(nequivalences) i = 0; i < nequivalences; ++i) {
        auto eq = grid.get_equivalence(i);

        for (std::size_t iq = 1; iq < eq.size(); ++iq)
            my_w0.col(eq[iq]) = my_w0.col(eq[0]);
    }
    // Sum over processors to obtain the total scattering rate for
    // each mode.
    Eigen::ArrayXXd nruter(nmodes, nqpoints);
    nruter.fill(0.);
    boost::mpi::reduce(comm,
                       my_w0.data(),
                       my_w0.size(),
                       nruter.data(),
                       std::plus<double>(),
                       0);
    // Share the information among MPI processes.
    boost::mpi::broadcast(comm, nruter.data(), nruter.size(), 0);
    return nruter;
}

Eigen::ArrayXXd calc_w0_threeph(
    const alma::Gamma_grid& grid,
    std::vector<alma::Threeph_process>& processes,
    double T,
    std::function<bool(const Threeph_process&)> filter,
    const boost::mpi::communicator& comm) {
    auto nqpoints = grid.nqpoints;
    auto nmodes = grid.get_spectrum_at_q(0).omega.size();

    // Compute all contributions belonging to the current
    // MPI process.
    Eigen::ArrayXXd my_w0(nmodes, nqpoints);

    my_w0.fill(0.);

    for (std::size_t i = 0; i < processes.size(); ++i) {
        if (!filter(processes[i])) {
            continue;
        }
        auto factor = 2. / (3. - static_cast<int>(processes[i].type));
        auto gamma = processes[i].compute_gamma(grid, T);
        my_w0(processes[i].alpha[0], processes[i].q[0]) += factor * gamma;
    }
    // Extend the calculation to all q points.
    auto nequivalences = grid.get_nequivalences();

    for (decltype(nequivalences) i = 0; i < nequivalences; ++i) {
        auto eq = grid.get_equivalence(i);

        for (std::size_t iq = 1; iq < eq.size(); ++iq)
            my_w0.col(eq[iq]) = my_w0.col(eq[0]);
    }
    // Sum over processors to obtain the total scattering rate for
    // each mode.
    Eigen::ArrayXXd nruter(nmodes, nqpoints);
    nruter.fill(0.);
    boost::mpi::reduce(comm,
                       my_w0.data(),
                       my_w0.size(),
                       nruter.data(),
                       std::plus<double>(),
                       0);
    // Share the information among MPI processes.
    boost::mpi::broadcast(comm, nruter.data(), nruter.size(), 0);
    return nruter;
}
} // namespace alma
