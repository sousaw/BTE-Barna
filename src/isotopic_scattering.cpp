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
/// Definitions corresponding to isotopic_scattering.hpp.

#include <cmath>
#include <complex>
#include <limits>
#include <constants.hpp>
#include <periodic_table.hpp>
#include <utilities.hpp>
#include <isotopic_scattering.hpp>

namespace alma {
/// Compute the 25th and 75th percentiles of log(sigma)
///
/// @param[in] sigma an Eigen array with all broadening parameters
/// @return an array containing the 25th and 75th percentiles
std::array<double, 2> calc_percentiles_log(
    const Eigen::Ref<const Eigen::ArrayXXd>& sigma) {
    auto rows = sigma.rows();
    auto cols = sigma.cols();

    std::vector<double> logsigma;
    logsigma.reserve(rows * cols);

    for (decltype(cols) c = 0; c < cols; ++c)
        for (decltype(rows) r = 0; r < rows; ++r)
            logsigma.emplace_back(sigma(r, c) == 0.
                                      ? std::numeric_limits<double>::lowest()
                                      : std::log(sigma(r, c)));
    std::sort(logsigma.begin(), logsigma.end());

    std::array<double, 2> nruter;
    nruter[0] = logsigma[25 * rows * cols / 100];
    nruter[1] = logsigma[75 * rows * cols / 100];
    return nruter;
}


std::vector<Twoph_process> find_allowed_twoph(
    const Gamma_grid& grid,
    const boost::mpi::communicator& communicator,
    double scalebroad) {
    auto nprocs = communicator.size();
    std::size_t nmodes = grid.get_spectrum_at_q(0).omega.size();

    // Precompute the broadening of all modes.
    Eigen::ArrayXXd sigmas(nmodes, grid.nqpoints);

    for (std::size_t iq = 0; iq < grid.nqpoints; ++iq)
        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            auto spectrum = grid.get_spectrum_at_q(iq);
            sigmas(im, iq) = scalebroad * grid.base_sigma(spectrum.vg.col(im));
        }
    // And refine them by removing outliers.
    auto percent = calc_percentiles_log(sigmas);
    double lbound = std::exp(percent[0] - 1.5 * (percent[1] - percent[0]));
    sigmas = (sigmas < lbound).select(lbound, sigmas);

    auto my_id = communicator.rank();
    auto limits = my_jobs(grid.get_nequivalences(), nprocs, my_id);
    std::vector<Twoph_process> nruter;

    for (auto ic = limits[0]; ic < limits[1]; ++ic) {
        auto iq1 = grid.get_representative(ic);
        auto spectrum1 = grid.get_spectrum_at_q(iq1);

        for (std::size_t iq2 = 0; iq2 < grid.nqpoints; ++iq2) {
            auto spectrum2 = grid.get_spectrum_at_q(iq2);

            for (decltype(nmodes) im1 = 0; im1 < nmodes; ++im1) {
                for (decltype(nmodes) im2 = 0; im2 < nmodes; ++im2) {
                    // Note that, since momentum is not conserved,
                    // the frequencies are uncorrelated random
                    // variables. The total standard deviation is
                    // the quadratic sum of both standard
                    // deviations.
                    auto sigma = std::hypot(sigmas(im1, iq1), sigmas(im2, iq2));
                    // The only criterion for accepting a process as
                    // allowed is conservation of energy after
                    // taking into account the discretization of
                    // reciprocal space with the usual adaptive
                    // broadening method.
                    auto delta =
                        std::fabs(spectrum1.omega(im1) - spectrum2.omega(im2));

                    if (delta <= constants::nsigma * sigma)
                        nruter.emplace_back(Twoph_process(
                            ic,
                            std::array<std::size_t, 2>({{iq1, iq2}}),
                            std::array<std::size_t, 2>({{im1, im2}}),
                            delta,
                            sigma));
                }
            }
        }
    }
    return nruter;
}


double Twoph_process::compute_gamma(const Crystal_structure& cell,
                                    const Gamma_grid& grid) const {
    // Obtain the set of g factors.
    auto natoms = cell.get_natoms();
    Eigen::VectorXd gfactors(natoms);

    for (decltype(natoms) iatom = 0; iatom < natoms; ++iatom)
        gfactors(iatom) = cell.get_gfactor(iatom);
    // And call the more explicit version of the method.
    return this->compute_gamma(cell, gfactors, grid);
}


double Twoph_process::compute_gamma(
    const Crystal_structure& cell,
    const Eigen::Ref<const Eigen::VectorXd>& gfactors,
    const Gamma_grid& grid) const {
    auto natoms = cell.get_natoms();

    if (gfactors.size() != natoms)
        throw value_error("There must be as many g factors as there are"
                          " atoms in the cell");
    // Obtain the weighted overlap between wave functions.
    Eigen::VectorXcd wf1 =
        grid.get_spectrum_at_q(this->q[0]).wfs.col(this->alpha[0]);
    Eigen::VectorXcd wf2 =
        grid.get_spectrum_at_q(this->q[1]).wfs.col(this->alpha[1]);
    double nruter{0.};

    for (decltype(natoms) iatom = 0; iatom < natoms; ++iatom)
        nruter +=
            gfactors(iatom) *
            std::norm(wf1.segment<3>(3 * iatom).dot(wf2.segment<3>(3 * iatom)));
    // Multiply by the right factors to obtain vp2.
    nruter *= alma::constants::pi *
              grid.get_spectrum_at_q(this->q[0]).omega(this->alpha[0]) *
              grid.get_spectrum_at_q(this->q[1]).omega(this->alpha[1]) / 2. /
              grid.nqpoints;
    // Multiply by the regularized delta and return.
    return this->gaussian * nruter;
}


Eigen::ArrayXXd calc_w0_twoph(const Crystal_structure& cell,
                              const alma::Gamma_grid& grid,
                              const std::vector<alma::Twoph_process>& processes,
                              const boost::mpi::communicator& comm) {
    // Obtain the set of g factors.
    auto natoms = cell.get_natoms();
    Eigen::VectorXd gfactors(natoms);

    for (decltype(natoms) iatom = 0; iatom < natoms; ++iatom)
        gfactors(iatom) = cell.get_gfactor(iatom);
    // And call the more explicit version of the function.
    return calc_w0_twoph(cell, gfactors, grid, processes, comm);
}


Eigen::ArrayXXd calc_w0_twoph(const Crystal_structure& cell,
                              const Eigen::Ref<const Eigen::VectorXd>& gfactors,
                              const alma::Gamma_grid& grid,
                              const std::vector<alma::Twoph_process>& processes,
                              const boost::mpi::communicator& comm) {
    auto nqpoints = grid.nqpoints;
    auto nmodes = grid.get_spectrum_at_q(0).omega.size();

    if (gfactors.size() != cell.get_natoms())
        throw value_error("There must be as many g factors as there are"
                          " atoms in the cell");
    // Compute all contributions belonging to the current
    // MPI process.
    Eigen::ArrayXXd my_w0{Eigen::ArrayXXd::Zero(nmodes, nqpoints)};

    for (std::size_t i = 0; i < processes.size(); ++i)
        my_w0(processes[i].alpha[0], processes[i].q[0]) +=
            processes[i].compute_gamma(cell, gfactors, grid);
    // Extend the calculation to all q points.
    auto nequivalences = grid.get_nequivalences();

    for (decltype(nequivalences) i = 0; i < nequivalences; ++i) {
        auto eq = grid.get_equivalence(i);

        for (std::size_t iq = 1; iq < eq.size(); ++iq)
            my_w0.col(eq[iq]) = my_w0.col(eq[0]);
    }
    // Sum over processors to obtain the total scattering rate for
    // each mode.
    Eigen::ArrayXXd nruter{Eigen::ArrayXXd::Zero(nmodes, nqpoints)};
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
