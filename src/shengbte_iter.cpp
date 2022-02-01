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
/// Definitions corresponding to shengbte_iter.hpp.
#include <shengbte_iter.hpp>

namespace alma {
ShengBTE_iterator::ShengBTE_iterator(
    const Crystal_structure& poscar,
    const Gamma_grid& grid,
    const Symmetry_operations& syms,
    std::vector<Threeph_process>& threeph_procs,
    std::vector<Twoph_process>& twoph_procs,
    double T,
    const boost::mpi::communicator& comm_)
    : nqpoints{grid.nqpoints}, nirred{grid.get_nequivalences()},
      nbranches{
          static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size())},
      V{poscar.V}, n{0}, omega{nbranches, nqpoints}, vg{3 * nbranches,
                                                        nqpoints},
      tau0{nbranches, nqpoints}, F{3 * nbranches, nqpoints}, comm(comm_) {
    // Rearrange frequencies group velocities and scattering rates in a
    // more convenient way.
    Eigen::ArrayXXd w0{calc_w0_threeph(grid, threeph_procs, T, this->comm) +
                       calc_w0_twoph(poscar, grid, twoph_procs, this->comm)};
    for (std::size_t iq = 0; iq < this->nqpoints; ++iq) {
        auto spectrum = grid.get_spectrum_at_q(iq);
        for (std::size_t im = 0; im < this->nbranches; ++im) {
            this->omega.col(iq) = spectrum.omega;
            this->vg.block<3, 1>(3 * im, iq) = spectrum.vg.col(im);
            if (w0(im, iq) == 0) {
                this->tau0(im, iq) = 0;
            }
            else {
                this->tau0(im, iq) = 1. / w0(im, iq);
            }
        }
    }
    // Compute the initial value of all elements of F.
    for (std::size_t iq = 0; iq < this->nqpoints; ++iq) {
        Eigen::Vector3d q{grid.get_q(iq)};
        for (std::size_t im = 0; im < this->nbranches; ++im) {
            this->F.block<3, 1>(3 * im, iq) = this->tau0(im, iq) *
                                              this->vg.block<3, 1>(3 * im, iq) *
                                              this->omega(im, iq);
            // Make sure that F is a vector.
            this->F.block<3, 1>(3 * im, iq) =
                grid.copy_symmetry(iq, syms, this->F.block<3, 1>(3 * im, iq));
        }
    }
}


void ShengBTE_iterator::next(const Crystal_structure& poscar,
                             const Gamma_grid& grid,
                             const Symmetry_operations& syms,
                             std::vector<Threeph_process>& threeph_procs,
                             std::vector<Twoph_process>& twoph_procs,
                             double T) {
    Eigen::MatrixXd my_delta{
        Eigen::MatrixXd::Zero(3 * this->nbranches, this->nqpoints)};
    // Compute the contribution of this worker to Delta.
    for (auto& p : threeph_procs) {
        double gamma = p.compute_gamma(grid, T);
        int s = static_cast<int>(p.type);
        double prefactor = 2. / (3. - s);
        auto eq = grid.get_equivalence(p.c);
        std::map<std::size_t, bool> done;
        for (const auto& iq : eq)
            done[iq] = false;
        const auto& images1 = grid.equivalent_qpoints(p.q[0]);
        auto nops = images1.size();
        const auto& images2 = grid.equivalent_qpoints(p.q[1]);
        const auto& images3 = grid.equivalent_qpoints(p.q[2]);
        for (std::size_t i = 0; i < nops; ++i) {
            auto q1 = images1[i];
            if (done[q1])
                continue;
            auto q2 = images2[i];
            auto q3 = images3[i];
            if (q1 == this->nqpoints || q2 == this->nqpoints ||
                q3 == this->nqpoints)
                continue;
            my_delta.block<3, 1>(3 * p.alpha[0], q1) +=
                prefactor * gamma *
                (this->F.block<3, 1>(3 * p.alpha[2], q3) -
                 s * this->F.block<3, 1>(3 * p.alpha[1], q2));
            done[q1] = true;
        }
    }
    for (auto& p : twoph_procs) {
        double gamma = p.compute_gamma(poscar, grid);
        auto eq = grid.get_equivalence(p.c);
        std::map<std::size_t, bool> done;
        for (const auto& iq : eq)
            done[iq] = false;
        const auto& images1 = grid.equivalent_qpoints(p.q[0]);
        auto nops = images1.size();
        const auto& images2 = grid.equivalent_qpoints(p.q[1]);
        for (std::size_t i = 0; i < nops; ++i) {
            auto q1 = images1[i];
            if (done[q1])
                continue;
            auto q2 = images2[i];
            if (q1 == this->nqpoints || q2 == this->nqpoints)
                continue;
            my_delta.block<3, 1>(3 * p.alpha[0], q1) +=
                gamma * this->F.block<3, 1>(3 * p.alpha[1], q2);
        }
    }
    // Sum the contributions to delta from all workers.
    Eigen::MatrixXd delta{
        Eigen::MatrixXd::Zero(3 * this->nbranches, this->nqpoints)};
    boost::mpi::all_reduce(this->comm,
                           my_delta.data(),
                           my_delta.size(),
                           delta.data(),
                           std::plus<double>());
    // Update F.
    for (std::size_t iq = 0; iq < this->nqpoints; ++iq) {
        Eigen::Vector3d q{grid.get_q(iq)};
        for (std::size_t im = 0; im < this->nbranches; ++im) {
            this->F.block<3, 1>(3 * im, iq) =
                this->tau0(im, iq) *
                (this->vg.block<3, 1>(3 * im, iq) * this->omega(im, iq) +
                 delta.block<3, 1>(3 * im, iq));
            // Make sure that F is a vector.
            this->F.block<3, 1>(3 * im, iq) =
                grid.copy_symmetry(iq, syms, this->F.block<3, 1>(3 * im, iq));
        }
    }
    ++(this->n);
}


Eigen::Matrix3d ShengBTE_iterator::calc_current_kappa(double T) const {
    Eigen::Matrix3d nruter{Eigen::Matrix3d::Zero()};
    for (std::size_t iq = 1; iq < this->nqpoints; ++iq) {
        for (std::size_t im = 0; im < this->nbranches; ++im) {
            Eigen::Matrix3d outer{this->vg.block<3, 1>(3 * im, iq) *
                                  this->F.block<3, 1>(3 * im, iq).transpose()};
            nruter += bose_einstein_kernel(this->omega(im, iq), T) * outer /
                      this->omega(im, iq);
        }
    }
    return (1e21 * constants::kB / this->V / this->nqpoints) * nruter;
}


Eigen::Matrix3d ShengBTE_iterator::calc_current_kappa_branch(
    double T,
    std::size_t branch) const {
    Eigen::Matrix3d nruter{Eigen::Matrix3d::Zero()};
    if (branch >= this->nbranches) {
        throw value_error("invalid branch index");
    }
    for (std::size_t iq = 1; iq < this->nqpoints; ++iq) {
        Eigen::Matrix3d outer{this->vg.block<3, 1>(3 * branch, iq) *
                              this->F.block<3, 1>(3 * branch, iq).transpose()};
        nruter += bose_einstein_kernel(this->omega(branch, iq), T) * outer /
                  this->omega(branch, iq);
    }
    return (1e21 * constants::kB / this->V / this->nqpoints) * nruter;
}


Eigen::ArrayXXd ShengBTE_iterator::calc_w() const {
    Eigen::ArrayXXd nruter(this->nbranches, this->nqpoints);
    for (std::size_t iq = 0; iq < this->nqpoints; ++iq) {
        for (std::size_t im = 0; im < this->nbranches; ++im) {
            nruter(im, iq) = this->vg.block<3, 1>(3 * im, iq).squaredNorm() *
                             this->omega(im, iq) /
                             (this->vg.block<3, 1>(3 * im, iq)
                                  .dot(this->F.block<3, 1>(3 * im, iq)));
        }
    }
    return nruter;
}


Eigen::ArrayXXd ShengBTE_iterator::calc_lambda() const {
    Eigen::ArrayXXd nruter(this->nbranches, this->nqpoints);
    for (std::size_t iq = 0; iq < this->nqpoints; ++iq) {
        for (std::size_t im = 0; im < this->nbranches; ++im) {
            nruter(im, iq) =
                this->vg.block<3, 1>(3 * im, iq)
                    .dot(this->F.block<3, 1>(3 * im, iq)) /
                (this->omega(im, iq) * this->vg.block<3, 1>(3 * im, iq).norm());
        }
    }
    return nruter;
}


std::vector<Eigen::Matrix3d> ShengBTE_iterator::calc_cumulative_kappa_omega(
    double T,
    Eigen::ArrayXd ticks) {
    std::vector<Eigen::Matrix3d> nruter;
    for (int i = 0; i < ticks.size(); ++i) {
        Eigen::Matrix3d kappa(Eigen::Matrix3d::Zero());
        nruter.emplace_back(kappa);
    }
    for (std::size_t iq = 1; iq < this->nqpoints; ++iq) {
        for (std::size_t im = 0; im < this->nbranches; ++im) {
            Eigen::Matrix3d outer{this->vg.block<3, 1>(3 * im, iq) *
                                  this->F.block<3, 1>(3 * im, iq).transpose()};
            Eigen::Matrix3d contribution{
                bose_einstein_kernel(this->omega(im, iq), T) * outer /
                this->omega(im, iq)};
            for (int i = 0; i < ticks.size(); ++i) {
                if (ticks(i) > omega(im, iq)) {
                    nruter[i] += contribution;
                }
            }
        }
    }
    for (int i = 0; i < ticks.size(); ++i) {
        nruter[i] *= 1e21 * constants::kB / this->V / this->nqpoints;
    }
    return nruter;
}


std::vector<Eigen::Matrix3d> ShengBTE_iterator::calc_cumulative_kappa_lambda(
    double T,
    Eigen::ArrayXd ticks) {
    Eigen::ArrayXXd lambda = this->calc_lambda();
    std::vector<Eigen::Matrix3d> nruter;
    for (int i = 0; i < ticks.size(); ++i) {
        Eigen::Matrix3d kappa(Eigen::Matrix3d::Zero());
        nruter.emplace_back(kappa);
    }
    for (std::size_t iq = 1; iq < this->nqpoints; ++iq) {
        for (std::size_t im = 0; im < this->nbranches; ++im) {
            Eigen::Matrix3d outer{this->vg.block<3, 1>(3 * im, iq) *
                                  this->F.block<3, 1>(3 * im, iq).transpose()};
            Eigen::Matrix3d contribution{
                bose_einstein_kernel(this->omega(im, iq), T) * outer /
                this->omega(im, iq)};
            for (int i = 0; i < ticks.size(); ++i) {
                if (ticks(i) > lambda(im, iq)) {
                    nruter[i] += contribution;
                }
            }
        }
    }
    for (int i = 0; i < ticks.size(); ++i) {
        nruter[i] *= 1e21 * constants::kB / this->V / this->nqpoints;
    }
    return nruter;
}


Eigen::MatrixXd calc_shengbte_kappa(
    const alma::Crystal_structure& poscar,
    const alma::Gamma_grid& grid,
    const alma::Symmetry_operations& syms,
    std::vector<alma::Threeph_process>& threeph_procs,
    std::vector<alma::Twoph_process>& twoph_procs,
    double T,
    const boost::mpi::communicator& comm,
    double tolerance,
    std::size_t maxiter) {
    ShengBTE_iterator iterator(
        poscar, grid, syms, threeph_procs, twoph_procs, T, comm);
    Eigen::Matrix3d kappa_shengbte;
    Eigen::Matrix3d kappa_shengbte_old;
    kappa_shengbte = iterator.calc_current_kappa(T);
    for (std::size_t iter = 1; iter < maxiter; ++iter) {
        kappa_shengbte_old = kappa_shengbte;
        iterator.next(poscar, grid, syms, threeph_procs, twoph_procs, T);
        kappa_shengbte = iterator.calc_current_kappa(T);
        double change = (kappa_shengbte - kappa_shengbte_old).norm() /
                        kappa_shengbte_old.norm();
        if (change < tolerance)
            return kappa_shengbte;
    }
    throw exception("maximum number of iterations exceeded");
}
} // namespace alma
