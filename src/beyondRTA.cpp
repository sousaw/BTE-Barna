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
/// Implements full BTE calculations defined in beyondRTA.hpp.

#include <iostream>
#include <map>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <utilities.hpp>
#include <beyondRTA.hpp>

namespace alma {
namespace beyondRTA {
Eigen::MatrixXd calc_kappa(
    const alma::Crystal_structure& poscar,
    const alma::Gamma_grid& grid,
    const alma::Symmetry_operations& syms,
    const std::vector<alma::Threeph_process>& threeph_procs,
    const std::vector<alma::Twoph_process>& twoph_procs,
    const Eigen::Ref<const Eigen::ArrayXXd>& w0,
    double T,
    bool iterative,
    boost::mpi::communicator& comm) {
    typedef std::array<int, 3> idx_triplet;
    typedef std::array<int, 2> idx_pair;

    // GATHER INFORMATION FOR FULL BTE SYSTEM AND DECLARE SOME VARIABLES

    int Ngridpoints = grid.nqpoints;

    // determine number of irreducible q-points
    int Nclasses = grid.get_nequivalences();

    int Nbranches = grid.get_spectrum_at_q(0).omega.size();
    int Ntot = Nclasses * Nbranches;

    // Unknowns in the linear system
    // (factor 3: each H unknown is a vector with x,y,z components)
    Eigen::VectorXd H(3 * Ntot);

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
    vg.fill(0.0);

    // table that maps a single-index value to the corresponding unknown
    std::map<int, int> idx_to_unknown;

    // table that maps an unknown to the single-index
    std::vector<int> unknown_to_idx;
    unknown_to_idx.resize(Ntot);

    // Gamma coefficients for 3-phonon processes
    std::map<idx_triplet, double> Gamma_plus;
    std::map<idx_triplet, double> Gamma_minus;

    // Gamma coefficients for 2-phonon processes
    std::map<idx_pair, double> Gamma2;

    // GATHER PHONON PROPERTIES FOR IRREDUCIBLE Q-POINTS

    int unknownCounter = 0;
    double C_factor = 1e27 * alma::constants::kB / Ngridpoints / poscar.V;

    // scan over all equivalence classes in the grid

    for (int nclass = 0; nclass < Nclasses; nclass++) {
        int nqbase = grid.get_representative(nclass);
        auto spectrum = grid.get_spectrum_at_q(nqbase);

        for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
            // register this phonon mode in the mapping tables
            int single_idx = nbranch * Ngridpoints + nqbase;
            idx_to_unknown[single_idx] = unknownCounter;
            unknown_to_idx[unknownCounter] = single_idx;

            // store heat capacity
            C(unknownCounter) = C_factor * alma::bose_einstein_kernel(
                                               spectrum.omega[nbranch], T);

            // store relaxation time
            double mytau =
                (w0(nbranch, nqbase) == 0.) ? 0. : (1. / w0(nbranch, nqbase));
            tau(unknownCounter) = mytau;

            // store phonon frequency
            omega(unknownCounter) = spectrum.omega(nbranch);

            // store group velocity vector
            Eigen::Vector3d myvg = spectrum.vg.col(nbranch);
            vg(unknownCounter, 0) = myvg(0);
            vg(unknownCounter, 1) = myvg(1);
            vg(unknownCounter, 2) = myvg(2);

            // update unkownCounter
            unknownCounter++;
        }
    }

    // CALCULATE NON-RTA CONDUCTIVITY BY SOLVING LINEAR SYSTEM

    // build the list of Gamma values for 2-phonon processes

    for (std::size_t nproc = 0; nproc < twoph_procs.size(); nproc++) {
        // retrieve Gamma value
        alma::Twoph_process myprocess = twoph_procs.at(nproc);
        double Gamma = myprocess.compute_gamma(poscar, grid);

        // obtain single indices
        int idx1 = myprocess.alpha[0] * Ngridpoints + myprocess.q[0];
        int idx2 = myprocess.alpha[1] * Ngridpoints + myprocess.q[1];

        // store Gamma value in the map
        idx_pair key({{idx1, idx2}});
        Gamma2[key] = Gamma;
    } // done scanning over all 2-phonon processes

    // build the list of Gamma values for 3-phonon processes

    for (std::size_t nproc = 0; nproc < threeph_procs.size(); nproc++) {
        alma::Threeph_process myprocess = threeph_procs.at(nproc);
        double Gamma = myprocess.compute_gamma(grid, T);
        bool gammaplus = (myprocess.type == alma::threeph_type::absorption);

        // store Gamma value

        int idx1 = myprocess.alpha[0] * Ngridpoints + myprocess.q[0];
        int idx2 = myprocess.alpha[1] * Ngridpoints + myprocess.q[1];
        int idx3 = myprocess.alpha[2] * Ngridpoints + myprocess.q[2];

        idx_triplet key({{idx1, idx2, idx3}});

        if (gammaplus) {
            Gamma_plus[key] = Gamma;
        }
        else {
            Gamma_minus[key] = Gamma;
        }
    } // done scanning over all 3-phonon processes

    // build system of equations "A*H = B"

    Eigen::MatrixXd A(3 * Ntot, 3 * Ntot);
    A.fill(0.0);

    Eigen::VectorXd B(3 * Ntot);

    // x-components of omega*MFP_RTA
    B.segment(0, Ntot) = omega.array() * tau.array() * vg.col(0).array();
    // y-components of omega*MFP_RTA
    B.segment(Ntot, Ntot) = omega.array() * tau.array() * vg.col(1).array();
    // z-components of omega*MFP_RTA
    B.segment(2 * Ntot, Ntot) = omega.array() * tau.array() * vg.col(2).array();

    // contributions from 3-phonon absorption processes
    for (auto& it3 : Gamma_plus) {
        idx_triplet key = it3.first;
        double Gamma = it3.second;

        // obtain unknownIndex for the irreducible point
        int rowBase = idx_to_unknown[key[0]];

        // PROCESS CHILD POINT 1

        // obtain unknownIndex for the parent of the non-irreducible point
        std::size_t child_nq = key[1] % Ngridpoints;
        std::size_t mybranch = key[1] / Ngridpoints;
        std::size_t parent_nq = grid.getParentIdx(child_nq);
        int colBase = idx_to_unknown[mybranch * Ngridpoints + parent_nq];

        // obtain the rotation matrix that maps the child point to its
        // irreducible parent
        std::size_t symmID = grid.getSymIdxToParent(child_nq);
        std::size_t symmop_idx = symmID / 2;
        // account for time reversal if needed
        double sign_correction = (symmID % 2 == 0) ? 1.0 : -1.0; //

        Eigen::Matrix3d ROT_buffer;
        ROT_buffer.col(0) =
            syms.rotate_v(Eigen::Vector3d(1.0, 0.0, 0.0), symmop_idx, true);
        ROT_buffer.col(1) =
            syms.rotate_v(Eigen::Vector3d(0.0, 1.0, 0.0), symmop_idx, true);
        ROT_buffer.col(2) =
            syms.rotate_v(Eigen::Vector3d(0.0, 0.0, 1.0), symmop_idx, true);

        // obtain the rotation matrix R for which H_child = R*H_parent.
        // This is the inverse of ROT_buffer, being its transpose,
        // corrected for sign.
        Eigen::Matrix3d R = sign_correction * ROT_buffer.transpose();

        // register contributions to the linear system

        for (int nRrow = 0; nRrow < 3; nRrow++) {
            for (int nRcol = 0; nRcol < 3; nRcol++) {
                A(rowBase + nRrow * Ntot, colBase + nRcol * Ntot) +=
                    R(nRrow, nRcol) * tau(rowBase) * Gamma;
            }
        }

        // PROCESS CHILD POINT 2

        // obtain unknownIndex for the parent of the non-irreducible point
        child_nq = key[2] % Ngridpoints;
        mybranch = key[2] / Ngridpoints;
        parent_nq = grid.getParentIdx(child_nq);
        colBase = idx_to_unknown[mybranch * Ngridpoints + parent_nq];

        // obtain the rotation matrix that maps the child point to its
        // irreducible parent
        symmID = grid.getSymIdxToParent(child_nq);
        symmop_idx = symmID / 2;
        // account for time reversal if needed
        sign_correction = (symmID % 2 == 0) ? 1.0 : -1.0;

        ROT_buffer.col(0) =
            syms.rotate_v(Eigen::Vector3d(1.0, 0.0, 0.0), symmop_idx, true);
        ROT_buffer.col(1) =
            syms.rotate_v(Eigen::Vector3d(0.0, 1.0, 0.0), symmop_idx, true);
        ROT_buffer.col(2) =
            syms.rotate_v(Eigen::Vector3d(0.0, 0.0, 1.0), symmop_idx, true);

        // obtain the rotation matrix R for which H_child = R*H_parent.
        // This is the inverse of ROT_buffer, being its transpose,
        // corrected for sign.
        R = sign_correction * ROT_buffer.transpose();

        // register contributions to the linear system

        for (int nRrow = 0; nRrow < 3; nRrow++) {
            for (int nRcol = 0; nRcol < 3; nRcol++) {
                A(rowBase + nRrow * Ntot, colBase + nRcol * Ntot) -=
                    R(nRrow, nRcol) * tau(rowBase) * Gamma;
            }
        }
    }

    // contributions from 3-phonon emission processes

    for (auto& it3 : Gamma_minus) {
        idx_triplet key = it3.first;
        double Gamma = it3.second;

        // obtain unknownIndex for the irreducible point
        int rowBase = idx_to_unknown[key[0]];

        // PROCESS CHILD POINT 1

        // obtain unknownIndex for the parent of the non-irreducible point
        std::size_t child_nq = key[1] % Ngridpoints;
        std::size_t mybranch = key[1] / Ngridpoints;
        std::size_t parent_nq = grid.getParentIdx(child_nq);
        int colBase = idx_to_unknown[mybranch * Ngridpoints + parent_nq];

        // obtain the rotation matrix that maps the child point to its
        // irreducible parent
        std::size_t symmID = grid.getSymIdxToParent(child_nq);
        std::size_t symmop_idx = symmID / 2;
        // account for time reversal if needed
        double sign_correction = (symmID % 2 == 0) ? 1.0 : -1.0;

        Eigen::Matrix3d ROT_buffer;
        ROT_buffer.col(0) =
            syms.rotate_v(Eigen::Vector3d(1.0, 0.0, 0.0), symmop_idx, true);
        ROT_buffer.col(1) =
            syms.rotate_v(Eigen::Vector3d(0.0, 1.0, 0.0), symmop_idx, true);
        ROT_buffer.col(2) =
            syms.rotate_v(Eigen::Vector3d(0.0, 0.0, 1.0), symmop_idx, true);

        // obtain the rotation matrix R for which H_child = R*H_parent.
        // This is the inverse of ROT_buffer, being its transpose,
        // corrected for sign.
        Eigen::Matrix3d R = sign_correction * ROT_buffer.transpose();

        // register contributions to the linear system

        for (int nRrow = 0; nRrow < 3; nRrow++) {
            for (int nRcol = 0; nRcol < 3; nRcol++) {
                A(rowBase + nRrow * Ntot, colBase + nRcol * Ntot) -=
                    0.5 * R(nRrow, nRcol) * tau(rowBase) * Gamma;
            }
        }

        // PROCESS CHILD POINT 2

        // obtain unknownIndex for the parent of the non-irreducible point
        child_nq = key[2] % Ngridpoints;
        mybranch = key[2] / Ngridpoints;
        parent_nq = grid.getParentIdx(child_nq);
        colBase = idx_to_unknown[mybranch * Ngridpoints + parent_nq];

        // obtain the rotation matrix that maps the child point to its
        // irreducible parent
        symmID = grid.getSymIdxToParent(child_nq);
        symmop_idx = symmID / 2;
        // account for time reversal if needed
        sign_correction = (symmID % 2 == 0) ? 1.0 : -1.0;

        ROT_buffer.col(0) =
            syms.rotate_v(Eigen::Vector3d(1.0, 0.0, 0.0), symmop_idx, true);
        ROT_buffer.col(1) =
            syms.rotate_v(Eigen::Vector3d(0.0, 1.0, 0.0), symmop_idx, true);
        ROT_buffer.col(2) =
            syms.rotate_v(Eigen::Vector3d(0.0, 0.0, 1.0), symmop_idx, true);

        // obtain the rotation matrix R for which H_child = R*H_parent.
        // This is the inverse of ROT_buffer, being its transpose,
        // corrected for sign.
        R = sign_correction * ROT_buffer.transpose();

        // register contributions to the linear system

        for (int nRrow = 0; nRrow < 3; nRrow++) {
            for (int nRcol = 0; nRcol < 3; nRcol++) {
                A(rowBase + nRrow * Ntot, colBase + nRcol * Ntot) -=
                    0.5 * R(nRrow, nRcol) * tau(rowBase) * Gamma;
            }
        }
    }

    // Share the contributions from 3-phonon processes among cores and
    // add them together.
    if (comm.size() > 1) {
        Eigen::MatrixXd my_A(A);
        A.fill(0.);
        boost::mpi::all_reduce(
            comm, my_A.data(), my_A.size(), A.data(), std::plus<double>());
    }

    // contributions from 2-phonon processes
    for (auto& it2 : Gamma2) {
        idx_pair key = it2.first;
        double Gamma = it2.second;

        // obtain unknownIndex for the irreducible point
        int rowBase = idx_to_unknown[key[0]];

        // obtain unknownIndex for the parent of the non-irreducible point
        std::size_t child_nq = key[1] % Ngridpoints;
        std::size_t mybranch = key[1] / Ngridpoints;
        std::size_t parent_nq = grid.getParentIdx(child_nq);
        int colBase = idx_to_unknown[mybranch * Ngridpoints + parent_nq];

        // obtain the rotation matrix that maps the child point to its
        // irreducible parent
        std::size_t symmID = grid.getSymIdxToParent(child_nq);
        std::size_t symmop_idx = symmID / 2;
        // account for time reversal if needed
        double sign_correction = (symmID % 2 == 0) ? 1.0 : -1.0;

        Eigen::Matrix3d ROT_buffer;
        ROT_buffer.col(0) =
            syms.rotate_v(Eigen::Vector3d(1.0, 0.0, 0.0), symmop_idx, true);
        ROT_buffer.col(1) =
            syms.rotate_v(Eigen::Vector3d(0.0, 1.0, 0.0), symmop_idx, true);
        ROT_buffer.col(2) =
            syms.rotate_v(Eigen::Vector3d(0.0, 0.0, 1.0), symmop_idx, true);

        // obtain the rotation matrix R for which H_child = R*H_parent.
        // This is the inverse of ROT_buffer (being its transpose)
        // corrected for sign.
        Eigen::Matrix3d R = sign_correction * ROT_buffer.transpose();

        // register contributions to the linear system

        for (int nRrow = 0; nRrow < 3; nRrow++) {
            for (int nRcol = 0; nRcol < 3; nRcol++) {
                A(rowBase + nRrow * Ntot, colBase + nRcol * Ntot) -=
                    R(nRrow, nRcol) * tau(rowBase) * Gamma;
            }
        }
    }

    // diagonal elements in the system
    for (int diag_idx = 0; diag_idx < 3 * Ntot; diag_idx++) {
        A(diag_idx, diag_idx) += 1.0;
    }

    // Normalise each row for better numerical stability

    for (int nrow = 0; nrow < 3 * Ntot; nrow++) {
        double scale = A.row(nrow).array().abs().maxCoeff();
        A.row(nrow) /= scale;
        B(nrow) /= scale;
    }

    // SOLVE SYSTEM

    if (iterative) {
        Eigen::BiCGSTAB<Eigen::MatrixXd> solver;
        solver.compute(A);

        // RTA solution to be used as initial guess
        Eigen::VectorXd H_guess(3 * Ntot);

        H_guess.segment(0, Ntot) =
            omega.array() * tau.array() * vg.col(0).array();
        H_guess.segment(Ntot, Ntot) =
            omega.array() * tau.array() * vg.col(1).array();
        H_guess.segment(2 * Ntot, Ntot) =
            omega.array() * tau.array() * vg.col(2).array();

        H = solver.solveWithGuess(B, H_guess);
    }

    else {
        H = A.partialPivLu().solve(B);
    }

    double rel_err = (A * H - B).norm() / B.norm();

    if (rel_err > 1e-3) {
        std::cout << "alma::beyondRTA::calc_kappa > WARNING:" << std::endl;
        std::cout << "solution of linear system might be unstable."
                  << std::endl;
        std::cout << "Relative error metric = " << rel_err << std::endl;
    }

    // PROCESS THE LINEAR SYSTEM SOLUTION

    // construct "generalised MFPs"
    Eigen::MatrixXd MFP_nonRTA(Ntot, 3);
    MFP_nonRTA.col(0) = H.segment(0, Ntot).array() / omega.array();
    MFP_nonRTA.col(1) = H.segment(Ntot, Ntot).array() / omega.array();
    MFP_nonRTA.col(2) = H.segment(2 * Ntot, Ntot).array() / omega.array();

    // fix potential NaN/Inf problems
    for (int n = 0; n < Ntot; n++) {
        if (omega(n) <= 0.0) {
            MFP_nonRTA(n, 0) = 0.0;
            MFP_nonRTA(n, 1) = 0.0;
            MFP_nonRTA(n, 2) = 0.0;
        }
    }

    // symmetrise obtained MFP_nonRTA

    for (int nclass = 0; nclass < Nclasses; nclass++) {
        // obtain the q-index of the class representative
        std::size_t nq_parent = grid.get_representative(nclass);

        for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
            int single_idx = nbranch * Ngridpoints + nq_parent;
            int unknownIndex = idx_to_unknown[single_idx];

            Eigen::Vector3d MFP_parent =
                MFP_nonRTA.row(unknownIndex).transpose();
            Eigen::Vector3d MFP_buffer =
                grid.copy_symmetry(nq_parent, syms, MFP_parent);

            MFP_nonRTA.row(unknownIndex) = MFP_buffer.transpose();
        }
    }

    // fix potential NaN/Inf problems
    for (int n = 0; n < Ntot; n++) {
        if (omega(n) <= 0.0) {
            MFP_nonRTA(n, 0) = 0.0;
            MFP_nonRTA(n, 1) = 0.0;
            MFP_nonRTA(n, 2) = 0.0;
        }
    }

    // CALCULATE NON-RTA CONDUCTIVITY

    Eigen::Matrix3d kappatensor;
    kappatensor.fill(0.0);

    for (int nclass = 0; nclass < Nclasses; nclass++) {
        // obtain the q-index of the class representative
        std::size_t nq_parent = grid.get_representative(nclass);

        for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
            int single_idx = nbranch * Ngridpoints + nq_parent;
            int unknownIndex = idx_to_unknown[single_idx];

            Eigen::Vector3d MFP_parent =
                MFP_nonRTA.row(unknownIndex).transpose();

            for (auto nq_member : grid.get_equivalence(nclass)) {
                // obtain the group velocity vector of the class member
                Eigen::Vector3d vg_member =
                    (grid.get_spectrum_at_q(nq_member)).vg.col(nbranch);

                // obtain the rotation matrix that maps the class member
                // to its irreducible parent
                std::size_t symmID = grid.getSymIdxToParent(nq_member);
                std::size_t symmop_idx = symmID / 2;
                double sign_correction = (symmID % 2 == 0) ? 1.0 : -1.0;

                Eigen::Matrix3d ROT_buffer;
                ROT_buffer.col(0) = syms.rotate_v(
                    Eigen::Vector3d(1.0, 0.0, 0.0), symmop_idx, true);
                ROT_buffer.col(1) = syms.rotate_v(
                    Eigen::Vector3d(0.0, 1.0, 0.0), symmop_idx, true);
                ROT_buffer.col(2) = syms.rotate_v(
                    Eigen::Vector3d(0.0, 0.0, 1.0), symmop_idx, true);

                // obtain the rotation matrix R for which v_child = R*v_parent.
                // This is the inverse of ROT_buffer, being its transpose,
                // corrected for sign.
                Eigen::Matrix3d R = sign_correction * ROT_buffer.transpose();

                // calculate the non-RTA MFP vector of this mode
                Eigen::Vector3d MFP_member = R * MFP_parent;

                // obtain conductivity contribution
                Eigen::Matrix3d outer =
                    (1e-9 * MFP_member) * (1e3 * vg_member.transpose());
                kappatensor += C(unknownIndex) * outer;
            }
        }
    }

    // symmetrise kappa tensor

    Eigen::Matrix3d kappa_accumulated;
    kappa_accumulated.fill(0.0);

    for (std::size_t nsymm = 0; nsymm < syms.get_nsym(); nsymm++) {
        kappa_accumulated += syms.rotate_m<double>(kappatensor, nsymm, true);
    }

    kappatensor = kappa_accumulated / static_cast<double>(syms.get_nsym());

    // RETURN RESULT

    return kappatensor;
} // end of calc_kappa
} // end of namespace beyondRTA
} // end of namespace alma
