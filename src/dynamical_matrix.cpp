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
/// Definitions corresponding to dynamical_matrix.hpp.

#include <iostream>
#include <complex>
#include <cmath>
#include <constants.hpp>
#include <utilities.hpp>
#include <periodic_table.hpp>
#include <dynamical_matrix.hpp>

namespace alma {
/// Return a square matrix with 3 * natoms rows where
/// element ij is equal to the square root of the products
/// of the masses of atom i/3 and atom j/3.
///
/// @param[in] structure - structure of the unit cell
/// @return - the aforementioned square matrix
Eigen::ArrayXXd build_mass_matrix(const Crystal_structure& structure) {
    auto natoms = structure.get_natoms();
    auto ndof = 3 * natoms;
    Eigen::ArrayXd m(ndof);

    for (auto i = 0; i < natoms; ++i)
        m.segment<3>(3 * i).setConstant(std::sqrt(structure.get_mass(i)));
    return m.matrix() * m.matrix().transpose();
}


/// POD class representing a pair of atoms - one in unit cell (0, 0,
/// 0) the other in an arbitrary unit cell cj, and the image of the
/// latter in a number of unit cells cjp.
class Atom_pair {
public:
    /// Index of the first atom in its unit cell.
    int i;
    /// Index of the second atom in its unit cell.
    int j;
    /// Unit cell the second atom belongs to in a regular
    /// supercell representation.
    Triple_int cj;
    /// All unit cells that the image of the second atom
    /// belongs to in a Wigner-Seitz supercell representation.
    std::vector<Triple_int> cjp;
};


/// Find all atom pairs in a Wigner-Seitz representation
/// of an na x nb x nc supercell.
///
/// @param[in] structure - description of the unit cell
/// @param[in] fcs - Harmomic_ifcs object adapted to the supercell
/// @return - a vector of Atom_pair
std::vector<Atom_pair> get_normal_pairs(const Crystal_structure& structure,
                                        const Harmonic_ifcs& fcs,
                                        int na,
                                        int nb,
                                        int nc) {
    std::vector<Atom_pair> nruter;
    auto natoms = structure.get_natoms();
    Eigen::Vector3d delta;
    Eigen::Vector3d sdelta;

    for (auto iatom1 = 0; iatom1 < natoms; ++iatom1)
        for (auto iatom2 = 0; iatom2 < natoms; ++iatom2)
            for (auto p : fcs.pos) {
                auto ia = p[0];
                auto ib = p[1];
                auto ic = p[2];
                delta << ia, ib, ic;
                delta += structure.positions.col(iatom1).transpose();
                delta -= structure.positions.col(iatom2).transpose();
                auto dmin = Min_keeper<Triple_int>();

                for (auto sa = -2; sa < 3; ++sa)
                    for (auto sb = -2; sb < 3; ++sb)
                        for (auto sc = -2; sc < 3; ++sc) {
                            sdelta << sa * na, sb * nb, sc * nc;
                            sdelta = structure.lattvec * (delta + sdelta);
                            dmin.update(Triple_int({{ia + sa * na,
                                                     ib + sb * nb,
                                                     ic + sc * nc}}),
                                        sdelta.squaredNorm());
                        }
                nruter.emplace_back(Atom_pair{
                    iatom1, iatom2, {{ia, ib, ic}}, dmin.get_vector()});
            }
    return nruter;
}


void Dynamical_matrix_builder::copy_blocks(const Harmonic_ifcs& fcs) {
    auto natoms = this->structure.get_natoms();
    auto ndof = 3 * natoms;

    this->massmatrix = build_mass_matrix(this->structure);

    auto pairs =
        get_normal_pairs(this->structure, fcs, this->na, this->nb, this->nc);

    // "Unfold" the blocks in fcs to obtain the blocks of the
    // dynamical matrix.
    Triple_int_map<Eigen::MatrixXd> blocks;
    Triple_int_map<Eigen::MatrixXd> masks;

    for (auto p : pairs) {
        auto ip = std::distance(
            fcs.pos.begin(), std::find(fcs.pos.begin(), fcs.pos.end(), p.cj));
        // The transposition is required by Phonopy's conventions
        // about indices.
        Eigen::MatrixXd block =
            ((fcs.ifcs[ip] / p.cjp.size()).array() / this->massmatrix)
                .transpose();
        Eigen::MatrixXd mask = Eigen::MatrixXd::Constant(
            fcs.ifcs[ip].rows(), fcs.ifcs[ip].cols(), 1. / p.cjp.size());

        for (auto pp : p.cjp) {
            if (blocks.find(pp) == blocks.end()) {
                blocks[pp] = Eigen::MatrixXd::Zero(ndof, ndof);
                masks[pp] = Eigen::MatrixXd::Zero(ndof, ndof);
            }
            blocks[pp].block<3, 3>(3 * p.i, 3 * p.j) =
                block.block<3, 3>(3 * p.i, 3 * p.j);
            masks[pp].block<3, 3>(3 * p.i, 3 * p.j) =
                mask.block<3, 3>(3 * p.i, 3 * p.j);
        }
    }

    auto kav = split_keys_and_values(blocks);
    this->pos.swap(std::get<0>(kav));
    this->blocks.swap(std::get<1>(kav));
    this->mpos.resize(3, this->blocks.size());
    kav = split_keys_and_values(masks);
    this->masks.swap(std::get<1>(kav));

    for (decltype(this->blocks.size()) i = 0; i < this->blocks.size(); ++i)
        for (auto j = 0; j < 3; ++j)
            this->mpos(j, i) = this->pos[i][j];
    this->cpos = this->structure.lattvec * this->mpos;

    // Convert from eV / A^2 / amu to (rad / ps)^2.
    for (auto& p : this->blocks)
        p *= constants::e / constants::amu * 1e-4;
}


Dynamical_matrix_builder::Dynamical_matrix_builder(
    const Crystal_structure& _structure,
    const Symmetry_operations& syms,
    const Harmonic_ifcs& fcs)
    : na(fcs.na), nb(fcs.nb), nc(fcs.nc), V(_structure.V),
      structure(_structure), rlattvec(_structure.rlattvec), symmetries(syms),
      nonanalytic(false) {
    this->copy_blocks(fcs);
}


Dynamical_matrix_builder::Dynamical_matrix_builder(
    const Crystal_structure& _structure,
    const Symmetry_operations& syms,
    const Harmonic_ifcs& fcs,
    const Dielectric_parameters& _born)
    : na(fcs.na), nb(fcs.nb), nc(fcs.nc), V(_structure.V),
      structure(_structure), rlattvec(_structure.rlattvec), symmetries(syms),
      nonanalytic(true), born(_born) {
    if (this->born.born.size() !=
        static_cast<std::size_t>(this->structure.get_natoms()))
        throw value_error("wrong number of Born charges");
    this->copy_blocks(fcs);
}


Eigen::ArrayXcd Dynamical_matrix_builder::get_exponentials(
    const Eigen::Ref<const Eigen::Vector3d>& q) const {
    auto args = (q.transpose() * this->cpos).array();

    return args.cos().cast<std::complex<double>>() -
           constants::imud * args.sin().cast<std::complex<double>>();
}


std::array<Eigen::ArrayXXd, 4> Dynamical_matrix_builder::build_nac(
    const Eigen::Ref<const Eigen::Vector3d>& q) const {
    constexpr double prefactor = constants::e * constants::e /
                                 constants::epsilon0 / constants::amu * 1e3;
    // The vector is reduced to the first BZ and normalized, since only
    // its direction matters.
    Eigen::Vector3d uq{this->structure.map_to_firstbz(q).col(0).normalized()};
    auto ndof = this->blocks[0].cols();
    auto natoms = ndof / 3;
    double epsilon = uq.dot(this->born.epsilon * uq);

    std::array<Eigen::ArrayXXd, 4> nruter;

    for (auto i = 0; i < 4; ++i)
        nruter[i].setZero(ndof, ndof);

    for (decltype(natoms) iatom1 = 0; iatom1 < natoms; ++iatom1) {
        Eigen::MatrixXd z1{uq.transpose() * this->born.born[iatom1]};
        for (decltype(natoms) iatom2 = 0; iatom2 < natoms; ++iatom2) {
            Eigen::MatrixXd z2{uq.transpose() * this->born.born[iatom2]};
            Eigen::MatrixXd zz{z1.transpose() * z2};
            nruter[0].block<3, 3>(3 * iatom1, 3 * iatom2) = zz;

            for (auto ip = 0; ip < 3; ++ip) {
                nruter[ip + 1].block<3, 3>(3 * iatom1, 3 * iatom2) =
                    this->born.born[iatom1].row(ip).transpose() * z2 +
                    z1.transpose() * this->born.born[iatom2].row(ip) +
                    -2. * zz * this->born.epsilon.row(ip).dot(uq) / epsilon;
            }
        }
    }
    for (auto i = 0; i < 4; ++i) {
        nruter[i] /= this->massmatrix;
        nruter[i] *=
            prefactor / epsilon / this->V / this->na / this->nb / this->nc;
    }
    return nruter;
}


/// Pairwise summation of a vector of matrices or arrays. This reduces the
/// expected error versus a standard summation.
///
/// @param[inout] v - vector of operands. All of them must have the same size.
/// The vector will be overwritten.
/// @return the result of the sum
template <typename T> T eigen_pairwise_sum(std::vector<T>& v) {
    std::size_t last = v.size();
    T nruter;
    if (last == 0) {
        nruter.fill(0.);
    }
    else {
        while (last != 1) {
            std::size_t half = last / 2;
            for (std::size_t i = 0; i < half; ++i) {
                v[i] += v[i + half];
            }
            if (last % 2 == 0) {
                last = half;
            }
            else {
                v[half] = v[last - 1];
                last = half + 1;
            }
        }
        nruter = v[0];
    }
    return nruter;
}


std::array<Eigen::MatrixXcd, 4> Dynamical_matrix_builder::build(
    const Eigen::Ref<const Eigen::Vector3d>& q) const {
    Eigen::MatrixXd qbzs = this->structure.map_to_firstbz(q);
    Eigen::Vector3d qbz{qbzs.col(0)};
    auto nblocks = this->blocks.size();
    auto ndof = this->blocks[0].cols();
    auto coefficients = this->get_exponentials(q);
    // The nonanalytic correction is never applied at Gamma or at points
    // on the surface of the BZ.
    const bool nonanalytic =
        this->nonanalytic && !almost_equal(0., qbz.norm()) && qbzs.cols() == 1;

    std::array<Eigen::ArrayXXd, 4> nac;
    if (nonanalytic) {
        nac = this->build_nac(q);
    }

    std::array<Eigen::MatrixXcd, 4> nruter;
    std::vector<Eigen::MatrixXcd> terms;
    Eigen::MatrixXcd term;

    for (auto i = 0; i < 4; i++) {
        nruter[i].setZero(ndof, ndof);
    }
    for (decltype(nblocks) i = 0; i < nblocks; ++i) {
        term.array() = coefficients(i) * this->blocks[i];
        if (nonanalytic) {
            term.array() += coefficients(i) * this->masks[i].array() * nac[0];
        }
        terms.emplace_back(term);
    }
    nruter[0] = eigen_pairwise_sum<Eigen::MatrixXcd>(terms);
    for (auto j = 0; j < 3; ++j) {
        terms.clear();
        for (decltype(nblocks) i = 0; i < nblocks; ++i) {
            term.array() = -this->cpos(j, i) * coefficients(i) *
                           this->blocks[i] * constants::imud;
            if (nonanalytic) {
                term.array() -= this->cpos(j, i) * coefficients(i) *
                                this->masks[i].array() * nac[0].array() *
                                constants::imud;
                term.array() += coefficients(i) * this->masks[i].array() *
                                nac[j + 1].array();
            }
            terms.emplace_back(term);
        }
        nruter[j + 1] = eigen_pairwise_sum<Eigen::MatrixXcd>(terms);
    }
    return nruter;
}


/// Choose a unique base of eigenvectors in a degenerate subspace using
/// perturbation theory.
///
/// Given an arbitrary basis of a degenerate eigenvector space and a suitable
/// perturbation matrix, this function applies an orthogonal transformation
/// to the basis so that the perturbation does not mix the states of the final
/// basis.
/// @param[in] dDdq - perturbation matrix (normally related to a group velocity
/// operator)
/// @param[inout] eigvecs - original basis
/// @return the new basis
Eigen::MatrixXcd solve_degeneracy(
    const Eigen::Ref<const Eigen::MatrixXcd>& pert,
    const Eigen::Ref<const Eigen::MatrixXcd>& eigvecs) {
    Eigen::MatrixXcd confusion = eigvecs.adjoint() * pert * eigvecs;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(confusion);
    return eigvecs * solver.eigenvectors();
}


std::unique_ptr<Spectrum_at_point> Dynamical_matrix_builder::get_spectrum(
    const Eigen::Ref<const Eigen::Vector3d>& q) const {
    auto ndof = this->blocks[0].cols();

    // To try and avoid violations of the space group symmetry, we always
    // obtain a standarized representative of the orbit of the q point provided,
    // compute the spectrum there, and then rotate the wave functions back to
    // the original q point.

    // Look for the representative point and prepare the rotation.
    Eigen::Vector3d direct_q(this->rlattvec.colPivHouseholderQr().solve(q));
    alma::Q_orbit orbit(direct_q, this->symmetries);
    auto i_repr = orbit.get_representative_index();
    auto i_rot = orbit.i_rot[i_repr];
    Eigen::Vector3d direct_q_repr = orbit.q_images[i_repr];
    Eigen::Vector3d c_q_repr = structure.rlattvec * direct_q_repr;
    Eigen::MatrixXcd big_rotation = alma::calc_big_rotation_matrix(
        orbit, this->structure, this->symmetries);

    // Obtain the dynamical matrix at the representative point and
    // diagonalize it.
    auto matrices = this->build(c_q_repr);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(matrices[0]);
    Eigen::VectorXd omega2 = solver.eigenvalues();
    Eigen::MatrixXcd wfs = solver.eigenvectors();

    // As scalars, frequencies can be copied between points without any
    // further transformation.
    Eigen::ArrayXd omega(ndof);
    for (auto i = 0; i < omega.size(); ++i) {
        omega(i) = alma::ssqrt(omega2(i));
    }


    // Compute the eigenvalues and group velocities at the representative point.
    Eigen::MatrixXd vg(Eigen::MatrixXd::Zero(3, ndof));
    auto start = 0;
    // Degenerate subspaces are treated together in order to have a univocal,
    // and hopefully physically correct, estimate of the group velocities.
    // The x component of the group velocity is estimated using the set of
    // eigenvectors that diagonalize d D / d q_x over the degenerate subspace,
    // and so on. This is, however, not enough for triple degeneracies and
    // will also fail for some particular orientations. There is room for
    // improvement (e.g.: avoiding degenerate directions).
    for (auto i = 1; i <= omega.size(); ++i) {
        if (i == omega.size() || !almost_equal(omega(i), omega(start))) {
            int dim = i - start;
            if (!almost_equal(omega(start), 0.)) {
                // Shortcut for non-degenerate cases.
                if (dim == 1) {
                    for (auto j = 0; j < 3; ++j)
                        vg(j, start) =
                            wfs.col(start)
                                .dot(matrices[j + 1] * wfs.col(start))
                                .real();
                    vg.col(start) /= (2. * omega(start));
                }
                // General implementation.
                else {
                    for (auto j = 0; j < 3; ++j) {
                        Eigen::MatrixXcd vectors = solve_degeneracy(
                            matrices[j + 1], wfs.block(0, start, ndof, dim));
                        for (auto l = 0; l < dim; ++l) {
                            vg(j, start + l) =
                                vectors.col(l)
                                    .dot(matrices[j + 1] * vectors.col(l))
                                    .real();
                        }
                    }
                    vg.block(0, start, 3, dim) /= (2. * omega(start));
                    // Finally, choose an arbitrary but unique set of
                    // eigenvectors, to reduce the variability of the results
                    // across platforms.
                    Eigen::MatrixXcd directional =
                        1. * matrices[1] + 2. * matrices[2] + 3. * matrices[3];
                    wfs.block(0, start, ndof, dim) =
                        solve_degeneracy(directional,
                                         wfs.block(0, start, ndof, dim))
                            .eval();
                }
            }
            start = i;
        }
    }

    // Rotate the eigenvectors to the original point. Take time reversal
    // symmetry into account as well.
    wfs = big_rotation * wfs;
    vg = this->symmetries.unrotate_v(vg, i_rot);
    if (orbit.time_reversal[i_repr]) {
        wfs = wfs.conjugate();
        vg = -vg.eval();
    }
    // Although this is not strictly necessary, remove a global phase to
    // obtain a more usual form of the eigenvectors.
    for (decltype(ndof) i_mode = 0; i_mode < ndof; ++i_mode) {
        double global_phase = std::arg(wfs(0, i_mode));
        std::complex<double> exp_factor = std::complex<double>(
            std::cos(global_phase), std::sin(-global_phase));
        wfs.col(i_mode).array() *= exp_factor;
    }

    return std::make_unique<Spectrum_at_point>(omega, wfs, vg);
}
} // namespace alma
