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
/// Definitions corresponding to symmetry.hpp.

extern "C" {
#include <spglib/spglib.h>
}

#include <symmetry.hpp>
#include <Eigen/LU>

namespace alma {
Symmetry_operations::Symmetry_operations(const Crystal_structure& structure,
                                         double _symprec)
    : symprec(_symprec) {
    int natoms = structure.get_natoms();
    // Create some C-friendly structures from the data
    // provided.
    double c_lattvec[3][3];

    for (auto i = 0; i < 3; ++i)
        for (auto j = 0; j < 3; ++j)
            c_lattvec[i][j] = structure.lattvec(i, j);
    std::vector<int> c_types;

    for (decltype(structure.numbers.size()) i = 0; i < structure.numbers.size();
         ++i)
        for (auto j = 0; j < structure.numbers[i]; ++j)
            c_types.push_back(i);
    auto c_positions = new double[natoms][3];

    for (decltype(natoms) i = 0; i < natoms; ++i)
        for (auto j = 0; j < 3; ++j)
            c_positions[i][j] = structure.positions(j, i);
    // Obtain a SpglibDataset structure containing
    // all information.
    SpglibDataset* data = spg_get_dataset(
        c_lattvec, c_positions, c_types.data(), natoms, this->symprec);

    if (data == nullptr)
        throw value_error("spglib's spg_get_dataset"
                          " returned a NULL pointer");
    // Copy all interesting information out of the
    // SpglibDataset.
    this->sg_number = data->spacegroup_number;
    this->sg_symbol = data->international_symbol;

    for (decltype(natoms) i = 0; i < natoms; ++i) {
        this->wyckoff += constants::alphabet[data->wyckoffs[i]];
        this->equivalences.push_back(data->equivalent_atoms[i]);
    }
    auto nsymm = data->n_operations;

    for (decltype(nsymm) i = 0; i < nsymm; ++i) {
        Eigen::VectorXd trans(3);

        for (auto j = 0; j < 3; ++j)
            trans(j) = data->translations[i][j];
        this->translations.emplace_back(trans);
        Eigen::MatrixXd rot(3, 3);

        for (auto j = 0; j < 3; ++j)
            for (auto k = 0; k < 3; ++k)
                rot(j, k) = data->rotations[i][j][k];
        this->rotations.emplace_back(rot);
    }
    // Free the allocated memory.
    delete[] c_positions;
    spg_free_dataset(data);

    // Compute the Cartesian representations of rotations
    // and translations.
    for (const auto& t : translations)
        this->ctranslations.emplace_back(structure.lattvec * t);
    auto solver = structure.lattvec.transpose().colPivHouseholderQr();

    for (const auto& r : rotations)
        this->crotations.emplace_back(
            (solver.solve(r.transpose()) * structure.lattvec.transpose())
                .transpose());
    // Express the rotations in direct reciprocal coordinates
    // as well.
    auto hermitian = structure.lattvec.transpose() * structure.lattvec;
    auto qsolver = hermitian.colPivHouseholderQr();

    for (const auto& r : rotations)
        this->qrotations.emplace_back(
            (qsolver.solve(r.transpose()) * hermitian).transpose());
    // Fill symmetry_map.
    this->fill_map(structure);
    // Compute the determinant of each rotation matrix.
    this->determinants.resize(nsymm);

    for (decltype(nsymm) i = 0; i < nsymm; ++i)
        this->determinants[i] = rotations[i].determinant();

    // Compute the displacements of the original image of each
    // atom.
    for (decltype(nsymm) i = 0; i < nsymm; ++i) {
        Eigen::MatrixXd delta(3, natoms);

        for (decltype(natoms) j = 0; j < natoms; ++j) {
            std::size_t jp = this->symmetry_map[j][i];
            delta.col(j) =
                structure.positions.col(jp) - structure.positions.col(j);
        }
        this->displacements.emplace_back(delta);
    }
}


void Symmetry_operations::fill_map(const Crystal_structure& structure) {
    auto natoms = static_cast<std::size_t>(structure.get_natoms());
    auto nops = this->get_nsym();
    Eigen::Vector3d r;
    Eigen::Vector3d newr;
    Eigen::Vector3d delta;

    for (std::size_t ia = 0; ia < natoms; ++ia) {
        r = structure.positions.col(ia);
        std::vector<std::size_t> images(nops);

        // Try all the symmetry operations on each atom.
        for (decltype(nops) iop = 0; iop < nops; ++iop) {
            newr = this->transform_v(r, iop);
            // Look for the corresponding atom.
            bool found = false;

            for (std::size_t ja = 0; ja < natoms; ++ja) {
                delta = newr - structure.positions.col(ja);

                for (auto i = 0; i < 3; ++i)
                    delta(i) -= std::round(delta(i));

                if (delta.lpNorm<Eigen::Infinity>() < this->symprec) {
                    images[iop] = ja;
                    found = true;
                    break;
                }
            }

            if (!found) {
                throw value_error("inconsistent symmetries");
            }
        }
        this->symmetry_map.emplace_back(images);
    }
}


std::vector<std::vector<Transformed_pair>>
Symmetry_operations::get_pair_classes() const {
    auto natoms = this->symmetry_map.size();
    auto nops = this->translations.size();

    // Only internal translations are considered.
    Eigen::Matrix3d eye{Eigen::Matrix3d::Identity()};

    std::vector<bool> toskip(nops);

    for (std::size_t iop = 0; iop < nops; ++iop)
        toskip[iop] = (this->rotations[iop] - eye).norm() >
                      std::numeric_limits<float>::epsilon();
    std::vector<std::vector<Transformed_pair>> nruter;

    for (std::size_t a1 = 0; a1 < natoms; ++a1)
        for (std::size_t a2 = a1; a2 < natoms; ++a2) {
            std::size_t iop;

            for (iop = 1; iop < nops; ++iop) {
                // Check that the operation is an internal
                // translation.
                if (toskip[iop] ||
                    !almost_equal(0.,
                                  (this->displacements[iop].col(a1) -
                                   this->displacements[iop].col(a2))
                                      .norm()))
                    continue;
                // And look for an equivalence class containing
                // the image of the pair.
                std::size_t b1 = this->symmetry_map[a1][iop];
                std::size_t b2 = this->symmetry_map[a2][iop];
                bool found = false;

                for (auto& eq : nruter) {
                    if ((b1 == eq[0].pair[0]) && (b2 == eq[0].pair[1])) {
                        eq.push_back(Transformed_pair(a1, a2, iop, false));
                        found = true;
                        break;
                    }
                    else if ((b1 == eq[1].pair[0]) && (b2 == eq[1].pair[1])) {
                        eq.push_back(Transformed_pair(a1, a2, iop, true));
                        found = true;
                        break;
                    }
                }

                if (found)
                    break;
            }

            // If a suitable equivalence class is not found, create
            // a new one.
            if (iop == nops) {
                nruter.push_back(std::vector<Transformed_pair>());
                nruter.back().push_back(Transformed_pair(a1, a2, 0, false));
            }
        }

    // For each pair (a1, a2) with a1 != a2, add (a2, a1) to the
    // same class.
    for (auto& eq : nruter) {
        std::size_t s = eq.size();

        for (std::size_t i = 0; i < s; ++i) {
            if (eq[i].pair[0] != eq[i].pair[1])
                eq.push_back(Transformed_pair(eq[i].pair[1],
                                              eq[i].pair[0],
                                              eq[i].operation,
                                              !eq[i].exchange));
            ;
        }
    }
    return nruter;
}

Eigen::MatrixXcd calc_big_rotation_matrix(const Q_orbit& orbit,
                                          const Crystal_structure& structure,
                                          const Symmetry_operations& symms) {
    std::size_t n_atoms = structure.get_natoms();
    Eigen::MatrixXcd nruter(Eigen::MatrixXcd::Zero(3 * n_atoms, 3 * n_atoms));
    auto pos = orbit.get_representative_index();
    Eigen::Vector3d q_repr = orbit.q_images[pos];
    auto i_rot = orbit.i_rot[pos];
    Eigen::Vector3d trans = symms.translations[i_rot];
    Eigen::Matrix3d d_rot = symms.rotations[i_rot];
    Eigen::Matrix3d c_rot = symms.crotations[i_rot];
    for (decltype(n_atoms) i_atom = 0; i_atom < n_atoms; ++i_atom) {
        auto j_atom = symms.map_atom(i_atom, i_rot);
        Eigen::Vector3d old_pos = structure.positions.col(i_atom);
        Eigen::Vector3d new_pos = structure.positions.col(j_atom);
        Eigen::Vector3d delta = new_pos - (d_rot * old_pos + trans);
        double arg = 2. * alma::constants::pi * q_repr.dot(delta);
        std::complex<double> exp_factor(std::cos(arg), std::sin(arg));
        nruter.block<3, 3>(3 * j_atom, 3 * i_atom) = exp_factor * c_rot;
    }
    return nruter.inverse();
}
} // namespace alma
