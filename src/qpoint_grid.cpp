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
/// Definitions corresponding to qpoint_grid.hpp.

#include <map>
#include <unordered_set>
#if BOOST_VERSION >= 106700
#include <boost/container_hash/hash.hpp>
#else
#include <boost/functional/hash.hpp>
#endif
#include <utilities.hpp>
#include <exceptions.hpp>
#include <qpoint_grid.hpp>

namespace std {
/// Trivial implementation of std::hash for arrays,
/// required to create an unordered_set of arrays.
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
Gamma_grid::Gamma_grid(const Crystal_structure& poscar,
                       const Symmetry_operations& symms,
                       const Harmonic_ifcs& force_constants,
                       int _na,
                       int _nb,
                       int _nc)
    : na(_na), nb(_nb), nc(_nc), nqpoints(_na * _nb * _nc),
      rlattvec(poscar.rlattvec),
      dq(rlattvec.array().rowwise() /
         Eigen::Map<Eigen::Array3i>(
             std::array<int, 3>({{_na, _nb, _nc}}).data())
             .cast<double>()
             .transpose()) {
    if (std::min({_na, _nb, _nc}) <= 0)
        throw value_error("all grid dimensions must be positive");
    this->initialize_cpos();
    this->fill_equivalences(symms);
    this->fill_map(symms);
    Dynamical_matrix_builder builder(poscar, symms, force_constants);
    boost::mpi::communicator world;
    auto my_spectrum = this->compute_my_spectrum(builder, symms, world);
    std::vector<decltype(my_spectrum)> all_spectra;
    boost::mpi::all_gather(world, my_spectrum, all_spectra);

    for (auto& s : all_spectra)
        this->spectrum.insert(this->spectrum.end(), s.begin(), s.end());
}

Gamma_grid::Gamma_grid(const Crystal_structure& poscar,
                       const Symmetry_operations& symms,
                       const Harmonic_ifcs& force_constants,
                       const Dielectric_parameters& born,
                       int _na,
                       int _nb,
                       int _nc)
    : na(_na), nb(_nb), nc(_nc), nqpoints(_na * _nb * _nc),
      rlattvec(poscar.rlattvec),
      dq(rlattvec.array().rowwise() /
         Eigen::Map<Eigen::Array3i>(
             std::array<int, 3>({{_na, _nb, _nc}}).data())
             .cast<double>()
             .transpose()) {
    if (std::min({_na, _nb, _nc}) <= 0)
        throw value_error("all grid dimensions must be positive");
    this->initialize_cpos();
    this->fill_equivalences(symms);
    this->fill_map(symms);
    Dynamical_matrix_builder builder(poscar, symms, force_constants, born);
    boost::mpi::communicator world;
    auto my_spectrum = this->compute_my_spectrum(builder, symms, world);
    std::vector<decltype(my_spectrum)> all_spectra;
    boost::mpi::all_gather(world, my_spectrum, all_spectra);

    for (auto& s : all_spectra)
        this->spectrum.insert(this->spectrum.end(), s.begin(), s.end());
}

void Gamma_grid::initialize_cpos() {
    this->cpos.resize(3, this->nqpoints);
    std::size_t iq = 0;

    for (auto ia = 0; ia < this->na; ++ia)
        for (auto ib = 0; ib < this->nb; ++ib)
            for (auto ic = 0; ic < this->nc; ++ic) {
                this->cpos(0, iq) = ia;
                this->cpos(1, iq) = ib;
                this->cpos(2, iq) = ic;
                ++iq;
            }
    this->cpos.row(0) /= this->na;
    this->cpos.row(1) /= this->nb;
    this->cpos.row(2) /= this->nc;
    this->cpos = this->rlattvec * this->cpos;
}


std::vector<Spectrum_at_point> Gamma_grid::compute_my_spectrum(
    const Dynamical_matrix_builder& factory,
    const Symmetry_operations& symms,
    const boost::mpi::communicator& communicator) {
    auto nprocs = communicator.size();
    auto my_id = communicator.rank();
    auto limits = my_jobs(this->nqpoints, nprocs, my_id);

    std::vector<Spectrum_at_point> my_spectrum;
    my_spectrum.reserve(limits[1] - limits[0]);

    // Compute the spectrum for each q point assigned to this
    // MPI process.
    for (auto iq = limits[0]; iq < limits[1]; ++iq) {
        my_spectrum.emplace_back(*factory.get_spectrum(this->cpos.col(iq)));
    }
    // Symmetrize the group velocities at each point.
    for (auto iq = limits[0]; iq < limits[1]; ++iq) {
        my_spectrum[iq - limits[0]].vg =
            this->copy_symmetry(
                    iq, symms, my_spectrum[iq - limits[0]].vg.matrix())
                .array();
    }

    return my_spectrum;
}


void Gamma_grid::fill_map(const Symmetry_operations& symms) {
    auto nops = symms.get_nsym();
    Eigen::Vector3d q;
    Eigen::Vector3d newq;
    Eigen::Vector3i intq;

    for (std::size_t iq = 0; iq < this->nqpoints; ++iq) {
        // A value of nqpoints in the this->symmetry map
        // signals an invalid symmetry operation.
        std::vector<std::size_t> images(2 * nops, this->nqpoints);
        auto indices = this->one_to_three(iq);
        q << indices[0] / static_cast<double>(this->na),
            indices[1] / static_cast<double>(this->nb),
            indices[2] / static_cast<double>(this->nc);

        // Try all the symmetry operations on each q point.
        for (decltype(nops) iop = 0; iop < nops; ++iop) {
            newq = symms.rotate_q(q, iop);
            newq(0) *= this->na;
            newq(1) *= this->nb;
            newq(2) *= this->nc;

            for (auto i = 0; i < 3; ++i)
                intq(i) = static_cast<int>(std::round(newq(i)));

            // Discard an operation if it does not map the q point
            // onto another valid q point. Some symmetry operations
            // may be incompatible with the q-point grid.
            if (!almost_equal(0., (newq - intq.cast<double>()).squaredNorm()))
                continue;
            // Compute the index of the q point obtained as the
            // image of q through the symmetry operation, as well
            // as the index of the q point obtained through time
            // reversal symmetry. Append both to the images vector.
            images[2 * iop] = this->three_to_one({{intq(0), intq(1), intq(2)}});
            images[2 * iop + 1] =
                this->three_to_one({{-intq(0), -intq(1), -intq(2)}});
        }
        this->symmetry_map.emplace_back(images);
    }
}


void Gamma_grid::fill_equivalences(const Symmetry_operations& symms) {
    auto nops = symms.get_nsym();
    Eigen::Vector3d q;
    Eigen::Vector3d newq;
    Eigen::Vector3i intq;

    this->equivalences.emplace_back(std::vector<std::size_t>{0});

    // For each q point.
    for (std::size_t iq = 1; iq < this->nqpoints; ++iq) {
        auto indices = this->one_to_three(iq);
        q << indices[0] / static_cast<double>(this->na),
            indices[1] / static_cast<double>(this->nb),
            indices[2] / static_cast<double>(this->nc);
        // Try all the symmetry operations.
        bool found = false;

        for (decltype(nops) iop = 0; iop < nops; ++iop) {
            newq = symms.rotate_q(q, iop);
            newq(0) *= this->na;
            newq(1) *= this->nb;
            newq(2) *= this->nc;

            for (auto i = 0; i < 3; ++i)
                intq(i) = static_cast<int>(std::round(newq(i)));

            // Discard an operation if it does not map the q point
            // onto another valid q point. Some symmetry operations
            // may be incompatible with the q-point grid.
            if (!almost_equal(0., (newq - intq.cast<double>()).squaredNorm()))
                continue;
            // Compute the index of the q point obtained as the
            // image of q through the symmetry operation.
            auto candidate1 = this->three_to_one({{intq(0), intq(1), intq(2)}});
            // And the index of the q point obtained through time
            // reversal symmetry.
            auto candidate2 =
                this->three_to_one({{-intq(0), -intq(1), -intq(2)}});

            // If either of candidate1 or candidate2 are the first
            // elements of an existing equivalence class, add iq
            // to that class.
            for (auto& c : this->equivalences)
                if ((candidate1 == c[0]) || (candidate2 == c[0])) {
                    c.emplace_back(iq);
                    found = true;
                    break;
                }

            if (found)
                break;
        }

        // If iq cannot be assigned to any existing equivalence class,
        // start a new one.
        if (!found) {
            auto size = this->equivalences.size();
            this->equivalences.resize(size + 1);
            this->equivalences[size].emplace_back(iq);
        }
    }

    this->fill_parentlookup();
}


void Gamma_grid::fill_parentlookup() {
    this->parentlookup.resize(this->nqpoints);

    for (std::size_t nclass = 0; nclass < this->equivalences.size(); nclass++) {
        for (std::size_t nmember = 0;
             nmember < this->equivalences[nclass].size();
             nmember++) {
            this->parentlookup.at(this->equivalences[nclass].at(nmember)) =
                this->equivalences[nclass].at(0);
        }
    }
}


std::size_t Gamma_grid::getParentIdx(std::size_t iq) const {
    if (iq >= this->nqpoints)
        throw value_error("invalid q point index");

    return this->parentlookup.at(iq);
}


std::size_t Gamma_grid::getSymIdxToParent(std::size_t iq) const {
    if (iq >= this->nqpoints)
        throw value_error("invalid q point index");

    std::size_t iq_parent = this->parentlookup.at(iq);

    std::size_t i;

    for (i = 0; i < this->symmetry_map[iq].size(); ++i) {
        auto iq_image = this->symmetry_map[iq][i];
        if (iq_image == iq_parent)
            return i;
    }
    throw exception("some point is not in symmetry_map");
}


Gamma_grid::Gamma_grid(const Crystal_structure& poscar,
                       const Symmetry_operations& symms,
                       int _na,
                       int _nb,
                       int _nc)
    : na(_na), nb(_nb), nc(_nc), nqpoints(_na * _nb * _nc),
      rlattvec(poscar.rlattvec),
      dq(rlattvec.array().rowwise() /
         Eigen::Map<Eigen::Array3i>(
             std::array<int, 3>({{_na, _nb, _nc}}).data())
             .cast<double>()
             .transpose()) {
    if (std::min({_na, _nb, _nc}) <= 0)
        throw value_error("all grid dimensions must be positive");
    this->initialize_cpos();
    this->fill_equivalences(symms);
    this->fill_map(symms);
}

std::vector<std::array<std::size_t, 2>> Gamma_grid::equivalent_qpairs(
    const std::array<std::size_t, 2>& original) const {
    auto iq1 = original[0];
    auto iq2 = original[1];

    if ((iq1 >= this->nqpoints) || (iq2 >= this->nqpoints))
        throw value_error("invalid q point index");

    std::unordered_set<std::array<std::size_t, 2>> unique;

    // Loop over the images of both q points looking
    // for all unique equivalent pairs.
    for (std::size_t i = 0; i < this->symmetry_map[0].size(); ++i) {
        auto jq1 = this->symmetry_map[iq1][i];
        auto jq2 = this->symmetry_map[iq2][i];
        std::array<std::size_t, 2> candidate({{jq1, jq2}});
        unique.emplace(candidate);
    }

    return std::vector<std::array<std::size_t, 2>>(unique.begin(),
                                                   unique.end());
}


std::vector<std::array<std::size_t, 3>> Gamma_grid::equivalent_qtriplets(
    const std::array<std::size_t, 3>& original) const {
    auto iq1 = original[0];
    auto iq2 = original[1];
    auto iq3 = original[2];

    if ((iq1 >= this->nqpoints) || (iq2 >= this->nqpoints) ||
        (iq3 >= this->nqpoints))
        throw value_error("invalid q point index");

    std::unordered_set<std::array<std::size_t, 3>> unique;

    // Loop over the images of all three q points looking
    // for all unique equivalent triplets.
    for (std::size_t i = 0; i < this->symmetry_map[0].size(); ++i) {
        auto jq1 = this->symmetry_map[iq1][i];
        auto jq2 = this->symmetry_map[iq2][i];
        auto jq3 = this->symmetry_map[iq3][i];
        std::array<std::size_t, 3> candidate({{jq1, jq2, jq3}});
        if (jq1 < this->nqpoints && jq2 < this->nqpoints &&
            jq3 < this->nqpoints) {
            unique.emplace(candidate);
        }
    }

    return std::vector<std::array<std::size_t, 3>>(unique.begin(),
                                                   unique.end());
}


std::vector<Triangle> Gamma_grid::get_triangles(std::size_t ia) const {
    std::vector<Triangle> triangles;
    int itriangle = 0;

    for (int ib = 0; ib < this->nb; ++ib)
        for (int ic = 0; ic < this->nc; ++ic) {
            std::array<int, 3> indices({{static_cast<int>(ia), ib, ic}});
            std::size_t i000 =
                this->three_to_one({{indices[0], indices[1], indices[2]}});
            std::size_t i010 =
                this->three_to_one({{indices[0], indices[1] + 1, indices[2]}});
            std::size_t i001 =
                this->three_to_one({{indices[0], indices[1], indices[2] + 1}});
            std::size_t i011 = this->three_to_one(
                {{indices[0], indices[1] + 1, indices[2] + 1}});
            triangles.emplace_back(Triangle({{i000, i010, i011}}));
            triangles.emplace_back(Triangle({{i000, i001, i011}}));
            itriangle += 2;
        }
    return triangles;
}
} // namespace alma
