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
/// Definitions corresponding to vc.hpp.

#include <Eigen/Dense>
#include <exceptions.hpp>
#include <periodic_table.hpp>
#include <vc.hpp>

namespace alma {
std::unique_ptr<Crystal_structure> vc_mix_structures(
    const std::vector<Crystal_structure>& components,
    const std::vector<double>& ratios) {
    auto ncomponents = components.size();

    // Perform some sanity checks. Note that even passing all of
    // those does not guarantee that the resulting virtual crystal
    // will make sense.
    if (ratios.size() != ncomponents)
        throw value_error("different numbers of components and ratios");

    if (ncomponents == 0)
        throw value_error("no components in the virtual crystal");

    for (auto r : ratios)
        if (r <= 0)
            throw value_error("all atomic ratios must be positive");
    auto natoms = components[0].get_natoms();

    for (decltype(ncomponents) i = 1; i < ncomponents; ++i) {
        if (components[i].get_natoms() != natoms)
            throw value_error("all components must have the same"
                              " number of atoms");
    }
    // Build the weighted lattice vectors.
    double total{ratios[0]};
    Eigen::Matrix3d lattvec(ratios[0] * components[0].lattvec);

    for (decltype(ncomponents) i = 1; i < ncomponents; ++i) {
        total += ratios[i];
        lattvec += ratios[i] * components[i].lattvec;
    }
    lattvec /= total;
    // Build the weighted atomic positions.
    Eigen::Matrix<double, 3, Eigen::Dynamic> positions(ratios[0] *
                                                       components[0].positions);

    for (decltype(ncomponents) i = 1; i < ncomponents; ++i)
        positions += ratios[i] * components[i].positions;
    positions /= total;
    // And the set of virtual elements.
    std::vector<std::string> vnames;
    std::vector<int> numbers;

    for (decltype(natoms) ia = 0; ia < natoms; ++ia) {
        std::vector<std::string> symbols;

        for (auto& c : components)
            symbols.emplace_back(c.get_element(ia));
        // Adjacent atoms belonging to the same element are grouped
        // together.
        auto name = Virtual_element(symbols, ratios).get_name();

        if ((vnames.size() == 0) || (vnames.back() != name)) {
            vnames.emplace_back(name);
            numbers.emplace_back(1);
        }
        else
            ++numbers.back();
    }
    return std::make_unique<Crystal_structure>(
        lattvec, positions, vnames, numbers);
}


std::unique_ptr<Dielectric_parameters> vc_mix_dielectric_parameters(
    const std::vector<Dielectric_parameters>& components,
    const std::vector<double>& ratios) {
    auto ncomponents = components.size();

    // Basic sanity checks.
    if (ratios.size() != ncomponents)
        throw value_error("different numbers of components and ratios");

    if (ncomponents == 0)
        throw value_error("no components in the virtual crystal");

    for (auto r : ratios)
        if (r <= 0)
            throw value_error("all atomic ratios must be positive");
    auto nborn = components[0].born.size();
    double total{ratios[0]};
    Eigen::Matrix3d epsilon{ratios[0] * components[0].epsilon};
    std::vector<Eigen::MatrixXd> born;

    for (decltype(nborn) ib = 0; ib < nborn; ++ib)
        born.emplace_back(Eigen::MatrixXd{ratios[0] * components[0].born[ib]});

    for (decltype(ncomponents) i = 1; i < ncomponents; ++i) {
        if (components[i].born.size() != nborn)
            throw value_error("all components must contain the same"
                              " number of Born charge tensors");
        total += ratios[i];
        epsilon += ratios[i] * components[i].epsilon;

        for (decltype(nborn) ib = 0; ib < nborn; ++ib)
            born[ib] += ratios[i] * components[i].born[ib];
    }
    epsilon /= total;

    for (auto& b : born)
        b /= total;
    return std::make_unique<Dielectric_parameters>(born, epsilon);
}


std::unique_ptr<Harmonic_ifcs> vc_mix_harmonic_ifcs(
    const std::vector<Harmonic_ifcs>& components,
    const std::vector<double>& ratios) {
    auto ncomponents = components.size();

    // Basic sanity checks.
    if (ratios.size() != ncomponents)
        throw value_error("different numbers of components and ratios");

    if (ncomponents == 0)
        throw value_error("no components in the virtual crystal");

    for (auto r : ratios)
        if (r <= 0)
            throw value_error("all atomic ratios must be positive");
    // Only the ifcs themselves are averaged. The remainin
    // attributes must be common to all objects.
    double total{ratios[0]};
    auto na = components[0].na;
    auto nb = components[0].nb;
    auto nc = components[0].nc;
    auto nblocks = components[0].ifcs.size();
    std::vector<Triple_int> pos{components[0].pos};
    std::vector<Eigen::MatrixXd> blocks;

    for (decltype(nblocks) ib = 0; ib < nblocks; ++ib)
        blocks.emplace_back(
            Eigen::MatrixXd{ratios[0] * components[0].ifcs[ib]});

    for (decltype(ncomponents) i = 1; i < ncomponents; ++i) {
        if (components[i].ifcs.size() != nblocks)
            throw value_error("all components must contain the same"
                              " number of IFC matrices");
        total += ratios[i];

        for (decltype(nblocks) ib = 0; ib < nblocks; ++ib) {
            if (components[i].pos[ib] != pos[ib])
                throw value_error("all components must contain IFC matrices"
                                  " for the same cell pairs");
            blocks[ib] += ratios[i] * components[i].ifcs[ib];
        }
    }

    for (auto& b : blocks)
        b /= total;
    return std::make_unique<Harmonic_ifcs>(pos, blocks, na, nb, nc);
}


std::unique_ptr<std::vector<Thirdorder_ifcs>> vc_mix_thirdorder_ifcs(
    const std::vector<std::vector<Thirdorder_ifcs>>& components,
    const std::vector<double>& ratios) {
    auto ncomponents = components.size();

    // Basic sanity checks.
    if (ratios.size() != ncomponents)
        throw value_error("different numbers of components and ratios");

    if (ncomponents == 0)
        throw value_error("no components in the virtual crystal");

    for (auto r : ratios)
        if (r <= 0)
            throw value_error("all atomic ratios must be positive");
    auto nifcs = components[0].size();

    for (decltype(ncomponents) i = 1; i < ncomponents; ++i)
        if (components[i].size() != nifcs)
            throw value_error("all components must contain the same number"
                              " of IFC blocks");
    // Create the final object.
    auto nruter = std::make_unique<std::vector<Thirdorder_ifcs>>();
    // We average the Cartesian coordinates of the unit cells and
    // the IFCs. We require the atom indices to be common to all
    // inputs.
    double total = std::accumulate(ratios.begin(), ratios.end(), 0.);

    for (decltype(nifcs) ic = 0; ic < nifcs; ++ic) {
        auto i = components[0][ic].i;
        auto j = components[0][ic].j;
        auto k = components[0][ic].k;
        Eigen::VectorXd rj{ratios[0] * components[0][ic].rj};
        Eigen::VectorXd rk{ratios[0] * components[0][ic].rk};

        for (decltype(ncomponents) ir = 1; ir < ncomponents; ++ir) {
            if ((components[ir][ic].i != i) || (components[ir][ic].j != j) ||
                (components[ir][ic].k != k))
                throw value_error("all components must contain IFC blocks"
                                  " for the same atom triplets");
            rj += ratios[ir] * components[ir][ic].rj;
            rk += ratios[ir] * components[ir][ic].rk;
        }
        rj /= total;
        rk /= total;
        Thirdorder_ifcs block{rj, rk, i, j, k};

        for (std::size_t alpha = 0; alpha < 3; ++alpha)
            for (std::size_t beta = 0; beta < 3; ++beta)
                for (std::size_t gamma = 0; gamma < 3; ++gamma) {
                    block.ifc(alpha, beta, gamma) =
                        ratios[0] * components[0][ic].ifc(alpha, beta, gamma);

                    for (decltype(ncomponents) ir = 1; ir < ncomponents; ++ir)
                        block.ifc(alpha, beta, gamma) +=
                            ratios[ir] *
                            components[ir][ic].ifc(alpha, beta, gamma);
                    block.ifc(alpha, beta, gamma) /= total;
                }
        nruter->emplace_back(block);
    }
    return nruter;
}
} // namespace alma
