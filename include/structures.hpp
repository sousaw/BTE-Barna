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

#pragma once

/// @file
/// Definitions of the basic data-handling classes in ALMA.

#include <map>
#include <array>
#include <vector>
#include <utility>
#include <numeric>
#include <limits>
#include <boost/math/special_functions/pow.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <cmakevars.hpp>
#include <constants.hpp>
#include <utilities.hpp>
#include <exceptions.hpp>
#include <periodic_table.hpp>

namespace alma {
// Forward declarations placed here for convenience.
class Symmetry_operations;
class Gamma_grid;
class Threeph_process;

/// Convenient shorthand for an array of three ints.
using Triple_int = std::array<int, 3>;

/// Specialized version of map whose keys are of class Triple_int.
template <typename T>
using Triple_int_map =
    std::map<Triple_int, T, Container_comparator<Triple_int>>;

/// Hold information about a crystal structure.
class Crystal_structure {
public:
    /// Lattice vectors in nm.
    const Eigen::Matrix3d lattvec;
    /// Atomic positions in lattice coordinates.
    const Eigen::Matrix<double, 3, Eigen::Dynamic> positions;
    /// Elements present in the structure.
    const std::vector<std::string> elements;
    /// Volume of the unit cell.
    const double V;
    /// Lattice vectors of the reciprocal lattice.
    const Eigen::MatrixXd rlattvec;
    /// Number of atoms of each element.
    ///
    /// The first numbers[0] atoms belong to elements[0], the next
    /// numbers[1] atoms belong to elements[1], etc.
    const std::vector<int> numbers;
    /// Basic constructor.
    Crystal_structure(Eigen::Matrix3d _lattvec,
                      Eigen::Matrix<double, 3, Eigen::Dynamic> _positions,
                      std::vector<std::string> _elements,
                      std::vector<int> _numbers)
        : lattvec(std::move(_lattvec)), positions(std::move(_positions)),
          elements(std::move(_elements)),
          V(std::fabs(this->lattvec.determinant())),
          rlattvec(2. * constants::pi * this->lattvec.inverse().transpose()),
          numbers(std::move(_numbers)) {
        std::partial_sum(this->numbers.begin(),
                         this->numbers.end(),
                         std::back_inserter(this->partsums));

        for (int i = 0; i < this->get_natoms(); ++i) {
            this->masses.emplace_back(alma::get_mass(this->get_element(i)));
            this->gfactors.emplace_back(
                alma::get_gfactor(this->get_element(i)));
        }
    }

    /// Specialized constructor allowing for custom masses.
    Crystal_structure(Eigen::Matrix3d _lattvec,
                      Eigen::Matrix<double, 3, Eigen::Dynamic> _positions,
                      std::vector<std::string> _elements,
                      std::vector<int> _numbers,
                      std::vector<double> _masses)
        : Crystal_structure(_lattvec, _positions, _elements, _numbers) {
        if (_masses.size() != static_cast<std::size_t>(this->get_natoms())) {
            throw value_error("one mass per atom must be provided");
        }
        for (int i = 0; i < this->get_natoms(); ++i) {
            if (_masses[i] <= 0.) {
                throw value_error("all masses must be positive");
            }
            this->masses[i] = _masses[i];
        }
    }

    /// Return the number of elements in the structure.
    inline std::vector<std::string>::size_type get_nelements() const {
        return this->elements.size();
    }


    /// Return the number of atoms in the motif.
    inline int get_natoms() const {
        return this->positions.cols();
    }
    /// Return the chemical symbol for an atom.
    ///
    /// @param[in] i - atom number
    /// @return the chemical symbol
    inline std::string get_element(int i) const {
        if (i >= this->get_natoms())
            throw value_error("invalid atom number");
        return this->elements[std::distance(
            this->partsums.begin(),
            std::lower_bound(
                this->partsums.begin(), this->partsums.end(), i + 1))];
    }


    /// Return the mass of the i-th atom.
    ///
    /// @param[in] i - atom number
    /// @return the mass in a.m.u
    inline double get_mass(int i) const {
        if (i >= this->get_natoms())
            throw value_error("invalid atom number");
        return this->masses[i];
    }


    /// Return the g factor of the i-th atom.
    ///
    /// @param[in] i - atom number
    /// @return the Pearson deviation coefficient of the i-th atom's
    /// mass
    inline double get_gfactor(int i) const {
        if (i >= this->get_natoms())
            throw value_error("invalid atom number");
        return this->gfactors[i];
    }


    /// Check if any atom belongs to a virtual element.
    ///
    /// @return true is the structure describes an alloy, or
    /// false otherwise.
    inline bool is_alloy() const {
        for (auto& e : this->elements)
            if (e.find(";") != std::string::npos)
                return true;

        return false;
    }

    /// Find all images of a q point in the first Brillouin zone.
    ///
    /// A point on the surface of the first BZ will have more than
    /// one possible image.
    /// @param[in] - original q point in Cartesian coordinates
    /// @return the Cartesian coordinates of all images of the point
    /// in the first Brillouin zone, as columns of a matrix.
    inline Eigen::MatrixXd map_to_firstbz(
        const Eigen::Ref<const Eigen::Vector3d>& q) const {
        constexpr int sbound = 3;
        // Step 1: find a single image using a standard approach
        Eigen::Vector3d qbz{this->rlattvec.colPivHouseholderQr().solve(q)};
        Eigen::Vector3d qbz_min;
        qbz -= qbz.array().round().matrix().eval();
        qbz = (this->rlattvec * qbz).eval();
        double n2 = qbz.squaredNorm();
        qbz_min = qbz;
        for (int i = -sbound; i <= sbound; ++i) {
            for (int j = -sbound; j <= sbound; ++j) {
                for (int k = -sbound; k <= sbound; ++k) {
                    Eigen::Vector3d qbzp{qbz + i * this->rlattvec.col(0) +
                                         j * this->rlattvec.col(1) +
                                         k * this->rlattvec.col(2)};
                    double n2p = qbzp.squaredNorm();
                    if (n2p < n2) {
                        n2 = n2p;
                        qbz_min = qbzp;
                    }
                }
            }
        }
        qbz = qbz_min;
        // Step 2: perform a search to find equivalent images.
        std::size_t found = 0;
        int ncols = boost::math::pow<3>(2 * sbound + 1);
        Eigen::MatrixXd nruter(3, ncols);
        for (int i = -sbound; i <= sbound; ++i) {
            for (int j = -sbound; j <= sbound; ++j) {
                for (int k = -sbound; k <= sbound; ++k) {
                    Eigen::Vector3d qbzp{qbz + i * this->rlattvec.col(0) +
                                         j * this->rlattvec.col(1) +
                                         k * this->rlattvec.col(2)};
                    double n2p = qbzp.squaredNorm();
                    if (alma::almost_equal(n2, n2p)) {
                        nruter.col(found) = qbzp;
                        ++found;
                    }
                }
            }
        }
        return nruter.leftCols(found);
    }


private:
    /// partsums[i] contains sum(numbers[0:i+1]).
    std::vector<int> partsums;
    /// Masses of each atom. Cached here to avoid costly
    /// calculations in the case of virtual elements.
    std::vector<double> masses;
    /// Pearson deviation coefficients of the atomic masses. Cached
    /// here to avoid costly calculations in the case of virtual
    /// elements.
    std::vector<double> gfactors;
};

/// Hold information about the harmonic interactions between atoms.
/// Normally T will be Eigen::MatrixXd since force constants are
/// real, but it might be useful to change it to Eigen::MatrixXcd
/// in order to operate in mixed real/reciprocal space.
template <class T> class General_harmonic_ifcs {
public:
    /// Coordinates of each unit cell for which constants
    /// are available.
    const std::vector<Triple_int> pos;
    /// Force constants between unit cell 0 and each unit cell.
    const std::vector<T> ifcs;
    /// Dimension of the supercell originally used for the IFC
    /// calculations along the first axis.
    const int na;
    /// Dimension of the supercell originally used for the IFC
    /// calculations along the second axis.
    const int nb;
    /// Dimension of the supercell originally used for the IFC
    /// calculations along the third axis.
    const int nc;
    /// Basic constructor.
    General_harmonic_ifcs(std::vector<Triple_int> _pos,
                          std::vector<T> _ifcs,
                          int _na,
                          int _nb,
                          int _nc)
        : pos(std::move(_pos)), ifcs(std::move(_ifcs)), na(_na), nb(_nb),
          nc(_nc) {
    }


    // Return the number of unit cells.
    inline std::vector<std::string>::size_type get_ncells() const {
        return this->pos.size();
    }
};

/// Specialization of General_harmonic_ifcs for the most common
/// use case.
using Harmonic_ifcs = General_harmonic_ifcs<Eigen::MatrixXd>;

/// Hold information about the polarization properties of the
/// structure.
class Dielectric_parameters {
public:
    /// Born charges.
    const std::vector<Eigen::MatrixXd> born;
    /// Dielectric tensor.
    const Eigen::Matrix3d epsilon;
    /// Basic constructor.
    Dielectric_parameters(std::vector<Eigen::MatrixXd> _born,
                          Eigen::Matrix3d _epsilon)
        : born(std::move(_born)), epsilon(std::move(_epsilon)) {
    }


    /// Default constructor. Create an empty object.
    Dielectric_parameters() {
    }
};

/// Phonopy-style atom index represented both as a single integer
/// and as four indices.
///
/// Objects from this class are not intended to be built
/// directly. Use a Supercell_index_builder instead.
class Supercell_index {
public:
    /// Atom index within the supercell.
    const int index;
    /// Unit cell index along the first axis.
    const int ia;
    /// Unit cell index along the second axis.
    const int ib;
    /// Unit cel index along the third axis.
    const int ic;
    /// Atom index within the unit cell.
    const int iatom;
    // Basic constructor.
    Supercell_index(const int _index,
                    const int _ia,
                    const int _ib,
                    const int _ic,
                    const int _iatom)
        : index(_index), ia(_ia), ib(_ib), ic(_ic), iatom(_iatom) {
    }


    /// Return an array of three integers {ia, ib, ic}.
    inline const Triple_int get_pos() const {
        return Triple_int({{this->ia, this->ib, this->ic}});
    }
};

/// Builder for Supercell_index objects sharing the same na, nb, nc.
class Supercell_index_builder {
public:
    /// Supercell dimension along the first axis.
    const int na;
    /// Supercell dimension along the second axis.
    const int nb;
    /// Supercell dimension along the third axis.
    const int nc;
    /// Number of atoms in the unit cell.
    const int natoms;
    /// Basic constructor.
    Supercell_index_builder(const int _na,
                            const int _nb,
                            const int _nc,
                            const int _natoms);
    /// Create a Supercell_index from a single integer.
    Supercell_index create_index(const int index) const;

    /// Create a Supercell_index from four integers.
    Supercell_index create_index(const int ia,
                                 const int ib,
                                 const int ic,
                                 const int iatom) const;

    /// Perform a bounds check and create a Supercell_index from a
    /// single integer.
    Supercell_index create_index_safely(const int index) const;

    /// Perform a bounds check and create a Supercell_index from
    /// four integers.
    Supercell_index create_index_safely(const int ia,
                                        const int ib,
                                        const int ic,
                                        const int iatom) const;
};

/// Class representing the anharmonic (third-order)
/// interaction between two atoms atoms.
///
/// Note that, since third-order IFCs tend to be expressed
/// in sparse formats, it is more natural to store each tensor
/// individually. There is no third-order equivalent to the
/// Harmonic_ifcs class. Vectors of Thirdorder_ifcs can be used
/// instead.
/// The first atom, i, is always assumed to be part of the first
/// unit cell, that is, R_i = {0., 0., 0.}.
class Thirdorder_ifcs {
public:
    /// Cartesian coordinates of the second unit cell.
    const Eigen::VectorXd rj;
    /// Cartesian coordinates of the third unit cell.
    const Eigen::VectorXd rk;
    /// Index of the first atom.
    const std::size_t i;
    /// Index of the second atom.
    const std::size_t j;
    /// Index of the third atom.
    const std::size_t k;
    /// Access a particular ifc using three indexes.
    ///
    /// Note that all indices must be positive and lower than 3,
    /// but that this condition is not checked.
    /// @param[in] alpha - first axis index
    /// @param[in] beta - second axis index
    /// @param[in] gamma - third axis index
    /// @return a mutable reference to the element
    double& ifc(std::size_t alpha, std::size_t beta, std::size_t gamma) {
        return this->ifcs[gamma + 3 * (beta + 3 * alpha)];
    }


    /// Access a particular ifc using three indexes.
    ///
    /// Note that all indices must be positive and lower than 3,
    /// but that this condition is not checked.
    /// @param[in] alpha - first axis index
    /// @param[in] beta - second axis index
    /// @param[in] gamma - third axis index
    /// @return a const reference to the element
    const double& ifc(std::size_t alpha,
                      std::size_t beta,
                      std::size_t gamma) const {
        return this->ifcs[gamma + 3 * (beta + 3 * alpha)];
    }


    /// Basic constructor. It does not initialize the ifcs
    /// member variable.
    Thirdorder_ifcs(const Eigen::VectorXd& _rj,
                    const Eigen::VectorXd& _rk,
                    std::size_t _i,
                    std::size_t _j,
                    std::size_t _k)
        : rj(std::move(_rj)), rk(std::move(_rk)), i(_i), j(_j), k(_k) {
    }


private:
    /// All third-order force constants between the three atoms.
    std::array<double, 27> ifcs;
};
} // namespace alma
