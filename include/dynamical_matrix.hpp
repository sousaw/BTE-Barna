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
/// Code related to the dynamical matrix.
#include <boost/serialization/complex.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/unique_ptr.hpp>
#include <structures.hpp>
#include <symmetry.hpp>

namespace alma {
/// POD class that holds all the information about the harmonic
/// properties of the system at a particular q point.
/// The public members are not declared as const to make
/// serialization simpler, but as a general rule it is probably not
/// a good idea to change them.
class Spectrum_at_point {
private:
    friend class boost::serialization::access;
    /// Serialize the data needed to reconstruct an
    /// object of this class.
    ///
    /// @param[in,out] ar - an output archive
    /// @param[in] version - version number used
    /// internally by boost::serialization
    template <class Archive>
    void save(Archive& ar, const unsigned int version) const {
        auto rows = this->omega.size();

        ar << rows;
        ar << boost::serialization::make_array(this->omega.data(), rows);
        rows = this->wfs.rows();
        auto cols = this->wfs.cols();
        ar << rows;
        ar << cols;
        ar << boost::serialization::make_array(this->wfs.data(), rows * cols);
        rows = this->vg.rows();
        cols = this->vg.cols();
        ar << rows;
        ar << cols;
        ar << boost::serialization::make_array(this->vg.data(), rows * cols);
    }


    /// Unserialize the data needed to reconstruct an
    /// object of this class.
    ///
    /// @param[in,out] ar - an output archive
    /// @param[in] version - version number used
    /// internally by boost::serialization
    template <class Archive>
    void load(Archive& ar, const unsigned int version) {
        decltype(this->omega.size()) rows;
        ar >> rows;
        this->omega.resize(rows);
        ar >> boost::serialization::make_array(this->omega.data(), rows);
        decltype(rows) cols;
        ar >> rows;
        ar >> cols;
        this->wfs.resize(rows, cols);
        ar >> boost::serialization::make_array(this->wfs.data(), rows * cols);
        ar >> rows;
        ar >> cols;
        this->vg.resize(rows, cols);
        ar >> boost::serialization::make_array(this->vg.data(), rows * cols);
    }


    // Instruct boost::serialization to expect separate load()
    // and save() methods.
    BOOST_SERIALIZATION_SPLIT_MEMBER()
public:
    /// Angular frequencies, in rad / ps. Imaginary values are
    /// represented as negative real numbers.
    Eigen::ArrayXd omega;
    /// Eigenvectors, normalized on a single unit cell. Each
    /// column is an eigenvector.
    Eigen::MatrixXcd wfs;
    /// Cartesian components of the group velocities in km / s.
    Eigen::ArrayXXd vg;
    /// Basic constructor.
    Spectrum_at_point(const Eigen::Ref<const Eigen::ArrayXd>& _omega,
                      const Eigen::Ref<const Eigen::MatrixXcd>& _wfs,
                      const Eigen::Ref<const Eigen::ArrayXXd>& _vg)
        : omega(_omega), wfs(_wfs), vg(_vg) {
    }


    /// Empty default constructor.
    Spectrum_at_point() {
    }
};

/// Factory of Dynamical_matrix objects.
class Dynamical_matrix_builder {
public:
    /// Constructor for nonpolar systems.
    Dynamical_matrix_builder(const Crystal_structure& _structure,
                             const Symmetry_operations& syms,
                             const Harmonic_ifcs& fcs);
    /// Constructor for polar systems.
    Dynamical_matrix_builder(const Crystal_structure& _structure,
                             const Symmetry_operations& syms,
                             const Harmonic_ifcs& fcs,
                             const Dielectric_parameters& born);
    /// Return the dynamical matrix and its derivatives
    /// at one point.
    ///
    /// @param[in] q - the q point in Cartesian coordinates
    /// @return an array containing the dynamical matrix and the
    /// three components of its gradient
    std::array<Eigen::MatrixXcd, 4> build(
        const Eigen::Ref<const Eigen::Vector3d>& q) const;

    /// Call build() and use the return value to obtain the
    /// harmonic properties at one q point.
    ///
    /// @param[in] q - the q point in Cartesian coordinates
    /// @return a Spectrum_at_point object with all relevant
    /// information.
    std::unique_ptr<Spectrum_at_point> get_spectrum(
        const Eigen::Ref<const Eigen::Vector3d>& q) const;

    /// Obtain the number of building blocks of unfolded,
    /// mass-reduced interatomic force constants used to build the
    /// dynamical matrix.
    ///
    /// @return the number of such blocks
    std::size_t get_nblocks() const {
        return this->blocks.size();
    }
    /// Obtain a copy of one of the building blocks of the
    /// dynamical matrix.
    ///
    /// @param[in] i - block index
    /// @return a matrix containing a copy of the block
    Eigen::MatrixXd get_block(std::size_t i) const {
        if (i >= this->blocks.size())
            throw value_error("invalid index");
        return this->blocks[i];
    }


    /// Obtain the direct coordinates of the lattice point
    /// associated to a block.
    ///
    /// @param[in] i - block index
    /// @return three integers identifying a lattice point.
    Triple_int get_pos(std::size_t i) const {
        if (i >= this->blocks.size())
            throw value_error("invalid index");
        return this->pos[i];
    }


private:
    /// Building blocks of the dynamical matrix. Those are built
    /// from the IFC matrices and the masses, taking into account
    /// the degeneracy of each pair of atoms.
    std::vector<Eigen::MatrixXd> blocks;
    /// Matrices used to kee track of the degeneracy of atom pairs in order to
    /// apply the nonanalytic correction
    std::vector<Eigen::MatrixXd> masks;
    /// Lattice coordinates of each block.
    std::vector<Triple_int> pos;
    /// Dimension of the supercell originally used for the IFC
    /// calculations along the first axis.
    const int na;
    /// Dimension of the supercell originally used for the IFC
    /// calculations along the second axis.
    const int nb;
    /// Dimension of the supercell originally used for the IFC
    /// calculations along the third axis.
    const int nc;
    /// Volume of the unit cell.
    const double V;
    /// Description of the crystal.
    const Crystal_structure structure;
    /// Reciprocal-space lattice vectors of the structure.
    const Eigen::MatrixXd rlattvec;
    /// Object describing the space group symmetries.
    const Symmetry_operations symmetries;
    /// Lattice coordinates in an alternative format.
    Eigen::MatrixXd mpos;
    /// Cartesian coordinates of each block.
    Eigen::MatrixXd cpos;
    /// Square matrix with 3 * natoms rows where element ij is equal
    /// to the square root of the products of the masses of atom i/3
    /// and atom j/3.
    Eigen::ArrayXXd massmatrix;
    /// True if the Coulomb nonanalytic correction is taken into
    /// account.
    const bool nonanalytic;
    /// Dielectric parameters required for the nonanalytic
    /// correction.
    const Dielectric_parameters born;
    /// Populate the "blocks" member variable using the data
    /// provided to the constructor.
    ///
    /// @param[in] structure - a description of the unit cell
    /// @param[in] fcs - an object containing the IFCs
    /// for a supercell
    void copy_blocks(const Harmonic_ifcs& fcs);

    /// Compute the complex exponential weights needed to build the
    /// dynamical matrix out of the blocks vector.
    ///
    /// @param[in] q - the q point in Cartesian coordinates
    /// @return - a vector of weights
    Eigen::ArrayXcd get_exponentials(
        const Eigen::Ref<const Eigen::Vector3d>& q) const;

    /// Return the nonanalytic part of the dynamical matrix and
    /// its derivatives at one point.
    ///
    /// @param[in] q - the q point in Cartesian coordinates
    /// @return an array containing the long-range contribution to
    /// the force constants and the three components of
    /// its gradient.
    std::array<Eigen::ArrayXXd, 4> build_nac(
        const Eigen::Ref<const Eigen::Vector3d>& q) const;
};
} // namespace alma
