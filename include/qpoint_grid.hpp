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
/// Classes and functions used to manipulate grids in reciprocal space.

#include <boost/mpi.hpp>
#include <structures.hpp>
#include <dynamical_matrix.hpp>

namespace alma {
/// Convenient shorthand for an array of four indices.
using Tetrahedron = std::array<std::size_t, 4>;
/// Convenient shorthand for an array of three indices.
using Triangle = std::array<std::size_t, 3>;

/// Objects of this class represent a regular grid with the Gamma
/// point in one corner.
class Gamma_grid {
public:
    /// Size of the grid along the first reciprocal axis.
    const int na;
    /// Size of the grid along the second reciprocal axis.
    const int nb;
    /// Size of the grid along the third reciprocal axis.
    const int nc;
    /// Total number of q points in the grid.
    const std::size_t nqpoints;
    /// Reciprocal lattice basis vectors.
    const Eigen::MatrixXd rlattvec;
    /// Side vectors of each element in reciprocal space.
    const Eigen::MatrixXd dq;
    /// Constructor: initialize all internal variables and compute
    /// the spectrum at each q point. Do not correct the dynamical
    /// matrix for the effect of long-range interactions.
    Gamma_grid(const Crystal_structure& poscar,
               const Symmetry_operations& symms,
               const Harmonic_ifcs& force_constants,
               int _na,
               int _nb,
               int _nc);


    /// Constructor: initialize all internal variables and compute
    /// the spectrum at each q point. Correct the dynamical
    /// matrix for the effect of long-range interactions.
    Gamma_grid(const Crystal_structure& poscar,
               const Symmetry_operations& symms,
               const Harmonic_ifcs& force_constants,
               const Dielectric_parameters& born,
               int _na,
               int _nb,
               int _nc);


    /// Set the three lowest frequencies at Gamma to zero.
    void enforce_asr() {
        this->spectrum[0].omega.head(3).fill(0.);
    }


    /// Return the index of a q point identified by its position
    /// along the three axes.
    ///
    /// Indices are interpreted modulo-{na, nb, nc}, so even
    /// negative values are accepted.
    /// @param[in] indices - positions of the q point
    /// along the three axes
    /// @return the index of the q point in this grid
    std::size_t three_to_one(const std::array<int, 3>& indices) const {
        auto ia = python_mod(indices[0], this->na);
        auto ib = python_mod(indices[1], this->nb);
        auto ic = python_mod(indices[2], this->nc);

        return ic + nc * (ib + nb * ia);
    }


    /// Return the coordinates of a q point identified by its
    /// index.
    ///
    /// @param[in] index - index of the q point
    /// @return - an array with the positions of the q point
    /// along the three axes.
    std::array<int, 3> one_to_three(std::size_t iq) const {
        std::array<int, 3> nruter;
        nruter[2] = iq % nc;
        iq /= nc;
        nruter[1] = iq % nb;
        nruter[0] = iq / nb;
        return nruter;
    }


    /// @return the number of q-point equivalence classes in the grid.
    std::size_t get_nequivalences() const {
        return this->equivalences.size();
    }


    /// Access the harmonic properties at a point in the grid.
    ///
    /// The index is interpreted using modular arithmetic,
    /// so even negative values are accepted.
    /// @param[in] iq - index of the q point
    /// @return the harmonic properties at the q point.
    const Spectrum_at_point& get_spectrum_at_q(int iq) const {
        std::size_t index = python_mod(iq, this->nqpoints);

        return this->spectrum[index];
    }


    /// Return the cardinal of an equivalence class.
    ///
    /// @param[in] ic - index of the equivalence class.
    /// @return the number of points in an equivalence class.
    std::size_t get_cardinal(std::size_t ic) const {
        if (ic > this->equivalences.size())
            throw value_error("wrong equivalence class index");
        return this->equivalences[ic].size();
    }


    /// Return a representative of an equivalence class.
    ///
    /// The method is guaranteed to always return the same point.
    /// @param[in] ic - index of the equivalence class.
    /// @return a representative of the equivalence class.
    std::size_t get_representative(std::size_t ic) const {
        if (ic > this->equivalences.size())
            throw value_error("wrong equivalence class index");
        return this->equivalences[ic][0];
    }


    /// Return the elements in an equivalence class.
    ///
    /// @param[in] ic - index of the equivalence class.
    /// @return a vector with the elements in the equivalence
    /// class.
    std::vector<std::size_t> get_equivalence(std::size_t ic) const {
        if (ic > this->equivalences.size())
            throw value_error("wrong equivalence class index");
        return this->equivalences[ic];
    }


    /// Return the Cartesian coordinates of a q point.
    ///
    /// @param[in] iq - a q point index
    /// @return three Cartesian coordinates
    Eigen::VectorXd get_q(std::size_t iq) const {
        std::size_t index = python_mod(iq, this->nqpoints);

        return this->cpos.col(index);
    }


    /// Return the base broadening (without any prefactor) for a
    /// mode.
    ///
    /// @param[in] v - group velocity, or difference
    /// of group velocities in the case of three-phonon processes
    /// @return - the standard deviation of a Gaussian
    double base_sigma(const Eigen::VectorXd& v) const {
        return (v.transpose() * this->dq).norm() / std::sqrt(12.);
    }


    /// Very basic constructor that builds a stub object. Useful
    /// for deserialization or for obtaining an equivalences vector
    /// without computing the spectrum.
    Gamma_grid(const Crystal_structure& poscar,
               const Symmetry_operations& symms,
               int _na,
               int _nb,
               int _nc);


    /// Find the index of the polar opposite q point.
    ///
    /// @param[in] q - a q point
    /// @return the index of the polar opposite of q
    std::size_t polar_opposite(std::size_t q) {
        auto indices = this->one_to_three(q);

        return this->three_to_one({{-indices[0], -indices[1], -indices[2]}});
    }


    /// Return the images of a q point through all the symmetry
    /// operations, including inversions.
    ///
    /// @param[in] q - a q point
    /// @return a vector of q point indices. Elements with even indices
    /// correspond to operations in the space group,
    /// while elements with odd indices compound them with time
    /// reversal.
    std::vector<size_t> equivalent_qpoints(std::size_t original) const {
        std::size_t index = python_mod(original, this->nqpoints);
        return this->symmetry_map[index];
    }


    /// Find all q-point pairs equivalent to the input.
    ///
    /// Given a pair of indices, obtain all equivalent pairs
    /// after looking the up in the symmetry_map.
    /// @param[in] pair a pair of q point indices
    /// @return a vector of pairs, including the input
    std::vector<std::array<std::size_t, 2>> equivalent_qpairs(
        const std::array<std::size_t, 2>& original) const;


    /// Find all q-point triplets equivalent to the input.
    ///
    /// Given a triplet of indices, obtain all equivalent triplets
    /// after looking the up in the symmetry_map.
    /// @param[in] a triplet of q point indices
    /// @return a vector of triplets, including the input
    std::vector<std::array<std::size_t, 3>> equivalent_qtriplets(
        const std::array<std::size_t, 3>& original) const;


    /// Decompose the q-th microcell in five tetrahedra.
    ///
    /// @param[in] q - the index of the q point
    /// @return a vector of five Tetrahedron objects
    std::vector<Tetrahedron> get_tetrahedra(std::size_t q) const {
        // Find out the indices of the eight corners of the
        // microcell.
        std::size_t i000 = q;
        auto indices = this->one_to_three(i000);
        std::size_t i001 =
            this->three_to_one({{indices[0], indices[1], indices[2] + 1}});
        std::size_t i010 =
            this->three_to_one({{indices[0], indices[1] + 1, indices[2]}});
        std::size_t i011 =
            this->three_to_one({{indices[0], indices[1] + 1, indices[2] + 1}});
        std::size_t i100 =
            this->three_to_one({{indices[0] + 1, indices[1], indices[2]}});
        std::size_t i101 =
            this->three_to_one({{indices[0] + 1, indices[1], indices[2] + 1}});
        std::size_t i110 =
            this->three_to_one({{indices[0] + 1, indices[1] + 1, indices[2]}});
        std::size_t i111 = this->three_to_one(
            {{indices[0] + 1, indices[1] + 1, indices[2] + 1}});

        // Build the four equivalent tetrahedra.
        std::vector<Tetrahedron> nruter;
        nruter.reserve(5);
        nruter.emplace_back(Tetrahedron({{i000, i001, i010, i100}}));
        nruter.emplace_back(Tetrahedron({{i110, i111, i100, i010}}));
        nruter.emplace_back(Tetrahedron({{i101, i100, i111, i001}}));
        nruter.emplace_back(Tetrahedron({{i011, i010, i001, i111}}));
        // And the central, inequivalent one.
        nruter.emplace_back(Tetrahedron({{i010, i100, i111, i001}}));
        return nruter;
    }


    /// @return a vector of triangle objects with each containing three
    /// indices
    std::vector<Triangle> get_triangles(std::size_t ia) const;


    /// @return the index of the representative of the equivalence
    /// class
    /// of the provided q-point
    std::size_t getParentIdx(std::size_t iq) const;


    /// @return the index of the symmetry operation that maps the
    /// provided q-point
    /// to the representative of the equivalence class to which it
    /// belongs
    std::size_t getSymIdxToParent(std::size_t iq) const;


    /// @return the grid's symmetry map
    std::vector<std::vector<std::size_t>> getSymmetryMap() {
        return this->symmetry_map;
    }


    /// Remove the component of a Cartesian vector which does transform as a
    /// q-point in the grid.
    ///
    /// @param[in] iq - index of the q point
    /// @param[in] symms - set of symmetry operations of the crystal
    /// @param[in] x - vector in Cartesian coordinates. Several vectors can
    /// be provided by making v a matrix and each vector a column
    /// @return the symmetrized version of x
    template <typename T>
    auto copy_symmetry(std::size_t iq,
                       const Symmetry_operations& symms,
                       const Eigen::MatrixBase<T>& x) const
        -> Eigen::Matrix<typename T::Scalar, Eigen::Dynamic, Eigen::Dynamic> {
        std::size_t index = python_mod(iq, this->nqpoints);
        std::size_t nops = symms.get_nsym();

        typename T::PlainObject nruter(x);
        nruter.fill(0.);
        std::size_t nfound = 0;

        for (std::size_t iop = 0; iop < nops; ++iop) {
            if (this->symmetry_map[index][2 * iop] == index) {
                nruter += symms.rotate_v(x, iop, true);
                ++nfound;
            }
        }
        nruter /= nfound;
        return nruter;
    }

private:
    friend void save_bulk_hdf5(const char* filename,
                               const std::string& description,
                               const Crystal_structure& cell,
                               const Symmetry_operations& symmetries,
                               const Gamma_grid& grid,
                               const std::vector<Threeph_process>& processes,
                               const boost::mpi::communicator& comm);

    friend std::tuple<std::string,
                      std::unique_ptr<Crystal_structure>,
                      std::unique_ptr<Symmetry_operations>,
                      std::unique_ptr<Gamma_grid>,
                      std::unique_ptr<std::vector<Threeph_process>>>
    load_bulk_hdf5(const char* filename, const boost::mpi::communicator& comm);

    /// Cartesian coordinates of each q point.
    Eigen::MatrixXd cpos;
    /// Harmonic properties at each q point.
    std::vector<Spectrum_at_point> spectrum;
    /// Vector of equivalence classes. Two q points in the same
    /// equivalence class are related by a symmetry operation.
    /// Note that both space group symmetries and time reversal
    /// symmetry are taken into account.
    std::vector<std::vector<std::size_t>> equivalences;
    /// Detailed map between q points through the symmetry
    /// operations.
    std::vector<std::vector<std::size_t>> symmetry_map;
    /// Initialize the Cartesian coordinates of the q points.
    /// @param[in] poscar - a description of the crystal structure
    void initialize_cpos();

    /// Compute the spectrum at a subset of the q points.
    ///
    /// That subset is selected on the basis of this process'
    /// position in an MPI communicator.
    /// @param[in] factory - object in charge of building the
    /// dynamical matrix at each point.
    /// @param[in] symms - set of symmetry operations of the crystal
    /// @param[in] communicator - MPI communicator to use
    std::vector<Spectrum_at_point> compute_my_spectrum(
        const Dynamical_matrix_builder& factory,
        const Symmetry_operations& symms,
        const boost::mpi::communicator& communicator);

    /// Fill the equivalences vector.
    ///
    /// @param[in] symms - the set of symmetry operations to try.
    void fill_equivalences(const Symmetry_operations& symms);

    /// Parentlookup table: stores for each q-point the index of
    /// the representative of the equivalence class to which it
    /// belongs.
    std::vector<std::size_t> parentlookup;

    /// Fill the parentlookup table
    void fill_parentlookup();

    /// Fill the symmetry_map vector.
    ///
    /// @param[in] symms - the set of symmetry operations to try.
    void fill_map(const Symmetry_operations& symms);
};
} // namespace alma
