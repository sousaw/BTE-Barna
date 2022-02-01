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
///
/// Code used to build full BZ collision operators as well as the functions and
/// classes used to take and store the exponential of such operator in an
/// efficient way

#include <vector>
#include <tuple>
#include <utilities.hpp>
#include <bulk_hdf5.hpp>
#include <boost/mpi.hpp>
#include <isotopic_scattering.hpp>
#include <Eigen/Dense>
#include <unordered_map>
#if BOOST_VERSION >= 106700
#include <boost/container_hash/hash.hpp>
#else
#include <boost/functional/hash.hpp>
#endif
/// Specialization of std space used for collision_operator.cpp
namespace std {
/// Creating a std template specialization of less functional
template <> struct less<std::array<std::size_t, 3>> {
    bool operator()(const std::array<std::size_t, 3>& a,
                    const std::array<std::size_t, 3>& b) const {
        return std::tuple<std::size_t, std::size_t, std::size_t>(
                   {a[0], a[1], a[2]}) <
               std::tuple<std::size_t, std::size_t, std::size_t>(
                   {b[0], b[1], b[2]});
    }
};
/// Trivial implementation of std::hash for pairs,
/// required to create an unordered_map with pair as key.
template <typename T> struct hash<std::pair<T, T>> {
    std::size_t operator()(const std::pair<T, T>& key) const {
        hash<T> backend;
        std::size_t nruter = 0;
        boost::hash_combine(nruter, backend(key.first));
        boost::hash_combine(nruter, backend(key.second));
        return nruter;
    }
};
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


/// This boost specialization of serialization is required to
/// perform MPI operations of the matrix stored quantities
namespace boost {
namespace serialization {
/// This one is for three phonon elements
template <typename Archive, typename T1, typename T2, typename T3>
void serialize(Archive& ar,
               std::tuple<T1, T2, T3>& t,
               const unsigned int version) {
    ar& std::get<0>(t);
    ar& std::get<1>(t);
    ar& std::get<2>(t);
}


template <typename Archive>
void serialize(Archive& ar,
               std::pair<std::size_t, double>& t,
               const unsigned int version) {
    ar& std::get<0>(t);
    ar& std::get<1>(t);
}

/// Eigen Matrix serialization:
template <class Archive>
void serialize(Archive& ar,
               Eigen::Matrix<double, -1, 1>& t,
               const unsigned int version) {
    Eigen::MatrixXd::Index rows = t.rows();
    Eigen::MatrixXd::Index cols = t.cols();

    ar& rows;
    ar& cols;
    // Because our matrix is dynamic we need to ensure resizing
    if (rows * cols != t.size())
        t.resize(rows, cols);

    ar& boost::serialization::make_array(t.data(), rows * cols);
}

template <class Archive>
void serialize(Archive& ar,
               Eigen::Matrix<double, -1, -1>& t,
               const unsigned int version) {
    Eigen::MatrixXd::Index rows = t.rows();
    Eigen::MatrixXd::Index cols = t.cols();

    ar& rows;
    ar& cols;
    // Because our matrix is dynamic we need to ensure resizing
    if (rows * cols != t.size())
        t.resize(rows, cols);

    ar& boost::serialization::make_array(t.data(), rows * cols);
}


} // namespace serialization
} // namespace boost

namespace alma {

/// Chunck vectors and scatter them
///@param[in] v2c - vector to chunck and scatter
///@param[in] world - MPI communicator
template <class T>
void chunck_and_scatter(std::vector<T>& v2c, boost::mpi::communicator world) {
    broadcast(world, v2c, 0);
    auto ranges = parrange(0, v2c.size(), world);
    std::vector<T> vproc(0);
    vproc.insert(
        vproc.begin(), v2c.begin() + ranges.first, v2c.begin() + ranges.second);
    std::swap(v2c, vproc);
}

/// Gather data from procs to given rank
///@param[in] v2c - vector to gather from each proc, the gather data is then
/// inserted in the selected process
///@param[in] world - MPI communicator
///@param[in] where - the id-process in which the vector is gathered
template <class T>
void gather_from_procs(std::vector<T>& v2c,
                       boost::mpi::communicator world,
                       std::size_t where) {
    std::vector<std::vector<T>> chunckedv;

    gather(world, v2c, chunckedv, where);
    v2c.clear();
    if ((std::size_t)world.rank() == where) {
        for (auto& chunk : chunckedv)
            std::move(chunk.begin(), chunk.end(), std::back_inserter(v2c));
        v2c.shrink_to_fit();
    }
    else {
        v2c.shrink_to_fit();
    }
}


/// It gives the id-process for a given index
/// @param[in] index - index to look for
/// @param[in] container_size - the size of the container
/// @param[in] world_size - the number of processes in that MPI communicator
inline std::size_t index_proc(std::size_t index,
                              std::size_t container_size,
                              std::size_t world_size) {
    for (std::size_t ip = 0; ip < world_size; ip++) {
        auto limits = alma::my_jobs(container_size, world_size, ip);
        if (index >= limits[0] and index < limits[1])
            return ip;
    }
    return world_size;
}

/// Defining alias for long tuples used to store the full BZ elements to build
/// collision operator
using matel3part = std::tuple<std::size_t, std::size_t, double>;
using matel2part = std::pair<std::size_t, double>;
// template <class T> using std2dvec_<T> = std::vector<std::vector<T>>;


/// This is to build B matrix as defined in
/// [http://dx.doi.org/10.1063/1.4898090] with the inclusion of isotopic
/// scattering terms
///@param[in] grid       - qpoint grid
///@param[in] cell       - crystal structure information
///@param[in] processes  - three phonon processes in irreductible wedge
///@param[in] anhIFC     - Anharmonic force constants
///@param[in] Treference - reference temperature
///@param[in] world      - MPI communicator
Eigen::MatrixXd get_collision_operator_dense(
    const Gamma_grid& grid,
    const Crystal_structure& cell,
    std::vector<alma::Thirdorder_ifcs>& anhIFC,
    double Treference,
    boost::mpi::communicator& world);

/// This calculates the upper bound error for the m-th sized krylov
/// approximation of exponential with a normalized vector
///@param[in] A -  matrix from which the exponential is requested
///@param[in] n -  the krylov subspace size
///@param[in] t -  real scalar constant multiplying the matrix i.e.: exp(t*A)
inline long double error_bound(const Eigen::MatrixXd& A, int n, double t = 1.0) {
    long double At_norm = (t * A).norm();
    return 2.0 * std::pow(At_norm, n) * std::exp(At_norm) / std::tgamma(n);
}


/// Arnoldi algorithm used to compute the basis n-th krylov subspace generated
/// by Ab {b,A**2 * b,...,A**(n-1)* b}
///@param[in]     A - matrix A (see function purpose)
///@param[in]     b - vector b (see function purpose)
///@param[in]     n - krylov subspace order
///@param[in,out] B - Matrix to store the basis (each vector of the basis is
/// stored in different column)
///@param[in,out] H - A in the krylov subspace basis
///@param[in]     t - constant multiplying the A matrix so that the real matrix
/// is A' = t*A
void Arnoldi_algorithm(const Eigen::MatrixXd& A,
                       const Eigen::VectorXd& b,
                       int n,
                       Eigen::MatrixXd& B,
                       Eigen::MatrixXd& H,
                       double t = 1.0);


/// Storage of propagator as histogram
class propagator_H {
private:
    mutable Eigen::VectorXd signed_cumm; /// Signed cummulative function
    double
        pcol_abssum; /// The summ of abs value of original matrix - this is only
                     /// requested to be able to reconstruct P from histograms
    
    /// Some definitions to serialize the class
    friend class boost::serialization::access;
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version) {
        ar& signed_cumm;
        ar& pcol_abssum;
    }

    struct abs_less {
        bool operator()(const double& a, const double& b) const {
            return std::abs(a) < std::abs(b);
        }
    };

    /// Adaptation of lower bound to work with signed cumm
    template <class T> std::size_t lower_bound(T value) const {
        auto it = std::lower_bound(this->signed_cumm.data(),
                                   this->signed_cumm.data() + this->signed_cumm.rows(),
                                   value,
                                   abs_less());
        return std::distance(signed_cumm.data(),it);
    }

    friend alma::propagator_H lirp(
        alma::propagator_H& H1,
        alma::propagator_H& H2,
        double T1, 
        double T2, 
        double Tx);
    
public:
    /// Empty constructor
    propagator_H();

    /// Constructor
    ///@param[in]  data      - pointer to data to construct the histogram
    propagator_H(Eigen::VectorXd& b);
    ///@param[in] signed_cumm_ - the signed cummulative function
    ///@param[in] pcol_abssum_ - the abs sum of column values
    propagator_H(Eigen::VectorXd& signed_cumm_, double pcol_abssum_)
        : signed_cumm(signed_cumm_), pcol_abssum(pcol_abssum_){};


    /// Destructor is compiler default
    ~propagator_H() = default;
    /// Move constructor
    propagator_H(propagator_H&& B);

    /// Copy constructor
    propagator_H(const propagator_H& B);

    /// Assignement operator
    propagator_H& operator=(const propagator_H& B);

    /// comparison operator is dummy constructed
    bool operator<(const propagator_H& B) const;

    /// Given a random number it selects a mode
    ///@param[in] - R - number [0,1] not [0,1) as the last bit will not be
    /// selected with proper probability
    std::pair<int, std::size_t> get(double R) const;

    /// Returns the vector with the info
    Eigen::VectorXd get_signed_cumm() const;

    /// Returns the abs col summ
    double get_pcol_abssum() const {
        return this->pcol_abssum;
    }

    /// It return the size in bytes of all class
    std::size_t byte_size() const;

    /// It return the number of non-zero elements
    std::size_t size() const;
};

/// Function to interpolate two historgrams at different temperatures:
///@param[in] H1 - histogram at low  temperature
///@param[in] H2 - histogram at high temperature
///@param[in] T1 - low  T
///@param[in] T2 - high T
///@param[in] Tx - T at which the histogram must be interpolated
alma::propagator_H lirp(alma::propagator_H& H1,
                        alma::propagator_H& H2,
                        double T1,
                        double T2,
                        double Tx);

/// Function to interpolate two propagators at different temperatures:
///@param[in] P1 - propagator at low  temperature
///@param[in] P2 - propagator at high temperature
///@param[in] T1 - low  T
///@param[in] T2 - high T
///@param[in] Tx - T at which the propagator must be interpolated
inline Eigen::MatrixXd lirp(Eigen::MatrixXd& P1,
                            Eigen::MatrixXd& P2,
                            double T1,
                            double T2,
                            double Tx) {
    return P1 + (Tx - T1) * (P2 - P1) / (T2 - T1);
}

/// Function to dump P matrix to eigen binary from histogram:
///@param[in] fname  - File name without format extension
///@param[in] data   - reference to P to save
///@param[in] world  - MPI communicator
void save_P(std::string fname,
            Eigen::MatrixXd& data,
            boost::mpi::communicator& world);


/// Function to load P matrix from eigen binary:
///@param[in]     fname  - File name without format extension
///@param[in,out] pmat   - Pmatrix
///@param[in]     world  - MPI communicator
void load_P(std::string fname,
            Eigen::MatrixXd& pmat,
            boost::mpi::communicator& world);

/// Builds up a histogram of propagator matrix.
///@param[in]     A - Collision operator in energy form
///@param[in,out] modes_histo - histogram of propagator matrix
///@param[in]     t - time step
///@param[in]     world - MPI communicator
///@param[in]     eps   - tolerance to select the order of krylov subspace
///@param[in]     print_info - print_info
void build_histogram(Eigen::MatrixXd& A,
                     std::unordered_map<std::size_t, propagator_H>& modes_histo,
                     double t,
                     boost::mpi::communicator& world,
                     double eps = 1.0e-8,
                     bool print_info = false);


/// Builds up propagator matrix.
///@param[in]     A - Collision operator in energy form
///@param[in,out] modes_histo - histogram of propagator matrix
///@param[in]     t - time step
///@param[in]     world - MPI communicator
///@param[in]     eps   - tolerance to select the order of krylov subspace
void build_P(Eigen::MatrixXd& A,
             Eigen::MatrixXd& P,
             double t,
             boost::mpi::communicator& world,
             double eps = 1.0e-8);


} // namespace alma
