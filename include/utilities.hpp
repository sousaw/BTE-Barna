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
/// Miscellaneous convenience resources.

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <limits>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <random>
#include <boost/integer/common_factor.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <boost/mpi.hpp>
#include <constants.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>


namespace alma {


/// Randomly choose an element from a sequence using the provided rng.
template <typename I, typename R> I choose(I start, I end, R& g) {
    std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
    std::advance(start, dis(g));
    return start;
}

/// Stable Sum using Neumaier
/// algorithm. See DOI: 10.1002/zamm.19740540106
///(in German)

/// Stable sum using pointer and size
///@param  A pointer to data
///@param  size size of data
///@return stable summation result
template <class T> T NeumaierSum(T* A, std::size_t size) {
    T sum = 0;
    T c = 0;

    for (std::size_t i = 0; i < size; i++) {
        T t = sum + A[i];
        if (std::abs(sum) >= std::abs(A[i]))
            c += (sum - t) + A[i];
        else
            c += (A[i] - t) + sum;
        sum = t;
    }
    return sum + c;
}

/// Neumaier Sum in which the summation
/// cannot be done all together
/// so that we keep the correction
/// and the summation
///@param result Neumaier sum until now
///@param b      the number to add
template <class T>
std::complex<T> NeumaierSum(std::complex<T>& result, const T& b) {
    T TemporalSum = std::real(result) + b;
    if (std::abs(std::real(result)) >= std::abs(b)) {
        result += std::complex<T>(0, (std::real(result) - TemporalSum) + b);
    }
    else {
        result += std::complex<T>(0, (b - TemporalSum) + std::real(result));
    }
    result = std::complex<T>(TemporalSum, std::imag(result));
    return result;
}

/// This is to eval the Neumaier sum
/// it add correction to the summation
/// and return the value
///@param N summation and correction
///@return the Neumaier summation
template <class T> T evalNeumaierSum(std::complex<T>& N) {
    return std::real(N) + std::imag(N);
}

/// Thought for MPI collapse
/// after gather
///@param Ns vector containing Neumaier sumation info
///          from different procs
///@return the Neumaier summation
template <class T> T combineNeumaierSums(std::vector<std::complex<T>>& Ns) {
    /// Get sums and errors and apply
    /// stable sum to them
    std::vector<T> sums, cs;
    for (auto& N : Ns) {
        sums.push_back(std::real(N));
        cs.push_back(std::imag(N));
    }

    auto NeuSum = NeumaierSum(sums.data(), sums.size());

    auto NeuCs = NeumaierSum(cs.data(), cs.size());

    return NeuSum + NeuCs;
}

/// Comparator function object template for a container of
/// comparable objects.
template <typename T> class Container_comparator {
public:
    inline bool operator()(const T& op1, const T& op2) const {
        return std::lexicographical_compare(
            std::begin(op1), std::end(op1), std::begin(op2), std::end(op2));
    }
};

/// Put the keys and values of a map (or similar) into separate
/// vectors.
template <typename T>
std::tuple<std::vector<typename T::key_type>,
           std::vector<typename T::mapped_type>>
split_keys_and_values(const T& input) {
    auto size = input.size();

    std::vector<typename T::key_type> keys;
    keys.reserve(size);
    std::vector<typename T::mapped_type> values;
    values.reserve(size);

    for (auto& i : input) {
        keys.emplace_back(i.first);
        values.emplace_back(i.second);
    }
    return std::make_tuple(keys, values);
}


/// Read from f until '\n' is found.
///
/// This function is meant to be called after reading data from f
/// using >> and before using std::getline() on it.
///
/// @param[in] f - istream from which to read
inline void flush_istream(std::istream& f) {
    if (f.peek() == '\n') {
        f.ignore(1, '\n');
    }
}


/// Try to tokenize a line into homogeneous tokens.
///
/// This function tries to split a line into fields that can all be
/// cast into the same type and return a vector of tokens.
/// boost::bad_lexical_cast will be thrown in case of failure.
///
/// @param[in] line - line of text to tokenize
/// @param[in] sep - field separators
/// @return a vector of tokens
template <typename T>
inline std::vector<T> tokenize_homogeneous_line(std::string& line,
                                                const std::string sep = " \t") {
    std::vector<T> nruter;
    boost::char_separator<char> separator(sep.c_str());
    boost::tokenizer<boost::char_separator<char>> tokenizer(line, separator);

    for (auto i : tokenizer)
        nruter.emplace_back(boost::lexical_cast<T>(i));
    return nruter;
}


/// Check whether a line starts with a character from a given set.
///
/// @param[in] line - string containing the line
/// @param[in] chars - string containing the set of characters
/// @return true if line starts with a character from chars
inline bool starts_with_character(const std::string& line,
                                  const std::string& chars) {
    if (line.length() == 0)
        return false;

    for (auto c : chars) {
        if (line[0] == c)
            return true;
    }
    return false;
}


/// Approximate comparation between doubles.
///
/// @param[in] a - first value
/// @param[in] b - second value
/// @param[in] abstol - Maximum absolute tolerance
/// @param[in] reltol - Maximum relative tolerance
/// @return - true if the two values are almost equal
inline bool almost_equal(const double a,
                         const double b,
                         const double abstol = 1e-8,
                         const double reltol = 1e-5) {
    // Method and default values inspired by
    // numpy.allclose().
    const auto tol = std::fabs(abstol) + std::fabs(reltol * b);

    return std::fabs(a - b) < tol;
}


/// Convenience class for updating a minimum value and keeping track
/// of all the objects associated to it.
template <typename T> class Min_keeper {
public:
    /// Basic constructor.
    Min_keeper(const double _abstol = 1e-8, const double _reltol = 1e-5)
        : abstol(_abstol), reltol(_reltol) {
    }


    /// @return the current minimum value.
    double get_value() const {
        return this->minimum;
    }


    /// @return the vector of objects associated to the
    /// current minimum.
    std::vector<T> get_vector() const {
        return this->objects;
    }


    /// Update the minimum value with a new one if the latter
    /// is lower.
    ///
    /// If the new value is almost equal to the current minimum,
    /// append obj to the vector of objects.
    /// If the new value is lower than the current minimum, empty the
    /// vector of objects, append obj and update the minimum value.
    /// If the new value is higher than the current minimum, do
    /// nothing.
    /// @param[in] obj - new object
    /// @param[in] value - associated value
    void update(const T& obj, double value) {
        if (almost_equal(value, this->minimum, this->abstol, this->reltol)) {
            this->objects.emplace_back(obj);
        }
        else if (value < this->minimum) {
            this->minimum = value;
            this->objects.clear();
            this->objects.emplace_back(obj);
        }
    }


private:
    /// Current minimum value.
    double minimum = std::numeric_limits<double>::infinity();
    /// Vector of objects associated to the current minimum value.
    std::vector<T> objects;
    /// Maximum absolute tolerance for equality between doubles.
    const double abstol;
    /// Maximum relative tolerance for equality between doubles.
    const double reltol;
};

/// "Signed square root" of real numbers, that converts purely
/// imaginary numbers to negative values.
///
/// @param[in] x - argument of the square root
/// @return a real number with the absolute value of sqrt(|x|) and
/// the sign of x
inline double ssqrt(double x) {
    return std::copysign(std::sqrt(std::fabs(x)), x);
}


/// Compute the sign of an argument.
///
/// @param[in] arg - anything that can be constructed passing 0
/// as the only argument and that supports operator -
/// @return -1, 0 or 1 depending on the sign of arg
template <typename T> inline int signum(const T& arg) {
    T zero(0);

    return (zero < arg) - (arg < zero);
}


/// Build a timestamp in our preferred format.
///
/// @return a string containing the current time.
inline std::string get_timestamp() {
    // A much more elegant solution using put_time() exists.
    // We chose the traditional implementation because put time
    // is not implemented in GCC < 5.
    auto t = std::time(nullptr);
    auto localtime = std::localtime(&t);
    std::size_t size = 0;

    std::vector<char> buffer;

    do {
        size += 64;
        buffer.reserve(size);
        // 64 bytesshould be enough for everybody, but we
        // check the return code of std::strftime anyway to see
        // if we need to allocate more space.
    } while (std::strftime(
                 buffer.data(), size, "%Y-%m-%dT%H:%M:%S", localtime) == 0u);
    return std::string(buffer.data());
}


/// Compute the minimum and maximum indices of the jobs that must
/// be executed in the current process.
///
/// Split njobs jobs among nproc processes and return the
/// minimum and maximum indices of the jobs that must be assigned
/// to the current process, bundled in an array. More
/// specifically, return jmin and jmax so that we can write a
/// for loop like
/// for (auto j = jmin; j < jmax; ++j)
/// @param[in] njobs - number of jobs
/// @param[in] nprocs - number of process
/// @param[in] my_id - id of the current process
inline std::array<std::size_t, 2> my_jobs(std::size_t njobs,
                                          std::size_t nprocs,
                                          std::size_t my_id) {
    // If nprocs divides njobs, each process gets exactly its
    // fair share of jobs. If there is a remainder R , an extra
    // job is assigned to the first R processes. If njobs > nprocs,
    // some processes do not get any job.
    auto quot = njobs / nprocs;
    auto rem = njobs % nprocs;

    std::array<std::size_t, 2> nruter;
    nruter[0] = quot * my_id + std::min(rem, my_id);
    nruter[1] = quot * (my_id + 1) + std::min(rem, my_id + 1);
    return nruter;
}


/// Alternative modulus operation with a behavior similar to the %
/// operator in Python.
///
/// @param[in] n - numerator
/// @param[in] d - denominator
/// @return the remainder of the integer division n / d with
/// the same sign as d.
template <typename T, typename U> inline U python_mod(const T& n, const U& d) {
    if (d < 0)
        return python_mod(-n, -d);
    else {
        auto nruter = n % d;

        if (nruter < 0)
            nruter += d;
        return nruter;
    }
}


/// Bose-Einstein distribution.
///
/// @param[in] omega - angular frequency in rad/ps
/// @param[in] T - temperature in K
/// @return the average occupation number at equilibrium
inline double bose_einstein(double omega, double T) {
    constexpr double prefactor = 1e12 * constants::hbar / constants::kB;

    return 1. / (std::exp(prefactor * omega / T) - 1.);
}


/// Integration kernel for the specific heat, thermal conductivity
/// and other integrals.
///
/// This function implements the calculation of
/// (x / sinh(x))^2,
/// where x = hbar * omega / (2 * kB *T).
/// @param[in] omega - angular frequency in rad/ps
/// @param[in] T - temperature in K
/// @return the value of the aforementioned function
inline double bose_einstein_kernel(double omega, double T) {
    constexpr double prefactor = 0.5 * 1e12 * constants::hbar / constants::kB;
    auto x = prefactor * omega / T;

    if (x < 1e-320) {
        return 1.;
    }
    else if (x > 300.) {
        return 0.;
    }
    else {
        auto s = x / std::sinh(x);
        return s * s;
    }
}


/// Divide a vector of integers by the GCD of their coefficients.
///
/// @param[in] vector - the original vector
/// @return the reduced vector
inline Eigen::VectorXi reduce_integers(
    const Eigen::Ref<const Eigen::VectorXi>& vector) {
    std::size_t d = vector.size();
    Eigen::VectorXi::Scalar gcd;

    if (d == 0)
        gcd = 1;
    else {
        gcd = std::abs(vector(0));

        for (std::size_t i = 1; i < d; ++i)
            gcd = boost::integer::gcd(gcd, vector(i));
    }
    return Eigen::VectorXi(vector / gcd);
}


/// Convert a numerical value to a string that uses engineering prefix
inline std::string engineer_format(double x, bool insert_space = false) {
    std::vector<char> prefix(
        {'f', 'p', 'n', 'u', 'm', ' ', 'k', 'M', 'G', 'T'});
    std::vector<double> scaling(
        {1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9, 1e12, 1e15});

    int idx =
        static_cast<int>(std::floor((std::log10(x) + 15.0) / (3.0 - 1e-12)));

    if (idx < 0) {
        idx = 0;
    }

    if (idx > 9) {
        idx = 9;
    }
    std::stringstream result_builder;
    result_builder << x / scaling.at(idx);

    if (idx != 5) { // avoid introducing blank space
        if (insert_space) {
            result_builder << " ";
        }
        result_builder << prefix.at(idx);
    }
    return result_builder.str();
}


/// Construct a vector with N logarithmically spaced values from min to
/// max.
inline Eigen::VectorXd logSpace(double min, double max, int N) {
    Eigen::VectorXd logresult(N);

    logresult.setLinSpaced(N, std::log10(min), std::log10(max));
    return Eigen::pow(10.0, logresult.array());
}


} // namespace alma

namespace boost {
namespace serialization {
/// Allow boost to serialize an Eigen::Triplet.
template <class Archive, typename S>
void save(Archive& ar, const Eigen::Triplet<S>& t, const unsigned int version) {
    ar& t.row();
    ar& t.col();
    ar& t.value();
}


/// Allow boost to unserialize an Eigen::Triplet.
template <class Archive, typename S>
void load(Archive& ar, Eigen::Triplet<S>& t, const unsigned int version) {
    int row;
    ar& row;
    int col;
    ar& col;
    S value;
    ar& value;

    t = Eigen::Triplet<S>(row, col, value);
}


/// Dummy function allowing us to implement
/// save() and load() separately.
template <class Archive, typename S>
void serialize(Archive& ar, Eigen::Triplet<S>& t, const unsigned int version) {
    split_free(ar, t, version);
}
} // namespace serialization
} // namespace boost
