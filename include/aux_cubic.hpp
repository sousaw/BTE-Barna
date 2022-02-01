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
/// This file contains some auxiliary code used for computing 1D
/// Green's functions and related quantities using piecewise cubic
/// interpolation.

#include <cmath>
#include <boost/math/special_functions/pow.hpp>

namespace alma {
namespace aux_cubic {
/// Auxiliary function used when computing rational integrals.
inline double I1(double x0) {
    return std::log(std::abs((1. - x0) / x0));
}

/// Auxiliary function used when computing rational integrals.
inline double I2(double dd, double ee) {
    double den{std::sqrt(4. * ee - boost::math::pow<2>(dd))};

    return -2. * (std::atan(dd / den) - std::atan((dd + 2. * ee) / den)) / den;
}

/// Auxiliary function used when computing rational integrals.
inline double I3(double dd, double ee) {
    double den = std::sqrt(4. * ee - boost::math::pow<2>(dd));

    return (2. * dd * (std::atan(dd / den) - std::atan((dd + 2. * ee) / den)) +
            den * std::log1p(dd + ee)) /
           (2 * ee * den);
}

/// Objects of this class perform rational integrals of the kind
/// found when approximating bands by piecewise cubic polynomials
/// to obtain Green's functions in 1D.
class Cubic_segment {
private:
    /// Coefficients of the cubic polynomial in the denominator.
    const std::array<double, 4> coeff;

public:
    /// Constructor.
    ///
    /// @param[in] a - value of the energy at the left
    /// @param[in] b - value of the energy at the right
    /// @param[in] ap - derivative of the energy at the left
    /// @param[in] bp - derivative of the energy at the right
    Cubic_segment(double a, double b, double ap, double bp)
        : coeff({{-(2. * a + ap - 2. * b + bp),
                  -(-3. * a - 2. * ap + 3. * b - bp),
                  -ap,
                  -a}}) {
    }
    /// Obtain the values of the four integrals needed for the
    /// calculation of the Green's function at a particular
    /// energy.
    ///
    /// @param[in] energ - value of the energy
    /// @return a vector with the four integrals in this order:
    /// ReG, ReGx, ImG, ImGx
    std::array<double, 4> calc_integrals(double ener) const {
        // Coefficients of the full polynomial in the
        // denominator.
        std::array<double, 4> p(this->coeff);
        p[3] += ener;
        // Get the number of roots based on the value of the
        // discriminant for the third-order equation.
        double a{p[1] / p[0]};
        double b{p[2] / p[0]};
        double c{p[3] / p[0]};
        double a2{boost::math::pow<2>(a)};
        double q{(a2 - 3. * b) / 9.};
        double r{(a * (2. * a2 - 9. * b) + 27. * c) / 54.};
        double r2{boost::math::pow<2>(r)};
        double q3{boost::math::pow<3>(q)};
        std::array<double, 4> nruter({{0., 0., 0., 0.}});

        // And use the formulae specific to each case.
        if (r2 < q3) {
            // Three real roots.
            double th{std::acos(r / std::sqrt(q3))};
            double q12{std::sqrt(q)};
            double x0{-2. * q12 * std::cos(th / 3.) - a / 3.};
            double x1{-2. * q12 * std::cos((th + 2. * constants::pi) / 3.) -
                      a / 3.};
            double x2{-2. * q12 * std::cos((th - 2. * constants::pi) / 3.) -
                      a / 3.};
            double pref{-x0 * x1 * x2 / p[3]};
            double aa{pref / ((x0 - x1) * (x0 - x2))};
            double bb{pref / ((x1 - x0) * (x1 - x2))};
            double cc{pref / ((x2 - x0) * (x2 - x1))};
            nruter[0] = aa * I1(x0) + bb * I1(x1) + cc * I1(x2);
            double aax{x0 * aa};
            double bbx{x1 * bb};
            double ccx{x2 * cc};
            nruter[1] = aax * I1(x0) + bbx * I1(x1) + ccx * I1(x2);

            if ((0. < x0) && (x0 < 1.)) {
                nruter[2] += std::abs(aa);
                nruter[3] += std::abs(aax);
            }

            if ((0. < x1) && (x1 < 1.)) {
                nruter[2] += std::abs(bb);
                nruter[3] += std::abs(bbx);
            }

            if ((0. < x2) && (x2 < 1.)) {
                nruter[2] += std::abs(cc);
                nruter[3] += std::abs(ccx);
            }
            nruter[2] *= constants::pi;
            nruter[3] *= constants::pi;
        }
        else {
            // Only one real root.
            double A{-signum(r) * std::cbrt(std::abs(r) + std::sqrt(r2 - q3))};
            double B{(A == 0. ? 0. : q / A)};
            double x0{A + B - a / 3.};
            double dd{A + B + 2. * a / 3.};
            double ee{boost::math::pow<2>(dd / 2.) +
                      .75 * boost::math::pow<2>(A - B)};
            double pref{-x0 * ee / p[3]};
            double aa{pref / (x0 * (x0 + dd) + ee)};
            double bb{-(x0 + dd) * aa};
            double cc{-aa};
            double aax{x0 * aa};
            double bbx{ee * aa};
            double ccx{-aax};
            nruter[0] = aa * I1(x0) + (bb / ee) * I2(dd / ee, 1. / ee) +
                        (cc / ee) * I3(dd / ee, 1. / ee);
            nruter[1] = aax * I1(x0) + (bbx / ee) * I2(dd / ee, 1. / ee) +
                        (ccx / ee) * I3(dd / ee, 1. / ee);

            if ((0. < x0) && (x0 < 1.)) {
                nruter[2] = constants::pi * std::abs(aa);
                nruter[3] = constants::pi * std::abs(aax);
            }
        }
        return nruter;
    }
};
} // namespace aux_cubic
} // namespace alma
