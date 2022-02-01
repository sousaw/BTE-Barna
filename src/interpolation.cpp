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
/// Definitions corresponding to interpolation.hpp.

#include <interpolation.hpp>
#include <algorithm>

namespace alma {

/// We are using the natural spline generation algorithm
/// see
/// https://en.wikipedia.org/wiki/Spline_(mathematics)#Algorithm_for_computing_natural_cubic_splines
/// accessed at 10/Dec/2020
/// The algorithm basically solves the tridiagonal problem
/// that the spline conditions build-up
CubicSpline1D::CubicSpline1D(const Eigen::Ref<const Eigen::VectorXd>& X,
                             const Eigen::Ref<const Eigen::VectorXd>& Y) {
    /// Given a set of coordinates C with size n+1:
    auto n = Y.rows() - 1;
    /// Create a new array a of size n+1 and set a_i = y_i
    Eigen::VectorXd a(Y);
    /// Create new arrays b and d of size n
    Eigen::VectorXd b(n), d(n);
    /// Create new array h of size n and set for
    /// i=0..n h(i) = x(i+1)-x(i)
    /// this contains the step size

    Eigen::VectorXd h(n);
    for (auto i = 0; i < n; i++)
        h(i) = X(i + 1) - X(i);

    /// Create new array alpha of size n and set for
    /// i=1..n alpha(i) = 3/h(i)*(a(i+1)-a(i)( - 3/h(i-1) * (a(i)-a(i-1))
    Eigen::VectorXd alpha(n);
    for (auto i = 1; i < n; i++)
        alpha(i) =
            3. / h(i) * (a(i + 1) - a(i)) - 3. / h(i - 1) * (a(i) - a(i - 1));

    /// Create new arrays c,l,mu and z of size n+1
    Eigen::VectorXd c(n + 1), l(n + 1), mu(n + 1), z(n + 1);
    /// Set l(0) = 1 and mu(0)=z(0)=0
    l(0) = 1;
    mu(0) = 0;
    z(0) = 0;
    /// Iterate to fill l,mu,z:
    for (auto i = 1; i < n; i++) {
        l(i) = 2 * (X(i + 1) - X(i - 1)) - h(i - 1) * mu(i - 1);
        mu(i) = h(i) / l(i);
        z(i) = (alpha(i) - h(i - 1) * z(i - 1)) / l(i);
    }
    l(n) = 1;
    z(n) = 0;
    c(n) = 0;

    for (auto j = n - 1; j >= 0; j--) {
        c(j) = z(j) - mu(j) * c(j + 1);
        b(j) = (a(j + 1) - a(j)) / h(j) - h(j) * (c(j + 1) + 2 * c(j)) / 3;
        d(j) = (c(j + 1) - c(j)) / (3 * h(j));
    }

    this->splines.resize(5, n);
    this->x.resize(n);
    for (auto i = 0; i < n; i++) {
        splines(0, i) = a(i);
        splines(1, i) = b(i);
        splines(2, i) = c(i);
        splines(3, i) = d(i);
        splines(4, i) = X(i);
        x[i] = X(i + 1);
    }
}

double CubicSpline1D::Interpolate(double newX) {
    /// Check which spline use
    auto idx =
        std::distance(this->x.begin(),
                      std::lower_bound(this->x.begin(), this->x.end(), newX));

    if (idx >= this->splines.cols()) {
        idx--;
    }

    Eigen::VectorXd sp3 = this->splines.col(idx);

    double dx = (newX - sp3(4));

    return sp3(0) + dx * (sp3(1) + dx * (sp3(2) + sp3(3) * dx));
}

} // namespace alma
