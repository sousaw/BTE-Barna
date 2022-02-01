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
/// Function interpolation stuff.

#include <unsupported/Eigen/Splines>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>


namespace alma {
/// Perform spline interpolation of a tabulated function Y(X) at
/// Xtarget.
inline Eigen::VectorXd splineInterpolation(
    const Eigen::Ref<const Eigen::VectorXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& Y,
    const Eigen::Ref<const Eigen::VectorXd>& Xtarget) {
    Eigen::VectorXd result(Xtarget.size());

    // map X values to 1D chord lengths ranging from 0 to 1
    Eigen::VectorXd X_norm =
        (X.array() - X.minCoeff()) / (X.maxCoeff() - X.minCoeff());

    // construct cubic spline over the base points
    typedef Eigen::Spline<double, 1> Spline1D;
    Spline1D S(Eigen::SplineFitting<Spline1D>::Interpolate(
        Y.transpose(), 3, X_norm.transpose()));

    // convert Xtarget values to their equivalent chord length
    Eigen::VectorXd Xtarget_norm =
        (Xtarget.array() - X.minCoeff()) / (X.maxCoeff() - X.minCoeff());

    // obtain spline value at each of these chord lengths

    for (int nx = 0; nx < Xtarget.size(); nx++) {
        result(nx) = S(Xtarget_norm(nx))(0);
    }

    return result;
}


/// Perform linear interpolation of a tabulated function Y(X) at
/// Xtarget.
/// X must be strictly ascending (no duplicate entries).
inline Eigen::VectorXd linearInterpolation(
    const Eigen::Ref<const Eigen::VectorXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& Y,
    const Eigen::Ref<const Eigen::VectorXd>& Xtarget) {
    Eigen::VectorXd result(Xtarget.size());
    int N = X.size();

    // perform linear interpolation for each of the specified targets

    for (int nx = 0; nx < Xtarget.size(); nx++) {
        if (Xtarget(nx) < X(0)) {
            std::cout << "alma::linearInterpolation > " << std::endl;
            std::cout << "WARNING: encountered target element out of bounds."
                      << std::endl;
            std::cout << "Interpolated result will be clipped." << std::endl;

            result(nx) = X(0);
        }

        else if (Xtarget(nx) > X(N - 1)) {
            std::cout << "alma::linearInterpolation > " << std::endl;
            std::cout << "WARNING: encountered target element out of bounds."
                      << std::endl;
            std::cout << "Interpolated result will be clipped." << std::endl;

            result(nx) = X(N - 1);
        }

        else {
            // use bisection algorithm to determine base array bin the target
            // sits
            // in.

            int leftindex = 0;
            int rightindex = N - 1;

            while (rightindex - leftindex > 1) {
                int midindex = (leftindex + rightindex) / 2;

                if (X(midindex) > Xtarget(nx)) {
                    rightindex = midindex;
                }
                else {
                    leftindex = midindex;
                }
            }

            double yleft = Y(leftindex);
            double yright = Y(rightindex);
            double xleft = X(leftindex);
            double xright = X(rightindex);

            result(nx) = yleft + (Xtarget(nx) - xleft) * (yright - yleft) /
                                     (xright - xleft);
        }
    }

    return result;
}





class CubicSpline1D {
private:
    Eigen::MatrixXd splines;
    std::vector<double> x;

public:
    /// Default constructors
    CubicSpline1D() = default;
    CubicSpline1D(const CubicSpline1D& A) = default;

    /// Constructor from data
    CubicSpline1D(const Eigen::Ref<const Eigen::VectorXd>& X,
                  const Eigen::Ref<const Eigen::VectorXd>& Y);

    /// Build interpolation from given data
    /// it discards the data
    void BuildCubicSpline1D(const Eigen::Ref<const Eigen::VectorXd>& X,
                            const Eigen::Ref<const Eigen::VectorXd>& Y);

    /// Interpolates Y for newX
    double Interpolate(double newX);
};


} // namespace alma
