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
/// Physical, mathematical and miscellaneous constants used in alma.
/// Values taken from
/// http://physics.nist.gov/cuu/Constants/Table/allascii.txt .

#include <cmath>
#include <complex>

namespace alma {
namespace constants {
/// Atomic mass unit, kg.
constexpr double amu = 1.660538921e-27;
/// Atomic unit of charge, C.
constexpr double e = 1.602176565e-19;
/// Bohr radius, m.
constexpr double a0 = 0.52917721092e-10;
/// Avogadro constant, mol^{-1}.
constexpr double NA = 6.02214129e23;
/// Boltzmann constant, J / K.
constexpr double kB = 1.3806488e-23;
/// Electric constant, F / m.
constexpr double epsilon0 = 8.854187817e-12;
/// Electron mass, kg.
constexpr double me = 9.10938291e-31;
/// Magnetic constant, Wb.
constexpr double mu0 = 12.566370614e-7;
/// Planck constant, J s.
constexpr double h = 6.62606957e-34;
/// Dirac constant, J s.
constexpr double hbar = 1.054571726e-34;
/// Speed of light in vacuum, m/s.
constexpr double c = 299792458.;
/// Value of pi (to 128 bits)
constexpr double pi = 3.1415926535897932384626433832795028841953;
/// Lower-case English alphabet.
constexpr char alphabet[] = "abcdefghijklmnopqrstuwxyz";
/// A Gaussian PDF is considered to be zero at any point beyond
/// nsigma standard deviations from its mean.
constexpr double nsigma = 2.828427124746190097603377448419396157138;
/// Imaginary unit in double precision
constexpr std::complex<double> imud(0.0, 1.0);
} // namespace constants
} // namespace alma
