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
/// Code related to the phonon density fo states (DOS).

#include <qpoint_grid.hpp>
#include <boost/math/distributions/normal.hpp>

namespace alma {
/// Objects of this class handle the contribution of a mode to the
/// phonon DOS. We use an adaptive Gaussian smearing algorithm
/// to broaden the isolated modes.
class Gaussian_for_DOS {
public:
    /// Average energy.
    const double mu;
    /// Standard deviation.
    const double sigma;
    /// True if the usual lower bound would be negative.
    const bool truncated;
    /// Lower bound to the values that can be considered compatible
    /// with the average energy.
    const double lbound;
    /// Upper bound to the values that can be considered compatible
    /// with the average energy.
    const double ubound;
    /// Constructor.
    ///
    /// @param[in] grid - phonon spectrum on a regular grid
    /// @param[in] iq - q point index
    /// @param[in] im - branch index
    /// @param[in] scalebroad - prefactor for the standard
    /// deviation.
    Gaussian_for_DOS(const Gamma_grid& grid,
                     std::size_t iq,
                     std::size_t im,
                     double scalebroad)
        : mu(grid.get_spectrum_at_q(iq).omega(im)),
          sigma(scalebroad *
                grid.base_sigma(grid.get_spectrum_at_q(iq).vg.col(im))),
          truncated(mu - constants::nsigma * sigma < 0.),
          lbound(truncated ? 0. : mu - constants::nsigma * sigma),
          ubound(mu + constants::nsigma * sigma) {
        if ((this->mu == 0.) || (this->sigma == 0.))
            this->dist = nullptr;
        else
            this->dist = std::make_unique<boost::math::normal>(mu, sigma);
    }


    /// Get the amplitude of this contribution at a given frequency.
    ///
    /// @param[in] omega - an angular frequency in rad / ps
    /// @return the value of the Gaussian, normalized to 1 with
    /// respect to integration over omega.
    double get_contribution(double omega) const {
        if (this->dist)
            return boost::math::pdf(*(this->dist), omega);
        else
            return 0.;
    }


private:
    /// Underlying Gaussian distribution object.
    std::unique_ptr<boost::math::normal> dist;
};
} // namespace alma
