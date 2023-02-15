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
/// Definitions corresponding to bulk_properties.hpp

#include <bulk_properties.hpp>
#include <analytic1d.hpp>

namespace alma {
Eigen::MatrixXd calc_kappa(const alma::Crystal_structure& poscar,
                           const alma::Gamma_grid& grid,
			   const alma::Symmetry_operations& syms,
                           const Eigen::Ref<const Eigen::ArrayXXd>& w,
                           double T) {
    auto nequiv = grid.get_nequivalences();
    auto nmodes =
        static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size());

    if ((static_cast<std::size_t>(w.rows()) != nmodes) ||
        (static_cast<std::size_t>(w.cols()) != grid.nqpoints))
        throw alma::value_error("inconsistent dimensions");
    Eigen::MatrixXd nruter(3, 3);
    nruter.fill(0.);

    // The Gamma point is ignored.
    for (decltype(nequiv) iequiv = 1; iequiv < nequiv; ++iequiv) {
        auto iq0 = grid.get_representative(iequiv);
        auto sp0 = grid.get_spectrum_at_q(iq0);

        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            double tau = (w(im, iq0) == 0.) ? 0. : (1. / w(im, iq0));
            Eigen::MatrixXd outer(3, 3);
            outer.fill(0.);

            for (auto iq : grid.get_equivalence(iequiv)) {
                auto sp = grid.get_spectrum_at_q(iq);
                Eigen::VectorXd vg = sp.vg.col(im);
                outer += vg * vg.transpose();
            }
            nruter +=
                alma::bose_einstein_kernel(sp0.omega[im], T) * tau * outer;
        }
    }

    nruter = (1e21 * alma::constants::kB / poscar.V / grid.nqpoints) * nruter;

    /// Symmetrise kappa tensor 

    Eigen::Matrix3d nruter_accumulated;
    nruter_accumulated.fill(0.0);

    for (std::size_t nsymm = 0; nsymm < syms.get_nsym(); nsymm++) {
        nruter_accumulated += syms.rotate_m<double>(nruter, nsymm, true);
    }

    nruter = nruter_accumulated / static_cast<double>(syms.get_nsym());

    return nruter;
}

Eigen::ArrayXd calc_l2_RTAisotropic(const alma::Crystal_structure& poscar,
                                    const alma::Gamma_grid& grid,
                                    const Eigen::Ref<const Eigen::ArrayXXd>& w,
                                    double T) {
    auto Nbranches =
        static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size());

    if ((static_cast<std::size_t>(w.rows()) != Nbranches) ||
        (static_cast<std::size_t>(w.cols()) != grid.nqpoints))
        throw alma::value_error("inconsistent dimensions");

    Eigen::ArrayXd l2num(3), l2den(3);
    l2num.setZero();
    l2den.setZero();

    // The Gamma point is ignored.
    for (decltype(Nbranches) iq = 1; iq < grid.nqpoints; iq++) {
        auto sp = grid.get_spectrum_at_q(iq);

        auto Qimages = poscar.map_to_firstbz(grid.get_q(iq));

        Eigen::ArrayXd moment(3);
        moment(0) = Qimages.row(0).mean();
        moment(1) = Qimages.row(1).mean();
        moment(2) = Qimages.row(2).mean();

        moment *= 1.0e+9 * alma::constants::hbar;

        for (decltype(Nbranches) im = 0; im < Nbranches; im++) {
            double tau = (w(im, iq) == 0.) ? 0. : (1. / w(im, iq));
            tau *= 1.0e-12;

            Eigen::ArrayXd vg = sp.vg.col(im);
            vg *= 1.0e+3;

            double dfBE_T = alma::bose_einstein_kernel(sp.omega[im], T) *
                            alma::constants::kB /
                            (alma::constants::hbar * 1.0e+12 * sp.omega[im]);

            if (alma::almost_equal(sp.omega[im], 0.))
                continue;
            Eigen::ArrayXd g1 = -tau * tau * dfBE_T * vg * vg;

            // std::cout << sp.omega[im] << '\t' << g1(0) << '\t' << g1(1) <<
            // '\t' << alma::bose_einstein_kernel(sp.omega[im], T) <<
            //             '\t' << alma::constants::kB /(alma::constants::hbar
            //             * 1.0e+12 * sp.omega[im] ) <<
            //             '\t' << vg(0) << '\t' << vg(1) << '\t' << moment(0)
            //             << '\t' << moment(1) << std::endl;

            /// we are not dividing g1 by kappa as
            /// it is canceled out by kappa outside integral

            l2num += moment * vg * g1;
            l2den += moment * vg * dfBE_T;
        }
    }

    Eigen::ArrayXd l2 = (-1. / 5.) * l2num / l2den;


    /// Check if isotropy is valid (it will fail for 2d materials
    /// even if they are isotropic)
    bool isotropic = ((l2 - l2.matrix().mean()).abs() < 1.0e-6).all();


    if (!isotropic) {
        std::cout << "# WARNING: Material might be anisotropic\n  ";
        std::cout << " or 2d. l**2 calculation might be wrong as\n";
        std::cout << "  it assumes material to be isotropic\n";
    }


    return l2;
}


Eigen::MatrixXd calc_kappa_sg(const alma::Crystal_structure& poscar,
                              const alma::Gamma_grid& grid,
                              double T) {
    auto nequiv = grid.get_nequivalences();
    auto nmodes =
        static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size());
    Eigen::MatrixXd nruter(3, 3);

    nruter.fill(0.);

    // The Gamma point is ignored.
    for (decltype(nequiv) iequiv = 1; iequiv < nequiv; ++iequiv) {
        auto iq0 = grid.get_representative(iequiv);
        auto sp0 = grid.get_spectrum_at_q(iq0);

        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            double modv = sp0.vg.col(im).matrix().norm();

            if (!almost_equal(0., modv)) {
                Eigen::MatrixXd outer(3, 3);
                outer.fill(0.);

                for (auto iq : grid.get_equivalence(iequiv)) {
                    auto sp = grid.get_spectrum_at_q(iq);
                    Eigen::VectorXd vg = sp.vg.col(im);
                    outer += vg * vg.transpose();
                }
                nruter +=
                    alma::bose_einstein_kernel(sp0.omega[im], T) * outer / modv;
            }
        }
    }
    return (1e21 * alma::constants::kB / poscar.V / grid.nqpoints) * nruter;
}


double calc_kappa_1d(const alma::Crystal_structure& poscar,
                     const alma::Gamma_grid& grid,
                     const Eigen::Ref<const Eigen::ArrayXXd>& w,
                     double T,
                     const Eigen::Ref<const Eigen::Vector3d>& direction) {
    auto nqpoints = grid.nqpoints;
    auto nmodes =
        static_cast<std::size_t>(grid.get_spectrum_at_q(0).omega.size());

    if ((static_cast<std::size_t>(w.rows()) != nmodes) ||
        (static_cast<std::size_t>(w.cols()) != grid.nqpoints))
        throw alma::value_error("inconsistent dimensions");
    double unorm = direction.norm();

    if (unorm == 0.)
        throw alma::value_error("invalid direction");
    Eigen::Vector3d u = direction / unorm;
    double nruter = 0.;

    // The Gamma point is ignored.
    for (decltype(nqpoints) iq = 1; iq < nqpoints; ++iq) {
        auto spectrum = grid.get_spectrum_at_q(iq);

        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            double tau = (w(im, iq) == 0.) ? 0. : (1. / w(im, iq));
            double v = u.dot(spectrum.vg.matrix().col(im));
            nruter +=
                alma::bose_einstein_kernel(spectrum.omega[im], T) * tau * v * v;
        }
    }
    return (1e21 * alma::constants::kB / poscar.V / grid.nqpoints) * nruter;
}
} // namespace alma
