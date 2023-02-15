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
/// Definitions corresponding to analytic1d.hpp

#include <complex>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <analytic1d.hpp>
#include <bulk_properties.hpp>
#include <io_utils.hpp>
#include <utilities.hpp>
#include <Eigen/Dense>
#include <Eigen/LU>

namespace alma {

namespace analytic1D {

/////////////////// BasicProperties_calculator ///////////////////

BasicProperties_calculator::BasicProperties_calculator(
    const alma::Crystal_structure* poscar_init,
    const alma::Gamma_grid* grid_init,
    const Eigen::ArrayXXd* w_init,
    double T_init)

    : poscar(poscar_init), grid(grid_init), w(w_init), T(T_init)

{
    auto nqpoints = this->grid->nqpoints;
    auto nmodes =
        static_cast<std::size_t>(this->grid->get_spectrum_at_q(0).omega.size());
    if (static_cast<std::size_t>(this->w->rows()) != nmodes ||
        static_cast<std::size_t>(this->w->cols()) != nqpoints)
        throw alma::value_error("BasicProperties_calculator > Dimensions of "
                                "scattering rate matrix are inconsistent with "
                                "wavevector grid.");

    this->unitvector = Eigen::Vector3d(1.0, 0.0, 0.0);
    this->thinfilm = false;
    this->kappacumulIdentifier = this->resolve_by_MFP;
    this->updateMe();
}

void BasicProperties_calculator::setDirection(const Eigen::Vector3d u) {
    this->unitvector = u;
    (this->unitvector).normalize();
    this->updateMe();
}

void BasicProperties_calculator::setLogMFPbins(double MFPmin,
                                               double MFPmax,
                                               int Nbins) {
    this->MFPbins = alma::logSpace(MFPmin, MFPmax, Nbins);
}

void BasicProperties_calculator::setAutoProjMFPbins(int Nbins) {
    this->setLogMFPbins(1e-6 * this->internal_MFPprojmax,
                        2.0 * this->internal_MFPprojmax,
                        Nbins);
}

void BasicProperties_calculator::setAutoMFPbins(int Nbins) {
    this->setLogMFPbins(
        1e-6 * this->internal_MFPmax, 2.0 * this->internal_MFPmax, Nbins);
}

void BasicProperties_calculator::setAutoRTbins(int Nbins) {
    this->setLogRTbins(
        1e-6 * this->internal_taumax, 2.0 * this->internal_taumax, Nbins);
}

void BasicProperties_calculator::setLogRTbins(double taumin,
                                              double taumax,
                                              int Nbins) {
    this->taubins = alma::logSpace(taumin, taumax, Nbins);
}

void BasicProperties_calculator::setLinOmegabins(double omegamin,
                                                 double omegamax,
                                                 int Nbins) {
    this->omegabins.setLinSpaced(Nbins, omegamin, omegamax);
}

void BasicProperties_calculator::setAutoOmegabins(int Nbins) {
    this->setLinOmegabins(0.0, 1.1 * this->internal_omegamax, Nbins);
}


Eigen::VectorXd BasicProperties_calculator::getMFPbins() {
    return this->MFPbins;
}

Eigen::VectorXd BasicProperties_calculator::getRTbins() {
    return this->taubins;
}

Eigen::VectorXd BasicProperties_calculator::getOmegabins() {
    return this->omegabins;
}


void BasicProperties_calculator::setBulk() {
    this->thinfilm = false;
    this->updateMe();
}

void BasicProperties_calculator::setCrossPlaneFilm(double d) {
    this->thinfilm = true;
    this->crossplane = true;
    this->filmthickness = d;
    this->updateMe();
}

void BasicProperties_calculator::setInPlaneFilm(double d,
                                                const Eigen::Vector3d n,
                                                double specul) {
    this->thinfilm = true;
    this->crossplane = false;
    this->filmthickness = d;
    this->normalvector = n;
    this->normalvector.normalize();
    this->specularity = specul;
    this->updateMe();
}

void BasicProperties_calculator::initFuchsRotation() {
    // obtain spherical coordinate angles that describe the film normal

    double theta = std::acos(this->normalvector(2));
    double phi = 0.0;

    if (std::abs(this->normalvector(2)) < 1.0) {
        phi = std::atan2(this->normalvector(1), this->normalvector(0));
    }

    // Construct the 3D rotation matrix that maps the film normal to (0,0,1)

    Eigen::Matrix3d R;
    R << std::cos(phi) * std::cos(theta), std::sin(phi) * std::cos(theta),
        -std::sin(theta), -std::sin(phi), std::cos(phi), 0.0,
        std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi),
        std::cos(theta);

    this->FuchsRotation = R;
}

void BasicProperties_calculator::updateMe() {
    if (this->thinfilm && !this->crossplane) {
        this->initFuchsRotation();
    }

    auto nqpoints = this->grid->nqpoints;
    auto nmodes =
        static_cast<std::size_t>(this->grid->get_spectrum_at_q(0).omega.size());

    this->kappa = 0.0;
    this->Cv = 0.0;
    this->dominantProjMFP = 0.0;
    this->dominantRT = 0.0;

    const double prefactor =
        1e27 * alma::constants::kB / this->grid->nqpoints / this->poscar->V;

    // obtain heat capacity at gamma point
    auto sp0 = this->grid->get_spectrum_at_q(0);

    for (decltype(nmodes) im = 0; im < nmodes; ++im) {
        // obtain volumetric heat capacity of these modes [J/m^3-K]
        double C = prefactor * alma::bose_einstein_kernel(sp0.omega[im], T);

        this->Cv += C;
    }

    // obtain heat capacity and conductivity of all other modes

    double MFPmax = -1.0;
    double MFPprojmax = -1.0;
    double taumax = -1.0;
    double omegamax = -1.0;

    for (decltype(nqpoints) iq = 1; iq < nqpoints; ++iq) {
        auto sp = grid->get_spectrum_at_q(iq);
        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            // scattering rate
            double w0 = w->operator()(im, iq);
            // obtain relaxation time [seconds]
            double tau0 = (w0 == 0.) ? 0. : (1e-12 / w0);

            // obtain phonon frequency
            double omega0 = 1e12 * sp.omega(im);
            if (omega0 > omegamax) {
                omegamax = omega0;
            }

            // obtain volumetric heat capacity [J/m^3-K]
            double C = prefactor * alma::bose_einstein_kernel(sp.omega[im], T);
            this->Cv += C;

            // obtain projected group velocity [m/s] and MFP [m]
            double vg = 1e3 * sp.vg.col(im).matrix().norm();
            Eigen::Vector3d vg_vector = sp.vg.col(im);
            double vg_proj = 1e3 * this->unitvector.dot(sp.vg.col(im).matrix());

            double MFP = vg * tau0;
            double MFP_proj = std::abs(vg_proj * tau0);

            // perform corrections for thin films if needed

            double suppression_factor = 1.0;

            if (this->thinfilm) {
                if (this->crossplane) {
                    // Obtain cross-plane suppression factor as derived in
                    // B. Vermeersch, J. Carrete, N. Mingo
                    // Applied Physics Letters 108, 193104 (2016)
                    // http://dx.doi.org/10.1063/1.4948968

                    suppression_factor =
                        1.0 / (1.0 + 2.0 * MFP_proj / this->filmthickness);

                } // end crossplane film

                else {
                    // Obtain in-plane suppression by applying Fuchs-Sondheimer
                    // correction
                    // As we process each mode within the discrete wavevector
                    // grid individually,
                    // F-S formula before integration over solid angle must be
                    // applied.

                    // (1) Obtain the group velocity within the transformed
                    // coordinate
                    //     system that describes the thin film
                    Eigen::Vector3d vg_rotated =
                        this->FuchsRotation * vg_vector;

                    // (2) Calculate generalised Knudsen number
                    double K_prime =
                        MFP * std::abs(vg_rotated(2) / vg_rotated.norm()) /
                        this->filmthickness;

                    // (3) Evaluate suppression factor
                    // This step is bypassed when
                    //  (a) K_prime = 0 (limit of suppression factor is 1)
                    //  (b) vg_proj = 0 (mode does not contribute to
                    //  conductivity anyway)

                    if (std::abs(vg_proj) > 0.0 && K_prime > 0.0) {
                        double buffer1 =
                            1.0 - this->specularity * std::exp(-1.0 / K_prime) -
                            (1.0 - this->specularity) * K_prime *
                                (1.0 - std::exp(-1.0 / K_prime));
                        double buffer2 =
                            1.0 - this->specularity * exp(-1.0 / K_prime);

                        suppression_factor = buffer1 / buffer2;
                    }

                } // end in-plane film

            } // end film corrections

            // correct phonon properties
            MFP *= suppression_factor;
            MFP_proj *= suppression_factor;
            tau0 *= suppression_factor;

            // keep track of maximum values
            if (MFP > MFPmax) {
                MFPmax = MFP;
            }
            if (MFP_proj > MFPprojmax) {
                MFPprojmax = MFP_proj;
            }
            if (tau0 > taumax) {
                taumax = tau0;
            };

            // register contributions of this mode
            double kappa_mode = vg_proj * vg_proj * tau0 * C;
            this->kappa += kappa_mode;
            this->dominantProjMFP += MFP_proj * kappa_mode;
            this->dominantRT += tau0 * kappa_mode;
        }
    }

    this->dominantProjMFP = this->dominantProjMFP / this->kappa;
    this->dominantRT = this->dominantRT / this->kappa;

    this->internal_MFPmax = MFPmax;
    this->internal_MFPprojmax = MFPprojmax;
    this->internal_omegamax = omegamax;
    this->internal_taumax = taumax;
}

double BasicProperties_calculator::getSpectralConductivity() {
    if (this->thinfilm && !this->crossplane) {
        if (std::abs(this->unitvector.dot(this->normalvector)) > 1e-12) {
            std::cout
                << "BasicProperties_calculator::getSpectralConductivity() > "
                << std::endl;
            std::cout << "WARNING: the transport axis and film normal that "
                         "were provided for in-plane film are not orthogonal."
                      << std::endl;
            std::cout << "Computation results likely invalid." << std::endl;
        }
    }

    // number of frequency bins to be used in calculations
    int Nbins = 100;

    int nqpoints = this->grid->nqpoints;
    int Nbranches = this->grid->get_spectrum_at_q(0).omega.size();

    // obtain largest phonon energy for each phonon branch
    Eigen::VectorXd omegamax(Nbranches);
    omegamax.setConstant(-1.0);
    double omega_mode;

    for (int nq = 1; nq < nqpoints; nq++) {
        auto sp = grid->get_spectrum_at_q(nq);

        for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
            omega_mode = sp.omega[nbranch];
            if (omega_mode > omegamax(nbranch)) {
                omegamax(nbranch) = omega_mode;
            }
        }
    }

    // set up frequency bins
    Eigen::MatrixXd omega(Nbins + 1, Nbranches);
    Eigen::VectorXd omegabranch;
    for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
        omegabranch.setLinSpaced(Nbins + 1, 0.0, omegamax(nbranch));
        omega.col(nbranch) = omegabranch;
    }

    Eigen::MatrixXd Cbin(Nbins, Nbranches); // total capacity in each bin
    Cbin.fill(0.0);
    Eigen::MatrixXd Dbin(Nbins, Nbranches); // average diffusivity in each bin
    Dbin.fill(0.0);
    Eigen::MatrixXi bincount(Nbins, Nbranches); // number of modes in each bin
    bincount.fill(0);

    int binindex;
    double tau, MFP, vg;
    const double prefactor =
        1e27 * alma::constants::kB / this->grid->nqpoints / this->poscar->V;

    // perform frequency binning

    for (int nq = 1; nq < nqpoints; nq++) {
        auto sp = grid->get_spectrum_at_q(nq);

        for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
            // scattering rate
            double w0 = w->operator()(nbranch, nq);
            // relaxation time [seconds]
            tau = (w0 == 0.) ? 0. : (1e-12 / w0);

            // calculate binindex
            omega_mode = sp.omega[nbranch];
            binindex = static_cast<int>(std::floor(
                omega_mode * static_cast<double>(Nbins) / omegamax(nbranch)));
            if (binindex == Nbins) { // occurs for maximum frequency
                binindex = Nbins - 1;
            }
            if (binindex < 0) {
                binindex = 0;
            }

            if (binindex >= 0 &&
                tau > 0.0) { // ignore anomalous negative frequency points

                Cbin(binindex, nbranch) +=
                    prefactor * alma::bose_einstein_kernel(omega_mode, T);
                bincount(binindex, nbranch) += 1;

                // obtain group velocity [m/s] and MFP [m]
                vg = 1e3 * sp.vg.col(nbranch).matrix().norm();
                MFP = vg * tau;
                Dbin(binindex, nbranch) += MFP * MFP / (3.0 * tau);
            }
        }
    }

    // calculate average diffusivity in each bin

    for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
        for (int nbin = 0; nbin < Nbins; nbin++) {
            if (bincount(nbin, nbranch) > 0) {
                Dbin(nbin, nbranch) /=
                    static_cast<double>(bincount(nbin, nbranch));
            }
        }
    }

    // evaluate conductivity

    double result = 0.0;

    for (int nbranch = 0; nbranch < Nbranches; nbranch++) {
        for (int nbin = 0; nbin < Nbins; nbin++) {
            if (bincount(nbin, nbranch) > 0) {
                result += Cbin(nbin, nbranch) * Dbin(nbin, nbranch);
            }
        }
    }

    return result;
}

void BasicProperties_calculator::setDOSgridsize(int nbins) {
    this->DOS_Nbins = nbins;
}

Eigen::VectorXd BasicProperties_calculator::getDOSgrid() {
    Eigen::VectorXd result;
    result.setLinSpaced(this->DOS_Nbins, 0.0, this->internal_omegamax);

    // conversion from phonon frequency to energy in meV
    return (1e3 * alma::constants::hbar / alma::constants::e) * result;
}

Eigen::VectorXd BasicProperties_calculator::getDOS() {
    // number of bins to be used in calculations
    int Nbins = this->DOS_Nbins;

    // set up frequency bins
    Eigen::VectorXi bincount(Nbins);
    bincount.setConstant(0);

    // Perform frequency binning
    auto nqpoints = this->grid->nqpoints;
    auto nmodes =
        static_cast<std::size_t>(this->grid->get_spectrum_at_q(0).omega.size());

    for (decltype(nqpoints) iq = 1; iq < nqpoints; ++iq) {
        auto sp = grid->get_spectrum_at_q(iq);
        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            // calculate binindex
            double omega_mode = 1e12 * sp.omega[im];

            if (omega_mode > 0.0) {
                int binindex = static_cast<int>(
                    std::floor(omega_mode * static_cast<double>(Nbins) /
                               this->internal_omegamax));
                if (binindex >= Nbins) {
                    binindex = Nbins - 1;
                };
                bincount(binindex)++;
            }
        }
    }

    // convert bincounts to DOS versus energy in meV

    double deltaomega = this->internal_omegamax / static_cast<double>(Nbins);
    int Nmodes = bincount.sum();
    double deltaE =
        1e3 * deltaomega * alma::constants::hbar / alma::constants::e;

    return bincount.cast<double>() / (deltaE * static_cast<double>(Nmodes));
}

double BasicProperties_calculator::getAnisotropyIndex(const alma::Symmetry_operations* syms) {
    Eigen::Matrix3d kappa_tensor =
        alma::calc_kappa(*(this->poscar), *(this->grid), *(syms) , *(this->w), this->T);

    Eigen::Vector3d kappa_diag = kappa_tensor.diagonal();

    return kappa_diag.maxCoeff() / kappa_diag.minCoeff();
}

double BasicProperties_calculator::getConductivity() {
    if (this->thinfilm && !this->crossplane) {
        if (std::abs(this->unitvector.dot(this->normalvector)) > 1e-12) {
            std::cout << "BasicProperties_calculator::getConductivity() > "
                      << std::endl;
            std::cout << "WARNING: the transport axis and film normal that "
                         "were provided for in-plane film are not orthogonal."
                      << std::endl;
            std::cout << "Computation results likely invalid." << std::endl;
        }
    }

    return this->kappa;
}

double BasicProperties_calculator::getCapacity() {
    return this->Cv;
}

double BasicProperties_calculator::getDiffusivity() {
    if (this->thinfilm && !this->crossplane) {
        if (std::abs(this->unitvector.dot(this->normalvector)) > 1e-12) {
            std::cout << "BasicProperties_calculator::getDiffusivity() > "
                      << std::endl;
            std::cout << "WARNING: the transport axis and film normal that "
                         "were provided for in-plane film are not orthogonal."
                      << std::endl;
            std::cout << "Computation results likely invalid." << std::endl;
        }
    }

    return (this->kappa / this->Cv);
}

double BasicProperties_calculator::getDominantProjMFP() {
    if (this->thinfilm && !this->crossplane) {
        if (std::abs(this->unitvector.dot(this->normalvector)) > 1e-12) {
            std::cout << "BasicProperties_calculator::getDominantProjMFP() > "
                      << std::endl;
            std::cout << "WARNING: the transport axis and film normal that "
                         "were provided for in-plane film are not orthogonal."
                      << std::endl;
            std::cout << "Computation results likely invalid." << std::endl;
        }
    }

    return this->dominantProjMFP;
}

double BasicProperties_calculator::getDominantRT() {
    if (this->thinfilm && !this->crossplane) {
        if (std::abs(this->unitvector.dot(this->normalvector)) > 1e-12) {
            std::cout << "BasicProperties_calculator::getDominantRT() > "
                      << std::endl;
            std::cout << "WARNING: the transport axis and film normal that "
                         "were provided for in-plane film are not orthogonal."
                      << std::endl;
            std::cout << "Computation results likely invalid." << std::endl;
        }
    }

    return this->dominantRT;
}

void BasicProperties_calculator::resolveByMFP() {
    this->kappacumulIdentifier = this->resolve_by_MFP;
}

void BasicProperties_calculator::resolveByProjMFP() {
    this->kappacumulIdentifier = this->resolve_by_ProjMFP;
}

void BasicProperties_calculator::resolveByRT() {
    this->kappacumulIdentifier = this->resolve_by_RT;
}

void BasicProperties_calculator::resolveByOmega() {
    this->kappacumulIdentifier = this->resolve_by_omega;
}

Eigen::VectorXd BasicProperties_calculator::getCumulativeConductivity() {
    if (this->thinfilm && !this->crossplane) {
        if (std::abs(this->unitvector.dot(this->normalvector)) > 1e-12) {
            std::cout
                << "BasicProperties_calculator::getCumulativeConductivity() > "
                << std::endl;
            std::cout << "WARNING: the transport axis and film normal that "
                         "were provided for in-plane film are not orthogonal."
                      << std::endl;
            std::cout << "Computation results likely invalid." << std::endl;
        }
    }

    int Nbins;

    if (this->kappacumulIdentifier == this->resolve_by_MFP ||
        this->kappacumulIdentifier == this->resolve_by_ProjMFP) {
        Nbins = this->MFPbins.size();
    }

    if (this->kappacumulIdentifier == this->resolve_by_RT) {
        Nbins = this->taubins.size();
    }

    if (this->kappacumulIdentifier == this->resolve_by_omega) {
        Nbins = this->omegabins.size();
    }

    this->kappacumul.resize(Nbins);
    this->kappacumul.setConstant(0.0);

    Eigen::VectorXd kappabins(Nbins);
    kappabins.setConstant(0.0);

    auto nqpoints = this->grid->nqpoints;
    auto nmodes =
        static_cast<std::size_t>(this->grid->get_spectrum_at_q(0).omega.size());

    for (decltype(nqpoints) iq = 1; iq < nqpoints; ++iq) {
        auto sp0 = grid->get_spectrum_at_q(iq);
        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            // angular frequency of this mode
            double omega0 = 1e12 * sp0.omega(im);

            // scattering rate
            double w0 = w->operator()(im, iq);
            // obtain relaxation time of these modes [seconds]
            double tau0 = (w0 == 0.) ? 0. : (1e-12 / w0);

            // obtain volumetric heat capacity of these modes [J/m^3-K]
            double prefactor = 1e27 * alma::constants::kB /
                               this->grid->nqpoints / this->poscar->V;
            double C = prefactor * alma::bose_einstein_kernel(sp0.omega[im], T);

            // obtain projected group velocity [m/s] and MFP [m]
            Eigen::Vector3d vg_vector = 1e3 * sp0.vg.col(im);
            double vg = vg_vector.norm();
            double vg_proj = this->unitvector.dot(vg_vector);

            double MFP = vg * tau0;
            double MFP_proj = std::abs(vg_proj * tau0);

            // perform corrections for thin films if needed

            double suppression_factor = 1.0;

            if (this->thinfilm) {
                if (this->crossplane) {
                    // Obtain cross-plane suppression factor as derived in
                    // B. Vermeersch, J. Carrete, N. Mingo
                    // Applied Physics Letters 108, 193104 (2016)
                    // http://dx.doi.org/10.1063/1.4948968

                    suppression_factor =
                        1.0 / (1.0 + 2.0 * MFP_proj / this->filmthickness);

                } // end crossplane film

                else {
                    // Obtain in-plane suppression by applying Fuchs-Sondheimer
                    // correction
                    // As we process each mode within the discrete wavevector
                    // grid individually,
                    // F-S formula before integration over solid angle must be
                    // applied.

                    // (1) Obtain the group velocity within the transformed
                    // coordinate
                    //     system that describes the thin film
                    Eigen::Vector3d vg_rotated =
                        this->FuchsRotation * vg_vector;

                    // (2) Calculate generalised Knudsen number
                    double K_prime =
                        MFP * std::abs(vg_rotated(2) / vg_rotated.norm()) /
                        this->filmthickness;

                    // (3) Evaluate suppression factor
                    // This step is bypassed when
                    //  (a) K_prime = 0 (limit of suppression factor is 1)
                    //  (b) vg_proj = 0 (mode does not contribute to
                    //  conductivity anyway)

                    if (std::abs(vg_proj) > 0.0 && K_prime > 0.0) {
                        double buffer1 =
                            1.0 - this->specularity * std::exp(-1.0 / K_prime) -
                            (1.0 - this->specularity) * K_prime *
                                (1.0 - std::exp(-1.0 / K_prime));
                        double buffer2 =
                            1.0 - this->specularity * exp(-1.0 / K_prime);

                        suppression_factor = buffer1 / buffer2;
                    }

                } // end in-plane film

            } // end film corrections

            // conductivity of these modes
            double contribution =
                suppression_factor * vg_proj * vg_proj * tau0 * C;

            // determine which bin these modes belong to

            double minvalue = 0.0;
            double maxvalue = 0.0;
            double targetvalue = 0.0;

            if (this->kappacumulIdentifier == this->resolve_by_MFP) {
                minvalue = std::log10(this->MFPbins(0));
                maxvalue = std::log10(this->MFPbins(Nbins - 1));
                targetvalue = std::log10(MFP);
            }
            else if (this->kappacumulIdentifier == this->resolve_by_ProjMFP) {
                minvalue = std::log10(this->MFPbins(0));
                maxvalue = std::log10(this->MFPbins(Nbins - 1));
                targetvalue = std::log10(MFP_proj);
            }
            else if (this->kappacumulIdentifier == this->resolve_by_RT) {
                minvalue = std::log10(this->taubins(0));
                maxvalue = std::log10(this->taubins(Nbins - 1));
                targetvalue = std::log10(tau0);
            }
            else if (this->kappacumulIdentifier == this->resolve_by_omega) {
                minvalue = 0.0;
                maxvalue = this->omegabins(Nbins - 1);
                targetvalue = omega0;
            }

            int binindex = static_cast<int>(
                std::floor((targetvalue - minvalue) *
                           static_cast<double>(Nbins) / (maxvalue - minvalue)));
            if (binindex < 0) {
                binindex = 0;
            }
            if (binindex >= Nbins) {
                binindex = Nbins - 1;
            }

            kappabins(binindex) += contribution;
        }
    }

    this->kappacumul(0) = kappabins(0);

    for (int nbin = 1; nbin < Nbins; nbin++) {
        this->kappacumul(nbin) = this->kappacumul(nbin - 1) + kappabins(nbin);
    }

    return this->kappacumul;
}


Eigen::VectorXd BasicProperties_calculator::getCumulativel2RTAiso() {
    if (this->thinfilm && !this->crossplane) {
        if (std::abs(this->unitvector.dot(this->normalvector)) > 1e-12) {
            std::cout
                << "BasicProperties_calculator::getCumulativel2RTAiso() > "
                << std::endl;
            std::cout << "WARNING: the transport axis and film normal that "
                         "were provided for in-plane film are not orthogonal."
                      << std::endl;
            std::cout << "Computation results likely invalid." << std::endl;
        }
    }

    int Nbins;

    if (this->kappacumulIdentifier == this->resolve_by_MFP ||
        this->kappacumulIdentifier == this->resolve_by_ProjMFP) {
        Nbins = this->MFPbins.size();
    }

    if (this->kappacumulIdentifier == this->resolve_by_RT) {
        Nbins = this->taubins.size();
    }

    if (this->kappacumulIdentifier == this->resolve_by_omega) {
        Nbins = this->omegabins.size();
    }

    this->l2cumul.resize(Nbins);
    this->l2cumul.setConstant(0.0);

    Eigen::VectorXd l2binsNum(Nbins);
    Eigen::VectorXd l2binsDen(Nbins);
    l2binsNum.setConstant(0.0);
    l2binsDen.setConstant(0.0);


    auto nqpoints = this->grid->nqpoints;
    auto nmodes =
        static_cast<std::size_t>(this->grid->get_spectrum_at_q(0).omega.size());

    for (decltype(nqpoints) iq = 1; iq < nqpoints; ++iq) {
        auto sp0 = grid->get_spectrum_at_q(iq);

        /// Get momentum

        double moment_proj;
        auto Qcarts = poscar->map_to_firstbz(grid->get_q(iq));
        Eigen::Vector3d momentum;

        momentum(0) = 1.0e+9 * alma::constants::hbar * Qcarts.row(0).mean();
        momentum(1) = 1.0e+9 * alma::constants::hbar * Qcarts.row(1).mean();
        momentum(2) = 1.0e+9 * alma::constants::hbar * Qcarts.row(2).mean();

        moment_proj = this->unitvector.dot(momentum);


        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            // angular frequency of this mode
            double omega0 = 1e12 * sp0.omega(im);

            // scattering rate
            double w0 = w->operator()(im, iq);
            // obtain relaxation time of these modes [seconds]
            double tau0 = (w0 == 0.) ? 0. : (1e-12 / w0);

            // obtain volumetric heat capacity of these modes [J/m^3-K]
            if (alma::almost_equal(sp0.omega[im], 0.))
                continue;

            double dfBE_T = alma::constants::kB *
                            alma::bose_einstein_kernel(sp0.omega[im], T) /
                            (alma::constants::hbar * 1.0e+12 * sp0.omega[im]);


            // obtain projected group velocity [m/s] and MFP [m]
            Eigen::Vector3d vg_vector = 1e3 * sp0.vg.col(im);
            double vg = vg_vector.norm();
            double vg_proj = this->unitvector.dot(vg_vector);

            double MFP = vg * tau0;
            double MFP_proj = std::abs(vg_proj * tau0);


            // perform corrections for thin films if needed

            double suppression_factor = 1.0;

            if (this->thinfilm) {
                /// TODO: Implement anisotropy
                std::cout << "#WARNING: current implementation is ";
                std::cout << "isotropic so it is not valid for nanosystems";
                std::cout << "the MFP resolved l2 for bulk is recommended";
                exit(1);

                if (this->crossplane) {
                    // Obtain cross-plane suppression factor as derived in
                    // B. Vermeersch, J. Carrete, N. Mingo
                    // Applied Physics Letters 108, 193104 (2016)
                    // http://dx.doi.org/10.1063/1.4948968

                    suppression_factor =
                        1.0 / (1.0 + 2.0 * MFP_proj / this->filmthickness);

                } // end crossplane film

                else {
                    // Obtain in-plane suppression by applying Fuchs-Sondheimer
                    // correction
                    // As we process each mode within the discrete wavevector
                    // grid individually,
                    // F-S formula before integration over solid angle must be
                    // applied.

                    // (1) Obtain the group velocity within the transformed
                    // coordinate
                    //     system that describes the thin film
                    Eigen::Vector3d vg_rotated =
                        this->FuchsRotation * vg_vector;

                    // (2) Calculate generalised Knudsen number
                    double K_prime =
                        MFP * std::abs(vg_rotated(2) / vg_rotated.norm()) /
                        this->filmthickness;

                    // (3) Evaluate suppression factor
                    // This step is bypassed when
                    //  (a) K_prime = 0 (limit of suppression factor is 1)
                    //  (b) vg_proj = 0 (mode does not contribute to
                    //  conductivity anyway)

                    if (std::abs(vg_proj) > 0.0 && K_prime > 0.0) {
                        double buffer1 =
                            1.0 - this->specularity * std::exp(-1.0 / K_prime) -
                            (1.0 - this->specularity) * K_prime *
                                (1.0 - std::exp(-1.0 / K_prime));
                        double buffer2 =
                            1.0 - this->specularity * exp(-1.0 / K_prime);

                        suppression_factor = buffer1 / buffer2;
                    }

                } // end in-plane film

            } // end film corrections

            double tauS = tau0 * suppression_factor;

            // conductivity of these modes
            double contributionNum =
                moment_proj * vg_proj *
                (-tauS * tauS * dfBE_T * vg_proj * vg_proj);
            double contributionDen = -5.0 * moment_proj * vg_proj * dfBE_T;

            // determine which bin these modes belong to

            double minvalue = 0.0;
            double maxvalue = 0.0;
            double targetvalue = 0.0;

            if (this->kappacumulIdentifier == this->resolve_by_MFP) {
                minvalue = std::log10(this->MFPbins(0));
                maxvalue = std::log10(this->MFPbins(Nbins - 1));
                targetvalue = std::log10(MFP);
            }
            else if (this->kappacumulIdentifier == this->resolve_by_ProjMFP) {
                minvalue = std::log10(this->MFPbins(0));
                maxvalue = std::log10(this->MFPbins(Nbins - 1));
                targetvalue = std::log10(MFP_proj);
            }
            else if (this->kappacumulIdentifier == this->resolve_by_RT) {
                minvalue = std::log10(this->taubins(0));
                maxvalue = std::log10(this->taubins(Nbins - 1));
                targetvalue = std::log10(tau0);
            }
            else if (this->kappacumulIdentifier == this->resolve_by_omega) {
                minvalue = 0.0;
                maxvalue = this->omegabins(Nbins - 1);
                targetvalue = omega0;
            }

            int binindex = static_cast<int>(
                std::floor((targetvalue - minvalue) *
                           static_cast<double>(Nbins) / (maxvalue - minvalue)));
            if (binindex < 0) {
                binindex = 0;
            }
            if (binindex >= Nbins) {
                binindex = Nbins - 1;
            }

            l2binsNum(binindex) += contributionNum;
            l2binsDen(binindex) += contributionDen;
        }
    }

    Eigen::VectorXd l2cumulNum(Nbins), l2cumulDen(Nbins);
    l2cumulDen.fill(0);
    l2cumulNum.fill(0);

    l2cumulNum(0) = l2binsNum(0);
    l2cumulDen(0) = l2binsDen(0);

    for (int nbin = 1; nbin < Nbins; nbin++) {
        l2cumulNum(nbin) = l2cumulNum(nbin - 1) + l2binsNum(nbin);
        l2cumulDen(nbin) = l2cumulDen(nbin - 1) + l2binsDen(nbin);
    }

    this->l2cumul = l2cumulNum.array() / l2cumulDen.array();

    return this->l2cumul;
}


Eigen::VectorXd BasicProperties_calculator::getCumulativeCapacity() {
    int Nbins = -1;

    if (this->kappacumulIdentifier == this->resolve_by_MFP ||
        this->kappacumulIdentifier == this->resolve_by_ProjMFP) {
        Nbins = this->MFPbins.size();
    }

    if (this->kappacumulIdentifier == this->resolve_by_RT) {
        Nbins = this->taubins.size();
    }

    if (this->kappacumulIdentifier == this->resolve_by_omega) {
        Nbins = this->omegabins.size();
    }

    this->Cvcumul.resize(Nbins);
    this->Cvcumul.setConstant(0.0);

    Eigen::VectorXd Cvbins(Nbins);
    Cvbins.setConstant(0.0);

    auto nqpoints = this->grid->nqpoints;
    auto nmodes =
        static_cast<std::size_t>(this->grid->get_spectrum_at_q(0).omega.size());

    for (decltype(nqpoints) iq = 0; iq < nqpoints; ++iq) {
        auto sp0 = grid->get_spectrum_at_q(iq);
        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            // angular frequency
            double omega0 = 1e12 * sp0.omega(im);

            // scattering rate
            double w0 = w->operator()(im, iq);
            // obtain relaxation time of these modes [seconds]
            double tau = (w0 == 0.) ? 0. : (1e-12 / w0);

            // obtain volumetric heat capacity of these modes [J/m^3-K]
            double prefactor = 1e27 * alma::constants::kB /
                               this->grid->nqpoints / this->poscar->V;
            double C = prefactor * alma::bose_einstein_kernel(sp0.omega[im], T);

            // obtain projected group velocity [m/s] and MFP [m]
            double vg = 1e3 * sp0.vg.col(im).matrix().norm();
            double vg_proj =
                1e3 * this->unitvector.dot(sp0.vg.col(im).matrix());

            double MFP = vg * tau;
            double MFP_proj = std::abs(vg_proj * tau);

            // heat capcity of these modes
            double contribution = C;

            // determine which bin these modes belong to

            double minvalue = 0.0;
            double maxvalue = 0.0;
            double targetvalue = 0.0;

            if (this->kappacumulIdentifier == this->resolve_by_MFP) {
                minvalue = std::log10(this->MFPbins(0));
                maxvalue = std::log10(this->MFPbins(Nbins - 1));
                targetvalue = std::log10(MFP);
            }
            else if (this->kappacumulIdentifier == this->resolve_by_ProjMFP) {
                minvalue = std::log10(this->MFPbins(0));
                maxvalue = std::log10(this->MFPbins(Nbins - 1));
                targetvalue = std::log10(MFP_proj);
            }
            else if (this->kappacumulIdentifier == this->resolve_by_RT) {
                minvalue = std::log10(this->taubins(0));
                maxvalue = std::log10(this->taubins(Nbins - 1));
                targetvalue = std::log10(tau);
            }
            else if (this->kappacumulIdentifier == this->resolve_by_omega) {
                minvalue = 0.0;
                maxvalue = this->omegabins(Nbins - 1);
                targetvalue = omega0;
            }

            int binindex = static_cast<int>(
                std::floor((targetvalue - minvalue) *
                           static_cast<double>(Nbins) / (maxvalue - minvalue)));
            if (binindex < 0) {
                binindex = 0;
            }
            if (binindex >= Nbins) {
                binindex = Nbins - 1;
            }

            Cvbins(binindex) += contribution;
        }
    }

    this->Cvcumul(0) = Cvbins(0);

    for (int nbin = 1; nbin < Nbins; nbin++) {
        this->Cvcumul(nbin) = this->Cvcumul(nbin - 1) + Cvbins(nbin);
    }

    return this->Cvcumul;
}

/////////////////// psi_calculator ///////////////////

psi_calculator::psi_calculator(const alma::Crystal_structure* poscar_init,
                               const alma::Gamma_grid* grid_init,
                               const Eigen::ArrayXXd* w_init,
                               double T_init)

    : poscar(poscar_init), grid(grid_init), w(w_init), T(T_init)

{
    auto nqpoints = this->grid->nqpoints;
    auto nmodes =
        static_cast<std::size_t>(this->grid->get_spectrum_at_q(0).omega.size());
    if (static_cast<std::size_t>(this->w->rows()) != nmodes ||
        static_cast<std::size_t>(this->w->cols()) != nqpoints)
        throw alma::value_error("psi_calculator > Dimensions of scattering "
                                "rate matrix are inconsistent with wavevector "
                                "grid.");

    this->unitvector = Eigen::Vector3d(1.0, 0.0, 0.0);
    this->scaleoutput = false;
    this->updateDiffusivity();
}

void psi_calculator::setDirection(const Eigen::Vector3d u) {
    this->unitvector = u;
    (this->unitvector).normalize();
    this->updateDiffusivity();
}

void psi_calculator::setLinGrid(double ximin, double ximax, int Nxi) {
    this->xi.setLinSpaced(Nxi, ximin, ximax);
}

void psi_calculator::setLogGrid(double ximin, double ximax, int Nxi) {
    this->xi = alma::logSpace(ximin, ximax, Nxi);
}

void psi_calculator::setXiGrid(const Eigen::Ref<const Eigen::VectorXd> xigrid) {
    this->xi = xigrid;
}

Eigen::VectorXd psi_calculator::getSpatialFrequencies() {
    return this->xi;
}

void psi_calculator::normaliseOutput(bool norm) {
    this->scaleoutput = norm;
}

void psi_calculator::updateDiffusivity() {
    alma::analytic1D::BasicProperties_calculator propCalc(
        this->poscar, this->grid, this->w, this->T);
    propCalc.setDirection(this->unitvector);

    this->Dbulk = propCalc.getDiffusivity();
}

double psi_calculator::getDiffusivity() {
    return this->Dbulk;
}

Eigen::VectorXd psi_calculator::getPsi() {
    auto nqpoints = this->grid->nqpoints;
    auto nmodes =
        static_cast<std::size_t>(this->grid->get_spectrum_at_q(0).omega.size());

    int Nxi = this->xi.size();
    Eigen::VectorXd ones(Nxi);
    ones.setConstant(1.0);

    Eigen::VectorXd numerator(Nxi);
    Eigen::VectorXd denominator(Nxi);
    numerator.setConstant(0.0);
    denominator.setConstant(0.0);

    // The Gamma point is ignored.
    for (decltype(nqpoints) iq = 1; iq < nqpoints; ++iq) {
        auto sp = this->grid->get_spectrum_at_q(iq);
        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            // scattering rate
            double w0 = w->operator()(im, iq);
            // relaxation time [seconds]
            double tau0 = (w0 == 0.) ? 0. : (1e-12 / w0);

            // obtain volumetric heat capacity [J/m^3-K]
            double prefactor = 1e27 * alma::constants::kB /
                               this->grid->nqpoints / this->poscar->V;
            double C = prefactor * alma::bose_einstein_kernel(sp.omega[im], T);

            // obtain projected group velocity [m/s] and MFP [m]
            double vg_proj = 1e3 * this->unitvector.dot(sp.vg.col(im).matrix());
            double MFP_proj = std::abs(vg_proj * tau0);

            numerator.array() += C * vg_proj * vg_proj * tau0 * ones.array() /
                                 (ones.array() + MFP_proj * MFP_proj *
                                                     this->xi.array().square());

            denominator.array() +=
                C * ones.array() /
                (ones.array() +
                 MFP_proj * MFP_proj * this->xi.array().square());
        }
    }

    this->psi = this->xi.array() * this->xi.array() *
                (numerator.array() / denominator.array());

    if (this->scaleoutput) {
        this->psi =
            this->psi.array() / (this->Dbulk * this->xi.array().square());
    }

    return this->psi;
}

/////////////////// SPR_calculator_FourierLaplace ///////////////////

SPR_calculator_FourierLaplace::SPR_calculator_FourierLaplace(
    const alma::Crystal_structure* poscar_init,
    const alma::Gamma_grid* grid_init,
    const Eigen::ArrayXXd* w_init,
    double T_init)

    : poscar(poscar_init), grid(grid_init), w(w_init), T(T_init)

{
    this->unitvector = Eigen::Vector3d(1.0, 0.0, 0.0);
}

void SPR_calculator_FourierLaplace::setDirection(const Eigen::Vector3d u) {
    this->unitvector = u;
    (this->unitvector).normalize();
}

void SPR_calculator_FourierLaplace::setLinSpatialGrid(double ximin,
                                                      double ximax,
                                                      int Nxi) {
    this->xi.setLinSpaced(Nxi, ximin, ximax);
}

void SPR_calculator_FourierLaplace::setLogSpatialGrid(double ximin,
                                                      double ximax,
                                                      int Nxi) {
    this->xi = alma::logSpace(ximin, ximax, Nxi);
}

void SPR_calculator_FourierLaplace::setLinTemporalGrid(double fmin,
                                                       double fmax,
                                                       int Nf) {
    this->f.setLinSpaced(Nf, fmin, fmax);
    this->s = std::complex<double>(0.0, 2.0 * alma::constants::pi) *
              f.cast<std::complex<double>>();
}

void SPR_calculator_FourierLaplace::setLogTemporalGrid(double fmin,
                                                       double fmax,
                                                       int Nf) {
    this->f = alma::logSpace(fmin, fmax, Nf);
    this->s = std::complex<double>(0.0, 2.0 * alma::constants::pi) *
              f.cast<std::complex<double>>();
}

Eigen::VectorXd SPR_calculator_FourierLaplace::getSpatialFrequencies() {
    return this->xi;
}

Eigen::VectorXd SPR_calculator_FourierLaplace::getTemporalFrequencies() {
    return this->f;
}

Eigen::MatrixXcd SPR_calculator_FourierLaplace::getSPR() {
    int Nxi = this->xi.size();
    int Ns = this->s.size();

    this->Pxis.resize(Nxi, Ns);

    auto nqpoints = this->grid->nqpoints;
    auto nmodes =
        static_cast<std::size_t>(this->grid->get_spectrum_at_q(0).omega.size());

    Eigen::MatrixXcd numerator(Nxi, Ns);
    Eigen::MatrixXcd denominator(Nxi, Ns);
    numerator.fill(std::complex<double>(0.0, 0.0));
    denominator.fill(std::complex<double>(0.0, 0.0));

    // The Gamma point is ignored.
    for (decltype(nqpoints) iq = 1; iq < nqpoints; ++iq) {
        auto sp = this->grid->get_spectrum_at_q(iq);
        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            // obtain relaxation time [seconds]
            double w0 = this->w->operator()(im, iq);
            double tau0 = (w0 == 0.) ? 0. : (1e-12 / w0);

            // obtain volumetric heat capacity [J/m^3-K]
            double prefactor = 1e27 * alma::constants::kB /
                               this->grid->nqpoints / this->poscar->V;
            double C = prefactor * alma::bose_einstein_kernel(sp.omega[im], T);

            // obtain projected group velocity [m/s] and MFP [m]
            double vg_proj = 1e3 * this->unitvector.dot(sp.vg.col(im).matrix());
            double MFP_proj = std::abs(vg_proj * tau0);

            for (int nxi = 0; nxi < Nxi; nxi++) {
                double xival = this->xi(nxi);

                for (int ns = 0; ns < Ns; ns++) {
                    std::complex<double> sval = this->s(ns);
                    std::complex<double> buffer =
                        (std::complex<double>(1.0, 0.0) + sval * tau0) /
                        ((std::complex<double>(1.0, 0.0) + sval * tau0) *
                             (std::complex<double>(1.0, 0.0) + sval * tau0) +
                         xival * xival * MFP_proj * MFP_proj);
                    numerator(nxi, ns) += C * buffer;
                    if (tau0 > 0.0)
                        denominator(nxi, ns) +=
                            (C / tau0) *
                            (std::complex<double>(1.0, 0.0) - buffer);
                }
            }
        }
    }

    this->Pxis = numerator.array() / denominator.array();

    return this->Pxis;
}


/////////////////// SPR_calculator_RealSpace ///////////////////

SPR_calculator_RealSpace::SPR_calculator_RealSpace(
    const alma::Crystal_structure* poscar_init,
    const alma::Gamma_grid* grid_init,
    const Eigen::ArrayXXd* w_init,
    double T_init)

    : poscar(poscar_init), grid(grid_init), w(w_init), T(T_init)

{
    this->unitvector = Eigen::Vector3d(1.0, 0.0, 0.0);
    this->normaliseoutput = false;
    this->gridIsNormalised = false;

    this->t = 10e-9;

    this->updateDiffusivity();
}

void SPR_calculator_RealSpace::setDirection(const Eigen::Vector3d u) {
    this->unitvector = u;
    (this->unitvector).normalize();

    this->updateDiffusivity();
}

void SPR_calculator_RealSpace::setLinGrid(double xmin, double xmax, int Nx) {
    this->x.setLinSpaced(Nx, xmin, xmax);
}

void SPR_calculator_RealSpace::setLogGrid(double xmin, double xmax, int Nx) {
    this->x = alma::logSpace(xmin, xmax, Nx);
}

void SPR_calculator_RealSpace::declareGridNormalised(bool norm) {
    this->gridIsNormalised = norm;
}

void SPR_calculator_RealSpace::normaliseOutput(bool norm) {
    this->normaliseoutput = norm;
}

void SPR_calculator_RealSpace::setTime(double t) {
    this->t = t;
}

void SPR_calculator_RealSpace::setLogMFPbins(double MFPmin,
                                             double MFPmax,
                                             int Nbins) {
    this->MFPbins = alma::logSpace(MFPmin, MFPmax, Nbins);
}

Eigen::VectorXd SPR_calculator_RealSpace::getGrid() {
    if (this->gridIsNormalised) {
        return std::sqrt(2.0 * this->Dbulk * this->t) * this->x;
    }
    else {
        return this->x;
    }
}

Eigen::VectorXd SPR_calculator_RealSpace::getNormalisedGrid() {
    if (this->gridIsNormalised) {
        return this->x;
    }
    else {
        return (this->x) / std::sqrt(2.0 * this->Dbulk * this->t);
    }
}

double SPR_calculator_RealSpace::getTime() {
    return this->t;
}

Eigen::VectorXd SPR_calculator_RealSpace::getMFPbins() {
    return this->MFPbins;
}

void SPR_calculator_RealSpace::updateDiffusivity() {
    alma::analytic1D::BasicProperties_calculator propCalc(
        this->poscar, this->grid, this->w, this->T);
    propCalc.setDirection(this->unitvector);

    this->Dbulk = propCalc.getDiffusivity();
}

double SPR_calculator_RealSpace::getDiffusivity() {
    return this->Dbulk;
}

Eigen::VectorXd SPR_calculator_RealSpace::getSourceTransient(
    const Eigen::Ref<Eigen::VectorXd> timegrid) {
    // output variable
    int Nt = timegrid.size();
    Eigen::VectorXd result(Nt);
    result.setConstant(0.0);

    // calculate the propagator function psi(xi) over a
    // logarithmic spatial frequency grid.
    // The following choices are recommended to assure sufficient accuracy
    // of the quadrature scheme: Nxi > 5000, ximin = 1e-3, ximax = 1e10;

    int Nxi = 5001;
    double ximin = 1e-3;
    double ximax = 1e10;

    psi_calculator psiCalc(this->poscar, this->grid, this->w, this->T);
    psiCalc.setDirection(unitvector);
    psiCalc.setLogGrid(ximin, ximax, Nxi);

    Eigen::VectorXd psi(psiCalc.getPsi());
    Eigen::VectorXd xi(psiCalc.getSpatialFrequencies());

    // fix potential numerical issues
    for (int nxi = 0; nxi < Nxi; nxi++) {
        if (xi(nxi) < 1e3) {
            psi(nxi) = this->Dbulk * xi(nxi) * xi(nxi);
        }
    }

    // perform piecewise integration of (1/pi)*exp(-psi(xi)*t) over xi

    for (int nt = 0; nt < Nt; nt++) {
        double t = timegrid(nt);

        for (int nxi = 0; nxi < Nxi - 1; nxi++) {
            if (std::abs(psi(nxi + 1) - psi(nxi)) <
                1e-12) { // "psileft = psiright"
                result(nt) += std::exp(-psi(nxi) * t) * (xi(nxi + 1) - xi(nxi));
            }
            else {
                result(nt) +=
                    (std::exp(-psi(nxi) * t) - std::exp(-psi(nxi + 1) * t)) *
                    (xi(nxi + 1) - xi(nxi)) / ((psi(nxi + 1) - psi(nxi)) * t);
            }
        }
    }

    return (1.0 / alma::constants::pi) * result.array();
}

Eigen::VectorXd SPR_calculator_RealSpace::getSPR() {
    // output variable
    int Nx = this->x.size();
    this->Pxt.resize(Nx, 1);
    this->Pxt.setConstant(0.0);

    // calculate the propagator function psi(xi) over a
    // logarithmic spatial frequency grid.
    // The number of xi points must be odd so that the grid contains
    // an integer number (Nxi-1)/2 of successive xi triplets.
    //
    // The following choices are recommended to assure sufficient accuracy
    // of the quadrature scheme: Nxi > 5000, ximin = 1e-3, ximax = 1e10;

    int Nxi = 5001;
    double ximin = 1e-3;
    double ximax = 1e10;

    psi_calculator psiCalc(this->poscar, this->grid, this->w, this->T);
    psiCalc.setDirection(unitvector);
    psiCalc.setLogGrid(ximin, ximax, Nxi);

    Eigen::VectorXd psi(psiCalc.getPsi());
    Eigen::VectorXd xi(psiCalc.getSpatialFrequencies());

    // group information with respect to successive xi triplets
    int N3 = (Nxi - 1) / 2;
    Eigen::VectorXd xileft(N3);
    Eigen::VectorXd psileft(N3);
    Eigen::VectorXd xiright(N3);
    Eigen::VectorXd psiright(N3);
    Eigen::VectorXd ximiddle(N3);
    Eigen::VectorXd psimiddle(N3);

    for (int n3 = 0; n3 < N3; n3++) {
        xileft(n3) = xi(2 * n3);
        psileft(n3) = psi(2 * n3);
        ximiddle(n3) = xi(2 * n3 + 1);
        psimiddle(n3) = psi(2 * n3 + 1);
        xiright(n3) = xi(2 * n3 + 2);
        psiright(n3) = psi(2 * n3 + 2);
    }

    // perform piecewise power law fit psi(xi) = A0*xi^alpha0 over the
    // successive xi intervals

    Eigen::VectorXd A0(N3);
    Eigen::VectorXd alpha0(N3);
    Eigen::VectorXd buffer1 = (psiright.array() / psileft.array()).log();
    Eigen::VectorXd buffer2 = (xiright.array() / xileft.array()).log();
    alpha0 = buffer1.array() / buffer2.array();
    A0 = psimiddle.array() / Eigen::pow(ximiddle.array(), alpha0.array());

    // use power law fittings for piecewise Taylor series expansions of
    // exp(-t*psi(xi)) around the midpoints of the xi intervals:
    // exp(-t*psi(xi)) \approx exp(-t*psimiddle) * (1 + B1*(xi-ximiddle) +
    // B2*(xi-xmiddle)^2)

    Eigen::VectorXd ones(N3);
    ones.setConstant(1.0);
    Eigen::VectorXd twos(N3);
    twos.setConstant(2.0);
    Eigen::VectorXd alphamin1 = alpha0 - ones;
    Eigen::VectorXd alphamin2 = alpha0 - twos;
    Eigen::VectorXd twoalphamin2 = 2.0 * alphamin1;

    Eigen::VectorXd B1 =
        -this->t * alpha0.array() * A0.array() *
        (Eigen::pow(ximiddle.array(), alphamin1.array())).array();
    buffer1 = this->t * alpha0.array().square() * A0.array().square() *
              (Eigen::pow(ximiddle.array(), (twoalphamin2).array())).array();

    buffer2 = alpha0.array() * alphamin1.array() * A0.array() *
              (Eigen::pow(ximiddle.array(), alphamin2.array())).array();

    Eigen::VectorXd B2 = 0.5 * this->t * (buffer1.array() - buffer2.array());

    // rewrite (1+B1*(xi-ximiddle)+B2*(xi-ximiddle)^2) as
    // (C0 + C1*xi + C2*xi^2) and perform piecewise evaluation of
    // Fourier inversion integral exp(-psi(xi)*t)*cos(xi*x)

    Eigen::VectorXd expfactors = (-this->t * psimiddle).array().exp();
    Eigen::VectorXd C0 =
        expfactors.array() * (ones.array() - B1.array() * ximiddle.array() +
                              B2.array() * ximiddle.array().square());
    Eigen::VectorXd C1 = expfactors.array() *
                         (B1.array() - (2.0 * B2).array() * ximiddle.array());
    Eigen::VectorXd C2 = expfactors.array() * B2.array();

    Eigen::VectorXd primitive_left(N3);
    Eigen::VectorXd primitive_right(N3);
    Eigen::VectorXd argument(N3);
    Eigen::VectorXd SIN(N3);
    Eigen::VectorXd COS(N3);

    for (int nx = 0; nx < Nx; nx++) {
        double xval;

        if (this->gridIsNormalised) {
            xval = this->x(nx) * std::sqrt(2.0 * this->Dbulk * this->t);
        }
        else {
            xval = this->x(nx);
        }

        argument = xval * xiright;
        SIN = argument.array().sin();
        COS = argument.array().cos();

        primitive_right =
            (C0 / xval).array() * SIN.array() +
            (C1 / (xval * xval)).array() *
                (COS.array() + argument.array() * SIN.array()) +
            (C2 / (xval * xval * xval)).array() *
                ((-2.0 * SIN).array() + (2.0 * argument).array() * COS.array() +
                 (argument.array().square()).array() * SIN.array());


        argument = xval * xileft;
        SIN = argument.array().sin();
        COS = argument.array().cos();

        primitive_left =
            (C0 / xval).array() * SIN.array() +
            (C1 / (xval * xval)).array() *
                (COS.array() + argument.array() * SIN.array()) +
            (C2 / (xval * xval * xval)).array() *
                ((-2.0 * SIN).array() + (2.0 * argument).array() * COS.array() +
                 (argument.array().square()).array() * SIN.array());

        double integral =
            (primitive_right.array() - primitive_left.array()).sum();

        this->Pxt(nx) = integral / alma::constants::pi;
    }

    if (this->normaliseoutput) {
        this->Pxt =
            std::sqrt(4.0 * alma::constants::pi * this->Dbulk * this->t) *
            this->Pxt;
    }

    return this->Pxt;
}

Eigen::MatrixXd SPR_calculator_RealSpace::resolveSPRbyMFP() {
    // output variable
    int Nx = this->x.size();
    int Nbins = this->MFPbins.size();

    this->Pmodes.resize(Nx, Nbins);
    this->Pmodes.fill(0.0);

    // calculate the propagator function psi(xi) over a
    // logarithmic spatial frequency grid.
    // The number of xi points must be odd so that the grid contains
    // an integer number (Nxi-1)/2 of successive xi triplets.
    //
    // The following choices are recommended to assure sufficient accuracy
    // of the quadrature scheme: Nxi > 5000, ximin = 1e-3, ximax = 1e10;

    int Nxi = 5001;
    double ximin = 1e-3;
    double ximax = 1e10;

    psi_calculator psiCalc(this->poscar, this->grid, this->w, this->T);
    psiCalc.setDirection(unitvector);
    psiCalc.setLogGrid(ximin, ximax, Nxi);

    Eigen::VectorXd psi(psiCalc.getPsi());
    Eigen::VectorXd xi(psiCalc.getSpatialFrequencies());

    // group information with respect to successive xi triplets
    int N3 = (Nxi - 1) / 2;

    Eigen::VectorXd xileft(N3);
    Eigen::VectorXd psileft(N3);
    Eigen::VectorXd xiright(N3);
    Eigen::VectorXd psiright(N3);
    Eigen::VectorXd ximiddle(N3);
    Eigen::VectorXd psimiddle(N3);

    for (int n3 = 0; n3 < N3; n3++) {
        xileft(n3) = xi(2 * n3);
        psileft(n3) = psi(2 * n3);
        ximiddle(n3) = xi(2 * n3 + 1);
        psimiddle(n3) = psi(2 * n3 + 1);
        xiright(n3) = xi(2 * n3 + 2);
        psiright(n3) = psi(2 * n3 + 2);
    }

    // perform piecewise power law fit psi(xi) = A0*xi^alpha0 over the
    // successive xi intervals

    Eigen::VectorXd A0(N3);
    Eigen::VectorXd alpha0(N3);

    Eigen::VectorXd buffer1 = (psiright.array() / psileft.array()).log();
    Eigen::VectorXd buffer2 = (xiright.array() / xileft.array()).log();
    alpha0 = buffer1.array() / buffer2.array();
    A0 = psimiddle.array() / Eigen::pow(ximiddle.array(), alpha0.array());

    // use power law fittings for piecewise Taylor series expansions of
    // exp(-t*psi(xi)) around the midpoints of the xi intervals:
    // exp(-t*psi(xi)) \approx exp(-t*psimiddle) * (1 + B1*(xi-ximiddle) +
    // B2*(xi-xmiddle)^2)

    Eigen::VectorXd ones(N3);
    ones.setConstant(1.0);
    Eigen::VectorXd twos(N3);
    twos.setConstant(2.0);

    Eigen::VectorXd alphamin1 = alpha0 - ones;
    Eigen::VectorXd alphamin2 = alpha0 - twos;
    Eigen::VectorXd twoalphamin2 = 2.0 * alphamin1;

    Eigen::VectorXd B1 =
        -t * alpha0.array() * A0.array() *
        (Eigen::pow(ximiddle.array(), alphamin1.array())).array();
    buffer1 = t * alpha0.array().square() * A0.array().square() *
              (Eigen::pow(ximiddle.array(), (twoalphamin2).array())).array();

    buffer2 = alpha0.array() * alphamin1.array() * A0.array() *
              (Eigen::pow(ximiddle.array(), alphamin2.array())).array();

    Eigen::VectorXd B2 = 0.5 * t * (buffer1.array() - buffer2.array());

    // the Fourier image of the energy contained within a mode pair
    // (forward and backward group velocities) is given by
    // (Cmode/Ctotal) * F(xi) * exp[-psi(xi)*t] with
    // F(xi) = [1 + tau*psi(xi)] / (1 + xi^2*MFPproj^2).
    // Expand F in a Taylor series around the midpoints of the xi intervals:
    // F(xi) \approx D0 + D1*(xi-ximiddle) + D2*(xi-ximiddle)^2

    auto nqpoints = this->grid->nqpoints;
    auto nmodes =
        static_cast<std::size_t>(this->grid->get_spectrum_at_q(0).omega.size());

    double Cv_tot = 1e27 * alma::calc_cv(*this->poscar, *this->grid, this->T);

    // The Gamma point is ignored.
    for (decltype(nqpoints) iq = 1; iq < nqpoints; ++iq) {
        auto sp = this->grid->get_spectrum_at_q(iq);
        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            // obtain relaxation time of these modes [seconds]
            double w0 = this->w->operator()(im, iq);
            double tau0 = (w0 == 0.) ? 0. : (1e-12 / w0);

            // obtain volumetric heat capacity of these modes [J/m^3-K]
            double prefactor = 1e27 * alma::constants::kB /
                               this->grid->nqpoints / this->poscar->V;
            double Cv = prefactor * alma::bose_einstein_kernel(sp.omega[im], T);

            // obtain projected group velocity [m/s] and MFP [m]
            double vg = 1e3 * sp.vg.col(im).matrix().norm();
            double vg_proj = 1e3 * this->unitvector.dot(sp.vg.col(im).matrix());

            double MFP = vg * tau0;
            double MFP_proj = std::abs(vg_proj * tau0);

            double tau = tau0; // initialise to intrinsic value

            Eigen::VectorXd spacedecay =
                (1.0 + MFP_proj * MFP_proj * ximiddle.array().square());

            // calculate Taylor expansion of F(xi)
            Eigen::VectorXd D0 =
                (1.0 + tau * psimiddle.array()).array() / spacedecay.array();

            Eigen::VectorXd D1 =
                tau * A0.array() * alpha0.array() *
                Eigen::pow(ximiddle.array(), alphamin1.array()) /
                spacedecay.array();
            D1 = D1.array() -
                 2.0 * MFP_proj * MFP_proj *
                     (1 + tau * A0.array() *
                              Eigen::pow(ximiddle.array(), alpha0.array())) *
                     ximiddle.array() / spacedecay.array().square();

            Eigen::VectorXd D2 =
                0.5 * tau * A0.array() * alpha0.array() * alphamin1.array() *
                Eigen::pow(ximiddle.array(), alphamin2.array()) /
                spacedecay.array();
            D2 = D2.array() - tau * MFP_proj * MFP_proj * A0.array() *
                                  Eigen::pow(ximiddle.array(), alpha0.array()) *
                                  (2.0 * alpha0.array() + 1.0) /
                                  spacedecay.array().square();
            D2 = D2.array() +
                 4.0 * std::pow(MFP_proj, 4.0) * ximiddle.array().square() *
                     (1 + tau * A0.array() *
                              Eigen::pow(ximiddle.array(), alpha0.array())) /
                     spacedecay.array().cube();
            D2 = D2.array() - MFP_proj * MFP_proj * ones.array() /
                                  spacedecay.array().square();

            // evaluate Taylor expansion of exp[-psi(xi)*t] * F(xi) up to second
            // order

            Eigen::VectorXd B0tot = D0;
            Eigen::VectorXd B1tot = D1.array() + D0.array() * B1.array();
            Eigen::VectorXd B2tot =
                D0.array() * B2.array() + D1.array() * B1.array() + D2.array();

            // rewrite (B0tot+B1tot*(xi-ximiddle)+B2tot*(xi-ximiddle)^2) as
            // (C0 + C1*xi + C2*xi^2) and perform piecewise evaluation of
            // Fourier inversion integral exp(-psi(xi)*t)*cos(xi*x)

            Eigen::VectorXd expfactors = (-t * psimiddle).array().exp();
            Eigen::VectorXd C0 =
                expfactors.array() *
                (B0tot.array() - B1tot.array() * ximiddle.array() +
                 B2tot.array() * ximiddle.array().square());
            Eigen::VectorXd C1 =
                expfactors.array() *
                (B1tot.array() - (2.0 * B2tot).array() * ximiddle.array());
            Eigen::VectorXd C2 = expfactors.array() * B2tot.array();

            // perform piecewise integration

            Eigen::VectorXd prim_left(N3);
            Eigen::VectorXd prim_right(N3);
            Eigen::VectorXd argument(N3);
            Eigen::VectorXd SIN(N3);
            Eigen::VectorXd COS(N3);

            for (int nx = 0; nx < Nx; nx++) {
                double xval;

                if (this->gridIsNormalised) {
                    xval = this->x(nx) * std::sqrt(2.0 * this->Dbulk * this->t);
                }
                else {
                    xval = this->x(nx);
                }

                argument = xval * xiright;
                SIN = argument.array().sin();
                COS = argument.array().cos();

                prim_right =
                    (C0 / xval).array() * SIN.array() +
                    (C1 / (xval * xval)).array() *
                        (COS.array() + argument.array() * SIN.array()) +
                    (C2 / (xval * xval * xval)).array() *
                        ((-2.0 * SIN).array() +
                         (2.0 * argument).array() * COS.array() +
                         (argument.array().square()).array() * SIN.array());

                argument = xval * xileft;
                SIN = argument.array().sin();
                COS = argument.array().cos();

                prim_left =
                    (C0 / xval).array() * SIN.array() +
                    (C1 / (xval * xval)).array() *
                        (COS.array() + argument.array() * SIN.array()) +
                    (C2 / (xval * xval * xval)).array() *
                        ((-2.0 * SIN).array() +
                         (2.0 * argument).array() * COS.array() +
                         (argument.array().square()).array() * SIN.array());

                double integral =
                    (prim_right.array() - prim_left.array()).sum() /
                    alma::constants::pi;

                // determine to which MFP bin this mode belongs

                double logL = std::log10(MFP);
                double logMFPmin = std::log10(this->MFPbins(0));
                double logMFPmax = std::log10(this->MFPbins(Nbins - 1));
                int binindex = static_cast<int>(
                    std::floor((logL - logMFPmin) * static_cast<double>(Nbins) /
                               (logMFPmax - logMFPmin)));
                if (binindex < 0) {
                    binindex = 0;
                }
                if (binindex >= Nbins) {
                    binindex = Nbins - 1;
                }

                this->Pmodes(nx, binindex) += (Cv / Cv_tot) * integral;
            }
        }
    }

    if (this->normaliseoutput) {
        this->Pmodes =
            std::sqrt(4.0 * alma::constants::pi * this->Dbulk * this->t) *
            this->Pmodes;
    }

    return this->Pmodes;
}

/////////////////// MSD_calculator_Laplace ///////////////////

MSD_calculator_Laplace::MSD_calculator_Laplace(
    const alma::Crystal_structure* poscar_init,
    const alma::Gamma_grid* grid_init,
    const Eigen::ArrayXXd* w_init,
    double T_init)

    : poscar(poscar_init), grid(grid_init), w(w_init), T(T_init)

{
    this->unitvector = Eigen::Vector3d(1.0, 0.0, 0.0);
}

void MSD_calculator_Laplace::setDirection(const Eigen::Vector3d u) {
    this->unitvector = u;
    (this->unitvector).normalize();
}

void MSD_calculator_Laplace::setLinGrid(double fmin, double fmax, int Nf) {
    this->f.setLinSpaced(Nf, fmin, fmax);
    this->s = std::complex<double>(0.0, 2.0 * alma::constants::pi) *
              f.cast<std::complex<double>>();
}

void MSD_calculator_Laplace::setLogGrid(double fmin, double fmax, int Nf) {
    this->f = alma::logSpace(fmin, fmax, Nf);
    this->s = std::complex<double>(0.0, 2.0 * alma::constants::pi) *
              f.cast<std::complex<double>>();
}

void MSD_calculator_Laplace::setLaplaceGrid(
    const Eigen::Ref<const Eigen::VectorXcd> sgrid) {
    this->s = sgrid;
}

Eigen::VectorXd MSD_calculator_Laplace::getGrid() {
    return this->f;
}

Eigen::VectorXcd MSD_calculator_Laplace::getLaplaceGrid() {
    return this->s;
}

Eigen::VectorXcd MSD_calculator_Laplace::getMSD() {
    int Ns = this->s.size();

    Eigen::VectorXcd numerator(Ns);
    Eigen::VectorXcd denominator(Ns);
    numerator.setConstant(std::complex<double>(0.0, 0.0));
    denominator.setConstant(std::complex<double>(0.0, 0.0));

    auto nqpoints = this->grid->nqpoints;
    auto nmodes =
        static_cast<std::size_t>(this->grid->get_spectrum_at_q(0).omega.size());

    // The Gamma point is ignored.
    for (decltype(nqpoints) iq = 1; iq < nqpoints; ++iq) {
        auto sp = this->grid->get_spectrum_at_q(iq);
        for (decltype(nmodes) im = 0; im < nmodes; ++im) {
            // obtain relaxation time [seconds]
            double w0 = this->w->operator()(im, iq);
            double tau0 = (w0 == 0.) ? 0. : (1e-12 / w0);

            // obtain volumetric heat capacity [J/m^3-K]
            double prefactor = 1e27 * alma::constants::kB /
                               this->grid->nqpoints / this->poscar->V;
            double C = prefactor * alma::bose_einstein_kernel(sp.omega[im], T);

            // obtain projected group velocity [m/s]
            double vg_proj = 1e3 * this->unitvector.dot(sp.vg.col(im).matrix());

            for (int ns = 0; ns < Ns; ns++) {
                std::complex<double> sval = this->s(ns);
                std::complex<double> buffer =
                    std::complex<double>(1.0, 0.0) + sval * tau0;
                numerator(ns) +=
                    C * vg_proj * vg_proj * tau0 / (buffer * buffer);
                denominator(ns) += C / buffer;
            }
        }
    }

    this->MSD = std::complex<double>(2.0, 0.0) * numerator.array() /
                (denominator.array() * this->s.array() * this->s.array());

    return this->MSD;
}

/////////////////// MSD_calculator_RealTime ///////////////////

MSD_calculator_RealTime::MSD_calculator_RealTime(
    const alma::Crystal_structure* poscar_init,
    const alma::Gamma_grid* grid_init,
    const Eigen::ArrayXXd* w_init,
    double T_init)

    : poscar(poscar_init), grid(grid_init), w(w_init), T(T_init)

{
    this->unitvector = Eigen::Vector3d(1.0, 0.0, 0.0);
    this->normaliseoutput = false;
    this->updateDiffusivity();

    // initialise Gaver-Stehfest coefficients

    this->GS_depth = 16;
    this->GS_coeffs.resize(this->GS_depth);
    this->GS_coeffs.setConstant(0.0);

    int nn2 = this->GS_depth / 2;

    for (int n = 1; n <= this->GS_depth; n++) {
        double coeff_buffer = 0.0;

        for (int k = std::floor((n + 1) / 2); k <= std::min(n, nn2); k++) {
            coeff_buffer += ((std::pow(k, nn2)) * this->factorial(2 * k)) /
                            (this->factorial(nn2 - k) * this->factorial(k) *
                             this->factorial(k - 1) * this->factorial(n - k) *
                             this->factorial(2 * k - n));
        }

        this->GS_coeffs(n - 1) = std::pow(-1, n + nn2) * coeff_buffer;
    }
}

void MSD_calculator_RealTime::setDirection(const Eigen::Vector3d u) {
    this->unitvector = u;
    (this->unitvector).normalize();
    this->updateDiffusivity();
}

void MSD_calculator_RealTime::setLinGrid(double tmin, double tmax, int Nt) {
    this->t.setLinSpaced(Nt, tmin, tmax);
}

void MSD_calculator_RealTime::setLogGrid(double tmin, double tmax, int Nt) {
    this->t = alma::logSpace(tmin, tmax, Nt);
}

void MSD_calculator_RealTime::setTimeGrid(
    const Eigen::Ref<const Eigen::VectorXd> tgrid) {
    this->t = tgrid;
}

Eigen::VectorXd MSD_calculator_RealTime::getGrid() {
    return this->t;
}

double MSD_calculator_RealTime::factorial(int n) {
    return (n == 1 || n == 0) ? 1
                              : this->factorial(n - 1) * static_cast<double>(n);
}

void MSD_calculator_RealTime::normaliseOutput(bool norm) {
    this->normaliseoutput = norm;
}

void MSD_calculator_RealTime::updateDiffusivity() {
    alma::analytic1D::BasicProperties_calculator propCalc(
        this->poscar, this->grid, this->w, this->T);
    propCalc.setDirection(this->unitvector);

    this->Dbulk = propCalc.getDiffusivity();
}

double MSD_calculator_RealTime::getDiffusivity() {
    return this->Dbulk;
}

Eigen::VectorXd MSD_calculator_RealTime::getMSD() {
    int Nt = this->t.size();
    this->MSD.resize(Nt);

    // set up Laplace calculator

    alma::analytic1D::MSD_calculator_Laplace LaplaceCalc(
        this->poscar, this->grid, this->w, this->T);
    LaplaceCalc.setDirection(this->unitvector);

    Eigen::VectorXcd MSD_Laplace(this->GS_depth * Nt);
    Eigen::VectorXd result;

    // calculate MSD in time domain

    Eigen::VectorXcd sgrid(this->GS_depth * Nt);

    Eigen::VectorXd sbuffer(this->GS_depth);
    sbuffer.setLinSpaced(
        this->GS_depth, 1.0, static_cast<double>(this->GS_depth));
    Eigen::VectorXcd sbase(sbuffer.cast<std::complex<double>>());

    Eigen::VectorXd prefactors(Nt);

    for (int nt = 0; nt < Nt; nt++) { // build Laplace grid

        double s0 = std::log(2.0) / this->t(nt);
        prefactors(nt) = s0;
        Eigen::VectorXcd ssegment = s0 * sbase;
        sgrid.segment(nt * this->GS_depth, this->GS_depth) = ssegment;
    }

    LaplaceCalc.setLaplaceGrid(sgrid);
    MSD_Laplace = LaplaceCalc.getMSD();
    Eigen::VectorXcd Laplace_segment(this->GS_depth);

    for (int nt = 0; nt < Nt; nt++) {
        Laplace_segment =
            MSD_Laplace.segment(nt * this->GS_depth, this->GS_depth);
        result = prefactors(nt) * this->GS_coeffs.transpose() *
                 (Laplace_segment.array().real()).matrix();
        this->MSD(nt) = result(0);
    }

    if (this->normaliseoutput) {
        this->MSD = MSD.array() / (2.0 * this->Dbulk * this->t.array());
    }

    return this->MSD;
}

} // end namespace analytic1D
} // end namespace alma
