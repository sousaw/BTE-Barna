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
/// Code related to fully analytical 1D RTA solutions.

#include <constants.hpp>
#include <structures.hpp>
#include <qpoint_grid.hpp>
#include <Eigen/Dense>

namespace alma {

namespace analytic1D {

/// Class for computing basic thermal properties
/// (kappa, Cv, diffusivity) and cumulative functions
/// (resolved for MFP, energy, etc.) along a given transport direction.
/// Calculations involve appropriate thin film corrections
/// for both cross-plane and in-plane transport.

class BasicProperties_calculator {
public:
    /// Constructor: initialise internal variables
    BasicProperties_calculator(const alma::Crystal_structure* poscar,
                               const alma::Gamma_grid* grid,
                               const Eigen::ArrayXXd* w,
                               double T);

    /// Function setting the thermal transport axis
    void setDirection(const Eigen::Vector3d unitvector);

    /// Construct a logarithmic grid of MFP bins
    /// from MFPmin to MFPmax with Nbins elements
    void setLogMFPbins(double MFPmin, double MFPmax, int Nbins);

    /// Construct a logarithmic grid of MFP bins witn Nbins elements.
    /// MFPmax is assigned automatically to twice the largest MFP.
    /// MFPmin is assigned to 1e-6*largest MFP.
    void setAutoMFPbins(int Nbins);

    /// Construct a logarithmic grid of MFP bins witn Nbins elements.
    /// MFPmax is assigned automatically to the twice the largest projected MFP.
    /// MFPmin is assigned to 1e-6*largest projected MFP.
    void setAutoProjMFPbins(int Nbins);

    /// Construct a logarithmic grid of tau bins
    /// from taumin to taumax with Nbins elements
    void setLogRTbins(double taumin, double taumax, int Nbins);

    /// Construct a logarithmic grid of tau bins witn Nbins elements.
    /// taumax is assigned automatically to the twice the largest relaxation
    /// time.
    /// taumin is assigned to 1e-6*largest relaxation time.
    void setAutoRTbins(int Nbins);

    /// Construct a linear grid of omega bins
    /// from omegamin to omegamax with Nbins elements
    void setLinOmegabins(double omegamin, double omegamax, int Nbins);

    /// Construct a linear grid of omega bins witn Nbins elements.
    /// omegamax is assigned automatically to 1.1*largest angular frequency.
    /// omegamin is automatically set to 0.
    void setAutoOmegabins(int Nbins);

    /// Retrieve the MFP bins
    Eigen::VectorXd getMFPbins();

    /// Retrieve the relaxation time bins
    Eigen::VectorXd getRTbins();

    /// Retrieve the relaxation time bins
    Eigen::VectorXd getOmegabins();

    /// Specify that the medium should be treated as
    /// infinite bulk (default option)
    void setBulk();

    /// Specify that the medium should be treated as
    /// a thin film with provided thickness.
    /// This option corrects MFPs and relaxation times
    /// in conductivity calculations.
    void setInPlaneFilm(double filmthickness,
                        const Eigen::Vector3d normal,
                        double specularity = 0.0);
    void setCrossPlaneFilm(double filmthickness);

    /// Retrieve thermal conductivity
    double getConductivity();

    /// Obtain spectrally computed thermal conductivity
    double getSpectralConductivity();

    /// Compute the heat capacity and retrieve it
    double getCapacity();

    /// Compute the Fourier diffusivity and retrieve it
    double getDiffusivity();

    /// Retrieve the dominant projected MFP as defined below
    double getDominantProjMFP();

    /// Retrieve the dominant phonon relaxation time
    double getDominantRT();

    /// Compute the anisotropy index kappa_max/kappa_min and retrieve it.
    double getAnisotropyIndex(const alma::Symmetry_operations* syms);

    /// Choose resolving cumulative curves by MFP
    void resolveByMFP();

    /// Choose resolving cumulative curves by projected MFP
    void resolveByProjMFP();

    /// Choose resolving cumulative curves by relaxation time
    void resolveByRT();

    /// Choose resolving cumulative curves by angular frequency
    void resolveByOmega();

    /// Compute the cumulative thermal conductivity curve
    /// with respect to the MFP bins and retrieve it
    Eigen::VectorXd getCumulativeConductivity();

    /// Compute the cumulative heat capacity curve
    /// with respect to the MFP bins and retrieve it
    Eigen::VectorXd getCumulativeCapacity();


    /// Compute the cumulative hydrodinamic l2
    /// using RTA and assuming isotropy
    Eigen::VectorXd getCumulativel2RTAiso();

    /// Set the number of bins to be used in DOS evaluation
    void setDOSgridsize(int nbins);

    /// Obtain the DOS energy grid (in meV)
    Eigen::VectorXd getDOSgrid();

    /// Calculate and obtain the DOS
    Eigen::VectorXd getDOS();

private:
    /// Pointer to description of the unit cell
    const alma::Crystal_structure* poscar;
    /// Pointer to phonon spectrum on a regular q-point grid
    const alma::Gamma_grid* grid;
    /// Pointer to scattering rates for all modes irreducible points in the grid
    const Eigen::ArrayXXd* w;
    /// Temperature
    double T;

    /// Unitvector describing 1D thermal transport axis
    Eigen::Vector3d unitvector;
    /// Unitvector describing the film normal for in-plane transport
    Eigen::Vector3d normalvector;
    /// MFP bins for cumulative curves
    Eigen::VectorXd MFPbins;
    /// relaxation time bins for cumulative curves
    Eigen::VectorXd taubins;
    /// omega bins for cumulative curves
    Eigen::VectorXd omegabins;
    /// Max MFP value present in medium
    double internal_MFPmax;
    /// Max proj MFP value present in medium
    double internal_MFPprojmax;
    /// Max relaxation time present in medium
    double internal_taumax;
    /// Max phonon frequency present in medium
    double internal_omegamax;
    /// Thin film specifiers
    bool thinfilm;
    bool crossplane;
    double filmthickness;
    double specularity; // for in-plane films
    /// Basic properties
    double kappa;
    double Cv;
    /// "Dominant" projected MFP, defined as kappa-weighted average
    /// <Lambda_proj> = sum[(kappamode/kappabulk)*Lambda_proj]
    double dominantProjMFP;
    /// "Dominant" relaxation time, defined as kappa-weighted average
    /// <RT> = sum[(kappamode/kappabulk)*RT_mode]
    double dominantRT;
    /// 3D rotation matrix that rotates the cartesian coordinate system such
    /// that
    /// the transport axis becomes (1,0,0) and film normal becomes (0,0,1)
    Eigen::Matrix3d FuchsRotation;
    /// Precalculate the 3D rotation matrix used for in-plane Fuch corrections.
    void initFuchsRotation();
    /// Precalculate thermal properties
    void updateMe();
    /// Calculated cumulative conductivity function
    Eigen::VectorXd kappacumul;
    /// Calculated cumulative capacity function
    Eigen::VectorXd Cvcumul;
    /// Calculated cumulative l**2
    Eigen::VectorXd l2cumul;

    /// Specifier for kappacumul calculation
    enum kappacumulID {
        resolve_by_MFP,
        resolve_by_ProjMFP,
        resolve_by_RT,
        resolve_by_omega
    };
    int kappacumulIdentifier;
    /// Number of bins to be used for DOS evaluation
    int DOS_Nbins;
};

/// Class for computing the RTA propagator function psi(xi) of a medium.
/// The psi function fully determines the analytical single pulse
/// response of the infinite bulk in weakly quasi-ballistic regime
/// (time scales exceeding phonon relaxation times):
/// Energy density in Fourier-Laplace domain = 1/[s + psi(xi)]
/// Energy density in Fourier-time domain = exp[-psi(xi)*t]

class psi_calculator {
public:
    /// Constructor: initialise internal variables
    psi_calculator(const alma::Crystal_structure* poscar,
                   const alma::Gamma_grid* grid,
                   const Eigen::ArrayXXd* w,
                   double T);

    /// Function setting the thermal transport axis
    void setDirection(const Eigen::Vector3d unitvector);

    /// Construct a linear grid of spatial frequencies
    /// from ximin to ximax with Nxi elements
    void setLinGrid(double ximin, double ximax, int Nxi);

    /// Construct a logarithmic grid of spatial frequencies
    /// from ximin to ximax with Nxi elements
    void setLogGrid(double ximin, double ximax, int Nxi);

    /// Manually set a grid of spatial frequencies
    void setXiGrid(const Eigen::Ref<const Eigen::VectorXd> xigrid);

    /// Retrieve the spatial frequency grid
    Eigen::VectorXd getSpatialFrequencies();

    /// Determine whether computation output should be
    /// normalised by the Fourier solution Dbulk*xi^2
    void normaliseOutput(bool norm);

    /// Obtain Fourier diffusivity of the medium
    double getDiffusivity();

    /// Compute psi function and retrieve it
    Eigen::VectorXd getPsi();

private:
    /// Pointer to description of the unit cell
    const alma::Crystal_structure* poscar;
    /// Pointer to phonon spectrum on a regular q-point grid
    const alma::Gamma_grid* grid;
    /// Pointer to scattering rates for all modes irreducible points in the grid
    const Eigen::ArrayXXd* w;
    /// Temperature
    double T;

    /// Unitvector describing 1D thermal transport axis
    Eigen::Vector3d unitvector;
    /// Spatial frequency grid
    Eigen::VectorXd xi;
    /// Normalisation specifier
    bool scaleoutput;
    /// Fourier diffusivity
    double Dbulk;
    void updateDiffusivity();
    /// Calculated psi function
    Eigen::VectorXd psi;
};

/////////////////// SPR_calculator_FourierLaplace ///////////////////

/// Class for computing the exact analytical RTA single pulse
/// energy density response of the infinite bulk medium
/// in Fourier-Laplace domain. [see PRB 91 085202 (2015)]

class SPR_calculator_FourierLaplace {
public:
    /// Constructor: initialise internal variables
    SPR_calculator_FourierLaplace(const alma::Crystal_structure* poscar,
                                  const alma::Gamma_grid* grid,
                                  const Eigen::ArrayXXd* w,
                                  double T);

    /// Function setting the thermal transport axis
    void setDirection(const Eigen::Vector3d unitvector);

    /// Construct a linear grid of spatial frequencies
    /// from ximin to ximax with Nxi elements
    void setLinSpatialGrid(double ximin, double ximax, int Nxi);

    /// Construct a logarithmic grid of spatial frequencies
    /// from ximin to ximax with Nxi elements
    void setLogSpatialGrid(double ximin, double ximax, int Nxi);

    /// Construct a linear grid of temporal frequencies
    /// from fmin to fmax with Nf elements.
    /// From this a Laplace grid s = 2*pi*1i*f is created.
    void setLinTemporalGrid(double fmin, double fmax, int Nf);

    /// Construct a logarithmic grid of temporal frequencies
    /// from fmin to fmax with Nf elements.
    /// From this a Laplace grid s = 2*pi*1i*f is created.
    void setLogTemporalGrid(double fmin, double fmax, int Nf);

    /// Retrieve the spatial frequency grid
    Eigen::VectorXd getSpatialFrequencies();

    /// Retrieve the temporal frequency grid
    Eigen::VectorXd getTemporalFrequencies();

    /// Compute SPR and retrieve it
    Eigen::MatrixXcd getSPR();

private:
    /// Pointer to description of the unit cell
    const alma::Crystal_structure* poscar;
    /// Pointer to phonon spectrum on a regular q-point grid
    const alma::Gamma_grid* grid;
    /// Pointer to scattering rates for all modes irreducible points in the grid
    const Eigen::ArrayXXd* w;
    /// Temperature
    double T;

    /// Unitvector describing 1D thermal transport axis
    Eigen::Vector3d unitvector;
    /// Spatial frequency grid
    Eigen::VectorXd xi;
    /// Temporal frequency grids
    Eigen::VectorXd f;
    Eigen::VectorXcd s;
    /// Calculated SPR
    Eigen::MatrixXcd Pxis;
};

/////////////////// SPR_calculator_RealSpace ///////////////////

/// Class for computing the approximate analytical RTA single pulse
/// energy density response in real space at a given time.
///
/// The approach is only valid in weakly quasiballistic regime
/// (time scales exceeding phonon relaxation times) so that
/// P(xi,t) \approx exp[-psi(xi)*t].
///
/// Calculations are performed by evaluating the Fourier inversion
/// (1/pi)*Integral(exp[-psi(xi)*t]*cos(xi*x),xi=0..infinity)
/// semi-analytically with 2nd order Filon-type quadrature.

class SPR_calculator_RealSpace {
public:
    /// Constructor: initialise internal variables
    SPR_calculator_RealSpace(const alma::Crystal_structure* poscar,
                             const alma::Gamma_grid* grid,
                             const Eigen::ArrayXXd* w,
                             double T);

    /// Function setting the thermal transport axis
    void setDirection(const Eigen::Vector3d unitvector);

    /// Construct a linear space grid
    /// from xmin to xmax with Nx elements
    void setLinGrid(double xmin, double xmax, int Nx);

    /// Construct a logarithmic space grid
    /// from xmin to xmax with Nx elements
    void setLogGrid(double xmin, double xmax, int Nx);

    /// Declare whether the space grid is normalised.
    /// If yes, space grid values are counted in
    /// Fourier diffusion lengths, i.e.
    /// x_actual = x_grid*sqrt(2*Dbulk*t)
    void declareGridNormalised(bool norm);

    // Optional normalisation of calculation output
    // by sqrt(4*pi*Dbulk*t)
    void normaliseOutput(bool norm);

    /// Set time value to be used for calculations
    void setTime(double t);

    /// Set MFP bins to be used for resolveSPRbyMFP()
    void setLogMFPbins(double MFPmin, double MFPmax, int Nbins);

    /// Retrieve the spatial grid
    Eigen::VectorXd getGrid();
    Eigen::VectorXd getNormalisedGrid();

    /// Retrieve time value
    double getTime();

    /// Retrieve MFP bins
    Eigen::VectorXd getMFPbins();

    /// Retrieve Fourier diffusivity along thermal transport axis
    double getDiffusivity();

    /// Compute source transient P(x=0) at the provided times
    Eigen::VectorXd getSourceTransient(
        const Eigen::Ref<Eigen::VectorXd> timegrid);

    /// Compute SPR and retrieve it
    Eigen::VectorXd getSPR();

    /// Compute SPR of individual modes resolved by MFP
    Eigen::MatrixXd resolveSPRbyMFP();

private:
    /// Pointer to description of the unit cell
    const alma::Crystal_structure* poscar;
    /// Pointer to phonon spectrum on a regular q-point grid
    const alma::Gamma_grid* grid;
    /// Pointer to scattering rates for all modes irreducible points in the grid
    const Eigen::ArrayXXd* w;
    /// Temperature
    double T;

    /// Unitvector describing 1D thermal transport axis
    Eigen::Vector3d unitvector;
    /// Unitvector describing the film normal for inplane transport
    Eigen::Vector3d filmnormal;
    /// Spatial grid
    Eigen::VectorXd x;
    bool gridIsNormalised;
    /// Output specifier
    bool normaliseoutput;
    /// MFP bins
    Eigen::VectorXd MFPbins;
    /// time value
    double t;
    /// diffusivity of the medium along the transport axis
    double Dbulk;
    void updateDiffusivity();
    /// Calculated SPR
    Eigen::VectorXd Pxt;
    /// Calculated SPR resolved by MFP
    Eigen::MatrixXd Pmodes;
};

/////////////////// MSD_calculator_Laplace ///////////////////

/// Class for computing the exact analytical RTA solution for
/// mean square thermal energy displacement. By definition,
/// the MSD is the variance of the macroscopic energy density:
/// MSD = Integral(x^2*P(x,s),x=-infinity..infinity).
/// Results are valid across all regimes
/// (from fully ballistic to fully diffusive transport).
/// Calculations are performed fully analytically by using
/// the moment generating properties of the energy density
/// in spatial frequency domain:
/// MSD = minus second derivative of P(xi,s) at xi = 0.

class MSD_calculator_Laplace {
public:
    /// Constructor: initialise internal variables
    MSD_calculator_Laplace(const alma::Crystal_structure* poscar,
                           const alma::Gamma_grid* grid,
                           const Eigen::ArrayXXd* w,
                           double T);

    /// Function setting the thermal transport axis
    void setDirection(const Eigen::Vector3d unitvector);

    /// Construct a linear temporal frequency grid
    /// from fmin to fmax with Nf elements
    void setLinGrid(double fmin, double fmax, int Nf);

    /// Construct a logarithmic temporal frequency grid
    /// from fmin to fmax with Nf elements
    void setLogGrid(double fmin, double fmax, int Nf);

    /// Manually set a Laplace grid
    void setLaplaceGrid(const Eigen::Ref<const Eigen::VectorXcd> sgrid);

    /// Retrieve the temporal frequency grid
    Eigen::VectorXd getGrid();
    Eigen::VectorXcd getLaplaceGrid();

    /// Compute MSD and retrieve it
    Eigen::VectorXcd getMSD();

private:
    /// Pointer to description of the unit cell
    const alma::Crystal_structure* poscar;
    /// Pointer to phonon spectrum on a regular q-point grid
    const alma::Gamma_grid* grid;
    /// Pointer to scattering rates for all modes irreducible points in the grid
    const Eigen::ArrayXXd* w;
    /// Temperature
    double T;

    /// Unitvector describing 1D thermal transport axis
    Eigen::Vector3d unitvector;
    /// Temporal frequency grid
    Eigen::VectorXd f;
    /// Laplace grid
    Eigen::VectorXcd s;
    /// Calculated MSD function
    Eigen::VectorXcd MSD;
};

/////////////////// MSD_calculator_RealTime ///////////////////

/// Class for computing the exact analytical RTA solution for
/// mean square thermal energy displacement in time domain.
///
/// Calculations are performed by numerically inverting
/// output from MSD_calculator_Laplace to time domain
/// with a Gaver-Stehfest scheme.

class MSD_calculator_RealTime {
public:
    /// Constructor: initialise internal variables
    MSD_calculator_RealTime(const alma::Crystal_structure* poscar,
                            const alma::Gamma_grid* grid,
                            const Eigen::ArrayXXd* w,
                            double T);

    /// Function setting the thermal transport axis
    void setDirection(const Eigen::Vector3d unitvector);

    /// Construct a linear time grid
    /// from tmin to tmax with Nt elements
    void setLinGrid(double tmin, double tmax, int Nt);

    /// Construct a logarithmic time grid
    /// from tmin to tmax with Nt elements
    void setLogGrid(double tmin, double tmax, int Nt);

    /// Manually set a time grid
    void setTimeGrid(const Eigen::Ref<const Eigen::VectorXd> tgrid);

    /// Retrieve the timegrid
    Eigen::VectorXd getGrid();

    /// Optional choice to normalise calculated results
    /// by Fourier solution 2*D*t.
    void normaliseOutput(bool norm);

    /// Retrieve Fourier diffusivity
    double getDiffusivity();

    /// Compute MSD and retrieve it
    Eigen::VectorXd getMSD();

private:
    /// Pointer to description of the unit cell
    const alma::Crystal_structure* poscar;
    /// Pointer to phonon spectrum on a regular q-point grid
    const alma::Gamma_grid* grid;
    /// Pointer to scattering rates for all modes irreducible points in the grid
    const Eigen::ArrayXXd* w;
    /// Temperature
    double T;

    /// Unitvector describing 1D thermal transport axis
    Eigen::Vector3d unitvector;
    /// Unitvector describing the film normal for inplane transport
    Eigen::Vector3d filmnormal;
    /// Time grid
    Eigen::VectorXd t;
    /// Fourier diffusivity
    double Dbulk;
    void updateDiffusivity();
    /// output specifier
    bool normaliseoutput;
    /// Calculated MSD function
    Eigen::VectorXd MSD;

    /// Gaver-Stehfest Laplace inversion coefficients
    int GS_depth;
    Eigen::VectorXd GS_coeffs;
    /// Helper function for Gaver-Stehfest Laplace inversion
    double factorial(int n);
};
} // end namespace analytic1D
} // end namespace alma
