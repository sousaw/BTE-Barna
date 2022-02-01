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
/// Classes and functions used for sampling from several distributions
/// used in the MC scheme.

#include <vector>
#include <utility>
#include <tuple>
#include <map>
#include <random>
#include <Eigen/Dense>
#include <pcg_random.hpp>
#include <structures.hpp>
#include <qpoint_grid.hpp>
#include <dos.hpp>
#include <processes.hpp>
#include <deviational_particle.hpp>
#include <mutex>

namespace alma {
/// Generate a random time delta from an exponential distribution
/// @param[in] w - a scattering rate
/// @param[in] rng - a random number generator
/// @return an exponential deviate
inline double random_dt(double w, pcg64& rng) {
    if (w == 0.)
        throw value_error("invalid scattering rate");
    ;
    double u = std::uniform_real_distribution(0., 1.)(rng);
    return -std::log(u) / w;
}


/// Base class for discrete distributions over a q-point grid.
class Grid_distribution {
protected:
    /// Distribution function.
    std::vector<double> cumulative;
    /// Random number generator.
    pcg64 rng;
    /// Vector with particle sign
    std::vector<alma::particle_sign> signs;
    /// The sum of cumulative
    double cumulsum = -1.0;
public:
    /// Number of q points.
    std::size_t nqpoints;
    /// Number of phonon modes at each q point.
    std::size_t nmodes;
    /// Empty constructor
    Grid_distribution() = default;

    /// Constructor.
    /// @param[in] grid - phonon spectrum on a regular grid
    /// @param[in] _rng - a random number generator
    Grid_distribution(const Gamma_grid& grid, pcg64& _rng)
        : rng(_rng), nqpoints(grid.nqpoints),
          nmodes(grid.get_spectrum_at_q(0).omega.size()) {
    }


    /// Trivial virtual destructor.
    virtual ~Grid_distribution() {
    }


    /// Fill the 'cumulative' vector.
    ///
    /// @param[in] p - vector with the unnormalized probabilities
    /// of each q point
    void fill_cumulative(const std::vector<double>& p);

    /// Draw a sample from the distribution.
    ///
    /// @return an array containing the mode number and the
    /// q-point number.
    std::array<std::size_t, 2> sample() {
        double u = std::uniform_real_distribution(0., 1.)(this->rng);
        auto pos = std::lower_bound(
                       this->cumulative.begin(), this->cumulative.end(), u) -
                   this->cumulative.begin();

        std::array<std::size_t, 2> nruter;
        nruter[0] = pos % this->nmodes;
        nruter[1] = pos / this->nmodes;
        return nruter;
    }

    /// @return a tuple containing the mode number, the
    /// q-point number and particle sign
    std::tuple<std::size_t, std::size_t, alma::particle_sign>
    sample_with_sign() {
        double u = std::uniform_real_distribution(0., 1.)(this->rng);
        auto pos = std::lower_bound(
                       this->cumulative.begin(), this->cumulative.end(), u) -
                   this->cumulative.begin();
        return std::make_tuple(
            pos % this->nmodes, pos / this->nmodes, this->signs[pos]);
    }
};

/// Objects of this class allow us to sample from a discrete
/// distribution over a q-point grid with a PMF proportional
/// to cv / tau, where cv is the contribution to the specific heat
/// and tau is the relaxation time.
class BE_derivative_distribution : public Grid_distribution {
public:
    /// Empty constructor

    BE_derivative_distribution() = default;

    /// Constructor.
    ///
    /// @param[in] grid - phonon spectrum on a regular grid
    /// @param[in] w - scattering rates
    /// @param[in] T - the temperature in K
    /// @param[in, out] _rng - a random number generator
    BE_derivative_distribution(const Gamma_grid& grid,
                               const Eigen::Ref<const Eigen::ArrayXXd>& w,
                               double T,
                               pcg64& _rng);
};

/// Objects of this class allow us to sample from a discrete
/// distribution over a q-point grid with a PMF proportional to one
/// component of the group velocity and to each mode's contribution
/// to the specific heat.
class Nabla_T_distribution : public Grid_distribution {
public:
    /// Empty constructor
    Nabla_T_distribution() = default;
    /// Constructor.
    ///
    /// @param[in] grid - phonon spectrum on a regular grid
    /// @param[in] nablaT - any vector parallel to the temperature
    /// gradient.
    /// @param[in] T - the temperature in K
    /// @param[in, out] _rng - a random number generator
    /// @param[in] theta - rotation of material with respect
    /// to x-axis
    Nabla_T_distribution(const Gamma_grid& grid,
                         const Eigen::Ref<const Eigen::Vector3d>& nablaT,
                         double T,
                         pcg64& _rng,
                         double theta = 0.);

    /// Get number of generated particles for given conditions
    ///
    ///@param[in] time - in ps
    ///@param[in] vol  - box_volume/unitcell_volume
    ///@param[in] Eff  - deviational particle energy
    std::size_t Ntogenerate(const double time,
                            const double vol,
                            const double Eff);

    /// Returns energy introduced by the generator per unit of time
    ///@param[in] vol  - box_volume/unitcell_volume
    double get_energy(const double vol);
};

/// Objects of this class allow us to sample from a discrete
/// distribution over a q-point grid given from a vector
class ref_distribution : public Grid_distribution {
private:
    std::mutex shield;
public:
    ///Empty constructor
    ref_distribution() = default;
    
    ref_distribution(const ref_distribution& that) : Grid_distribution(that) {
    }
    
    ref_distribution& operator=(const ref_distribution& that) {
        this->cumulative = that.cumulative;
        this->rng        = that.rng;
        this->signs      = that.signs;
        this->cumulsum   = that.cumulsum;
        this->nqpoints   = that.nqpoints;
        this->nmodes     = that.nmodes;
        return *this;
    }
    
    
    
    /// Constructor.
    ///
    /// @param[in] grid - phonon spectrum on a regular grid
    /// @param[in] fd   - the distribution
    /// @param[in, out] _rng - a random number generator
    ref_distribution(const Gamma_grid& grid,
                     const Eigen::VectorXd&  fd,
                     pcg64& _rng);
    
    /// Get number of generated particles for given conditions
    ///@param [in] vol - vol_box / volume_unit_cell
    ///@param [in] Eff - packet energy
    std::size_t Ntogenerate(const double vol,
                            const double Eff);
    
    
    std::tuple<std::size_t,std::size_t,
        alma::particle_sign> sample_dist() {
        
        shield.lock();
        auto res = this->sample_with_sign();
        shield.unlock();
        return res;
    }
    
};


/// Objects of this class allow us to sample from a discrete
/// distribution over a q-point grid with a PMF corresponding to an
/// isothermal wall of given orientation and equilibrium
/// temperature.  They are also useful for simple periodic systems
/// if Teq is set accordingly.
class Isothermal_wall_distribution : public Grid_distribution {
public:
    /// Empty constructor
    Isothermal_wall_distribution() = default;
    /// Constructor.
    ///
    /// @param[in] grid - phonon spectrum on a regular grid
    /// @param[in] Twall - the temperature of the wall in K
    /// @param[in] Teq - the simulation temperature in K
    /// @param[in] normal - a normal vector pointing out
    /// of the wall
    /// @param[in, out] _rng - a random number generator
    /// @param[in] theta - rotation of material with respect
    /// to x-axis
    Isothermal_wall_distribution(
        const Gamma_grid& grid,
        double Twall,
        double Teq,
        const Eigen::Ref<const Eigen::Vector3d>& normal,
        pcg64& _rng,
        double theta = 0.);

    /// Get number of generated particles for given conditions
    ///
    ///@param[in] time - in ps
    ///@param[in] spf  - area_boundary/unitcell_volume
    ///@param[in] Eff  - deviational particle energy
    std::size_t Ntogenerate(const double time,
                            const double spf,
                            const double Eff);

    /// Get the flux:
    ///@param[in] vuc - unitcell_volume
    double get_flux(const double vuc);
};


/// Objects of this class allow us to sample from a discrete
/// distribution over a q-point grid with a temperature
/// different from that of reference
class outTref_distribution : public Grid_distribution {
public:
    /// Empty constructor
    outTref_distribution() = default;
    /// Constructor.
    ///
    /// @param[in] grid - phonon spectrum on a regular grid
    /// @param[in] Tint - the temperature in K
    /// @param[in] Tref - reference temperature
    /// @param[in, out] _rng - a random number generator
    outTref_distribution(const Gamma_grid& grid,
                         double Tinit,
                         double Tref,
                         pcg64& _rng);

    /// Get number of generated particles for given conditions
    ///
    ///@param[in] vol  - box_volume/unitcell_volume
    ///@param[in] Eff  - deviational particle energy
    std::size_t Ntogenerate(const double vol, const double Eff);

    /// Returns energy introduced by the generator
    ///@param[in] vol  - box_volume/unitcell_volume
    double get_energy(const double vol);
};


/// planar_source_distribution
/// Emission probability for outgoing modes is proportional to
/// heat capacity * normal velocity.
class planar_source_distribution : public Grid_distribution {
public:
    /// Constructor.
    ///
    /// @param[in] grid - phonon spectrum on a regular grid
    /// @param[in] Tref - temperature in K at which to compute heat capacities
    /// @param[in] normal - a normal vector pointing out of the source
    /// @param[in, out] _rng - a random number generator
    planar_source_distribution(const Gamma_grid& grid,
                               double Tref,
                               const Eigen::Ref<const Eigen::Vector3d>& normal,
                               pcg64& _rng);
};


/// For layered systems, this class
/// defines the layers of each state
/// and use them to calculate the coupling
/// of two states in two stacks by using localization
/// distribution and Jensen-Shanon divergence

class layer_coupling {
private:
    /// Jensen shanon diveregence plus -1
    /// the function is defined for both STD maps
    /// and STD vectors
    inline double JSDm1(std::map<std::string, double>& P,
                        std::map<std::string, double>& Q) const {
        double PM = 0., QM = 0.;


        for (auto& [layer_id, p] : P) {
            double q = Q.at(layer_id);

            double M = 0.5 * (p + q);

            if (alma::almost_equal(M, 0.)) {
                continue;
            }

            if (!alma::almost_equal(q, 0.)) {
                QM += q * std::log2(q / M);
            }

            if (!alma::almost_equal(p, 0.)) {
                PM += p * std::log2(p / M);
            }
        }

        return std::max({1.0 - (0.5 * QM + 0.5 * PM), 0.});
    }


    inline double JSDm1(std::vector<double>& P, std::vector<double>& Q) const {
        double PM = 0., QM = 0.;


        for (std::size_t ii = 0; ii < P.size(); ii++) {
            double p = P[ii];
            double q = Q[ii];

            double M = 0.5 * (p + q);

            if (alma::almost_equal(M, 0.)) {
                continue;
            }

            if (!alma::almost_equal(q, 0.)) {
                QM += q * std::log2(q / M);
            }

            if (!alma::almost_equal(p, 0.)) {
                PM += p * std::log2(p / M);
            }
        }

        return std::max({1.0 - (0.5 * QM + 0.5 * PM), 0.});
    }


public:
    /// To store for each material the layers
    std::map<std::string, std::map<std::string, std::vector<int>>> layers;

    std::map<std::string, std::string> connection;
    bool stack_injection = false;

    /// Default constructor
    layer_coupling() = default;

    /// Calculates the coupling rate between two states
    double get_injection_rate(Eigen::VectorXcd& wfin_,
                              Eigen::VectorXcd& wfout_,
                              std::string material_in,
                              std::string material_out) const;
};

/// Diffuse mismatch distribution

class Diffuse_mismatch_distribution {
private:
    /// Number of branches, indexed via 'A' and 'B'.
    std::map<char, std::size_t> Nbranches;
    /// Number of phonon modes, indexed via 'A' and 'B'.
    std::map<char, std::size_t> Ntot;
    /// Unit cell volumes, indexed via 'A' and 'B'.
    std::map<char, double> volume;

    /// Material names:
    std::string material_A, material_B;

    /// Tuple describing the origin of each available mode in A or B and its
    /// contribution to the DOS.
    /// The first element is the material side
    /// The second element is a branch index
    /// The third element is a q-point index
    /// The fourth element is the projection of its velocity on the normal to
    /// the surface. The fifth element is the heat capacity The sixth element is
    /// a contribution to the DOS
    using dos_tuple = std::
        tuple<char, std::size_t, std::size_t, double, double, Gaussian_for_DOS>;
    /// Short notation for tuple specifying the material side, mode index, and
    /// cumulative probability
    typedef std::tuple<char, std::size_t, double> lookup_entry;
    /// Lookup table for incident A modes
    std::vector<std::vector<lookup_entry>> lookup_incidentA;
    /// Lookup table for incident B modes
    std::vector<std::vector<lookup_entry>> lookup_incidentB;
    /// Random number generator.
    pcg64 rng;
    /// Reference temperature (for computing heat capacities)
    double Tref;
    /// Gather information about all available phonon modes.
    ///
    /// @param[in] gridA - phonon spectrum of material A
    /// @param[in] gridB - phonon spectrum of material B
    /// @param[in] normal - a normal vector pointing from A to B.
    /// @param[in] scalebroad - factor modulating all the broadenings
    /// @return a vector of dos_tuples describing all modes
    std::vector<dos_tuple> get_modes(
        const Gamma_grid& gridA,
        const Gamma_grid& gridB,
        const Eigen::Ref<const Eigen::Vector3d>& normal,
        double scalebroad);

public:
    /// Empty Constructor.
    ///
    ///
    Diffuse_mismatch_distribution() = default;
    ///
    ///
    /// Constructor.
    ///
    /// @param[in] gridA - phonon spectrum of material A
    /// @param[in] poscarA - crystal structure of material A
    /// @param[in] gridB - phonon spectrum of material B
    /// @param[in] poscarB - crystal structure of material B
    /// @param[in] normal - a normal vector pointing from A to B.
    /// @param[in] scalebroad - factor modulating all the broadenings
    /// @param[in, out] _rng - a random number generator
    /// @param[in] Tref - reference temperature [K]
    /// @param[in] thicknessA - material A thickness for 2D simulations [nm]
    /// @param[in] thicknessB - material B thickness for 2D simulations [nm]
    /// @param[in] coupling - coupling between layers
    Diffuse_mismatch_distribution(
        const Gamma_grid& gridA,
        const Crystal_structure& poscarA,
        const Gamma_grid& gridB,
        const Crystal_structure& poscarB,
        const Eigen::Ref<const Eigen::Vector3d>& normal,
        double scalebroad,
        pcg64& _rng,
        double Tref = 300.0,
        double thicknessA = -1.,
        double thicknessB = -1.,
        layer_coupling coupling = layer_coupling(),
        std::string materialA = "None",
        std::string materialB = "None");

    /// Draw a final state for a particle incident from A or B.
    ///
    /// @param[in] incidence - 'A' or 'B'
    /// @param[in, out] particle - information about the particle, before and
    /// after the interaction.
    /// @param[in] account_for_velocity: if true, emission probability is
    /// proportional to abs(projected velocity).
    /// @return 'A' or 'B', depending on the direction of emission
    char reemit(char incidence, D_particle& particle);
};

/// Objects of this class allow us to simulate completely diffusive
/// interfaces between two media. An incident phonon undergoes
/// elastic diffusion and exits the interface in a mode chosen at
/// random.
///
/// In contrast with Diffuse_mismatch_distribution, in this case
/// the normal vector is not fixed at construction time.
class Elastic_interface_distribution {
private:
    /// Number of q points in grid A.
    std::size_t nqA;
    /// Number of q points in grid B.
    std::size_t nqB;
    /// Number of branches in grid A.
    std::size_t nmodesA;
    /// Number of branches in grid B.
    std::size_t nmodesB;
    /// Volume of the unit cell in A.
    double VA;
    /// Volume of the unit cell in B.
    double VB;
    /// Tuple describing the origin of each available mode
    /// and its contribution to the DOS.
    /// The first element is either 'A' or 'B'
    /// The second element is a branch index
    /// The third element is a q-point index
    /// The fourth element is a contribution to the DOS
    using dos_tuple =
        std::tuple<char, std::size_t, std::size_t, Gaussian_for_DOS>;
    /// Mode-to-mode cumulative transition probabilities.
    /// Only allowed transitions are considered.
    std::vector<std::vector<double>> cumulative;
    /// Modes to which the elements of cumulative refer.
    std::vector<std::vector<std::size_t>> allowed;
    /// Unit vectors parallel to each of the group velocities
    /// from the original grid. Each column is a 3-vector.
    Eigen::MatrixXd directions;
    /// Random number generator.
    pcg64& rng;
    /// Gather information about all available modes.
    ///
    /// @param[in] gridA - phonon spectrum of material A
    /// @param[in] gridB - phonon spectrum of material B
    /// @param[in] scalebroad - factor modulating all
    /// the broadenings
    /// @return a vector of tuples describing all modes
    std::vector<dos_tuple> get_all_modes(const Gamma_grid& gridA,
                                         const Gamma_grid& gridB,
                                         double scalebroad) const;

public:
    /// Constructor.
    ///
    /// @param[in] gridA - phonon spectrum of material A
    /// @param[in] gridB - phonon spectrum of material B
    /// @param[in] scalebroad - factor modulating all
    /// the broadenings
    /// @param[in, out] _rng - a random number generator
    Elastic_interface_distribution(const Gamma_grid& gridA,
                                   const Gamma_grid& gridB,
                                   double scalebroad,
                                   pcg64& _rng);
    /// Draw a final state for a particle incident from A or B or
    /// viceversa.
    ///
    /// @param[in] incidence - 'A' or 'B'
    /// @param[in] normal - normal to the interface, pointing
    /// from A to B.
    /// @param[in, out] particle - information about the particle,
    /// before and after the interaction.
    /// @return 'A' or 'B', depending on the direction of emission
    char reemit(char incidence,
                const Eigen::Ref<const Eigen::Vector3d>& normal,
                D_particle& particle);
};

// Simple and general class that allows us to simulate completely
// random elastic scattering.
class Elastic_distribution {
private:
    /// Number of q points in the grid.
    std::size_t nq;
    /// Number of branches in the grid..
    std::size_t nmodes;
    /// Tuple describing the origin of each available mode
    /// and its contribution to the DOS.
    /// The first element is a branch index
    /// The second element is a q-point index
    /// The third element is a contribution to the DOS
    using dos_tuple = std::tuple<std::size_t, std::size_t, Gaussian_for_DOS>;
    /// Mode-to-mode cumulative transition probabilities.
    /// Only allowed transitions are considered.
    std::vector<std::vector<double>> cumulative;
    /// Modes to which the elements of cumulative refer.
    std::vector<std::vector<std::size_t>> allowed;
    /// Unit vectors parallel to each of the group velocities
    /// from the original grid. Each column is a 3-vector.
    Eigen::MatrixXd directions;
    /// Random number generator.
    pcg64& rng;
    /// Gather information about all available modes.
    ///
    /// @param[in] grid - description of the phonon spectrum
    /// @param[in] scalebroad - factor modulating all
    /// the broadenings
    /// @return a vector of tuples describing all modes
    std::vector<dos_tuple> get_all_modes(const Gamma_grid& grid,
                                         double scalebroad) const;

public:
    /// Constructor.
    ///
    /// @param[in] grid - phonon spectrum of the material
    /// @param[in] scalebroad - factor modulating all
    /// the broadenings
    /// @param[in, out] _rng - a random number generator
    Elastic_distribution(const Gamma_grid& grid,
                         double scalebroad,
                         pcg64& _rng);
    /// Draw a final state for a particle.
    ///
    /// @param[in, out] particle - information about the particle,
    /// before and after the interaction.
    void scatter(D_particle& particle);

    /// Draw a final state for a particule with the constraint that
    /// one component of its group velocity has a predefined sign.
    ///
    /// @param[in] normal - direction of the projection
    /// @param[in] refsign - an integer with the desired sign
    /// @param[in, out] particle - information about the particle,
    /// before and after the interaction.
    void scatter(const Eigen::Ref<const Eigen::Vector3d>& normal,
                 const int refsign,
                 D_particle& particle);
};
} // namespace alma
