// Copyright 2015-2020 The ALMA Project Developers
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
/// General MC for 2d materials beyond RTA
/// The function definitions are also located in this file

#include <iostream>
#include <fstream>
#include <structures.hpp>
#include <boost/filesystem.hpp>
#include <boost/mpi.hpp>
#include <Eigen/Dense>
#include <collision_operator.hpp>
#include <sstream>
#include <sampling.hpp>
#include <constants.hpp>
#include <deviational_particle.hpp>
#include <sampling.hpp>
#include <exceptions.hpp>
#include <functional>
#include <unordered_map>
#include <map>

#include <boost/functional/hash.hpp>
#include <boost/format.hpp>
#include <geometry_2d.hpp>
#include <bulk_properties.hpp>

#include <io_utils.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/math/distributions/gamma.hpp>
#include <boost/property_tree/json_parser.hpp>


#define TBB_PREVIEW_GLOBAL_CONTROL true
#include <tbb/concurrent_vector.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <tbb/parallel_sort.h>
#include <tbb/enumerable_thread_specific.h>
#include <mutex>

/// Probability density function for a Gamma distribution.
///
/// @param[in] k - shape parameter
/// @param[in] theta - scale parameter
/// @param[in] x - point at which to evaluate the function
/// @return the value of the pdf
double gamma_pdf(double k, double theta, double x) {
    return boost::math::gamma_p_derivative(k, x / theta) / theta;
}


/// Key for DMM identification:

using DMM_key = std::tuple<std::string, std::string, Eigen::Vector3d>;


/// Some Specialization of std needed for DMM_key
namespace std {
/// Trivial implementation of std::hash for DMM_key
template <> struct hash<DMM_key> {
    std::size_t operator()(const DMM_key& key) const {
        hash<std::string> backend_str;
        hash<double> backend_double;

        std::size_t nruter = 0;

        boost::hash_combine(nruter, backend_str(std::get<0>(key)));
        boost::hash_combine(nruter, backend_str(std::get<1>(key)));

        auto vec3d = std::get<2>(key);
        boost::hash_combine(nruter, backend_double(vec3d(0)));
        boost::hash_combine(nruter, backend_double(vec3d(1)));
        boost::hash_combine(nruter, backend_double(vec3d(2)));

        return nruter;
    }
};
/// Implementation of std::equal_to for DMM
template <> struct equal_to<DMM_key> {
    bool operator()(const DMM_key& A, const DMM_key& B) const {
        bool mat_1 = (std::get<0>(A) == std::get<0>(B));
        bool mat_2 = (std::get<1>(A) == std::get<1>(B));
        /// Vectors compariso
        auto va = std::get<2>(A);
        auto vb = std::get<2>(B);
        bool veq = alma::almost_equal((va - vb).norm(), 0.);

        return (mat_1 and mat_2 and veq);
    }
};
}; // namespace std

/// Some alias
using gridData =
    std::unordered_map<std::string, std::unique_ptr<alma::Gamma_grid>>;
using cellData =
    std::unordered_map<std::string, std::unique_ptr<alma::Crystal_structure>>;

using DMM = std::unordered_map<DMM_key, alma::Diffuse_mismatch_distribution>;


/// Specialization of STD
namespace std {
/// Trivial implementation of std::hash for arrays,
/// required to create an unordered_set of arrays.
template <typename T> struct hash<std::array<T, 2>> {
    std::size_t operator()(const array<T, 2>& key) const {
        hash<T> backend;
        std::size_t nruter = 0;

        for (auto& e : key)
            boost::hash_combine(nruter, backend(e));
        return nruter;
    }
};

} // namespace std

/// Helper function that determines in which bin of a grid a particle is
/// situated. The result is clipped to [0, grid.size() - 2], even if the
/// particle happens to be out of the grid.
///
/// @param[in] grid - sorted grid of values defining the bins
/// @param[in] target - value (time or position) from the trajectory
/// @return the index of the bin where the particle is
inline Eigen::Index get_bin_index(const Eigen::Ref<const Eigen::VectorXd>& grid,
                                  double target) {
    auto first = grid.data();
    auto last = grid.data() + grid.size();
    auto nruter = static_cast<decltype(grid.size())>(
                      std::lower_bound(first, last, target) - first) -
                  1;
    return std::clamp(nruter, static_cast<Eigen::Index>(0), grid.size() - 2);
}

struct spectral_decomposition {
    /// Size of spectral decomposition
    std::size_t nomega = 0;
    std::unordered_map<std::size_t, Eigen::ArrayXXd> gz_omega;
    std::unordered_map<std::size_t, Eigen::ArrayXXd> gzjx_omega;
    std::unordered_map<std::size_t, Eigen::ArrayXXd> gzjy_omega;
    std::unordered_map<std::size_t, Eigen::ArrayXXd> fd_qmesh;
    /// Spectral grid per material
    std::unordered_map<std::size_t, Eigen::ArrayXd> omegagrid;
    /// Broadening widths (with outliers removed) for all materials in the grid.
    std::unordered_map<std::size_t, Eigen::ArrayXXd> broadening_sigmas;
    /// Information about qpoints in 1st BZ
    std::unordered_map<std::size_t, Eigen::ArrayXXd> qpoints1stBZ;


    void clean() {
        for (auto& [loc, g] : this->gz_omega)
            g.setZero();
        for (auto& [loc, g] : this->gzjx_omega)
            g.setZero();
        for (auto& [loc, g] : this->gzjy_omega)
            g.setZero();
        for (auto& [loc, g] : this->fd_qmesh)
            g.setZero();
    }
};


/// Parameters of input file
struct input_parameters {
    /// Geometry
    std::vector<alma::geometry_2d> system;

    /// Time
    double dt, maxtime;
    /// Material data
    cellData system_cell;
    gridData system_grid;
    // VARIABLES USED WHILE RUNNING SIMULATION
    // Volumetric heat capacity for each material
    std::map<std::string, double> material_cv;
    // RTA scattering rates for all modes of all materials
    std::map<std::string, Eigen::ArrayXXd> material_w0;
    /// Sampler for new phonons generation
    std::map<std::string, alma::BE_derivative_distribution> material_sampler;
    /// Reference temperature
    double T0 = -1;
    /// Temperature gradient applied to system [K/nm]
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();


    /// Number of deviational energy particles
    std::size_t Nparticles;
    /// Vector of thickness
    std::vector<double> thicknesses;

    /// Stacking stuff
    alma::layer_coupling couplings;

    /// This is for gradient simulations
    double eps_energy = -42.0;
    double eps_flux = -42.0;

    /// Spectral decomposition
    spectral_decomposition sd;

    /// Only ballistic
    bool ballistic = false;
};

/// Compute the deviational power per unit area at an isothermal
/// surface.
///
/// @param[in] poscar - a description of the unit cell
/// @param[in] grid - phonon spectrum on a regular grid
/// @param[in] Twall - the temperature of the wall in K
/// @param[in] Tref - the simulation temperature in K
/// @param[in] normal - a normal vector pointing out
/// of the wall [inside material]
/// @return  the deviational power per unit area in units of
/// J / ps / nm ** 2
inline double calc_dotEeff_wall(
    const alma::Crystal_structure& poscar,
    const alma::Gamma_grid& grid,
    double Twall,
    double Tref,
    const Eigen::Ref<const Eigen::Vector3d>& normal) {
    if (normal.norm() == 0.) {
        throw alma::value_error("invalid normal vector");
    }
    Eigen::Vector3d u(normal.array() / normal.norm());
    double nruter = 0.;
    double nqpoints = grid.nqpoints;
    double nmodes = grid.get_spectrum_at_q(0).omega.size();

    for (std::size_t iq = 0; iq < nqpoints; ++iq) {
        auto& spectrum = grid.get_spectrum_at_q(iq);

        for (std::size_t im = 0; im < nmodes; ++im) {
            double vn = u.dot(spectrum.vg.col(im).matrix());

            if (vn >= 0.) {
                nruter +=
                    alma::bose_einstein_kernel(spectrum.omega(im), Tref) * vn;
            }
        }
    }
    nruter *= std::abs((Twall - Tref) / nqpoints);
    return alma::constants::kB * nruter;
}


namespace generators {

/// It contains all generators
///@param[in] sys   - system boxes
///@param[in] grids - grid data for all materials
///@param[in] cells - cell data for all materials
///@param[in] rng   - random number generator
///@param[in] thickness - thickness of all boxes
///@param[in,out] einfo - contains absolute flux and energy introduced to system
///per ps
alma::D_particle Init_generator(std::vector<alma::geometry_2d>& sys,
                                gridData& grids,
                                cellData& cells,
                                pcg64& rng,
                                std::vector<double>& thickness,
                                Eigen::Vector3d& gradient,
                                Eigen::Vector2d& einfo) {
    /// Generators
    static std::unordered_map<
        std::array<std::size_t, 2>,
        std::pair<alma::Isothermal_wall_distribution, alma::geom2d_border>>
        isogens;
    static std::unordered_map<std::size_t, alma::Nabla_T_distribution>
        grad_generators;

    static std::map<double, std::array<std::size_t, 2>> weight_gen;
    static std::map<double, std::size_t> wgrad_gen, winit_gen;

    static alma::particle_sign gsign = alma::get_particle_sign(-1);

    /// Energy per generator
    static std::array<double, 2> ExGen = {0., 0.};

    einfo.setZero();


    if (isogens.size() == 0 and grad_generators.size() == 0) {
        /// Iterate through system to build up isothermal generators
        for (auto& s : sys) {
            /// If not reservoir ignore
            if (!s.reservoir)
                continue;
            /// Retrive material and box id
            auto smat = s.material;
            auto sid = s.get_id();

            /// Iterate through contacts:
            /// This is done each timestep
            /// only in case we need to recalculate
            /// the generator
            for (auto& [c, b] : s.get_contacts()) {
                if (sys[c].reservoir or sys[c].periodic)
                    continue;

                if (alma::almost_equal(b[0].get_length(), 0.))
                    continue;

                if (sys[c].material != smat) {
                    throw alma::input_error("Injection from material A to B "
                                            "is not allowed\n");
                }

                /// Ignore if Temperatures are equal
                if (alma::almost_equal(s.Teq, sys[c].Teq))
                    continue;
                std::array<std::size_t, 2> ids = {sid, c};
                Eigen::Vector3d nw;
                nw << b[0].np(0), b[0].np(1), 0;

                alma::Isothermal_wall_distribution igen(
                    *(grids.at(smat)), s.Teq, sys[c].Teq, nw, rng);

                // double ed   = calc_dotEeff_wall(
                //            *(cells.at(smat)),
                //            *(grids.at(smat)),
                //            s.Teq,
                //            sys[c].Teq,
                //            nw);
                double vuc = cells.at(s.material)->V * thickness[s.get_id()] /
                             cells.at(s.material)->lattvec(2, 2);

                double ed = igen.get_flux(vuc);

                einfo(0) += ed / vuc;

                isogens[ids] = std::make_pair(igen, b[0]);
            } // contacts loop
        }

        // Iterate through system to build up initial condition generators
        for (auto& s : sys) {
            /// If reservoir or periodic ignore
            if (s.reservoir or s.periodic)
                continue;

            /// If no gradient applied to system
            if (alma::almost_equal(gradient.norm(), 0.))
                continue;

            auto mat = s.material;


            alma::Nabla_T_distribution myGradGen(
                *(grids[mat]), gradient, s.Teq, rng);

            grad_generators[s.get_id()] = myGradGen;
        }

    /// We now iterate through isothermal generators to get the weights
    /// to know which source the particle will be generated from
    {
        std::vector<double> ws, weight_vec;
        for (auto& [ids, G] : isogens) {
            auto idr = ids[0];
            auto mat = sys[idr].material;
            auto b = G.second;
            auto Lb = b.get_length();
            double vuc = cells.at(mat)->V * thickness[idr] /
                         cells.at(mat)->lattvec(2, 2);

            Eigen::Vector3d nw;
            nw << b.np(0), b.np(1), 0;
            double jd = G.first.get_flux(vuc);

            double e_inserted = std::abs(jd) * Lb * thickness[idr];
            /// This gives the energy inserted per ps in system (not care about
            /// sign) per unit
            einfo(1) += e_inserted;
            ExGen[0] += e_inserted;

            ws.push_back(e_inserted);
        }

        double wnorm = std::accumulate(ws.begin(), ws.end(), 0.);
        double cum = 0.;
        for (auto w : ws) {
            cum += w / wnorm;
            weight_vec.push_back(cum);
        }

        std::size_t ii = 0;
        for (auto& [ids, G] : isogens) {
            weight_gen[weight_vec[ii]] = ids;
            ii++;
        }
    }

    {
        std::vector<double> ws, weight_vec;
        for (auto& [id, G] : grad_generators) {
            auto& s = sys[id];

            auto Vucell = cells.at(s.material)->V * thickness[s.get_id()] /
                          cells.at(s.material)->lattvec(2, 2);

            auto Vbox = s.get_area() * thickness[s.get_id()];

            auto eintro = G.get_energy(Vbox / Vucell);

            ExGen[1] += eintro;

            ws.push_back(eintro);
        }

        double wnorm = std::accumulate(ws.begin(), ws.end(), 0.);
        double cum = 0.;
        for (auto w : ws) {
            cum += w / wnorm;
            weight_vec.push_back(cum);
        }

        std::size_t ii = 0;
        for (auto& [id, G] : grad_generators) {
            wgrad_gen[weight_vec[ii]] = id;
            ii++;
        }
    }

        /// Building array to generate
        double total_exgen = std::accumulate(ExGen.begin(), ExGen.end(), 0.);

        einfo(1) = total_exgen;

        ExGen[0] = ExGen[0] / total_exgen;
        ExGen[1] = ExGen[1] / total_exgen + ExGen[0];
        std::cout << "#Distribution by generators => " << ExGen[0] << '\t'
                  << ExGen[1] << std::endl;
    }

    double selectGenR = std::uniform_real_distribution(0., 1.)(rng);

    auto genID =
        std::distance(ExGen.begin(),
                      std::lower_bound(ExGen.begin(), ExGen.end(), selectGenR));

    switch (genID) {
        case 0: {
            /// Selecting the generator
            double whereGenR = std::uniform_real_distribution(0., 1.)(rng);
            std::array<std::size_t, 2> ids =
                weight_gen.lower_bound(whereGenR)->second;
            auto& G = isogens[ids];
            auto idr = ids[0];
            auto idi = ids[1];
            auto mat = sys[idr].material;
            auto b = G.second;
            auto& gen = G.first;

            Eigen::Vector2d ppos = b.get_random_point(rng);

            // Sanity check
            // if that fails is due to
            // numerical errors:
            if (!sys[idi].inside(ppos)) {
                decltype(ppos) pos0 = ppos;
                decltype(ppos) d = sys[idi].get_center() - ppos;
                ppos += 1.0e-6 * d / d.norm();
                if (!sys[idi].inside(ppos)) {
                    std::cout << "Error report" << std::endl;
                    std::cout << "reservoir id: " << idr << std::endl;
                    std::cout << "orig pos:\n" << pos0 << std::endl;
                    std::cout << "correction\n" << d << std::endl;
                    std::cout << "corpos:\n" << ppos << std::endl;
                    std::cout << "id inj box: " << idi << std::endl;
                    std::cout << "center inj box\n"
                              << sys[idi].get_center() << std::endl;
                    throw alma::geometry_error("Generated particle "
                                               "out of insertion box\n");
                }
            }

            auto pinfo = gen.sample_with_sign();
            alma::D_particle newp(ppos,
                                  std::get<1>(pinfo),
                                  std::get<0>(pinfo),
                                  std::get<2>(pinfo),
                                  0.,
                                  idi);

            return newp;
        }
        case 1: {
            double whereGenR = std::uniform_real_distribution(0., 1.)(rng);
            auto id = wgrad_gen.lower_bound(whereGenR)->second;
            auto& G = grad_generators[id];

            Eigen::Vector2d ppos = sys[id].get_random_point(rng);

            auto pinfo = G.sample_with_sign();
            while (std::get<2>(pinfo) != gsign) {
                pinfo = G.sample_with_sign();
            }
            gsign = alma::get_particle_sign(
                -1 * static_cast<int>(std::get<2>(pinfo)));
            alma::D_particle newp(ppos,
                                  std::get<1>(pinfo),
                                  std::get<0>(pinfo),
                                  std::get<2>(pinfo),
                                  0.,
                                  id);

            return newp;
        }

        default: {
            throw alma::value_error("Error selecting generator");
        }
    }

    throw alma::value_error("Error selecting generator");
    return alma::D_particle();
}


/// This class is to make the diffusive
/// scattering which is approximated by the Lambert
/// cosine law with the energy conservation as
/// constraint.
/// The probability of going for i to f is the defined as:
///$P_{i \rightarrow f} = \frac{v_f \cdot
///\hat{e}_{\perp}\delta(\omega_f-\omega_i)}{\sum_j v_j \cdot
///\hat{e}_{\perp}\delta(\omega_j-\omega_i)}$ This energy "conservation" is
/// treated in the same way that for three phonon processes, see section 2.4 of
/// https://doi.org/10.1016/j.cpc.2014.02.015 for reference. Code is partially
/// adapted from isotopic scattering algorithms.
class diffusive_bouncing {
public:
    /// Table with energy conservation from state to state
    // for each material
    std::map<std::string, std::vector<std::unordered_map<std::size_t, double>>>
        lambert;

    /// It creates a table in the full BZ for diffusive bouncing
    //@param[in] gridData - map containing qgrid data for each material
    diffusive_bouncing(gridData& d) {
        for (auto& [m, info] : d) {
            auto nq = info->nqpoints;
            auto nb = static_cast<std::size_t>(
                info->get_spectrum_at_q(0).omega.size());
            // Precompute the broadening of all modes.
            Eigen::ArrayXXd sigmas(nb, nq);

            lambert[m] =
                std::vector<std::unordered_map<std::size_t, double>>(nq * nb);

            for (std::size_t iq = 0; iq < nq; ++iq)
                for (std::size_t im = 0; im < nb; ++im) {
                    auto spectrum = info->get_spectrum_at_q(iq);
                    sigmas(im, iq) = info->base_sigma(spectrum.vg.col(im));
                }
            // And refine them by removing outliers.
            auto percent = alma::calc_percentiles_log(sigmas);
            double lbound =
                std::exp(percent[0] - 1.5 * (percent[1] - percent[0]));
            sigmas = (sigmas < lbound).select(lbound, sigmas);

            for (std::size_t ic = 0; ic < info->get_nequivalences(); ++ic) {
                /// Gamma phonons cannot couple
                if (ic == 0)
                    continue;
                auto iq1 = info->get_representative(ic);
                auto spectrum1 = info->get_spectrum_at_q(iq1);
                /// Gamma phonons cannot couple (we start from 1)
                for (std::size_t iq2 = 1; iq2 < info->nqpoints; ++iq2) {
                    auto spectrum2 = info->get_spectrum_at_q(iq2);

                    for (decltype(nb) im1 = 0; im1 < nb; ++im1) {
                        for (decltype(nb) im2 = 0; im2 < nb; ++im2) {
                            if (alma::almost_equal(0., spectrum1.omega(im1)) or
                                alma::almost_equal(0., spectrum2.omega(im2)))
                                continue;
                            // Note that, since momentum is not conserved,
                            // the frequencies are uncorrelated random
                            // variables. The total standard deviation is
                            // the quadratic sum of both standard
                            // deviations.
                            auto sigma =
                                std::hypot(sigmas(im1, iq1), sigmas(im2, iq2));
                            // sigma      = 0.1;
                            // The only criterion for accepting a process as
                            // allowed is conservation of energy after
                            // taking into account the discretization of
                            // reciprocal space with the usual adaptive
                            // broadening method.
                            auto delta = std::fabs(spectrum1.omega(im1) -
                                                   spectrum2.omega(im2));

                            if (delta <= alma::constants::nsigma * sigma) {
                                auto distr = boost::math::normal(0., sigma);
                                /// Recovering full BZ using symmetry operations
                                for (auto qp :
                                     info->equivalent_qpairs({iq1, iq2})) {
                                    auto mode1 = nb * qp[0] + im1;
                                    auto mode2 = nb * qp[1] + im2;
                                    this->lambert[m][mode1][mode2] =
                                        boost::math::pdf(distr, delta);
                                }
                            }
                        }
                    }
                }
            }
        }
        // Ended for
    }

    /// It returns the new state after diffusive bouncing
    ///@param[in] npout - the vector of the wall pointing outside
    ///@param[in] mat   - material name
    ///@param[in] imode - phonon mode id
    ///@param[in] grid  - map with qgrid data for all materials
    ///@param[in] rng   - random number generator
    ///@returns   pair containing mode band and qpoint id
    std::pair<std::size_t, std::size_t> bounce(Eigen::Vector3d npout,
                                               std::string mat,
                                               std::size_t imode,
                                               gridData& grid,
                                               pcg64& rng) {
        auto& tm = this->lambert[mat][imode];
        auto nb = grid[mat]->get_spectrum_at_q(0).omega.size();
        /// Iterate over the energy conserving states
        /// and building histogram for random
        /// selection of final state
        std::map<std::size_t, double> states;
        std::map<double, std::size_t> states2;
        double cum = 0.;
        // for (int imf = 3; imf < grid[mat]->nqpoints*nb; imf++){
        for (auto& [imf, econ] : tm) {
            auto ib = imf % nb;
            auto iq = imf / nb;
            Eigen::Vector3d v = grid[mat]->get_spectrum_at_q(iq).vg.col(ib);

            for (const auto axis : {0, 1, 2}) {
                if (alma::almost_equal(v(axis), 0.))
                    v(axis) = 0.;
            }

            double vn = -(npout.dot(v));
            if (vn < 0. or alma::almost_equal(vn, 0.))
                continue;
            cum += vn * econ;
            states[imf] = vn * econ;
        }

        double cum2 = 0;

        for (auto& [imf, tran] : states) {
            cum2 += tran / cum;
            states2[cum2] = imf;
        }

        double s = std::uniform_real_distribution(0., 1.)(rng);

        std::size_t IMF = states2.lower_bound(s)->second;

        return std::make_pair(IMF % nb, IMF / nb);
    }
};


}; // namespace generators

/// Helper function for processing trajectories
///@param[in] ibox - r bin id
///@param[in] tstart - initial time of the trajectory
///@param[in] tstop  - end time for the trajectory
///@param[in] sign   - particle sign
///@param[in] timegrid - grid of times for register
///@param[in] vg       - particle velocity
///@param[in,out] gz - matrix where to save the contribution of trajectory
//                     to bins
///@param[in,out] gz_jx - matrix where to save the contribution to jx
///@param[in,out] gz_jy - matrix where to save the contribution to jy
///@param[in] omega - particle frequency
///@param[in] ib - particle branch
///@param[in] iq - particle qpoint id
///@param[in,out] sd - class containing data of spectral decompositions
void process_segment(std::size_t ibox,
                     double tstart,
                     double tstop,
                     alma::particle_sign& sign,
                     Eigen::VectorXd& timegrid,
                     Eigen::Vector2d& vg,
                     Eigen::ArrayXXd& gz,
                     Eigen::ArrayXXd& gz_jx,
                     Eigen::ArrayXXd& gz_jy,
                     double omega,
                     std::size_t iq,
                     std::size_t ib,
                     spectral_decomposition& sd) {
    // Compute bounds for the subgrid in (t, r) space covered by the
    // segment.
    auto startbin_t = get_bin_index(timegrid, tstart);
    auto stopbin_t = get_bin_index(timegrid, tstop);
    double delta_t = tstop - tstart;

    bool spectral = !sd.gz_omega.empty() and sd.gz_omega.count(ibox) != 0;


    Eigen::ArrayXd gamma_weights(0);
    /// Spectral decomposition stuff
    if (spectral) {
        gamma_weights.resize(sd.omegagrid[ibox].size());
        double k = omega * omega / sd.broadening_sigmas[ibox](ib, iq);
        double theta = sd.broadening_sigmas[ibox](ib, iq) / omega;

        for (Eigen::Index i = 0; i < sd.omegagrid[ibox].size(); ++i) {
            gamma_weights(i) = gamma_pdf(k, theta, sd.omegagrid[ibox](i));
        }
    }


    // Compute the time spent in each cell of the (t, z) subgrid.
    Eigen::ArrayXd lo_time{stopbin_t - startbin_t + 1};
    Eigen::ArrayXd hi_time{stopbin_t - startbin_t + 1};
    for (auto i = startbin_t; i <= stopbin_t; ++i) {
        lo_time(i - startbin_t) = timegrid(i) - tstart;
        hi_time(i - startbin_t) = timegrid(i + 1) - tstart;
    }

    /// Shared among threads
    static std::mutex protector;

    // We are only putting contributions from one single space bin
    // as function is called always when we change from box

    for (auto i_time = startbin_t; i_time <= stopbin_t; ++i_time) {
        double lo = std::max(lo_time(i_time - startbin_t), 0.);
        double hi = std::min(hi_time(i_time - startbin_t), delta_t);
        double contr = static_cast<double>(sign) * std::max(hi - lo, 0.);
        gz(ibox, i_time) += contr;
        gz_jx(ibox, i_time) += vg(0) * contr;
        gz_jy(ibox, i_time) += vg(1) * contr;

        if (spectral) {
            protector.lock();
            sd.gz_omega[ibox].row(i_time) += contr * gamma_weights;
            sd.gzjx_omega[ibox].row(i_time) += vg(0) * contr * gamma_weights;
            sd.gzjy_omega[ibox].row(i_time) += vg(1) * contr * gamma_weights;
            /// Storing distribution in q-mesh
            sd.fd_qmesh[ibox](i_time, iq) += contr / omega;
            protector.unlock();
            // std::cout << "sp->done" << std::endl;
        }
    }

    return;
}

/// Helper function for processing trajectories
///@param[in] ibox - r bin id
///@param[in] tstart - initial time of the trajectory
///@param[in] tstop  - end time for the trajectory
///@param[in] sign   - particle sign
///@param[in] timegrid - grid of times for register
///@param[in] vg       - particle velocity
///@param[in,out] gz - matrix where to save the contribution of trajectory
//                     to bins
///@param[in,out] gz_jx - matrix where to save the contribution to jx
///@param[in,out] gz_jy - matrix where to save the contribution to jy
///@param[in,out] phi   -
///@param[in]     evalphi - bool to eval phi
void process_segment(std::size_t ibox,
                     alma::particle_sign& sign,
                     double dt,
                     Eigen::Vector2d& vg,
                     Eigen::ArrayXd& gz,
                     Eigen::ArrayXd& gz_jx,
                     Eigen::ArrayXd& gz_jy,
                     Eigen::ArrayXd& phi,
                     double omega,
                     std::size_t iq,
                     std::size_t ib,
                     spectral_decomposition& sd,
                     bool evalphi = false) {
    // Compute bounds for the subgrid in (t, r) space covered by the
    // segment.
    double contr = static_cast<double>(sign);
    gz(ibox) += contr * dt;
    gz_jx(ibox) += vg(0) * contr * dt;
    gz_jy(ibox) += vg(1) * contr * dt;


    bool spectral = !sd.gz_omega.empty() and sd.gz_omega.count(ibox) != 0;

    /// Shared among threads
    static std::mutex protector;

    /// Spectral decomposition stuff
    if (spectral) {
        Eigen::ArrayXd gamma_weights{sd.omegagrid[ibox].size()};
        double k = omega * omega / sd.broadening_sigmas[ibox](ib, iq);
        double theta = sd.broadening_sigmas[ibox](ib, iq) / omega;
        for (Eigen::Index i = 0; i < sd.omegagrid[ibox].size(); ++i) {
            gamma_weights(i) = gamma_pdf(k, theta, sd.omegagrid[ibox](i));
        }

        protector.lock();
        sd.gz_omega[ibox].row(0) += gamma_weights * contr * dt;
        sd.gzjx_omega[ibox].row(0) += vg(0) * gamma_weights * contr * dt;
        sd.gzjy_omega[ibox].row(0) += vg(1) * gamma_weights * contr * dt;

        sd.fd_qmesh[ibox](0, iq) += contr / omega;

        protector.unlock();
    }

    if (evalphi) {
        phi(ibox) += contr;
    }

    return;
}


/// Get v:
/// We need to clean v values of small values
///@param[in] grid - qpoint grid
///@param[in] ib   - band index
///@param[in] iq   - qpoint index
///@return    phonon group velocity
inline Eigen::Vector2d get_v(const alma::Gamma_grid& grid,
                             std::size_t ib,
                             std::size_t iq) {
    Eigen::Vector3d v3 = grid.get_spectrum_at_q(iq).vg.col(ib);
    // return (v3.block(0,0,2,1)).eval();
    Eigen::Vector2d v2;
    v2 << v3(0), v3(1);
    for (auto i = 0; i < 2; i++) {
        if (alma::almost_equal(v2(i), 0.))
            v2(i) = 0.;
    }

    return v2;
}


/// Function controling the change of material it allows for DMM and LDMM
///@param[in]     inew      - possible box id to check transmission to 
///@param[in]     cboxes    - contact boxes
///@param[in,out] particle  - deviational particle to evolve
///@param[in]     sys       - geometry and box info
///@param[in]     dt        - time to evolve
///@param[in]     rnd       - random generator
///@param[in]     grids     - qgrid data per material
///@param[in]     DMM_gen   - DMM generators for interface scattering
///@param[in]     thickness - correct materials thickness
///@param[in]     cells     - cells data
///@param[in]     coupling  - information for LDMM model for interface
template<class Random,class Mutex>
void change_material_diffuse(std::size_t inew,
    std::set<std::size_t>& cboxes,
    alma::D_particle& particle,
    std::vector<alma::geometry_2d>& sys,
    double& dt,
    Random& rnd,
    Mutex& pmutex,
    gridData& grids,
    DMM& DMM_gen,
    std::vector<double>& thickness,
    cellData& cells,
    alma::layer_coupling& couplings) {

    /// Getting index info
    std::size_t iold = particle.boxid;

    auto matA = sys[iold].material;
    auto matB = sys[inew].material;

    /// Getting vector info
    Eigen::Vector3d nw;
    nw.setZero();
    // Get incident vector
    for (auto& [c, b] : sys[iold].get_contacts()) {
        if (c == inew) {
            Eigen::Vector3d nw_;
            nw_ << b[0].np(0), b[0].np(1), 0.;
            nw += nw_;
        }
    }

    DMM_key DMM_ID = {matA, matB, nw};

    /// If the model for that interface does not exist create it
    if (DMM_gen.count(DMM_ID) == 0) {
        bool layer_coupling = couplings.stack_injection;

        if (matA == matB) {
            throw alma::geometry_error("DMM in same material");
        }

        /// In the case of layered stacks if
        /// coupling information is provided
        /// we use LDMM to take into account
        /// the coupling of vibrations 
        /// between different layers. Otherwise
        /// we use traditional DMM
        if (layer_coupling) {
            DMM_gen[DMM_ID] =
                alma::Diffuse_mismatch_distribution(*grids[matA],
                                                    *cells[matA],
                                                    *grids[matB],
                                                    *cells[matB],
                                                    nw,
                                                    0.1,
                                                    rnd,
                                                    sys[iold].Teq,
                                                    thickness[iold],
                                                    thickness[inew],
                                                    couplings,
                                                    matA,
                                                    matB);
        }
        else {
            DMM_gen[DMM_ID] =
                alma::Diffuse_mismatch_distribution(*grids[matA],
                                                    *cells[matA],
                                                    *grids[matB],
                                                    *cells[matB],
                                                    nw,
                                                    0.1,
                                                    rnd,
                                                    sys[iold].Teq,
                                                    thickness[iold],
                                                    thickness[inew]);
        }
    }

    /// Setting incidence (we are always incident from A)
    char incidence = 'A';
    /// Scatter at interface
    auto out_incidence = DMM_gen[DMM_ID].reemit(incidence, particle);

    /// If there is transmission accross interface
    if (incidence != out_incidence) {
        particle.boxid = inew;
        if (cboxes.find(particle.boxid) == cboxes.end()) {
            throw alma::geometry_error("Teleport is not allowed");
        }
    }
    dt = 0.;
    return;
}




/// Particle evolution:
///@param[in,out] particle - deviational particle to evolve
///@param[in] v        - particle velocity
///@param[in] sys      - geometry and box info
///@param[in] dt       - time to evolve
///@param[in] rnd     - random generator
///@param[in] pmutex  - mutex
///@param[in] db      - diffusive boundary scattering table
///@param[in] grids   - qgrid data per material
///@param[in] timegrid - time mesh
///@param[in,out] gz   - matrix to save particles contributions to bins
///@param[in,out] gz_jx - matrix where to save the contribution to jx
///@param[in,out] gz_jy - matrix where to save the contribution to jy
///@param[in]     DMM_gen - DMM generators for interface scattering
///@param[in]     thickness - correct materials thickness
///@param[in]     cells     - cells data
///@param[in]     material_sampler - postscacttering distribution for sampling
///after scattering
///@param[in]     sd - information about spectral decompostions
///@param[in]     coupling - information for LDMM model for interface
///@param[in]     ballistic - to turn on/off intrinsic scattering
template <class Random, class Mutex>
void evolparticle(
    alma::D_particle& particle,
    Eigen::Vector2d& v,
    std::vector<alma::geometry_2d>& sys,
    double& dt,
    Random& rnd,
    Mutex& pmutex,
    generators::diffusive_bouncing& db,
    gridData& grids,
    Eigen::VectorXd& timegrid,
    Eigen::ArrayXXd& gz,
    Eigen::ArrayXXd& gz_jx,
    Eigen::ArrayXXd& gz_jy,
    DMM& DMM_gen,
    std::vector<double>& thickness,
    cellData& cells,
    std::map<std::string, alma::BE_derivative_distribution>& material_sampler,
    spectral_decomposition& sd,
    alma::layer_coupling& couplings,
    bool ballistic) {
    Eigen::Vector2d opos(particle.pos);

    auto pq = particle.q;
    auto pb = particle.alpha;
    double mu =
        grids[sys[particle.boxid].material]->get_spectrum_at_q(pq).omega(pb);

    // std::cout << "# " << particle.t << '\t' <<
    // particle.boxid <<'\t'<< particle.pos.transpose()
    //<< '\t' << v.transpose() << '\t' << dt << std::endl;

    /// Make some v check
    for (auto iv : {0, 1}) {
        if (alma::almost_equal(v(iv), 0.) and v(iv) != 0.) {
            throw alma::geometry_error(
                "Error, please set all that"
                " evaluates almost_equal to 0 as 0. Otherwise,"
                " geometric algorithms will fail\n"
                "Specifically the rnew check as it contains "
                "v in spatial search of boxes to go\n");
        }
    }

    /// Apply PBC
    if (sys[particle.boxid].periodic) {
        pmutex.lock();
        auto PBCsol = sys[particle.boxid].translate(opos, v, sys, rnd);
        pmutex.unlock();

        particle.boxid = PBCsol.first;
        particle.pos = PBCsol.second;
        return;
    }


    /// Check if in reservoir
    if (sys[particle.boxid].reservoir) {
        /// Terminated particle
        particle.q = 0;
        particle.alpha = 0;
        dt = 0.;
        return;
    }

    Eigen::Vector2d rnew = particle.pos + dt * v;
    // particle.pos(0) = 25;

    /// Check if inside the same box:

    if (sys[particle.boxid].inside(rnew)) {
        if (sys[particle.boxid].periodic) {
            std::cout << "Periodic boxes do not exist\n";
            std::cout << "r0\n" << opos << std::endl;
            std::cout << "rf\n" << rnew << std::endl;
            std::cout << "dt " << dt << std::endl;
            std::cout << particle.pos << std::endl;
            std::cout << "v\n" << v << std::endl;
            throw alma::geometry_error("Peridic error\n");
        }

        particle.pos = rnew;


        /// Make here the contribution to the segment
        process_segment(particle.boxid,
                        particle.t,
                        particle.t + dt,
                        particle.sign,
                        timegrid,
                        v,
                        gz,
                        gz_jx,
                        gz_jy,
                        mu,
                        pq,
                        pb,
                        sd);
        particle.t += dt;

        if (!ballistic) {
            /// Scattering here (serial because of rng protection):
            auto mat = sys[particle.boxid].material;
            pmutex.lock();
            auto mode = material_sampler.at(mat).sample();
            pmutex.unlock();
            particle.q = mode[1];
            particle.alpha = mode[0];
        }
        dt = 0.;
        return;
    }

    /// Check the place it left
    Eigen::Vector2d rp;
    rp << particle.pos(0), particle.pos(1);

    /// Here we calculate where the particle is
    /// going out
    std::tuple<double, Eigen::Vector2d, std::vector<int>> MRU_sol;

    try {
        MRU_sol = sys[particle.boxid].get_inter_side(rp, v, dt);
    }
    catch (const alma::geometry_error& geomerror) {
        Eigen::Vector2d d = sys[particle.boxid].get_center() - opos;
        particle.pos += 1.0e-6 * d / d.norm();
        // evolparticle(particle, v, sys, dt, rnd,
        // pmutex,db,grids,timegrid,gz,gz_jx,gz_jy,DMM_gen,thickness,cells,material_sampler,sd,couplings);
        return;
    }

    if (std::get<0>(MRU_sol) < 0) {
        throw alma::geometry_error("Error in kinematics\n");
        exit(EXIT_FAILURE);
    }
    /// Calculating leftime
    double left_time = dt - std::get<0>(MRU_sol);
    double tf = std::get<0>(MRU_sol);
    if (alma::almost_equal(left_time, 0.)) {
        left_time = 0.;
    }

    dt = left_time;
    rnew = std::get<1>(MRU_sol);

    decltype(rnew) reps = rnew + 1.0e-6 * v / v.norm();

    /// Get contact boxes
    std::set<std::size_t> cboxes;
    for (auto& [ibc, border] : sys[particle.boxid].get_contacts()) {
        cboxes.insert(ibc);
    }

    /// Check possible boxes to assign:
    std::vector<std::size_t> pboxes;


    for (auto test : cboxes) {
        if (test == particle.boxid) {
            continue;
        }

        auto border_contact = sys[particle.boxid].get_contacts()[test];
        /// If going along border we do not change
        bool parallel = alma::almost_equal((border_contact[0].np).dot(v), 0.);

        bool oriented = ((border_contact[0].np).dot(v) > 0.);

        if ((sys[test].inside(rnew) or sys[test].inside(reps)) and !parallel and
            oriented) {
            pboxes.push_back(test);
        }
    }

    /// If going inside the void
    if (pboxes.size() == 0) {

        /// Make here the contribution to the segment
        process_segment(particle.boxid,
                        particle.t,
                        particle.t + tf,
                        particle.sign,
                        timegrid,
                        v,
                        gz,
                        gz_jx,
                        gz_jy,
                        mu,
                        pq,
                        pb,
                        sd);
        particle.t += tf;


        /// Correcting three corner problem
        auto corner_problem = in_corner3(rnew, sys);
        if (corner_problem.first) {
            auto csol = correct_corner_problem(
                rnew, v, sys, corner_problem.second, rnd);

            particle.boxid = csol.first;
            Eigen::Vector2d correction =
                sys[particle.boxid].get_center() - rnew;
            Eigen::Vector2d corrected_pos =
                rnew + 1.0e-6 * correction / correction.norm();

            MRU_sol =
                sys[particle.boxid].get_inter_side(corrected_pos, v, 1.0e+18);

            rnew = std::get<1>(MRU_sol);
        }

        /// Diffusive scattering
        std::size_t iborder;
        auto iborders = std::get<2>(MRU_sol);
        if (iborders.size() > 1) {
            /// We first check the borders
            /// that touch void
            std::vector<std::size_t> voidborders;

            for (auto& tvb : iborders) {
                auto& mb = sys[particle.boxid].get_border(tvb);

                Eigen::Vector2d rc = rnew + 1.0e-4 * mb.np;

                bool isvoid = true;
                for (auto& cb : cboxes) {
                    if (sys[cb].inside(rc)) {
                        isvoid = false;
                        break;
                    }
                }

                if (isvoid) {
                    voidborders.push_back(tvb);
                }
            }

            if (voidborders.empty()) {
                throw alma::geometry_error("Error in void borders\n");
            }

            // In the case two borders pointing at void
            iborder = voidborders[std::uniform_int_distribution(
                0, static_cast<int>(voidborders.size()))(rnd)];
        }
        else {
            iborder = iborders[0];
        }

        auto u = sys[particle.boxid].get_border(iborder).np;

        Eigen::Vector3d unp3;
        unp3 << u(0), u(1), 0.;

        std::size_t Nb =
            static_cast<std::size_t>(grids[sys[particle.boxid].material]
                                         ->get_spectrum_at_q(0)
                                         .omega.size());
        auto newmode = db.bounce(unp3,
                                 sys[particle.boxid].material,
                                 particle.q * Nb + particle.alpha,
                                 grids,
                                 rnd);

        particle.pos = rnew;
        particle.q = newmode.second;
        particle.alpha = newmode.first;
        dt = 0.;
        return;
    }
    else if (pboxes.size() == 1) {
        /// Change material
        if (sys[particle.boxid].material != sys[pboxes[0]].material) {
            /// Save contribution here
            process_segment(particle.boxid,
                            particle.t,
                            particle.t + tf,
                            particle.sign,
                            timegrid,
                            v,
                            gz,
                            gz_jx,
                            gz_jy,
                            mu,
                            pq,
                            pb,
                            sd);

            particle.pos = rnew;
            particle.t += tf;
            change_material_diffuse(pboxes[0],cboxes,particle,sys,dt,rnd,pmutex,grids,
                                    DMM_gen,thickness,cells,couplings);
            return;
        }
        else { /// Same material
            particle.pos = rnew;


            /// Make here the contribution to the segment
            process_segment(particle.boxid,
                            particle.t,
                            particle.t + tf,
                            particle.sign,
                            timegrid,
                            v,
                            gz,
                            gz_jx,
                            gz_jy,
                            mu,
                            pq,
                            pb,
                            sd);
            particle.t += tf;
            particle.boxid = pboxes[0];
            if (cboxes.find(particle.boxid) == cboxes.end()) {
                throw alma::geometry_error("Teleport is not allowed");
            }
        }

        // std::cout << "#boxchange" << std::endl;
        // evolparticle(particle, v, sys, left_time, rnd,
        // pmutex,db,grids,timegrid,gz,gz_jx,gz_jy,DMM_gen,thickness,cells,material_sampler,sd,couplings);
        return;
    }
    else {
        // Make here the contribution to the segment
        process_segment(particle.boxid,
                        particle.t,
                        particle.t + tf,
                        particle.sign,
                        timegrid,
                        v,
                        gz,
                        gz_jx,
                        gz_jy,
                        mu,
                        pq,
                        pb,
                        sd);

        particle.t += tf;
        particle.pos = rnew;


        std::vector<std::size_t> pbox2;

        Eigen::Vector2d rcheck = rnew + 1.0e-6 * v;

        for (auto& PB : pboxes) {
            if (sys[PB].inside(rcheck)) {
                pbox2.push_back(PB);
            }
        }

        std::size_t newid;
        if (pbox2.empty()) {
            newid = *(alma::choose(pboxes.begin(), pboxes.end(), rnd));

            Eigen::Vector2d d = sys[newid].get_center() - rnew;
            if (!sys[newid].periodic)
                particle.pos += 1.0e-6 * d / d.norm();

            if (sys[particle.boxid].material != sys[newid].material) {
                change_material_diffuse(newid,cboxes,particle,sys,dt,rnd,pmutex,grids,
                                        DMM_gen,thickness,cells,couplings);
                return;
            }

            particle.boxid = newid;
        }
        else {
            newid = *(alma::choose(pbox2.begin(), pbox2.end(), rnd));

            if (sys[particle.boxid].material != sys[newid].material) {
                change_material_diffuse(newid,cboxes,particle,sys,dt,rnd,pmutex,grids,
                                        DMM_gen,thickness,cells,couplings);
                return;
            }

            particle.boxid = newid;
        }


        if (cboxes.find(particle.boxid) == cboxes.end()) {
            throw alma::geometry_error("Teleport is not allowed");
        }

        return;
    }


    throw alma::geometry_error("This point should not be reached");

    return;
}


/// Particle evolution:
///@param[in,out] particle - deviational particle to evolve
///@param[in] v        - particle velocity
///@param[in] sys      - geometry and box info
///@param[in] dt       - time to evolve
///@param[in] rnd     - random generator
///@param[in] pmutex  - mutex
///@param[in] db      - diffusive boundary scattering table
///@param[in] grids   - qgrid data per material
///@param[in] timegrid - time mesh
///@param[in,out] gz   - matrix to save particles contributions to bins
///@param[in,out] gz_jx - matrix where to save the contribution to jx
///@param[in,out] gz_jy - matrix where to save the contribution to jy
///@param[in,out] phi   - energy loss
///@param[in]     DMM_gen - DMM generators for interface scattering
///@param[in]     thickness - correct materials thickness
///@param[in]     cells     - cells data
template <class Random, class Mutex>
void evolparticle(
    alma::D_particle& particle,
    Eigen::Vector2d& v,
    std::vector<alma::geometry_2d>& sys,
    double& dt,
    Random& rnd,
    Mutex& pmutex,
    generators::diffusive_bouncing& db,
    gridData& grids,
    Eigen::VectorXd& timegrid,
    Eigen::ArrayXd& gz,
    Eigen::ArrayXd& gz_jx,
    Eigen::ArrayXd& gz_jy,
    Eigen::ArrayXd& phi,
    DMM& DMM_gen,
    std::vector<double>& thickness,
    cellData& cells,
    std::map<std::string, alma::BE_derivative_distribution>& material_sampler,
    std::vector<alma::D_particle>& particles,
    spectral_decomposition& sd,
    alma::layer_coupling& couplings) {
    Eigen::Vector2d opos(particle.pos);

    /// Make some v check
    for (auto iv : {0, 1}) {
        if (alma::almost_equal(v(iv), 0.) and v(iv) != 0.) {
            throw alma::geometry_error(
                "Error, please set all that"
                " evaluates almost_equal to 0 as 0. Otherwise,"
                " geometric algorithms will fail\n"
                "Specifically the rnew check as it contains "
                "v in spatial search of boxes to go\n");
        }
    }


    auto pq = particle.q;
    auto pb = particle.alpha;
    double mu =
        grids[sys[particle.boxid].material]->get_spectrum_at_q(pq).omega(pb);


    /// Apply PBC
    if (sys[particle.boxid].periodic) {
        pmutex.lock();
        auto PBCsol = sys[particle.boxid].translate(opos, v, sys, rnd);
        pmutex.unlock();

        particle.boxid = PBCsol.first;
        particle.pos = PBCsol.second;
        return;
    }

    /// Check if in reservoir
    if (sys[particle.boxid].reservoir) {
        /// Terminated particle
        particle.q = 0;
        particle.alpha = 0;
        dt = 0.;
        return;
    }

    Eigen::Vector2d rnew = particle.pos + dt * v;

    /// Check if inside the same box:

    if (sys[particle.boxid].inside(rnew)) {
        if (sys[particle.boxid].periodic) {
            std::cout << "Periodic boxes do not exist\n";
            std::cout << "r0\n" << opos << std::endl;
            std::cout << "rf\n" << rnew << std::endl;
            std::cout << "dt " << dt << std::endl;
            std::cout << particle.pos << std::endl;
            std::cout << "v\n" << v << std::endl;
            throw alma::geometry_error("Peridic error\n");
        }

        particle.pos = rnew;
        particle.t += dt;

        /// Make here the contribution to the segment
        process_segment(particle.boxid,
                        particle.sign,
                        dt,
                        v,
                        gz,
                        gz_jx,
                        gz_jy,
                        phi,
                        mu,
                        pq,
                        pb,
                        sd,
                        true);


        auto mat = sys[particle.boxid].material;
        
        dt = 0.;        

        return;
    }

    /// Check the place it left
    Eigen::Vector2d rp;
    rp << particle.pos(0), particle.pos(1);

    /// Here we calculate where the particle is
    /// going out
    std::tuple<double, Eigen::Vector2d, std::vector<int>> MRU_sol;

    try {
        MRU_sol = sys[particle.boxid].get_inter_side(rp, v, dt);
    }
    catch (const alma::geometry_error& geomerror) {
        Eigen::Vector2d d = sys[particle.boxid].get_center() - opos;
        particle.pos += 1.0e-6 * d / d.norm();
        // evolparticle(particle, v, sys, dt, rnd,
        // pmutex,db,grids,timegrid,gz,gz_jx,gz_jy,phi,DMM_gen,thickness,cells,material_sampler,particles,sd,couplings);
        return;
    }

    if (std::get<0>(MRU_sol) < 0) {
        throw alma::geometry_error("Error in kinematics\n");
        exit(EXIT_FAILURE);
    }
    /// Calculating leftime
    double left_time = dt - std::get<0>(MRU_sol);
    double tf = std::get<0>(MRU_sol);
    if (alma::almost_equal(left_time, 0.)) {
        left_time = 0.;
    }
    rnew = std::get<1>(MRU_sol);
    dt = left_time;

    decltype(rnew) reps = rnew + 1.0e-6 * v / v.norm();


    /// Get contact boxes
    std::set<std::size_t> cboxes;
    for (auto& [ibc, border] : sys[particle.boxid].get_contacts()) {
        cboxes.insert(ibc);
    }

    /// Check possible boxes to assign:
    std::vector<std::size_t> pboxes;

    for (auto test : cboxes) {
        if (test == particle.boxid) {
            continue;
        }

        auto border_contact = sys[particle.boxid].get_contacts()[test];
        /// If going along border we do not change
        bool parallel = alma::almost_equal((border_contact[0].np).dot(v), 0.);

        bool oriented = ((border_contact[0].np).dot(v) > 0.);

        if ((sys[test].inside(rnew) or sys[test].inside(reps)) and !parallel and
            oriented) {
            pboxes.push_back(test);
        }
    }

    /// If going inside the void
    if (pboxes.size() == 0) {
        
        /// Make here the contribution to the segment
        process_segment(particle.boxid,
                        particle.sign,
                        tf,
                        v,
                        gz,
                        gz_jx,
                        gz_jy,
                        phi,
                        mu,
                        pq,
                        pb,
                        sd);
        particle.t += tf;


        /// Correcting three corner problem
        auto corner_problem = in_corner3(rnew, sys);
        if (corner_problem.first) {
            auto csol = correct_corner_problem(
                rnew, v, sys, corner_problem.second, rnd);
            particle.boxid = csol.first;
            Eigen::Vector2d correction =
                sys[particle.boxid].get_center() - rnew;
            Eigen::Vector2d corrected_pos =
                rnew + 1.0e-6 * correction / correction.norm();

            MRU_sol =
                sys[particle.boxid].get_inter_side(corrected_pos, v, 1.0e+18);

            rnew = std::get<1>(MRU_sol);
        }


        /// Diffusive scattering
        std::size_t iborder;
        auto iborders = std::get<2>(MRU_sol);
        if (iborders.size() > 1) {
            /// We first check the borders
            /// that touch void
            std::vector<std::size_t> voidborders;

            for (auto& tvb : iborders) {
                auto& mb = sys[particle.boxid].get_border(tvb);

                Eigen::Vector2d rc = rnew + 1.0e-4 * mb.np;

                bool isvoid = true;
                for (auto& cb : cboxes) {
                    if (sys[cb].inside(rc)) {
                        isvoid = false;
                        break;
                    }
                }

                if (isvoid) {
                    voidborders.push_back(tvb);
                }
            }

            if (voidborders.empty()) {
                throw alma::geometry_error("Error in void borders\n");
            }

            // In the case two borders pointing at void
            iborder = voidborders[std::uniform_int_distribution(
                0, static_cast<int>(voidborders.size()))(rnd)];
        }
        else {
            iborder = iborders[0];
        }

        auto u = sys[particle.boxid].get_border(iborder).np;

        Eigen::Vector3d unp3;
        unp3 << u(0), u(1), 0.;

        std::size_t Nb =
            static_cast<std::size_t>(grids[sys[particle.boxid].material]
                                         ->get_spectrum_at_q(0)
                                         .omega.size());
        auto newmode = db.bounce(unp3,
                                 sys[particle.boxid].material,
                                 particle.q * Nb + particle.alpha,
                                 grids,
                                 rnd);

        particle.pos = rnew;
        particle.q = newmode.second;
        particle.alpha = newmode.first;
        
        ///Relaxation event (diffuse boundary)
        dt = 0.;
        pmutex.lock();
        particles.push_back(particle);
        pmutex.unlock();
        return;
    }
    else if (pboxes.size() == 1) {
        /// Change material
        if (sys[particle.boxid].material != sys[pboxes[0]].material) {
            /// Save contribution here
            process_segment(particle.boxid,
                            particle.sign,
                            tf,
                            v,
                            gz,
                            gz_jx,
                            gz_jy,
                            phi,
                            mu,
                            pq,
                            pb,
                            sd);
            particle.t += tf;
            particle.pos = rnew;
            change_material_diffuse(pboxes[0],cboxes,particle,sys,dt,rnd,pmutex,grids,
                                    DMM_gen,thickness,cells,couplings);
            pmutex.lock();
            particles.push_back(particle);
            pmutex.unlock();
            ///Relaxation event interface scattering
            dt = 0.;
            return;
        }
        else { /// Same material
            particle.pos = rnew;
            particle.t += tf;

            /// Make here the contribution to the segment
            process_segment(particle.boxid,
                            particle.sign,
                            tf,
                            v,
                            gz,
                            gz_jx,
                            gz_jy,
                            phi,
                            mu,
                            pq,
                            pb,
                            sd);
            particle.boxid = pboxes[0];
            if (cboxes.find(particle.boxid) == cboxes.end()) {
                throw alma::geometry_error("Teleport is not allowed");
            }
            
            return;
        }
    }
    else {
        // Make here the contribution to the segment
        process_segment(particle.boxid,
                        particle.sign,
                        tf,
                        v,
                        gz,
                        gz_jx,
                        gz_jy,
                        phi,
                        mu,
                        pq,
                        pb,
                        sd);

        particle.t += tf;
        particle.pos = rnew;


        std::vector<std::size_t> pbox2;

        Eigen::Vector2d rcheck = rnew + 1.0e-6 * v;

        for (auto& PB : pboxes) {
            if (sys[PB].inside(rcheck)) {
                pbox2.push_back(PB);
            }
        }

        std::size_t newid;
        if (pbox2.empty()) {
            newid = *(alma::choose(pboxes.begin(), pboxes.end(), rnd));

            Eigen::Vector2d d = sys[newid].get_center() - rnew;
            if (!sys[newid].periodic)
                particle.pos += 1.0e-6 * d / d.norm();

            if (sys[particle.boxid].material != sys[newid].material) {
                change_material_diffuse(newid,cboxes,particle,sys,dt,rnd,pmutex,grids,
                                        DMM_gen,thickness,cells,couplings);
                pmutex.lock();
                particles.push_back(particle);
                pmutex.unlock();
                return;
            }

            particle.boxid = newid;
        }
        else {
            newid = *(alma::choose(pbox2.begin(), pbox2.end(), rnd));

            if (sys[particle.boxid].material != sys[newid].material) {
                change_material_diffuse(newid,cboxes,particle,sys,dt,rnd,pmutex,grids,DMM_gen,
                                        thickness,cells,couplings);
                pmutex.lock();
                particles.push_back(particle);
                pmutex.unlock();
                return;
            }

            particle.boxid = newid;
        }


        if (cboxes.find(particle.boxid) == cboxes.end()) {
            throw alma::geometry_error("Teleport is not allowed");
        }
        
        return;
    }


    throw alma::geometry_error("This point should not be reached");

    return;
}


/// This reads the input file
///@param[in] filename - input filename with path
///@param[in] world    - mpi communicator
///@return structure with input parameters
input_parameters process_input(std::string& filename,
                               boost::mpi::communicator& world) {
    // Create empty property tree object
    boost::property_tree::ptree tree;

    // Parse XML input file into the tree
    boost::property_tree::read_xml(filename, tree);

    input_parameters inpars;
    /// Map to store thickness
    std::map<std::string, double> zmap;

    for (const auto& v : tree.get_child("RTA_MC2d")) {
        /// Reading geometry
        if (v.first == "geometry") {
            std::string gfname = alma::parseXMLfield<std::string>(v, "file");
            inpars.system = alma::read_geometry_XML(gfname);
        }
        if (v.first == "time") {
            inpars.dt = alma::parseXMLfield<double>(v, "dt");
            inpars.maxtime = alma::parseXMLfield<double>(v, "maxtime");
        }
        if (v.first == "gradient") {
            inpars.gradient(0) = alma::parseXMLfield<double>(v, "x");
            inpars.gradient(1) = alma::parseXMLfield<double>(v, "y");
        }

        if (v.first == "ballistic") {
            inpars.ballistic = true;
        }

        if (v.first == "convergence") {
            inpars.eps_energy = alma::parseXMLfield<double>(v, "energy");
            inpars.eps_flux = alma::parseXMLfield<double>(v, "flux");
        }

        if (v.first == "layer") {
            std::string material =
                alma::parseXMLfield<std::string>(v, "material");

            std::string layerid =
                alma::parseXMLfield<std::string>(v, "layer_name");

            std::string list_atoms =
                alma::parseXMLfield<std::string>(v, "atoms");

            std::vector<int> atoms_ids;
            std::istringstream buf(list_atoms);
            std::istream_iterator<std::string> beg(buf), end;
            std::vector<std::string> tokens(beg, end); // done!
            for (auto& s : tokens) {
                atoms_ids.push_back(boost::lexical_cast<int>(s));
            }

            inpars.couplings.stack_injection = true;

            inpars.couplings.layers[material][layerid] = atoms_ids;

            std::cout << "#Layer info: \n"
                      << "#   -material: " << material << std::endl
                      << "#   -name: " << layerid << std::endl
                      << "#   -atoms: ";
            for (auto at_id : atoms_ids)
                std::cout << at_id << '\t';
            std::cout << std::endl;
        }

        if (v.first == "layers_connection") {
            std::string matA = alma::parseXMLfield<std::string>(v, "layer_A");
            std::string matB = alma::parseXMLfield<std::string>(v, "layer_B");

            std::cout << "#Layer connection: " << matA << "->" << matB
                      << std::endl;
            std::cout << "#Layer connection: " << matB << "->" << matA
                      << std::endl;
            inpars.couplings.connection[matA] = matB;
            inpars.couplings.connection[matB] = matA;
        }


        if (v.first == "material") {
            std::string name = alma::parseXMLfield<std::string>(v, "name");
            std::string hdf5file =
                alma::parseXMLfield<std::string>(v, "database");
            double zsize = alma::parseXMLfield<double>(v, "thickness");
            zmap[name] = zsize;
            inpars.T0 = alma::parseXMLfield<double>(v, "T0");
            auto data = alma::load_bulk_hdf5(hdf5file.c_str(), world);
            inpars.system_cell[name] = std::move(std::get<1>(data));
            inpars.system_grid[name] = std::move(std::get<3>(data));

            auto processes = std::move(std::get<4>(data));

            Eigen::ArrayXXd w3(alma::calc_w0_threeph(
                *inpars.system_grid[name], *processes, inpars.T0, world));
            auto twoph_processes =
                alma::find_allowed_twoph(*inpars.system_grid[name], world);
            Eigen::ArrayXXd w2(alma::calc_w0_twoph(*inpars.system_cell[name],
                                                   *inpars.system_grid[name],
                                                   twoph_processes,
                                                   world));
            Eigen::ArrayXXd w0(w3 + w2);

            pcg_extras::seed_seq_from<std::random_device> seed_source;
            pcg64 rng(seed_source);
            /// Init the internal material properties for RTA loop

            /// The cv is in J/(K*nm**3)
            inpars.material_cv[name] = alma::calc_cv(*inpars.system_cell[name],
                                                     *inpars.system_grid[name],
                                                     inpars.T0);
            /// The w are in 1/ps
            inpars.material_w0[name] = w0;
            inpars.material_sampler[name] =
                alma::BE_derivative_distribution(*inpars.system_grid[name],
                                                 inpars.material_w0[name],
                                                 inpars.T0,
                                                 rng);
        }

        if (v.first == "particles") {
            inpars.Nparticles = alma::parseXMLfield<std::size_t>(v, "N");
        }


        if (v.first == "spectral") {
            for (auto it = v.second.begin(); it != v.second.end(); it++) {
                if (it->first == "resolution") {
                    inpars.sd.nomega =
                        alma::parseXMLfield<std::size_t>(*it, "ticks");
                }
                if (it->first == "location") {
                    std::size_t loc =
                        alma::parseXMLfield<std::size_t>(*it, "bin");
                    inpars.sd.gz_omega[loc] = Eigen::ArrayXXd(0, 0);
                    inpars.sd.gzjx_omega[loc] = Eigen::ArrayXXd(0, 0);
                    inpars.sd.gzjy_omega[loc] = Eigen::ArrayXXd(0, 0);
                    inpars.sd.fd_qmesh[loc] = Eigen::ArrayXXd(0, 0);
                }
            }
        }
    }

    for (auto& s : inpars.system) {
        if (s.reservoir or s.periodic)
            continue;
        if (s.Teq != inpars.T0) {
            throw alma::input_error("Tref in xml is different from provided T0"
                                    " which is incorrect\n");
        }
        if (s.Treal != s.Teq) {
            throw alma::input_error(
                "Initial temperature out of reference is not allowed\n"
                "use bRTA simulator with RTA option for those kind of "
                "simulations\n");
        }
    }

    /// Fill thicknesses
    for (std::size_t i = 0; i < inpars.system.size(); i++) {
        auto mat = inpars.system[i].material;
        inpars.thicknesses.push_back(zmap[mat]);
    }


    bool nograd = alma::almost_equal(inpars.gradient.norm(), 0.);

    auto ntime = 2;

    if (nograd)
        ntime = std::floor(inpars.maxtime / inpars.dt + 1.0e-6) + 1;


    /// Fill information for spectral decomposition
    if (!inpars.sd.gz_omega.empty()) {
        /// Init storing arrays
        for (auto& [loc, gzo] : inpars.sd.gzjx_omega) {
            gzo.resize(ntime - 1, inpars.sd.nomega);
            gzo.setZero();
        }

        for (auto& [loc, gzo] : inpars.sd.fd_qmesh) {
            auto mat = inpars.system[loc].material;
            auto nq = inpars.system_grid[mat]->nqpoints;
            gzo.resize(ntime - 1, nq);
            gzo.setZero();
        }


        for (auto& [loc, gzo] : inpars.sd.gzjy_omega) {
            gzo.resize(ntime - 1, inpars.sd.nomega);
            gzo.setZero();
        }

        for (auto& [loc, gzo] : inpars.sd.gz_omega) {
            gzo.resize(ntime - 1, inpars.sd.nomega);
            gzo.setZero();

            auto mat = inpars.system[loc].material;

            /// If material is not registered
            if (inpars.sd.omegagrid.count(loc) == 0) {
                double maxomega = 0;


                auto nq = inpars.system_grid[mat]->nqpoints;
                auto nb = static_cast<int>(
                    inpars.system_grid[mat]->get_spectrum_at_q(0).omega.size());

                Eigen::ArrayXXd sigmas(nb, nq);
                Eigen::ArrayXXd qpoints1stBZ(nq, 2);

                /// Get maxomega
                for (decltype(nq) iq = 0; iq < nq; iq++) {
                    auto& sp = inpars.system_grid[mat]->get_spectrum_at_q(iq);

                    auto q1BZ = inpars.system_cell[mat]->map_to_firstbz(
                        inpars.system_grid[mat]->get_q(iq));

                    qpoints1stBZ(iq, 0) = q1BZ.row(0).mean();
                    qpoints1stBZ(iq, 1) = q1BZ.row(1).mean();

                    for (int ib = 0; ib < nb; ib++) {
                        if (sp.omega(ib) > maxomega)
                            maxomega = sp.omega(ib);

                        sigmas(ib, iq) =
                            inpars.system_grid[mat]->base_sigma(sp.vg.col(ib));
                    }
                }

                auto percent = alma::calc_percentiles_log(sigmas);
                sigmas = sigmas.max(
                    std::exp(percent[0] - 1.5 * (percent[1] - percent[0])));

                double delta_omega =
                    maxomega / static_cast<double>(inpars.sd.nomega);
                inpars.sd.omegagrid[loc] =
                    Eigen::ArrayXd::LinSpaced(inpars.sd.nomega, 0., maxomega);
                inpars.sd.omegagrid[loc].array() += .5 * delta_omega;

                inpars.sd.qpoints1stBZ[loc] = qpoints1stBZ;

                inpars.sd.broadening_sigmas[loc] = sigmas;
            }
        }
    }

    return inpars;
}


/// Main loop of MC


int main(int argc, char** argv) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    auto tA = std::chrono::high_resolution_clock::now();

    if (world.size() > 1) {
        std::cout << "Cannot run in MPI" << std::endl;
        std::cout << "Multithreading is allowed via TBB" << std::endl;
        world.abort(1);
    }

    auto my_id = world.rank();

    if (my_id == 0) {
        std::cout << "********************************************"
                  << std::endl;
        std::cout << "This is ALMA/RTA_MC2d version " << ALMA_VERSION_MAJOR
                  << "." << ALMA_VERSION_MINOR << std::endl;
        std::cout << "********************************************"
                  << std::endl;
    }

    // Check that the right number of arguments have been provided.
    if (argc != 3 and argc != 4) {
        if (my_id == 0) {
            std::cerr << boost::format(
                             "Usage: %1% <inputfile.xml> nthreads nruns") %
                             argv[0]
                      << std::endl;
        }
        return 1;
    }


    std::cout << "#Init variables:\n";

    std::size_t nthreadsTBB = std::atoi(argv[2]);
    std::size_t nruns = 1;
    if (argc == 4)
        nruns = std::atoi(argv[3]);

    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
                                nthreadsTBB);

    /// Process the file:
    std::string inputfile = argv[1];

    input_parameters inpars = std::move(process_input(inputfile, world));

    /// Geometry
    std::vector<alma::geometry_2d> system;
    std::swap(inpars.system, system);

    std::vector<double> thicknesses;
    std::swap(thicknesses, inpars.thicknesses);

    /// Time
    double dt = inpars.dt;
    double maxtime = inpars.maxtime;
    std::size_t ntime = std::floor(maxtime / dt + 1.0e-6) + 1;
    maxtime = (ntime - 1) * dt;

    Eigen::VectorXd timegrid = Eigen::VectorXd::LinSpaced(ntime, 0.0, maxtime);

    /// Material data
    cellData system_cell = std::move(inpars.system_cell);
    gridData system_grid = std::move(inpars.system_grid);

    /// The random generator
    pcg_extras::seed_seq_from<std::random_device> seed_source;
    pcg64 rng(seed_source);
    /// Make the thread safe version
    tbb::enumerable_thread_specific<pcg64> rng_tbb(rng);

    if (inpars.couplings.stack_injection) {
        std::cout << "\n#Layered system corrections ON\n" << std::endl;
    }

    auto Nparticles = inpars.Nparticles;

    auto tB = std::chrono::high_resolution_clock::now();
    std::cout << "#Init variables -> Done ("
              << (static_cast<std::chrono::duration<double>>(tB - tA)).count()
              << " s )" << std::endl
              << std::endl;

    /// Building the generators and getting energy info
    std::cout << "#Building generators and tables:\n";
    std::cout << "# Isothermal generators \n" << std::endl;
    Eigen::Vector2d einfo;
    generators::Init_generator(system,
                               system_grid,
                               system_cell,
                               rng,
                               thicknesses,
                               inpars.gradient,
                               einfo);
    auto tC = std::chrono::high_resolution_clock::now();
    std::cout << "# Isothermal generators -> Done ("
              << (static_cast<std::chrono::duration<double>>(tC - tB)).count()
              << " s )\n"
              << std::endl;

    std::cout << "# Diffusive table \n" << std::endl;
    generators::diffusive_bouncing db(system_grid);
    auto tD = std::chrono::high_resolution_clock::now();
    std::cout << "# Diffusive table -> Done ("
              << (static_cast<std::chrono::duration<double>>(tD - tC)).count()
              << " s )\n"
              << std::endl;


    std::cout << "#Energy and particles information:\n";
    std::cout << "#|E| =" << einfo(1) << " J/ps" << std::endl;
    std::cout << "#Nparticles = " << Nparticles << std::endl;
    double Eeff = einfo(1) / Nparticles;
    std::cout << "#Eeff = " << Eeff << " J/ps" << std::endl;

    std::mutex sphinx;
    std::cout << "#Generating particles:" << std::endl;

    /// Map for the DMM (it is private for each thread)
    tbb::enumerable_thread_specific<DMM> DMM_gen_tbb;

    std::size_t irun = 0;

    while (irun < nruns) {
        std::cout << "#Run: " << irun << std::endl;

        unsigned int previous_percentage = 0;
        std::size_t icounter = 0;

        std::vector<alma::D_particle> particles;
        particles.reserve(Nparticles);
        for (decltype(Nparticles) i = 0; i < Nparticles; i++) {
            Eigen::Vector2d einfo_;
            auto particle = generators::Init_generator(system,
                                                       system_grid,
                                                       system_cell,
                                                       rng,
                                                       thicknesses,
                                                       inpars.gradient,
                                                       einfo_);
            //           std::cout << particle.q << '\t'<< particle.alpha <<
            //              '\t' << particle.boxid << std::endl;
            particles.push_back(particle);
        }
        std::cout << "#Starting MC loop" << std::endl;
        /// We evolve the particles

        if (alma::almost_equal(inpars.gradient.norm(), 0)) {
            /// Print info about intrinsic scattering
            std::cout << "#Intrinsic scattering: " << std::boolalpha
                      << !(inpars.ballistic) << std::endl;
            /// To store data
            Eigen::ArrayXXd gz(system.size(), ntime - 1);
            gz.setZero();
            Eigen::ArrayXXd gz_jx(gz);
            Eigen::ArrayXXd gz_jy(gz);


            tbb::enumerable_thread_specific<Eigen::ArrayXXd> gz_tbb(gz);
            tbb::enumerable_thread_specific<Eigen::ArrayXXd> gz_jx_tbb(gz_jx);
            tbb::enumerable_thread_specific<Eigen::ArrayXXd> gz_jy_tbb(gz_jy);


            tbb::parallel_for(
                static_cast<std::size_t>(0),
                static_cast<std::size_t>(Nparticles),
                static_cast<std::size_t>(1),
                [&](std::size_t i) {
                    double time = 0.;
                    alma::D_particle& particle = particles[i];
                    sphinx.lock();
                    unsigned int current_percentage = static_cast<unsigned int>(
                        100. * (icounter + 1) / Nparticles);

                    if (current_percentage > previous_percentage) {
                        unsigned int nchars = static_cast<unsigned int>(
                            72. * (icounter + 1) / Nparticles);
                        std::cout << "[";

                        for (auto i = 0u; i < nchars; ++i) {
                            std::cout << "-";
                        }
                        std::cout << ">";

                        for (auto i = nchars; i < 72; ++i) {
                            std::cout << " ";
                        }
                        std::cout << "] " << current_percentage << "%\r";
                        std::cout.flush();
                        previous_percentage = current_percentage;
                    }

                    icounter++;
                    sphinx.unlock();

                    /// Particle evoulution and info collection [Multithreading:
                    /// TBB]
                    while (time < maxtime) {
                        // std::cout << i << '\t' << time << '\t'
                        //<< particle.t  <<'\t' <<
                        // particle.pos.transpose(); //<< std::endl;
                        auto mat = system[particle.boxid].material;
                        auto v = get_v(
                            *system_grid[mat], particle.alpha, particle.q);
                        double dt_;
                        try {
                            dt_ =
                                alma::random_dt(inpars.material_w0[mat](
                                                    particle.alpha, particle.q),
                                                rng_tbb.local());
                        }
                        catch (const alma::value_error& error_) {
                            std::cout << particle.boxid << '\t'
                                      << particle.alpha << '\t' << particle.q
                                      << '\t' << particle.t << '\t' << time
                                      << std::endl;
                            throw alma::value_error("invalid scattering rate");
                        }
                        // std::cout << '\t' << dt_ << std::endl;
                        if (time + dt_ > maxtime) {
                            dt_ = maxtime - time + 1.0e-8;
                        }

                        while (!alma::almost_equal(dt_, 0.)) {
                            /// We move the particle
                            evolparticle(particle,
                                         v,
                                         system,
                                         dt_,
                                         rng_tbb.local(),
                                         sphinx,
                                         db,
                                         system_grid,
                                         timegrid,
                                         gz_tbb.local(),
                                         gz_jx_tbb.local(),
                                         gz_jy_tbb.local(),
                                         DMM_gen_tbb.local(),
                                         thicknesses,
                                         system_cell,
                                         inpars.material_sampler,
                                         inpars.sd,
                                         inpars.couplings,
                                         inpars.ballistic);
                        }
                        /// Check if arrived to reservoir
                        /// In that case terminate with that particle
                        if (particle.q == 0 and particle.alpha == 0)
                            break;

                        time = particle.t;
                    }
                });

            auto tE = std::chrono::high_resolution_clock::now();
            std::cout
                << "\n#MC loop -> Done ( "
                << (static_cast<std::chrono::duration<double>>(tE - tD)).count()
                << " s ) " << std::endl;

            std::cout << "#Collecting and processing information\n";
            for (auto& g : gz_tbb)
                gz += g;

            for (auto& g : gz_jx_tbb)
                gz_jx += g;

            for (auto& g : gz_jy_tbb)
                gz_jy += g;

            /// The gz is now processed to get the time evolution of the system
            for (auto ir = 0; ir < gz.rows(); ir++) {
                /// To obtain the energy density [J/nm**3]
                /// we divide by the volume of each box
                double vbox = system[ir].get_area() * thicknesses[ir];
                gz.row(ir) /= vbox;
                gz_jx.row(ir) /= vbox;
                gz_jy.row(ir) /= vbox;
            }
            gz *= Eeff;
            /// We want flux in [J/(m**2 * s)] it is in [J/(nm**2 * ps)]
            gz_jx *= Eeff * 1e12 * 1e9 * 1e9;
            gz_jy *= Eeff * 1e12 * 1e9 * 1e9;


            /// Spectral decomposition treatment
            if (!inpars.sd.gz_omega.empty()) {
                for (auto& [loc, g] : inpars.sd.gz_omega) {
                    double vbox = system[loc].get_area() * thicknesses[loc];
                    g *= Eeff / vbox;
                }

                for (auto& [loc, g] : inpars.sd.gzjx_omega) {
                    double vbox = system[loc].get_area() * thicknesses[loc];
                    g *= Eeff * 1e12 * 1e9 * 1e9 / vbox;
                }

                for (auto& [loc, g] : inpars.sd.gzjy_omega) {
                    double vbox = system[loc].get_area() * thicknesses[loc];
                    g *= Eeff * 1e12 * 1e9 * 1e9 / vbox;
                }

                for (auto& [loc, g] : inpars.sd.fd_qmesh) {
                    auto mat = system[loc].material;
                    double vbox = system[loc].get_area() * thicknesses[loc];
                    double vcorr =
                        thicknesses[loc] / system_cell[mat]->lattvec(2, 2);
                    int NQs = system_grid[mat]->nqpoints;

                    g *= Eeff * vcorr * NQs /
                         (vbox * 1.0e+12 * alma::constants::hbar);
                }
            }

            // Integrate gz in time to get the time response.
            for (auto ic = 1; ic < gz.cols(); ic++) {
                gz.col(ic) += gz.col(ic - 1);
                gz_jx.col(ic) += gz_jx.col(ic - 1);
                gz_jy.col(ic) += gz_jy.col(ic - 1);
            }

            /// Spectral decomposition
            if (!inpars.sd.gz_omega.empty()) {
                for (auto& [loc, g] : inpars.sd.gz_omega) {
                    for (int ir = 1; ir < g.rows(); ir++)
                        g.row(ir) += g.row(ir - 1);
                }

                for (auto& [loc, g] : inpars.sd.fd_qmesh) {
                    for (int ir = 1; ir < g.rows(); ir++)
                        g.row(ir) += g.row(ir - 1);
                }

                for (auto& [loc, g] : inpars.sd.gz_omega) {
                    auto mat = system[loc].material;
                    double vcorr =
                        thicknesses[loc] / system_cell[mat]->lattvec(2, 2);
                    g = g / (inpars.material_cv[system[loc].material] / vcorr);
                }


                for (auto& [loc, g] : inpars.sd.gzjx_omega) {
                    for (int ir = 1; ir < g.rows(); ir++)
                        g.row(ir) += g.row(ir - 1);
                }

                for (auto& [loc, g] : inpars.sd.gzjy_omega) {
                    for (int ir = 1; ir < g.rows(); ir++)
                        g.row(ir) += g.row(ir - 1);
                }
            }


            // Translate gz into a "macroscopic" temperature.
            Eigen::ArrayXXd T(gz);
            for (auto ir = 0; ir < gz.rows(); ir++) {
                /// We divide by Cv to get deviational temperature
                /// more accurate as deviation gets larger will be a cubic
                /// interpolator

                /// We need to correct the volume for the Cv
                auto mat = system[ir].material;
                double vcorr =
                    thicknesses[ir] / system_cell[mat]->lattvec(2, 2);


                T.row(ir) =
                    T.row(ir) /
                        (inpars.material_cv[system[ir].material] / vcorr) +
                    inpars.T0;
            }

            // Write out temperature profile
            std::cout << "# Writing temperature" << std::endl;
            std::string filename{
                (boost::format("temperature_%|g|K_run_%|i|.csv") % inpars.T0 %
                 irun)
                    .str()};
            Eigen::MatrixXd output{ntime - 1, system.size() + 1};
            for (std::size_t i = 0; i < ntime - 1; i++)
                output(i, 0) = (i + 0.5) * dt;
            output.bottomRightCorner(ntime - 1, system.size()) = T.transpose();
            alma::write_to_csv(filename, output, ',');


            /// Write fluxes
            std::cout << "# Writing Jx" << std::endl;
            std::string filename_jx{
                (boost::format("jx_%|g|K_run_%|i|.csv") % inpars.T0 % irun)
                    .str()};
            Eigen::MatrixXd output_jx{ntime - 1, system.size() + 1};
            for (std::size_t i = 0; i < ntime - 1; i++)
                output_jx(i, 0) = (i + 0.5) * dt;
            output_jx.bottomRightCorner(ntime - 1, system.size()) =
                gz_jx.transpose();
            alma::write_to_csv(filename_jx, output_jx, ',');

            std::cout << "# Writing Jy" << std::endl;
            std::string filename_jy{
                (boost::format("jy_%|g|K_run_%|i|.csv") % inpars.T0 % irun)
                    .str()};
            Eigen::MatrixXd output_jy{ntime - 1, system.size() + 1};
            for (std::size_t i = 0; i < ntime - 1; i++)
                output_jy(i, 0) = (i + 0.5) * dt;
            output_jy.bottomRightCorner(ntime - 1, system.size()) =
                gz_jy.transpose();
            alma::write_to_csv(filename_jy, output_jy, ',');

            if (!inpars.sd.gz_omega.empty()) {
                std::cout << "# Writing spectral decompositions" << std::endl;
                for (auto& [loc, g] : inpars.sd.gz_omega) {
                    std::string filename_{
                        (boost::format("deltaT_omega_%|i|_%|g|K_run_%|i|.csv") %
                         loc % inpars.T0 % irun)
                            .str()};
                    Eigen::MatrixXd output{ntime, inpars.sd.nomega + 1};
                    output(0, 0) = -1.0;
                    for (std::size_t i = 1; i < ntime; i++)
                        output(i, 0) = (i + 0.5) * dt;
                    output.block(0, 1, 1, inpars.sd.nomega) =
                        inpars.sd.omegagrid[loc].transpose();
                    output.bottomRightCorner(ntime - 1, inpars.sd.nomega) = g;
                    alma::write_to_csv(filename_, output, ',');
                }

                for (auto& [loc, g] : inpars.sd.fd_qmesh) {
                    std::string filename_{
                        (boost::format("fd_q_%|i|_%|g|K_run_%|i|.csv") % loc %
                         inpars.T0 % irun)
                            .str()};
                    std::string header_msg = "";
                    int nq = g.cols();
                    header_msg += "# ";
                    for (int iq = 0; iq < nq; iq++)
                        header_msg +=
                            std::to_string(inpars.sd.qpoints1stBZ[loc](iq, 0)) +
                            ",";
                    header_msg.pop_back();
                    header_msg += "\n# ";
                    for (int iq = 0; iq < nq; iq++)
                        header_msg +=
                            std::to_string(inpars.sd.qpoints1stBZ[loc](iq, 1)) +
                            ",";
                    header_msg.pop_back();
                    header_msg += "\n# ";
                    alma::write_to_csv(filename_, g, ',', false, header_msg);
                }

                for (auto& [loc, g] : inpars.sd.gzjx_omega) {
                    std::string filename_{
                        (boost::format("jx_omega_%|i|_%|g|K_run_%|i|.csv") %
                         loc % inpars.T0 % irun)
                            .str()};
                    Eigen::MatrixXd output{ntime, inpars.sd.nomega + 1};
                    output(0, 0) = -1.0;
                    for (std::size_t i = 1; i < ntime; i++)
                        output(i, 0) = (i + 0.5) * dt;
                    output.block(0, 1, 1, inpars.sd.nomega) =
                        inpars.sd.omegagrid[loc].transpose();
                    output.bottomRightCorner(ntime - 1, inpars.sd.nomega) = g;
                    alma::write_to_csv(filename_, output, ',');
                }
                for (auto& [loc, g] : inpars.sd.gzjy_omega) {
                    std::string filename_{
                        (boost::format("jy_omega_%|i|_%|g|K_run_%|i|.csv") %
                         loc % inpars.T0 % irun)
                            .str()};
                    Eigen::MatrixXd output{ntime, inpars.sd.nomega + 1};
                    output(0, 0) = -1.0;
                    for (std::size_t i = 1; i < ntime; i++)
                        output(i, 0) = (i + 0.5) * dt;
                    output.block(0, 1, 1, inpars.sd.nomega) =
                        inpars.sd.omegagrid[loc].transpose();
                    output.bottomRightCorner(ntime - 1, inpars.sd.nomega) = g;
                    alma::write_to_csv(filename_, output, ',');
                }
                inpars.sd.clean();
            }
        }
        else {
            /// This is the case of periodic structures the algorithm is
            /// different
            std::cout << "PERIODIC CASE: [only steady state is simulated]\n";
            if (inpars.eps_energy > 0.)
                std::cout << "#energy convergence " << inpars.eps_energy
                          << std::endl;
            if (inpars.eps_flux > 0.)
                std::cout << "#flux convergence " << inpars.eps_flux
                          << std::endl;

            /// To store data
            Eigen::ArrayXd gz(system.size());
            gz.setZero();
            Eigen::ArrayXd gz_jx(gz);
            Eigen::ArrayXd gz_jy(gz);
            Eigen::ArrayXd phi(gz);

            bool converged = false;
            std::size_t counter = 1;

            while (particles.size() != 0 and !converged) {
                Eigen::ArrayXd null(system.size());
                null.setZero();
                tbb::enumerable_thread_specific<Eigen::ArrayXd> phi_tbb(null);
                tbb::enumerable_thread_specific<Eigen::ArrayXd> gz_tbb(null);
                tbb::enumerable_thread_specific<Eigen::ArrayXd> gz_jx_tbb(null);
                tbb::enumerable_thread_specific<Eigen::ArrayXd> gz_jy_tbb(null);

                std::vector<alma::D_particle> newparticles;

                tbb::parallel_for(
                    static_cast<std::size_t>(0),
                    static_cast<std::size_t>(particles.size()),
                    static_cast<std::size_t>(1),
                    [&](std::size_t i) {
                        auto& particle = particles[i];
                        auto mat = system[particle.boxid].material;
                        auto v = get_v(
                            *system_grid[mat], particle.alpha, particle.q);
                        double dt_;
                        try {
                            dt_ =
                                alma::random_dt(inpars.material_w0[mat](
                                                    particle.alpha, particle.q),
                                                rng_tbb.local());
                        }
                        catch (const alma::value_error& error_) {
                            std::cout << particle.boxid << '\t'
                                      << particle.alpha << '\t' << particle.q
                                      << '\t' << particle.t << std::endl;
                            throw alma::value_error("invalid scattering rate");
                        }

                        while (!alma::almost_equal(dt_, 0.)) {
                            /// We move the particle
                            evolparticle(particle,
                                         v,
                                         system,
                                         dt_,
                                         rng_tbb.local(),
                                         sphinx,
                                         db,
                                         system_grid,
                                         timegrid,
                                         gz_tbb.local(),
                                         gz_jx_tbb.local(),
                                         gz_jy_tbb.local(),
                                         phi_tbb.local(),
                                         DMM_gen_tbb.local(),
                                         thicknesses,
                                         system_cell,
                                         inpars.material_sampler,
                                         newparticles,
                                         inpars.sd,
                                         inpars.couplings);
                        }
                    });

                Eigen::ArrayXd oldgz(gz);
                Eigen::ArrayXd oldgzjx(gz_jx);
                Eigen::ArrayXd oldgzjy(gz_jy);

                for (auto& e : gz_tbb)
                    gz += e;

                for (auto& e : gz_jx_tbb)
                    gz_jx += e;


                for (auto& e : gz_jy_tbb)
                    gz_jy += e;

                /// Reset phi
                phi.setZero();
                for (auto& e : phi_tbb)
                    phi += e;

                /// Adding new particles
                for (auto is = 0; is < phi.rows(); is++) {
                    if (!alma::almost_equal(phi(is), 0)) {
                        auto sign = alma::get_particle_sign(phi(is));
                        for (int ig = 0; ig < (int)std::abs(phi(is)); ig++) {
                            Eigen::Vector2d ppos =
                                system[is].get_random_point(rng);

                            auto mode =
                                inpars.material_sampler.at(system[is].material)
                                    .sample();

                            alma::D_particle newp(
                                ppos, mode[1], mode[0], sign, 0., is);

                            newparticles.push_back(newp);
                        }
                    }
                }

                /// Calculating the new diff
                const static double eps_gz = inpars.eps_energy;
                const static double eps_flux = inpars.eps_flux;
                double rgz = (oldgz - gz).matrix().norm() / gz.matrix().norm();
                double rjx =
                    (oldgzjx - gz_jx).matrix().norm() / gz_jx.matrix().norm();
                double rjy =
                    (oldgzjy - gz_jy).matrix().norm() / gz_jy.matrix().norm();

                particles.clear();
                particles.assign(newparticles.begin(), newparticles.end());
                std::swap(newparticles, particles);


                /// For convergence we will only take
                if (alma::almost_equal(inpars.gradient(0), 0.)) {
                    converged = rgz < eps_gz and rjy < eps_flux;
                    std::cout << counter << '\t' << particles.size() << '\t'
                              << rgz << '\t' << rjy << std::endl;
                }
                else if (alma::almost_equal(inpars.gradient(1), 0.)) {
                    converged = rgz < eps_gz and rjx < eps_flux;
                    std::cout << counter << '\t' << particles.size() << '\t'
                              << rgz << '\t' << rjx << std::endl;
                }
                else {
                    converged =
                        rgz < eps_gz and rjx < eps_flux and rjy < eps_flux;
                    std::cout << counter << '\t' << particles.size() << '\t'
                              << rgz << '\t' << rjx << '\t' << rjy << std::endl;
                }

                counter++;
            }


            /// The gz is now processed to get the time evolution of the system
            for (auto ir = 0; ir < gz.rows(); ir++) {
                /// To obtain the energy density [J/nm**3]
                /// we divide by the volume of each box
                double vbox = system[ir].get_area() * thicknesses[ir];
                gz.row(ir) /= vbox;
                gz_jx.row(ir) /= vbox;
                gz_jy.row(ir) /= vbox;
            }
            gz *= Eeff;
            /// We want flux in [J/(m**2 * s)] it is in [J/(nm**2 * ps)]
            gz_jx *= Eeff * 1e12 * 1e9 * 1e9;
            gz_jy *= Eeff * 1e12 * 1e9 * 1e9;

            // Translate gz into a "macroscopic" temperature.
            Eigen::ArrayXXd T(gz);
            for (auto ir = 0; ir < gz.rows(); ir++) {
                /// We divide by Cv to get deviational temperature
                /// more accurate as deviation gets larger will be a cubic
                /// interpolator

                /// We need to correct the volume for the Cv
                auto mat = system[ir].material;
                double vcorr =
                    thicknesses[ir] / system_cell[mat]->lattvec(2, 2);

                gz(ir) =
                    gz(ir) / (inpars.material_cv[system[ir].material] / vcorr) +
                    inpars.T0;
            }

            /// Spectral decomposition
            if (!inpars.sd.gz_omega.empty()) {
                for (auto& [loc, g] : inpars.sd.gz_omega) {
                    double vbox = system[loc].get_area() * thicknesses[loc];
                    g *= Eeff / vbox;

                    auto mat = system[loc].material;
                    double vcorr =
                        thicknesses[loc] / system_cell[mat]->lattvec(2, 2);
                    g = g / (inpars.material_cv[system[loc].material] / vcorr);
                }


                for (auto& [loc, g] : inpars.sd.gzjx_omega) {
                    double vbox = system[loc].get_area() * thicknesses[loc];
                    g *= Eeff * 1e12 * 1e9 * 1e9 / vbox;
                }

                for (auto& [loc, g] : inpars.sd.gzjy_omega) {
                    double vbox = system[loc].get_area() * thicknesses[loc];
                    g *= Eeff * 1e12 * 1e9 * 1e9 / vbox;
                }

                for (auto& [loc, g] : inpars.sd.fd_qmesh) {
                    auto mat = system[loc].material;
                    double vbox = system[loc].get_area() * thicknesses[loc];
                    double vcorr =
                        thicknesses[loc] / system_cell[mat]->lattvec(2, 2);
                    int NQs = system_grid[mat]->nqpoints;

                    g *= Eeff * vcorr * NQs /
                         (vbox * 1.0e+12 * alma::constants::hbar);
                }
            }


            // Write out temperature profile
            std::cout << "# Writing steady state" << std::endl;
            std::string filename{
                (boost::format("steady_%|g|K_run_%|i|.csv") % inpars.T0 % irun)
                    .str()};
            Eigen::MatrixXd output{3, system.size()};
            output.row(0) = gz;
            output.row(1) = gz_jx;
            output.row(2) = gz_jy;
            alma::write_to_csv(filename, output, ',');


            if (!inpars.sd.gz_omega.empty()) {
                std::cout << "# Writing spectral decompositions" << std::endl;
                for (auto& [loc, g] : inpars.sd.gz_omega) {
                    std::string filename_{
                        (boost::format(
                             "steady_deltaT_omega_%|i|_%|g|K_run_%|i|.csv") %
                         loc % inpars.T0 % irun)
                            .str()};
                    Eigen::MatrixXd output{2, inpars.sd.nomega};
                    output.block(0, 0, 1, inpars.sd.nomega) =
                        inpars.sd.omegagrid[loc].transpose();
                    output.bottomRightCorner(1, inpars.sd.nomega) = g.row(0);
                    alma::write_to_csv(filename_, output, ',');
                }


                for (auto& [loc, g] : inpars.sd.fd_qmesh) {
                    std::string filename_{
                        (boost::format("steady_fd_q_%|i|_%|g|K_run_%|i|.csv") %
                         loc % inpars.T0 % irun)
                            .str()};
                    std::string header_msg = "";
                    int nq = g.cols();
                    header_msg += "# ";
                    for (int iq = 0; iq < nq; iq++)
                        header_msg +=
                            std::to_string(inpars.sd.qpoints1stBZ[loc](iq, 0)) +
                            ",";
                    header_msg.pop_back();
                    header_msg += "\n# ";
                    for (int iq = 0; iq < nq; iq++)
                        header_msg +=
                            std::to_string(inpars.sd.qpoints1stBZ[loc](iq, 1)) +
                            ",";
                    header_msg.pop_back();
                    header_msg += "\n# ";
                    alma::write_to_csv(filename_, g, ',', false, header_msg);
                }

                for (auto& [loc, g] : inpars.sd.gzjx_omega) {
                    std::string filename_{
                        (boost::format(
                             "steady_jx_omega_%|i|_%|g|K_run_%|i|.csv") %
                         loc % inpars.T0 % irun)
                            .str()};
                    Eigen::MatrixXd output{2, inpars.sd.nomega};
                    output.block(0, 0, 1, inpars.sd.nomega) =
                        inpars.sd.omegagrid[loc].transpose();
                    output.bottomRightCorner(1, inpars.sd.nomega) = g.row(0);
                    alma::write_to_csv(filename_, output, ',');
                }
                for (auto& [loc, g] : inpars.sd.gzjy_omega) {
                    std::string filename_{
                        (boost::format(
                             "steady_jy_omega_%|i|_%|g|K_run_%|i|.csv") %
                         loc % inpars.T0 % irun)
                            .str()};
                    Eigen::MatrixXd output{2, inpars.sd.nomega};
                    output.block(0, 0, 1, inpars.sd.nomega) =
                        inpars.sd.omegagrid[loc].transpose();
                    output.bottomRightCorner(1, inpars.sd.nomega) = g.row(0);
                    alma::write_to_csv(filename_, output, ',');
                }
                inpars.sd.clean();
            }
        }
        irun++;
    }
    auto t_final = std::chrono::high_resolution_clock::now();
    std::cout
        << "# MC -> Done ("
        << (static_cast<std::chrono::duration<double>>(t_final - tD)).count()
        << " s )\n"
        << std::endl;
    std::cout << "END" << std::endl;

    return EXIT_SUCCESS;
}
