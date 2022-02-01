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
/// 1D Monte Carlo solver for multilayer structures that uses
/// a heat source with prescribed power density at the top
/// and heat sink at the bottom instead of isothermal reservoirs.

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <Eigen/Dense>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/serialization/map.hpp>
#include <boost/mpi.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/math/distributions/gamma.hpp>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include <pcg_random.hpp>
#pragma GCC diagnostic pop
#include <cmakevars.hpp>
#include <constants.hpp>
#include <utilities.hpp>
#include <io_utils.hpp>
#include <structures.hpp>
#include <bulk_hdf5.hpp>
#include <sampling.hpp>
#include <bulk_properties.hpp>
#include <deviational_particle.hpp>
#include <isotopic_scattering.hpp>
#include <analytic1d.hpp>

typedef std::pair<std::size_t, std::size_t> idx_pair;
typedef std::array<double, 3> emissionID;

/// Working directory of the user
boost::filesystem::path launch_path;
/// Target directory for writing output
std::string target_directory = "AUTO";

/// Probability density function for a Gamma distribution.
///
/// @param[in] k - shape parameter
/// @param[in] theta - scale parameter
/// @param[in] x - point at which to evaluate the function
/// @return the value of the pdf
double gamma_pdf(double k, double theta, double x) {
    return boost::math::gamma_p_derivative(k, x / theta) / theta;
}

/// Helper function that determines in which bin a particle is situated.
///
/// @param[in] grid - sorted grid of values defining the bins
/// @para[in] target - position of the particle
/// @return the index of the bin where the particle is
inline int get_bin_index(const Eigen::Ref<const Eigen::VectorXd>& grid,
                         double target) {
    auto first = grid.data();
    auto last = grid.data() + grid.size();

    int result =
        static_cast<int>(std::lower_bound(first, last, target) - first) - 1;
    if (result >= grid.size()) {
        result = grid.size() - 1;
    }
    if (result < 0) { // occurs when target == grid.minCoeff()
        result = 0;
    }
    return result;
}

/// Class that helps run simulations of general 1D structures sandwiched
/// between a top power source and bottom heat sink, in the steady-state regime.
class Steady_1d_simulator {
private:
    /// Random number generator.
    pcg64 rng;
    /// Description of H5 repository root directory
    std::string h5_repository = ".";
    /// Vector of base names for the materials in the structure.
    std::vector<std::string> material_base;
    /// Vector of "compound" names for the materials in the structure.
    std::vector<std::string> material_compound;
    /// Vector of superlattice flags for the materials in the structure.
    std::vector<bool> material_superlattice;
    /// Vector of superlattice UIDs for the materials in the structure.
    std::vector<std::string> material_superlattice_UID;
    /// Vector of superlattice distorder scattering rates for the materials in
    /// the structure.
    std::vector<Eigen::ArrayXXd> material_w0_SLdisorder;
    /// Vector of superlattice barrier scattering rates for the materials in the
    /// structure.
    std::vector<Eigen::ArrayXXd> material_w0_SLbarriers;
    /// Vector containing the q-point grid density along A axis for all
    /// materials in the structure.
    std::vector<int> material_grid_densityA;
    /// Vector containing the q-point grid density along B axis for all
    /// materials in the structure.
    std::vector<int> material_grid_densityB;
    /// Vector containing the q-point grid density along C axis for all
    /// materials in the structure.
    std::vector<int> material_grid_densityC;
    /// Vector of structures holding configurational information about each
    /// material in the structure.
    std::vector<std::unique_ptr<alma::Crystal_structure>> material_structure;
    /// Vector of structures holding information about the phonon spectrum of
    /// each material in the structure.
    std::vector<std::unique_ptr<alma::Gamma_grid>> material_grid;
    /// Vector of structures containing information about allowed three-phonon
    /// processes in each structure.
    std::vector<std::unique_ptr<std::vector<alma::Threeph_process>>>
        material_processes;
    /// 1D thermal transport axis
    Eigen::Vector3d u;
    /// Material making up each layer in the structure.
    std::vector<int> layer_material;
    /// Thickness of each layer in the structure.
    std::vector<double> layer_thickness;
    /// Value of the coordinate marking the beginning of each layer in the
    /// structure.
    std::vector<double> layer_top;
    /// Value of the coordinate marking the end of each layer in the structure.
    std::vector<double> layer_bottom;
    /// Number of particles to be used for the whole simulation.
    int nparticles;
    /// Number of particles considered in a partial run.
    int nparticles_partial;
    /// Ambient (heat sink) temperature relative to which the deviational
    /// thermal field is computed. Also serves as reference temperature
    /// at which all phonon properties are evaluated.
    double Tambient;
    /// Power density (W/m^2) emitted by the source
    double p0source;
    /// Description of the layer structure
    std::stringstream structureinfo;
    /// Number of ticks in the real-space grid.
    ///
    /// Note that the number of bins equals this number minus 1.
    int nspace;
    /// Spatial grid containing the edged of the bins where the
    /// steady-state distribution should be evaluated.
    Eigen::VectorXd spacegrid;
    /// Auxiliary grid with the midpoint of each bin.
    Eigen::VectorXd bgrid;
    /// Width of each bin.
    double deltaspace;
    /// Map from bins to layers.
    std::vector<int> bin_layer;
    /// Number of bins in each layer.
    Eigen::ArrayXi layer_nbins;
    /// Diffuse mismatch samplers for each interface.
    std::vector<alma::Diffuse_mismatch_distribution> interface_sampler;
    /// MPI communicator used to synchronize with other processes.
    boost::mpi::communicator comm;
    /// Coordinates [nm] of the surfaces where spectral flux should be resolved.
    std::vector<double> jq_surfaces;
    /// Angular frequencies [rad/ps] for spectral flux evaluation.
    Eigen::ArrayXd jq_omega;
    /// Spectral heat flux at selected surface [J/nm^2/rad].
    Eigen::ArrayXXd djq;
    // Deviational energy density in space.
    Eigen::VectorXd gzeta;
    // Pseudotemperature in space [= sum(gzeta/tau)/sum(C/tau)]
    Eigen::VectorXd Tpseudo;
    // "Macroscopic" temperature in space [= sum(gzeta)/sum(C)]
    Eigen::VectorXd Tmacro;

    // VARIABLES USED WHILE RUNNING SIMULATION

    std::vector<double> layer_cv;
    std::vector<Eigen::ArrayXXd> layer_w0;
    std::vector<double> layer_cv_over_tau;
    std::vector<alma::BE_derivative_distribution> layer_sampler;

    // VARIABLES NEEDED TO HANDLE DIFFUSE SCATTERING AT THE SOURCE

    /// Lookup table to preserve phonon frequency in diffuse scatterings
    std::vector<std::vector<idx_pair>> omega_lookup;
    /// Number of omega divisions considered for the lookup table
    int Nomega_lookup = 100;
    /// Omega grid
    Eigen::VectorXd omegagrid;
    /// Lookup table for diffuse particle reemissions
    std::map<emissionID,
             std::vector<std::pair<std::vector<idx_pair>, std::vector<double>>>>
        emission_lookup;


public:
    /// Create an object based on the description contained in an XML file.
    ///
    /// @param[in] filename - path to the input XML file
    /// @param[inout] comm_ - MPI communicator used to synchronize with other
    /// processes
    Steady_1d_simulator(const std::string& filename,
                        boost::mpi::communicator comm_)
        : rng{pcg_extras::seed_seq_from<std::random_device>()}, comm{comm_} {
        auto my_id = comm.rank();

        // Parse the inputs and create the structures, only in the master
        // process.
        std::size_t njq_omega = 100;
        if (my_id == 0) {
            std::cout << "PARSING " << filename << " ..." << std::endl;
            // Create empty property tree object
            boost::property_tree::ptree tree;
            // Parse XML input file into the tree
            boost::property_tree::read_xml(filename, tree);

            // Traverse the tree and create the structure
            // 1. Parse materials

            std::map<std::string, int> material_catalog;

            for (const auto& v : tree.get_child("materials")) {
                if (v.first == "H5repository") {
                    h5_repository =
                        alma::parseXMLfield<std::string>(v, "root_directory");
                }

                if (v.first == "material") {
                    std::string label{
                        alma::parseXMLfield<std::string>(v, "label")};
                    std::string base{
                        alma::parseXMLfield<std::string>(v, "directory")};
                    std::string compound{
                        alma::parseXMLfield<std::string>(v, "compound")};
                    std::string SL_UID("NULL");

                    if (alma::probeXMLfield<std::string>(v,
                                                         "superlattice_UID")) {
                        SL_UID = alma::parseXMLfield<std::string>(
                            v, "superlattice_UID");
                    }

                    int gridDensityA{alma::parseXMLfield<int>(v, "gridA")};
                    int gridDensityB{alma::parseXMLfield<int>(v, "gridB")};
                    int gridDensityC{alma::parseXMLfield<int>(v, "gridC")};

                    if (material_catalog.count(label) == 0) {
                        // material is not yet in the list; add it
                        std::size_t next_index = material_catalog.size();
                        material_catalog[label] = next_index;
                        material_base.emplace_back(base);
                        material_compound.emplace_back(compound);
                        material_superlattice_UID.emplace_back(SL_UID);
                        material_grid_densityA.emplace_back(gridDensityA);
                        material_grid_densityB.emplace_back(gridDensityB);
                        material_grid_densityC.emplace_back(gridDensityC);
                    }
                    else {
                        std::cerr << "WARNING: found duplicate entry for "
                                     "material label "
                                  << label << std::endl;
                    }
                }
            }
            auto nmaterials = material_base.size();
            std::cout << " Structure contains " << nmaterials << " materials."
                      << std::endl;

            // 2. Parse layer structure

            std::map<int, std::string> layer_catalog_material;
            std::map<int, double> layer_catalog_thickness;

            for (const auto& v : tree.get_child("layers")) {
                if (v.first == "layer") {
                    std::string label{
                        alma::parseXMLfield<std::string>(v, "label")};
                    int index{alma::parseXMLfield<int>(v, "index")};
                    std::string material{
                        alma::parseXMLfield<std::string>(v, "material")};
                    double thickness{
                        alma::parseXMLfield<double>(v, "thickness")};

                    if (layer_catalog_material.count(index) == 0) {
                        // this layer index has not been created yet
                        layer_catalog_material[index] = material;
                        layer_catalog_thickness[index] = thickness;
                    }
                    else {
                        std::cerr << "ERROR: duplicate layer index found in "
                                     "input file."
                                  << std::endl;
                        comm.abort(1);
                    }
                }
            }
            // All layer information is now known, create simulation
            // structure: determine and verify total number of layers

            auto nlayers = static_cast<int>(layer_catalog_material.size());

            if (my_id == 0) {
                std::cout << " Structure contains " << nlayers
                          << " layers:" << std::endl;
            }

            for (auto& it : layer_catalog_material) {
                if (it.first > nlayers) {
                    std::cerr << "ERROR: layer index in input file should not "
                                 "exceed total number of layers."
                              << std::endl;
                    comm.abort(1);
                }
            }

            for (int index = 1; index <= nlayers; ++index) {
                std::cout << "  " << layer_catalog_material[index] << " ("
                          << alma::engineer_format(
                                 1e-9 * layer_catalog_thickness[index], true)
                          << "m)" << std::endl;
                structureinfo
                    << "LAYER " << index << " " << layer_catalog_material[index]
                    << " ("
                    << alma::engineer_format(
                           1e-9 * layer_catalog_thickness[index], true)
                    << "m)" << std::endl;
            }

            layer_material.resize(nlayers);
            layer_thickness.resize(nlayers);

            for (auto& it : layer_catalog_material) {
                layer_material.at(it.first - 1) = material_catalog[it.second];
                layer_thickness.at(it.first - 1) =
                    layer_catalog_thickness[it.first];
            }

            // 3. Parse simulation settings.

            for (const auto& v : tree.get_child("simulation")) {
                if (v.first == "core") {
                    Tambient = alma::parseXMLfield<double>(v, "Tambient");
                    p0source = alma::parseXMLfield<double>(v, "powerdensity");
                    nparticles = static_cast<int>(
                        alma::parseXMLfield<double>(v, "particles"));
                    nspace = static_cast<int>(
                                 alma::parseXMLfield<double>(v, "bins")) +
                             1;
                }

                if (v.first == "transportAxis") {
                    u(0) = alma::parseXMLfield<double>(v, "x");
                    u(1) = alma::parseXMLfield<double>(v, "y");
                    u(2) = alma::parseXMLfield<double>(v, "z");
                    u.normalize();
                }

                if (v.first == "target") {
                    target_directory =
                        alma::parseXMLfield<std::string>(v, "directory");
                }
            }

            // 4. Parse settings for resolving spectral flux, if applicable

            bool spectralflux = true;
            try {
                tree.get_child("spectralflux");
            }
            catch (boost::property_tree::ptree_bad_path& bad_path_error) {
                spectralflux = false;
            }

            if (spectralflux) {
                for (const auto& v : tree.get_child("spectralflux")) {
                    if (v.first == "resolution") {
                        njq_omega = alma::parseXMLfield<std::size_t>(
                            v, "frequencybins");
                    }

                    if (v.first == "location") {
                        this->jq_surfaces.emplace_back(
                            alma::parseXMLfield<double>(v, "position"));
                    }

                    if (v.first == "locationrange") {
                        double start = alma::parseXMLfield<double>(v, "start");
                        double stop = alma::parseXMLfield<double>(v, "stop");
                        double step = alma::parseXMLfield<double>(v, "step");

                        for (double pos = start; pos <= stop; pos += step) {
                            this->jq_surfaces.emplace_back(pos);
                        }
                    }
                }
            }

            // Calculate coordinates of layer boundaries
            layer_top.resize(nlayers);
            layer_bottom.resize(nlayers);
            layer_top.at(0) = 0.0;

            for (int nlayer = 0; nlayer < nlayers; ++nlayer) {
                layer_bottom.at(nlayer) =
                    layer_top.at(nlayer) + layer_thickness.at(nlayer);

                if (nlayer < nlayers - 1) {
                    layer_top.at(nlayer + 1) = layer_bottom.at(nlayer);
                }
            }
        }

        // Broadcast all relevant information to all processes with id != 0.
        boost::mpi::broadcast(comm, material_base, 0);
        boost::mpi::broadcast(comm, h5_repository, 0);
        boost::mpi::broadcast(comm, material_compound, 0);
        boost::mpi::broadcast(comm, material_superlattice_UID, 0);
        boost::mpi::broadcast(comm, material_grid_densityA, 0);
        boost::mpi::broadcast(comm, material_grid_densityB, 0);
        boost::mpi::broadcast(comm, material_grid_densityC, 0);
        boost::mpi::broadcast(comm, layer_material, 0);
        boost::mpi::broadcast(comm, layer_thickness, 0);
        boost::mpi::broadcast(comm, Tambient, 0);
        boost::mpi::broadcast(comm, p0source, 0);
        boost::mpi::broadcast(comm, nparticles, 0);
        boost::mpi::broadcast(comm, nspace, 0);
        boost::mpi::broadcast(comm, u.data(), u.size(), 0);
        boost::mpi::broadcast(comm, layer_bottom, 0);
        boost::mpi::broadcast(comm, layer_top, 0);
        boost::mpi::broadcast(comm, njq_omega, 0);
        boost::mpi::broadcast(comm, jq_surfaces, 0);

        // Create the spatial grids used for binning.
        spacegrid =
            Eigen::VectorXd::LinSpaced(nspace, 0.0, layer_bottom.back());
        deltaspace = spacegrid(1) - spacegrid(0);
        bgrid = spacegrid.head(nspace - 1).array() + .5 * deltaspace;
        // Assign each bin to a layer. Note that only the midpoint is
        // taken into account, regardless of whether a bin overlaps with
        // several layers.
        bin_layer.resize(nspace - 1);
        auto nlayers = static_cast<int>(layer_material.size());
        layer_nbins.resize(nlayers);
        int firstbin = 0;

        for (auto i = 0; i < nlayers; ++i) {
            int lastbin = get_bin_index(bgrid, layer_bottom[i]) + 1;

            if (lastbin == firstbin) {
                if (my_id == 0) {
                    std::cerr << "Error: one of the bins is empty" << std::endl;
                }
                comm.abort(1);
            }

            for (auto j = firstbin; j < lastbin; ++j) {
                bin_layer[j] = i;
            }

            layer_nbins(i) = lastbin - firstbin;
            firstbin = lastbin;
        }

        // Load and save data about phonons for the materials involved in the
        // calculation.

        // Initialise file system and verify that directories actually exist

        launch_path = boost::filesystem::current_path();
        auto basedir = boost::filesystem::path(h5_repository);

        if (!(boost::filesystem::exists(boost::filesystem::path(basedir)))) {
            std::cout << "ERROR:" << std::endl;
            std::cout << "Repository directory " << basedir
                      << " does not exist." << std::endl;
            comm.abort(1);
        }

        int nmaterials = material_base.size();

        for (int nmat = 0; nmat < nmaterials; ++nmat) {
            if (my_id == 0) {
                std::cout << "Importing phonon data for "
                          << material_compound.at(nmat) << std::endl;
            }

            if (!(boost::filesystem::exists(boost::filesystem::path(
                    basedir / material_base.at(nmat))))) {
                if (my_id == 0) {
                    std::cerr << "ERROR:" << std::endl;
                    std::cerr << "Material directory " << material_base.at(nmat)
                              << " does not exist within the HDF5 repository."
                              << std::endl;
                }
                comm.abort(1);
            }

            auto path =
                basedir / boost::filesystem::path(material_base.at(nmat)) /
                boost::filesystem::path((boost::format("%1%_%2%_%3%_%4%.h5") %
                                         material_compound.at(nmat) %
                                         material_grid_densityA.at(nmat) %
                                         material_grid_densityB.at(nmat) %
                                         material_grid_densityC.at(nmat))
                                            .str());

            if (!(boost::filesystem::exists(path))) {
                if (my_id == 0) {
                    std::cerr << "ERROR:" << std::endl;
                    std::cerr << "HDF5 file " << path << " does not exist."
                              << std::endl;
                }
                comm.abort(1);
            }

            auto data = alma::load_bulk_hdf5(path.string().c_str(), comm);
            material_structure.emplace_back(std::move(std::get<1>(data)));
            material_grid.emplace_back(std::move(std::get<3>(data)));
            material_processes.emplace_back(std::move(std::get<4>(data)));

            // Check if we are dealing with a superlattice.
            // If so, load the applicable scattering data.

            bool superlattice = false;
            std::string superlattice_UID = material_superlattice_UID.at(nmat);

            auto subgroups =
                alma::list_scattering_subgroups(path.string().c_str(), comm);

            int superlattice_count = 0;

            for (std::size_t ngroup = 0; ngroup < subgroups.size(); ngroup++) {
                if (subgroups.at(ngroup).find("superlattice") !=
                    std::string::npos) {
                    superlattice_count++;
                }
            }

            superlattice_count /= 2;

            if (superlattice_count > 0) {
                superlattice = true;
            }

            Eigen::ArrayXXd w0_SLdisorder;
            Eigen::ArrayXXd w0_SLbarriers;

            if (superlattice) {
                // complain if there are multiple possibilities

                if ((superlattice_count > 1) && (superlattice_UID == "NULL")) {
                    if (my_id == 0) {
                        std::cerr << "ERROR:" << std::endl;
                        std::cerr << "H5 file contains scattering information "
                                     "for multiple superlattices."
                                  << std::endl;
                        std::cerr << "Must provide the superlattice UID via "
                                     "the <superlattice> XML tag."
                                  << std::endl;
                    }
                    comm.abort(1);
                }

                // if the user provided a UID, verify that corresponding data
                // exists

                if (superlattice_UID != "NULL") {
                    int UIDcount = 0;

                    for (std::size_t ngroup = 0; ngroup < subgroups.size();
                         ngroup++) {
                        if (subgroups.at(ngroup).find(superlattice_UID) !=
                            std::string::npos) {
                            UIDcount++;
                        }
                    }

                    if (UIDcount != 2) {
                        if (my_id == 0) {
                            std::cerr << "ERROR:" << std::endl;
                            std::cerr << "H5 file does not contain any "
                                         "superlattice data with provided UID "
                                      << superlattice_UID << "." << std::endl;
                        }
                        comm.abort(1);
                    }
                }

                // load the scattering rates from the H5 file

                bool UIDmatch = true;

                for (std::size_t ngroup = 0; ngroup < subgroups.size();
                     ngroup++) {
                    bool contains_SLdisorder =
                        (subgroups.at(ngroup).find("superlattice") !=
                         std::string::npos) &&
                        (subgroups.at(ngroup).find("disorder") !=
                         std::string::npos);
                    bool contains_SLbarriers =
                        (subgroups.at(ngroup).find("superlattice") !=
                         std::string::npos) &&
                        (subgroups.at(ngroup).find("barriers") !=
                         std::string::npos);

                    if (superlattice_count > 1) {
                        UIDmatch = (subgroups.at(ngroup).find(
                                        superlattice_UID) != std::string::npos);
                    }

                    if (contains_SLdisorder && UIDmatch) {
                        auto mysubgroup = alma::load_scattering_subgroup(
                            path.string().c_str(), subgroups.at(ngroup), comm);
                        w0_SLdisorder = mysubgroup.w0;
                    }

                    if (contains_SLbarriers && UIDmatch) {
                        auto mysubgroup = alma::load_scattering_subgroup(
                            path.string().c_str(), subgroups.at(ngroup), comm);
                        w0_SLbarriers = mysubgroup.w0;
                    }
                }
            }

            material_superlattice.push_back(superlattice);
            material_w0_SLdisorder.emplace_back(w0_SLdisorder);
            material_w0_SLbarriers.emplace_back(w0_SLbarriers);
        }

        // Now the spectrum is known. Compute the frequencies at which
        // the heat flux density will be evaluated.

        double maxfreq = 0.;
        for (auto im = 0; im < nmaterials; ++im) {
            const auto& grid = material_grid[im];
            for (std::size_t iq = 0; iq < grid->nqpoints; ++iq) {
                const auto& spectrum = grid->get_spectrum_at_q(iq);
                maxfreq = std::max(maxfreq, spectrum.omega.maxCoeff());
            }
        }

        this->jq_omega.resize(njq_omega);

        for (std::size_t iw = 0; iw < njq_omega; ++iw) {
            this->jq_omega(iw) = iw * maxfreq / static_cast<double>(njq_omega);
        }

        this->djq.resize(njq_omega, this->jq_surfaces.size());
        this->djq.fill(0.);

        // Process the interfaces.
        int ninterfaces = layer_material.size() - 1;

        for (int nint = 0; nint < ninterfaces; ++nint) {
            if (my_id == 0)
                std::cout << "Creating diffuse mismatch distribution "
                          << nint + 1 << " of " << ninterfaces << std::endl;
            int id1 = layer_material[nint];
            int id2 = layer_material[nint + 1];

            interface_sampler.emplace_back(
                alma::Diffuse_mismatch_distribution(*material_grid[id1],
                                                    *material_structure[id1],
                                                    *material_grid[id2],
                                                    *material_structure[id2],
                                                    u,
                                                    0.1,
                                                    rng));
        }

    } // END CONSTRUCTOR

    /// Return the number of real-space bins used in this calculation.
    ///
    /// @return the number of ticks in the real-space grid minus one
    int get_nbins() const {
        return this->nspace - 1;
    }
    /// Return the power density of the heat source
    ///
    /// @return the source density [W/m^2]
    double get_p0source() const {
        return p0source;
    }
    /// Return the ambient (heat sink) temperature
    ///
    /// @return Tambient [K]
    double get_Tambient() const {
        return Tambient;
    }
    /// Return a description of the layer structure
    ///
    /// @return description
    std::string get_layerdescription() const {
        return structureinfo.str();
    }
    /// Return the total thickness of the structure.
    ///
    /// @return the sum of all layer widths [nm]
    double get_thickness() const {
        return layer_bottom.back();
    }
    /// Return the number of surfaces at which heat flux is computed.
    ///
    /// @return the number of surfaces defined in the input XML file
    std::size_t get_nsurfaces() const {
        return this->jq_surfaces.size();
    }
    /// Return the location where spectral flux is computed.
    ///
    /// @param[in] isurf - a surface index
    /// @return location of the surface
    double get_surfacelocation(std::size_t nsurf) const {
        return this->jq_surfaces.at(nsurf);
    }

    /// Return the computed heat flux density profile at a surface.
    ///
    /// @param[in] isurf - a surface index
    /// @return an array with two columns, one for angular frequencies
    /// [rad/ps] and one for flux densities [J/m^2/rad]
    Eigen::ArrayXXd get_flux_at_surface(std::size_t isurf) const {
        if (isurf >= this->jq_surfaces.size()) {
            throw alma::value_error("invalid surface index");
        }
        Eigen::ArrayXXd nruter(this->jq_omega.size(), 2);
        nruter.col(0) = this->jq_omega;
        nruter.col(1) = 1e30 * this->djq.col(isurf);
        return nruter;
    }

    /// Return the number of particles used for the whole simulation.
    ///
    /// @return number of particles
    int get_nparticles() const {
        return this->nparticles;
    }

    /// Helper function for processing trajectories

    void processSegment(double zetastart,
                        double zetastop,
                        double duration,
                        double w0,
                        double sign,
                        double mu,
                        double sigma) {
        auto nsurfaces = this->jq_surfaces.size();
        int startbin = get_bin_index(spacegrid, zetastart);
        startbin = std::max(0, std::min(startbin, nspace - 2));
        int stopbin = get_bin_index(spacegrid, zetastop);
        stopbin = std::max(0, std::min(stopbin, nspace - 2));

        if (startbin == stopbin) {
            gzeta(startbin) += duration * sign;
            Tpseudo(startbin) += w0 * duration * sign;
        }
        else {
            int minbin = std::min(startbin, stopbin);
            int maxbin = std::max(startbin, stopbin);
            double zetamin = std::min(zetastart, zetastop);
            double zetamax = std::max(zetastart, zetastop);
            double deltazeta = zetamax - zetamin;

            // process internal segments
            for (int nbin = minbin + 1; nbin < maxbin; ++nbin) {
                gzeta(nbin) += sign * duration * deltaspace / deltazeta;
                Tpseudo(nbin) += w0 * sign * duration * deltaspace / deltazeta;
            }
            // process left edge of segment
            gzeta(minbin) +=
                sign * duration * (spacegrid(minbin + 1) - zetamin) / deltazeta;
            Tpseudo(minbin) += w0 * sign * duration *
                               (spacegrid(minbin + 1) - zetamin) / deltazeta;
            // process right edge of segment
            gzeta(maxbin) +=
                sign * duration * (zetamax - spacegrid(maxbin)) / deltazeta;
            Tpseudo(maxbin) += w0 * sign * duration *
                               (zetamax - spacegrid(maxbin)) / deltazeta;
        }

        // Obtain the contribution of this segment to the spectral
        // heat flux at each surface.
        // This is performed through kernel density estimation with
        // Gamma distributions,
        // to avoid loss of norm towards negative frequencies.

        double k = mu * mu / sigma;
        double theta = sigma / mu;

        for (std::size_t is = 0; is < nsurfaces; ++is) {
            double zetas = this->jq_surfaces[is];
            // If the particle has crossed the surface, count
            // its contribution.
            if ((zetas - zetastart) * (zetas - zetastop) < 0) {
                double corrected_sign = (zetastop > zetastart) ? sign : -sign;
                for (std::size_t iw = 0;
                     iw < static_cast<std::size_t>(this->jq_omega.size());
                     ++iw) {
                    this->djq(iw, is) +=
                        corrected_sign *
                        gamma_pdf(k, theta, this->jq_omega(iw));
                }
            }
        }
    }

    /// Helper function to prebuild the emission lookup map
    void build_emission_lookup(emissionID my_ID) {
        Eigen::Vector3d inward_normal;
        inward_normal << my_ID[0], my_ID[1], my_ID[2];
        inward_normal.normalize();

        std::vector<std::pair<std::vector<idx_pair>, std::vector<double>>>
            my_table;

        for (int omega_idx = 0; omega_idx < Nomega_lookup; omega_idx++) {
            // grab the list of potentially selectable modes
            std::vector<idx_pair> longlist = omega_lookup.at(omega_idx);

            // build list of modes that point inward and their corresponding
            // unnormalised selection probability
            std::vector<idx_pair> shortlist;
            std::vector<double> raw_prob;
            double total_prob = 0.0;

            for (std::size_t nlong = 0; nlong < longlist.size(); nlong++) {
                idx_pair candidate = longlist.at(nlong);
                auto& spectrum =
                    material_grid[layer_material.front()]->get_spectrum_at_q(
                        candidate.first);
                Eigen::Vector3d vg = spectrum.vg.col(candidate.second).array();
                double vproj = vg.dot(inward_normal);

                if (vproj > 0.0) { // selectable mode

                    // register this mode in the list
                    shortlist.emplace_back(candidate);

                    // register the raw probability
                    // (note that heat capacity is omitted since it is virtually
                    // constant among the selectable modes (they all belong to
                    // the same omega bin)
                    raw_prob.emplace_back(vproj);
                    total_prob += raw_prob.back();
                }
            }

            // construct the cumulative probability distribution
            std::vector<double> cumul_prob;

            if (raw_prob.size() > 0) {
                cumul_prob.emplace_back(raw_prob.front() / total_prob);
            }

            for (std::size_t ncumul = 1; ncumul < raw_prob.size(); ncumul++) {
                double accumul = cumul_prob.back();
                cumul_prob.emplace_back(accumul +
                                        raw_prob.at(ncumul) / total_prob);
            }

            my_table.emplace_back(
                std::pair<std::vector<idx_pair>, std::vector<double>>{
                    shortlist, cumul_prob});
        }

        emission_lookup[my_ID] = my_table;
    }

    /// Helper function for particle reemission from diffusely scattering walls
    idx_pair diffuse_reemit(emissionID emID, int omega_idx) {
        if (emission_lookup.count(emID) == 0) {
            std::cout << "diffuse_reemit > error: unknown emID <" << emID[0]
                      << "," << emID[1] << "," << emID[2] << ">" << std::endl;
        }

        // obtain data from the lookup table
        std::vector<idx_pair> shortlist =
            emission_lookup[emID].at(omega_idx).first;
        std::vector<double> cumul_prob =
            emission_lookup[emID].at(omega_idx).second;

        // draw a mode and return the result

        double u = std::uniform_real_distribution(0., 1.)(rng);
        int selected_idx =
            std::lower_bound(cumul_prob.begin(), cumul_prob.end(), u) -
            cumul_prob.begin();

        return shortlist.at(selected_idx);
    }


    /// Function that runs the simulation

    Eigen::MatrixXd run(int npartial, int Npartial) {
        double T0 = this->Tambient;

        if (npartial == Npartial - 1) {
            nparticles_partial =
                nparticles - (Npartial - 1) * (nparticles / Npartial);
        }
        else {
            nparticles_partial = nparticles / Npartial;
        }

        auto nprocs = comm.size();
        auto my_id = comm.rank();

        auto nlayers = static_cast<int>(layer_material.size());

        if (npartial == 0) {
            // Compute specific heats and scattering rates in each layer at the
            // reference temperature.

            for (auto i = 0; i < nlayers; ++i) {
                auto id = layer_material[i];
                layer_cv.emplace_back(alma::calc_cv(
                    *material_structure[id], *material_grid[id], T0));
                Eigen::ArrayXXd w3{alma::calc_w0_threeph(
                    *material_grid[id], *material_processes[id], T0, comm)};

                Eigen::ArrayXXd w_elastic;

                if (material_superlattice.at(id)) {
                    w_elastic = material_w0_SLdisorder.at(id).array() +
                                material_w0_SLbarriers.at(id).array();
                }
                else {
                    auto twoph_proc =
                        alma::find_allowed_twoph(*material_grid[id], comm);
                    w_elastic = alma::calc_w0_twoph(*material_structure[id],
                                                    *material_grid[id],
                                                    twoph_proc,
                                                    comm);
                }
                layer_w0.emplace_back(w_elastic + w3);
            }

            // Compute sum(C/tau) in each layer at the reference temperature.

            double omega_max = -1.0; // maximum frequency in the uppermost layer
                                     // (next to source)

            for (auto i = 0; i < nlayers; ++i) {
                double C_over_tau = 0.0;
                auto id = layer_material[i];

                std::size_t Nq = material_grid[id]->nqpoints;
                std::size_t Nbranches =
                    material_grid[id]->get_spectrum_at_q(0).omega.size();

                double Cscaling = alma::constants::kB /
                                  static_cast<double>(Nq) /
                                  material_structure[id]->V;

                for (std::size_t nq = 0; nq < Nq; nq++) {
                    auto& spectrum = material_grid[id]->get_spectrum_at_q(nq);

                    for (std::size_t nbranch = 0; nbranch < Nbranches;
                         nbranch++) {
                        double myC =
                            Cscaling * alma::bose_einstein_kernel(
                                           spectrum.omega(nbranch), T0);
                        double myw0 = layer_w0.at(i).operator()(nbranch, nq);
                        C_over_tau += myC * myw0;

                        if (i == 0 && spectrum.omega(nbranch) > omega_max) {
                            omega_max = spectrum.omega(nbranch);
                        }
                    }
                }

                layer_cv_over_tau.emplace_back(C_over_tau);
            }

            // Create omega lookup table needed for diffuse scatterings at the
            // source

            omegagrid.setLinSpaced(Nomega_lookup + 1, 0.0, omega_max);

            std::vector<idx_pair> dummy;
            for (int nomega_lookup = 0; nomega_lookup < Nomega_lookup;
                 nomega_lookup++) {
                omega_lookup.emplace_back(dummy);
            }

            for (std::size_t nq = 0;
                 nq < material_grid[layer_material.front()]->nqpoints;
                 nq++) {
                auto& spectrum =
                    material_grid[layer_material.front()]->get_spectrum_at_q(
                        nq);
                for (int nbranch = 0; nbranch < spectrum.omega.size();
                     nbranch++) {
                    double myomega = spectrum.omega(nbranch);
                    int omega_idx = get_bin_index(omegagrid, myomega);
                    idx_pair lookup_entry({nq, nbranch});
                    omega_lookup.at(omega_idx).emplace_back(lookup_entry);
                }
            }

            // Prebuild the diffuse emission map
            emissionID em_source({{u(0), u(1), u(2)}});
            build_emission_lookup(em_source);

            // Create a set of random samplers for intrinsic scattering.

            for (auto i = 0; i < nlayers; ++i) {
                auto id = layer_material[i];
                layer_sampler.emplace_back(alma::BE_derivative_distribution(
                    *material_grid[id], layer_w0[i], T0, rng));
            }
        }

        // Distribution for the source emission.
        alma::planar_source_distribution dist_source{
            *material_grid[layer_material.front()], T0, u, rng};

        // Total deviational power per unit area.
        // This quantity is expressed in J/ps/nm^2.
        double dotEeff = 1e-30 * this->p0source;
        // Total deviational power per particle.
        double dotepsilon = dotEeff / static_cast<double>(nparticles_partial);

        auto indices = alma::my_jobs(nparticles_partial, nprocs, my_id);
        auto nindices = indices[1] - indices[0];
        unsigned int previous_percentage = 0;

        // Initialise energy density and temperature
        gzeta.resize(nspace - 1);
        gzeta.setConstant(0.0);
        Tpseudo.resize(nspace - 1);
        Tpseudo.setConstant(0.0);
        Tmacro.resize(nspace - 1);
        Tmacro.setConstant(0.0);

        // Initialise spectral flux
        this->djq.fill(0.0);

        // *** MAIN MONTE CARLO LOOP ***

        for (auto nparticle = indices[0]; nparticle < indices[1]; ++nparticle) {
            // Variables that define a trajectory segment.
            // zeta signifies the coordinate measured along the 1D transport
            // axis.
            double traj_zeta1;
            double traj_t1;
            double traj_zeta2;
            double traj_t2;
            double traj_w0;
            double traj_omega;
            double traj_sigma;

            unsigned int current_percentage = static_cast<unsigned int>(
                100. * (nparticle + 1) / (Npartial * nindices));

            if (npartial > 0) {
                current_percentage += 100 * npartial / Npartial;
            }

            if ((my_id == 0) && (current_percentage > previous_percentage)) {
                unsigned int nchars = static_cast<unsigned int>(
                    72. * (nparticle + 1) / (Npartial * nindices));

                if (npartial > 0) {
                    nchars += static_cast<unsigned int>(
                        std::round(72.0 * static_cast<double>(npartial) /
                                   static_cast<double>(Npartial)));
                }

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

            // LAUNCH A NEW PARTICLE FROM THE HEAT SOURCE.

            traj_t1 = 0.0;
            traj_zeta1 = layer_top.front();

            std::unique_ptr<alma::D_particle> particle;

            int inside_layer = 0;
            int at_interface = -1;

            auto mode = dist_source.sample();
            particle = std::make_unique<alma::D_particle>(
                Eigen::Vector3d(0., 0., layer_top.front()),
                mode[1],
                mode[0],
                alma::get_particle_sign(1.0),
                0.);

            // TRACK PARTICLE UNTIL IT IS ABSORBED
            for (;;) {
                bool scattered_at_source = false;
                bool scattered_at_sink = false;
                bool scattered_at_interface = false;
                bool scattered_inside = false;
                char interface_side = 'X';

                int mat_id = layer_material[inside_layer];
                double dt = alma::random_dt(
                    layer_w0[inside_layer](particle->alpha, particle->q), rng);

                // Provisionally move the particle
                auto& spectrum =
                    material_grid[mat_id]->get_spectrum_at_q(particle->q);
                Eigen::Vector3d vg = spectrum.vg.col(particle->alpha);
                double zetastart = traj_zeta1;
                double deltazeta = u.dot(vg) * dt;
                double zetaend = zetastart + deltazeta;
                traj_omega = spectrum.omega[particle->alpha];
                traj_sigma =
                    material_grid[layer_material[inside_layer]]->base_sigma(vg);
                traj_w0 = layer_w0[inside_layer](particle->alpha, particle->q);

                // fix numerical artifacts that can trap a particle
                if ((zetastart == layer_top.at(inside_layer)) &&
                    (zetaend == zetastart)) { // trapped at top
                    zetaend = zetastart + 1e-8;
                }

                if ((zetastart == layer_bottom.at(inside_layer)) &&
                    (zetaend == zetastart)) { // trapped at bottom
                    zetaend = zetastart - 1e-8;
                }

                // Check if particle would still be within the same
                // layer. If not, enforce scattering at the encountered
                // boundary and determine where scattering took place.

                if (zetaend >= layer_bottom.at(inside_layer)) {
                    if (inside_layer ==
                        nlayers - 1) { // particle reached heat sink
                        scattered_at_sink = true;
                    }
                    else { // not at heat sink
                        scattered_at_interface = true;
                        at_interface = inside_layer;
                        // bottom of n-th layer = n-th interface
                        interface_side = 'A';
                    }
                    traj_zeta2 = layer_bottom.at(inside_layer);
                }
                else if (zetaend <= layer_top.at(inside_layer)) {
                    if (inside_layer == 0) {
                        // particle is at the source
                        scattered_at_source = true;
                    }
                    else { // not at the source
                        scattered_at_interface = true;
                        at_interface = inside_layer - 1;
                        // top of n-th layer is (n-1)-th interface
                        interface_side = 'B';
                    }
                    traj_zeta2 = layer_top.at(inside_layer);
                }
                else { // particle scattered intrinsically inside a layer
                    scattered_inside = true;
                    traj_zeta2 = zetaend;
                }

                if (alma::almost_equal(zetaend, zetastart)) {
                    traj_t2 = traj_t1 + dt;
                }
                else {
                    traj_t2 = traj_t1 + dt * std::abs((traj_zeta2 - zetastart) /
                                                      (zetaend - zetastart));
                }

                // PROCESS THIS SEGMENT
                processSegment(traj_zeta1,
                               traj_zeta2,
                               traj_t2 - traj_t1,
                               traj_w0,
                               static_cast<double>(particle->sign),
                               traj_omega,
                               traj_sigma);

                // Draw new particle information

                // Scattering took place inside a layer: draw new mode and
                // accept
                if (scattered_inside) {
                    auto mode = layer_sampler.at(inside_layer).sample();
                    particle->q = mode[1];
                    particle->alpha = mode[0];
                }

                // Scattering took place at the heat sink: absorb and terminate
                // particle.
                else if (scattered_at_sink) {
                    break;
                }

                // Scattering took place at the heat source: diffusely reemit
                // the particle.
                else if (scattered_at_source) {
                    int omega_idx =
                        get_bin_index(omegagrid,
                                      material_grid[layer_material.front()]
                                          ->get_spectrum_at_q(particle->q)
                                          .omega(particle->alpha));
                    emissionID em_source({{u(0), u(1), u(2)}});
                    idx_pair selected_mode =
                        diffuse_reemit(em_source, omega_idx);
                    particle->q = selected_mode.first;
                    particle->alpha = selected_mode.second;
                }

                // scattering took place at an interface
                else if (scattered_at_interface) {
                    char newside = interface_sampler.at(at_interface)
                                       .reemit(interface_side, *particle);

                    if ((interface_side == 'A') && (newside == 'B')) {
                        // crosses interface left to right
                        inside_layer++;
                    }

                    if ((interface_side == 'B') && (newside == 'A')) {
                        // crosses interface right to left
                        inside_layer--;
                    }

                    if (newside == 'X') { // something went wrong
                        break;
                    }
                }

                // INITIALISE NEXT SEGMENT
                traj_zeta1 = traj_zeta2;
                traj_t1 = traj_t2;

            } // this particle is terminated

        } // END OF MONTE CARLO LOOP

        // Obtain the total spectral flux.

        if (this->jq_surfaces.size() != 0) {
            Eigen::ArrayXXd mydjq{this->djq * dotepsilon};
            this->djq.fill(0.);
            boost::mpi::all_reduce(comm,
                                   mydjq.data(),
                                   mydjq.size(),
                                   this->djq.data(),
                                   std::plus<double>());
        }

        if (my_id == 0 && npartial == Npartial - 1) {
            std::cout << std::endl;
        }

        // Obtain the output total deviational power profile.
        Eigen::ArrayXd gzeta_final{nspace - 1};
        boost::mpi::all_reduce(comm,
                               gzeta.data(),
                               gzeta.size(),
                               gzeta_final.data(),
                               std::plus<double>());
        gzeta_final *= (dotepsilon / deltaspace);

        // Determine the deviational pseudotemperature
        Eigen::VectorXd Tpseudo_final{nspace - 1};
        boost::mpi::all_reduce(comm,
                               Tpseudo.data(),
                               Tpseudo.size(),
                               Tpseudo_final.data(),
                               std::plus<double>());
        Tpseudo_final *= (dotepsilon / deltaspace);

        for (auto i = 0; i < nspace - 1; ++i) {
            Tpseudo_final(i) /= layer_cv_over_tau[bin_layer[i]];
        }

        // Determine the deviational "macroscopic" temperature
        Eigen::VectorXd Tmacro_final{nspace - 1};

        for (auto i = 0; i < nspace - 1; ++i) {
            Tmacro_final(i) = gzeta_final(i) / layer_cv[bin_layer[i]];
        }

        // Add the reference to get absolute temperatures.
        for (auto i = 0; i < nspace - 1; ++i) {
            Tpseudo_final(i) += T0;
            Tmacro_final(i) += T0;
        }

        // Return the result.
        Eigen::MatrixXd result(nspace - 1, 2);
        result.col(0) = Tpseudo_final;
        result.col(1) = Tmacro_final;

        return result;
    } // END run()
};    // END CLASS

int main(int argc, char** argv) {
    boost::mpi::environment env;
    boost::mpi::communicator world;
    auto my_id = world.rank();

    if (my_id == 0) {
        std::cout << "*******************************************************"
                  << std::endl;
        std::cout << "This is ALMA/steady_montecarlo1d_powersource version "
                  << ALMA_VERSION_MAJOR << "." << ALMA_VERSION_MINOR
                  << std::endl;
        std::cout << "*******************************************************"
                  << std::endl;
    }

    // Check that the right number of arguments have been provided.
    if (argc < 2) {
        if (my_id == 0) {
            std::cerr << boost::format("Usage: %1% <inputfile.xml>") % argv[0]
                      << std::endl;
        }
        return 1;
    }

    // verify that input file exists.
    if (!boost::filesystem::exists(boost::filesystem::path{argv[1]})) {
        if (my_id == 0) {
            std::cout << "ERROR: input file " << argv[1] << " does not exist."
                      << std::endl;
        }
        world.abort(1);
    }

    // Build the simulator object.

    boost::filesystem::path xmlfile{argv[1]};
    Steady_1d_simulator sim(xmlfile.string(), world);
    world.barrier();

    if (my_id == 0) {
        std::cout << "Initialization finished" << std::endl;
    }

    // Resolve output directory and create it when needed

    boost::filesystem::current_path(launch_path);
    boost::filesystem::path outputbase;

    if (target_directory.compare("AUTO") == 0) {
        // Strip path information and extension from filename to build output
        // path.
        outputbase = boost::filesystem::path("output") /
                     boost::filesystem::path("steady_montecarlo1d") /
                     xmlfile.stem();
    }

    else {
        outputbase = boost::filesystem::path(target_directory);
    }

    if (!(boost::filesystem::exists(outputbase))) {
        boost::filesystem::create_directories(outputbase);
    }

    if (my_id == 0) {
        std::cout << "Files will be written to " << outputbase.string()
                  << std::endl;
    }

    if (my_id == 0) {
        std::cout << "RUNNING SIMULATION..." << std::endl;
    }

    // Run a series of partial simulations

    int Npartial = 5;
    Eigen::VectorXd kappaeff_partial(Npartial);
    Eigen::VectorXd deltaTsource_partial(Npartial);
    Eigen::VectorXd Tpseudo;
    Eigen::VectorXd Tmacro;

    int Nsurfaces = sim.get_nsurfaces();
    Eigen::MatrixXd spectralflux;

    for (int npartial = 0; npartial < Npartial; npartial++) {
        Eigen::MatrixXd result = sim.run(npartial, Npartial);

        // Obtain results
        Eigen::VectorXd Tpseudo_partial = result.col(0);
        Eigen::VectorXd Tmacro_partial = result.col(1);

        if (npartial == 0) {
            Tpseudo.resize(Tpseudo_partial.size());
            Tpseudo.setConstant(0.0);
            Tmacro.resize(Tmacro_partial.size());
            Tmacro.setConstant(0.0);
        }

        for (int nrow = 0; nrow < Tpseudo_partial.size(); nrow++) {
            Tpseudo(nrow) +=
                Tpseudo_partial(nrow) / static_cast<double>(Npartial);
            Tmacro(nrow) +=
                Tmacro_partial(nrow) / static_cast<double>(Npartial);
        }

        deltaTsource_partial(npartial) = Tmacro_partial(0) - sim.get_Tambient();

        kappaeff_partial(npartial) =
            sim.get_p0source() / ((Tmacro_partial(0) - sim.get_Tambient()) /
                                  (sim.get_thickness() * 1e-9));

        for (int nsurface = 0; nsurface < Nsurfaces; nsurface++) {
            Eigen::VectorXd surfaceflux =
                sim.get_flux_at_surface(nsurface).col(1);

            if (npartial == 0 && nsurface == 0) {
                spectralflux.resize(surfaceflux.size(), Nsurfaces);
                spectralflux.fill(0.0);
            }

            for (int nomega = 0; nomega < surfaceflux.size(); nomega++) {
                spectralflux(nomega, nsurface) +=
                    surfaceflux(nomega) / static_cast<double>(Npartial);
            }
        }
    }

    // Determine temperature rise at the heat source and kappa_eff and their
    // stochastic uncertainty

    double deltaTsource =
        deltaTsource_partial.sum() / static_cast<double>(Npartial);
    double sigma_deltaTsource =
        std::sqrt((deltaTsource_partial.array() - deltaTsource).square().sum() /
                  static_cast<double>(Npartial * (Npartial - 1)));

    double kappa_eff = kappaeff_partial.sum() / static_cast<double>(Npartial);
    double sigma_kappaeff =
        std::sqrt((kappaeff_partial.array() - kappa_eff).square().sum() /
                  static_cast<double>(Npartial * (Npartial - 1)));

    // Obtain some basic information
    std::size_t Nbins = sim.get_nbins();
    double Tambient = sim.get_Tambient();
    double p0source = sim.get_p0source();

    if (my_id == 0) {
        std::cout << "Tambient = " << Tambient << " K" << std::endl;
        std::cout << "source dissipation = "
                  << alma::engineer_format(p0source, true) << "W/m^2"
                  << std::endl;
    }

    // Save all relevant data to output files.
    if (my_id == 0) {
        // Write basic information
        std::string filename{
            (boost::format("basicproperties_%|g|K.txt") % Tambient).str()};
        std::string path{(outputbase / filename).string()};

        std::ofstream infowriter;
        infowriter.open(path);
        infowriter << sim.get_layerdescription();
        infowriter << "T_AMBIENT " << Tambient << " K" << std::endl;
        infowriter << "SOURCE_DISSIPATION "
                   << alma::engineer_format(p0source, true) << "W/m^2"
                   << std::endl;
        infowriter << "N_PARTICLES " << sim.get_nparticles() << std::endl;
        infowriter << "DELTAT_SOURCE " << deltaTsource << " K" << std::endl;
        infowriter << "DELTAT_SOURCE_TOLERANCE " << sigma_deltaTsource << " K"
                   << std::endl;
        infowriter << "EFF_CONDUCTIVITY " << kappa_eff << " W/m-K" << std::endl;
        infowriter << "EFF_CONDUCTIVITY_TOLERANCE " << sigma_kappaeff
                   << " W/m-K" << std::endl;
        double r_eff = 1e-9 * sim.get_thickness() / kappa_eff;
        infowriter << "EFF_RESISTIVITY " << alma::engineer_format(r_eff, true)
                   << "K-m^2/W" << std::endl;
        infowriter << "EFF_CONDUCTANCE "
                   << alma::engineer_format(1.0 / r_eff, true) << "W/K-m^2"
                   << std::endl;
        infowriter.close();

        // Write temperature profile
        filename = (boost::format("temperature_%|g|K.csv") % Tambient).str();
        path = (outputbase / filename).string();
        int nspace = Nbins + 1;
        Eigen::ArrayXd bgrid{
            Eigen::VectorXd::LinSpaced(nspace, 0.0, sim.get_thickness())
                .head(Nbins)};
        Eigen::MatrixXd output{Nbins, 2};
        output.col(0).array() = bgrid.array() + .5 * (bgrid(1) - bgrid(0));
        output.col(1) = Tmacro;
        alma::write_to_csv(path, output, ',');

        // Write spectral heat flux
        auto nsurfaces = sim.get_nsurfaces();
        for (std::size_t is = 0; is < nsurfaces; ++is) {
            filename = (boost::format("spectralflux_surface_%|d|_%|g|K.csv") %
                        (is + 1) % Tambient)
                           .str();
            path = (outputbase / filename).string();
            std::stringstream fileheader;
            fileheader << "PHONON_FREQUENCY,SPECTRALFLUX_AT_POSITION_";
            fileheader << sim.get_surfacelocation(is) << std::endl;

            Eigen::MatrixXd dummy = sim.get_flux_at_surface(is);
            Eigen::MatrixXd output(dummy.rows(), 2);
            output.col(0) = dummy.col(0);
            output.col(1) = spectralflux.col(is);
            alma::write_to_csv(path, output, ',', false, fileheader.str());
        }
    }

    if (my_id == 0) {
        std::cout << "Temperature rise at the source: " << deltaTsource << " K";
        std::cout << " +/- " << sigma_deltaTsource << " K" << std::endl;
        std::cout << "Effective conductivity: " << kappa_eff << " +/- "
                  << sigma_kappaeff << " W/m-K" << std::endl;
    }

    return 0;
}
