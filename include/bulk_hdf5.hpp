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
/// Functions to save and load data about a bulk system to an HDF5
/// file.

#include <string>
#include <vector>
#include <tuple>
#include <boost/mpi.hpp>
#include <structures.hpp>
#include <symmetry.hpp>
#include <qpoint_grid.hpp>
#include <processes.hpp>

namespace alma {
/// Save all relevant information about a bulk material to an
/// HDF5 file.
///
/// @param[in] filename - path to the HDF5 file. It can be NULL
/// for all processes except the one with rank == 0.
/// @param[in] description - a free-format description
/// @param[in] cell - description of the unit cell
/// @param[in] symmetries - symmetry operations of the unit cell
/// @param[in] grid - phonon spectrum on a refular grid
/// @param[in] processes - allowed three-phonon processes
/// @param[in] comm - MPI communicator used to coordinate with all
/// other processes
void save_bulk_hdf5(const char* filename,
                    const std::string& description,
                    const Crystal_structure& cell,
                    const Symmetry_operations& symmetries,
                    const Gamma_grid& grid,
                    const std::vector<Threeph_process>& processes,
                    const boost::mpi::communicator& comm);

/// Reconstruct all data structures from a file saved with
/// save_bulk_hdf5().
///
/// @param[in] filename - path to the HDF5 file
/// @param[in] comm - MPI communicator used to determine which
/// part of the data to load in this process.
/// @return a tuple with all the information in the file,
/// namely:
/// description - a free-format description
/// cell - description of the unit cell
/// symmetries - symmetry operations of the unit cell
/// grid - phonon spectrum on a refular grid
/// processes - allowed three-phonon processes
std::tuple<std::string,
           std::unique_ptr<Crystal_structure>,
           std::unique_ptr<Symmetry_operations>,
           std::unique_ptr<Gamma_grid>,
           std::unique_ptr<std::vector<Threeph_process>>>
load_bulk_hdf5(const char* filename, const boost::mpi::communicator& comm);

/// Return the names of all subgroups in the "/scattering" group of
/// an HDF5 file.
///
/// An empty vector is returned both when the "/scattering" group
/// does not exist and when it exists but does not contain any
/// subset.
/// @param[in] filename - path to the HDF5 file
/// @param[in] comm - MPI communicator used to coordinate with all
/// other processes
/// @return a vector of subgroup names
std::vector<std::string> list_scattering_subgroups(
    const char* filename,
    const boost::mpi::communicator& comm);

/// POD class describing a subgroup of the "/scattering" group of an
/// HDF5 file.
class Scattering_subgroup {
public:
    friend Scattering_subgroup load_scattering_subgroup(
        const char* filename,
        const std::string& groupname,
        const boost::mpi::communicator& comm);

    /// Name of the subgroup.
    const std::string name;
    /// True unless the scattering source alters the q-point
    /// equivalence classes.
    const bool preserves_symmetry;
    /// Free-format description.
    const std::string description;
    /// Set of scattering rates [1 / ps].
    const Eigen::ArrayXXd w0;
    /// List with the names of all non-mandatory attributes
    /// in the subgroup.
    const std::vector<std::string> attributes;
    /// List with the names of all non-mandatory datasets
    /// in the subgroup.
    const std::vector<std::string> datasets;
    /// List with the names of all subsubgroups in the subgroup.
    const std::vector<std::string> groups;
    /// Public constructor that does not allow extra attributes,
    /// datasets or groups.
    Scattering_subgroup(const std::string& _name,
                        const bool _preserves_symmetry,
                        const std::string& _description,
                        const Eigen::Ref<const Eigen::ArrayXXd>& _w0)
        : name(_name), preserves_symmetry(_preserves_symmetry),
          description(_description), w0(_w0), attributes({}), datasets({}),
          groups({}) {
    }

private:
    /// Basic constructor.
    Scattering_subgroup(const std::string& _name,
                        const bool _preserves_symmetry,
                        const std::string& _description,
                        const Eigen::Ref<const Eigen::ArrayXXd>& _w0,
                        const std::vector<std::string>& _attributes,
                        const std::vector<std::string>& _datasets,
                        const std::vector<std::string>& _groups)
        : name(_name), preserves_symmetry(_preserves_symmetry),
          description(_description), w0(_w0), attributes(_attributes),
          datasets(_datasets), groups(_groups) {
    }
};
/// Read the scattering rates stored in a subgroup from the
/// "/scattering" group.
///
/// The existence of the group is not checked.
/// @param[in] filename - path to the HDF5 file
/// @param[in] groupname - name of the group
/// @param[in] comm - MPI communicator used to coordinate with all
/// other processes
/// @return a Scattering_subgroup structure
Scattering_subgroup load_scattering_subgroup(
    const char* filename,
    const std::string& groupname,
    const boost::mpi::communicator& comm);

/// Add a subgroup to the "/scattering" group of the HDF5 file.
///
/// The existence of the group is not checked.
/// @param[in] filename - path to the HDF5 file
/// @param[in] group - information about the group
/// @param[in] comm - MPI communicator used to coordinate with all
/// other processes
void write_scattering_subgroup(const char* filename,
                               const Scattering_subgroup& group,
                               const boost::mpi::communicator& comm);
} // namespace alma
