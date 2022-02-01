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
/// Definitions corresponding to structures.hpp.

#include <boost/format.hpp>
#include <exceptions.hpp>
#include <structures.hpp>

namespace alma {
Supercell_index_builder::Supercell_index_builder(const int _na,
                                                 const int _nb,
                                                 const int _nc,
                                                 const int _natoms)
    : na(_na), nb(_nb), nc(_nc), natoms(_natoms) {
    if (std::min({_na, _nb, _nc, _natoms}) <= 0)
        throw value_error("na, nb, nc and natoms must be positive");
}


Supercell_index Supercell_index_builder::create_index(const int index) const {
    auto result = std::div(index, this->na);
    auto ia = result.rem;

    result = std::div(result.quot, this->nb);
    auto ib = result.rem;
    result = std::div(result.quot, this->nc);
    return Supercell_index(index, ia, ib, result.rem, result.quot);
}


Supercell_index Supercell_index_builder::create_index(const int ia,
                                                      const int ib,
                                                      const int ic,
                                                      const int iatom) const {
    auto index = ia + this->na * (ib + this->nb * (ic + this->nc * iatom));

    return Supercell_index(index, ia, ib, ic, iatom);
}


Supercell_index Supercell_index_builder::create_index_safely(
    const int index) const {
    if (index < 0)
        throw value_error("indices must be positive");
    auto nmax = this->na * this->nb * this->nc * this->natoms;

    if (index >= nmax)
        throw value_error(boost::str(
            boost::format("invalid index = %1% >= %2%") % index % nmax));
    return this->create_index(index);
}


Supercell_index Supercell_index_builder::create_index_safely(
    const int ia,
    const int ib,
    const int ic,
    const int iatom) const {
    if (std::min({ia, ib, ic, iatom}) < 0)
        throw value_error("indices must be positive");

    if (ia >= this->na)
        throw value_error(
            boost::str(boost::format("invalid ia = %1% >= %2%") % ia % na));

    if (ib >= this->nb)
        throw value_error(
            boost::str(boost::format("invalid ib = %1% >= %2%") % ib % nb));

    if (ic >= this->nc)
        throw value_error(
            boost::str(boost::format("invalid ic = %1% >= %2%") % ic % nc));

    if (iatom >= this->natoms)
        throw value_error(boost::str(
            boost::format("invalid iatom = %1% >= %2%") % iatom % natoms));
    return this->create_index(ia, ib, ic, iatom);
}
} // namespace alma
