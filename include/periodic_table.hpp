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
/// Data about the elements in the periodic table.

#include <string>
#include <vector>
#include <algorithm>
#include <exceptions.hpp>

namespace alma {
/// Names of all supported elements (and then some) sorted by atomic
/// number.
const std::vector<std::string> elements{
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na",
    "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti",
    "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs",
    "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra",
    "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
    "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt"};

/// Return the atomic number of an element.
///
/// @param[in] element - chemical symbol
/// @return the atomic number
inline int symbol_to_Z(const std::string& element) {
    auto pos = std::find(elements.begin(), elements.end(), element);

    if (pos == elements.end()) {
        throw value_error("unknown chemical symbol");
    }
    return std::distance(elements.begin(), pos) + 1;
}


/// Return the symbol corresponding to an atomic number.
///
/// @param[in] Z - atomic number
/// @return a chemical symbol from the periodic table
inline std::string Z_to_symbol(decltype(elements.size()) Z) {
    if ((Z < 1) || (Z > elements.size()))
        throw value_error("invalid atomic number");
    return elements[Z - 1];
}


/// Compute the average mass of an element.
///
/// @param[in] element - chemical symbol or virtual element
/// description
/// @param[in] disablev - disable parsing of virtual elements
/// @return the mass in a.m.u.
double get_mass(const std::string& element, bool disablev = false);

/// Compute the squared Pearson deviation factor of the mass
/// distribution for an element.
///
/// @param[in] element - chemical symbol or virtual element
/// description
/// @param[in] disablev - disable parsing of virtual elements
/// @return the g factor in a.m.u.
double get_gfactor(const std::string& element, bool disablev = false);

/// Class whose objects describe a virtual "element" in an alloy,
/// i.e., a statistical mixture of chemical elements.
class Virtual_element {
public:
    /// Trivial constructor.
    ///
    /// The vector of atomic ratios need not be normalized,
    /// though its elements cannot be negative.
    Virtual_element(const std::vector<std::string>& symbols,
                    const std::vector<double>& ratios);
    /// Create a Virtual_element from a string describing its
    /// composition.
    ///
    /// The format of the string is:
    /// "element1:ratio1;element2:ratio2"
    /// with as many elements as necessary. Whitespace is not
    /// significative. ratios must be positive decimal numbers, but
    /// need not be normalized.
    /// @param[in] composition - a string describing the composition
    /// ratios. The elements of the latter add up to 1.
    Virtual_element(const std::string& composition);
    /// Return the number of elements in the mix.
    ///
    /// @return the number of chemical elements making up the
    /// virtual element.
    std::size_t get_nelements() const {
        return this->symbols.size();
    }


    /// Return the symbol of the i-th component of the mixture.
    ///
    /// @param[in] i - index of the element
    /// @return the chemical symbol of the requested element
    std::string get_symbol(std::size_t i) const {
        if (i >= this->symbols.size())
            throw value_error("invalid element number");
        return this->symbols[i];
    }


    /// Return the atomic ratio of the i-th component of the
    /// mixture.
    ///
    /// @param[in] i - index of the element
    /// @return the atomic ratio of the requested element
    double get_ratio(std::size_t i) const {
        if (i >= this->symbols.size())
            throw value_error("invalid element number");
        return this->ratios[i];
    }


    /// Obtain a textual representation of the virtual element.
    ///
    /// @return a string in a format suitable to be passed to the
    /// class constructor.
    std::string get_name() const;

private:
    /// Symbols of chemical elements in the mixture.
    std::vector<std::string> symbols;
    /// Ratio of each element. The elements of this vector add up
    /// to 1.
    std::vector<double> ratios;
};
} // namespace alma
