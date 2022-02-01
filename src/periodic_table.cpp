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
/// Definitions corresponding to periodic_table.hpp.

#include <map>
#include <cmath>
#include <cctype>
#include <numeric>
#include <algorithm>
#include <boost/format.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <utilities.hpp>
#include <periodic_table.hpp>

namespace alma {
/// Plain old data class containing the mass and abundance of
/// one isotope.
class Isotope {
public:
    /// Chemical symbol.
    const std::string element;
    /// Mass in a.m.u.
    const double mass;
    /// Abundance in %.
    const double percentage;
};

/// Table of isotopes, alphabetically sorted.
const std::vector<Isotope> isotopes{
    {"Ag", 106.905095, 51.84}, {"Ag", 108.904754, 48.16},
    {"Al", 26.981541, 100.0},  {"Ar", 35.967546, 0.34},
    {"Ar", 37.962732, 0.063},  {"Ar", 39.962383, 99.6},
    {"As", 74.921596, 100.0},  {"Au", 196.96656, 100.0},
    {"B", 10.012938, 19.8},    {"B", 11.009305, 80.2},
    {"Ba", 129.906277, 0.11},  {"Ba", 131.905042, 0.1},
    {"Ba", 133.90449, 2.42},   {"Ba", 134.905668, 6.59},
    {"Ba", 135.904556, 7.85},  {"Ba", 136.905816, 11.23},
    {"Ba", 137.905236, 71.7},  {"Be", 9.012183, 100.0},
    {"Bi", 208.980388, 100.0}, {"Br", 78.918336, 50.69},
    {"Br", 80.91629, 49.31},   {"C", 12.0, 98.9},
    {"C", 13.003355, 1.1},     {"Ca", 39.962591, 96.95},
    {"Ca", 41.958622, 0.65},   {"Ca", 42.95877, 0.14},
    {"Ca", 43.955485, 2.086},  {"Ca", 45.953689, 0.004},
    {"Ca", 47.952532, 0.19},   {"Cd", 105.906461, 1.25},
    {"Cd", 107.904186, 0.89},  {"Cd", 109.903007, 12.49},
    {"Cd", 110.904182, 12.8},  {"Cd", 111.902761, 24.13},
    {"Cd", 112.904401, 12.22}, {"Cd", 113.903361, 28.73},
    {"Cd", 115.904758, 7.49},  {"Ce", 135.90714, 0.19},
    {"Ce", 137.905996, 0.25},  {"Ce", 139.905442, 88.48},
    {"Ce", 141.909249, 11.08}, {"Cl", 34.968853, 75.77},
    {"Cl", 36.965903, 24.23},  {"Co", 58.933198, 100.0},
    {"Cr", 49.946046, 4.35},   {"Cr", 51.94051, 83.79},
    {"Cr", 52.940651, 9.5},    {"Cr", 53.938882, 2.36},
    {"Cs", 132.905433, 100.0}, {"Cu", 62.929599, 69.17},
    {"Cu", 64.927792, 30.83},  {"Dy", 155.924287, 0.06},
    {"Dy", 157.924412, 0.1},   {"Dy", 159.925203, 2.34},
    {"Dy", 160.926939, 18.9},  {"Dy", 161.926805, 25.5},
    {"Dy", 162.928737, 24.9},  {"Dy", 163.929183, 28.2},
    {"Er", 161.928787, 0.14},  {"Er", 163.929211, 1.61},
    {"Er", 165.930305, 33.6},  {"Er", 166.932061, 22.95},
    {"Er", 167.932383, 26.8},  {"Er", 169.935476, 14.9},
    {"Eu", 150.91986, 47.8},   {"Eu", 152.921243, 52.2},
    {"F", 18.998403, 100.0},   {"Fe", 53.939612, 5.8},
    {"Fe", 55.934939, 91.72},  {"Fe", 56.935396, 2.2},
    {"Fe", 57.933278, 0.28},   {"Ga", 68.925581, 60.1},
    {"Ga", 70.924701, 39.9},   {"Gd", 151.919803, 0.2},
    {"Gd", 153.920876, 2.18},  {"Gd", 154.822629, 14.8},
    {"Gd", 155.92213, 20.47},  {"Gd", 156.923967, 15.65},
    {"Gd", 157.924111, 24.84}, {"Gd", 159.927061, 21.86},
    {"Ge", 69.92425, 20.5},    {"Ge", 71.92208, 27.4},
    {"Ge", 72.923464, 7.8},    {"Ge", 73.921179, 36.5},
    {"Ge", 75.921403, 7.8},    {"H", 1.007825, 99.99},
    {"H", 2.014102, 0.015},    {"He", 3.016029, 0.0001},
    {"He", 4.002603, 100.0},   {"Hf", 173.940065, 0.16},
    {"Hf", 175.94142, 5.2},    {"Hf", 176.943233, 18.6},
    {"Hf", 177.94371, 27.1},   {"Hf", 178.945827, 13.74},
    {"Hf", 179.946561, 35.2},  {"Hg", 195.965812, 0.15},
    {"Hg", 197.96676, 10.1},   {"Hg", 198.968269, 17.0},
    {"Hg", 199.968316, 23.1},  {"Hg", 200.970293, 13.2},
    {"Hg", 201.970632, 29.65}, {"Hg", 203.973481, 6.8},
    {"Ho", 164.930332, 100.0}, {"I", 126.904477, 100.0},
    {"In", 112.904056, 4.3},   {"In", 114.903875, 95.7},
    {"Ir", 190.960603, 37.3},  {"Ir", 192.962942, 62.7},
    {"K", 38.963708, 93.2},    {"K", 39.963999, 0.012},
    {"K", 40.961825, 6.73},    {"Kr", 77.920397, 0.35},
    {"Kr", 79.916375, 2.25},   {"Kr", 81.913483, 11.6},
    {"Kr", 82.914134, 11.5},   {"Kr", 83.911506, 57.0},
    {"Kr", 85.910614, 17.3},   {"La", 137.907114, 0.09},
    {"La", 138.906355, 99.91}, {"Li", 6.015123, 7.42},
    {"Li", 7.016005, 92.58},   {"Lu", 174.940785, 97.4},
    {"Lu", 175.942694, 2.6},   {"Mg", 23.985045, 78.9},
    {"Mg", 24.985839, 10.0},   {"Mg", 25.982595, 11.1},
    {"Mn", 54.938046, 100.0},  {"Mo", 91.906809, 14.84},
    {"Mo", 93.905086, 9.25},   {"Mo", 94.905838, 15.92},
    {"Mo", 95.904676, 16.68},  {"Mo", 96.906018, 9.55},
    {"Mo", 97.905405, 24.13},  {"Mo", 99.907473, 9.63},
    {"N", 14.003074, 99.63},   {"N", 15.000109, 0.37},
    {"Na", 22.98977, 100.0},   {"Nb", 92.906378, 100.0},
    {"Nd", 141.907731, 27.13}, {"Nd", 142.909823, 12.18},
    {"Nd", 143.910096, 23.8},  {"Nd", 144.912582, 8.3},
    {"Nd", 145.913126, 17.19}, {"Nd", 147.916901, 5.76},
    {"Nd", 149.9209, 5.64},    {"Ne", 19.992439, 90.6},
    {"Ne", 20.993845, 0.26},   {"Ne", 21.991384, 9.2},
    {"Ni", 57.935347, 68.27},  {"Ni", 59.930789, 26.1},
    {"Ni", 60.931059, 1.13},   {"Ni", 61.928346, 3.59},
    {"Ni", 63.927968, 0.91},   {"O", 15.994915, 99.76},
    {"O", 16.999131, 0.038},   {"O", 17.999159, 0.2},
    {"Os", 183.952514, 0.02},  {"Os", 185.953852, 1.58},
    {"Os", 186.955762, 1.6},   {"Os", 187.95585, 13.3},
    {"Os", 188.958156, 16.1},  {"Os", 189.958455, 26.4},
    {"Os", 191.961487, 41.0},  {"P", 30.973763, 100.0},
    {"Pb", 203.973037, 1.4},   {"Pb", 205.974455, 24.1},
    {"Pb", 206.975885, 22.1},  {"Pb", 207.976641, 52.4},
    {"Pd", 101.905609, 1.02},  {"Pd", 103.904026, 11.14},
    {"Pd", 104.905075, 22.33}, {"Pd", 105.903475, 27.33},
    {"Pd", 107.903894, 26.46}, {"Pd", 109.905169, 11.72},
    {"Pr", 140.907657, 100.0}, {"Pt", 189.959937, 0.01},
    {"Pt", 191.961049, 0.79},  {"Pt", 193.962679, 32.9},
    {"Pt", 194.964785, 33.8},  {"Pt", 195.964947, 25.3},
    {"Pt", 197.967879, 7.2},   {"Rb", 84.9118, 72.17},
    {"Rb", 86.909184, 27.84},  {"Re", 184.952977, 37.4},
    {"Re", 186.955765, 62.6},  {"Rh", 102.905503, 100.0},
    {"Ru", 95.907596, 5.52},   {"Ru", 97.905287, 1.88},
    {"Ru", 98.905937, 12.7},   {"Ru", 99.904218, 12.6},
    {"Ru", 100.905581, 17.0},  {"Ru", 101.904348, 31.6},
    {"Ru", 103.905422, 18.7},  {"S", 31.972072, 95.02},
    {"S", 32.971459, 0.75},    {"S", 33.967868, 4.21},
    {"S", 35.967079, 0.02},    {"Sb", 120.903824, 57.3},
    {"Sb", 122.904222, 42.7},  {"Sc", 44.955914, 100.0},
    {"Se", 73.922477, 0.9},    {"Se", 75.919207, 9.0},
    {"Se", 76.919908, 7.6},    {"Se", 77.917304, 23.5},
    {"Se", 79.916521, 49.6},   {"Se", 81.916709, 9.4},
    {"Si", 27.976928, 92.23},  {"Si", 28.976496, 4.67},
    {"Si", 29.973772, 3.1},    {"Sm", 143.912009, 3.1},
    {"Sm", 146.914907, 15.0},  {"Sm", 147.914832, 11.3},
    {"Sm", 148.917193, 13.8},  {"Sm", 149.917285, 7.4},
    {"Sm", 151.919741, 26.7},  {"Sm", 153.922218, 22.7},
    {"Sn", 111.904826, 0.97},  {"Sn", 113.902784, 0.65},
    {"Sn", 114.903348, 0.36},  {"Sn", 115.901744, 14.7},
    {"Sn", 116.902954, 7.7},   {"Sn", 117.901607, 24.3},
    {"Sn", 118.90331, 8.6},    {"Sn", 119.902199, 32.4},
    {"Sn", 121.90344, 4.6},    {"Sn", 123.905271, 5.6},
    {"Sr", 83.913428, 0.56},   {"Sr", 85.909273, 9.86},
    {"Sr", 86.908902, 7.0},    {"Sr", 87.905625, 82.58},
    {"Ta", 179.947489, 0.012}, {"Ta", 180.948014, 99.99},
    {"Tb", 158.92535, 100.0},  {"Te", 119.904021, 0.096},
    {"Te", 121.903055, 2.6},   {"Te", 122.904278, 0.91},
    {"Te", 123.902825, 4.82},  {"Te", 124.904435, 7.14},
    {"Te", 125.90331, 18.95},  {"Te", 127.904464, 31.69},
    {"Te", 129.906229, 33.8},  {"Th", 232.038054, 100.0},
    {"Ti", 45.952633, 8.0},    {"Ti", 46.951765, 7.3},
    {"Ti", 47.947947, 73.8},   {"Ti", 48.947871, 5.5},
    {"Ti", 49.944786, 5.4},    {"Tl", 202.972336, 29.52},
    {"Tl", 204.97441, 70.48},  {"Tm", 168.934225, 100.0},
    {"U", 234.040947, 0.006},  {"U", 235.043925, 0.72},
    {"U", 238.050786, 99.27},  {"V", 49.947161, 0.25},
    {"V", 50.943963, 99.75},   {"W", 179.946727, 0.13},
    {"W", 181.948225, 26.3},   {"W", 182.950245, 14.3},
    {"W", 183.950953, 30.67},  {"W", 185.954377, 28.6},
    {"Xe", 123.905894, 0.1},   {"Xe", 125.904281, 0.09},
    {"Xe", 127.903531, 1.91},  {"Xe", 128.90478, 26.4},
    {"Xe", 129.90351, 4.1},    {"Xe", 130.905076, 21.2},
    {"Xe", 131.904148, 26.9},  {"Xe", 133.905395, 10.4},
    {"Xe", 135.907219, 8.9},   {"Y", 88.905856, 100.0},
    {"Yb", 167.933908, 0.13},  {"Yb", 169.934774, 3.05},
    {"Yb", 170.936338, 14.3},  {"Yb", 171.936393, 21.9},
    {"Yb", 172.938222, 16.12}, {"Yb", 173.938873, 31.8},
    {"Yb", 175.942576, 12.7},  {"Zn", 63.929145, 48.6},
    {"Zn", 65.926035, 27.9},   {"Zn", 66.927129, 4.1},
    {"Zn", 67.924846, 18.8},   {"Zn", 69.925325, 0.6},
    {"Zr", 89.904708, 51.45},  {"Zr", 90.905644, 11.27},
    {"Zr", 91.905039, 17.17},  {"Zr", 93.906319, 17.33},
    {"Zr", 95.908272, 2.78}};

/// Find the range of entries in the isotope table that
/// belong to one particular element.
///
/// @param[in] element - a chemical symbol
/// @return a pair of iterators into the table
std::pair<std::vector<Isotope>::const_iterator,
          std::vector<Isotope>::const_iterator>
get_isotope_range(const std::string& element) {
    if (std::find(elements.begin(), elements.end(), element) == elements.end())
        throw value_error("unknown chemical symbol");
    auto reference = Isotope{element, 0., 0.};
    auto range = std::equal_range(isotopes.begin(),
                                  isotopes.end(),
                                  reference,
                                  [](const Isotope& i1, const Isotope& i2) {
                                      return i1.element < i2.element;
                                  });

    if (range.first == range.second)
        throw value_error("unknown isotopic distribution");
    return range;
}


double get_mass(const std::string& element, bool disablev) {
    try {
        auto range = get_isotope_range(element);
        return std::accumulate(
            range.first,
            range.second,
            0.,
            [](const double& previous, const Isotope& current) {
                return previous + current.mass * current.percentage / 100.;
            });
    }
    catch (value_error& e) {
        // If we are not allowed to parse virtual elements,
        // re-throw the exception.
        if (disablev)
            throw;
        // Otherwise, try parsing the string.
        auto v = Virtual_element(element);
        auto nelements = v.get_nelements();
        double nruter{0.};

        for (decltype(nelements) i = 0; i < nelements; ++i)
            nruter += v.get_ratio(i) * get_mass(v.get_symbol(i), true);
        return nruter;
    }
}


double get_gfactor(const std::string& element, bool disablev) {
    auto mean = get_mass(element, disablev);
    double nruter{0.};

    try {
        auto range = get_isotope_range(element);

        for (auto i = range.first; i < range.second; ++i)
            nruter +=
                (i->mass - mean) * (i->mass - mean) * i->percentage / 100.;
    }
    catch (value_error& e) {
        if (disablev)
            throw;
        auto v = Virtual_element(element);
        auto nelements = v.get_nelements();

        for (decltype(nelements) j = 0; j < nelements; ++j) {
            auto range = get_isotope_range(v.get_symbol(j));

            for (auto i = range.first; i < range.second; ++i)
                nruter += (i->mass - mean) * (i->mass - mean) * i->percentage /
                          100. * v.get_ratio(j);
        }
    }
    return nruter / mean / mean;
}


Virtual_element::Virtual_element(const std::vector<std::string>& symbols,
                                 const std::vector<double>& ratios) {
    auto nelements = symbols.size();

    if (nelements != ratios.size())
        throw value_error("inconsistent number of elements");
    double total = 0;

    for (decltype(nelements) i = 0; i < nelements; ++i) {
        if (ratios[i] <= 0)
            throw value_error("atomic ratios cannot be negative");
        total += ratios[i];
    }
    // Account for possible duplicates.
    std::map<std::string, double> dict;

    for (decltype(nelements) i = 0; i < nelements; ++i) {
        if (dict.count(symbols[i]) == 0)
            dict[symbols[i]] = ratios[i];
        else
            dict[symbols[i]] += ratios[i];
    }
    std::tie(this->symbols, this->ratios) = split_keys_and_values(dict);
    nelements = this->symbols.size();

    // Normalize the atomic ratios for internal use.
    for (decltype(nelements) i = 0; i < nelements; ++i)
        this->ratios[i] /= total;
}


Virtual_element::Virtual_element(const std::string& composition) {
    std::string tmp(composition);

    // Remove all whitespace from the string
    tmp.erase(
        std::remove_if(
            tmp.begin(),
            tmp.end(),
            [](char c) { return std::isspace(static_cast<unsigned char>(c)); }),
        tmp.end());
    // Split it at semicolons.
    boost::tokenizer<boost::char_separator<char>> tok(
        tmp, boost::char_separator<char>(";", " ", boost::keep_empty_tokens));

    if (tmp.size() == 0)
        throw value_error("invalid virtual element specification");
    // And try to parse each of the substrings into a valid
    // symbol:ratio pair.
    std::map<std::string, double> dict;
    double total{0.};

    for (auto c : tok) {
        if (c.size() == 0)
            throw value_error("invalid virtual element specification");
        std::vector<std::string> fields;
        boost::split(
            fields,
            c,
            [](char c) { return c == ':'; },
            boost::algorithm::token_compress_off);

        if (fields.size() != 2)
            throw value_error("invalid virtual element specification");
        std::string symbol{fields[0]};

        if (std::find(elements.begin(), elements.end(), symbol) ==
            elements.end())
            throw value_error("unknown chemical symbol");
        double ratio{0.};
        try {
            ratio = boost::lexical_cast<double>(fields[1]);
        }
        catch (const boost::bad_lexical_cast& e) {
            throw value_error("invalid atomic ratio");
        }

        if (ratio <= 0.) {
            throw value_error("atomic ratios must be positive");
        }

        if (dict.count(symbol) == 0)
            dict[symbol] = ratio;
        else
            dict[symbol] += ratio;
        total += ratio;
    }
    std::tie(this->symbols, this->ratios) = split_keys_and_values(dict);
    // Normalize the atomic ratios for internal use.
    auto nelements = this->symbols.size();

    for (decltype(nelements) i = 0; i < nelements; ++i)
        this->ratios[i] /= total;
}


std::string Virtual_element::get_name() const {
    auto nelements = this->symbols.size();
    std::string nruter;

    // Our using 8 decimal places may cause a loss of precision for
    // very small concentrations. Then again, the virtual crystal
    // approximation should not be used in such cases.
    for (decltype(nelements) i = 0; i < nelements - 1; ++i)
        nruter += boost::str(boost::format("%s:%.8f;") % this->symbols[i] %
                             this->ratios[i]);
    nruter += boost::str(boost::format("%s:%.8f") % this->symbols.back() %
                         this->ratios.back());
    return nruter;
}
} // namespace alma
