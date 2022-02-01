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
/// Test the code that computes average masses and g factors.

#include <iostream>
#include "gtest/gtest.h"
#include <cmakevars.hpp>
#include <utilities.hpp>
#include <periodic_table.hpp>

TEST(mass_test_case, resolves_valid_symbols) {
    EXPECT_EQ(1, alma::symbol_to_Z("H"));
    EXPECT_EQ(28, alma::symbol_to_Z("Ni"));
    EXPECT_EQ(109, alma::symbol_to_Z("Mt"));
}

TEST(mass_test_case, handles_invalid_symbols) {
    EXPECT_THROW(alma::symbol_to_Z("Bar"), alma::value_error);
}

TEST(mass_test_case, resolves_valid_Zs) {
    EXPECT_EQ("H", alma::Z_to_symbol(1));
    EXPECT_EQ("Ni", alma::Z_to_symbol(28));
    EXPECT_EQ("Mt", alma::Z_to_symbol(109));
}

TEST(mass_test_case, handles_invalid_Zs) {
    EXPECT_THROW(alma::Z_to_symbol(-1), alma::value_error);
    EXPECT_THROW(alma::Z_to_symbol(0), alma::value_error);
    EXPECT_THROW(alma::Z_to_symbol(200), alma::value_error);
}

TEST(mass_test_case, gets_mass) {
    EXPECT_DOUBLE_EQ(28.085509989600006, alma::get_mass("Si"));
    EXPECT_DOUBLE_EQ(15.998985191780001, alma::get_mass("O"));
}

TEST(mass_test_case, get_mass_throws) {
    EXPECT_THROW(alma::get_mass("Bar"), alma::value_error);
    EXPECT_THROW(alma::get_mass("Mt"), alma::value_error);
}

TEST(mass_test_case, gets_gfactor) {
    EXPECT_DOUBLE_EQ(0.00020091202549504095, alma::get_gfactor("Si"));
    EXPECT_DOUBLE_EQ(3.280895878894621e-05, alma::get_gfactor("O"));
}

TEST(mass_test_case, get_gfactor_throws) {
    EXPECT_THROW(alma::get_gfactor("Bar"), alma::value_error);
    EXPECT_THROW(alma::get_gfactor("Mt"), alma::value_error);
}

TEST(mass_test_case, parses_valid_virtual_atoms) {
    auto element1 = alma::Virtual_element("Si:1");

    EXPECT_EQ(1u, element1.get_nelements());
    EXPECT_EQ("Si", element1.get_symbol(0));
    EXPECT_DOUBLE_EQ(1.0, element1.get_ratio(0));
    auto element2 = alma::Virtual_element("Si: 0.1e1");
    EXPECT_EQ(1u, element2.get_nelements());
    EXPECT_EQ("Si", element2.get_symbol(0));
    EXPECT_DOUBLE_EQ(1.0, element2.get_ratio(0));
    auto element3 = alma::Virtual_element("Si: 0.1e1; Ge: 10.0e-1");
    EXPECT_EQ(2u, element3.get_nelements());
    EXPECT_EQ("Ge", element3.get_symbol(0));
    EXPECT_DOUBLE_EQ(0.5, element3.get_ratio(0));
    EXPECT_EQ("Si", element3.get_symbol(1));
    EXPECT_DOUBLE_EQ(0.5, element3.get_ratio(1));
    auto element4 = alma::Virtual_element({"Si"}, {1.0});
    EXPECT_EQ(1u, element1.get_nelements());
    EXPECT_EQ("Si", element1.get_symbol(0));
    EXPECT_DOUBLE_EQ(1.0, element1.get_ratio(0));
    auto element5 = alma::Virtual_element({"Si", "Si"}, {1.0, 1.0});
    EXPECT_EQ(1u, element1.get_nelements());
    EXPECT_EQ("Si", element1.get_symbol(0));
    EXPECT_DOUBLE_EQ(1.0, element1.get_ratio(0));
    auto element6 = alma::Virtual_element({"Si", "Ge"}, {1.0, 1.0});
    EXPECT_EQ(2u, element6.get_nelements());
    EXPECT_EQ("Ge", element6.get_symbol(0));
    EXPECT_DOUBLE_EQ(0.5, element6.get_ratio(0));
    EXPECT_EQ("Si", element6.get_symbol(1));
    EXPECT_DOUBLE_EQ(0.5, element6.get_ratio(1));
    auto element7 = alma::Virtual_element({"Si", "Ge", "Si"}, {0.7, 1.0, 0.3});
    EXPECT_EQ(2u, element7.get_nelements());
    EXPECT_EQ("Ge", element7.get_symbol(0));
    EXPECT_DOUBLE_EQ(0.5, element7.get_ratio(0));
    EXPECT_EQ("Si", element7.get_symbol(1));
    EXPECT_DOUBLE_EQ(0.5, element7.get_ratio(1));
}

TEST(mass_test_case, handles_invalid_virtual_atoms) {
    EXPECT_THROW(alma::Virtual_element(";"), alma::value_error);
    EXPECT_THROW(alma::Virtual_element("Si"), alma::value_error);
    EXPECT_THROW(alma::Virtual_element("Si:1.0;"), alma::value_error);
    EXPECT_THROW(alma::Virtual_element("SI:1.0"), alma::value_error);
    EXPECT_THROW(alma::Virtual_element("Si:-1.0"), alma::value_error);
    EXPECT_THROW(alma::Virtual_element("Si:1/2"), alma::value_error);
    EXPECT_THROW(alma::Virtual_element("Si:1.:3."), alma::value_error);
    EXPECT_THROW(alma::Virtual_element("Si:1.;;"), alma::value_error);
}

TEST(mass_test_case, gets_virtual_mass) {
    EXPECT_DOUBLE_EQ(.5 * (alma::get_mass("Si") + alma::get_mass("O")),
                     alma::get_mass("Si:0.5;O:0.5"));
    EXPECT_DOUBLE_EQ(.5 * (alma::get_mass("Si") + alma::get_mass("O")),
                     alma::get_mass("Si:3;O:3"));
}

TEST(mass_test_case, gets_virtual_gfactor) {
    EXPECT_TRUE(
        alma::almost_equal(0.0753347, alma::get_gfactor("Si:0.5;O:0.5")));
}

TEST(mass_test_case, virtual_mass_and_gfactor_throw) {
    EXPECT_THROW(alma::get_mass("Si:0.5;O:0.5", true), alma::value_error);
    EXPECT_THROW(alma::get_gfactor("Si:0.5;O:0.5", true), alma::value_error);
}

TEST(mass_test_case, virtual_element_names_work) {
    EXPECT_EQ("Si:1.00000000", alma::Virtual_element("Si:1.0").get_name());
    EXPECT_EQ("Ge:0.50000000;Si:0.50000000",
              alma::Virtual_element("Si:0.5;Ge:0.5").get_name());
}
