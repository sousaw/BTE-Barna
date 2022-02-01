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
/// Test if boost::mpi works.

#include <boost/mpi.hpp>
#include "gtest/gtest.h"

TEST(mpi_test_case, mpi_test) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    // We need more than one process to perform a proper MPI test.
    ASSERT_GT(world.size(), 1);

    auto myvalue = world.rank();
    std::vector<decltype(myvalue)> allvalues;

    boost::mpi::all_gather(world, myvalue, allvalues);

    if (myvalue == 0) {
        for (auto i = 0; i < world.size(); ++i) {
            EXPECT_EQ(i, allvalues[i]);
        }
    }
}
