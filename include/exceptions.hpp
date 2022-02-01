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
/// Exceptions used in ALMA.

#include <string>
#include <stdexcept>

namespace alma {
/// Base class for all exceptions in ALMA.
class exception : public std::runtime_error {
public:
    /// Basic constructor.
    exception(const std::string& message) : std::runtime_error(message) {
    }
};

/// Exception related to the parameters passed to a function.
class value_error : public exception {
public:
    /// Basic constructor.
    value_error(const std::string& message) : exception(message) {
    }
};

/// Exception related to the contents of an input file.
class input_error : public exception {
public:
    /// Basic constructor.
    input_error(const std::string& message) : exception(message) {
    }
};

/// Exception related to an inconsistent geometric result.
class geometry_error : public exception {
public:
    /// Basic constructor.
    geometry_error(const std::string& message) : exception(message) {
    }
};
} // namespace alma
