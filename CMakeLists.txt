# Standard boilerplate.
cmake_minimum_required(VERSION 3.1.0 FATAL_ERROR)
project(ALMA CXX C)

# Assume C++17 for all C++ source,
# and require compiler support.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Try to find a threading library. Superfluous in most systems,
# but lack of this seems to lead to errors in some cases.
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Find the Intel Threading Building Blocks.
find_package(TBB COMPONENTS tbbmalloc REQUIRED)

# Find msgpack, which is now required in order to compile
find_path(MSGPACK_INCLUDE_DIRS NAMES msgpack.hpp REQUIRED)

# Try to associate an HDF5 type to std::size_t.
include(CheckTypeSize)
set(CMAKE_EXTRA_INCLUDE_FILES "cstddef")
check_type_size("std::size_t" ALMA_SIZEOF_SIZE_T LANGUAGE CXX)
set(CMAKE_EXTRA_INCLUDE_FILES)
if ("${ALMA_SIZEOF_SIZE_T}" STREQUAL "")
  message(FATAL_ERROR "Cannot evaluate the size of std::size_t")
endif()
check_type_size("unsigned short" ALMA_SIZEOF_USHORT LANGUAGE CXX)
check_type_size("unsigned int" ALMA_SIZEOF_UINT LANGUAGE CXX)
check_type_size("unsigned long" ALMA_SIZEOF_ULONG LANGUAGE CXX)
check_type_size("unsigned long long" ALMA_SIZEOF_ULLONG LANGUAGE CXX)
if (${ALMA_SIZEOF_SIZE_T} EQUAL ${ALMA_SIZEOF_USHORT})
  set(NATIVE_SIZE_T "NATIVE_USHORT")
  message("std::size_t is unsigned short")
elseif (${ALMA_SIZEOF_SIZE_T} EQUAL ${ALMA_SIZEOF_UINT})
  set(NATIVE_SIZE_T "NATIVE_UINT")
  message("std::size_t is unsigned int")
elseif (${ALMA_SIZEOF_SIZE_T} EQUAL ${ALMA_SIZEOF_ULONG})
  set(NATIVE_SIZE_T "NATIVE_ULONG")
  message("std::size_t is unsigned long")
elseif (${ALMA_SIZEOF_SIZE_T} EQUAL ${ALMA_SIZEOF_ULLONG})
  set(NATIVE_SIZE_T "NATIVE_ULLONG")
  message("std::size_t is unsigned long long")
else (${ALMA_SIZEOF_SIZE_T} EQUAL ${ALMA_SIZEOF_USHORT})
  message(FATAL_ERROR "Cannot find an HDF5 type for std::size_t")
endif (${ALMA_SIZEOF_SIZE_T} EQUAL ${ALMA_SIZEOF_USHORT})

# Do not use Eigen's parallel algorithms.
add_definitions(-DEIGEN_DONT_PARALLELIZE)

# Detect and set up MPI.
find_package(MPI)
if(NOT MPI_CXX_FOUND)
  message(FATAL_ERROR "No MPI C++ compiler was found.")
endif()
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
string(CONCAT CMAKE_CXX_FLAGS ${MPI_CXX_COMPILE_FLAGS})
string(CONCAT CMAKE_CXX_LINK_FLAGS ${MPI_CXX_LINK_FLAGS})

# Compiler flags (currently only for g++ or clang++).
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  string(CONCAT CMAKE_CXX_FLAGS_RELEASE "-O3 -g -Wall")
  string(CONCAT CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(CONCAT CMAKE_CXX_FLAGS_RELEASE "-O3 -g -Wall -Wno-unused-local-typedef")
  string(CONCAT CMAKE_CXX_FLAGS_RELEASE "-O0 -g -Wall -Wno-unused-local-typedef")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE
      STRING "Choose the type of build." FORCE)
endif()

# Set up some variables so that they can be used in the code.
set(ALMA_VERSION_MAJOR 1)
set(ALMA_VERSION_MINOR 3)
set(BULK_HDF5_MAJOR_VERSION 0)
set(BULK_HDF5_MINOR_VERSION 2)
set(TEST_RESOURCE_DIR "${PROJECT_SOURCE_DIR}/test_resources")


# Tell the compiler where to find the headers.
include_directories(BEFORE "${PROJECT_SOURCE_DIR}/external/eigen")
include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_BINARY_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/external/pcg-cpp-0.98/include")
include_directories("${PROJECT_SOURCE_DIR}/external/base-n/include")

# Find some components from Boost.
find_package(Boost COMPONENTS system filesystem mpi serialization log REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})
add_definitions(-DBOOST_LOG_DYN_LINK)
set(Boost_USE_MULTITHREADED OFF)

# Check for Boost 1.58.0, which contains a bug that prevents compilation.
if(Boost_MAJOR_VERSION EQUAL 1 AND Boost_MINOR_VERSION EQUAL 58)
  message(FATAL_ERROR "Boost 1.58.0 found.
This particular version is not compatible with ALMA due to a bug. See:
http://stackoverflow.com/a/30437359/85371
https://svn.boost.org/trac/boost/ticket/11342
Please use an older or a newer version of Boost.
")
endif()

# Try to find hdf5 in the system.
find_package(HDF5 COMPONENTS C CXX REQUIRED)
add_definitions(${HDF5_DEFINITIONS})
include_directories(${HDF5_INCLUDE_DIRS})
link_directories(${HDF5_LIBRARY_DIRS})
set(HDF5_LIBS ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES})

# Instruct CMake to compile spglib as an external project.
include(ExternalProject)
# For some reason the directory where spglib is extracted gets
# renamed on some hosts, leading to a different path for the
# configure script.
set(SPGLIB_CONFIGURE_COMMAND1 configure)
set(SPGLIB_CONFIGURE_COMMAND2 spglib-1.9.4/configure)
ExternalProject_Add(
  spglib
  URL file://${PROJECT_SOURCE_DIR}/external/spglib-1.9.4.tar.gz
  PREFIX external/spglib
  CONFIGURE_COMMAND <SOURCE_DIR>/${SPGLIB_CONFIGURE_COMMAND1} --disable-shared --with-pic --enable-static --prefix=<INSTALL_DIR> || <SOURCE_DIR>/${SPGLIB_CONFIGURE_COMMAND2}  ${SPGLIB_CONFIGURE_OPTIONS} --disable-shared --enable-static --prefix=<INSTALL_DIR>
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  BUILD_IN_SOURCE 1
  )
# Make sure that the spglib headers and libraries are available.
ExternalProject_Get_Property(spglib install_dir)
include_directories(${install_dir}/include)
link_directories(${install_dir}/lib)
add_library(libsymspg STATIC IMPORTED)
set_property(TARGET libsymspg PROPERTY IMPORTED_LOCATION ${install_dir}/lib/libsymspg.a)
add_dependencies(libsymspg spglib)

# Add the main source directory.
add_subdirectory(src)

# Set up Google Test and Google Mock.
add_subdirectory(external/googletest-release-1.10.0)
enable_testing()
include_directories(external/googletest-release-1.10.0/googletest/include)
include_directories(external/googletest-release-1.10.0/googlemock/include)

# Add the directory containing the tests.
add_subdirectory(test)

# Generate the configuration with Doxygen if possible.
find_package(Doxygen)
if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile @ONLY)
  file(INSTALL ${CMAKE_CURRENT_SOURCE_DIR}/doc/logo_for_doxygen.png
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/doc)
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc
    COMMENT "Generating documentation with Doxygen" VERBATIM
    )
else()
  message(WARNING "Doxygen not found, documentation cannot be generated")
endif()

# Write out a header file with all information needed at compile time
configure_file (
  "${PROJECT_SOURCE_DIR}/include/cmakevars.hpp.in"
  "${PROJECT_BINARY_DIR}/include/cmakevars.hpp"
  )
