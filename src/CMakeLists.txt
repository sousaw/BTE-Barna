# Library that will be used in the executable and in the unit tests.
add_library(alma_framework STATIC structures.cpp
  vasp_io.cpp
  periodic_table.cpp
  dynamical_matrix.cpp
  symmetry.cpp
  qpoint_grid.cpp
  isotopic_scattering.cpp
  processes.cpp
  bulk_hdf5.cpp
  sampling.cpp
  bulk_properties.cpp
  vc.cpp
  analytic1d.cpp
  beyondRTA.cpp
  shengbte_iter.cpp
  collision_operator.cpp
  geometry_2d.cpp
  interpolation.cpp
  nanos.cpp
)

# Define our executables.
add_executable(VCAbuilder VCAbuilder.cpp)
target_link_libraries(VCAbuilder alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(RTAMC2D RTA_MC2D.cpp)
target_link_libraries(RTAMC2D alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(dist_reader dist_reader.cpp)
target_link_libraries(dist_reader alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(beRTAMC2D beyondRTA_MC2d.cpp)
target_link_libraries(beRTAMC2D alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(PropagatorBuilder PropagatorBuilder.cpp)
target_link_libraries(PropagatorBuilder alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(phononinfo phononinfo.cpp)
target_link_libraries(phononinfo alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(kappa_Tsweep kappa_Tsweep.cpp)
target_link_libraries(kappa_Tsweep alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(kappa_Tsweep_nanos kappa_Tsweep_nanos.cpp)
target_link_libraries(kappa_Tsweep_nanos alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(cumulativecurves cumulativecurves.cpp)
target_link_libraries(cumulativecurves alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(transient_analytic1d transient_analytic1d.cpp)
target_link_libraries(transient_analytic1d alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(steady_montecarlo1d steady_montecarlo1d.cpp)
target_link_libraries(steady_montecarlo1d alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(steady_montecarlo1d_powersource steady_montecarlo1d_powersource.cpp)
target_link_libraries(steady_montecarlo1d_powersource alma_framework
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

# These custom commands provide the tests with the right path to some of the executables in ALMA
add_custom_command(TARGET kappa_Tsweep POST_BUILD COMMAND echo const std::string kappa_Tsweep = \"$<TARGET_FILE:kappa_Tsweep>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_kappa_Tsweep.hpp VERBATIM)
add_custom_command(TARGET cumulativecurves POST_BUILD COMMAND echo const std::string cumulativecurves = \"$<TARGET_FILE:cumulativecurves>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_cumulativecurves.hpp VERBATIM)
add_custom_command(TARGET steady_montecarlo1d POST_BUILD COMMAND echo const std::string steady_montecarlo1d = \"$<TARGET_FILE:steady_montecarlo1d>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_steady_montecarlo1d.hpp VERBATIM)
add_custom_command(TARGET transient_analytic1d POST_BUILD COMMAND echo const std::string transient_analytic1d = \"$<TARGET_FILE:transient_analytic1d>\"\; > ${PROJECT_BINARY_DIR}/include/where_is_transient_analytic1d.hpp VERBATIM)
