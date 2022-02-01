# Build all the necessary executables.
add_executable(metatest metatest.cpp)
target_link_libraries(metatest gtest gtest_main)

add_executable(poscar_test poscar_test.cpp)
target_link_libraries(poscar_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(force_constants_test force_constants_test.cpp)
target_link_libraries(force_constants_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(born_test born_test.cpp)
target_link_libraries(born_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(mass_test mass_test.cpp)
target_link_libraries(mass_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(vc_test vc_test.cpp)
target_link_libraries(vc_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(dynamical_matrix_test dynamical_matrix_test.cpp)
target_link_libraries(dynamical_matrix_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(symmetry_test symmetry_test.cpp)
target_link_libraries(symmetry_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(mpi_test mpi_test.cpp)
target_link_libraries(mpi_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(qpoint_grid_test qpoint_grid_test.cpp)
target_link_libraries(qpoint_grid_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(cv_test cv_test.cpp)
target_link_libraries(cv_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(kappa_sg_test kappa_sg_test.cpp)
target_link_libraries(kappa_sg_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(processes_test processes_test.cpp)
target_link_libraries(processes_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(p3_test p3_test.cpp)
target_link_libraries(p3_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(force_constants_3rd_test force_constants_3rd_test.cpp)
target_link_libraries(force_constants_3rd_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(conductivity_test conductivity_test.cpp)
target_link_libraries(conductivity_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(beyondRTA_test beyondRTA_test.cpp)
target_link_libraries(beyondRTA_test alma_framework gtest gtest_main
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg Threads::Threads tbb)

add_executable(proc_serialization_test proc_serialization_test.cpp)
target_link_libraries(proc_serialization_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg Threads::Threads tbb)

add_executable(hdf5_write_test hdf5_write_test.cpp)
target_link_libraries(hdf5_write_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(hdf5_scattering_test hdf5_scattering_test.cpp)
target_link_libraries(hdf5_scattering_test alma_framework gtest
  gtest_main ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES} libsymspg
  ${HDF5_LIBS} Threads::Threads tbb)

add_executable(hdf5_read_test hdf5_read_test.cpp)
target_link_libraries(hdf5_read_test alma_framework gtest gtest_main
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(isotopic_scattering_test isotopic_scattering_test.cpp)
target_link_libraries(isotopic_scattering_test alma_framework gtest
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(analytic1D_basicprop_test analytic1D_basicprop_test.cpp)
target_link_libraries(analytic1D_basicprop_test alma_framework gtest 
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(analytic1D_psi_test analytic1D_psi_test.cpp)
target_link_libraries(analytic1D_psi_test alma_framework gtest 
  ${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
  libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(analytic1D_SPR-FL_test analytic1D_SPR-FL_test.cpp)
target_link_libraries(analytic1D_SPR-FL_test alma_framework gtest
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(analytic1D_SPR_test analytic1D_SPR_test.cpp)
target_link_libraries(analytic1D_SPR_test alma_framework gtest 
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(analytic1D_MSD_test analytic1D_MSD_test.cpp)
target_link_libraries(analytic1D_MSD_test alma_framework gtest
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)

add_executable(kappa_Tsweep_test kappa_Tsweep_test.cpp)
target_link_libraries(kappa_Tsweep_test alma_framework gtest
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)
add_dependencies(kappa_Tsweep_test kappa_Tsweep)

add_executable(cumulativecurves_test cumulativecurves_test.cpp)
target_link_libraries(cumulativecurves_test alma_framework gtest
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)
add_dependencies(cumulativecurves_test cumulativecurves)

add_executable(steady_montecarlo1d_test steady_montecarlo1d_test.cpp)
target_link_libraries(steady_montecarlo1d_test alma_framework gtest
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)
add_dependencies(steady_montecarlo1d_test steady_montecarlo1d)

add_executable(transient_analytic1d_test transient_analytic1d_test.cpp)
target_link_libraries(transient_analytic1d_test alma_framework gtest
${MPI_CXX_LIBRARIES} ${Boost_LIBRARIES}
libsymspg ${HDF5_LIBS} Threads::Threads tbb)
add_dependencies(transient_analytic1d_test transient_analytic1d)

# Define all tests.
add_test(NAME metatest COMMAND metatest)
add_test(NAME poscar_test COMMAND poscar_test)
add_test(NAME force_constants_test COMMAND force_constants_test)
add_test(NAME born_test COMMAND born_test)
add_test(NAME mass_test COMMAND mass_test)
add_test(NAME vc_test COMMAND vc_test)
add_test(NAME dynamical_matrix_test COMMAND dynamical_matrix_test)
add_test(NAME symmetry_test COMMAND symmetry_test)
add_test(NAME mpi_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 4 $<TARGET_FILE:symmetry_test>)
add_test(NAME qpoint_grid_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:qpoint_grid_test>)
add_test(NAME cv_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:cv_test>)
add_test(NAME kappa_sg_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:kappa_sg_test>)
add_test(NAME processes_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:processes_test>)
add_test(NAME p3_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:p3_test>)
add_test(NAME force_constants_3rd_test COMMAND force_constants_3rd_test)
add_test(NAME conductivity_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:conductivity_test>)
add_test(NAME beyondRTA_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:beyondRTA_test>)
add_test(NAME proc_serialization_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:proc_serialization_test>)
add_test(NAME hdf5_write_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:hdf5_write_test>)
add_test(NAME hdf5_scattering_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:hdf5_scattering_test>)
add_test(NAME hdf5_read_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:hdf5_read_test>)
add_test(NAME isotopic_scattering_test COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} 2 $<TARGET_FILE:isotopic_scattering_test>)
add_test(NAME analytic1D_basicprop_test COMMAND analytic1D_basicprop_test)
add_test(NAME analytic1D_psi_test COMMAND analytic1D_psi_test)
add_test(NAME analytic1D_SPR-FL_test COMMAND analytic1D_SPR-FL_test)
add_test(NAME analytic1D_SPR_test COMMAND analytic1D_SPR_test)
add_test(NAME analytic1D_MSD_test COMMAND analytic1D_MSD_test)
add_test(NAME kappa_Tsweep_test COMMAND kappa_Tsweep_test)
add_test(NAME cumulativecurves_test COMMAND cumulativecurves_test)
add_test(NAME transient_analytic1d_test COMMAND transient_analytic1d_test)
add_test(NAME steady_montecarlo1d_test COMMAND steady_montecarlo1d_test)
