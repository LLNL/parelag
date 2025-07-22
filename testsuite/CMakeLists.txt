# Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-745557. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the ParElag library. For more information and source code
# availability see http://github.com/LLNL/parelag.
#
# ParElag is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

list(APPEND TEST_SRCS
  coarseningMetis.cpp
  coarseningTopology.cpp 
  coarseSpaces.cpp
  refineTopology.cpp
  topology.cpp
  twentyseven.cpp
  unstructuredDarcy.cpp
  Darcy.cpp
  UpscalingGeneralForm.cpp
  test_GeometricBoxPartitioner.cpp)

if(${PROJECT_NAME}_ENABLE_PETSC)
  list(APPEND TEST_SRCS unstructuredDarcy_PETSc.cpp)
endif()

add_parelag_executables(TEST_SRCS)

# a better solution to the below is to query the Petsc install somehow
set(PETSC_HAS_MUMPS OFF)

if (${PROJECT_NAME}_ENABLE_TESTING)
  # topology test: connected components
  add_test(connected_components twentyseven.exe --partition disconnected
    --no-visualize)
  set_tests_properties(connected_components
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "N_elements =          3")

  # topology test: donut/torus
  add_test(donut twentyseven.exe --partition donut --no-visualize)
  set_tests_properties(donut
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "Element 0 has 1 tunnels.")

  # topology test: void/hole
  add_test(void twentyseven.exe --partition void --no-visualize)
  set_tests_properties(void
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "Element 1 has 1 holes.")

  # topology test: disconnected face
  add_test(discface twentyseven.exe --partition discface --no-visualize)
  set_tests_properties(discface
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "Facet 3 is disconnected.")

  # topology test: hole in face
  add_test(facehole twentyseven.exe --partition facehole --no-visualize)
  set_tests_properties(facehole
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "Facet 3 has 1 holes.")

  # topology test: disconnected edge
  add_test(discedge twentyseven.exe --partition discedge --no-visualize)
  set_tests_properties(discedge
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "Ridge 16 is disconnected.")

  # topology test: boundary edge adjacent to multiple boundary faces
  # (not sure this should be a problem!)
  add_test(connectivity twentyseven.exe --partition tet --no-visualize)
  set_tests_properties(connectivity
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "codim 0 iAE 0 has bad connectivity")

  # topology test: torus-like with only shared edge
  # --check fixes this, but I am not sure it does so for the right
  # reasons/in the right way (ie, it fully de-agglomerates everything)
  add_test(sharededge twentyseven.exe --partition sharededge --no-visualize)
  set_tests_properties(sharededge
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "codim 0 iAE 1 has bad connectivity")

  if(${PROJECT_NAME}_ENABLE_FAILING_TOPOLOGY_TESTS)
    # topology test fails: getting this to pass is one step to fixing
    # our topology problems and clearing ELAG-19
    add_test(sharedvertex twentyseven.exe --partition sharedvertex --no-visualize)
    set_tests_properties(sharedvertex
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "Success.")

    # topology test fails: this is the simplest example of a bad
    # topology that gets past all our checks and corrections, but see
    # also --partition sharedvertex
    add_test(sharedvertex2 twentyseven.exe --partition sv2 --no-visualize)
    set_tests_properties(sharedvertex
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "Success.")
  endif()

  # Upscaling general form -- form 0
  add_test(form0 UpscalingGeneralForm.exe --form 0 --nref_parallel 1 --no-visualize)
  set_tests_properties(form0
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 1.8389e-02 \nu energy-like errors: 2.1485e-01")

  # Upscaling general form -- form 0 -- with Petsc BDDC solver
  if (${PROJECT_NAME}_ENABLE_PETSC)
    add_test(form0petsc UpscalingGeneralForm.exe
      --form 0 --nref_parallel 1 --use-petsc
      --petsc_opts ${PROJECT_SOURCE_DIR}/testsuite/.petsc_rc_test)
    set_tests_properties(form0petsc
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "u l2-like errors: 1.8389e-02 \nu energy-like errors: 2.1485e-01")

    add_mpi_test(par_form0petsc 2 UpscalingGeneralForm.exe
      --form 0 --nref_parallel 1 --use-petsc --unassembled
      --petsc_opts ${PROJECT_SOURCE_DIR}/testsuite/.petsc_rc_testbddc)
    set_tests_properties(par_form0petsc
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "u l2-like errors: 1.8389e-02 \nu energy-like errors: 2.1485e-01")

    add_mpi_test(par_form1petsc 4 UpscalingGeneralForm.exe
      --form 1 --nref_parallel 1 --use-petsc --unassembled
      --petsc_opts ${PROJECT_SOURCE_DIR}/testsuite/.petsc_rc_testbddc)
    set_tests_properties(par_form1petsc
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "u l2-like errors: 3.1436e-02 \nu energy-like errors: 3.2016e-01")

    add_mpi_test(par_form2petsc 4 UpscalingGeneralForm.exe
      --form 2 --nref_parallel 1 --use-petsc --unassembled
      --petsc_opts ${PROJECT_SOURCE_DIR}/testsuite/.petsc_rc_testbddc)
    set_tests_properties(par_form2petsc
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "u l2-like errors: 9.1847e-03 \nu energy-like errors: 1.2515e-01")
  endif()

  # Upscaling general form -- form 1
  add_test(form1 UpscalingGeneralForm.exe
    --form 1 --nref_parallel 1 --no-visualize)
  set_tests_properties(form1
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 3.1436e-02 \nu energy-like errors: 3.2016e-01")

  add_mpi_test(par_form1_np2 2 UpscalingGeneralForm.exe
    --form 1 --nref_parallel 1 --no-visualize)
  set_tests_properties(par_form1_np2
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 3.143[67]e-02 \nu energy-like errors: 3.2016e-01")

  # Upscaling general form -- form 2
  add_test(form2 UpscalingGeneralForm.exe
    --form 2 --nref_parallel 1 --no-visualize)
  set_tests_properties(form2
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 9.1847e-03 \nu energy-like errors: 1.2515e-01")

  # Upscaling general form -- form 1, unstructured
  add_test(unstructured_form1
    UpscalingGeneralForm.exe
    --form 1 --nref_parallel 1 --unstructured --no-visualize)
  set_tests_properties(unstructured_form1
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 4.8096e-02 \nu energy-like errors: 4.1826e-01;u l2-like errors: 7.1756e-02 \nu energy-like errors: 4.3777e-01")

  # Upscaling, geometric partitioning
  add_test(geometric_form1
    UpscalingGeneralForm.exe
    --form 1 --nref_parallel 1 --geometric --no-visualize)
  set_tests_properties(geometric_form1
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 3.1436e-02 \nu energy-like errors: 3.2016e-01")

  # Unstructured Darcy flow
  add_test(unstructuredDarcy
    unstructuredDarcy.exe --nref_parallel 1 --mesh none --no-visualize)
  set_tests_properties(unstructuredDarcy
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 7.3721e-02 \np l2-like errors: 2.8944e-03 \nu energy-like errors: 1.1226e-06;\
u l2-like errors: 7.3721e-02 \np l2-like errors: 2.8944e-03 \nu energy-like errors: 1.1089e-06;\
u l2-like errors: 7.3721e-02 \np l2-like errors: 3.1029e-03 \nu energy-like errors: 9.2277e-07")

  if (${PROJECT_NAME}_ENABLE_PETSC)
    add_mpi_test(unstructuredDarcy_PETSc_fieldsplit 4 unstructuredDarcy_PETSc.exe
      --assembled --nref_parallel 1
      --petsc_opts ${PROJECT_SOURCE_DIR}/testsuite/.petsc_rc_darcy_fieldsplit
      --mesh none)
    set_tests_properties(unstructuredDarcy_PETSc_fieldsplit
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "u l2-like errors: 7.3721e-02 \np l2-like errors: 2.9438e-03 \nu energy-like errors: 1.0143e-06")

    if (PETSC_HAS_MUMPS)
      add_mpi_test(unstructuredDarcy_PETSc_bddc 4 unstructuredDarcy_PETSc.exe
        --unassembled --nref_parallel 1
        --petsc_opts ${PROJECT_SOURCE_DIR}/testsuite/.petsc_rc_darcy_bddc
        --mesh none)
      set_tests_properties(unstructuredDarcy_PETSc_bddc
        PROPERTIES
        PASS_REGULAR_EXPRESSION
        "u l2-like errors: 7.3721e-02 \np l2-like errors: 2.9438e-03 ")
    endif()
  endif()

  # test to use external meshes with UpscalingGeneralForm
  if (ElagMeshes_FOUND)
    add_test(upscaling_outside_mesh
      UpscalingGeneralForm.exe --mesh ${ElagMeshes_DIR}/cube474.mesh3d
      --nref_parallel 1 --nref_serial 0 --rhs-one --no-visualize)
    set_tests_properties(upscaling_outside_mesh
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "u l2-like errors: 4.4287e-08 \nu energy-like errors: 3.3158e-07;\
u l2-like errors: 5.1508e-08 \nu energy-like errors: 3.6821e-07;\
u l2-like errors: 6.8347e-08 \nu energy-like errors: 4.9492e-07")
  endif()

  # Improve topology coarsening (new algorithm for coarse facets)
  add_test(newcoarsefacets
    unstructuredDarcy.exe --nref_parallel 2 --mesh none --no-visualize
    --topo-algo 2 --nref_serial 1)
  set_tests_properties(newcoarsefacets
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "mpi processes: 1"
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 8.5165e-02 4.3735e-02 \np l2-like errors: 1.9534e-04 1.0103e-04 \nu energy-like errors: [1-2].[0-9]*e-06 [1-2].[0-9]*e-06;\
u l2-like errors: 8.6573e-02 4.4377e-02 \np l2-like errors: 1.9933e-04 1.0208e-04 \nu energy-like errors: [1-2].[0-9]*e-06 [1-2].[0-9]*e-06;
u l2-like errors: 8.6902e-02 4.3905e-02 \np l2-like errors: 1.8645e-04 1.0685e-04 \nu energy-like errors: [1-2].[0-9]*e-06 [1-2].[0-9]*e-06")

  add_test(geometric_partitioner
    test_GeometricBoxPartitioner.exe --x-elem 12 --y-elem 16 --partitions 9)
  set_tests_properties(geometric_partitioner
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "mean size: 21.3333")

  # Only run the valgrind tests if Valgrind is available
  if (MEMORYCHECK_COMMAND AND ${PROJECT_NAME}_ENABLE_VALGRIND_TESTS)
    add_valgrind_test(vform0
      UpscalingGeneralForm.exe --form 0 --nref_parallel 1 --no-visualize)
    add_valgrind_test(vunstructuredDarcy
      unstructuredDarcy.exe --nref_parallel 1 --mesh none --no-visualize)
    add_valgrind_test(vnewcoarsefacets
      unstructuredDarcy.exe --nref_parallel 2 --mesh none --no-visualize
      --topo-algo 2 --nref_serial 1)
    add_valgrind_test(vgeometric_partitioner
      test_GeometricBoxPartitioner.exe)
    if (${PROJECT_NAME}_ENABLE_PETSC)
      add_valgrind_test(vform0petsc UpscalingGeneralForm.exe --form 0
        --nref_parallel 1 --use-petsc
        --petsc_opts ${PROJECT_SOURCE_DIR}/testsuite/.petsc_rc_test)
      add_valgrind_test(vunstructuredDarcy_PETSc
        unstructuredDarcy_PETSc.exe
        --petsc_opts ${PROJECT_SOURCE_DIR}/testsuite/.petsc_rc_darcy_fieldsplit
        --nref_parallel 1 --mesh none)
    endif (${PROJECT_NAME}_ENABLE_PETSC)
  endif (MEMORYCHECK_COMMAND AND ${PROJECT_NAME}_ENABLE_VALGRIND_TESTS)
  
endif (${PROJECT_NAME}_ENABLE_TESTING)
