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

list(APPEND EXAMPLE_SRCS
  3DHcurlWeakScaling.cpp
  3DHdivWeakScaling.cpp
  ElectricPotential.cpp
  ElectricPotential_bfly.cpp
  EmbeddedMeshPartitionerDemo.cpp
  HdivL2SampleGenerator.cpp
  HdivL2ExtensionVisualize.cpp
  LogicalPartitionerDemo.cpp
  ElasticityUpscaling0Form.cpp
  Upscaling0Form.cpp
  Upscaling0FormMFEM.cpp
  Upscaling0FormMFEMonly.cpp
  Upscaling0FormSpectral.cpp
  Upscaling2Form.cpp
  Upscaling2FormAMGe.cpp
  Upscaling2FormSpectralAMGe.cpp
  UpscalingMaxwell.cpp
  )

list(APPEND EXAMPLE_SRCS
  MultigridTest0Form.cpp
  MultigridTest1Form.cpp
  MultigridTest2Form.cpp
  MultigridTestDarcy.cpp
  MultigridTestDarcy_Hybrid.cpp
  MultigridTestSPE10.cpp
  )

if (${PROJECT_NAME}_USE_MARS)
  list(APPEND EXAMPLE_SRCS
    Upscaling0FormMARS.cpp
    )
endif()

add_parelag_executables(EXAMPLE_SRCS)

if (${PROJECT_NAME}_ENABLE_TESTING)

  # Form 2, AMGE
  add_test(form2AMGe Upscaling2FormAMGe.exe --meshfile none --no-visualize)
  set_tests_properties(form2AMGe
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 1.9010e-02 3.9570e-03 \nu energy-like errors: 1.2883e-01 5.7793e-02")

  # Form 2, AMGE, parallel run with 4 processes
  add_mpi_test(par_form2AMGe_np4 4 Upscaling2FormAMGe.exe --meshfile none --no-visualize)
  set_tests_properties(form2AMGe
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 1.9010e-02 3.9570e-03 \nu energy-like errors: 1.2883e-01 5.7793e-02")

  # Form 2, Spectral AMGe
  add_test(form2spectralAMGe
    Upscaling2FormSpectralAMGe.exe --meshfile none --no-visualize)
    set_tests_properties(form2spectralAMGe
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "u l2-like errors: 7.4780e-04 1.3267e-07 \nu energy-like errors: 1.3227e-02 1.6039e-07;\
u l2-like errors: 7.4780e-04 7.2804e-08 \nu energy-like errors: 1.3227e-02 6.6910e-08;\
u l2-like errors: 7.4780e-04 1.6799e-07 \nu energy-like errors: 1.3227e-02 2.8143e-07")

  # Form 0, Spectral
  # the local_targets tests have ridiculously high energy errors, see
  # JIRA ticket ELAG-10
  add_test(local_targets
    Upscaling0FormSpectral.exe --par_ref_levels 2 --meshfile none --no-visualize)
  set_tests_properties(local_targets
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 8\\.6752e-02 2\\.9799e-02 \nu energy-like errors: 6\\.5713e\\+02 4\\.3955e\\+02;u l2-like errors: 4.9853e-02 2.0129e-02 \nu energy-like errors: 5.4201e\\+02 3.4931e\\+02")

  # Another form 0, spectral with different options
  add_test(local_targets_2
    Upscaling0FormSpectral.exe
    --par_ref_levels 2 --spect_tol 0.001 --coarsening_step 2
    --meshfile none --no-visualize)
  set_tests_properties(local_targets_2
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 8\\.6759e-02 \nu energy-like errors: 6\\.5712e\\+02;u l2-like errors: 4.9880e-02 \nu energy-like errors: 5.4200e\\+02")

  # Electric potential test
  add_test(ElectricPotential_bfly
    ElectricPotential_bfly.exe --feorder 0 --upscalingorder 0 --no-visualize)
  set_tests_properties(ElectricPotential_bfly
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 5.5000e-04 2.1996e-04 \np l2-like errors: 9.9192e-03 4.5006e-03 \nu energy-like errors: 2.4379e-04 1.3407e-04;\
u l2-like errors: 5.4999e-04 2.1996e-04 \np l2-like errors: 9.9192e-03 4.5006e-03 \nu energy-like errors: 2.4379e-04 1.3407e-04")

  # LogicalPartitioner test
  add_test(logical_partitioner
    LogicalPartitionerDemo.exe --Nx 12 --Ny 12 --Nz 12 --no-visualize)
  set_tests_properties(logical_partitioner
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 4.3386e-03 4.3386e-03 1.1209e-03 \nu energy-like errors: 5.6634e-02 5.6634e-02 2.8076e-02;u l2-like errors: 6.3801e-04 5.8036e-04 3.5011e-04 \nu energy-like errors: 1.7760e-02 1.6425e-02 1.1493e-02")

  # LogicalPartitioner test with metis
  add_test(logical_partitioner_metis
    LogicalPartitionerDemo.exe --Nx 12 --Ny 12 --Nz 12 --metis_agglomeration
    --no-visualize)
  set_tests_properties(logical_partitioner_metis
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 9.5170e-03 7.7270e-03 1.8229e-03 \nu energy-like errors: 8.6474e-02 7.7053e-02 3.5714e-02")

  # EmbeddedMeshPartitioner test
  add_test(embedded_mesh_partitioner
    EmbeddedMeshPartitionerDemo.exe --mesh none --par_ref_levels 2
    --no-visualize)
  set_tests_properties(embedded_mesh_partitioner
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 9.361[0-9]e-03 2.554[0-9]e-03 \nu energy-like errors: 8.60[0-9]*e-02 4.292[0-9]e-02")

  # test 3DHdivWeakScaling.exe - structured mesh with several refinements
  add_test(3DHdivWeakScaling 3DHdivWeakScaling.exe --nref_parallel 2
    --no-visualize)
  set_tests_properties(3DHdivWeakScaling
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 3.4325e-01 1.2642e-01 \nu energy-like errors: 2.9404e-01 1.3420e-01")

  # 3DHcurlWeakScaling is really not much different from any other test
  add_test(3DHcurlWeakScaling 3DHcurlWeakScaling.exe
    --nref_parallel 2 --no-visualize)
  set_tests_properties(3DHcurlWeakScaling
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "u l2-like errors: 1.6197e-01 3.0947e-02 \nu energy-like errors: 7.0872e-01 2.3455e-01")

  add_test(MG0Form MultigridTest0Form.exe)
  set_tests_properties(MG0Form
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "Final residual norm: 5.92369e-07;Final residual norm: 6.1043e-07")

  add_test(MG1Form MultigridTest1Form.exe)
  set_tests_properties(MG1Form
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "Final residual norm: 1.97197e-06;\
Final residual norm: 5.01914e-07;\
Final residual norm: 1.36916e-06")

  add_test(MG2Form MultigridTest2Form.exe)
  set_tests_properties(MG2Form
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "Final residual norm: 2.61854e-07;\
Final residual norm: 2.26239e-07;\
Final residual norm: 2.54525e-07")

  add_test(MGDarcyHybrid MultigridTestDarcy_Hybrid.exe)
  set_tests_properties(MGDarcyHybrid
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "Final residual norm: 2.86484e-07;\
Final residual norm: 1.95524e-06;\
Final residual norm: 2.09814e-06")

  add_test(MGDarcyWithL2MassHybrid MultigridTestDarcy_Hybrid.exe --L2mass-weight 1.0)
  set_tests_properties(MGDarcyWithL2MassHybrid
    PROPERTIES
    PASS_REGULAR_EXPRESSION
    "Final residual norm: 2.60987e-07;\
Final residual norm: 1.80874e-06;\
Final residual norm: 1.9406e-06")

  if (${PROJECT_NAME}_USE_MARS)
    add_test(0FormMARS Upscaling0FormMARS.exe -nv)
    set_tests_properties(0FormMARS
      PROPERTIES
      PASS_REGULAR_EXPRESSION
      "% || uh - uH ||;2.325457e-03;% || grad ( uh - uH ) ||;4.196067e-02")
  endif()

  # Only run the valgrind tests if Valgrind is available
  if (MEMORYCHECK_COMMAND AND ${PROJECT_NAME}_ENABLE_VALGRIND_TESTS)
    add_valgrind_test(vlocal_targets
      Upscaling0FormSpectral.exe --par_ref_levels 1 --meshfile none
      --no-visualize)
    add_valgrind_test(vform2spectral
      Upscaling2FormSpectralAMGe.exe --par_ref_levels 1 --meshfile none
      --no-visualize)
  endif()

endif (${PROJECT_NAME}_ENABLE_TESTING)
