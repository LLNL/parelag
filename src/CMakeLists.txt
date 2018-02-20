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

# All the source files for the library
set(SOURCES "")

# All the header files for the library
set(HEADERS "")

#
# Add all sources
#
add_subdirectory(amge)
add_subdirectory(hypreExtension)
add_subdirectory(linalg)
add_subdirectory(partitioning)
add_subdirectory(structures)
add_subdirectory(topology)
add_subdirectory(utilities)
add_subdirectory(SPE10)

# Create the ParELAG library from the collected sources. The headers
# are also included for the convenience of those using IDEs.
add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Add the TPL_LIBRARIES dependency
target_link_libraries(${PROJECT_NAME} PUBLIC ${TPL_LIBRARIES} MPI::MPI_CXX)

# Add the old debug flags to debug builds
target_compile_definitions(${PROJECT_NAME} PUBLIC
  $<$<CONFIG:Debug>:ELAG_DEBUG>)

# This seems like the best, albeit funky to read, way to ensure safely
# adding a compiler option...
target_compile_options(${PROJECT_NAME} PUBLIC
  $<$<CONFIG:Debug>:$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:-O0>>)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include/parelag>
  )


#
# Install target for the library
#
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include/parelag
  )

install(EXPORT ${PROJECT_NAME}Targets
  NAMESPACE parelag::
  DESTINATION lib/cmake/parelag
  )
