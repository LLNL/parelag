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

find_package(Doxygen)

if (DOXYGEN_FOUND)

  # Setup Dot support
  set(DOXYGEN_HAVE_DOT "NO")
  if (DOXYGEN_DOT_FOUND)
    set(DOXYGEN_HAVE_DOT "YES")
    get_filename_component(DOXYGEN_DOT_PATH ${DOXYGEN_DOT_EXECUTABLE} DIRECTORY)
  endif()

  set(DOXYGEN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/documentation/doxygen)
  
  # Setup the doxyfile
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_BINARY_DIR}/Doxyfile @ONLY)

  # Apparently the Makefile generator doesn't do this properly
  add_custom_target(make_doxygen_dir
    COMMAND ${CMAKE_COMMAND} -E make_directory ${DOXYGEN_OUTPUT_DIR}
    COMMENT "Creating the doxygen output directory."
    VERBATIM)
  
  
  # These are not just called "doc" so that we may one day support
  # many different in-source documentation platforms. This is probably
  # overly optimistic.
  add_custom_target(doxygen
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
    BYPRODUCTS ${DOXYGEN_OUTPUT_DIR}/html/index.html
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating Doxygen documentation as ${DOXYGEN_OUTPUT_DIR}/html/index.html"
    VERBATIM)
  add_dependencies(doxygen make_doxygen_dir)
  
  add_custom_target(clean-doxygen
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOXYGEN_OUTPUT_DIR}
    COMMENT "Removing Doxygen documentation."
    VERBATIM)

  # Add shortcuts that will, in the future, build all types of
  # documentation (e.g. Doxygen, Sphinx, the next great thing in code
  # documentation, etc...)
  add_custom_target(doc)
  add_dependencies(doc doxygen)

  add_custom_target(clean-doc)
  add_dependencies(clean-doc clean-doxygen)
endif()
