# Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
# Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the ParElag library. For more information and source code
# availability see http://github.com/LLNL/parelag.
#
# ParElag is free software; you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License (as published by the Free
# Software Foundation) version 2.1 dated February 1999.

include ../Makefile.in

# Include $(MFEM_DIR)/config.mk unless some of the $(SKIP_INCLUDE_TARGETS) are given
SKIP_INCLUDE_TARGETS = clean clean-lib doc
HAVE_SKIP_INCLUDE_TARGET = $(filter $(SKIP_INCLUDE_TARGETS),$(MAKECMDGOALS))
ifeq (,$(HAVE_SKIP_INCLUDE_TARGET))
	-include $(MFEM_DIR)/config.mk

	ifneq ($(MFEM_USE_MPI), YES)
		$(error Please rebuild MFEM with option MFEM_USE_MPI=YES)
	endif

	ifneq ($(MFEM_USE_METIS_5), YES)
		$(error Please rebuild MFEM with option MFEM_USE_METIS_5=YES)
	endif

	ifneq ($(MFEM_USE_LAPACK), YES)
		$(error Please rebuild MFEM with option MFEM_USE_LAPACK=YES)
	endif

	ifneq ($(MFEM_USE_SUITESPARSE), YES)
		$(error Please rebuild MFEM with option MFEM_USE_SUITESPARSE=YES)
	endif
else
   # Do not allow skip-include targets to be combined with other targets
   ifneq (1,$(words $(MAKECMDGOALS)))
      $(error Target '$(firstword $(HAVE_SKIP_INCLUDE_TARGET))' can not be\
      combined with other targets)
   endif
endif

INCLUDE_DIRS=-I$(MFEM_INC_DIR) -I$(HYPRE_DIR)/include -I$(METIS_DIR)/include -I$(SUITESPARSE_DIR)/include
USE_OPENMP_NO  =
USE_OPENMP_YES = -fopenmp -DELAG_USE_OPENMP
USE_OPENMP_DEF = $(USE_OPENMP_$(USE_OPENMP))

CFLAGS=-O3 -Wall $(EXTRA_CFLAGS)
CFLAGS_DBG=-O0 -ggdb -Wall -DELAG_DEBUG $(EXTRA_CFLAGS)

FFOPTS = -O2 -Wall
FFOPTS_DBG =  -O0 -ggdb -Wall

MYDEF = $(USE_OPENMP_DEF)

DIRS = partitioning topology amge linalg utilities structures hypreExtension ma57d
SOURCE_FILES = $(foreach dir,$(DIRS),$(wildcard $(dir)/*.cpp))
OBJECT_FILES = $(SOURCE_FILES:.cpp=.o)

FORTRAN_FILES = $(foreach dir,$(DIRS),$(wildcard $(dir)/*.f))
FORTRAN_OBJECTS = $(FORTRAN_FILES:.f=.o)

C_FILES = $(foreach dir,$(DIRS),$(wildcard $(dir)/*.c))
C_OBJECTS = $(C_FILES:.c=.o)

LIB_FILE = libElag.a

all: lib

debug:
	make 'CFLAGS=$(CFLAGS_DBG)' 'FFOPTS=$(FFOPTS_DBG)' MFEM_DIR=$(MFEM_DIR_DBG)

obj: $(OBJECT_FILES) $(FORTRAN_OBJECTS) $(C_OBJECTS)

lib: $(OBJECT_FILES) $(FORTRAN_OBJECTS) $(C_OBJECTS)
	rm -f  $(LIB_FILE)
	ar cruv $(LIB_FILE) $^
	ranlib $(LIB_FILE)

%.o: %.cpp
	cd $(<D); $(CXX) $(CFLAGS) $(MYDEF) -c $(<F) $(INCLUDE_DIRS)
.f.o:
	cd $(<D); $(FF) $(FFOPTS) -c $(<F)
.c.o:
	cd $(<D); $(CC) $(CFLAGS) -c $(<F)  $(INCLUDE_DIRS)

clean:
	rm -f $(OBJECT_FILES)
	
clean-lib: clean
	rm -f $(LIB_FILE)	
	
doc:
	doxygen ../Doxyfile
