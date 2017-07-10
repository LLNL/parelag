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
SKIP_INCLUDE_TARGETS = clean clean-obj clean-exec
HAVE_SKIP_INCLUDE_TARGET = $(filter $(SKIP_INCLUDE_TARGETS),$(MAKECMDGOALS))
ifeq (,$(HAVE_SKIP_INCLUDE_TARGET))
	-include $(MFEM_DIR)/config.mk
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

MYDEFS = $(USE_OPENMP_DEF)
LIBS= -L../src/ -lElag -L$(MFEM_LIB_DIR) -lmfem  -L$(HYPRE_DIR)/lib/ -lHYPRE -L$(METIS_DIR)/lib -lmetis -L$(METIS_DIR)/lib -lparmetis $(SUITESPARSE_LIBS) -lm $(LAPACK_LIBS)

SOURCE_FILES = $(wildcard *.cpp)
OBJECT_FILES = $(SOURCE_FILES:.cpp=.o)
EXEC_FILES = $(SOURCE_FILES:.cpp=.exe)

all : $(EXEC_FILES)

debug:
	make 'CFLAGS=$(CFLAGS_DBG)' MFEM_DIR=$(MFEM_DIR_DBG)

%.exe: %.o
	$(CXX) $(CFLAGS) $(MYDEFS) $^ -o $@ $(LIBS)

%.o: %.cpp
	$(CXX) $(CFLAGS) $(MYDEFS) -c $(INCLUDE_DIRS) $^ 


clean-obj:
	rm -f *.o

clean-exec:
	rm -f *.exe

clean: clean-obj clean-exec
