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

all:
	cd src; make;
	
debug:
	cd src; make debug;
	
doc:
	doxygen Doxyfile
	
clean:
	cd src; make clean;
	cd examples; make clean;
	cd testsuite; make clean;
	
deps-install:
	cd dependencies; make deps-install;
	
deps-clean:
	cd dependencies; make deps-clean;
