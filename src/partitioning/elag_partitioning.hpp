/*
  Copyright (c) 2018, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-745557. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#ifndef ELAG_PARTITIONING_HPP_
#define ELAG_PARTITIONING_HPP_

#include "CartesianPartitioner.hpp"
#include "LogicalPartitioner.hpp"
#include "MetisGraphPartitioner.hpp"
#include "MFEMRefinedMeshPartitioner.hpp"
#include "GeometricBoxPartitioner.hpp"

#ifdef ParELAG_ENABLE_PARMETIS
#include "ParmetisGraphPartitioner.hpp"
#endif /* ParELAG_ENABLE_PARMETIS */

#endif /* ELAG_PARTITIONING_HPP_ */
