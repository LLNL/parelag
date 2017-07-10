/*
  Copyright (c) 2015, Lawrence Livermore National Security, LLC. Produced at the
  Lawrence Livermore National Laboratory. LLNL-CODE-669695. All Rights reserved.
  See file COPYRIGHT for details.

  This file is part of the ParElag library. For more information and source code
  availability see http://github.com/LLNL/parelag.

  ParElag is free software; you can redistribute it and/or modify it under the
  terms of the GNU Lesser General Public License (as published by the Free
  Software Foundation) version 2.1 dated February 1999.
*/

#ifndef ELAG_AMGE_HPP_
#define ELAG_AMGE_HPP_

#include <mfem.hpp>
using namespace mfem;

#include "../utilities/elag_utilities.hpp"
#include "../structures/elag_structures.hpp"
#include "../linalg/elag_linalg.hpp"

#include "../topology/elag_topology.hpp"

#include "bilinIntegrators.hpp"
#include "Coefficient.hpp"
#include "DofHandler.hpp"
#include "DOFAgglomeration.hpp"
#include "ElementalMatricesContainier.hpp"
#include "CochainProjector.hpp"
#include "DeRhamSequence.hpp"
#include "DeRhamSequenceFE.hpp"


#endif /* ELAG_AMGE_HPP_ */
