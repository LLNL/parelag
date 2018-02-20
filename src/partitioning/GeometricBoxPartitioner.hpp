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

#ifndef GEOMETRICBOXPARTITIONER_HPP_
#define GEOMETRICBOXPARTITIONER_HPP_

#include <mfem.hpp>

#include "topology/Topology.hpp"

namespace parelag
{

/**
   Interface is supposed to roughly match MetisGraphPartitioner
*/
class GeometricBoxPartitioner
{
public:
    GeometricBoxPartitioner() {}
    ~GeometricBoxPartitioner() {}
    

    void doPartition(mfem::Mesh & mesh,
                     int & num_partitions,
                     mfem::Array<int> & partitioning);

    const mfem::Array<int>& GetPartitionSizes() const {return partition_sizes_;}

private:
    mfem::Array<int> partition_sizes_;
};

}

#endif
