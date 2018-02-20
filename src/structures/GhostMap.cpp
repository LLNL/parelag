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

#include "GhostMap.hpp"

#include "utilities/MemoryUtils.hpp"

namespace parelag
{
using namespace mfem;

GhostMap::GhostMap(const SharingMap & in) :
    Map_(make_unique<SharingMap>(in))
{
}

GhostMap::~GhostMap()
{
}

int GhostMap::AssemblePlus(const Vector & data, Vector & trueData)
{
    return assemble(data,1.0,trueData);
}

int GhostMap::AssembleMinus(const Vector & data, Vector & trueData)
{
    return assemble(data,-1.0,trueData);
}

int GhostMap::assemble(const Vector & data, double offd_values, Vector & trueData)
{
    hypre_VectorData(hypre_ParVectorLocalVector(Map_->xTrue_))
        = trueData.GetData();
    hypre_VectorData(hypre_ParVectorLocalVector(Map_->x_))
        = data.GetData();

    hypre_ParCSRMatrix * map_ent_trueEnt = *(Map_->entity_trueEntity);

    double * diag
        = hypre_CSRMatrixData(
            hypre_ParCSRMatrixDiag(map_ent_trueEnt) );
    double * offd
        = hypre_CSRMatrixData(
            hypre_ParCSRMatrixOffd(map_ent_trueEnt) );

    const int nnz_diag
        = hypre_CSRMatrixNumNonzeros(
            hypre_ParCSRMatrixDiag(map_ent_trueEnt) );
    const int nnz_offd
        = hypre_CSRMatrixNumNonzeros(
            hypre_ParCSRMatrixOffd(map_ent_trueEnt) );

    // Copy data from entity_trueEntity such that we can restore it
    double * orig_diag = new double[nnz_diag]; // ?? +1 ??
    double * orig_offd = new double[nnz_offd]; // ?? +1 ??
    std::copy(diag,diag+nnz_diag,orig_diag);
    std::copy(offd,offd+nnz_offd,orig_offd);

    // Change data
    for(int i(0); i < nnz_diag; ++i)
        diag[i] = 1.0;
    for(int i(0); i < nnz_offd; ++i)
        offd[i] = offd_values;

    int ierr = hypre_ParCSRMatrixMatvecT(1.,
                                         map_ent_trueEnt,
                                         Map_->x_.get(),
                                         0.,
                                         Map_->xTrue_.get());

    hypre_VectorData(hypre_ParVectorLocalVector(Map_->xTrue_)) = nullptr;
    hypre_VectorData(hypre_ParVectorLocalVector(Map_->x_)) = nullptr;

    // Restore original entity_trueEntity data
    std::copy(orig_diag,orig_diag+nnz_diag,diag);
    std::copy(orig_offd,orig_offd+nnz_offd,offd);

    delete[] orig_diag;
    delete[] orig_offd;
    return ierr;
}

int GhostMap::DistributePlus(const Vector & trueData, Vector & data)
{
    return distribute(trueData,1.0,data);
}

int GhostMap::DistributeMinus(const Vector & trueData, Vector & data)
{
    return distribute(trueData,-1.0,data);
}


int GhostMap::distribute(const Vector & trueData, double offd_values, Vector & data)
{
    hypre_VectorData(hypre_ParVectorLocalVector(Map_->xTrue_))
        = trueData.GetData();
    hypre_VectorData(hypre_ParVectorLocalVector(Map_->x_))
        = data.GetData();

    hypre_ParCSRMatrix * map_ent_trueEnt = *(Map_->entity_trueEntity);

    double * diag
        = hypre_CSRMatrixData(hypre_ParCSRMatrixDiag(map_ent_trueEnt));
    double * offd
        = hypre_CSRMatrixData(hypre_ParCSRMatrixOffd(map_ent_trueEnt));

    const int nnz_diag
        = hypre_CSRMatrixNumNonzeros(
            hypre_ParCSRMatrixDiag(map_ent_trueEnt) );
    const int nnz_offd
        = hypre_CSRMatrixNumNonzeros(
            hypre_ParCSRMatrixOffd(map_ent_trueEnt) );

    // Copy data from entity_trueEntity such that we can restore it
    double * orig_diag = new double[nnz_diag]; // ?? +1 ??
    double * orig_offd = new double[nnz_offd]; // ?? +1 ??
    std::copy(diag,diag+nnz_diag,orig_diag);
    std::copy(offd,offd+nnz_offd,orig_offd);

    // Change data
    for(int i(0); i < nnz_diag; ++i)
        diag[i] = 1.0;
    for(int i(0); i < nnz_offd; ++i)
        offd[i] = offd_values;

    int ierr = hypre_ParCSRMatrixMatvec(1.,
                                        map_ent_trueEnt,
                                        Map_->xTrue_.get(),
                                        0.,
                                        Map_->x_.get());

    hypre_VectorData(hypre_ParVectorLocalVector(Map_->xTrue_)) = nullptr;
    hypre_VectorData(hypre_ParVectorLocalVector(Map_->x_))     = nullptr;

    // Restore original entity_trueEntity data
    std::copy(orig_diag,orig_diag+nnz_diag,diag);
    std::copy(orig_offd,orig_offd+nnz_offd,offd);

    delete[] orig_diag;
    delete[] orig_offd;
    return ierr;
}
}//namespace parelag
