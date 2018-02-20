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

#include "hypreExtension.hpp"

void hypre_CSRDataTransformationSign(hypre_CSRMatrix * mat)
{
    double * data = hypre_CSRMatrixData(mat);
    int nnz       = hypre_CSRMatrixNumNonzeros(mat);

    double * idata;
    for(idata = data; idata != data + nnz; ++idata)
        if( fabs(*idata) > 1e-9 )
            *idata = ( (*idata > 0.) ? 1.0 : -1.0 );
        else
            *idata = 0.0;
}

void hypre_ParCSRDataTransformationSign(hypre_ParCSRMatrix * mat)
{
    hypre_CSRDataTransformationSign( hypre_ParCSRMatrixDiag(mat) );
    hypre_CSRDataTransformationSign( hypre_ParCSRMatrixOffd(mat) );
}

