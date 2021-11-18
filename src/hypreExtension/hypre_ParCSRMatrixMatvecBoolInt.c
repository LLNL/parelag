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

HYPRE_Int
hypre_CSRMatrixMatvecBoolInt( int           alpha,
              hypre_CSRMatrix *A,
              int    *x,
              int           beta,
              int    *y     )
{

   HYPRE_Int        *A_i      = hypre_CSRMatrixI(A);
   HYPRE_Int        *A_j      = hypre_CSRMatrixJ(A);
   HYPRE_Int         num_rows = hypre_CSRMatrixNumRows(A);
   //HYPRE_Int         num_cols = hypre_CSRMatrixNumCols(A);

   HYPRE_Int        *A_rownnz = hypre_CSRMatrixRownnz(A);
   HYPRE_Int         num_rownnz = hypre_CSRMatrixNumRownnz(A);

   int      temp, tempx;

   HYPRE_Int         i, jj;
   //HYPRE_Int         i, j, jj;

   HYPRE_Int         m;

   double     xpar=0.7;

   HYPRE_Int         ierr = 0;

   /*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/

    if (alpha == 0)
    {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
       for (i = 0; i < num_rows; i++)
          y[i] *= beta;

       return ierr;
    }

   /*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/

   temp = beta / alpha;

   hypre_assert(temp * alpha == beta);

   if (temp != 1.0)
   {
      if (temp == 0.0)
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
     for (i = 0; i < num_rows; i++)
        y[i] = 0.0;
      }
      else
      {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
     for (i = 0; i < num_rows; i++)
        y[i] *= temp;
      }
   }

   /*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/

/* use rownnz pointer to do the A*x multiplication  when num_rownnz is smaller than num_rows */

   if (num_rownnz < xpar*(num_rows))
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,m,tempx) HYPRE_SMP_SCHEDULE
#endif

      for (i = 0; i < num_rownnz; i++)
      {
         m = A_rownnz[i];
         tempx = 0.0;
         for (jj = A_i[m]; jj < A_i[m+1]; jj++)
             tempx +=  x[A_j[jj]];
         y[m] += tempx;
      }

   }
   else
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,jj,temp) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows; i++)
      {
          temp = 0.0;
          for (jj = A_i[i]; jj < A_i[i+1]; jj++)
             temp += x[A_j[jj]];
          y[i] += temp;
      }
   }


   /*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/

   if (alpha != 1.0)
   {
#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i) HYPRE_SMP_SCHEDULE
#endif
      for (i = 0; i < num_rows; i++)
     y[i] *= alpha;
   }

   return ierr;
}


HYPRE_Int
hypre_ParCSRMatrixMatvecBoolInt(
                 int alpha,
                 hypre_ParCSRMatrix *A,
                 int *x_local,
                 int beta,
                 int *y_local )
{
   hypre_ParCSRCommHandle	*comm_handle;
   hypre_ParCSRCommPkg	*comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix      *diag   = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix      *offd   = hypre_ParCSRMatrixOffd(A);
   //HYPRE_Int         num_rows = hypre_ParCSRMatrixGlobalNumRows(A);
   //HYPRE_Int         num_cols = hypre_ParCSRMatrixGlobalNumCols(A);

   HYPRE_Int	    num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_Int        ierr = 0;
   HYPRE_Int	    num_sends, i, j, index, start;

   int     *x_tmp, *x_buf;


   x_tmp = parelag_hypre_CTAlloc(HYPRE_Int, num_cols_offd );

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);

   x_buf = parelag_hypre_CTAlloc(int, hypre_ParCSRCommPkgSendMapStart(comm_pkg, num_sends));

   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            x_buf[index++]
               = x_local[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)];
      }
   }

   comm_handle = hypre_ParCSRCommHandleCreate( 11, comm_pkg, x_buf, x_tmp );

   hypre_CSRMatrixMatvecBoolInt( alpha, diag, x_local, beta, y_local);

   hypre_ParCSRCommHandleDestroy(comm_handle);
   comm_handle = NULL;

   if (num_cols_offd) hypre_CSRMatrixMatvecBoolInt( alpha, offd, x_tmp, 1, y_local);

   parelag_hypre_TFree(x_tmp);
   parelag_hypre_TFree(x_buf);

   return ierr;
}

#if 0
/*--------------------------------------------------------------------------
 * hypre_ParCSRMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_ParCSRMatrixMatvecT( double           alpha,
                  hypre_ParCSRMatrix *A,
                  hypre_ParVector    *x,
                  double           beta,
                  hypre_ParVector    *y     )
{
   hypre_ParCSRCommHandle	**comm_handle;
   hypre_ParCSRCommPkg	*comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   hypre_CSRMatrix *diag = hypre_ParCSRMatrixDiag(A);
   hypre_CSRMatrix *offd = hypre_ParCSRMatrixOffd(A);
   hypre_Vector *x_local = hypre_ParVectorLocalVector(x);
   hypre_Vector *y_local = hypre_ParVectorLocalVector(y);
   hypre_Vector *y_tmp;
   HYPRE_Int           vecstride = hypre_VectorVectorStride( y_local );
   HYPRE_Int           idxstride = hypre_VectorIndexStride( y_local );
   double       *y_tmp_data, **y_buf_data;
   double       *y_local_data = hypre_VectorData(y_local);

   HYPRE_Int         num_rows  = hypre_ParCSRMatrixGlobalNumRows(A);
   HYPRE_Int         num_cols  = hypre_ParCSRMatrixGlobalNumCols(A);
   HYPRE_Int	       num_cols_offd = hypre_CSRMatrixNumCols(offd);
   HYPRE_Int         x_size = hypre_ParVectorGlobalSize(x);
   HYPRE_Int         y_size = hypre_ParVectorGlobalSize(y);
   HYPRE_Int         num_vectors = hypre_VectorNumVectors(y_local);

   HYPRE_Int         i, j, jv, index, start, num_sends;

   HYPRE_Int         ierr  = 0;

   /*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/

    if (num_rows != x_size)
              ierr = 1;

    if (num_cols != y_size)
              ierr = 2;

    if (num_rows != x_size && num_cols != y_size)
              ierr = 3;
   /*-----------------------------------------------------------------------
    *-----------------------------------------------------------------------*/

    comm_handle = parelag_hypre_CTAlloc(hypre_ParCSRCommHandle*,num_vectors);

    if ( num_vectors==1 )
    {
       y_tmp = hypre_SeqVectorCreate(num_cols_offd);
    }
    else
    {
       y_tmp = hypre_SeqMultiVectorCreate(num_cols_offd,num_vectors);
    }
    hypre_SeqVectorInitialize(y_tmp);

   /*---------------------------------------------------------------------
    * If there exists no CommPkg for A, a CommPkg is generated using
    * equally load balanced partitionings
    *--------------------------------------------------------------------*/
   if (!comm_pkg)
   {
      hypre_MatvecCommPkgCreate(A);
      comm_pkg = hypre_ParCSRMatrixCommPkg(A);
   }

   num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
   y_buf_data = parelag_hypre_CTAlloc( double*, num_vectors );
   for ( jv=0; jv<num_vectors; ++jv )
      y_buf_data[jv] = parelag_hypre_CTAlloc(double, hypre_ParCSRCommPkgSendMapStart
                                     (comm_pkg, num_sends));
   y_tmp_data = hypre_VectorData(y_tmp);
   y_local_data = hypre_VectorData(y_local);

   hypre_assert( idxstride==1 ); /* >>> only 'column' storage of multivectors implemented so far */

   if (num_cols_offd) hypre_CSRMatrixMatvecT(alpha, offd, x_local, 0.0, y_tmp);

   for ( jv=0; jv<num_vectors; ++jv )
   {
      /* >>> this is where we assume multivectors are 'column' storage */
      comm_handle[jv] = hypre_ParCSRCommHandleCreate
         ( 2, comm_pkg, &(y_tmp_data[jv*num_cols_offd]), y_buf_data[jv] );
   }

   hypre_CSRMatrixMatvecT(alpha, diag, x_local, beta, y_local);

   for ( jv=0; jv<num_vectors; ++jv )
   {
      hypre_ParCSRCommHandleDestroy(comm_handle[jv]);
      comm_handle[jv] = NULL;
   }
   parelag_hypre_TFree(comm_handle);

   if ( num_vectors==1 )
   {
      index = 0;
      for (i = 0; i < num_sends; i++)
      {
         start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
         for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
            y_local_data[hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j)]
               += y_buf_data[0][index++];
      }
   }
   else
      for ( jv=0; jv<num_vectors; ++jv )
      {
         index = 0;
         for (i = 0; i < num_sends; i++)
         {
            start = hypre_ParCSRCommPkgSendMapStart(comm_pkg, i);
            for (j = start; j < hypre_ParCSRCommPkgSendMapStart(comm_pkg, i+1); j++)
               y_local_data[ jv*vecstride +
                             idxstride*hypre_ParCSRCommPkgSendMapElmt(comm_pkg,j) ]
                  += y_buf_data[jv][index++];
         }
      }

   hypre_SeqVectorDestroy(y_tmp);
   y_tmp = NULL;
   for ( jv=0; jv<num_vectors; ++jv ) parelag_hypre_TFree(y_buf_data[jv]);
   parelag_hypre_TFree(y_buf_data);

   return ierr;
}
#endif
