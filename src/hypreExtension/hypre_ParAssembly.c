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

#include "hypreExtension.hpp"

// Based on hypre_BoomerAMGBuildCoarseOperator
//TODO Reimplement this using par_Tmatmul
HYPRE_Int
hypre_ParCSRMatrixAssembly( hypre_ParCSRMatrix  *RT,
                                    hypre_ParCSRMatrix  *A,
                                    hypre_ParCSRMatrix  *P,
                                    int ignoreNonLocalRows,
                                    hypre_ParCSRMatrix **RAP_ptr )

{
   MPI_Comm        comm = hypre_ParCSRMatrixComm(A);

   hypre_CSRMatrix *RT_diag = hypre_ParCSRMatrixDiag(RT);
   hypre_CSRMatrix *RT_offd = hypre_ParCSRMatrixOffd(RT);

   if(ignoreNonLocalRows)
	   RT_offd = hypre_ZerosCSRMatrix( hypre_CSRMatrixNumRows(RT_diag), 0);

   HYPRE_Int             num_cols_diag_RT = hypre_CSRMatrixNumCols(RT_diag);
   HYPRE_Int             num_cols_offd_RT = hypre_CSRMatrixNumCols(RT_offd);
   HYPRE_Int             num_rows_offd_RT = hypre_CSRMatrixNumRows(RT_offd);

   hypre_ParCSRCommPkg   *comm_pkg_RT = hypre_ParCSRMatrixCommPkg(RT);
   HYPRE_Int             num_recvs_RT = 0;
   HYPRE_Int             num_sends_RT = 0;
   HYPRE_Int             *send_map_starts_RT;
   HYPRE_Int             *send_map_elmts_RT;

   if(ignoreNonLocalRows)
	   comm_pkg_RT = NULL;

   hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(A);

   double          *A_diag_data = hypre_CSRMatrixData(A_diag);
   HYPRE_Int             *A_diag_i = hypre_CSRMatrixI(A_diag);
   HYPRE_Int             *A_diag_j = hypre_CSRMatrixJ(A_diag);

   hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(A);

   double          *A_offd_data = hypre_CSRMatrixData(A_offd);
   HYPRE_Int             *A_offd_i = hypre_CSRMatrixI(A_offd);
   HYPRE_Int             *A_offd_j = hypre_CSRMatrixJ(A_offd);

   HYPRE_Int  num_cols_diag_A = hypre_CSRMatrixNumCols(A_diag);
   HYPRE_Int  num_cols_offd_A = hypre_CSRMatrixNumCols(A_offd);

   hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(P);

   double          *P_diag_data = hypre_CSRMatrixData(P_diag);
   HYPRE_Int             *P_diag_i = hypre_CSRMatrixI(P_diag);
   HYPRE_Int             *P_diag_j = hypre_CSRMatrixJ(P_diag);

   hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(P);
   HYPRE_Int             *col_map_offd_P = hypre_ParCSRMatrixColMapOffd(P);

   double          *P_offd_data = hypre_CSRMatrixData(P_offd);
   HYPRE_Int             *P_offd_i = hypre_CSRMatrixI(P_offd);
   HYPRE_Int             *P_offd_j = hypre_CSRMatrixJ(P_offd);

   HYPRE_Int  first_col_diag_P = hypre_ParCSRMatrixFirstColDiag(P);
   HYPRE_Int  last_col_diag_P;
   HYPRE_Int  num_cols_diag_P = hypre_CSRMatrixNumCols(P_diag);
   HYPRE_Int  num_cols_offd_P = hypre_CSRMatrixNumCols(P_offd);
   HYPRE_Int *coarse_partitioning = hypre_ParCSRMatrixColStarts(P);
   HYPRE_Int *RT_partitioning = hypre_ParCSRMatrixColStarts(RT);

   hypre_ParCSRMatrix *RAP;
   HYPRE_Int                *col_map_offd_RAP;
   HYPRE_Int                *new_col_map_offd_RAP;

   hypre_CSRMatrix *RAP_int = NULL;

   double          *RAP_int_data;
   HYPRE_Int             *RAP_int_i;
   HYPRE_Int             *RAP_int_j;

   hypre_CSRMatrix *RAP_ext;

   double          *RAP_ext_data = NULL;
   HYPRE_Int             *RAP_ext_i = NULL;
   HYPRE_Int             *RAP_ext_j = NULL;

   hypre_CSRMatrix *RAP_diag;

   double          *RAP_diag_data;
   HYPRE_Int             *RAP_diag_i;
   HYPRE_Int             *RAP_diag_j;

   hypre_CSRMatrix *RAP_offd;

   double          *RAP_offd_data = NULL;
   HYPRE_Int             *RAP_offd_i = NULL;
   HYPRE_Int             *RAP_offd_j = NULL;

   HYPRE_Int              RAP_size;
   HYPRE_Int              RAP_ext_size;
   HYPRE_Int              RAP_diag_size;
   HYPRE_Int              RAP_offd_size;
   HYPRE_Int              P_ext_diag_size;
   HYPRE_Int              P_ext_offd_size;
   HYPRE_Int              first_col_diag_RAP;
   HYPRE_Int              last_col_diag_RAP;
   HYPRE_Int              num_cols_offd_RAP = 0;

   hypre_CSRMatrix *R_diag;

   double          *R_diag_data;
   HYPRE_Int             *R_diag_i;
   HYPRE_Int             *R_diag_j;

   hypre_CSRMatrix *R_offd;

   double          *R_offd_data;
   HYPRE_Int             *R_offd_i;
   HYPRE_Int             *R_offd_j;

   hypre_CSRMatrix *Ps_ext;

   double          *Ps_ext_data;
   HYPRE_Int             *Ps_ext_i;
   HYPRE_Int             *Ps_ext_j;

   double          *P_ext_diag_data = NULL;
   HYPRE_Int             *P_ext_diag_i = NULL;
   HYPRE_Int             *P_ext_diag_j = NULL;

   double          *P_ext_offd_data = NULL;
   HYPRE_Int             *P_ext_offd_i = NULL;
   HYPRE_Int             *P_ext_offd_j = NULL;

   HYPRE_Int             *col_map_offd_Pext;
   HYPRE_Int             *map_P_to_Pext = NULL;
   HYPRE_Int             *map_P_to_RAP = NULL;
   HYPRE_Int             *map_Pext_to_RAP = NULL;

   HYPRE_Int             *P_marker;
   HYPRE_Int            **P_mark_array;
   HYPRE_Int            **A_mark_array;
   HYPRE_Int             *A_marker;
   HYPRE_Int             *temp;

   HYPRE_Int              n_coarse, n_coarse_RT;
   HYPRE_Int              square = 1;
   HYPRE_Int              num_cols_offd_Pext = 0;

   HYPRE_Int              ic, i, j, k;
   HYPRE_Int              i1, i2, i3, ii, ns, ne, size, rest;
   HYPRE_Int              cnt, cnt_offd, cnt_diag, value;
   HYPRE_Int              jj1, jj2, jj3, jcol;

   HYPRE_Int             *jj_count, *jj_cnt_diag, *jj_cnt_offd;
   HYPRE_Int              jj_counter, jj_count_diag, jj_count_offd;
   HYPRE_Int              jj_row_begining, jj_row_begin_diag, jj_row_begin_offd;
   HYPRE_Int              start_indexing = 0; /* start indexing for RAP_data at 0 */
   HYPRE_Int              num_nz_cols_A;
   HYPRE_Int              num_procs;
   HYPRE_Int              num_threads;

   double           r_entry;
   double           r_a_product;
   double           r_a_p_product;

   double           zero = 0.0;

   /*-----------------------------------------------------------------------
    *  Copy ParCSRMatrix RT into CSRMatrix R so that we have row-wise access
    *  to restriction .
    *-----------------------------------------------------------------------*/

   hypre_MPI_Comm_size(comm,&num_procs);
   num_threads = hypre_NumThreads();

   if (comm_pkg_RT)
   {
        num_recvs_RT = hypre_ParCSRCommPkgNumRecvs(comm_pkg_RT);
        num_sends_RT = hypre_ParCSRCommPkgNumSends(comm_pkg_RT);
        send_map_starts_RT =hypre_ParCSRCommPkgSendMapStarts(comm_pkg_RT);
        send_map_elmts_RT = hypre_ParCSRCommPkgSendMapElmts(comm_pkg_RT);
   }

   hypre_CSRMatrixTranspose(RT_diag,&R_diag,1);
   if (num_cols_offd_RT)
   {
        hypre_CSRMatrixTranspose(RT_offd,&R_offd,1);
        R_offd_data = hypre_CSRMatrixData(R_offd);
        R_offd_i    = hypre_CSRMatrixI(R_offd);
        R_offd_j    = hypre_CSRMatrixJ(R_offd);
   }

   /*-----------------------------------------------------------------------
    *  Access the CSR vectors for R. Also get sizes of fine and
    *  coarse grids.
    *-----------------------------------------------------------------------*/

   R_diag_data = hypre_CSRMatrixData(R_diag);
   R_diag_i    = hypre_CSRMatrixI(R_diag);
   R_diag_j    = hypre_CSRMatrixJ(R_diag);

   n_coarse = hypre_ParCSRMatrixGlobalNumCols(P);
   num_nz_cols_A = num_cols_diag_A + num_cols_offd_A;

   n_coarse_RT = hypre_ParCSRMatrixGlobalNumCols(RT);
   if (n_coarse != n_coarse_RT)
      square = 0;

   /*-----------------------------------------------------------------------
    *  Generate Ps_ext, i.e. portion of P that is stored on neighbor procs
    *  and needed locally for triple matrix product
    *-----------------------------------------------------------------------*/

   if (num_procs > 1)
   {
        Ps_ext = hypre_ParCSRMatrixExtractBExt(P,A,1);
        Ps_ext_data = hypre_CSRMatrixData(Ps_ext);
        Ps_ext_i    = hypre_CSRMatrixI(Ps_ext);
        Ps_ext_j    = hypre_CSRMatrixJ(Ps_ext);
   }

   P_ext_diag_i = hypre_CTAlloc(HYPRE_Int,num_cols_offd_A+1);
   P_ext_offd_i = hypre_CTAlloc(HYPRE_Int,num_cols_offd_A+1);
   P_ext_diag_size = 0;
   P_ext_offd_size = 0;
   last_col_diag_P = first_col_diag_P + num_cols_diag_P - 1;

   for (i=0; i < num_cols_offd_A; i++)
   {
      for (j=Ps_ext_i[i]; j < Ps_ext_i[i+1]; j++)
         if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P)
            P_ext_offd_size++;
         else
            P_ext_diag_size++;
      P_ext_diag_i[i+1] = P_ext_diag_size;
      P_ext_offd_i[i+1] = P_ext_offd_size;
   }

   if (P_ext_diag_size)
   {
      P_ext_diag_j = hypre_CTAlloc(HYPRE_Int, P_ext_diag_size);
      P_ext_diag_data = hypre_CTAlloc(double, P_ext_diag_size);
   }
   if (P_ext_offd_size)
   {
      P_ext_offd_j = hypre_CTAlloc(HYPRE_Int, P_ext_offd_size);
      P_ext_offd_data = hypre_CTAlloc(double, P_ext_offd_size);
   }

   cnt_offd = 0;
   cnt_diag = 0;
   cnt = 0;
   for (i=0; i < num_cols_offd_A; i++)
   {
      for (j=Ps_ext_i[i]; j < Ps_ext_i[i+1]; j++)
         if (Ps_ext_j[j] < first_col_diag_P || Ps_ext_j[j] > last_col_diag_P)
         {
            P_ext_offd_j[cnt_offd] = Ps_ext_j[j];
            P_ext_offd_data[cnt_offd++] = Ps_ext_data[j];
         }
         else
         {
            P_ext_diag_j[cnt_diag] = Ps_ext_j[j] - first_col_diag_P;
            P_ext_diag_data[cnt_diag++] = Ps_ext_data[j];
         }
   }
   if (num_procs > 1)
   {
      hypre_CSRMatrixDestroy(Ps_ext);
      Ps_ext = NULL;
   }

   if (P_ext_offd_size || num_cols_offd_P)
   {
      temp = hypre_CTAlloc(HYPRE_Int, P_ext_offd_size+num_cols_offd_P);
      for (i=0; i < P_ext_offd_size; i++)
         temp[i] = P_ext_offd_j[i];
      cnt = P_ext_offd_size;
      for (i=0; i < num_cols_offd_P; i++)
         temp[cnt++] = col_map_offd_P[i];
   }
   if (cnt)
   {
      qsort0(temp, 0, cnt-1);

      num_cols_offd_Pext = 1;
      value = temp[0];
      for (i=1; i < cnt; i++)
      {
         if (temp[i] > value)
         {
            value = temp[i];
            temp[num_cols_offd_Pext++] = value;
         }
      }
   }

   if (num_cols_offd_Pext)
        col_map_offd_Pext = hypre_CTAlloc(HYPRE_Int,num_cols_offd_Pext);

   for (i=0; i < num_cols_offd_Pext; i++)
      col_map_offd_Pext[i] = temp[i];

   if (P_ext_offd_size || num_cols_offd_P)
      hypre_TFree(temp);

   for (i=0 ; i < P_ext_offd_size; i++)
      P_ext_offd_j[i] = hypre_BinarySearch(col_map_offd_Pext,
                                           P_ext_offd_j[i],
                                           num_cols_offd_Pext);
   if (num_cols_offd_P)
   {
      map_P_to_Pext = hypre_CTAlloc(HYPRE_Int,num_cols_offd_P);

      cnt = 0;
      for (i=0; i < num_cols_offd_Pext; i++)
         if (col_map_offd_Pext[i] == col_map_offd_P[cnt])
         {
            map_P_to_Pext[cnt++] = i;
            if (cnt == num_cols_offd_P) break;
         }
   }

   /*-----------------------------------------------------------------------
    *  First Pass: Determine size of RAP_int and set up RAP_int_i if there
    *  are more than one processor and nonzero elements in R_offd
    *-----------------------------------------------------------------------*/

  P_mark_array = hypre_CTAlloc(HYPRE_Int *, num_threads);
  A_mark_array = hypre_CTAlloc(HYPRE_Int *, num_threads);

  if (num_cols_offd_RT)
  {
   jj_count = hypre_CTAlloc(HYPRE_Int, num_threads);

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,size,rest,jj_counter,jj_row_begining,A_marker,P_marker) HYPRE_SMP_SCHEDULE
#endif
   for (ii = 0; ii < num_threads; ii++)
   {
     size = num_cols_offd_RT/num_threads;
     rest = num_cols_offd_RT - size*num_threads;
     if (ii < rest)
     {
        ns = ii*size+ii;
        ne = (ii+1)*size+ii+1;
     }
     else
     {
        ns = ii*size+rest;
        ne = (ii+1)*size+rest;
     }

   /*-----------------------------------------------------------------------
    *  Allocate marker arrays.
    *-----------------------------------------------------------------------*/

   if (num_cols_offd_Pext || num_cols_diag_P)
   {
      P_mark_array[ii] = hypre_CTAlloc(HYPRE_Int, num_cols_diag_P+num_cols_offd_Pext);
      P_marker = P_mark_array[ii];
   }
   A_mark_array[ii] = hypre_CTAlloc(HYPRE_Int, num_nz_cols_A);
   A_marker = A_mark_array[ii];
   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   jj_counter = start_indexing;
   for (ic = 0; ic < num_cols_diag_P+num_cols_offd_Pext; ic++)
   {
      P_marker[ic] = -1;
   }
   for (i = 0; i < num_nz_cols_A; i++)
   {
      A_marker[i] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over exterior c-points
    *-----------------------------------------------------------------------*/

   for (ic = ns; ic < ne; ic++)
   {

      jj_row_begining = jj_counter;

      /*--------------------------------------------------------------------
       *  Loop over entries in row ic of R_offd.
       *--------------------------------------------------------------------*/

      for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic+1]; jj1++)
      {
         i1  = R_offd_j[jj1];

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/

         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
         {
            i2 = A_offd_j[jj2];

            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2] = ic;

               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_ext.
                *-----------------------------------------------------------*/

               for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3++)
               {
                  i3 = P_ext_diag_j[jj3];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     jj_counter++;
                  }
               }
               for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3++)
               {
                  i3 = P_ext_offd_j[jj3] + num_cols_diag_P;

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     jj_counter++;
                  }
               }
            }
         }
         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_diag.
          *-----------------------------------------------------------------*/

         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];

            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2+num_cols_offd_A] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2+num_cols_offd_A] = ic;

               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_diag.
                *-----------------------------------------------------------*/

               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     jj_counter++;
                  }
               }
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_offd.
                *-----------------------------------------------------------*/

               for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
               {
                  i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     jj_counter++;
                  }
               }
            }
         }
      }
    }

    jj_count[ii] = jj_counter;

   }

   /*-----------------------------------------------------------------------
    *  Allocate RAP_int_data and RAP_int_j arrays.
    *-----------------------------------------------------------------------*/
   for (i = 0; i < num_threads-1; i++)
      jj_count[i+1] += jj_count[i];

   RAP_size = jj_count[num_threads-1];
   RAP_int_i = hypre_CTAlloc(HYPRE_Int, num_cols_offd_RT+1);
   RAP_int_data = hypre_CTAlloc(double, RAP_size);
   RAP_int_j    = hypre_CTAlloc(HYPRE_Int, RAP_size);

   RAP_int_i[num_cols_offd_RT] = RAP_size;

   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in RAP_int_data and RAP_int_j.
    *-----------------------------------------------------------------------*/

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,size,rest,jj_counter,jj_row_begining,A_marker,P_marker,r_entry,r_a_product,r_a_p_product) HYPRE_SMP_SCHEDULE
#endif
   for (ii = 0; ii < num_threads; ii++)
   {
     size = num_cols_offd_RT/num_threads;
     rest = num_cols_offd_RT - size*num_threads;
     if (ii < rest)
     {
        ns = ii*size+ii;
        ne = (ii+1)*size+ii+1;
     }
     else
     {
        ns = ii*size+rest;
        ne = (ii+1)*size+rest;
     }

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/
   if (num_cols_offd_Pext || num_cols_diag_P)
      P_marker = P_mark_array[ii];
   A_marker = A_mark_array[ii];

   jj_counter = start_indexing;
   if (ii > 0) jj_counter = jj_count[ii-1];

   for (ic = 0; ic < num_cols_diag_P+num_cols_offd_Pext; ic++)
   {
      P_marker[ic] = -1;
   }
   for (i = 0; i < num_nz_cols_A; i++)
   {
      A_marker[i] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over exterior c-points.
    *-----------------------------------------------------------------------*/

   for (ic = ns; ic < ne; ic++)
   {

      jj_row_begining = jj_counter;
      RAP_int_i[ic] = jj_counter;

      /*--------------------------------------------------------------------
       *  Loop over entries in row ic of R_offd.
       *--------------------------------------------------------------------*/

      for (jj1 = R_offd_i[ic]; jj1 < R_offd_i[ic+1]; jj1++)
      {
         i1  = R_offd_j[jj1];
         r_entry = R_offd_data[jj1];

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/

         for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
         {
            i2 = A_offd_j[jj2];
            r_a_product = r_entry * A_offd_data[jj2];

            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2] = ic;

               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_ext.
                *-----------------------------------------------------------*/

               for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3++)
               {
                  i3 = P_ext_diag_j[jj3];
                  r_a_p_product = r_a_product * P_ext_diag_data[jj3];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     RAP_int_data[jj_counter] = r_a_p_product;
                     RAP_int_j[jj_counter] = i3 + first_col_diag_P;
                     jj_counter++;
                  }
                  else
                  {
                     RAP_int_data[P_marker[i3]] += r_a_p_product;
                  }
               }

               for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3++)
               {
                  i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                  r_a_p_product = r_a_product * P_ext_offd_data[jj3];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     RAP_int_data[jj_counter] = r_a_p_product;
                     RAP_int_j[jj_counter]
                                = col_map_offd_Pext[i3-num_cols_diag_P];
                     jj_counter++;
                  }
                  else
                  {
                     RAP_int_data[P_marker[i3]] += r_a_p_product;
                  }
               }
            }

            /*--------------------------------------------------------------
             *  If i2 is previously visited ( A_marker[12]=ic ) it yields
             *  no new entries in RAP and can just add new contributions.
             *--------------------------------------------------------------*/

            else
            {
               for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3++)
               {
                  i3 = P_ext_diag_j[jj3];
                  r_a_p_product = r_a_product * P_ext_diag_data[jj3];
                  RAP_int_data[P_marker[i3]] += r_a_p_product;
               }
               for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3++)
               {
                  i3 = P_ext_offd_j[jj3] + num_cols_diag_P;
                  r_a_p_product = r_a_product * P_ext_offd_data[jj3];
                  RAP_int_data[P_marker[i3]] += r_a_p_product;
               }
            }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_diag.
          *-----------------------------------------------------------------*/

         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];
            r_a_product = r_entry * A_diag_data[jj2];

            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2+num_cols_offd_A] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2+num_cols_offd_A] = ic;

               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_diag.
                *-----------------------------------------------------------*/

               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3];
                  r_a_p_product = r_a_product * P_diag_data[jj3];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     RAP_int_data[jj_counter] = r_a_p_product;
                     RAP_int_j[jj_counter] = i3 + first_col_diag_P;
                     jj_counter++;
                  }
                  else
                  {
                     RAP_int_data[P_marker[i3]] += r_a_p_product;
                  }
               }
               for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
               {
                  i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                  r_a_p_product = r_a_product * P_offd_data[jj3];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begining)
                  {
                     P_marker[i3] = jj_counter;
                     RAP_int_data[jj_counter] = r_a_p_product;
                     RAP_int_j[jj_counter] =
                                col_map_offd_Pext[i3-num_cols_diag_P];
                     jj_counter++;
                  }
                  else
                  {
                     RAP_int_data[P_marker[i3]] += r_a_p_product;
                  }
               }
            }

            /*--------------------------------------------------------------
             *  If i2 is previously visited ( A_marker[12]=ic ) it yields
             *  no new entries in RAP and can just add new contributions.
             *--------------------------------------------------------------*/

            else
            {
               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3];
                  r_a_p_product = r_a_product * P_diag_data[jj3];
                  RAP_int_data[P_marker[i3]] += r_a_p_product;
               }
               for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
               {
                  i3 = map_P_to_Pext[P_offd_j[jj3]] + num_cols_diag_P;
                  r_a_p_product = r_a_product * P_offd_data[jj3];
                  RAP_int_data[P_marker[i3]] += r_a_p_product;
               }
            }
         }
      }
   }
   if (num_cols_offd_Pext || num_cols_diag_P)
      hypre_TFree(P_mark_array[ii]);
   hypre_TFree(A_mark_array[ii]);
   }

   RAP_int = hypre_CSRMatrixCreate(num_cols_offd_RT,num_rows_offd_RT,RAP_size);
   hypre_CSRMatrixI(RAP_int) = RAP_int_i;
   hypre_CSRMatrixJ(RAP_int) = RAP_int_j;
   hypre_CSRMatrixData(RAP_int) = RAP_int_data;
   hypre_TFree(jj_count);
  }

   RAP_ext_size = 0;
   if (num_sends_RT || num_recvs_RT)
   {
        RAP_ext = hypre_ExchangeRAPData(RAP_int,comm_pkg_RT);
        RAP_ext_i = hypre_CSRMatrixI(RAP_ext);
        RAP_ext_j = hypre_CSRMatrixJ(RAP_ext);
        RAP_ext_data = hypre_CSRMatrixData(RAP_ext);
        RAP_ext_size = RAP_ext_i[hypre_CSRMatrixNumRows(RAP_ext)];
   }
   if (num_cols_offd_RT)
   {
      hypre_CSRMatrixDestroy(RAP_int);
      RAP_int = NULL;
   }

   RAP_diag_i = hypre_CTAlloc(HYPRE_Int, num_cols_diag_RT+1);
   RAP_offd_i = hypre_CTAlloc(HYPRE_Int, num_cols_diag_RT+1);

   first_col_diag_RAP = first_col_diag_P;
   last_col_diag_RAP = first_col_diag_P + num_cols_diag_P - 1;

   /*-----------------------------------------------------------------------
    *  check for new nonzero columns in RAP_offd generated through RAP_ext
    *-----------------------------------------------------------------------*/

   if (RAP_ext_size || num_cols_offd_Pext)
   {
      temp = hypre_CTAlloc(HYPRE_Int,RAP_ext_size+num_cols_offd_Pext);
      cnt = 0;
      for (i=0; i < RAP_ext_size; i++)
         if (RAP_ext_j[i] < first_col_diag_RAP
                        || RAP_ext_j[i] > last_col_diag_RAP)
            temp[cnt++] = RAP_ext_j[i];
      for (i=0; i < num_cols_offd_Pext; i++)
         temp[cnt++] = col_map_offd_Pext[i];


      if (cnt)
      {
         qsort0(temp,0,cnt-1);
         value = temp[0];
         num_cols_offd_RAP = 1;
         for (i=1; i < cnt; i++)
         {
            if (temp[i] > value)
            {
               value = temp[i];
               temp[num_cols_offd_RAP++] = value;
            }
         }
      }

   /* now evaluate col_map_offd_RAP */
      if (num_cols_offd_RAP)
         col_map_offd_RAP = hypre_CTAlloc(HYPRE_Int, num_cols_offd_RAP);

      for (i=0 ; i < num_cols_offd_RAP; i++)
         col_map_offd_RAP[i] = temp[i];

      hypre_TFree(temp);
   }

   if (num_cols_offd_P)
   {
      map_P_to_RAP = hypre_CTAlloc(HYPRE_Int,num_cols_offd_P);

      cnt = 0;
      for (i=0; i < num_cols_offd_RAP; i++)
         if (col_map_offd_RAP[i] == col_map_offd_P[cnt])
         {
            map_P_to_RAP[cnt++] = i;
            if (cnt == num_cols_offd_P) break;
         }
   }

   if (num_cols_offd_Pext)
   {
      map_Pext_to_RAP = hypre_CTAlloc(HYPRE_Int,num_cols_offd_Pext);

      cnt = 0;
      for (i=0; i < num_cols_offd_RAP; i++)
         if (col_map_offd_RAP[i] == col_map_offd_Pext[cnt])
         {
            map_Pext_to_RAP[cnt++] = i;
            if (cnt == num_cols_offd_Pext) break;
         }
   }

   /*-----------------------------------------------------------------------
    *  Convert RAP_ext column indices
    *-----------------------------------------------------------------------*/

   for (i=0; i < RAP_ext_size; i++)
      if (RAP_ext_j[i] < first_col_diag_RAP
                        || RAP_ext_j[i] > last_col_diag_RAP)
            RAP_ext_j[i] = num_cols_diag_P
                                + hypre_BinarySearch(col_map_offd_RAP,
                                                RAP_ext_j[i],num_cols_offd_RAP);
      else
            RAP_ext_j[i] -= first_col_diag_RAP;

/*   need to allocate new P_marker etc. and make further changes */
   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/
   jj_cnt_diag = hypre_CTAlloc(HYPRE_Int, num_threads);
   jj_cnt_offd = hypre_CTAlloc(HYPRE_Int, num_threads);

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,k,jcol,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,size,rest,jj_count_diag,jj_count_offd,jj_row_begin_diag,jj_row_begin_offd,A_marker,P_marker) HYPRE_SMP_SCHEDULE
#endif
   for (ii = 0; ii < num_threads; ii++)
   {
     size = num_cols_diag_RT/num_threads;
     rest = num_cols_diag_RT - size*num_threads;
     if (ii < rest)
     {
        ns = ii*size+ii;
        ne = (ii+1)*size+ii+1;
     }
     else
     {
        ns = ii*size+rest;
        ne = (ii+1)*size+rest;
     }

   P_mark_array[ii] = hypre_CTAlloc(HYPRE_Int, num_cols_diag_P+num_cols_offd_RAP);
   A_mark_array[ii] = hypre_CTAlloc(HYPRE_Int, num_nz_cols_A);
   P_marker = P_mark_array[ii];
   A_marker = A_mark_array[ii];
   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;

   for (ic = 0; ic < num_cols_diag_P+num_cols_offd_RAP; ic++)
   {
      P_marker[ic] = -1;
   }
   for (i = 0; i < num_nz_cols_A; i++)
   {
      A_marker[i] = -1;
   }

   /*-----------------------------------------------------------------------
    *  Loop over interior c-points.
    *-----------------------------------------------------------------------*/

   for (ic = ns; ic < ne; ic++)
   {

      /*--------------------------------------------------------------------
       *  Set marker for diagonal entry, RAP_{ic,ic}. and for all points
       *  being added to row ic of RAP_diag and RAP_offd through RAP_ext
       *--------------------------------------------------------------------*/

      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;

      if (square)
         P_marker[ic] = jj_count_diag++;

      for (i=0; i < num_sends_RT; i++)
        for (j = send_map_starts_RT[i]; j < send_map_starts_RT[i+1]; j++)
            if (send_map_elmts_RT[j] == ic)
            {
                for (k=RAP_ext_i[j]; k < RAP_ext_i[j+1]; k++)
                {
                   jcol = RAP_ext_j[k];
                   if (jcol < num_cols_diag_P)
                   {
                        if (P_marker[jcol] < jj_row_begin_diag)
                        {
                                P_marker[jcol] = jj_count_diag;
                                jj_count_diag++;
                        }
                   }
                   else
                   {
                        if (P_marker[jcol] < jj_row_begin_offd)
                        {
                                P_marker[jcol] = jj_count_offd;
                                jj_count_offd++;
                        }
                   }
                }
                break;
            }

      /*--------------------------------------------------------------------
       *  Loop over entries in row ic of R_diag.
       *--------------------------------------------------------------------*/

      for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic+1]; jj1++)
      {
         i1  = R_diag_j[jj1];

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/

         if (num_cols_offd_A)
         {
           for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
           {
            i2 = A_offd_j[jj2];

            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2] = ic;

               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_ext.
                *-----------------------------------------------------------*/

               for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3++)
               {
                  i3 = P_ext_diag_j[jj3];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begin_diag)
                  {
                     P_marker[i3] = jj_count_diag;
                     jj_count_diag++;
                  }
               }
               for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3++)
               {
                  i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]]+num_cols_diag_P;

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begin_offd)
                  {
                     P_marker[i3] = jj_count_offd;
                     jj_count_offd++;
                  }
               }
            }
           }
         }
         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_diag.
          *-----------------------------------------------------------------*/

         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];

            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2+num_cols_offd_A] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2+num_cols_offd_A] = ic;

               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_diag.
                *-----------------------------------------------------------*/

               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begin_diag)
                  {
                     P_marker[i3] = jj_count_diag;
                     jj_count_diag++;
                  }
               }
               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_offd.
                *-----------------------------------------------------------*/

               if (num_cols_offd_P)
               {
                 for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
                 {
                  i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, mark it and increment
                   *  counter.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begin_offd)
                  {
                     P_marker[i3] = jj_count_offd;
                     jj_count_offd++;
                  }
                 }
               }
            }
         }
      }

      /*--------------------------------------------------------------------
       * Set RAP_diag_i and RAP_offd_i for this row.
       *--------------------------------------------------------------------*/
/*
      RAP_diag_i[ic] = jj_row_begin_diag;
      RAP_offd_i[ic] = jj_row_begin_offd;
*/
    }
    jj_cnt_diag[ii] = jj_count_diag;
    jj_cnt_offd[ii] = jj_count_offd;
   }

   for (i=0; i < num_threads-1; i++)
   {
      jj_cnt_diag[i+1] += jj_cnt_diag[i];
      jj_cnt_offd[i+1] += jj_cnt_offd[i];
   }

   jj_count_diag = jj_cnt_diag[num_threads-1];
   jj_count_offd = jj_cnt_offd[num_threads-1];

   RAP_diag_i[num_cols_diag_RT] = jj_count_diag;
   RAP_offd_i[num_cols_diag_RT] = jj_count_offd;

   /*-----------------------------------------------------------------------
    *  Allocate RAP_diag_data and RAP_diag_j arrays.
    *  Allocate RAP_offd_data and RAP_offd_j arrays.
    *-----------------------------------------------------------------------*/

   RAP_diag_size = jj_count_diag;
   if (RAP_diag_size)
   {
      RAP_diag_data = hypre_CTAlloc(double, RAP_diag_size);
      RAP_diag_j    = hypre_CTAlloc(HYPRE_Int, RAP_diag_size);
   }

   RAP_offd_size = jj_count_offd;
   if (RAP_offd_size)
   {
        RAP_offd_data = hypre_CTAlloc(double, RAP_offd_size);
        RAP_offd_j    = hypre_CTAlloc(HYPRE_Int, RAP_offd_size);
   }

   if (RAP_offd_size == 0 && num_cols_offd_RAP != 0)
   {
      num_cols_offd_RAP = 0;
      hypre_TFree(col_map_offd_RAP);
   }
   /*-----------------------------------------------------------------------
    *  Second Pass: Fill in RAP_diag_data and RAP_diag_j.
    *  Second Pass: Fill in RAP_offd_data and RAP_offd_j.
    *-----------------------------------------------------------------------*/

#ifdef HYPRE_USING_OPENMP
#pragma omp parallel for private(i,j,k,jcol,ii,ic,i1,i2,i3,jj1,jj2,jj3,ns,ne,size,rest,jj_count_diag,jj_count_offd,jj_row_begin_diag,jj_row_begin_offd,A_marker,P_marker,r_entry,r_a_product,r_a_p_product) HYPRE_SMP_SCHEDULE
#endif
   for (ii = 0; ii < num_threads; ii++)
   {
     size = num_cols_diag_RT/num_threads;
     rest = num_cols_diag_RT - size*num_threads;
     if (ii < rest)
     {
        ns = ii*size+ii;
        ne = (ii+1)*size+ii+1;
     }
     else
     {
        ns = ii*size+rest;
        ne = (ii+1)*size+rest;
     }

   /*-----------------------------------------------------------------------
    *  Initialize some stuff.
    *-----------------------------------------------------------------------*/

   P_marker = P_mark_array[ii];
   A_marker = A_mark_array[ii];
   for (ic = 0; ic < num_cols_diag_P+num_cols_offd_RAP; ic++)
   {
      P_marker[ic] = -1;
   }
   for (i = 0; i < num_nz_cols_A ; i++)
   {
      A_marker[i] = -1;
   }

   jj_count_diag = start_indexing;
   jj_count_offd = start_indexing;
   if (ii > 0)
   {
      jj_count_diag = jj_cnt_diag[ii-1];
      jj_count_offd = jj_cnt_offd[ii-1];
   }

   /*-----------------------------------------------------------------------
    *  Loop over interior c-points.
    *-----------------------------------------------------------------------*/

   for (ic = ns; ic < ne; ic++)
   {

      /*--------------------------------------------------------------------
       *  Create diagonal entry, RAP_{ic,ic} and add entries of RAP_ext
       *--------------------------------------------------------------------*/

      jj_row_begin_diag = jj_count_diag;
      jj_row_begin_offd = jj_count_offd;
      RAP_diag_i[ic] = jj_row_begin_diag;
      RAP_offd_i[ic] = jj_row_begin_offd;

      if (square)
      {
         P_marker[ic] = jj_count_diag;
         RAP_diag_data[jj_count_diag] = zero;
         RAP_diag_j[jj_count_diag] = ic;
         jj_count_diag++;
      }

      for (i=0; i < num_sends_RT; i++)
        for (j = send_map_starts_RT[i]; j < send_map_starts_RT[i+1]; j++)
            if (send_map_elmts_RT[j] == ic)
            {
                for (k=RAP_ext_i[j]; k < RAP_ext_i[j+1]; k++)
                {
                   jcol = RAP_ext_j[k];
                   if (jcol < num_cols_diag_P)
                   {
                        if (P_marker[jcol] < jj_row_begin_diag)
                        {
                                P_marker[jcol] = jj_count_diag;
                                RAP_diag_data[jj_count_diag]
                                        = RAP_ext_data[k];
                                RAP_diag_j[jj_count_diag] = jcol;
                                jj_count_diag++;
                        }
                        else
                                RAP_diag_data[P_marker[jcol]]
                                        += RAP_ext_data[k];
                   }
                   else
                   {
                        if (P_marker[jcol] < jj_row_begin_offd)
                        {
                                P_marker[jcol] = jj_count_offd;
                                RAP_offd_data[jj_count_offd]
                                        = RAP_ext_data[k];
                                RAP_offd_j[jj_count_offd]
                                        = jcol-num_cols_diag_P;
                                jj_count_offd++;
                        }
                        else
                                RAP_offd_data[P_marker[jcol]]
                                        += RAP_ext_data[k];
                   }
                }
                break;
            }

      /*--------------------------------------------------------------------
       *  Loop over entries in row ic of R_diag.
       *--------------------------------------------------------------------*/

      for (jj1 = R_diag_i[ic]; jj1 < R_diag_i[ic+1]; jj1++)
      {
         i1  = R_diag_j[jj1];
         r_entry = R_diag_data[jj1];

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_offd.
          *-----------------------------------------------------------------*/

         if (num_cols_offd_A)
         {
          for (jj2 = A_offd_i[i1]; jj2 < A_offd_i[i1+1]; jj2++)
          {
            i2 = A_offd_j[jj2];
            r_a_product = r_entry * A_offd_data[jj2];

            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2] = ic;

               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_ext.
                *-----------------------------------------------------------*/

               for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3++)
               {
                  i3 = P_ext_diag_j[jj3];
                  r_a_p_product = r_a_product * P_ext_diag_data[jj3];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/
                  if (P_marker[i3] < jj_row_begin_diag)
                  {
                        P_marker[i3] = jj_count_diag;
                        RAP_diag_data[jj_count_diag] = r_a_p_product;
                        RAP_diag_j[jj_count_diag] = i3;
                        jj_count_diag++;
                  }
                  else
                        RAP_diag_data[P_marker[i3]] += r_a_p_product;
               }
               for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3++)
               {
                  i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;
                  r_a_p_product = r_a_product * P_ext_offd_data[jj3];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/
                  if (P_marker[i3] < jj_row_begin_offd)
                  {
                        P_marker[i3] = jj_count_offd;
                        RAP_offd_data[jj_count_offd] = r_a_p_product;
                        RAP_offd_j[jj_count_offd] = i3 - num_cols_diag_P;
                        jj_count_offd++;
                  }
                  else
                        RAP_offd_data[P_marker[i3]] += r_a_p_product;
               }
            }

            /*--------------------------------------------------------------
             *  If i2 is previously visited ( A_marker[12]=ic ) it yields
             *  no new entries in RAP and can just add new contributions.
             *--------------------------------------------------------------*/
            else
            {
               for (jj3 = P_ext_diag_i[i2]; jj3 < P_ext_diag_i[i2+1]; jj3++)
               {
                  i3 = P_ext_diag_j[jj3];
                  r_a_p_product = r_a_product * P_ext_diag_data[jj3];
                  RAP_diag_data[P_marker[i3]] += r_a_p_product;
               }
               for (jj3 = P_ext_offd_i[i2]; jj3 < P_ext_offd_i[i2+1]; jj3++)
               {
                  i3 = map_Pext_to_RAP[P_ext_offd_j[jj3]] + num_cols_diag_P;
                  r_a_p_product = r_a_product * P_ext_offd_data[jj3];
                  RAP_offd_data[P_marker[i3]] += r_a_p_product;
               }
            }
          }
         }

         /*-----------------------------------------------------------------
          *  Loop over entries in row i1 of A_diag.
          *-----------------------------------------------------------------*/

         for (jj2 = A_diag_i[i1]; jj2 < A_diag_i[i1+1]; jj2++)
         {
            i2 = A_diag_j[jj2];
            r_a_product = r_entry * A_diag_data[jj2];

            /*--------------------------------------------------------------
             *  Check A_marker to see if point i2 has been previously
             *  visited. New entries in RAP only occur from unmarked points.
             *--------------------------------------------------------------*/

            if (A_marker[i2+num_cols_offd_A] != ic)
            {

               /*-----------------------------------------------------------
                *  Mark i2 as visited.
                *-----------------------------------------------------------*/

               A_marker[i2+num_cols_offd_A] = ic;

               /*-----------------------------------------------------------
                *  Loop over entries in row i2 of P_diag.
                *-----------------------------------------------------------*/

               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3];
                  r_a_p_product = r_a_product * P_diag_data[jj3];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begin_diag)
                  {
                     P_marker[i3] = jj_count_diag;
                     RAP_diag_data[jj_count_diag] = r_a_p_product;
                     RAP_diag_j[jj_count_diag] = P_diag_j[jj3];
                     jj_count_diag++;
                  }
                  else
                  {
                     RAP_diag_data[P_marker[i3]] += r_a_p_product;
                  }
               }
               if (num_cols_offd_P)
               {
                for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
                {
                  i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;
                  r_a_p_product = r_a_product * P_offd_data[jj3];

                  /*--------------------------------------------------------
                   *  Check P_marker to see that RAP_{ic,i3} has not already
                   *  been accounted for. If it has not, create a new entry.
                   *  If it has, add new contribution.
                   *--------------------------------------------------------*/

                  if (P_marker[i3] < jj_row_begin_offd)
                  {
                     P_marker[i3] = jj_count_offd;
                     RAP_offd_data[jj_count_offd] = r_a_p_product;
                     RAP_offd_j[jj_count_offd] = i3 - num_cols_diag_P;
                     jj_count_offd++;
                  }
                  else
                  {
                     RAP_offd_data[P_marker[i3]] += r_a_p_product;
                  }
                }
               }
            }

            /*--------------------------------------------------------------
             *  If i2 is previously visited ( A_marker[12]=ic ) it yields
             *  no new entries in RAP and can just add new contributions.
             *--------------------------------------------------------------*/

            else
            {
               for (jj3 = P_diag_i[i2]; jj3 < P_diag_i[i2+1]; jj3++)
               {
                  i3 = P_diag_j[jj3];
                  r_a_p_product = r_a_product * P_diag_data[jj3];
                  RAP_diag_data[P_marker[i3]] += r_a_p_product;
               }
               if (num_cols_offd_P)
               {
                for (jj3 = P_offd_i[i2]; jj3 < P_offd_i[i2+1]; jj3++)
                {
                  i3 = map_P_to_RAP[P_offd_j[jj3]] + num_cols_diag_P;
                  r_a_p_product = r_a_product * P_offd_data[jj3];
                  RAP_offd_data[P_marker[i3]] += r_a_p_product;
                }
               }
            }
         }
      }
   }
      hypre_TFree(P_mark_array[ii]);
      hypre_TFree(A_mark_array[ii]);
   }

   /* check if really all off-diagonal entries occurring in col_map_offd_RAP
	are represented and eliminate if necessary */

   P_marker = hypre_CTAlloc(HYPRE_Int,num_cols_offd_RAP);
   for (i=0; i < num_cols_offd_RAP; i++)
      P_marker[i] = -1;

   jj_count_offd = 0;
   for (i=0; i < RAP_offd_size; i++)
   {
      i3 = RAP_offd_j[i];
      if (P_marker[i3])
      {
	 P_marker[i3] = 0;
	 jj_count_offd++;
      }
   }

   if (jj_count_offd < num_cols_offd_RAP)
   {
      new_col_map_offd_RAP = hypre_CTAlloc(HYPRE_Int,jj_count_offd);
      jj_counter = 0;
      for (i=0; i < num_cols_offd_RAP; i++)
         if (!P_marker[i])
         {
	    P_marker[i] = jj_counter;
	    new_col_map_offd_RAP[jj_counter++] = col_map_offd_RAP[i];
         }

      for (i=0; i < RAP_offd_size; i++)
      {
	 i3 = RAP_offd_j[i];
	 RAP_offd_j[i] = P_marker[i3];
      }

      num_cols_offd_RAP = jj_count_offd;
      hypre_TFree(col_map_offd_RAP);
      col_map_offd_RAP = new_col_map_offd_RAP;
   }
   hypre_TFree(P_marker);

   RAP = hypre_ParCSRMatrixCreate(comm, n_coarse_RT, n_coarse,
                                  RT_partitioning, coarse_partitioning,
                                  num_cols_offd_RAP, RAP_diag_size,
                                  RAP_offd_size);

/* Make sure RAP does not own the coarse_partitioning instead of P or R*/
   hypre_ParCSRMatrixSetColStartsOwner(RAP,0);
   hypre_ParCSRMatrixSetRowStartsOwner(RAP,0);

   RAP_diag = hypre_ParCSRMatrixDiag(RAP);
   hypre_CSRMatrixI(RAP_diag) = RAP_diag_i;
   if (RAP_diag_size)
   {
      hypre_CSRMatrixData(RAP_diag) = RAP_diag_data;
      hypre_CSRMatrixJ(RAP_diag) = RAP_diag_j;
   }

   RAP_offd = hypre_ParCSRMatrixOffd(RAP);
   hypre_CSRMatrixI(RAP_offd) = RAP_offd_i;
   if (num_cols_offd_RAP)
   {
        hypre_CSRMatrixData(RAP_offd) = RAP_offd_data;
        hypre_CSRMatrixJ(RAP_offd) = RAP_offd_j;
        hypre_ParCSRMatrixColMapOffd(RAP) = col_map_offd_RAP;
   }
   if (num_procs > 1)
   {
        /* hypre_GenerateRAPCommPkg(RAP, A); */
        hypre_MatvecCommPkgCreate(RAP);
   }

   *RAP_ptr = RAP;

   /*-----------------------------------------------------------------------
    *  Free R, P_ext and marker arrays.
    *-----------------------------------------------------------------------*/

   hypre_CSRMatrixDestroy(R_diag);
   R_diag = NULL;

   if (num_cols_offd_RT)
   {
      hypre_CSRMatrixDestroy(R_offd);
      R_offd = NULL;
   }

   if (num_sends_RT || num_recvs_RT)
   {
      hypre_CSRMatrixDestroy(RAP_ext);
      RAP_ext = NULL;
   }
   hypre_TFree(P_mark_array);
   hypre_TFree(A_mark_array);
   hypre_TFree(P_ext_diag_i);
   hypre_TFree(P_ext_offd_i);
   hypre_TFree(jj_cnt_diag);
   hypre_TFree(jj_cnt_offd);
   if (num_cols_offd_P)
   {
      hypre_TFree(map_P_to_Pext);
      hypre_TFree(map_P_to_RAP);
   }
   if (num_cols_offd_Pext)
   {
      hypre_TFree(col_map_offd_Pext);
      hypre_TFree(map_Pext_to_RAP);
   }
   if (P_ext_diag_size)
   {
      hypre_TFree(P_ext_diag_data);
      hypre_TFree(P_ext_diag_j);
   }
   if (P_ext_offd_size)
   {
      hypre_TFree(P_ext_offd_data);
      hypre_TFree(P_ext_offd_j);
   }

   if(ignoreNonLocalRows)
	   hypre_CSRMatrixDestroy(RT_offd);

   return(0);

}
