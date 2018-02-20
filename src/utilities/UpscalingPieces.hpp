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

/**
   Many of the example codes in examples/ and testsuite/
   share a huge amount of code. This module is an attempt to
   resuse some of that code and keep it in one place
   instead of copy/pasting it into twenty different files.

   These routines have no real relevance for the element
   agglomeration algorithms in parelag, they are just for
   convenience in our examples and tests.

   Andrew T. Barker
   atb@llnl.gov
   2 September 2015
*/

#ifndef __UPSCALINGPIECES_HPP_
#define __UPSCALINGPIECES_HPP_

#include <memory>

#include <mfem.hpp>

#include "amge/DeRhamSequence.hpp"
#include "linalg/legacy/ParELAG_HypreExtension.hpp"

namespace parelag
{

/**
   this was called myRAP and repeated a million times
   in all the test/example programs, now written
   just once here (though I think we can probably use
   an MFEM routine instead?)
*/
std::unique_ptr<mfem::SparseMatrix> ExampleRAP(const mfem::SparseMatrix & Rt,
                                               const mfem::SparseMatrix & A,
                                               const mfem::SparseMatrix & P);

int UpscalingHypreSolver(int form,
                         mfem::HypreParMatrix *pA,
                         const mfem::Vector &prhs,
                         DeRhamSequence * sequence,
                         int k,
                         int prec_timing_index,
                         int solver_timing_index,
                         int print_iter,
                         int max_num_iter,
                         double rtol,
                         double atol,
                         mfem::DenseMatrix& timings,
                         const SharingMap& form_dofTrueDof,
                         mfem::Vector &solution_out,
                         bool report_timing=true);

#ifdef ParELAG_ENABLE_PETSC
int UpscalingPetscSolver(int form,
                         mfem::PetscParMatrix *pA,
                         const mfem::Vector &prhs,
                         DeRhamSequence * sequence,
                         int k,
                         mfem::Array<int>* ebdr,
                         mfem::Array<int>* nbdr,
                         int prec_timing_index,
                         int solver_timing_index,
                         int print_iter,
                         int max_num_iter,
                         double rtol,
                         double atol,
                         mfem::DenseMatrix& timings,
                         const SharingMap& form_dofTrueDof,
                         mfem::Vector &solution_out,
                         bool report_timing=true);
#endif

void OutputUpscalingTimings(const mfem::Array<int>& ndofs,
                            const mfem::Array<int>& nnz,
                            const mfem::Array2D<int>& iter,
                            const mfem::DenseMatrix& timings,
                            const char ** solver_names,
                            const char ** stage_names);

void OutputUpscalingTimings(const mfem::Array<int>& ndofs,
                            const mfem::Array<int>& nnz,
                            const mfem::Array<int>& iter,
                            const mfem::DenseMatrix& timings,
                            const char ** stage_names);

void OutputUpscalingErrors(const mfem::DenseMatrix& u_errors_L2,
                           const mfem::Vector& u_norms_L2,
                           const mfem::DenseMatrix& p_errors_L2,
                           const mfem::Vector& p_norms_L2,
                           const mfem::DenseMatrix& errors_der,
                           const mfem::Vector& norms_der);

void OutputUpscalingErrors(const mfem::DenseMatrix& u_errors_L2,
                           const mfem::Vector& u_norms_L2,
                           const mfem::DenseMatrix& errors_der,
                           const mfem::Vector& norms_der);

void ReduceAndOutputUpscalingErrors(const mfem::DenseMatrix& u_errors_L2_2,
                                    const mfem::Vector& u_norm_L2_2,
                                    const mfem::DenseMatrix& p_errors_L2_2,
                                    const mfem::Vector& p_norm_L2_2,
                                    const mfem::DenseMatrix& errors_der_2,
                                    const mfem::Vector& norm_der_2);

void ReduceAndOutputUpscalingErrors(const mfem::DenseMatrix& u_errors_L2_2,
                                    const mfem::Vector& u_norm_L2_2,
                                    const mfem::DenseMatrix& errors_der_2,
                                    const mfem::Vector& norm_der_2);

}//namespace parelag
#endif
