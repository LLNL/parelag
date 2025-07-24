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


#include "ParELAG_MonolithicBlockedOperatorFactory.hpp"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ParELAG_MfemBlockOperator.hpp"
#include "ParELAG_MG_Utils.hpp"

#include "utilities/elagError.hpp"
#include "utilities/MPIDataTypes.hpp"
#include "hypreExtension/hypreExtension.hpp"

namespace parelag
{

// Assumptions:
//  - Each block in a block-row has the same row distribution.
//  - Each block in a block-col has the same col distribution.
template <typename BlockedOperatorType>
std::unique_ptr<mfem::HypreParMatrix>
MonolithicBlockedOperatorFactory::BuildOperator(BlockedOperatorType& op)
{
    using helper = mg_utils::BlockOpHelper<BlockedOperatorType>;

    MPI_Comm comm =
        dynamic_cast<mfem::HypreParMatrix&>(op.GetBlock(0,0)).GetComm();

    // Debugging
    int myid;
    MPI_Comm_rank(comm,&myid);
    double blocked_fro_norm = 0.;


    bool fix_singular_system = false;

    // Get the number of blocks and the block sizes
    auto row_offsets = helper::CopyRowOffsets(op);
    auto col_offsets = helper::CopyColumnOffsets(op);

    const size_t num_blk_rows = helper::GetNumBlockRows(op),
        num_blk_cols = helper::GetNumBlockCols(op);

    // Find the size of the matrix
    size_t total_diag_nnz = 0, total_offd_nnz = 0;
    size_t num_global_rows = 0, num_global_cols = 0;
    size_t num_cols_offd = 0;

    // Resize the dof-function map
    DofToFunction_.resize(row_offsets.back());

    // Create the row/col starts
    global_index_type * row_starts, * col_starts;
    row_starts = parelag_hypre_CTAlloc(global_index_type,2);
    col_starts = parelag_hypre_CTAlloc(global_index_type,2);
    row_starts[0] = 0; row_starts[1] = 0;
    col_starts[0] = 0; col_starts[1] = 0;

    // Simplify storage and compute my {row,col}_starts, as well as
    // the set of unique columns to offd

    std::vector<global_index_type> my_col_starts;

    std::vector<std::unordered_set<global_index_type>> offd_cols(num_blk_cols);

    std::vector<std::vector<hypre_ParCSRMatrix*>> mats(
        num_blk_rows,std::vector<hypre_ParCSRMatrix*>(num_blk_cols,nullptr));

    std::set<HYPRE_Int> send_neighbors, recv_neighbors;
    for (size_t brow = 0; brow < num_blk_rows; ++brow)
    {
        auto& blk_row = mats[brow];
        for (size_t bcol = 0; bcol < num_blk_cols; ++bcol)
        {
            if (op.IsZeroBlock(brow,bcol))
            {
                if (brow == bcol)
                    fix_singular_system = false;
                continue;
            }
            auto& blk_offd_cols = offd_cols[bcol];

            hypre_ParCSRMatrix* tmp = dynamic_cast<mfem::HypreParMatrix&>(
                op.GetBlock(brow,bcol));

            blk_row[bcol] = tmp;
            const auto& tmp_diag = hypre_ParCSRMatrixDiag(tmp);
            const auto& tmp_offd = hypre_ParCSRMatrixOffd(tmp);

            total_diag_nnz += hypre_CSRMatrixNumNonzeros(tmp_diag);
            total_offd_nnz += hypre_CSRMatrixNumNonzeros(tmp_offd);

            std::copy_n(hypre_ParCSRMatrixColMapOffd(tmp),
                        hypre_CSRMatrixNumCols(tmp_offd),
                        std::inserter(blk_offd_cols,blk_offd_cols.end()));

            if (brow == 0)
            {
                col_starts[0] += hypre_ParCSRMatrixColStarts(tmp)[0];
                col_starts[1] += hypre_ParCSRMatrixColStarts(tmp)[1];

                num_global_cols += hypre_ParCSRMatrixGlobalNumCols(tmp);

                std::copy_n(tmp->col_starts,2,
                            std::back_inserter(my_col_starts));
            }

            if (bcol == 0)
            {
                row_starts[0] += hypre_ParCSRMatrixRowStarts(tmp)[0];
                row_starts[1] += hypre_ParCSRMatrixRowStarts(tmp)[1];

                num_global_rows += hypre_ParCSRMatrixGlobalNumRows(tmp);
            }

            // Do the neighbor-finding
            hypre_ParCSRCommPkg* commPkg = hypre_ParCSRMatrixCommPkg(tmp);
            if (!commPkg)
            {
                hypre_MatvecCommPkgCreate(tmp);
                commPkg = hypre_ParCSRMatrixCommPkg(tmp);
            }

            std::copy_n(commPkg->send_procs,commPkg->num_sends,
                        std::inserter(send_neighbors,send_neighbors.end()));

            std::copy_n(commPkg->recv_procs,commPkg->num_recvs,
                        std::inserter(recv_neighbors,recv_neighbors.end()));
        }
    }

    if (!myid)
        std::cout << "||A_BLOCKED|| = " << std::sqrt(blocked_fro_norm) << "\n";

    // Check a bunch of stuff
    PARELAG_ASSERT(row_starts[1] - row_starts[0] == row_offsets.back());
    PARELAG_ASSERT(col_starts[1] - col_starts[0] == col_offsets.back());
    PARELAG_ASSERT(my_col_starts.size() == 2*num_blk_cols);

    // I cannot figure out a way to build col_map_offd without
    // communicating, which sucks. The issue is that I only know how
    // to map from block-local global IDs to block-global global IDs
    // within my own block -- everything else is just "someone
    // else's". One approach would be to receive
    //
    // [ block0_col_start, block0_col_end, block1_col_start,
    // block1_col_end, ... , blockN_col_start, blockN_col_end ]
    //
    // from each of my neighbor processes. Then I could compute their
    // new col_starts and map block-local global IDs into that
    // range. So far, I think this is the best option.

    // Now I need to send my_col_starts to all of my send neighbors
    // and receive the analogous data from all of my recv neighbors.

    // First I need to figure out who my send/recv neighbors are... I
    // use "unordered_set" here to get the unique list of neighbors. I
    // make assumptions about the distributions of rows/cols, but not
    // about the sparsity pattern of the data. Thus, the send/recv
    // lists might differ among matrices.

    // Launch the recvs
    std::vector<std::vector<global_index_type>> neighbor_starts(
        recv_neighbors.size(), std::vector<global_index_type>(2*num_blk_cols));
    std::vector<MPI_Request> recv_requests(recv_neighbors.size());
    {
        size_t recv_count = 0;
        for (const auto& recv_neighbor : recv_neighbors)
        {
            auto& starts = neighbor_starts[recv_count];
            auto mpi_info =
                MPI_Irecv(starts.data(), 2*num_blk_cols,
                          GetMPIType<global_index_type>(0),
                          recv_neighbor, 0, comm, &recv_requests[recv_count]);
            PARELAG_ASSERT(mpi_info == 0);

            ++recv_count;
        }
    }

    // Launch the sends
    std::vector<MPI_Request> send_requests(send_neighbors.size());
    {
        size_t send_count = 0;
        for (const auto& send_neighbor : send_neighbors)
        {
            auto mpi_info =
                MPI_Isend(my_col_starts.data(),2*num_blk_cols,
                          GetMPIType<global_index_type>(0),
                          send_neighbor, 0, comm, &send_requests[send_count]);
            PARELAG_ASSERT(mpi_info == 0);
            ++send_count;
        }
    }

    MPI_Waitall(send_requests.size(),send_requests.data(),MPI_STATUSES_IGNORE);
    MPI_Waitall(recv_requests.size(),recv_requests.data(),MPI_STATUSES_IGNORE);

    // Add up the number of cols each block will contribute to offd
    for (const auto& offd_col_set : offd_cols)
        num_cols_offd += offd_col_set.size();

    // Convert each block-local GID to a monolithic GID once
    std::vector<std::unordered_map<global_index_type,global_index_type>>
        blgid_to_gids(num_blk_cols);
    std::unordered_map<global_index_type,global_index_type> gid_to_lid;
    std::vector<global_index_type> all_gids;

    {
        // Create aux data for this computation
        std::vector<std::vector<global_index_type>>
            neighbor_starts_by_blk(num_blk_cols),
            neighbor_ends_by_blk(num_blk_cols);
        std::vector<global_index_type> new_col_start(recv_neighbors.size(),0);

        // Loop over the data that I communicated
        size_t counter = 0;
        for (const auto& starts : neighbor_starts)
        {
            for (size_t blk = 0; blk < num_blk_cols; ++blk)
            {
                neighbor_starts_by_blk[blk].push_back(starts[2*blk]);
                new_col_start[counter] += starts[2*blk];
                neighbor_ends_by_blk[blk].push_back(starts[2*blk+1]);
            }
            ++counter;
        }

        // DEBUG
        {
            for (const auto& starts : neighbor_starts_by_blk)
            {
                PARELAG_ASSERT(starts.size() == recv_neighbors.size());
                PARELAG_ASSERT(std::is_sorted(starts.begin(),starts.end()));
            }
            for (const auto& ends : neighbor_ends_by_blk)
            {
                PARELAG_ASSERT(ends.size() == recv_neighbors.size());
                PARELAG_ASSERT(std::is_sorted(ends.begin(),ends.end()));
            }
        }

        auto find_owning_neighbor =
            [&neighbor_ends_by_blk](const global_index_type& bl_GID, const size_t& blk)
            {
                const auto& neighbor_ends = neighbor_ends_by_blk[blk];
                return std::distance(neighbor_ends.begin(),
                                     std::upper_bound(neighbor_ends.begin(),
                                                      neighbor_ends.end(),
                                                      bl_GID));
            };
        auto to_mono_gid =
            [&find_owning_neighbor, &new_col_start,
             &neighbor_starts_by_blk, &neighbor_ends_by_blk]
            (const global_index_type& bl_GID, const size_t& blk)
            {
                const auto owner = find_owning_neighbor(bl_GID,blk);
                global_index_type new_GCID = new_col_start[owner];
                for (size_t col = 0; col < blk; ++col)
                    new_GCID += neighbor_ends_by_blk[col][owner] -
                        neighbor_starts_by_blk[col][owner];

                new_GCID += bl_GID - neighbor_starts_by_blk[blk][owner];

                return new_GCID;
            };

        for (size_t bcol = 0; bcol < num_blk_cols; ++bcol)
        {
            auto& blgid_gid_map = blgid_to_gids[bcol];
            auto& col_blgids = offd_cols[bcol];

            for (const auto& blgid : col_blgids)
            {
                const auto GID = to_mono_gid(blgid,bcol);
                blgid_gid_map[blgid] = GID;
                all_gids.push_back(GID);
            }
        }

        std::sort(all_gids.begin(),all_gids.end());

        local_index_type LID = 0;
        for (const auto& GID : all_gids)
            gid_to_lid[GID] = LID++;
    }

    // Clear the temporary vector of sets
    std::vector<std::unordered_set<global_index_type>>().swap(offd_cols);

    //
    // Finally I can start filling the matrix!
    //

    // Create the matrix that will be returned
    hypre_ParCSRMatrix* mono_mat = hypre_ParCSRMatrixCreate(
        comm, num_global_rows, num_global_cols,
        row_starts, col_starts, num_cols_offd,
        total_diag_nnz, total_offd_nnz);

    PARELAG_ASSERT(hypre_ParCSRMatrixInitialize(mono_mat) == 0);

    // Copy the GIDs to col_map_offd
    std::copy(all_gids.begin(),all_gids.end(),
              hypre_ParCSRMatrixColMapOffd(mono_mat));

#if MFEM_HYPRE_VERSION <= 22200
    PARELAG_ASSERT(hypre_ParCSRMatrixOwnsRowStarts(mono_mat));
    PARELAG_ASSERT(hypre_ParCSRMatrixOwnsColStarts(mono_mat));
#endif

    // Grab the pieces of the monolithic guy
    hypre_CSRMatrix* m_diag = hypre_ParCSRMatrixDiag(mono_mat),
        * m_offd = hypre_ParCSRMatrixOffd(mono_mat);

    HYPRE_Int* m_diag_I = hypre_CSRMatrixI(m_diag);
    HYPRE_Int* m_diag_J = hypre_CSRMatrixJ(m_diag);
    HYPRE_Complex* m_diag_D = hypre_CSRMatrixData(m_diag);

    HYPRE_Int* m_offd_I = hypre_CSRMatrixI(m_offd);
    HYPRE_Int* m_offd_J = hypre_CSRMatrixJ(m_offd);
    HYPRE_Complex* m_offd_D = hypre_CSRMatrixData(m_offd);

    // Some useful counters
    size_t current_diag_nnz = 0, current_offd_nnz = 0;
    HYPRE_Int col_offset = 0;
    auto iter = DofToFunction_.begin();

    // Loop over the blocks and fill in the monolithic guy
    for (size_t blk_row = 0; blk_row < num_blk_rows; ++blk_row)
    {
        const auto& par_mats = mats[blk_row];

        const size_t num_rows_in_blk =
            row_offsets[blk_row+1] - row_offsets[blk_row];
        iter = std::fill_n(iter,num_rows_in_blk,blk_row);

        for (size_t row = 0; row < num_rows_in_blk; ++row)
        {
            *m_diag_I = current_diag_nnz;
            *m_offd_I = current_offd_nnz;

            HYPRE_Int* this_row_start_ind = m_diag_J;
            HYPRE_Complex* this_row_start_val = m_diag_D;

            for (size_t blk_col = 0; blk_col < num_blk_cols; ++blk_col)
            {
                if (!par_mats[blk_col])
                    continue;

                // Do the diagonal block
                const auto& diag = hypre_ParCSRMatrixDiag(par_mats[blk_col]);
                auto endJ = std::transform(
                    hypre_CSRMatrixJ(diag)+hypre_CSRMatrixI(diag)[row],
                    hypre_CSRMatrixJ(diag)+hypre_CSRMatrixI(diag)[row+1],
                    m_diag_J,
                    [&col_offset](const HYPRE_Int& a){return a+col_offset;});

                auto endD = std::copy(
                    hypre_CSRMatrixData(diag)+hypre_CSRMatrixI(diag)[row],
                    hypre_CSRMatrixData(diag)+hypre_CSRMatrixI(diag)[row+1],
                    m_diag_D);

                // Correct the diagonal
                if ((blk_row > 0) && (blk_row == blk_col)) // this is the diagonal block
                {
                    std::swap(*this_row_start_ind,*m_diag_J);
                    std::swap(*this_row_start_val,*m_diag_D);
                }

                col_offset += hypre_CSRMatrixNumCols(diag);
                current_diag_nnz += std::distance(m_diag_J,endJ);
                m_diag_J = endJ;
                m_diag_D = endD;

                // Do the offd block
                const auto& offd = hypre_ParCSRMatrixOffd(par_mats[blk_col]);
                const auto& col_map =
                    hypre_ParCSRMatrixColMapOffd(par_mats[blk_col]);
                const auto& blgid_to_gid = blgid_to_gids[blk_col];

                endJ = std::transform(
                    hypre_CSRMatrixJ(offd)+hypre_CSRMatrixI(offd)[row],
                    hypre_CSRMatrixJ(offd)+hypre_CSRMatrixI(offd)[row+1],
                    m_offd_J,
                    [&col_map,&gid_to_lid,&blgid_to_gid](const HYPRE_Int& a)
                    { return gid_to_lid[blgid_to_gid.at(col_map[a])];} );

                m_offd_D = std::copy(
                    hypre_CSRMatrixData(offd)+hypre_CSRMatrixI(offd)[row],
                    hypre_CSRMatrixData(offd)+hypre_CSRMatrixI(offd)[row+1],
                    m_offd_D);

                current_offd_nnz += std::distance(m_offd_J,endJ);
                m_offd_J = endJ;
            }// For each block column in the block row

            ++m_diag_I;
            ++m_offd_I;
            col_offset = 0;
        }// For each row in the block row
    }// for each block row
    *m_diag_I = current_diag_nnz;
    *m_offd_I = current_offd_nnz;

    // DEBUG
    PARELAG_ASSERT(current_diag_nnz == total_diag_nnz);
    PARELAG_ASSERT(current_offd_nnz == total_offd_nnz);

    if (fix_singular_system && (num_global_rows == (size_t)row_starts[1]))
    {
        std::cout << "FIXING SINGULARITY!\n";
        // Assert that I own the last column too
        PARELAG_ASSERT(row_starts[1] == col_starts[1]);

        // Find the last row; one on diagonal; zero elsewhere
        auto last_row_id = m_diag->num_rows-1;
        auto num_entries_in_last_row =
            m_diag_I[last_row_id+1] - m_diag_I[last_row_id];

        // Set the col idx to be the last column
        m_diag_J[m_diag_I[last_row_id]] = m_diag->num_cols-1;

        // Set the row to be [1., 0., 0., 0., ...]
        std::fill_n(m_diag_D + m_diag_I[last_row_id],
                    num_entries_in_last_row, 0.0);
        m_diag_D[m_diag_I[last_row_id]] = 1.0;

        std::fill(m_offd_D + m_offd_I[last_row_id],
                  m_diag_D + m_diag_I[last_row_id+1],0.0);

        m_diag_I[last_row_id+1] = m_diag_I[last_row_id]+1;
        m_diag->num_nonzeros = m_diag_I[last_row_id+1];

        m_offd_I[last_row_id+1] = m_diag_I[last_row_id];
    }

    return make_unique<mfem::HypreParMatrix>(mono_mat);
}// MonolithicBlockedOperator::BuildOperator()


// ETI for mfem::BlockOperator
template std::unique_ptr<mfem::HypreParMatrix>
MonolithicBlockedOperatorFactory::BuildOperator<mfem::BlockOperator>(
    mfem::BlockOperator& op);


// ETI for MfemBlockOperator
template std::unique_ptr<mfem::HypreParMatrix>
MonolithicBlockedOperatorFactory::BuildOperator<MfemBlockOperator>(
    MfemBlockOperator& op);

}// namespace parelag
