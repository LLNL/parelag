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


#include "ParELAG_TimeManager.hpp"

#include <iomanip>
#include <sstream>
#include <vector>

#include <mpi.h>

#include "utilities/elagError.hpp"
#include "utilities/MPIDataTypes.hpp"

namespace parelag
{

TimeManager::map_type TimeManager::Map_;

void TimeManager::Print(std::ostream& os, int root, MPI_Comm comm)
{
    using time_type = double;

    int myrank, mysize;
    MPI_Comm_size(comm, &mysize);
    MPI_Comm_rank(comm, &myrank);

    if (mysize == 1)
    {
        PrintSerial(os,root,comm);
        return;
    }

    // Verify we have a valid root to output to
    PARELAG_ASSERT(root < mysize);

    // Now start gathering data
    std::vector<key_type> my_keys;

    int name_width = -1;
    if (myrank == root)
        for (auto& iter : Map_)
        {
            my_keys.push_back(iter.first);
            // Make sure we've got a wide enough column for the
            // names. The "+1" is so that the key looks
            // null-terminated. This can go away if you'd rather use
            // the slicing constructor for std::string.
            if (static_cast<int>(iter.first.size()) >= name_width)
                name_width = static_cast<int>(iter.first.size())+1;
        }

    // Broadcast the maximum key size
    MPI_Bcast(&name_width,1,MPI_INT,root,comm);

    // Broadcast the number of keys
    int num_keys = static_cast<int>(my_keys.size());
    MPI_Bcast(&num_keys,1,MPI_INT,root,comm);

    // Create the key buffer
    std::vector<char> all_keys(num_keys*name_width, 0);

    // Fill the key buffer
    if (myrank == root)
    {
        char * key_start = all_keys.data();
        for (int keyid = 0; keyid < num_keys; ++keyid)
        {
            std::copy(my_keys[keyid].c_str(),
                      my_keys[keyid].c_str()+my_keys[keyid].size(),
                      key_start);
            key_start += name_width;
        }

        // Sanity check
        PARELAG_ASSERT(key_start == all_keys.data() + all_keys.size());
    }

    // Broadcast the key buffer
    MPI_Bcast(all_keys.data(),all_keys.size(), MPI_CHAR, root, comm);

    // Now get the times
    std::vector<time_type> my_times(num_keys);

    char* this_key = all_keys.data();
    for (int jj = 0; jj < num_keys; ++jj)
    {
        if (Map_.find(this_key) != Map_.end())
            my_times[jj] = Map_[this_key].GetElapsedTime();
        else
            my_times[jj] = -1.0;

        this_key += name_width;
    }

    std::vector<double> all_values;

    if (myrank == root)
        all_values.resize(mysize*num_keys,0.0);


    MPI_Gather(my_times.data(),
               my_times.size(),
               GetMPIType(my_times[0]),
               all_values.data(),
               num_keys,
               MPI_DOUBLE,
               root,
               comm);

    // Create the preable
    if (name_width < 36) name_width = 36;
    const int total_width = name_width + 38;
    if (myrank == root)
    {
        // Buffer everything to here first
        std::ostringstream oss;

        oss << std::endl
            << std::string(total_width,'=') << std::endl << std::endl
            << "parelag::TimeManager::Print() -- Root = " << root
            << ", Size = " << mysize << std::endl << std::endl
            << std::string(total_width,'-') << std::endl
            << std::setw(name_width+2) << std::left << "Timer Name"
            << "Minimum     Maximum     Mean" << std::endl;

        for (int ii = 0; ii < num_keys; ++ii)
        {
            double min = -1.0;
            double max = -1.0;
            int count = 0;
            double mean = 0.0;

            for (int jj = 0; jj < mysize; ++jj)
            {
                double this_value = all_values[ii + jj*num_keys];
                if (this_value == -1.0)
                    continue;
                else
                {
                    ++count;
                    mean += this_value;
                }

                if ((min == -1.0) || (this_value < min))
                    min = this_value;
                if ((max == -1.0) || (this_value > max))
                    max = this_value;
            }
            mean = mean / count;

            if (ii % 10 == 0)
                oss << std::string(total_width,'-') << std::endl;

            oss << std::setw(name_width+2) << std::left << my_keys[ii]
                << std::setw(10) << std::left << min << "  "
                << std::setw(10) << std::left << max << "  "
                << std::setw(10) << std::left << mean << std::endl;
        }

        oss << std::string(total_width,'=') << std::endl
            << "End of Time Output (#Timers = " << num_keys << ")" << std::endl
            << std::string(total_width,'=') << std::endl;

        os << oss.str() << std::endl;
    }
}/* Print() */

/* Sample output:

================================================================================

parelag::TimeManager::Print() -- Root = 0, Size = 8

--------------------------------------------------------------------------------
Timer Name                                  Minimum     Maximum     Mean
--------------------------------------------------------------------------------
Do something                                1.23456789  1.23456789  1.23456789
I am a long key a very long key indeeeeeed  1.23456789  1.23456789  1.23456789
Do something                                1.23456789  1.23456789  1.23456789
I am a long key a very long key indeeeeeed  1.23456789  1.23456789  1.23456789
Do something                                1.23456789  1.23456789  1.23456789
I am a long key a very long key indeeeeeed  1.23456789  1.23456789  1.23456789
Do something                                1.23456789  1.23456789  1.23456789
I am a long key a very long key indeeeeeed  1.23456789  1.23456789  1.23456789
Do something                                1.23456789  1.23456789  1.23456789
I am a long key a very long key indeeeeeed  1.23456789  1.23456789  1.23456789
--------------------------------------------------------------------------------
Do something                                1.23456789  1.23456789  1.23456789
I am a long key a very long key indeeeeeed  1.23456789  1.23456789  1.23456789
Do something                                1.23456789  1.23456789  1.23456789
I am a long key a very long key indeeeeeed  1.23456789  1.23456789  1.23456789
Do something                                1.23456789  1.23456789  1.23456789
I am a long key a very long key indeeeeeed  1.23456789  1.23456789  1.23456789
Do something                                1.23456789  1.23456789  1.23456789
I am a long key a very long key indeeeeeed  1.23456789  1.23456789  1.23456789
Do something                                1.23456789  1.23456789  1.23456789
I am a long key a very long key indeeeeeed  1.23456789  1.23456789  1.23456789
================================================================================
End of Time Output (Timers = 20)
================================================================================



 */
void TimeManager::PrintSerial(std::ostream& os, const int root, MPI_Comm comm)
{
    int myrank, mysize;
    MPI_Comm_size(comm, &mysize);
    MPI_Comm_rank(comm, &myrank);

    // Verify we have a valid root to output to
    PARELAG_ASSERT(root < mysize);

    const int common_output_rank = (root < 0 ? 0 : root);

    int my_max_width = -1,name_width;
    if ((root < 0) || (myrank == root))
        for (auto& iter : Map_)
            if (static_cast<int>(iter.first.size()) >= my_max_width)
                my_max_width = static_cast<int>(iter.first.size())+1;

    // Find the maximum width
    if (root < 0)
        MPI_Allreduce(&my_max_width,&name_width,1,MPI_INT,MPI_MAX,comm);
    else
        name_width = my_max_width;

    if (name_width < 60) name_width = 60;
    const int total_width = name_width + 20;

    std::ostringstream my_oss;
    if ((root < 0) || (root == myrank))
        _do_create_my_time_output(my_oss,myrank,name_width);

    MPI_Barrier(comm);
    if (common_output_rank == myrank)
        os << std::endl
           << std::string(total_width,'=') << std::endl << std::endl
           << "parelag::TimeManager::PrintSerial() -- Root = " << root << " " << myrank
           << ", Size = " << mysize <<  std::endl << std::endl
           << std::string(total_width,'=') << std::endl;

    if (root == myrank)
        os << my_oss.str();

    if (root < 0)
    {
        int my_string_size = my_oss.str().size();

        std::vector<int> all_msg_sizes;
        if (common_output_rank == myrank) all_msg_sizes.resize(mysize);

        MPI_Gather(&my_string_size,1,GetMPIType(my_string_size),
                   (int*)all_msg_sizes.data(),1,GetMPIType(my_string_size),
                   common_output_rank,comm);

        std::vector<int> msg_offsets;
        std::vector<char> big_msg;
        if (common_output_rank == myrank)
        {
            msg_offsets.resize(mysize+1,0);
            for (int ii = 0; ii < mysize; ++ii)
                msg_offsets[ii+1] = msg_offsets[ii] + all_msg_sizes[ii];

            big_msg.resize(msg_offsets.back());
        }

        MPI_Gatherv((char*)my_oss.str().c_str(),my_string_size,MPI_CHAR,
                    (char*)big_msg.data(), all_msg_sizes.data(), msg_offsets.data(),
                    MPI_CHAR, common_output_rank, comm);

        if (common_output_rank == myrank)
        {
            char * msg_start = big_msg.data();
            for (int ii = 0; ii < mysize; ++ii)
                os << std::string(msg_start+msg_offsets[ii],all_msg_sizes[ii])
                   << std::endl;
        }
    }

    if (common_output_rank == myrank)
    {
        os << std::string(total_width,'=') << std::endl
           << "End of Serial Time Output" << std::endl
           << std::string(total_width,'=') << std::endl << std::endl;
    }
}

void TimeManager::_do_create_my_time_output(
    std::ostream& os, const int rank, const int max_name_width)
{
    const int total_width = max_name_width + 20;

    os << "Processor ID: " << rank << std::endl
       << std::setw(max_name_width+2) << std::left << "Timer Name"
       << std::setw(18) << std::left << "Local Time" << std::endl;

    int count = 0;
    for (auto& iter : Map_)
    {
        if (count % 10 == 0)
            os << std::string(total_width,'-') << std::endl;
        os << std::setw(max_name_width) << std::left << iter.first << "  "
           << std::setw(18) << iter.second.GetElapsedTime() << std::endl;
        ++count;
    }
    os << std::string(total_width,'=') << std::endl
       << "End of Processor " << rank << " Timers (#Timers = " << count << ")"
       << std::endl
       << std::string(total_width,'=') << std::endl;

}

}// namespace parelag
