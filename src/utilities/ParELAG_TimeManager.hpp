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


#ifndef PARELAG_TIMEMANAGER_HPP_
#define PARELAG_TIMEMANAGER_HPP_

#include <iostream>
#include <map>
#include <mpi.h>

#include "ParELAG_Watch.hpp"
#include "ParELAG_Timer.hpp"

namespace parelag
{

/** \class TimeManager
 *  \brief Manages Watch objects by mapping them to strings.
 *
 *  \todo This class is a shoddy implementation of a singleton. It
 *        should be refactored to either look like a "proper"
 *        singleton or we should allow multiple instances. We
 *        currently have not needed this, but it could happen?
 *
 *  \todo This code is not thread-safe, and is not even single-thread
 *        safe! (One could restart a watch by creating multiple Timers
 *        around it while it is already running. What to do??
 */
class TimeManager
{
public:
    using key_type = std::string;
    using watch_type = Watch;
    using map_type = std::map<key_type,watch_type>;
    // This is std::map instead of std::unordered map so they are
    // printed alphabetically automatically.

public:

    /** \brief Check if the specified timer exists */
    static bool IsTimer(const std::string& name)
    { return !(Map_.find(name) == Map_.end()); }

    /** \brief Add a stopwatch with the specified name to the Manager
     *  and return a reference to it.
     *
     *  \param name The name of the requested stopwatch.
     *
     *  \return A reference to the desired stopwatch, wrapped in a Timer.
     */
    static Timer AddTimer(const std::string& name)
    { return Timer(Map_[name]); }

    /** \brief Remove the specified stopwatch from the Manager.
     *
     *  \param name The name of the requested stopwatch.
     *
     *  \return \c true if the timer was removed; \c false otherwise.
     */
    static bool DeleteTimer(const std::string& name)
    { return Map_.erase(name); }

    /** \brief Get a direct reference to the specified stopwatch from
     *         the Manager.
     *
     *  \param name The name of the requested stopwatch.
     *
     *  \throws std::out_of_range if the specified watch does not exist.
     */
    static watch_type& GetWatch(const std::string& name)
    { return Map_.at(name); }

    /** \brief Get a timer wrapped around the watch with the specified name.
     *
     *  \param name The name of the requested stopwatch.
     *
     *  \throws std::out_of_range if the specified watch does not exist.
     */
    static Timer GetTimer(const std::string& name)
    { return Timer(Map_.at(name)); }

    /** \brief Clear the time manager completely. */
    static void ClearAllTimers()
    { map_type().swap(Map_); }

    /** \brief Print the timer with nice, serialized output.
     *
     *  This will gather data for all timers known by proc "root" on
     *  communicator "comm". If a timer exists on proc "root" but not
     *  on some other processors, those processors will return a time
     *  of -1.0, which will be excluded from min, max, mean. If a
     *  timer does not exist on proc "root" but does exist on some
     *  other procs, it will be excluded from this print.
     *
     *  If the communicator size is 1, a serial output will occur
     *  instead (min = max = mean, so why print all that?).
     *
     *  Currently "RealTime()" is printed. We could change this via a
     *  function pointer or some such thing.
     *
     *  \param os The ostream object to which to print.
     *  \param root The root rank for deciding which fields to print.
     *  \param comm The MPI communicator for the given rank.
     */
    static void Print(std::ostream& os = std::cout,
                      int root = 0,
                      MPI_Comm comm = MPI_COMM_WORLD);


    /** \brief Print the serial times on the proc of given rank on the given
     *         communicator.
     *
     *  If rank==-1, this will print the serial times for each
     *  processor in the communicator.
     *
     *  \param os The ostream object to which to print.
     *  \param rank The rank that should do the printing.
     *  \param comm The MPI communicator for the given rank.
     */
    static void PrintSerial(std::ostream& os = std::cout,
                            int rank = -1,
                            MPI_Comm comm = MPI_COMM_WORLD);

private:

    /** \brief Write the timer output for a single rank. */
    static void _do_create_my_time_output(
        std::ostream& os, int rank, int max_name_width);

private:

    /** \brief A map from strings to persistent Watch objects. */
    static map_type Map_;

};// class TimeManager

extern TimeManager::map_type Map_;

}// namespace parelag
#endif /* PARELAG_TIMEMANAGER_HPP_ */
