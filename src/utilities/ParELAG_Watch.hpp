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


#ifndef PARELAG_WATCH_HPP_
#define PARELAG_WATCH_HPP_

#include <chrono>
#include <type_traits>

namespace parelag
{

/** \class Watch
 *  \brief Simple stopwatch-like timing utility.
 *
 *  Wraps the std::chrono clocks providing simple start, stop,
 *  time_elapsed, and reset functionality.
 *
 *  This class guarantees use of a steady clock for timing
 *  correctness. If the high resolution clock is steady, it is used.
 */
class Watch
{
public:
    using clock_type =
        std::conditional<std::chrono::high_resolution_clock::is_steady,
                         std::chrono::high_resolution_clock,
                         std::chrono::steady_clock>::type;
    using duration_type = clock_type::duration;
    using time_point = clock_type::time_point;
    using seconds_type =
        std::chrono::duration<double, std::chrono::seconds::period>;

public:
    /** \brief Default constructor */
    Watch() = default;

    /** \brief Starts the timer running.
     *
     *  \warning If "Start()" is called on an already-running Watch,
     *  no elapsed time will be recorded and the current duration will
     *  be set to zero.
     */
    void Start() { Start_ = clock_type::now(); }

    /** \brief Stop the timer. */
    void Stop()
    {
        if (IsRunning()) ElapsedTime_ += (clock_type::now() - Start_);

        Start_ = time_point();
    }

    /** \brief Set the elapsed time counter to zero.
     *
     *  Also returns the watch to the default stopped state.
     */
    void Reset()
    {
        Start_ = time_point();
        ElapsedTime_ = duration_type::zero();
    }

    /** \brief Get the current accumulated elapsed time.
     *
     *  \return The current accumulated elapsed time in seconds.
     */
    double GetElapsedTime()
    {
        if (IsRunning()) { Stop(); Start(); }
        return std::chrono::duration_cast<seconds_type>(ElapsedTime_).count();
    }

    /** \brief Determine if the clock is running.
     *
     *  \return \c true if the clock is running.
     */
    bool IsRunning() { return (Start_ != time_point()); }

private:

    /** \brief The current elapsed time recorded by this watch. */
    duration_type ElapsedTime_ =  duration_type::zero();

    /** \brief Time point for when the watch was most recently
     *         started.
     */
    time_point Start_ =  time_point();

};// class Watch
}// namespace parelag

#endif /* PARELAG_WATCH_HPP_ */
