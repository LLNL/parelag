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


#ifndef PARELAG_TIMER_HPP_
#define PARELAG_TIMER_HPP_

#include "ParELAG_Watch.hpp"

namespace parelag
{

/** \class Timer
 *  \brief A simple RAII-style wrapper around a Watch to start and
 *         stop it.
 *
 *  \todo This is really an implemenation detail of
 *        TimeManager. Perhaps it belongs there instead.
 */
class Timer
{
public:
    /** \brief Default construction is prohibited.
     *
     *  Currently binds to a reference, which can't be null.
     */
    Timer() = delete;

    /** \brief Construct with a reference to a watch.
     *
     *  Starts the timer.
     *
     *  \param watch  A reference to a Watch object managed elsewhere.
     */
    Timer(Watch& watch) : Watch_(watch) { Watch_.Start(); }

    /** \brief Stops the watch.
     */
    void Stop() { Watch_.Stop(); }

    /** \brief Destructor.
     *
     *  Stops the watch.
     */
    ~Timer() { Watch_.Stop(); }

private:
    /** \brief The held watch reference. */
    Watch& Watch_;
};// class Timer

}// namespace parelag
#endif /* PARELAG_TIMER_HPP_ */
