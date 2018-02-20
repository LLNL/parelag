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

#ifndef TRACE_HPP_
#define TRACE_HPP_

#include <fstream>
#include <mpi.h>

#ifdef ELAG_USE_TRACE
#define elag_trace_init(comm) { Trace::Init(comm); }
#define elag_trace_enter_block(msg) { elag_trace("Enter Block " << msg); ++Trace::indent; }
#define elag_trace(msg) { if(Trace::logger) Trace::logger << std::setw(Trace::indent) << Trace::indent_symbol << " " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl; }
#define elag_trace_leave_block(msg) { --Trace::indent; elag_trace("Leave Block " << msg); }

class Trace
{
public:
    static std::ofstream logger;
    static int indent;
    static char indent_symbol;
    static void Init(MPI_Comm comm, const std::string & tname = "trace");
};
#else

#define elag_trace_init(comm) {}
#define elag_trace_enter_block(msg) {}
#define elag_trace_leave_block(msg) {}
#define elag_trace(msg) {}

#endif

#endif /* TRACE_H_ */
