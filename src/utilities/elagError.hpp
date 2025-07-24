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
#ifndef ELAGERROR_HPP
#define ELAGERROR_HPP

#include <exception>
#include <stdexcept>
#include <iostream>
#include <sstream>

#include <_hypre_utilities.h>

#include "ParELAG_Config.h"
#include "ParELAG_Exceptions.hpp"
#include PARELAG_MFEM_CONFIG_HEADER

namespace parelag
{

/// This function only exists for setting breakpoints and
/// providing minimal debugging information
void Debug_BreakOnMe(const std::string& errorMessage);

/// Get the number of throws
size_t Debug_GetThrowCount();

/// Increase the throw count
void Debug_IncreaseThrowCount();
}// namespace parelag

/*--------------------------------------------------------------------------
 * Global variable used in hypre error checking
 *--------------------------------------------------------------------------*/

#if MFEM_HYPRE_VERSION >= 22900
extern hypre_Error hypre__global_error;
#define hypre_error_flag  hypre__global_error.error_flag
#else
extern HYPRE_Int hypre__global_error;
#define hypre_error_flag  hypre__global_error
#endif


#define NOT_IMPLEMENTED_YET 999

/// Prints an error, with file name and line number, to stderr
/// and then throws.
void elag_error_handler(
    const char *filename, HYPRE_Int line,HYPRE_Int ierr, const char *msg);

/// Throws error IERR with message MSG
#define elag_error(IERR)                                \
    elag_error_handler(__FILE__, __LINE__, IERR, NULL)

/// Throws error IERR with message MSG
#define elag_error_msg(IERR, MSG)                       \
    elag_error_handler(__FILE__, __LINE__, IERR, MSG)

/// Option for pretty function name printing
#ifdef ParELAG_HAVE_PRETTY_FUNCTION
#define PRETTY_FUNCTION_NAME() \
    "\nFunction:\n" << __PRETTY_FUNCTION__ << "\n"
#else
#define PRETTY_FUNCTION_NAME()
#endif

/// Assert the given hypre error code is valid
/// ierr should be an integral type, hence the comparison to zero
///
/// \note The ParELAG policy is that ALL raw hypre calls that return
/// an error code should either be directly wrapped by this macro or
/// immediately test the return value using this macro.
#define PARELAG_ASSERT_HYPRE_ERROR_FLAG(IERR)                           \
    {                                                                   \
        if (IERR != 0)                                                  \
        {                                                               \
            parelag::Debug_IncreaseThrowCount();                        \
            char* msg = new char[256];                                  \
            HYPRE_DescribeError(IERR,msg);                              \
            std::ostringstream outmsg;                                  \
            outmsg << "An error in HYPRE has been detected at:\n"       \
                   << __FILE__ << ":" << __LINE__ << "\n"               \
                   << PRETTY_FUNCTION_NAME() << "\n"                    \
                   << "Throw number: "                                  \
                   << parelag::Debug_GetThrowCount() << "\n\n"          \
                   << "HYPRE reports the error code: " << IERR << "\n"  \
                   << "HYPRE describes the error as: " << msg           \
                   << std::endl;                                        \
            const std::string& outmsgstr = outmsg.str();                \
            parelag::Debug_BreakOnMe(outmsgstr);                        \
            throw parelag::hypre_runtime_error(outmsgstr);              \
        }                                                               \
    }

/// Pull the global hypre error code and test it
#define PARELAG_ASSERT_GLOBAL_HYPRE_ERROR_FLAG()   \
    {                                              \
        const auto ierr = HYPRE_GetError();        \
        PARELAG_ASSERT_HYPRE_ERROR_FLAG(ierr);     \
    }


/// Throw std::exception if test evaluates to TRUE
#define PARELAG_TEST_FOR_EXCEPTION(Exception_Test,Exception,Message)    \
    {                                                                   \
        const bool do_throw = (Exception_Test);                         \
        if (do_throw)                                                   \
        {                                                               \
            parelag::Debug_IncreaseThrowCount();                        \
            std::ostringstream outmsg;                                  \
            outmsg << __FILE__ << ":" << __LINE__                       \
                   << PRETTY_FUNCTION_NAME() << "\n"                    \
                   << std::endl                                         \
                   << "Throw number: "                                  \
                   << parelag::Debug_GetThrowCount()                    \
                   << std::endl << std::endl                            \
                   << "Test that evaluated to true: "#Exception_Test    \
                << std::endl << std::endl                               \
                << Message;                                             \
                const std::string & outmsgstr = outmsg.str();           \
                parelag::Debug_BreakOnMe(outmsgstr);                    \
                throw Exception(outmsgstr);                             \
        }                                                               \
    }

// PARELAG_ASSERT is compiled / tested regardless of build options
#define PARELAG_ASSERT(test)                                            \
    PARELAG_TEST_FOR_EXCEPTION(!(test),std::logic_error,"Error!")

#define PARELAG_NOT_IMPLEMENTED()                                       \
    {                                                                   \
        bool function_not_implemented = true;                           \
        PARELAG_TEST_FOR_EXCEPTION(                                     \
            function_not_implemented,                                   \
            parelag::not_implemented_error,                             \
            "This function has not been implemented yet!");             \
    }

// PARELAG_ASSERT_DEBUG and elag_assert only active for debug builds
#ifndef ELAG_DEBUG
#define PARELAG_ASSERT_DEBUG(test) \
    (void)(test)
#define elag_assert(EX)
#else
#define PARELAG_ASSERT_DEBUG(test)              \
    PARELAG_ASSERT(test)
#define elag_assert(EX)                                                 \
    {                                                                   \
        if (!(EX))                                                      \
        {                                                               \
            hypre_fprintf(stderr,"elag_assert failed: %s\n", #EX);      \
            elag_error(1);                                              \
        }                                                               \
    }
#define PARELAG_ASSERTING
#endif

#ifndef ELAG_DEBUG_PEDANTIC
#define PARELAG_ASSERT_PEDANTIC(test)
#define elag_assert_pedantic(EX)
#else
#define PARELAG_ASSERT_PEDANTIC(test)           \
    PARELAG_ASSERT(test)
#define elag_assert_pedantic(EX) if (!(EX)) {hypre_fprintf(stderr,"elag_assert failed: %s\n", #EX); elag_error(1);}
#endif

#endif
