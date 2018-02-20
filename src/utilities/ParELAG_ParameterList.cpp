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


#include "ParELAG_ParameterList.hpp"
#include "ParELAG_VariableContainer.hpp"
#include "utilities/elagError.hpp"

#include <cxxabi.h>

namespace parelag
{

bool ParameterList::IsValid(const key_type& name) const noexcept
{
    auto iter = Map_.find(name);

    // key does not exist; return false
    if (iter == Map_.end()) return false;

    // return true if the data is nonempty
    return !(iter->second.IsEmpty());
}

bool ParameterList::IsSublist(const key_type& name) const noexcept
{
    auto iter = SublistMap_.find(name);

    // Key is not found in the list
    if (iter == SublistMap_.end()) return false;

    return true;
}

void ParameterList::Merge(const ParameterList& other)
{
    // Just loop over the parameters in other and set them in this
    // parameter list.
    for (const auto& iter : other.Map_)
        Map_[iter.first] = iter.second;

    for (const auto& iter : other.SublistMap_)
        SublistMap_[iter.first] = make_unique<ParameterList>(*iter.second);
}

ParameterList& ParameterList::Sublist(const std::string& name, bool must_exist)
{
    if (IsSublist(name))
        return *SublistMap_[name];
    else
        PARELAG_TEST_FOR_EXCEPTION(
            must_exist,
            std::runtime_error,
            "ParameterList::Sublist(): The list \"" << name
            << "\" does not exist in ParameterList \"" << Name_ << "\".");

    // Create a new list
    SublistMap_[name] = make_unique<ParameterList>(name);
    return *(SublistMap_[name]);
}

const ParameterList& ParameterList::Sublist(const std::string& name) const
{
    PARELAG_TEST_FOR_EXCEPTION(
        !IsSublist(name),
        std::runtime_error,
        "ParameterList::Sublist(): The list \"" << name
        << "\" does not exist in ParameterList \"" << Name_ << "\".");

    return *SublistMap_.at(name);
}

void ParameterList::Print(std::ostream& os, unsigned indent) const noexcept
{
    os << std::string(indent,' ')
       << "ParameterList (\"" << Name_ << "\")" << std::endl;

    // Variables for the demangling; malloc is necessary here...
    int status;
    size_t buffer_size = 128;
    char * buffer = (char*) malloc(buffer_size);

    // Print the parameters first...
    for (auto& thing : Map_)
    {
        auto& dtype = thing.second.DataType();

        buffer = abi::__cxa_demangle(dtype.name(),buffer,&buffer_size,&status);

        os << std::string(indent+2,' ') << "\"" << thing.first << "\": "
           << thing.second << " (" << buffer << ")" << std::endl;
    }
    // Cleanup the memory from the demangling
    free(buffer);

    // ...then the sublists.
    for (const auto& thing : SublistMap_)
    {
        thing.second->Print(os,indent+2);
    }

    os << std::string(indent,' ')
       << "END ParameterList (\"" << Name_ << "\")" << std::endl;
}

}// namespace parelag
