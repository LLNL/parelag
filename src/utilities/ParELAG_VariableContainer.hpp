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


#ifndef PARELAG_VARIABLECONTAINER_HPP_
#define PARELAG_VARIABLECONTAINER_HPP_

#include <list>
#include <memory>
#include <sstream>
#include <typeinfo>
#include <type_traits>
#include <vector>
#include <utility>

#include <cxxabi.h>

#include "utilities/ParELAG_Meta.hpp"
#include "utilities/MemoryUtils.hpp"
#include "utilities/elagError.hpp"

namespace parelag
{

/// Overloaded ostream operator for vectors
template <typename T>
inline std::ostream& operator<<(std::ostream& os,const std::vector<T>& rhs)
{
#ifdef ParELAG_USE_VECTOR_PRINT_COMMAS
    std::ostringstream oss;
    oss << "{";
    for (const auto& ent : rhs)
        oss << " " << ent << ",";
    // Un-put the comma
    oss.seekp(-1,std::ios_base::end);
    oss << " }";
    os << oss.str();
    return os;
#else
    os << "{ ";
    for (const auto& ent : rhs)
        os << ent << " ";
    os << "}";
    return os;
#endif
}

/// Overloaded ostream operator for lists
template <typename T>
inline std::ostream& operator<<(std::ostream& os,const std::list<T>& rhs)
{
#ifdef ParELAG_USE_VECTOR_PRINT_COMMAS
    std::ostringstream oss;
    oss << "{";
    for (const auto& ent : rhs)
        oss << " " << ent << ",";
    // Un-put the comma
    oss.seekp(-1,std::ios_base::end);
    oss << " }";
    os << oss.str();
    return os;
#else
    os << "{ ";
    for (const auto& ent : rhs)
        os << ent << " ";
    os << "}";
    return os;
#endif
}

/** \class VariableContainer
 *  \brief A class to store any object (standard type-erasure class).
 *
 *  The stored object needs to be copiable and have "operator<<"
 *  overloaded for it (for "print()"). I think those are the only
 *  requirements.
 *
 *  This class is basically "boost::any" or "std::any (c++17)" but
 *  slightly lighter.
 */
class VariableContainer
{
public:

    /** \name Constructors and destructors */
    ///@{

    /** \brief Default constructor creates empty container */
    VariableContainer()
        : data_{nullptr}
    {}

    /** \brief Constructor stores an object of type T
     *
     *  \tparam T  The (deduced) type of the object being stored.
     *
     *  \param value  The object being stored.
     */
    template <typename T>
    VariableContainer(T const& value)
        : data_{make_unique<Data<typename std::decay<T>::type>>(value)}
    {}

    /** \brief "Universal reference" constructor.
     *
     *  This disables VariableContainers of VariableContainers and
     *  things like VariableContainer<const T>.
     *
     *  \tparam T  The (deduced) type of the object being stored.
     *
     *  \param value  The object being stored.
     */
    template <typename T>
    VariableContainer(
        T&& value,
        EnableIf<not IsSame<VariableContainer&,T>()>* = nullptr,
        EnableIf<not IsConst<T>()>* = nullptr)
        : data_{make_unique<Data<Decay<T>>>(static_cast<T&&>(value))}
    {}

    /** \brief Copy constructor.
     *
     *  Copies the underlying object, too.
     */
    VariableContainer(const VariableContainer & rhs)
        : data_{rhs.data_ ? rhs.data_->clone() : nullptr}
    {}

    /** \brief Move constructor. */
    VariableContainer(VariableContainer&& rhs) noexcept
        : data_{std::move(rhs.data_)}
    {}

    /** \brief Destructor */
    ~VariableContainer() = default;

    ///@}
    /** \name Assignment operators and modifiers */
    ///@{

    /** \brief Swap held data with another VariableContainer object */
    VariableContainer& Swap(VariableContainer& rhs) noexcept
    {
        std::swap(data_,rhs.data_);
        return *this;
    }

    /** \brief Copy assignment. */
    VariableContainer& operator=(const VariableContainer& rhs)
    {
        VariableContainer(rhs).Swap(*this);
        return *this;
    }

    /** \brief Move assignmen. */
    VariableContainer& operator=(VariableContainer&& rhs) noexcept
    {
        rhs.Swap(*this);
        return *this;
    }

    /** \brief Assign from value. */
    template <typename T>
    VariableContainer& operator=(T&& rhs)
    {
        VariableContainer(std::forward<T>(rhs)).Swap(*this);
        return *this;
    }

    ///@}
    /** \name Query functions */
    ///@{

    /** |brief Query if the container holds an object.
     *
     *  \return \c true if the container is empty.
     */
    bool IsEmpty() const noexcept
    {
        return !data_;
    }

    /** \brief Get the type_info object for the held object.
     *
     *  \return A reference to the type_info object for the held
     *          object.
     */
    std::type_info const& DataType() const noexcept
    {
        return data_ ? data_->type() : typeid(void);
    }

    ///@}
    /** \name Type-protected data access */
    ///@{

    /** \brief Get back the held data.
     *
     *  Because the held object has been type-erased, the user must
     *  provide the actual type underlying the VariableContainer's
     *  held object. If this type is incorrect, an exception will be
     *  thrown.
     *
     *  \throws bad_var_cast  The user-provided type was not correct.
     *
     *  \tparam T  The type of the held data.
     *
     *  \return  A const reference to the held data.
     */
    template <typename T>
    T const& GetData() const;

    /** \brief Get back the held data.
     *
     *  Because the held object has been type-erased, the user must
     *  provide the actual type underlying the VariableContainer's
     *  held object. If this type is incorrect, an exception will be
     *  thrown.
     *
     *  \throws bad_var_cast  The user-provided type was not correct.
     *
     *  \tparam T  The type of the held data.
     *
     *  \return  A non-const reference to the held data.
     */
    template <typename T>
    T& GetData()
    {
        return const_cast<T&>(
            static_cast<const VariableContainer&>(*this).GetData<T>() );
    }

    ///@}

    /** \brief Print the held object to a stream.
     *
     *  \param os  The stream to which the object is printed.
     */
    void Print(std::ostream& os) const noexcept
    {
        if (data_) data_->Print(os);
    }

    /** \brief Clear the container.
     *
     *  This will delete the held object.
     */
    void Clear() noexcept
    {
        data_.reset();
    }

private:
    /** \class DataHolder
     *  \brief Base class for the held object container.
     */
    struct DataHolder
    {
        /** \brief Destructor. */
        virtual ~DataHolder() = default;

        /** \brief Creates a copy of the held data. */
        virtual std::unique_ptr<DataHolder> clone() const noexcept = 0;

        /** \brief Get the type_info for the held object. */
        virtual std::type_info const& type() const noexcept = 0;

        /** \brief Print the held object to the given stream. */
        virtual void Print(std::ostream& os) const noexcept = 0;
    };

    /** \class Data
     *  \brief The concrete data-holder object.
     */
    template <typename T>
    struct Data : DataHolder
    {
        /** \brief Create the data by making a copy. */
        Data(T const& value)
            : val_{value}
        {}

        /** \brief Create the data by moving an object. */
        Data(T&& value)
            : val_{std::move(value)}
        {}

        /** \brief Destructor. */
        ~Data() = default;

        /** \brief Creates a copy of the held data. */
        std::unique_ptr<DataHolder> clone() const noexcept final
        {
            return make_unique<Data>(val_);
        }

        /** \brief Get the type_info for the held object. */
        const std::type_info & type() const noexcept final
        {
            return typeid(T);
        }

        /** \brief Print the held object to the given stream. */
        void Print(std::ostream& os) const noexcept final
        {
            os << val_;
        }

        /** \brief The held object. */
        T val_;
    };

private:
    /** \brief The held data. */
    std::unique_ptr<DataHolder> data_;
};// class VariableContainer


/** \brief Operator<< overload for VariableContainer. */
inline std::ostream& operator<<(std::ostream& os,const VariableContainer& rhs)
{
    rhs.Print(os);
    return os;
}


template <typename T>
inline T const& VariableContainer::GetData() const
{
    PARELAG_TEST_FOR_EXCEPTION(
        not data_,
        bad_var_cast,
        "VariableContainer::GetData(): Container is empty.");

    const std::type_info & type_T = typeid(T);
    const std::type_info & type_THIS = this->DataType();

    if (type_T.hash_code() != type_THIS.hash_code())
    {
        int status;
        // We are responsible for freeing this memory (using free)
        char* type_T_name = abi::__cxa_demangle(type_T.name(),0,0,&status);
        char* type_THIS_name = abi::__cxa_demangle(type_THIS.name(),0,0,&status);

        std::ostringstream oss;
        oss << "Something has gone wrong casting a VariableContainer. "
            << "type_T = " << type_T_name << ". "
            << "type_THIS = " << type_THIS_name << ".";

        // See. Here we free() it.
        if (status == 0)
        {
            free(type_T_name);
            free(type_THIS_name);
        }

        PARELAG_TEST_FOR_EXCEPTION(true,bad_var_cast,oss.str());
    }

    VariableContainer::Data<T> * dyn_cast_data
        = dynamic_cast<VariableContainer::Data<T> *>(data_.get());

    PARELAG_TEST_FOR_EXCEPTION(
        not dyn_cast_data,
        bad_var_cast,
        "VariableContainer::GetData(): Dynamic cast failed but shouldn't have. "
        "You might have issues with incompatible RTTI systems in static and shared "
        "libraries. But since I'm pulling that info from a third party, I can't be "
        "more helpful to you than just making the suggestion. Sorry. :(");

    return dyn_cast_data->val_;
}

}// namespace parelag
#endif /* PARELAG_VARIABLECONTAINER_HPP_ */
