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


#ifndef PARELAG_LEVEL_HPP_
#define PARELAG_LEVEL_HPP_

#include <sstream>
#include <unordered_map>

#include "utilities/ParELAG_VariableContainer.hpp"

// Basically a "Level" is a blackbox map object, not dissimilar from a
// ParameterList.

namespace parelag
{

/** \class Level
 *  \brief Stores all information for a specific level in the MG hierarchy.
 *
 *  This stores everything in VariableContainer objects, which allows
 *  users to store a wide variety of data on each level.
 *
 *  Note that while DeRhamSequence stores many objects (P, D, etc) for
 *  use in upscaling or whatever else, this class is specifically
 *  solver-focused.
 */
class Level
{

    /** \brief The type used to identify data members. */
    using key_type = std::string;

    /** \brief The type of object storing the members. */
    using mapped_type = VariableContainer;

public:

    /** \brief Default constructor.
     *
     *  The level has no ID and no data.
     */
    Level() = default;

    /** \brief Create a level with a specific ID.
     *
     *  \param ID The identifier of this level. 0 is the finest grid.
     */
    Level(int ID) : LevelID_(ID) {}

    /** \brief Destructor. */
    ~Level() = default;

    /** \name Deleted special functions */
    ///@{
    /** \brief Prevent move/copy construction/assignment.
     *
     *  \todo Reevaluate whether these can be enabled.
     */
    Level(const Level&) = delete;
    Level(Level&&) = delete;
    Level& operator=(const Level&) = delete;
    Level& operator=(Level&&) = delete;
    ///@}

    /** \brief Get the ID of the current level.
     *
     *  \return The ID for this level.
     */
    int GetLevelID() const noexcept {return LevelID_;}

    /** \brief Set the ID of the current level.
     *
     *  \param ID The ID to be set for this level.
     */
    void SetLevelID(int ID) noexcept {LevelID_ = ID;}

    /** \brief Get the previous level.
     *
     *  \return A shared_ptr to the previous level. If no previous
     *          level, returns empty shared_ptr.
     */
    std::shared_ptr<Level> GetPreviousLevel()
    {
        return PreviousLevel_.lock();
    }

    /** \brief Set the previous level.
     *
     *  \param PreviousLevel A shared_ptr to the preceding level (one
     *                       level finer) in the hierarchy.
     */
    void SetPreviousLevel(const std::shared_ptr<Level>& PreviousLevel)
    {
        PreviousLevel_ = PreviousLevel;
    }

    /** \brief Test if the key exists in the Level.
     *
     *  No guarantee is made that the corresponding data is not
     *  null. This just indicates that there is an entry to which \c
     *  key maps.
     *
     *  \param key The identifier to search for in the Level.
     *
     *  \return \c true if \c key is found.
     */
    bool IsKey(key_type const& key) const noexcept
    {
        return not (Map_.find(key) == Map_.end());
    }

    /** \brief Test if the key exists in the level and holds data.
     *
     *  \param key The identifier to search for in the Level.
     *
     *  \return \c true if \c key is found and matching container is
     *          not empty.
     *
     *  \todo This should maybe be templated on type to verify the
     *        held data is of the right type.
     */
    bool IsValidKey(key_type const& key) const noexcept
    {
        auto iter = Map_.find(key);
        if (iter != Map_.end())
        {
            return not (iter->second.IsEmpty());
        }

        // return false by default
        return false;
    }

    /** \brief Adds a new member to the Level with given key.
     *
     *  If \c key exists in the Level, no insertion happens.
     *
     *  \param key The identifier for the new data.
     *  \param value The new data.
     *
     *  \return \c true if adding the value succeeded.
     */
    template <typename T>
    bool Set(key_type const& key,T&& value)
    {
        bool key_exists = IsKey(key);
        if (not key_exists)
            Map_[key] = std::forward<T>(value);

        return not key_exists;
    }

    /** \brief Replaces the member with given key with a new value.
     *
     *  If \c key exists, any data associated with it is replaced by new data.
     *
     *  \param key The identifier for the new data.
     *  \param value The new data.
     */
    template <typename T>
    void Reset(key_type const& key, const T& value)
    {
        Map_[key] = value;
    }

    /** \brief Get the data corresponding to the given key from the Level.
     *
     *  \throws std::out_of_range Thrown when key doesn't exist.
     *  \throws bad_var_cast Thrown if found data is not the right type.
     *
     *  \param key Identifier assigned to the desired data.
     *
     *  \return A reference to the underlying data.
     */
    template <typename T>
    T& Get(key_type const& key)
    {
        auto iter = Map_.find(key);

        PARELAG_TEST_FOR_EXCEPTION(
            iter == Map_.end(),
            std::out_of_range,
            "Level::Get<T>(...): Key \"" << key <<
            "\" not found on Level " << LevelID_ << ".")

        return iter->second.GetData<T>();
    }

private:

    /** \brief The map from IDs to data. */
    std::unordered_map<key_type,mapped_type> Map_;

    /** \brief The ID of this level. -1 if not set. */
    int LevelID_ = -1;

    /** \brief A weak reference to the previous level. */
    std::weak_ptr<Level> PreviousLevel_;
    // No strong dependence, so there's no need to preserve the
    // shared_ptr...

};// class Level
}// namespace parelag
#endif /* PARELAG_LEVEL_HPP_ */
