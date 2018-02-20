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


#ifndef PARELAG_SIMPLEXMLPARAMETERLISTREADER_HPP_
#define PARELAG_SIMPLEXMLPARAMETERLISTREADER_HPP_

#include <fstream>

#include "utilities/MemoryUtils.hpp"
#include "utilities/ParELAG_ParameterList.hpp"

namespace parelag
{

/** \class SimpleXMLParameterListReader
 *  \brief Read ParameterList from XML file.
 *
 *  This class supports reading given pseudo-XML-like file and streams
 *  and returning working ParameterList objects.
 *
 *  Definition of "pseudo-XML-like file" is as follows:
 *
 *    1. It must parse as valid XML. The user is responsible for
 *       validating the XML as the checking facilities of this
 *       class are very limited.
 *
 *    2. There are only two valid tags that elements may have:
 *       \c %ParameterList and \c Parameter
 *
 *    3. \c %ParameterList must use a start-tag and an end-tag. The
 *       start-tag must contain one attribute called
 *       "name". Extraneous attributes will be ignored. E.g.,
 *       .
 *       \code{.xml}
             <ParameterList name="my parameter list name">
             ...
             </ParamterList>
         \endcode
 *
 *    4. \c Parameter must use an empty-element-tag with three tags
 *       called "name", "type", and "value". Extraneous attributes
 *       will be ignored.
 *       .
 *       \code{.xml}
              <Parameter name="my param" type="a type" value="the value"/>
         \endcode
 *
 *    5. The value of the "type" attribute must be either a
 *       fundamental type or a predefined "special" type. In
 *       particular, the following types are supported:
 *
 *         * "char"
 *         * "bool"
 *         * "int"
 *         * "long"
 *         * "unsigned long"
 *         * "long long"
 *         * "unsigned long long"
 *         * "float"
 *         * "double"
 *         * "long double"
 *         * "string"
 *         * "vector_int" (Creates an std::vector<int>)
 *         * "vector(int)" (Same as "vector_int")
 *         * "vector_double" (Creates an std::vector<double>)
 *         * "vector(double)" (Same as "vector_double")
 *
 *       \note This only limits the types that can be stored in a
 *             parameter list *file*; any type that obeys the
 *             VariableContainer rules can be stored in the
 *             ParameterList object.
 *
 *    6. All \c Parameter elements must be nested under a
 *       \c %ParameterList. \c Parameters that are not part of a
 *       \c %ParameterList are erroneous and result in an exception.
 *
 *    7. There must be only one master list per file. That is,
 *       there must be one and only one \c %ParameterList that is
 *       not a sublist of another \c %ParameterList.
 *
 *  \todo Replace with real XML reader. TinyXML or something?
 */
class SimpleXMLParameterListReader
{
public:

    /** \brief Return a ParameterList from the stream.
     *
     *  There are rules; see the class documentation a list of the
     *  rules for streams and files.
     *
     *  \param is The input stream from which to read the XML.
     *
     *  \return A constructed ParameterList representing the input XML
     *          deck.
     */
    std::unique_ptr<ParameterList> GetParameterList(std::istream& is) const;

    /** \brief Read in a parameter list from the given file.
     *
     *  There are rules; see the class documentation a list of the
     *  rules for streams and files.
     *
     *  \param filename The name of the file to read.
     *
     *  \return A constructed ParameterList representing the input XML
     *          deck.
     */
    std::unique_ptr<ParameterList> GetParameterList(const std::string& filename)
    {
        std::ifstream in_file(filename);

        PARELAG_TEST_FOR_EXCEPTION(
            not in_file.good(),
            std::runtime_error,
            "SimpleXMLParameterListReader::GetParameterList(...): "
            "Problem opening file \"" << filename << "\".");

        auto ret = GetParameterList(in_file);
        in_file.close();

        return ret;
    }

};// class SimpleXMLParameterListReader
}// namespace parelag
#endif /* PARELAG_SIMPLEXMLPARAMETERLISTREADER_HPP_ */
