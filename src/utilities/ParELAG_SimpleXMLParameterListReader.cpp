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


#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include "ParELAG_SimpleXMLParameterListReader.hpp"


namespace
{
/** \brief Auxiliary function to convert strings to parameter list
 *         values.
 */
parelag::ParameterList& add_parameter_to_list(
    std::vector<std::string>& param_attrs,parelag::ParameterList& list);

/** \brief Auxiliary function to extract the next attribute value from
 *         the given input stream.
 */
std::string get_next_attr_value(std::istream& in_ss);

/** \brief Auxiliary function to get a reference to the desired
 *         sublist given a root and a path.
 */
parelag::ParameterList& get_current_sublist(
    parelag::ParameterList& root, std::vector<std::string>& path_to_sublist);

/** \brief Trim all leading or trailing whitespace from a string */
std::string trim_string(char const* str);

}// namespace <anonymous>


namespace parelag
{

std::unique_ptr<ParameterList>
SimpleXMLParameterListReader::GetParameterList(std::istream& is) const
{
    std::unique_ptr<ParameterList> out;

    // Allocate some stuff to be reused
    std::string buff, element_name;
    //std::istringstream iss;
    std::vector<std::string> current_list_name;

    // Skip any gibberish at the front of the stream
    is.ignore(std::numeric_limits<std::streamsize>::max(),'<');

    auto clear_white_space = [](std::string& str)
    {
        auto back = std::remove_if(
                        str.begin(),str.end(), [](char x)
        {
            return std::isspace(x);
        });
        if (back != str.end())
            str.erase(back,str.end());
    };

    auto get_next_attr_name = [&clear_white_space](std::istream& in_ss)
    {
        std::string attr_name;
        std::getline(in_ss,attr_name,'=');

        clear_white_space(attr_name);

        if (attr_name.find('/') != std::string::npos)
            attr_name = "";

        return attr_name;
    };


    // Iterate through the stream, parsing as I go
    while (is.good())
    {
        // Short-circuit if we're ending a list
        if (is.get() == '/')
        {
            current_list_name.pop_back();
            is.ignore(std::numeric_limits<std::streamsize>::max(),'<');
            continue;
        }
        else
            is.unget();

        // Now grab the leftover line
        std::getline(is,buff,'>');

        // Skip over comments and xml descriptor garbage
        if ((buff.front() == '!') || (buff.front() == '?'))
        {
            // Skip to the next element and continue
            is.ignore(std::numeric_limits<std::streamsize>::max(),'<');
            continue;
        }

        // Copy the buffer into the string stream
        std::istringstream iss(buff);

        // Element name must be ParameterList or Parameter
        iss >> element_name;

        // If we get here, we know we should be creating a
        // ParameterList
        if (element_name == "ParameterList")
        {
            // Make sure we're not trying to create a new master list...
            PARELAG_TEST_FOR_EXCEPTION(
                out && current_list_name.size() == 0,
                std::runtime_error,
                "SimpleXMLParameterListReader::GetParamterList(...): "
                "Trying to create a new master list. "
                "Only one master list is allowed per call.");

            auto attr_name = get_next_attr_name(iss);

            PARELAG_TEST_FOR_EXCEPTION(
                attr_name != "name",
                std::runtime_error,
                "SimpleXMLParameterListReader::GetParamterList(...): "
                "Found attribute \"" << attr_name << "\". The only attribute "
                "allowed for ParameterList elements is \"name\".");

            // Since this guy is new, add to the top of the queue
            current_list_name.emplace_back(::get_next_attr_value(iss));

            if (!out)
                out = make_unique<ParameterList>(current_list_name.back());
        }
        else if (element_name == "Parameter")
        {
            PARELAG_TEST_FOR_EXCEPTION(
                current_list_name.size() == 0,
                std::runtime_error,
                "SimpleXMLParameterListReader::GetParamterList(...): "
                "Found a parameter outside of a parameter list. "
                "This is not allowed!");

            ParameterList& list = get_current_sublist(*out,current_list_name);

            std::vector<std::string> param_attrs(3);
            while (iss.good())
            {
                // Try to get the next name
                auto tmp = get_next_attr_name(iss);

                // Try to get the next attribute (the previous call
                // may have killed the stream if there are no more
                // parameters to be extracted in the first place!)
                if (iss.good())
                {
                    auto val = ::get_next_attr_value(iss);

                    if (tmp == "name")
                        param_attrs[0] = std::move(val);
                    else if (tmp == "type")
                        param_attrs[1] = std::move(val);
                    else if (tmp == "value")
                        param_attrs[2] = std::move(val);
                    // Ignore everything else
                }
            }

            // Pass the strings
            ::add_parameter_to_list(param_attrs,list);
        }
        else
        {
            const bool invalid_element_tag = true;
            PARELAG_TEST_FOR_EXCEPTION(
                invalid_element_tag,
                std::runtime_error,
                "SimpleXMLParameterListReader::GetParamterList(...): "
                "Invalid element tag = \"" << element_name << "\". "
                "Only acceptable options are \"ParameterList\" and "
                "\"Parameter\".");
        }

        // Move to the next element
        is.ignore(std::numeric_limits<std::streamsize>::max(),'<');
    }
    return out;
}
}// namespace parelag


namespace
{
parelag::ParameterList& add_parameter_to_list(
    std::vector<std::string>& param_attrs,parelag::ParameterList& list)
{
    std::string& param_name = param_attrs[0];
    std::string& val_type = param_attrs[1];
    std::string& val = param_attrs[2];

    // Make sure nothing is empty
    PARELAG_ASSERT(param_name != "");
    PARELAG_ASSERT(val != "");

    if (val_type == "char")
        // Go int -> char.
        list.Set<char>(param_name,std::stoi(val));
    else if (val_type == "bool")
    {
        // convert to uppercase, just in case
        std::transform(val.begin(),val.end(),val.begin(),
                       [](unsigned char c)
        {
            return std::toupper(c);
        });

        list.Set<bool>(param_name,(val == "TRUE"));
    }
    else if (val_type == "int")
        list.Set<int>(param_name,std::stoi(val));
    else if (val_type == "long")
        list.Set<long>(param_name,std::stol(val));
    else if (val_type == "unsigned long")
        list.Set<unsigned long>(param_name,std::stoul(val));
    else if (val_type == "long long")
        list.Set<long long>(param_name,std::stoll(val));
    else if (val_type == "unsigned long long")
        list.Set<unsigned long long>(param_name,std::stoull(val));
    else if (val_type == "size_t")
        list.Set<size_t>(param_name,std::stoull(val));
    else if (val_type == "float")
        list.Set<float>(param_name,std::stof(val));
    else if (val_type == "double")
        list.Set<double>(param_name,std::stod(val));
    else if (val_type == "long double")
        list.Set<long double>(param_name,std::stold(val));
    else if (val_type == "string")
        list.Set(param_name,val);
    else if ((val_type == "vector(int)") || (val_type == "vector_int"))
    {
        std::vector<int> tmp;
        std::istringstream iss(val);
        std::copy(std::istream_iterator<int>(iss),
                  std::istream_iterator<int>(),
                  std::back_inserter(tmp));
        list.Set<std::vector<int>>(param_name,std::move(tmp));
    }
    else if ((val_type == "vector(double)") || (val_type == "vector_double"))
    {
        std::vector<double> tmp;
        std::istringstream iss(val);
        std::copy(std::istream_iterator<double>(iss),
                  std::istream_iterator<double>(),
                  std::back_inserter(tmp));
        list.Set<std::vector<double>>(param_name,std::move(tmp));
    }
    else if (val_type == "list(string)")
    {
        using string_list = std::list<std::string>;
        string_list tmp;
        std::istringstream iss(val);

        for (std::array<char, 512> str; iss.getline(&str[0],512,',') ;)
        {
            tmp.emplace_back(trim_string(&str[0]));
            while (iss.peek() == ' ')
                iss.get();
        }

        list.Set<string_list>(param_name,std::move(tmp));
    }
    else
    {
        const bool type_is_invalid = true;
        PARELAG_TEST_FOR_EXCEPTION(
            type_is_invalid,
            std::runtime_error,
            "add_parameter_to_list(...): Encountered unknown type = \"" <<
            val_type << "\".");
    }
    return list;
}

std::string get_next_attr_value(std::istream& in_ss)
{
    auto current_pos = in_ss.tellg();

    bool use_single_quote = false;
    while (in_ss.good())
    {
        char c = in_ss.get();
        if (c == '\'')
        {
            use_single_quote = true;
            break;
        }
        else if (c == '\"')
            break;
    }

    PARELAG_TEST_FOR_EXCEPTION(
        !in_ss.good(),
        std::runtime_error,
        "SimpleXMLParameterListReader::GetParamterList(...): "
        "Cannot find any quotes left in the stream!");

    // Rewind!
    in_ss.seekg(current_pos);

    std::string attr_value;
    const char stop_char = (use_single_quote ? '\'' : '\"');

    // Ignore everything up to the quote, than grab everything
    // before the next quote.
    in_ss.ignore(std::numeric_limits<std::streamsize>::max(),stop_char);
    std::getline(in_ss,attr_value,stop_char);

    // Make sure we didn't get rubbish
    PARELAG_TEST_FOR_EXCEPTION(
        attr_value == "/" || attr_value == "",
        std::runtime_error,
        "SimpleXMLParameterListReader::GetParamterList(...): "
        "Found invalid attribute value!");

    return attr_value;
}


// FIXME: I feel like this is very unsafe, but if it works, it works,
// and I won't complain...
parelag::ParameterList& get_current_sublist(
    parelag::ParameterList& root, std::vector<std::string>& path_to_sublist)
{
    if (path_to_sublist.size() == 1) return root;

    parelag::ParameterList* out = &root;

    auto iter = path_to_sublist.begin() + 1;
    while (iter != path_to_sublist.end())
    {
        out = &(out->Sublist(*iter,false));
        ++iter;
    }
    return *out;
}


std::string trim_string(char const* str)
{
    std::string tmp(str);
    auto is_space_lambda =
        [](unsigned char const& a)
        {
            return !std::isspace(a);
        };

    auto beg = std::find_if(tmp.begin(), tmp.end(), is_space_lambda);
    auto end = std::find_if(tmp.rbegin(), tmp.rend(), is_space_lambda);

    return std::string{beg,end.base()};
}

}// namespace <anonymous>
