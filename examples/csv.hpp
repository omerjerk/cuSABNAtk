/***
 *  $Id$
 **
 *  File: csv.hpp
 *  Created: Nov 12, 2015
 *
 *  Author: Jaroslaw Zola <jaroslaw.zola@hush.com>
 *  Copyright (c) 2015 SCoRe Group http://www.score-group.org/
 *  Distributed under the MIT License.
 *  See accompanying file LICENSE.
 */

#ifndef CSV_HPP
#define CSV_HPP

#include <algorithm>
#include <fstream>
#include <sstream>
#include <tuple>
#include <vector>

#include <jaz/iterator.hpp>


template <typename T>
std::tuple<bool, int, int> read_csv(std::ifstream& f, std::vector<T>& data) {
    jaz::getline_iterator<> it(f);
    jaz::getline_iterator<> end;

    int n = 0;
    int m = 0;

    std::istringstream is;
    T t;

    for (; it != end; ++it, ++n) {
        auto s = *it;

        is.clear();
        is.str(s);

        // currently we allow only categorical data
        std::istream_iterator<int> iit(is);
        std::istream_iterator<int> iend;

        int l = data.size();
        std::copy(iit, iend, std::back_inserter(data));

        l = data.size() - l;
        if (l == 0) return std::make_tuple(false, -1, -1);

        if (m == 0) m = l;
        else if (m != l) return std::make_tuple(false, -1, -1);
    } // for it

    // sanity check
    for (int i = 0; i < n; ++i) {
        auto mm = std::minmax_element(data.data() + i * m, data.data() + (i + 1) * m);
        int d = *mm.second - *mm.first;
        if ((d < 1) || (d > 254)) return std::make_tuple(false, -1, -1);
    }

    return std::make_tuple(true, n, m);
} // read_csv

template <typename T>
std::tuple<bool, int, int> read_csv(const std::string& name, std::vector<T>& data) {
    std::ifstream f(name.c_str());
    if (!f) return std::make_tuple(false, -1, -1);
    return read_csv(f, data);
} // read_csv

std::vector<std::string> split(std::string s, std::string delimiter) {
    std::vector<std::string> list;
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        list.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    list.push_back(s);
    return list;
}

#endif // CSV_HPP
