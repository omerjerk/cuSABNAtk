/***
 *  $Id$
 **
 *  File: MDL.hpp
 *  Created: Nov 22, 2016
 *
 *  Author: Jaroslaw Zola <jaroslaw.zola@hush.com>
 *  Copyright (c) 2016 SCoRe Group http://www.score-group.org/
 *  Distributed under the MIT License.
 *  See accompanying file LICENSE.
 */

#ifndef MDL_HPP
#define MDL_HPP

#include <cmath>
#include <utility>


class MDL {
public:
    typedef std::pair<double, double> score_type;

    explicit MDL(int m = 0) : m_(m) { }

    void init(int ri, int qi) { score_ = 0.0; nc_ = 0.0; ri_ = ri; qi_ = qi; }

    void finalize(int qi) {
        // Uncomment to change how number of observed states in handled
        // qi_ = qi;
        nc_ = 0.5 * std::log2(m_) * (ri_ - 1) * qi_;
    } // finalize

    void operator()(int Nij) { }

    void operator()(int Nijk, int Nij) {
        double p = static_cast<double>(Nijk) / Nij;
        score_ += (Nijk * std::log2(p));
    } // operator()

    int r() const { return ri_; }

    score_type score() const { return {-(score_ - nc_), -score_}; }


private:
    int m_ = 0;

    double score_ = 0.0;
    double nc_ = 0.0;

    int ri_ = 1;
    int qi_ = 1;

}; // class MDL

#endif // MDL_HPP
