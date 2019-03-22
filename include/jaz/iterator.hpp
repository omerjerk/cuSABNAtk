/***
 *  $Id$
 **
 *  File: iterator.hpp
 *  Created: Apr 28, 2010
 *
 *  Author: Jaroslaw Zola <jaroslaw.zola@hush.com>
 *  Copyright (c) 2010-2012 Jaroslaw Zola
 *  Distributed under the Boost Software License, Version 1.0.
 *  See accompanying file LICENSE_BOOST.txt.
 *
 *  This file is part of jaz.
 */

#ifndef JAZ_ITERATOR_HPP
#define JAZ_ITERATOR_HPP

#include <iostream>
#include <iterator>
#include <string>


/** File: iterator.hpp
 */
namespace jaz {

  /** Class: ostream_iterator
   */
  template <typename T, typename charT = char, typename traits = std::char_traits<charT>, typename dist = std::ptrdiff_t>
  class ostream_iterator : public std::iterator<std::output_iterator_tag, T> {
  public:
      typedef charT char_type;
      typedef traits traits_type;
      typedef std::basic_ostream<char_type, traits_type> ostream_type;


      /** Constructor: ostream_iterator
       */
      ostream_iterator(ostream_type& os, unsigned int lsz,
                       const std::basic_string<charT, traits>& sep = " ")
          : os_(&os), lsz_(lsz), sep_(sep), pos_(0) { }


      /** Function: operator=
       */
      ostream_iterator& operator=(const T& val) {
          *os_ << val;
          ++pos_;
          if ((pos_ % lsz_) == 0) *os_ << std::endl;
          else *os_ << sep_;
          return *this;
      } // operator=


      /** Function: operator*
       */
      ostream_iterator& operator*() { return *this; }

      /** Function: operator++
       */
      ostream_iterator& operator++() { return *this; }

      /** Function: operator++
       */
      ostream_iterator& operator++(int) { return *this; }


  private:
      ostream_type* os_;
      unsigned int lsz_;
      std::basic_string<charT, traits> sep_;

      unsigned int pos_;

  }; // ostream_iterator


  /** Class: getline_iterator
   */
  template <typename charT = char, typename traits = std::char_traits<charT>, typename dist = std::ptrdiff_t>
  class getline_iterator : public std::iterator<std::input_iterator_tag, std::basic_string<charT, traits>, dist> {
  public:
      typedef charT char_type;
      typedef traits traits_type;
      typedef std::basic_string<charT, traits> value_type;
      typedef std::basic_istream<char_type, traits_type> istream_type;

      /** Constructor: getline_iterator
       */
      getline_iterator() : delim_(), value_(), state_(false), is_(0) { }

      /** Constructor: getline_iterator
       */
      getline_iterator(istream_type& is) : is_(&is) {
          delim_ = std::use_facet<std::ctype<char_type> >(is_->getloc()).widen('\n');
          m_read__();
      } // getline_iterator

      /** Constructor: getline_iterator
       */
      getline_iterator(istream_type& is, char_type delim)
          : delim_(delim), is_(&is) { m_read__(); }

      /** Constructor: getline_iterator
       */
      getline_iterator(const getline_iterator& gi)
          : delim_(gi.delim_), value_(gi.value_), state_(gi.state_), is_(gi.is_) { }


      /** Function: operator*
       */
      const value_type& operator*() const { return value_; }

      /** Function: operator->
       */
      const value_type* operator->() const { return &(operator*()); }


      /** Function: operator++
       */
      getline_iterator& operator++() {
          m_read__();
          return *this;
      } // operator++

      /** Function: operator++
       */
      getline_iterator operator++(int) {
          getline_iterator tmp = *this;
          m_read__();
          return tmp;
      } // operator++


  private:
      void m_read__() {
          state_ = (is_ && *is_) ? true : false;
          if (state_ == true) {
              std::getline(*is_, value_, delim_);
              state_ = *is_ ? true : false;
          }
      } // m_read__

      char_type delim_;
      value_type value_;

      bool state_;
      istream_type* is_;


      friend bool operator==(const getline_iterator& lhs, const getline_iterator& rhs) {
          return ((lhs.state_ == rhs.state_) && (!lhs.state_ || (lhs.is_ == rhs.is_)));
      } // operator==

      friend bool operator!=(const getline_iterator& lhs, const getline_iterator& rhs) {
          return !(lhs == rhs);
      } // operator!=

  }; // class getline_iterator

} // namespace jaz

#endif // JAZ_ITERATOR_HPP
