/***
 *  $Id$
 **
 *  File: logger.hpp
 *  Created: May 11, 2018
 *
 *  Author: Jaroslaw Zola <jaroslaw.zola@hush.com>
 *  Copyright (c) 2018 Jaroslaw Zola
 *  Distributed under the Boost Software License, Version 1.0.
 *  See accompanying file LICENSE_BOOST.txt.
 *
 *  This file is part of jaz.
 */

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>


namespace jaz {

  struct null_ostream : public std::ostream {
      null_ostream() : std::ios(0), std::ostream(0) { }
  }; // struct null_ostream


  namespace log {

    inline std::string byte_to_size(unsigned long long int sz) {
        std::ostringstream ss;

        if (sz < 1024) ss << sz << "B";
        else if (sz < (1024 * 1024)) {
            ss << std::setprecision(4) << (static_cast<float>(sz) / 1024) << "KB";
        } else if (sz < (1024 * 1024 * 1024)) {
            ss << std::setprecision(4) << (static_cast<float>(sz) / (1024 * 1024)) << "MB";
        } else {
            ss << std::setprecision(4) << (static_cast<float>(sz) / (1024 * 1024 * 1024)) << "GB";
        }

        return ss.str();
    } // byte_to_size

    inline std::string second_to_time(double t) {
        std::ostringstream ss;

        unsigned int tt = static_cast<unsigned int>(t);
        unsigned int ht = tt / 3600;
        unsigned int mt = (tt % 3600) / 60;
        unsigned int st = (tt % 3600) % 60;

        ss << t << "s (" << ht << "h" << mt << "m" << st << "s)";

        return ss.str();
    } // second_to_time

  } // namespace log


  class Logger {
  public:
      enum Level { DEBUG, INFO, WARN, ERROR };

      explicit Logger(std::ostream& os) : os_(os) { }

      explicit Logger(Level level = INFO, std::ostream& os = std::cout) : os_(os), level_(level) { }

      void level(Level l) { level_ = l; }

      std::ostream& debug() {
          if (level_ > DEBUG) return nout_;
          return os_ << m_time__() << " DEBUG ";
      } // debug

      std::ostream& info() {
          if (level_ > INFO) return nout_;
          return os_ << m_time__() << " INFO ";
      } // info

      std::ostream& warn() {
          if (level_ > WARN) return nout_;
          return os_ << m_time__() << " WARN ";
      } // warn

      std::ostream& error() {
          return os_ << m_time__() << " ERROR ";
      } // warn


  private:
      std::string m_time__() const {
          auto t = std::chrono::system_clock::now();
          auto tt = std::chrono::system_clock::to_time_t(t);

          std::stringstream ss;
          ss << std::put_time(std::localtime(&tt), "%Y-%m-%d %X");
          return ss.str();
      } // m_time__

      null_ostream nout_;

      std::ostream& os_ = std::cout;
      Level level_ = INFO;

  }; // class Logger

} // namespace jaz

#endif // LOGGER_HPP
