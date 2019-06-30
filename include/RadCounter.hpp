/***
 *    $Id$
 **
 *    File: RadCounter.hpp
 *    Created: Oct 18, 2017
 *
 *    Author: Subhadeep Karan <skaran@buffalo.edu>
 *    Copyright (c) 2017 SCoRe Group http://www.score-group.org/
 *    Distributed under the MIT License.
 *    See accompanying file LICENSE.
 */

#ifndef RAD_COUNTER_HPP
#define RAD_COUNTER_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

#include <bit_util.hpp>


template <int N, typename Data = uint8_t> class RadCounter {
public:
    using set_type = uint_type<N>;
    using data_type = Data;
    using pair_data_type = std::pair<data_type, data_type>;
    using pair_int = std::pair<int, int>;

    int n() const { return n_; }

    int m() const { return m_; }

    int r(int xi) const { return r_[xi]; }

    bool is_reorderable() { return true; }


    template <typename score_functor>
    void apply(const set_type& set_xi, const set_type& pa, const std::vector<data_type>& state_xi, const std::vector<data_type>& state_pa, std::vector<score_functor>& F) const {
        std::vector<pair_data_type> state_range_pa(state_pa.size());
        std::vector<pair_data_type> state_range_xi(state_xi.size());

        for (int i = 0; i < state_pa.size(); ++i) { state_range_pa[i] = {state_pa[i], state_pa[i] + 1}; }
        for (int i = 0; i < state_xi.size(); ++i) { state_range_xi[i] = {state_xi[i], state_xi[i] + 1}; }

        m_radcounter_core__(false, set_xi, pa, state_range_xi, state_range_pa, F);
    } // apply (state_specific_queries)


    template <typename score_functor>
    void apply(const set_type& set_xi, const set_type& pa, std::vector<score_functor>& F) const {
        std::vector<pair_data_type> state_pa(set_size(pa));
        std::vector<pair_data_type> state_xi(F.size());

        for (int xi = 0, idx_pa = 0, idx_xi = 0; xi < n_; ++xi) {
            if (in_set(pa, xi)) { state_pa[idx_pa++] = {0, r_[xi]}; }
            else if (in_set(set_xi, xi)) { state_xi[idx_xi++] = {0, r_[xi] }; }
        }

        m_radcounter_core__(true, set_xi, pa, state_xi, state_pa, F);
    } // apply

    template <typename score_functor>
    void apply(const std::vector<int>& xi_vect, const set_type& pa, std::vector<score_functor>& F) const {
        set_type set_xi = as_set<set_type>(std::begin(xi_vect), std::end(xi_vect));
        apply(set_xi, pa, F);
    } // apply

    template <typename score_functor>
    void apply(const std::vector<int>& xi_vect, const std::vector<int>& pa_vect, std::vector<score_functor>& F) const {
        set_type set_xi = as_set<set_type>(std::begin(xi_vect), std::end(xi_vect));
        auto pa = as_set<set_type>(std::begin(pa_vect), std::end(pa_vect));
        apply(set_xi, pa, F);
    } // apply

    template <typename score_functor>
    void apply(int xi, const set_type& pa, score_functor& F) const {
        set_type set_xi = set_empty<set_type>();
        set_xi = set_add(set_xi, xi);
        std::vector<score_functor> F_vect{F};
        apply(set_xi, pa, F_vect);
        F = F_vect[0];
    } // apply


    bool reorder(const std::vector<int>& norder) {
        std::vector<data_type> r_temp;
        std::vector<data_type> D_temp;
        std::vector<std::vector<int>> Rxi_temp;
        std::vector<int> idx_r_temp;

        r_temp.reserve(n_);
        D_temp.reserve(D_.size());
        Rxi_temp.reserve(Rxi_.size());
        idx_r_temp.reserve(n_);

        for (const int xi : norder) {
            D_temp.insert(D_temp.end(), D_.begin() + xi * m_, D_.begin() + (xi+1) * m_);
            idx_r_temp.push_back(Rxi_temp.size());
            for (int r = 0; r < r_[xi]; ++r) { Rxi_temp.push_back(Rxi_[idx_r_[xi] + r]); }
            r_temp.push_back(r_[xi]);
        }

        D_ = std::move(D_temp);
        idx_r_ = std::move(idx_r_temp);
        Rxi_ = std::move(Rxi_temp);
        r_ = std::move(r_temp);

        return true;
    } // reorder

private:
    template <int M, typename data_type_copy, typename Iter>
    friend RadCounter <M, data_type_copy> create_RadCounter(int n, int m, Iter it);

    template <typename score_functor>
    void m_radcounter_core__(bool skip_unique_row, const set_type& set_xi, const set_type pa, const std::vector<pair_data_type>& state_xi, const std::vector<pair_data_type>& state_pa, std::vector<score_functor>& F) const {
        int q_pa = m_compute_q_pa__(pa, state_pa);
        int pa_size = set_size(pa);

        if (pa_size == 0)  {
            for (int xi = 0, idx_xi = 0; xi < n_; ++xi) {
                if (!in_set(set_xi, xi)) { continue; }
                F[idx_xi].init(r_[xi], q_pa);
                F[idx_xi](m_);
                for (int r = state_xi[idx_xi].first; r < state_xi[idx_xi].second; ++r) { F[idx_xi](Rxi_[idx_r_[xi] + r].size(), m_); }
                F[idx_xi].finalize(1);
                ++idx_xi;
            }
            return;
        }

        int min_xi = 0;
        for (; min_xi < n_ && !in_set(pa, min_xi); ++min_xi);

        std::vector<int> row_id;
        std::vector<pair_int> ro_bracket;
        std::vector<pair_int> wo_bracket;
        std::vector<std::vector<int>> Rxi(max_r_);
        row_id.reserve(m_);
        ro_bracket.reserve(m_);
        wo_bracket.reserve(m_);

        for (int r = 0, idx = 0; r < max_r_; ++r) {
            Rxi[r].reserve(max_count_r_[r]);
            if (!(r >= state_pa[0].first) || !(r < state_pa[0].second)) { continue; }
            row_id.insert(row_id.end(), Rxi_[idx_r_[min_xi] + r].begin(), Rxi_[idx_r_[min_xi] + r].end());
            wo_bracket.push_back({idx, Rxi_[idx_r_[min_xi] + r].size()});
            idx += Rxi_[idx_r_[min_xi]+r].size();
        }

        for (int xi = 0, idx_pa = 0; xi < n_; ++xi) {
            if (!in_set(pa, xi)) { continue; }
            ro_bracket.clear();
            ro_bracket.swap(wo_bracket);
            m_radix_sort_core__(skip_unique_row, xi, false, state_pa[idx_pa++], ro_bracket, wo_bracket, Rxi, row_id, F[0]);
            if (wo_bracket.empty()) { return; }
            for (int i = row_id.size() - (wo_bracket.back().second + wo_bracket.back().first); i > 0; --i) { row_id.pop_back();}
        }

        ro_bracket.swap(wo_bracket);

        std::vector<int> row_id_copy;
        row_id_copy.reserve(m_);
        for (int xi = 0, idx_xi = 0, count_unique_row = 0; xi < n_; ++xi) {
            if (!in_set(set_xi, xi)) { continue; }
            row_id_copy = row_id;
            F[idx_xi].init(r_[xi], q_pa);

            count_unique_row = m_radix_sort_core__(skip_unique_row, xi, true, state_xi[idx_xi], ro_bracket, wo_bracket, Rxi, row_id_copy, F[idx_xi]);
            for (;count_unique_row > 0; --count_unique_row) {
                F[idx_xi](1);
                F[idx_xi](1, 1);
            }

            F[idx_xi].finalize(ro_bracket.size());
            ++idx_xi;
        } // for xi

    } // m_rad_counter_core__

    template <typename score_functor>
    int m_radix_sort_core__ (bool skip_unique_row, int xi, bool is_lsb, const pair_data_type& range_r, const std::vector<pair_int>& ro_bracket, std::vector<pair_int>& wo_bracket, std::vector<std::vector<int>>& Rxi, std::vector<int>& row_id, score_functor& F) const {
        int idx = 0;
        int count_unique_row = 0;

        for (auto& val : ro_bracket) {
            if (val.second == 1 && skip_unique_row) {
                if (!is_lsb) { wo_bracket.push_back({idx, 1}); }
                else { ++count_unique_row; }
                ++idx;
                continue;
            }

            for (int i = val.first; i < val.first + val.second; ++i) {
                data_type temp = D_[xi * m_ + row_id[i]];
                if (temp >= range_r.first && temp < range_r.second) { Rxi[temp].push_back(row_id[i]); }
            }

            if (is_lsb) { F(val.second); }

            for (int r = range_r.first, i = val.first; r < range_r.second; ++r) {
                if (Rxi[r].empty()) { continue; }
                for (auto& rid : Rxi[r]) { row_id[i++] = rid; }
                if (!is_lsb) { wo_bracket.push_back({idx, Rxi[r].size()}); }
                else { F(Rxi[r].size(), val.second); }
                idx += Rxi[r].size();
                Rxi[r].clear();
            } // for r_[xi]
        } // for ro_bracket

        return count_unique_row;
    } // m_radix_sort_core__

    int m_compute_q_pa__(const set_type& pa, const std::vector<pair_data_type>& state_range_pa) const {
        int q = 1;

        for (int xi = 0, idx_xi = 0; xi < n_; ++xi) {
            if (in_set(pa, xi)) {
                q *= state_range_pa[idx_xi].second - state_range_pa[idx_xi].first;
                ++idx_xi;
            }
        }

        return q;
    } // m_compute_q_pa__

    int n_;
    int m_;

    int max_r_ = -1;
    std::vector<data_type> r_; // we do not expect more than 255 states
    std::vector<int> idx_r_;
    std::vector<int> max_count_r_;
    // stores the row id's for each xi in state 'r'
    std::vector<std::vector<int>> Rxi_;

    std::vector<data_type> D_;

}; // class RadCounter


template <int N, typename data_type = uint8_t, typename Iter>
RadCounter<N, data_type> create_RadCounter(int n, int m, Iter it) {
    RadCounter<N, data_type> rad;

    rad.n_ = n;
    rad.m_ = m;
    rad.D_.reserve(n * m);
    rad.r_.resize(n, -1);
    rad.idx_r_.resize(n, -1);

    for (int i = 0; i < n * m; ++i, ++it) { rad.D_.push_back(*it); }

    for (int xi = 0, r_sum = 0; xi < n; ++xi) {
        auto min_max = std::minmax_element(rad.D_.begin() + xi * m, rad.D_.begin() + (xi + 1) * m);
        std::transform(rad.D_.begin() + xi * m, rad.D_.begin() + (xi + 1) * m, rad.D_.begin() + xi * m, [min_max](data_type x) { return x - *(min_max.first); } );
        rad.r_[xi] = *min_max.second - *min_max.first + 1;
        rad.max_r_ = std::max<int>(rad.max_r_, rad.r_[xi]);
        rad.idx_r_[xi] = r_sum;
        r_sum += rad.r_[xi];
    }

    rad.Rxi_.resize(rad.idx_r_[n - 1] + rad.r_[n - 1]);
    rad.max_count_r_.resize(rad.max_r_, -1);

    for (int xi = 0; xi < n; ++xi) {
        for (int i = 0; i < m; ++i) { rad.Rxi_[rad.idx_r_[xi] + rad.D_[xi * m + i]].push_back(i); }
        for (int r = 0, idx = 0; r < rad.r_[xi]; ++r) {
            const int temp = rad.Rxi_[rad.idx_r_[xi] + r].size();
            rad.max_count_r_[r] = rad.max_count_r_[r] > temp ? rad.max_count_r_[r] : temp;
            idx += temp;
        }
    }

    return rad;
} // create_RadCounter

#endif // RAD_COUNTER_HPP
