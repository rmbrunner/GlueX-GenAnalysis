// knuth_bins.hpp
#pragma once

#include <vector>

namespace Knuth {

/**
 * Compute the optimal histogram bin width using Knuth's rule.
 * Implementation in knuth_bins.cpp (link with -lMinuit2 -lMathCore).
 *
 * @param data  A non‐empty vector of data values (size ≥ 2).
 * @return      Optimal fixed bin width Δx.
 * @throws      std::invalid_argument if data.size()<2,
 *              std::runtime_error if all values are identical.
 */
double computeNumberBins(const std::vector<double>& data);

} // namespace knuth_hist

