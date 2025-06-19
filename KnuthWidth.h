// knuth_bins.hpp
#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <memory>

#include "Math/Minimizer.h"
#include "Math/Factory.h"
#include "Math/Functor.h"

namespace knuth_hist {

//——— Knuth’s log‑posterior F(M) for a given M ———
inline double knuthF(const std::vector<double>& data, double Mval) {
    if (Mval <= 0.0) return -INFINITY;
    std::size_t M = static_cast<std::size_t>(std::round(Mval));
    if (M == 0) return -INFINITY;

    auto n = data.size();
    double mn = data.front(), mx = data.back();
    double width = (mx - mn) / double(M);

    std::vector<std::size_t> counts(M, 0);
    for (auto x : data) {
        std::size_t k = std::min<std::size_t>(
            M - 1,
            static_cast<std::size_t>(std::floor((x - mn) / width))
        );
        counts[k]++;
    }

    double F = n * std::log(double(M))
        + std::lgamma(0.5 * M)
        - double(M) * std::lgamma(0.5)
        - std::lgamma(double(n) + 0.5 * M);

    for (auto nk : counts) {
        F += std::lgamma(double(nk) + 0.5);
    }

    return -F; // Negative log-posterior for minimization
}

//——— compute optimal bin width using ROOT minimizer ———
inline double knuth_bin_width(const std::vector<double>& data_) {
    if (data_.size() < 2)
        throw std::invalid_argument("need at least two data points");

    std::vector<double> data = data_;
    std::sort(data.begin(), data.end());

    auto n = data.size();
    double mn = data.front(), mx = data.back();

    ROOT::Math::Functor f([&](const double* x) { return knuthF(data, x[0]); }, 1);
    std::unique_ptr<ROOT::Math::Minimizer> minimizer(
        ROOT::Math::Factory::CreateMinimizer("Minuit2", "Simplex"));

    minimizer->SetFunction(f);
    minimizer->SetLimitedVariable(0, "M", 10.0, 1.0, 1.0, std::min<double>(200, n));
    minimizer->Minimize();

    double Mopt = std::round(minimizer->X()[0]);
    if (Mopt < 1) Mopt = 1;

    return (mx - mn) / Mopt;
}

} // namespace knuth_hist


