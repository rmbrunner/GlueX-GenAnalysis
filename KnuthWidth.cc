#include "KnuthWidth.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>

#include "Math/Factory.h"
#include "Math/Functor.h"
#include "Math/Minimizer.h"

namespace Knuth
{

static double knuthF(const std::vector<double> &data, const double *x)
{
    double Mval = x[0];
    if (Mval <= 0.0)
    {
        return -INFINITY;
    }
    std::size_t M = static_cast<std::size_t>(Mval);
    if (M == 0)
    {
        return -INFINITY;
    }

    auto n = data.size();
    double mn = data.front(), mx = data.back();
    if (mn == mx)
    {
        return -INFINITY;
    }
    double width = (mx - mn) / double(M);
    if (width <= 0.0)
    {
        return -INFINITY;
    }

    std::vector<std::size_t> counts(M, 0);
    for (auto xval : data)
    {
        std::size_t k =
            std::min<std::size_t>(M - 1, static_cast<std::size_t>(std::floor((xval - mn) / width)));
        counts[k]++;
    }

    double F = n * std::log(double(M)) + std::lgamma(0.5 * M) - double(M) * std::lgamma(0.5) -
               std::lgamma(double(n) + 0.5 * M);
    for (auto nk : counts)
    {
        F += std::lgamma(double(nk) + 0.5);
    }

    return -F;
}

// Public API: optimize M via ROOT Minuit (regular)
double computeNumberBins(const std::vector<double> &data_)
{
    if (data_.size() < 2)
    {
        throw std::invalid_argument("need at least two data points");
    }
    // copy & sort
    std::vector<double> data = data_;
    std::sort(data.begin(), data.end());

    auto n = data.size();
    double mn = data.front(), mx = data.back();
    if (mn == mx)
    {
        throw std::runtime_error("data range is zero");
    }

    // double phys_bin_width = 0.020;
    // size_t M0 = std::max<size_t>(1, std::round((mx - mn) / phys_bin_width));
    size_t M0 = 104;
    size_t Mmax = std::min(n, std::max<size_t>(1, M0 * 2));

    // std::cerr << "[knuth] Physical seed M0=" << M0 << ", Mmax=" << Mmax << std::endl;

    // define functor
    ROOT::Math::Functor f([&data](const double *x) { return knuthF(data, x); }, 1);
    double test = 80;
    std::cout << knuthF(data, &test) << std::endl;

    // create minimizer
    auto minimizer = std::unique_ptr<ROOT::Math::Minimizer>(
        ROOT::Math::Factory::CreateMinimizer("Minuit", "Migrad"));
    if (!minimizer)
    {
        throw std::runtime_error("Failed to create ROOT minimizer");
    }

    minimizer->SetFunction(f);
    minimizer->SetPrintLevel(2);
    minimizer->SetStrategy(2);
    // minimizer->SetErrorDef(0.5);
    // minimizer->SetTolerance(1e-4);

    // parameter M: seed at M0, step M0/10
    double initM = double(M0);
    double step = std::max(1.0, M0 * 0.1);
    // std::cerr << "[knuth] init M = " << initM << ", step = " << step << std::endl;
    minimizer->SetLimitedVariable(0, "M", initM, step, 1.0, double(Mmax));

    // bool ok = minimizer->Minimize();
    //std::cerr << "[knuth] Minimize returned " << ok << std::endl;

    const double *xs = minimizer->X();
    // std::cerr << "[knuth] Final Mval = " << (xs ? xs[0] : NAN) << std::endl;
    // std::cerr << "[knuth] MinValue = " << minimizer->MinValue() << std::endl;
    // std::cerr << "[knuth] Status = " << minimizer->Status() << std::endl;

    // if (!ok || !xs || std::isnan(xs[0]) || xs[0] < 1.0 || minimizer->Status() != 0) {
    //     // fallback brute-force
    //     std::cerr << "[knuth] Fallback brute-force scan" << std::endl;
    //     double bestF = -INFINITY;
    //     size_t bestM = 1;
    //     for (size_t M = 1; M <= Mmax; ++M) {
    //         double val = static_cast<double>(M);
    //         double v = knuthF(data, &val);
    //         if (v > bestF) { bestF = v; bestM = M; }
    //     }
    //     double Mopt = bestM;
    //     std::cerr << "[knuth] Brute M=" << Mopt << std::endl;
    //     return Mopt;
    // }

    double Mopt = xs[0];
    if (Mopt < 1)
    {
        Mopt = 1;
    }
    return Mopt;
}

} // namespace Knuth
