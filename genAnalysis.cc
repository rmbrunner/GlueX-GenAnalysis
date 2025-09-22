// #include "KnuthWidth.h"
#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TF1.h>
#include <TFile.h>
#include <TH1D.h>
#include <TKey.h>
#include <TLegend.h>
#include <TTree.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <ostream>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace ROOT;
using std::map;
using std::string;
using std::vector;

static std::vector<std::string> splitTokens(const std::string &s)
{
    std::vector<std::string> toks;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, '_'))
    {
        toks.push_back(item);
    }
    return toks;
}

string particleNameToLatex(const string &branch)
{
    static const map<string, string> particleMap = {
        {"PiMinus", "#pi^{-}"}, {"PiPlus", "#pi^{+}"},  {"KMinus", "K^{-}"},
        {"KPlus", "K^{+}"},     {"Proton", "p"},        {"AntiProton", "#bar{p}"},
        {"Neutron", "n"},       {"Electron", "e^{-}"},  {"Positron", "e^{+}"},
        {"Photon", "#gamma"},   {"KShort", "K^{0}_{S}"}};

    string s = branch;
    if (s.rfind("mass_", 0) == 0)
    {
        s = s.substr(5);
    }

    std::stringstream ss(s);
    string segment;
    vector<string> particles;
    while (std::getline(ss, segment, '_'))
    {
        std::regex re("([A-Za-z]+)([0-9]*)$");
        std::smatch match;
        if (std::regex_match(segment, match, re))
        {
            string base = match[1];
            string index = match[2];
            auto it = particleMap.find(base);
            string latex = (it != particleMap.end()) ? it->second : base;
            if (!index.empty())
            {
                latex += "_{" + index + "}";
            }
            particles.push_back(latex);
        }
    }
    string joined;
    for (size_t i = 0; i < particles.size(); ++i)
    {
        if (i)
        {
            joined += " ";
        }
        joined += particles[i];
    }
    return joined;
}

// histogram helpers for the generated macro
string makeHisto1D(const string &b, size_t bins, double min, double max, bool showErrors,
                   const string &weightCol = "")
{
    std::ostringstream out;
    double binWidth = static_cast<int>(1000 * ((max - min) / bins));
    string histName = "h1_" + b;
    string title = "Mass[" + particleNameToLatex(b) + "] (GeV)";
    if (weightCol.empty())
    {
        out << "  auto " << histName << " = df.Histo1D({\"" << histName << "\",\""
            << "" << "\"," << bins << "," << min << "," << max << "}, \"" << b << "\");\n";
    }
    else
    {
        out << "  auto " << histName << " = df.Histo1D({\"" << histName << "\",\""
            << "" << "\"," << bins << "," << min << "," << max << "}, \"" << b << "\", \""
            << weightCol << "\");\n";
    }
    out << histName << "->GetXaxis()->SetTitle(\"" << title << "\");" << std::endl;
    out << histName << "->GetYaxis()->SetTitle(\"Counts / " << binWidth << " MeV\");" << std::endl;
    out << histName << "->Write();\n";
    out << "progress++;\n";
    out << "std::cout << \"\\r\" << \"Progress: \" << progress << \"/\" << total << std::flush;\n";
    return out.str();
}

string makeHisto2D(const string &x, const string &y, size_t bx, double xmin, double xmax, size_t by,
                   double ymin, double ymax, const string &xlabel, const string &ylabel,
                   bool showErrors, const string &dfName, const string &weightCol = "")
{
    std::ostringstream out;
    string histName = "h2_" + x + "_" + y;
    string title = xlabel + " vs. " + ylabel;
    if (weightCol.empty())
    {
        out << "  auto " << histName << " = " << dfName << ".Histo2D({\"" << histName << "\",\""
            << ""
            << "\"," << bx << "," << xmin << "," << xmax << "," << by << "," << ymin << "," << ymax
            << "}, \"" << x << "\", \"" << y << "\");\n";
    }
    else
    {
        out << "  auto " << histName << " = " << dfName << ".Histo2D({\"" << histName << "\",\""
            << ""
            << "\"," << bx << "," << xmin << "," << xmax << "," << by << "," << ymin << "," << ymax
            << "}, \"" << x << "\", \"" << y << "\", \"" << weightCol << "\");\n";
    }
    out << histName << "->GetXaxis()->SetTitle(\"" << xlabel << "\");" << std::endl;
    out << histName << "->GetYaxis()->SetTitle(\"" << ylabel << "\");" << std::endl;
    out << histName << "->Write();\n";
    out << "progress++;\n";
    out << "std::cout << \"\\r\" << \"Progress: \" << progress << \"/\" << total << std::flush;\n";
    return out.str();
}

// ---------------------------------------------------------------------
// Use the user-provided peak+sigma estimator (fits a gaussian to a local window)
// ---------------------------------------------------------------------
static std::pair<double, double> estimatePeakAndSigmaFromVec(const vector<double> &vec,
                                                             int nbins = 400)
{
    if (vec.empty())
    {
        return {0., 0.};
    }
    double xmin = 1e300, xmax = -1e300;
    for (double v : vec)
    {
        if (std::isfinite(v))
        {
            xmin = std::min(xmin, v);
            xmax = std::max(xmax, v);
        }
    }
    if (!(xmin < xmax))
    {
        return {0., 0.};
    }
    TH1D htmp("htmp", "", nbins, xmin, xmax);
    for (double v : vec)
    {
        if (std::isfinite(v))
        {
            htmp.Fill(v);
        }
    }

    int imax = htmp.GetMaximumBin();
    double peak = htmp.GetBinCenter(imax);
    double fullRange = xmax - xmin;
    double window = std::max(0.03, 0.05 * fullRange);
    double fitLow = std::max(xmin, peak - window);
    double fitHigh = std::min(xmax, peak + window);
    TF1 f("fgaus", "gaus", fitLow, fitHigh);
    int fitStatus = 1;

    try
    {
        fitStatus = htmp.Fit(&f, "QRS", "", fitLow, fitHigh);
    }
    catch (...)
    {
        fitStatus = 1;
    }

    double mean = 0., sigma = 0.;

    if (fitStatus == 0 || f.GetNpar() >= 3)
    {
        mean = f.GetParameter(1);
        sigma = fabs(f.GetParameter(2));
        if (!(sigma > 0) || !(mean > xmin && mean < xmax))
        {
            mean = htmp.GetMean();
            sigma = htmp.GetRMS();
        }
    }
    else
    {
        mean = htmp.GetMean();
        sigma = htmp.GetRMS();
    }
    return {mean, sigma};
}
// ---------------------------------------------------------------------

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input.root> <selectedBranches.csv> <output.cc> [--fractal-sb|-f [levels]] "
                     "[showErrors=0]\n";
        return 1;
    }
    string inputFile = argv[1];
    string cpy = inputFile;
    std::size_t pos = cpy.rfind('_');
    if (pos != std::string::npos)
    {
        cpy.erase(pos); // erase from '_' onwards
    }

    string csvFile = argv[2];
    string outFile = argv[3];

    // parse optional args
    bool fractalSB = false;
    int sbLevels = 3; // default fractal depth
    bool showErrors = false;

    // scan remaining argv for our flags and showErrors (last)
    for (int i = 4; i < argc; ++i)
    {
        string a = argv[i];
        if (a == "--fractal-sb" || a == "-f")
        {
            fractalSB = true;
            // check next arg if exists and is integer -> levels
            if (i + 1 < argc)
            {
                // try parse integer
                try
                {
                    int v = std::stoi(argv[i + 1]);
                    if (v > 1)
                    {
                        sbLevels = v;
                        ++i; // consume
                    }
                }
                catch (...)
                { /* not an integer; ignore */
                }
            }
        }
        else
        {
            // fallback: interpret as showErrors integer if it looks like one
            try
            {
                int val = std::stoi(a);
                showErrors = (val != 0);
            }
            catch (...)
            { /* ignore unknown args */
            }
        }
    }

    size_t totalPlots = 0;

    // ------------------------------
    // USER: define resonance groups here in the generator (this example is used
    // by the generator to find branches named "mass_<suffix>" and compute
    // sideband windows). The same array is copied into the generated macro for
    // user reference/editing.
    // ------------------------------
    // format: pair<resonance_name, branch_suffix_string>
    // example: {"omega", "PiPlus_PiMinus1_Photon1_Photon2"}
    // ------------------------------
    static const std::pair<std::string, std::string> RESONANCE_GROUPS[] = {
        {"omega", "PiPlus_PiMinus1_Photon1_Photon2"}, {"Lambda", "PiMinus2_Proton"}};
    // ------------------------------

    // parameters controlling widths (you can change these defaults)
    double n_sig = 2.0; // signal window = mean +/- n_sig * sigma
    double sb_inner =
        3.0; // (kept for backward compatibility with previous single-sideband calculation)
    double sb_outer = 5.0; // outermost sigma for sidebands

    std::ifstream ifs(csvFile);
    vector<TString> sel;
    std::string line;
    // read in and sanitize combo names
    string finalState;
    while (std::getline(ifs, line))
    {
        if (!line.empty())
        {
            line.erase(line.find_last_not_of(" \n\r\t") + 1);
            sel.push_back(line);
            finalState = line;
        }
    }

    TFile *f = TFile::Open(inputFile.c_str(), "READ");
    if (!f || f->IsZombie())
    {
        std::cerr << "ERROR: Cannot open file " << inputFile << "\n";
        return 1;
    }
    TIter itKey(f->GetListOfKeys());
    TKey *key;
    string treeName;
    while ((key = (TKey *)itKey()))
    {
        if (std::string(key->GetClassName()) == "TTree")
        {
            treeName = key->GetName();
            break;
        }
    }
    if (treeName.empty())
    {
        std::cerr << "ERROR: No TTree found in file.\n";
        return 1;
    }

    RDataFrame df(treeName, inputFile);
    auto allCols = df.GetColumnNames();

    // find mass branches corresponding to user-selected final states
    vector<string> cols;
    for (const auto &b : allCols)
    {
        string stripped = b;
        const string prefix = "mass_";
        if (stripped.rfind(prefix, 0) == 0)
        {
            stripped = stripped.substr(prefix.size());
        }
        TString tmp = stripped;
        for (auto &i : sel)
        {
            TString thing = i;

            if (i == tmp)
            {
                cols.push_back(b);
                break;
            }
        }
    }

    // -----------------------
    // For each resonance group, try to locate the corresponding mass branch
    // and compute mean & sigma (used to create signal/sideband windows).
    // Also compute fractal-level boundaries & coefficients if fractalSB is enabled.
    // -----------------------
    struct SBBounds
    {
        std::string branch; // full branch name: e.g. mass_PiPlus_...
        double mean = NAN;
        double sigma = NAN;
        // basic windows (for non-fractal, backward compat)
        double sig_lo = NAN, sig_hi = NAN;
        double left_lo = NAN, left_hi = NAN;
        double right_lo = NAN, right_hi = NAN;
        // fractal: per-level inner/outer (levels count)
        std::vector<double> level_inner; // positive distances in sigma: n_sig + ...
        std::vector<double> level_outer;
        std::vector<double> level_inner_val; // absolute mass values (mean - inner*sigma etc)
        std::vector<double> level_outer_val;
        std::vector<double> coeff; // coefficient per level
    };

    std::map<std::string, SBBounds> resonanceBounds; // key: resonance name (omega, Lambda)

    // Build a set of all column names for quick membership
    std::unordered_set<std::string> allColSet;
    for (const auto &c : allCols)
    {
        allColSet.insert((std::string)c);
    }

    // For each config entry, find mass_<suffix> branch and compute stats
    for (const auto &entry : RESONANCE_GROUPS)
    {
        std::string rname = entry.first;
        std::string suffix = entry.second;
        std::string branch = "mass_" + suffix;
        if (allColSet.find(branch) == allColSet.end())
        {
            std::cerr << "Warning: resonance group '" << rname << "' -> branch '" << branch
                      << "' not found in file. Skipping.\n";
            continue;
        }
        // take values
        auto vec = df.Take<double>(branch).GetValue();
        if (vec.empty())
        {
            std::cerr << "Warning: branch '" << branch << "' empty. Skipping " << rname << ".\n";
            continue;
        }

        // ---- use fitter-based estimate of peak & sigma (user-provided routine) ----
        auto [mean, sigma] = estimatePeakAndSigmaFromVec(vec, 400);
        if (!(sigma > 0) || !(mean > -1e299 && mean < 1e299)) // sanity fallback
        {
            // fall back to simple RMS/mean if fit failed
            mean = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
            double acc = 0.0;
            for (auto v : vec)
            {
                acc += (v - mean) * (v - mean);
            }
            sigma = (vec.size() > 1) ? std::sqrt(acc / (vec.size() - 1)) : 0.0;
        }
        // -------------------------------------------------------------------------

        SBBounds b;
        b.branch = branch;
        b.mean = mean;
        b.sigma = sigma;
        b.sig_lo = mean - n_sig * sigma;
        b.sig_hi = mean + n_sig * sigma;
        b.left_lo = mean - sb_outer * sigma;
        b.left_hi = mean - sb_inner * sigma;
        b.right_lo = mean + sb_inner * sigma;
        b.right_hi = mean + sb_outer * sigma;

        // Fractal setup if requested: create sbLevels nested rings from n_sig..sb_outer
        if (fractalSB)
        {
            int L = std::max(2, sbLevels);
            b.level_inner.resize(L);
            b.level_outer.resize(L);
            b.level_inner_val.resize(L);
            b.level_outer_val.resize(L);
            b.coeff.resize(L);
            // compute level edges in sigma-units:
            // level 0 = [0, n_sig]
            // levels 1..L-1 split [n_sig, sb_outer] linearly
            double range = std::max(0.0, sb_outer - n_sig);
            double delta = (L > 1) ? (range / double(L - 1)) : 0.0;
            for (int l = 0; l < L; ++l)
            {
                if (l == 0)
                {
                    b.level_inner[l] = 0.0;
                    b.level_outer[l] = n_sig;
                }
                else
                {
                    b.level_inner[l] = n_sig + (l - 1) * delta;
                    b.level_outer[l] = n_sig + (l)*delta;
                }
                // convert to absolute mass ranges (we'll use these numeric values directly in
                // generated code)
                b.level_inner_val[l] =
                    mean - b.level_outer[l] * sigma; // left-most outer for lower bound
                b.level_outer_val[l] =
                    mean -
                    b.level_inner[l] * sigma; // left-most inner for upper bound (for left side)
                // coeff rule: pow(-0.5, level)
                b.coeff[l] = std::pow(-0.5, l);
            }
        }

        resonanceBounds[rname] = b;

        std::cout << "Resonance '" << rname << "' -> branch " << branch << ": mean=" << mean
                  << " sigma=" << sigma << " sig[" << b.sig_lo << "," << b.sig_hi << "] "
                  << " Lsb[" << b.left_lo << "," << b.left_hi << "] Rsb[" << b.right_lo << ","
                  << b.right_hi << "]\n";

        if (fractalSB)
        {
            std::cout << "  Fractal levels = " << (int)b.level_inner.size() << " coeffs: ";
            for (double c : b.coeff)
            {
                std::cout << c << " ";
            }
            std::cout << "\n";
        }
    }

    std::ofstream ofs(outFile);

    ofs << "#include <ROOT/RDataFrame.hxx>\n"
        << "#include <TFile.h>\n"
        << "#include <TCanvas.h>\n"
        << "#include <TKey.h>\n"
        << "#include <TGaxis.h>\n"
        << "#include <string>\n"
        << "#include <TLorentzVector.h>\n"
        << "using namespace ROOT;\n"
        << "void plots(){\n"
        << "  TFile *f = TFile::Open(\"" << inputFile << "\", \"READ\");\n"
        << "  if (!f || f->IsZombie()) throw std::runtime_error(\"Cannot open "
           "\\\" "
        << inputFile << "\\\"\");\n"
        << "  TIter itKey(f->GetListOfKeys()); TKey *key; std::string tree;\n"
        << "  while ((key = (TKey*)itKey())) {\n    if "
           "(std::string(key->GetClassName()) == \"TTree\") { tree = "
           "key->GetName(); break; }\n  }\n"
        << "  if (tree.empty()) throw std::runtime_error(\"No TTree\");\n"
        << "  auto df = RDataFrame(tree, \"" << cpy << "*.root\");\n\n";

    // write user-visible resonance config into generated macro
    ofs << "  // ----------------------\n";
    ofs << "  // Resonance groups (user-configurable)\n";
    ofs << "  // Format: pair<resonance_name, branch_suffix>\n";
    ofs << "  // Example (generator used these to compute windows):\n";
    ofs << "  static const std::pair<std::string, std::string> RESONANCE_GROUPS[] = {\n";
    for (const auto &entry : RESONANCE_GROUPS)
    {
        ofs << "    {\"" << entry.first << "\", \"" << entry.second << "\"},\n";
    }
    ofs << "  };\n";
    ofs << "  // ----------------------\n\n";

    // Print computed numeric windows into the generated macro as literals
    ofs << "  // ----------------------\n";
    if (!fractalSB)
    {
        ofs << "  // Sideband windows computed by generator (numeric literals) - classic 3x3 "
               "style\n";
    }
    else
    {
        ofs << "  // Fractal sideband mode enabled. Numeric level boundaries & coefficients "
               "below.\n";
    }

    for (const auto &p : resonanceBounds)
    {
        const auto &r = p.first;
        const auto &b = p.second;
        ofs << "  // resonance '" << r << "' for branch \"" << b.branch << "\"\n";
        ofs << "  const double " << "res_" << r << "_mean = " << b.mean << ";\n";
        ofs << "  const double " << "res_" << r << "_sigma = " << b.sigma << ";\n";
        ofs << "  const double " << "res_" << r << "_sig_lo = " << b.sig_lo << ";\n";
        ofs << "  const double " << "res_" << r << "_sig_hi = " << b.sig_hi << ";\n";
        ofs << "  const double " << "res_" << r << "_left_lo = " << b.left_lo << ";\n";
        ofs << "  const double " << "res_" << r << "_left_hi = " << b.left_hi << ";\n";
        ofs << "  const double " << "res_" << r << "_right_lo = " << b.right_lo << ";\n";
        ofs << "  const double " << "res_" << r << "_right_hi = " << b.right_hi << ";\n";
        if (fractalSB)
        {
            int L = (int)b.level_inner.size();
            ofs << "  // fractal levels: " << L << "\n";
            // print coefficients array
            ofs << "  const int res_" << r << "_levels = " << L << ";\n";
            ofs << "  const double res_" << r << "_coeffs[] = {";
            for (int l = 0; l < L; ++l)
            {
                if (l)
                {
                    ofs << ", ";
                }
                ofs << b.coeff[l];
            }
            ofs << "};\n";
            // print numeric boundaries for each level (we'll provide left and right pairs)
            // We'll emit level boundaries in mass units for left and right symmetric check
            for (int l = 0; l < L; ++l)
            {
                double left_lo = b.mean - b.level_outer[l] * b.sigma;
                double left_hi = b.mean - b.level_inner[l] * b.sigma;
                double right_lo = b.mean + b.level_inner[l] * b.sigma;
                double right_hi = b.mean + b.level_outer[l] * b.sigma;
                ofs << "  // level " << l << " boundaries (mass): left[" << left_lo << ","
                    << left_hi << "] right[" << right_lo << "," << right_hi << "]\n";
                ofs << "  const double res_" << r << "_lvl" << l << "_left_lo = " << left_lo
                    << ";\n";
                ofs << "  const double res_" << r << "_lvl" << l << "_left_hi = " << left_hi
                    << ";\n";
                ofs << "  const double res_" << r << "_lvl" << l << "_right_lo = " << right_lo
                    << ";\n";
                ofs << "  const double res_" << r << "_lvl" << l << "_right_hi = " << right_hi
                    << ";\n";
            }
        }
        ofs << "\n";
    }
    ofs << "  // ----------------------\n\n";

    // Create per-resonance 1D sideband weight columns in the generated macro
    // If fractalSB is enabled, create weights using per-level coefficients.
    for (const auto &p : resonanceBounds)
    {
        const auto &r = p.first;
        const auto &b = p.second;
        string wname = "sb_w_" + r;
        ofs << "  // define 1D sideband weight for resonance '" << r << "'\n";
        if (!fractalSB)
        {
            ofs << "  df = df.Define(\"" << wname << "\", [](double m){\n";
            ofs << "    if (m >= " << b.sig_lo << " && m <= " << b.sig_hi << ") return 1.0;\n";
            ofs << "    if ((m >= " << b.left_lo << " && m <= " << b.left_hi
                << ") || (m >= " << b.right_lo << " && m <= " << b.right_hi << ")) return -0.5;\n";
            ofs << "    return 0.0;\n";
            ofs << "  }, {\"" << b.branch << "\"});\n\n";
        }
        else
        {
            int L = (int)b.level_inner.size();
            ofs << "  df = df.Define(\"" << wname << "\", [](double m){\n";
            // check levels from 0..L-1
            for (int l = 0; l < L; ++l)
            {
                // signal level 0 is symmetric between left and right
                if (l == 0)
                {
                    ofs << "    if (m >= " << (b.mean - b.level_outer[0] * b.sigma)
                        << " && m <= " << (b.mean + b.level_outer[0] * b.sigma) << ") return "
                        << b.coeff[0] << ";\n";
                }
                else
                {
                    double left_lo = b.mean - b.level_outer[l] * b.sigma;
                    double left_hi = b.mean - b.level_inner[l] * b.sigma;
                    double right_lo = b.mean + b.level_inner[l] * b.sigma;
                    double right_hi = b.mean + b.level_outer[l] * b.sigma;
                    ofs << "    if ((m >= " << left_lo << " && m <= " << left_hi
                        << ") || (m >= " << right_lo << " && m <= " << right_hi << ")) return "
                        << b.coeff[l] << ";\n";
                }
            }
            ofs << "    return 0.0;\n";
            ofs << "  }, {\"" << b.branch << "\"});\n\n";
        }
    }

    ofs << "TFile* histograms =TFile::Open(\"histograms.root\", \"RECREATE\"); \n"
        << "std::ifstream totalplots(\"totalPlots.txt\");\n"
        << "std::string total;\n"
        << "std::getline(totalplots,total);\n"
        << "size_t numplots = std::stoul(total);\n"
        << "size_t progress = 0;\n"
        << "histograms->cd();\n";

    std::cout << "Creating 1D Histograms..." << std::endl;

    // map from branch -> resonance name (if any)
    std::map<std::string, std::string> branchToRes;
    for (const auto &p : resonanceBounds)
    {
        branchToRes[p.second.branch] = p.first;
    }

    for (auto &b : cols)
    {
        double minv = df.Min<double>(b).GetValue();
        double maxv = df.Max<double>(b).GetValue();
        auto data_b = df.Take<double>(b).GetValue();
        size_t bins1 = 300; // static_cast<size_t>(Knuth::computeNumberBins(data_b));
        string weightCol = "";
        auto it = branchToRes.find(b);
        if (it != branchToRes.end())
        {
            weightCol = "sb_w_" + it->second; // use the 1D sideband weight for this resonance
        }
        ofs << makeHisto1D(b, bins1, minv, maxv, showErrors, weightCol);
        totalPlots++;
    }

    std::cout << "Creating Angular Distribution Plots..." << std::endl;
    for (auto &m : cols)
    {
        // only match if mass column is for the same particle
        string mass = m;
        const string prefix = "mass_";
        if (m.rfind(prefix, 0) == 0)
        {
            m = m.substr(prefix.size());
        }
        string costh = "costh_";
        string phi = "phi_";
        string dfName = "df";
        string clab = costh + "lab_" + m, plab = phi + "lab_" + m;
        string cGJ = "costh_GJ_" + m, pGJ = "phi_GJ_" + m, cH = "costh_H_" + m, pH = "phi_H_" + m;

        double xmin = df.Min<double>(mass).GetValue();
        double xmax = df.Max<double>(mass).GetValue();

        double clabmin = 0;
        double clabmax = 0;
        double plabmin = 0;
        double plabmax = 0;
        double cGJmin = 0;
        double cGJmax = 0;
        double pGJmin = 0;
        double pGJmax = 0;
        double cHmin = 0;
        double cHmax = 0;
        double pHmin = 0;
        double pHmax = 0;

        if (m != finalState)
        {
            clabmin = df.Min<double>(clab).GetValue();
            clabmax = df.Max<double>(clab).GetValue();
            plabmin = df.Min<double>(plab).GetValue();
            plabmax = df.Max<double>(plab).GetValue();
            cGJmin = df.Min<double>(cGJ).GetValue();
            cGJmax = df.Max<double>(cGJ).GetValue();
            pGJmin = df.Min<double>(pGJ).GetValue();
            pGJmax = df.Max<double>(pGJ).GetValue();
            cHmin = df.Min<double>(cH).GetValue();
            cHmax = df.Max<double>(cH).GetValue();
            pHmin = df.Min<double>(pH).GetValue();
            pHmax = df.Max<double>(pH).GetValue();
        }

        size_t bm = 300; // static_cast<size_t>(Knuth::computeNumberBins(data_m));
        size_t ba = 300; // static_cast<size_t>(Knuth::computeNumberBins(data_a));

        string xtitle = "Mass[" + particleNameToLatex(mass) + "] (GeV)";
        string ytitle = "";
        ytitle = particleNameToLatex(clab);
        ofs << makeHisto2D(mass, clab, bm, xmin, xmax, ba, clabmin, clabmax, xtitle, ytitle,
                           showErrors, dfName);
        totalPlots++;
        ytitle = particleNameToLatex(plab);
        ofs << makeHisto2D(mass, plab, bm, xmin, xmax, ba, plabmin, plabmax, xtitle, ytitle,
                           showErrors, dfName);
        totalPlots++;
        if (m != finalState)
        {
            ytitle = particleNameToLatex(cGJ);
            ofs << makeHisto2D(mass, cGJ, bm, xmin, xmax, ba, cGJmin, cGJmax, xtitle, ytitle,
                               showErrors, dfName);
            totalPlots++;
            ytitle = particleNameToLatex(pGJ);
            ofs << makeHisto2D(mass, pGJ, bm, xmin, xmax, ba, pGJmin, pGJmax, xtitle, ytitle,
                               showErrors, dfName);
            totalPlots++;
            ytitle = particleNameToLatex(cH);
            ofs << makeHisto2D(mass, cH, bm, xmin, xmax, ba, cHmin, cHmax, xtitle, ytitle,
                               showErrors, dfName);
            totalPlots++;
            ytitle = particleNameToLatex(pH);
            ofs << makeHisto2D(mass, pH, bm, xmin, xmax, ba, pHmin, pHmax, xtitle, ytitle,
                               showErrors, dfName);
            totalPlots++;
        }
    }

    std::cout << "Creating 2D Mass Correlation Plots..." << std::endl;
    for (size_t i = 0; i < cols.size(); ++i)
    {
        for (size_t j = i + 1; j < cols.size(); ++j)
        {
            string dfName = "df";
            string x = "mass_" + cols[i];
            string y = "mass_" + cols[j];
            double xmin = df.Min<double>(x).GetValue();
            double xmax = df.Max<double>(x).GetValue();
            double ymin = df.Min<double>(y).GetValue();
            double ymax = df.Max<double>(y).GetValue();
            auto data_x = df.Take<double>(x).GetValue();
            auto data_y = df.Take<double>(y).GetValue();
            size_t bx = 300; // static_cast<size_t>(Knuth::computeNumberBins(data_x));
            size_t by = 300; // static_cast<size_t>(Knuth::computeNumberBins(data_y));
            string xtitle = "Mass[" + particleNameToLatex(x) + "] (GeV)";
            string ytitle = "Mass[" + particleNameToLatex(y) + "] (GeV)";

            // Determine if x and/or y are resonances we computed
            std::string res_x = "";
            std::string res_y = "";
            for (const auto &p : resonanceBounds)
            {
                if (p.second.branch == x)
                {
                    res_x = p.first;
                }
                if (p.second.branch == y)
                {
                    res_y = p.first;
                }
            }

            // If both are resonances same one, use 1D weight; complex 2D combined weights will be
            // defined for Dalitz below
            ofs << makeHisto2D(x, y, bx, xmin, xmax, by, ymin, ymax, xtitle, ytitle, showErrors,
                               dfName,
                               (res_x.empty() && res_y.empty()
                                    ? ""
                                    : (res_x == res_y && !res_x.empty() ? "sb_w_" + res_x : "")));
            totalPlots++;
        }
    }

    std::cout << "Creating 2D Dalitz Plots..." << std::endl;
    size_t counter = 0;
    std::vector<std::vector<std::string>> tokens;
    tokens.reserve(cols.size());
    for (auto &c : cols)
    {
        tokens.push_back(splitTokens(c));
    }
    for (size_t i = 0; i < cols.size(); ++i)
    {
        std::unordered_set<std::string> set_i(tokens[i].begin(), tokens[i].end());
        for (size_t j = i + 1; j < cols.size(); ++j)
        {
            bool hasCommon = false;
            for (auto &t : tokens[j])
            {
                if (set_i.count(t))
                {
                    hasCommon = true;
                    break;
                }
                if (!hasCommon)
                {
                    continue;
                }
            }
            string b1 = "mass_" + cols[i], b2 = "mass_" + cols[j];
            string sq1 = b1 + "_sq";
            string sq2 = b2 + "_sq";
            string dfName = "df2_" + std::to_string(counter);
            ofs << "  auto " << dfName << " = df.Define(\"" << sq1
                << "\",[](double x){return x*x;}, {\"" << b1 << "\"}).Define(\"" << sq2
                << "\",[](double x){return x*x;}, {\"" << b2 << "\"});\n";

            double min1 = df.Min<double>(b1).GetValue();
            double max1 = df.Max<double>(b1).GetValue();
            double min2 = df.Min<double>(b2).GetValue();
            double max2 = df.Max<double>(b2).GetValue();
            auto vec1 = df.Take<double>(b1).GetValue();
            auto vec2 = df.Take<double>(b2).GetValue();
            std::vector<double> sqx, sqy;
            sqx.reserve(vec1.size());
            sqy.reserve(vec2.size());
            for (auto v : vec1)
            {
                sqx.push_back(v * v);
            }
            for (auto v : vec2)
            {
                sqy.push_back(v * v);
            }
            size_t binsx = 300; // static_cast<size_t>(Knuth::computeNumberBins(sqx));
            size_t binsy = 300; // static_cast<size_t>(Knuth::computeNumberBins(sqy));
            string xtitle = "Mass[" + particleNameToLatex(b1) + "]^{2} (GeV^{2})";
            string ytitle = "Mass[" + particleNameToLatex(b2) + "]^{2} (GeV^{2})";

            // If both mass branches correspond to resonance groups that we computed, create a
            // combined 2D weight.
            std::string r1 = "", r2 = "";
            for (const auto &p : resonanceBounds)
            {
                if (p.second.branch == b1)
                {
                    r1 = p.first;
                }
                if (p.second.branch == b2)
                {
                    r2 = p.first;
                }
            }

            string weightName = "";
            if (!r1.empty() && !r2.empty())
            {
                // create combined weight name
                weightName = "sb_w_" + r1 + "_" + r2;
                // build the lambda with numeric literals for boundaries
                const auto &A = resonanceBounds[r1];
                const auto &B = resonanceBounds[r2];

                ofs << "  // combined 2D sideband weight for resonances '" << r1 << "' and '" << r2
                    << "'\n";
                ofs << "  " << dfName << " = " << dfName << ".Define(\"" << weightName
                    << "\", [](double m1, double m2){\n";

                if (!fractalSB)
                {
                    // original simple 3x3 logic
                    ofs << "    int r1 = 0; // 2=Sig, 1=Sideband, 0=Other\n";
                    ofs << "    if (m1 >= " << A.sig_lo << " && m1 <= " << A.sig_hi
                        << ") r1 = 2;\n";
                    ofs << "    else if ((m1 >= " << A.left_lo << " && m1 <= " << A.left_hi
                        << ") || (m1 >= " << A.right_lo << " && m1 <= " << A.right_hi
                        << ")) r1 = 1;\n";
                    ofs << "    int r2 = 0;\n";
                    ofs << "    if (m2 >= " << B.sig_lo << " && m2 <= " << B.sig_hi
                        << ") r2 = 2;\n";
                    ofs << "    else if ((m2 >= " << B.left_lo << " && m2 <= " << B.left_hi
                        << ") || (m2 >= " << B.right_lo << " && m2 <= " << B.right_hi
                        << ")) r2 = 1;\n";
                    ofs << "    if (r1 == 2 && r2 == 2) return 1.0;\n";
                    ofs << "    if ((r1 == 2 && r2 == 1) || (r1 == 1 && r2 == 2)) return -0.5;\n";
                    ofs << "    if (r1 == 1 && r2 == 1) return 0.25;\n";
                    ofs << "    return 0.0;\n";
                }
                else
                {
                    // fractal multi-level logic: compute level index for each axis and multiply
                    // coefficients
                    int LA = (int)A.level_inner.size();
                    int LB = (int)B.level_inner.size();
                    ofs << "    int lvl1 = -1;\n";
                    // levels for axis1
                    for (int l = 0; l < LA; ++l)
                    {
                        if (l == 0)
                        {
                            double lo = A.mean - A.level_outer[0] * A.sigma;
                            double hi = A.mean + A.level_outer[0] * A.sigma;
                            ofs << "    if (m1 >= " << lo << " && m1 <= " << hi << ") lvl1 = 0;\n";
                        }
                        else
                        {
                            double left_lo = A.mean - A.level_outer[l] * A.sigma;
                            double left_hi = A.mean - A.level_inner[l] * A.sigma;
                            double right_lo = A.mean + A.level_inner[l] * A.sigma;
                            double right_hi = A.mean + A.level_outer[l] * A.sigma;
                            ofs << "    if ((m1 >= " << left_lo << " && m1 <= " << left_hi
                                << ") || (m1 >= " << right_lo << " && m1 <= " << right_hi
                                << ")) lvl1 = " << l << ";\n";
                        }
                    }
                    ofs << "    int lvl2 = -1;\n";
                    // levels for axis2
                    for (int l = 0; l < LB; ++l)
                    {
                        if (l == 0)
                        {
                            double lo = B.mean - B.level_outer[0] * B.sigma;
                            double hi = B.mean + B.level_outer[0] * B.sigma;
                            ofs << "    if (m2 >= " << lo << " && m2 <= " << hi << ") lvl2 = 0;\n";
                        }
                        else
                        {
                            double left_lo = B.mean - B.level_outer[l] * B.sigma;
                            double left_hi = B.mean - B.level_inner[l] * B.sigma;
                            double right_lo = B.mean + B.level_inner[l] * B.sigma;
                            double right_hi = B.mean + B.level_outer[l] * B.sigma;
                            ofs << "    if ((m2 >= " << left_lo << " && m2 <= " << left_hi
                                << ") || (m2 >= " << right_lo << " && m2 <= " << right_hi
                                << ")) lvl2 = " << l << ";\n";
                        }
                    }
                    // now multiply coefficients if both levels found
                    ofs << "    if (lvl1 < 0 || lvl2 < 0) return 0.0;\n";
                    // create an array of coeffs inline for axis1 and axis2 (we'll simply hardcode
                    // arrays)
                    ofs << "    const double coeff1[] = {";
                    for (int l = 0; l < LA; ++l)
                    {
                        if (l)
                        {
                            ofs << ", ";
                        }
                        ofs << A.coeff[l];
                    }
                    ofs << "};\n";
                    ofs << "    const double coeff2[] = {";
                    for (int l = 0; l < LB; ++l)
                    {
                        if (l)
                        {
                            ofs << ", ";
                        }
                        ofs << B.coeff[l];
                    }
                    ofs << "};\n";
                    ofs << "    return coeff1[lvl1] * coeff2[lvl2];\n";
                }

                ofs << "  }, {\"" << b1 << "\", \"" << b2 << "\"});\n";
            }

            ofs << makeHisto2D(sq1, sq2, binsx, min1 * min1, max1 * max1, binsy, min2 * min2,
                               max2 * max2, xtitle, ytitle, showErrors, dfName, weightName);
            counter++;
            totalPlots++;
        }
    }
    ofs << "std::cout << std::endl;";

    ofs << "histograms->Close();\n";
    ofs << "}";

    ofs.close();
    ofs.open("totalPlots.txt");
    ofs << totalPlots;
}
