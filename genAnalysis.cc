#include <ROOT/RDataFrame.hxx>
#include <TF1.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TKey.h>
#include <TString.h>
#include <TTree.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace ROOT;
using std::string;
using std::vector;

// ----------------------------- USER CONFIGURATION (edit here) -----------------------------
static const int NUM_THREADS = 8; // 0 => ROOT decides

struct NamedFilter
{
    std::string name;
    std::string expr;
    bool enabled;
};
static const NamedFilter DEFAULT_FILTERS[] = {
  //{"name", "expression", bool},
    {"candidate pt", "pt > 0.5", true},
    {"candidate eta", "fabs(eta) < 2.5", true},
    {"vertex chi2", "vtx_chi2 < 10", true},
    {"flight significance", "l_xy / l_xy_err > 3", false}};

// Resonance groups: name -> vector of tokens to match in mass_<tokens...> branch
static const std::pair<std::string, std::vector<std::string>> RESONANCE_GROUPS[] = {
    {"omega", {"PiPlus", "PiMinus1", "Photon1", "Photon2"}}, {"Lambda", {"Proton", "PiMinus2"}}};

// sideband geometry (in sigmas)
static const double SIG_NSIGMA = 2.0;
static const bool ENABLE_SIDEBAND_SUBTRACTION = true;
// -------------------------------------------------------------------------------------------

// LaTeX particle names helper
static std::string particleNameToLatex(const std::string &branch)
{
    static const std::map<std::string, std::string> particleMap = {
        {"PiMinus", "#pi^{-}"},    {"PiPlus", "#pi^{+}"}, {"Pi0", "#pi^{0}"},
        {"KMinus", "K^{-}"},       {"KPlus", "K^{+}"},    {"Proton", "p"},
        {"AntiProton", "#bar{p}"}, {"Neutron", "n"},      {"Electron", "e^{-}"},
        {"Positron", "e^{+}"},     {"Photon", "#gamma"},  {"KShort", "K^{0}_{S}"}};

    std::string s = branch;
    const std::string prefix = "mass_";
    if (s.rfind(prefix, 0) == 0)
    {
        s = s.substr(prefix.size());
    }

    std::stringstream ss(s);
    std::string seg;
    std::vector<std::string> parts;
    while (std::getline(ss, seg, '_'))
    {
        if (seg.empty())
        {
            continue;
        }
        std::regex re("([A-Za-z]+)([0-9]*)$");
        std::smatch m;
        if (std::regex_match(seg, m, re))
        {
            std::string base = m[1];
            std::string idx = m[2];
            auto it = particleMap.find(base);
            std::string latex = (it != particleMap.end()) ? it->second : base;
            if (!idx.empty())
            {
                latex += "_{" + idx + "}";
            }
            parts.push_back(latex);
        }
        else
        {
            parts.push_back(seg);
        }
    }
    std::string joined;
    for (size_t i = 0; i < parts.size(); ++i)
    {
        if (i)
        {
            joined += " ";
        }
        joined += parts[i];
    }
    return joined;
}

// small helpers
static std::vector<std::string> splitTokens(const std::string &s)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    string t;
    while (std::getline(ss, t, '_'))
    {
        if (!t.empty())
        {
            out.push_back(t);
        }
    }
    return out;
}

// Estimate peak and sigma from vector (gaussian fit fallback to mean/RMS)
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

int main(int argc, char **argv)
{
    // implicit MT
    if (NUM_THREADS > 0)
    {
        ROOT::EnableImplicitMT(NUM_THREADS);
        std::cout << "[INFO] Threads requested: " << NUM_THREADS << "\n";
    }
    else
    {
        ROOT::EnableImplicitMT();
        std::cout << "[INFO] ROOT decides threads\n";
    }

    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <files*|comma-separated-files> <selectedBranches.csv> ";
        // "[showErrors=0]\n";
        return 1;
    }

    std::string filespec = argv[1];
    std::string csvFile = argv[2];
    // bool showErrors = (argc > 3 ? std::stoi(argv[3]) != 0 : false);

    // read selected branches csv
    std::ifstream ifs(csvFile);
    if (!ifs.is_open())
    {
        std::cerr << "ERROR: cannot open CSV: " << csvFile << "\n";
        return 1;
    }
    vector<string> sel;
    string line;
    string finalState;
    while (std::getline(ifs, line))
    {
        if (line.empty())
        {
            continue;
        }
        line.erase(line.find_last_not_of(" \n\r\t") + 1);
        sel.push_back(line);
        finalState = line;
    }
    ifs.close();
    if (sel.empty())
    {
        std::cerr << "ERROR: CSV empty\n";
        return 1;
    }

    // determine TTree name from first file
    std::vector<std::string> fileVec;
    bool passVectorToRDF = false;
    if (filespec.find(',') != string::npos)
    {
        // split comma-separated into vector and pass vector
        std::stringstream ss(filespec);
        string token;
        while (std::getline(ss, token, ','))
        {
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);
            if (!token.empty())
            {
                fileVec.push_back(token);
            }
        }
        if (fileVec.empty())
        {
            std::cerr << "ERROR: no file names parsed from comma-list\n";
            return 1;
        }
        passVectorToRDF = true;
    }

    // find tree name from first real root file
    std::string firstFile = passVectorToRDF ? fileVec.front() : filespec;
    TFile *tf0 = TFile::Open(firstFile.c_str(), "READ");
    if (!tf0 || tf0->IsZombie())
    {
        std::cerr << "ERROR: cannot open " << firstFile << "\n";
        return 1;
    }
    std::string treeName;
    TIter itKey(tf0->GetListOfKeys());
    TKey *key;
    while ((key = (TKey *)itKey()))
    {
        TString cn = key->GetClassName();
        if (std::string(cn.Data()) == "TTree")
        {
            treeName = key->GetName();
            break;
        }
    }
    tf0->Close();
    if (treeName.empty())
    {
        std::cerr << "ERROR: no TTree found in file.\n";
        return 1;
    }
    std::cout << "[INFO] Using tree: " << treeName << "\n";

    // Build RDataFrame using ROOT-native file handling:
    // - if we collected a vector (from a globbed path or comma-list) pass the vector;
    // - otherwise pass the filespec string (may contain wildcard like file_*.root)
    RDataFrame df =
        passVectorToRDF ? RDataFrame(treeName, fileVec) : RDataFrame(treeName, filespec);

    // collect mass_* columns that match selected CSV combos; fallback to all mass_*
    auto allCols = df.GetColumnNames();
    vector<string> cols;
    for (const auto &c : allCols)
    {
        TString tc = c;
        std::string s = std::string(tc.Data());
        const std::string prefix = "mass_";
        std::string stripped = s;
        if (s.rfind(prefix, 0) == 0)
        {
            stripped = s.substr(prefix.size());
        }
        for (auto &t : sel)
        {
            if (stripped == t)
            {
                cols.push_back(s);
                break;
            }
        }
    }
    if (cols.empty())
    {
        for (const auto &c : allCols)
        {
            TString tc = c;
            std::string s = std::string(tc.Data());
            if (s.rfind("mass_", 0) == 0)
            {
                cols.push_back(s);
            }
        }
    }
    if (cols.empty())
    {
        std::cerr << "ERROR: no mass_* columns found\n";
        return 1;
    }

    // build combined preselection
    std::string combinedSel;
    for (const auto &nf : DEFAULT_FILTERS)
    {
        if (!nf.enabled)
        {
            continue;
        }
        if (!combinedSel.empty())
        {
            combinedSel += " && ";
        }
        combinedSel += "(" + nf.expr + ")";
    }
    if (!combinedSel.empty())
    {
        std::cout << "[INFO] Preselection: " << combinedSel << "\n";
    }
    else
    {
        std::cout << "[INFO] No preselection\n";
    }

    ROOT::RDF::RNode df_pre = df;
    if (!combinedSel.empty())
    {
        df_pre = df_pre.Filter(combinedSel.c_str(), "preselection");
    }

    // pre-no-mass node for sideband fitting (exclude any mass-based cuts if existed)
    std::string nonMassSel;
    for (const auto &nf : DEFAULT_FILTERS)
    {
        if (!nf.enabled)
        {
            continue;
        }
        if (nf.expr.find("mass_") != std::string::npos)
        {
            continue;
        }
        if (!nonMassSel.empty())
        {
            nonMassSel += " && ";
        }
        nonMassSel += "(" + nf.expr + ")";
    }
    ROOT::RDF::RNode df_pre_noMass = df;
    if (!nonMassSel.empty())
    {
        df_pre_noMass = df_pre_noMass.Filter(nonMassSel.c_str(), "pre_no_mass");
    }

    // auto-detect resonance groups (generalized)
    std::map<std::string, std::string> detectedMassByGroup;
    std::map<std::string, std::pair<double, double>> meanSigmaByGroup;
    for (const auto &rg : RESONANCE_GROUPS)
    {
        const std::string &gname = rg.first;
        const auto &tokensWanted = rg.second;
        std::string found;
        for (const auto &c : cols)
        {
            auto toks = splitTokens(c);
            bool ok = true;
            for (auto &tw : tokensWanted)
            {
                bool f = false;
                for (auto &t : toks)
                {
                    if (t.find(tw) != std::string::npos)
                    {
                        f = true;
                        break;
                    }
                }
                if (!f)
                {
                    ok = false;
                    break;
                }
            }
            if (ok)
            {
                found = c;
                break;
            }
        }
        if (!found.empty())
        {
            detectedMassByGroup[gname] = found;
            auto vec = df_pre_noMass.Take<double>(found).GetValue();
            auto ms = estimatePeakAndSigmaFromVec(vec);
            meanSigmaByGroup[gname] = ms;
            std::cout << "[AUTO] Group '" << gname << "' -> " << found << " (mean=" << ms.first
                      << ", sigma=" << ms.second << ")\n";
        }
        else
        {
            std::cout << "[AUTO] Group '" << gname << "' not found\n";
        }
    }

    // precompute sideband windows
    struct SBWin
    {
        double sig_lo, sig_hi;
        double sb1_lo, sb1_hi;
        double sb2_lo, sb2_hi;
    };
    std::map<std::string, SBWin> sidebandWindows;
    if (ENABLE_SIDEBAND_SUBTRACTION)
    {
        for (const auto &kv : meanSigmaByGroup)
        {
            const std::string &g = kv.first;
            double mean = kv.second.first, sigma = kv.second.second;
            if (!(sigma > 0 && std::isfinite(mean)))
            {
                continue;
            }
            double sig_lo = mean - SIG_NSIGMA * sigma, sig_hi = mean + SIG_NSIGMA * sigma;
            double sb1_lo = mean - 2.0 * SIG_NSIGMA * sigma, sb1_hi = mean - SIG_NSIGMA * sigma;
            double sb2_lo = mean + SIG_NSIGMA * sigma, sb2_hi = mean + 2.0 * SIG_NSIGMA * sigma;
            sidebandWindows[g] = {sig_lo, sig_hi, sb1_lo, sb1_hi, sb2_lo, sb2_hi};
            std::cout << "[SB] " << g << " sig=[" << sig_lo << "," << sig_hi << "] sb1=[" << sb1_lo
                      << "," << sb1_hi << "] sb2=[" << sb2_lo << "," << sb2_hi << "]\n";
        }
    }

    // compute totalPlots (must be exact)
    size_t totalPlots = 0;
    totalPlots += cols.size(); // 1D mass
    auto preCols = df_pre.GetColumnNames();
    auto hasPreCol = [&](const std::string &cname) -> bool {
        for (const auto &cc : preCols)
        {
            TString t = cc;
            if (std::string(t.Data()) == cname)
            {
                return true;
            }
        }
        return false;
    };
    for (const auto &c : cols)
    {
        std::string ms = c;
        if (ms.rfind("mass_", 0) == 0)
        {
            ms = ms.substr(5);
        }
        if (hasPreCol("costh_lab_" + ms))
        {
            totalPlots++;
        }
        if (hasPreCol("phi_lab_" + ms))
        {
            totalPlots++;
        }
        if (hasPreCol("costh_GJ_" + ms))
        {
            totalPlots++;
        }
        if (hasPreCol("phi_GJ_" + ms))
        {
            totalPlots++;
        }
        if (hasPreCol("costh_H_" + ms))
        {
            totalPlots++;
        }
        if (hasPreCol("phi_H_" + ms))
        {
            totalPlots++;
        }
    }
    for (size_t i = 0; i < cols.size(); ++i)
    {
        for (size_t j = i + 1; j < cols.size(); ++j)
        {
            totalPlots++;
        }
    }
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
            }
            if (hasCommon)
            {
                totalPlots++;
            }
        }
    }
    {
        std::ofstream ofs("totalPlots.txt");
        if (!ofs.is_open())
        {
            std::cerr << "ERROR: cannot write totalPlots.txt\n";
            return 1;
        }
        ofs << totalPlots;
    }
    std::cout << "[INFO] totalPlots=" << totalPlots << " (written to totalPlots.txt)\n";

    // Create histograms using RNode::Histo methods for speed
    TFile *fout = TFile::Open("histograms.root", "RECREATE");
    if (!fout || fout->IsZombie())
    {
        std::cerr << "ERROR: cannot create histograms.root\n";
        return 1;
    }
    fout->cd();

    size_t progress = 0;

    // 1D Mass histograms with RNode::Histo1D
    for (const auto &col : cols)
    {
        // determine range
        double xmin = 0., xmax = 1.;
        try
        {
            xmin = df_pre.Min<double>(col).GetValue();
            xmax = df_pre.Max<double>(col).GetValue();
        }
        catch (...)
        {
        }
        if (!(xmin < xmax))
        {
            xmin = 0.;
            xmax = 1.;
        }
        int nbins = 104;
        std::string hname = "h1_" + col;
        std::string title = "Mass[" + particleNameToLatex(col) + "] (GeV)";
        // Using Histo1D on prefiltered RNode
        auto r = df_pre.Histo1D({hname.c_str(), title.c_str(), nbins, xmin, xmax}, col);
        // set axis title & y-title then write
        r->GetXaxis()->SetTitle(title.c_str());
        int bw = static_cast<int>(1000.0 * ((xmax - xmin) / nbins));
        r->GetYaxis()->SetTitle(Form("Counts / %d MeV", bw));
        r->Write();
        progress++;
        std::cout << "\rProgress: " << progress << "/" << totalPlots << std::flush;
    }

    // Angular 2D: use Histo2D on df_pre
    preCols = df_pre.GetColumnNames();
    auto hasCol = [&](const std::string &cname) -> bool {
        for (const auto &cc : preCols)
        {
            TString t = cc;
            if (std::string(t.Data()) == cname)
            {
                return true;
            }
        }
        return false;
    };

    for (const auto &m : cols)
    {
        std::string ms = m;
        if (ms.rfind("mass_", 0) == 0)
        {
            ms = ms.substr(5);
        }
        std::string clab = "costh_lab_" + ms, plab = "phi_lab_" + ms;
        std::string cGJ = "costh_GJ_" + ms, pGJ = "phi_GJ_" + ms, cH = "costh_H_" + ms,
                    pH = "phi_H_" + ms;
        double xmin = 0., xmax = 1.;
        try
        {
            xmin = df_pre.Min<double>(m).GetValue();
            xmax = df_pre.Max<double>(m).GetValue();
        }
        catch (...)
        {
        }
        if (!(xmin < xmax))
        {
            xmin = 0.;
            xmax = 1.;
        }
        int bx = 104, by = 104;

        auto make2D = [&](const std::string &ycol) {
            std::string name = "h2_" + m + "_" + ycol;
            std::string title =
                "Mass[" + particleNameToLatex(m) + "] vs " + particleNameToLatex(ycol);
            // Use Histo2D on df_pre
            auto r =
                df_pre.Histo2D({name.c_str(), title.c_str(), bx, xmin, xmax, by, -1., 1.}, m, ycol);
            // set nicer y-range & axis titles if possible (we cannot quickly get min/max for y
            // without another RDF call)
            r->GetXaxis()->SetTitle(("Mass[" + particleNameToLatex(m) + "] (GeV)").c_str());
            r->GetYaxis()->SetTitle(particleNameToLatex(ycol).c_str());
            r->Write();
            progress++;
            std::cout << "\rProgress: " << progress << "/" << totalPlots << std::flush;
        };

        if (hasCol(clab))
        {
            make2D(clab);
        }
        if (hasCol(plab))
        {
            make2D(plab);
        }
        if (hasCol(cGJ))
        {
            make2D(cGJ);
        }
        if (hasCol(pGJ))
        {
            make2D(pGJ);
        }
        if (hasCol(cH))
        {
            make2D(cH);
        }
        if (hasCol(pH))
        {
            make2D(pH);
        }
    }

    // 2D mass correlations (and sideband subtraction using Histo2D)
    for (size_t i = 0; i < cols.size(); ++i)
    {
        for (size_t j = i + 1; j < cols.size(); ++j)
        {
            std::string x = cols[i], y = cols[j];
            double xmin = 0., xmax = 1., ymin = 0., ymax = 1.;
            try
            {
                xmin = df_pre.Min<double>(x).GetValue();
                xmax = df_pre.Max<double>(x).GetValue();
            }
            catch (...)
            {
            }
            try
            {
                ymin = df_pre.Min<double>(y).GetValue();
                ymax = df_pre.Max<double>(y).GetValue();
            }
            catch (...)
            {
            }
            if (!(xmin < xmax))
            {
                xmin = 0.;
                xmax = 1.;
            }
            if (!(ymin < ymax))
            {
                ymin = 0.;
                ymax = 1.;
            }
            int bx = 104, by = 104;

            bool didSB = false;
            if (ENABLE_SIDEBAND_SUBTRACTION)
            {
                // find if x and y match two different detected resonance groups
                std::string gx, gy;
                for (const auto &kv : detectedMassByGroup)
                {
                    if (kv.second == x)
                    {
                        gx = kv.first;
                    }
                    if (kv.second == y)
                    {
                        gy = kv.first;
                    }
                }
                if (!gx.empty() && !gy.empty() && gx != gy && sidebandWindows.count(gx) &&
                    sidebandWindows.count(gy))
                {
                    // construct safe selection strings
                    auto &wx = sidebandWindows[gx];
                    auto &wy = sidebandWindows[gy];
                    auto mk = [&](const std::string &branch, double lo, double hi) {
                        std::ostringstream os;
                        os << "(" << branch << " > " << std::setprecision(10) << lo << " && "
                           << branch << " < " << std::setprecision(10) << hi << ")";
                        return os.str();
                    };
                    std::string ss = "(" + mk(x, wx.sig_lo, wx.sig_hi) + " && " +
                                     mk(y, wy.sig_lo, wy.sig_hi) + ")";
                    std::string sb = "(" + mk(x, wx.sig_lo, wx.sig_hi) + " && (" +
                                     mk(y, wy.sb1_lo, wy.sb1_hi) + " || " +
                                     mk(y, wy.sb2_lo, wy.sb2_hi) + "))";
                    std::string bs = "((" + mk(x, wx.sb1_lo, wx.sb1_hi) + " || " +
                                     mk(x, wx.sb2_lo, wx.sb2_hi) + ") && " +
                                     mk(y, wy.sig_lo, wy.sig_hi) + ")";
                    std::string bb = "((" + mk(x, wx.sb1_lo, wx.sb1_hi) + " || " +
                                     mk(x, wx.sb2_lo, wx.sb2_hi) + ") && (" +
                                     mk(y, wy.sb1_lo, wy.sb1_hi) + " || " +
                                     mk(y, wy.sb2_lo, wy.sb2_hi) + "))";

                    // filter nodes and create Histo2D for each region
                    auto node_ss = df_pre_noMass.Filter(ss.c_str(), "ss_node");
                    auto node_sb = df_pre_noMass.Filter(sb.c_str(), "sb_node");
                    auto node_bs = df_pre_noMass.Filter(bs.c_str(), "bs_node");
                    auto node_bb = df_pre_noMass.Filter(bb.c_str(), "bb_node");

                    std::string name_ss = "h_ss_" + x + "_" + y;
                    std::string title = "ss(" + x + "," + y + ")";
                    auto r_ss = node_ss.Histo2D(
                        {name_ss.c_str(), title.c_str(), bx, xmin, xmax, by, ymin, ymax}, x, y);
                    auto r_sb =
                        node_sb.Histo2D({"h_sb_tmp", "sb", bx, xmin, xmax, by, ymin, ymax}, x, y);
                    auto r_bs =
                        node_bs.Histo2D({"h_bs_tmp", "bs", bx, xmin, xmax, by, ymin, ymax}, x, y);
                    auto r_bb =
                        node_bb.Histo2D({"h_bb_tmp", "bb", bx, xmin, xmax, by, ymin, ymax}, x, y);

                    // force evaluation by using the hist pointers (operator-> triggers computation)
                    TH2D *h_ss = r_ss.operator->(); // trigger
                    TH2D *h_sb = r_sb.operator->();
                    TH2D *h_bs = r_bs.operator->();
                    TH2D *h_bb = r_bb.operator->();

                    // clone and combine: out = ss - 0.5*sb - 0.5*bs + 0.25*bb
                    std::string outName = "h2_sbsub_" + x + "_vs_" + y;
                    TH2D *hout = (TH2D *)h_ss->Clone(outName.c_str());
                    hout->Add(h_sb, -0.5);
                    hout->Add(h_bs, -0.5);
                    hout->Add(h_bb, 0.25);

                    // label axes using LaTeX names
                    hout->GetXaxis()->SetTitle(
                        ("Mass[" + particleNameToLatex(x) + "] (GeV)").c_str());
                    hout->GetYaxis()->SetTitle(
                        ("Mass[" + particleNameToLatex(y) + "] (GeV)").c_str());
                    hout->Write();

                    // cleanup temporaries created by Histo2D (RDF owns them, but cloned hist
                    // exists) Note: r_ss/r_sb/... are RResultPtr; their underlying histogram
                    // objects are managed by RDF until program exit. We cloned h_ss into hout; we
                    // don't need to delete the r_* histograms (RDF manages them), but we do delete
                    // hout.
                    delete hout;
                    didSB = true;
                    progress++;
                    std::cout << "\rProgress: " << progress << "/" << totalPlots << std::flush;
                }
            }

            if (!didSB)
            {
                // simple 2D using Histo2D on prefiltered df
                std::string name = "h2_" + x + "_" + y;
                std::string title =
                    "Mass[" + particleNameToLatex(x) + "] vs Mass[" + particleNameToLatex(y) + "]";
                auto r = df_pre.Histo2D(
                    {name.c_str(), title.c_str(), bx, xmin, xmax, by, ymin, ymax}, x, y);
                r->GetXaxis()->SetTitle(("Mass[" + particleNameToLatex(x) + "] (GeV)").c_str());
                r->GetYaxis()->SetTitle(("Mass[" + particleNameToLatex(y) + "] (GeV)").c_str());
                r->Write();
                progress++;
                std::cout << "\rProgress: " << progress << "/" << totalPlots << std::flush;
            }
        }
    }

    // Dalitz plots: m^2 vs m^2 for combos that share tokens
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
            }
            if (!hasCommon)
            {
                continue;
            }

            std::string b1 = cols[i], b2 = cols[j];
            std::string sq1 = b1 + "_sq", sq2 = b2 + "_sq";
            // Define squares and histogram with Histo2D
            auto node_sq = df.Define(sq1, [](double x) { return x * x; }, {b1})
                               .Define(sq2, [](double x) { return x * x; }, {b2});
            double min1 = node_sq.Min<double>(sq1).GetValue();
            double max1 = node_sq.Max<double>(sq1).GetValue();
            double min2 = node_sq.Min<double>(sq2).GetValue();
            double max2 = node_sq.Max<double>(sq2).GetValue();
            TString histname = "h2_" + sq1 + "_" + sq2;
            auto r =
                node_sq.Histo2D({histname, "dalitz", 104, min1, max1, 104, min2, max2}, sq1, sq2);
            r->GetXaxis()->SetTitle(
                ("Mass[" + particleNameToLatex(b1) + "]^{2} (GeV^{2})").c_str());
            r->GetYaxis()->SetTitle(
                ("Mass[" + particleNameToLatex(b2) + "]^{2} (GeV^{2})").c_str());
            r->Write();
            progress++;
            std::cout << "\rProgress: " << progress << "/" << totalPlots << std::flush;
        }
    }

    std::cout << "\nDone. Wrote histograms.root with " << progress << " histograms.\n";
    fout->Close();
    return 0;
}
