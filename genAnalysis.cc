// CodeGen.cc
// This program generates a C++ analysis file (e.g. AnalyzeD.cc) based on
// ROOT RDataFrame and a list of selected branches in selectedBranches.csv.
// It inspects an input ROOT file, computes histogram ranges and bin widths,
// and writes out code that uses RDataFrame to create 1D and 2D histograms

// (including Dalitz plots) with LaTeX axes, error bars, and optimized
// bins.

// #include "KnuthWidth.h"
#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TFile.h>
#include <TKey.h>
#include <TLegend.h>
#include <TTree.h>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
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

string makeHisto1D(const string &b, size_t bins, double min, double max, bool showErrors)
{
    std::ostringstream out;
    double binWidth = static_cast<int>(1000 * ((max - min) / bins));
    string histName = "h1_" + b;
    string title = "Mass[" + particleNameToLatex(b) + "] (GeV)";
    out << "  auto " << histName << " = df.Histo1D({\"" << histName << "\",\""
        << "" << "\"," << bins << "," << min << "," << max << "}, \"" << b << "\");\n";
    out << histName << "->GetXaxis()->SetTitle(\"" << title << "\");" << std::endl;
    out << histName << "->GetYaxis()->SetTitle(\"Counts / " << binWidth << " MeV\");" << std::endl;
    out << histName << "->Write();\n";
    out << "progress++;\n";
    out << "std::cout << \"\\r\" << \"Progress: \" << progress << \"/\" << total << std::flush;\n";

    // if (!showErrors)
    // out << "  " << histName << "->SetErrorOption(\"\");\n";
    return out.str();
}

string makeHisto2D(const string &x, const string &y, size_t bx, double xmin, double xmax, size_t by,
                   double ymin, double ymax, const string &xlabel, const string &ylabel,
                   bool showErrors, string dfName)
{
    std::ostringstream out;
    string histName = "h2_" + x + "_" + y;
    string title = xlabel + " vs. " + ylabel;
    out << "  auto " << histName << " = " << dfName << ".Histo2D({\"" << histName << "\",\"" << ""
        << "\"," << bx << "," << xmin << "," << xmax << "," << by << "," << ymin << "," << ymax
        << "}, \"" << x << "\", \"" << y << "\");\n";
    out << histName << "->GetXaxis()->SetTitle(\"" << xlabel << "\");" << std::endl;
    out << histName << "->GetYaxis()->SetTitle(\"" << ylabel << "\");" << std::endl;
    out << histName << "->Write();\n";
    out << "progress++;\n";
    out << "std::cout << \"\\r\" << \"Progress: \" << progress << \"/\" << total << std::flush;\n";
    // if (!showErrors)
    // out << "  " << histName << "->SetErrorOption(\"\");\n";
    return out.str();
}

int main(int argc, char **argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <input.root> <selectedBranches.csv> <output.cc> [showErrors=0]\n";
        return 1;
    }
    string inputFile = argv[1];

    string csvFile = argv[2];
    string outFile = argv[3];
    bool showErrors = (argc > 4 ? std::stoi(argv[4]) : 0);
    size_t totalPlots = 0;

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

    std::ofstream ofs(outFile);

    ofs << "#include <ROOT/RDataFrame.hxx>\n"
        << "#include <TFile.h>\n"
        << "#include <TCanvas.h>\n"
        << "#include <TKey.h>\n"
        << "#include <TGaxis.h>\n"
        << "#include <string>\n"
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
        << "  auto df = RDataFrame(tree, \"" << inputFile << "\");\n"
        << "TFile* histograms =TFile::Open(\"histograms.root\", \"RECREATE\"); \n"
        << "std::ifstream totalplots(\"totalPlots.txt\");\n"
        << "std::string total;\n"
        << "std::getline(totalplots,total);\n"
        << "size_t numplots = std::stoul(total);\n"
        << "size_t progress = 0;\n"
        << "histograms->cd();\n";

    std::cout << "Creating 1D Histograms..." << std::endl;

    for (auto &b : cols)
    {
        double minv = df.Min<double>(b).GetValue();
        double maxv = df.Max<double>(b).GetValue();
        auto data_b = df.Take<double>(b).GetValue();
        size_t bins1 = 104; // static_cast<size_t>(Knuth::computeNumberBins(data_b));
        ofs << makeHisto1D(b, bins1, minv, maxv, showErrors);
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
        double clabmin = df.Min<double>(clab).GetValue();
        double clabmax = df.Max<double>(clab).GetValue();
        double plabmin = df.Min<double>(plab).GetValue();
        double plabmax = df.Max<double>(plab).GetValue();

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
            cGJmin = df.Min<double>(cGJ).GetValue();
            cGJmax = df.Max<double>(cGJ).GetValue();
            pGJmin = df.Min<double>(pGJ).GetValue();
            pGJmax = df.Max<double>(pGJ).GetValue();
            cHmin = df.Min<double>(cH).GetValue();
            cHmax = df.Max<double>(cH).GetValue();
            pHmin = df.Min<double>(pH).GetValue();
            pHmax = df.Max<double>(pH).GetValue();
        }
        // auto data_cGJ = df.Take<double>(cGJ).GetValue();
        // auto data_pGJ = df.Take<double>(pGJ).GetValue();
        // auto data_cH = df.Take<double>(cH).GetValue();
        // auto data_pH = df.Take<double>(pH).GetValue();

        size_t bm = 104; // static_cast<size_t>(Knuth::computeNumberBins(data_m));
        size_t ba = 104; // static_cast<size_t>(Knuth::computeNumberBins(data_a));

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
            size_t bx = 104; // static_cast<size_t>(Knuth::computeNumberBins(data_x));
            size_t by = 104; // static_cast<size_t>(Knuth::computeNumberBins(data_y));
            string xtitle = "Mass[" + particleNameToLatex(x) + "] (GeV)";
            string ytitle = "Mass[" + particleNameToLatex(y) + "] (GeV)";
            ofs << makeHisto2D(x, y, bx, xmin, xmax, by, ymin, ymax, xtitle, ytitle, showErrors,
                               dfName);
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
            ofs << "  auto df2_" << counter << " = df.Define(\"" << sq1
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
            size_t binsx = 104; // static_cast<size_t>(Knuth::computeNumberBins(sqx));
            size_t binsy = 104; // static_cast<size_t>(Knuth::computeNumberBins(sqy));
            string xtitle = "Mass[" + particleNameToLatex(b1) + "]^{2} (GeV^{2})";
            string ytitle = "Mass[" + particleNameToLatex(b2) + "]^{2} (GeV^{2})";
            ofs << makeHisto2D(sq1, sq2, binsx, min1 * min1, max1 * max1, binsy, min2 * min2,
                               max2 * max2, xtitle, ytitle, showErrors, dfName);
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
