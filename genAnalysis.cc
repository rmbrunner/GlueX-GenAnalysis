// CodeGen.cc
// This program generates a C++ analysis file (e.g. AnalyzeD.cc) based on
// ROOT RDataFrame and a list of selected branches in selectedBranches.csv.
// It inspects an input ROOT file, computes histogram ranges and bin widths,
// and writes out code that uses RDataFrame to create 1D and 2D histograms

// (including Dalitz plots) with LaTeX axes, error bars, and optimized
// bins.

#include "KnuthWidth.h"
#include <ROOT/RDataFrame.hxx>
#include <TCanvas.h>
#include <TFile.h>
#include <TKey.h>
#include <TLegend.h>
#include <TTree.h>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <map>
#include <ostream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ROOT;
using std::map;
using std::set;
using std::string;
using std::vector;

string particleNameToLatex(const string &branch) {
  static const map<string, string> particleMap = {
      {"PiMinus", "#pi^{-}"}, {"PiPlus", "#pi^{+}"}, {"KMinus", "K^{-}"},
      {"KPlus", "K^{+}"},     {"Proton", "p"},       {"AntiProton", "#bar{p}"},
      {"Neutron", "n"},       {"Electron", "e^{-}"}, {"Positron", "e^{+}"},
      {"Photon", "#gamma"}};

  string s = branch;
  if (s.rfind("mass_", 0) == 0)
    s = s.substr(5);

  std::stringstream ss(s);
  string segment;
  vector<string> particles;
  while (std::getline(ss, segment, '_')) {
    std::regex re("([A-Za-z]+)([0-9]*)$");
    std::smatch match;
    if (std::regex_match(segment, match, re)) {
      string base = match[1];
      string index = match[2];
      auto it = particleMap.find(base);
      string latex = (it != particleMap.end()) ? it->second : base;
      if (!index.empty())
        latex += "_{" + index + "}";
      particles.push_back(latex);
    }
  }
  string joined;
  for (size_t i = 0; i < particles.size(); ++i) {
    if (i)
      joined += " ";
    joined += particles[i];
  }
  return joined;
}

string makeHisto1D(const string &b, size_t bins, double min, double max,
                   bool showErrors) {
  std::ostringstream out;
  double binWidth = static_cast<int>(1000 * ((max - min) / bins));
  string histName = "h1_" + b;
  string title = "Mass[" + particleNameToLatex(b) + "] (GeV)";
  out << "  auto " << histName << " = df.Histo1D({\"" << histName << "\",\""
      << "" << "\"," << bins << "," << min << "," << max << "}, \"" << b
      << "\");\n";
  out << histName << "->GetXaxis()->SetTitle(\"" << title << "\");"
      << std::endl;
  out << histName << "->GetYaxis()->SetTitle(\"Counts / " << binWidth
      << " MeV\");" << std::endl;
  out << histName << "->Write();\n";

  // if (!showErrors)
  // out << "  " << histName << "->SetErrorOption(\"\");\n";
  return out.str();
}

string makeHisto2D(const string &x, const string &y, size_t bx, double xmin,
                   double xmax, size_t by, double ymin, double ymax,
                   const string &xlabel, const string &ylabel, bool showErrors,
                   string dfName) {
  std::ostringstream out;
  string histName = "h2_" + x + "_" + y;
  string title = xlabel + " vs. " + ylabel;
  out << "  auto " << histName << " = " << dfName << ".Histo2D({\"" << histName
      << "\",\"" << "" << "\"," << bx << "," << xmin << "," << xmax << "," << by
      << "," << ymin << "," << ymax << "}, \"" << x << "\", \"" << y
      << "\");\n";
  out << histName << "->GetXaxis()->SetTitle(\"" << xlabel << "\");"
      << std::endl;
  out << histName << "->GetYaxis()->SetTitle(\"" << ylabel << "\");"
      << std::endl;
  out << histName << "->Write();\n";
  // if (!showErrors)
  // out << "  " << histName << "->SetErrorOption(\"\");\n";
  return out.str();
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr
        << "Usage: " << argv[0]
        << " <input.root> <selectedBranches.csv> <output.cc> [showErrors=0]\n";
    return 1;
  }
  string inputFile = argv[1];

  string csvFile = argv[2];
  string outFile = argv[3];
  bool showErrors = (argc > 4 ? std::stoi(argv[4]) : 0);

  std::ifstream ifs(csvFile);
  vector<TString> sel;
  std::string line;
  while (std::getline(ifs, line)) {
    if (!line.empty()) {
      line.erase(line.find_last_not_of(" \n\r\t") + 1);
      sel.push_back(line);
    }
  }

  TFile *f = TFile::Open(inputFile.c_str(), "READ");
  if (!f || f->IsZombie()) {
    std::cerr << "ERROR: Cannot open file " << inputFile << "\n";
    return 1;
  }
  TIter itKey(f->GetListOfKeys());
  TKey *key;
  string treeName;
  while ((key = (TKey *)itKey())) {
    if (std::string(key->GetClassName()) == "TTree") {
      treeName = key->GetName();
      break;
    }
  }
  if (treeName.empty()) {
    std::cerr << "ERROR: No TTree found in file.\n";
    return 1;
  }

  RDataFrame df(treeName, inputFile);
  auto allCols = df.GetColumnNames();

  vector<string> cols;
  for (const auto &b : allCols) {
    TString tmp = b;
    for (auto &i : sel) {
      TString thing = i;

      if (i == tmp) {
        cols.push_back(b);
        break;
      }
    }
  }

  vector<string> massCols, angleCols;
  for (auto &b : cols) {
    if (b.find("costh") != string::npos || b.find("phi") != string::npos)
      angleCols.push_back(b);
    else
      massCols.push_back(b);
  }

  std::ofstream ofs(outFile);

  ofs << "#include <ROOT/RDataFrame.hxx>\n"
      << "#include <TFile.h>\n"
      << "#include <TCanvas.h>\n"
      << "#include <TKey.h>\n"
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
      << "histograms->cd();\n";

  std::cout << "Creating 1D Histograms..." << std::endl;

  for (auto &b : massCols) {
    double minv = df.Min<double>(b).GetValue();
    double maxv = df.Max<double>(b).GetValue();
    auto data_b = df.Take<double>(b).GetValue();
    size_t bins1 = static_cast<size_t>(Knuth::computeNumberBins(data_b));
    ofs << makeHisto1D(b, bins1, minv, maxv, showErrors);
  }

  std::cout << "Creating 2D Mass Correlation Plots..." << std::endl;
  for (size_t i = 0; i < massCols.size(); ++i) {
    for (size_t j = i + 1; j < massCols.size(); ++j) {
      string dfName = "df";
      auto &x = massCols[i];
      auto &y = massCols[j];
      double xmin = df.Min<double>(x).GetValue();
      double xmax = df.Max<double>(x).GetValue();
      double ymin = df.Min<double>(y).GetValue();
      double ymax = df.Max<double>(y).GetValue();
      auto data_x = df.Take<double>(x).GetValue();
      auto data_y = df.Take<double>(y).GetValue();
      size_t bx = static_cast<size_t>(Knuth::computeNumberBins(data_x));
      size_t by = static_cast<size_t>(Knuth::computeNumberBins(data_y));
      string xtitle = "Mass[" + particleNameToLatex(x) + "] (GeV)";
      string ytitle = "Mass[" + particleNameToLatex(y) + "] (GeV)";
      ofs << makeHisto2D(x, y, bx, xmin, xmax, by, ymin, ymax, xtitle, ytitle,
                         showErrors, dfName);
    }
  }

  std::cout << "Creating Angular Distribution Plots..." << std::endl;
  for (auto &ang : angleCols) {
    for (auto &m : massCols) {
      string dfName = "df";
      double xmin = df.Min<double>(m).GetValue();
      double xmax = df.Max<double>(m).GetValue();
      double ymin = df.Min<double>(ang).GetValue();
      double ymax = df.Max<double>(ang).GetValue();
      auto data_m = df.Take<double>(m).GetValue();
      auto data_a = df.Take<double>(ang).GetValue();
      size_t bm = static_cast<size_t>(Knuth::computeNumberBins(data_m));
      size_t ba = static_cast<size_t>(Knuth::computeNumberBins(data_a));
      string xtitle = "Mass[" + particleNameToLatex(m) + "] (GeV)";
      string ytitle = particleNameToLatex(ang);
      ofs << makeHisto2D(m, ang, bm, xmin, xmax, ba, ymin, ymax, xtitle, ytitle,
                         showErrors, dfName);
    }
  }
  std::cout << "Creating 2D Dalitz Plots..." << std::endl;
  unsigned long int counter = 0;
  for (size_t i = 0; i < massCols.size(); ++i) {
    for (size_t j = i + 1; j < massCols.size(); ++j) {
      auto &b1 = massCols[i], &b2 = massCols[j];
      string sq1 = b1 + "_sq";
      string sq2 = b2 + "_sq";
      string dfName = "df2_" + std::to_string(counter);
      ofs << "  auto df2_" << counter << " = df.Define(\"" << sq1
          << "\",[](double x){return x*x;}, {\"" << b1 << "\"}).Define(\""
          << sq2 << "\",[](double x){return x*x;}, {\"" << b2 << "\"});\n";
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
        sq1.push_back(v * v);
      for (auto v : vec2)
        sq2.push_back(v * v);
      size_t binsx = static_cast<size_t>(Knuth::computeNumberBins(sqx));
      size_t binsy = static_cast<size_t>(Knuth::computeNumberBins(sqy));
      string xtitle = "Mass[" + particleNameToLatex(b1) + "]^{2} (GeV^{2})";
      string ytitle = "Mass[" + particleNameToLatex(b2) + "]^{2} (GeV^{2})";
      ofs << makeHisto2D(sq1, sq2, binsx, min1 * min1, max1 * max1, binsy,
                         min2 * min2, max2 * max2, xtitle, ytitle, showErrors,
                         dfName);
      counter++;
    }
  }
  ofs << "histograms->Close();\n";
  ofs << "}";
}
