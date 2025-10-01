// MakeEfficiencies_fixed_nan.C
// Usage:
//   root -l
//   .L MakeEfficiencies_fixed_nan.C
//   MakeEfficiencies();

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TKey.h>
#include <TIterator.h>
#include <TCollection.h>
#include <TObject.h>
#include <TString.h>
#include <TError.h>
#include <TStyle.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <limits> // <-- for quiet_NaN()

// Optimized area-weighted rebin for uniform 1D histograms
TH1D* Rebin1D(const TH1D* h, double newMin, double newMax) {
    int old_n = h->GetNbinsX();
    double old_xmin = h->GetXaxis()->GetXmin();
    double old_xmax = h->GetXaxis()->GetXmax();
    TH1D* hnew = new TH1D(Form("%s_reb1d", h->GetName()), h->GetTitle(), old_n, newMin, newMax);
    hnew->GetXaxis()->SetTitle(h->GetXaxis()->GetTitle());
    hnew->Sumw2();

    std::vector<double> err2_accum(hnew->GetNbinsX() + 3, 0.0);
    double eps = (old_xmax - old_xmin) * 1e-12;

    for (int i = 1; i <= old_n; ++i) {
        double xlow = h->GetXaxis()->GetBinLowEdge(i);
        double xhigh = xlow + h->GetXaxis()->GetBinWidth(i);
        if (xhigh <= newMin || xlow >= newMax) continue;

        double content = h->GetBinContent(i);
        double err = h->GetBinError(i);
        double width = xhigh - xlow;
        if (width <= 0) continue;

        int jfirst = hnew->GetXaxis()->FindBin(std::max(xlow + eps, newMin));
        int jlast  = hnew->GetXaxis()->FindBin(std::min(xhigh - eps, newMax - 1e-300));
        jfirst = std::max(1, jfirst);
        jlast  = std::min(hnew->GetNbinsX(), jlast);
        if (jfirst > jlast) continue;

        for (int j = jfirst; j <= jlast; ++j) {
            double nxlow = hnew->GetXaxis()->GetBinLowEdge(j);
            double nxhigh = nxlow + hnew->GetXaxis()->GetBinWidth(j);
            double overlap = std::min(xhigh, nxhigh) - std::max(xlow, nxlow);
            if (overlap <= 0) continue;
            double frac = overlap / width;
            double add = content * frac;
            int global = j;
            hnew->AddBinContent(global, add);
            double err_contrib = err * frac;
            if (global >= (int)err2_accum.size()) err2_accum.resize(global + 1, 0.0);
            err2_accum[global] += err_contrib * err_contrib;
        }
    }

    for (int j = 1; j <= hnew->GetNbinsX(); ++j) {
        double e2 = (j < (int)err2_accum.size()) ? err2_accum[j] : 0.0;
        hnew->SetBinError(j, std::sqrt(e2));
    }

    return hnew;
}

// Optimized area-weighted rebin for uniform 2D histograms
TH2D* Rebin2D(const TH2D* h, double minX, double maxX, double minY, double maxY) {
    int old_nx = h->GetNbinsX();
    int old_ny = h->GetNbinsY();
    double old_xmin = h->GetXaxis()->GetXmin();
    double old_xmax = h->GetXaxis()->GetXmax();
    double old_ymin = h->GetYaxis()->GetXmin();
    double old_ymax = h->GetYaxis()->GetXmax();

    TH2D* hnew = new TH2D(Form("%s_reb2d", h->GetName()), h->GetTitle(),
                          old_nx, minX, maxX, old_ny, minY, maxY);
    hnew->GetXaxis()->SetTitle(h->GetXaxis()->GetTitle());
    hnew->GetYaxis()->SetTitle(h->GetYaxis()->GetTitle());
    hnew->Sumw2();

    int nx_new = hnew->GetNbinsX();
    int ny_new = hnew->GetNbinsY();
    int maxGlobal = (nx_new + 2) * (ny_new + 2);
    std::vector<double> err2_accum(maxGlobal, 0.0);

    double eps_x = (old_xmax - old_xmin) * 1e-12;
    double eps_y = (old_ymax - old_ymin) * 1e-12;

    for (int ix = 1; ix <= old_nx; ++ix) {
        double xlow = h->GetXaxis()->GetBinLowEdge(ix);
        double xhigh = xlow + h->GetXaxis()->GetBinWidth(ix);
        if (xhigh <= minX || xlow >= maxX) continue;

        int jx_first = hnew->GetXaxis()->FindBin(std::max(xlow + eps_x, minX));
        int jx_last  = hnew->GetXaxis()->FindBin(std::min(xhigh - eps_x, maxX - 1e-300));
        jx_first = std::max(1, jx_first);
        jx_last  = std::min(nx_new, jx_last);
        if (jx_first > jx_last) continue;

        for (int iy = 1; iy <= old_ny; ++iy) {
            double ylow = h->GetYaxis()->GetBinLowEdge(iy);
            double yhigh = ylow + h->GetYaxis()->GetBinWidth(iy);
            if (yhigh <= minY || ylow >= maxY) continue;

            int jy_first = hnew->GetYaxis()->FindBin(std::max(ylow + eps_y, minY));
            int jy_last  = hnew->GetYaxis()->FindBin(std::min(yhigh - eps_y, maxY - 1e-300));
            jy_first = std::max(1, jy_first);
            jy_last  = std::min(ny_new, jy_last);
            if (jy_first > jy_last) continue;

            double content = h->GetBinContent(ix, iy);
            double err = h->GetBinError(ix, iy);
            double xwidth = xhigh - xlow;
            double ywidth = yhigh - ylow;
            if (xwidth <= 0 || ywidth <= 0) continue;

            for (int jx = jx_first; jx <= jx_last; ++jx) {
                double nxlow = hnew->GetXaxis()->GetBinLowEdge(jx);
                double nxhigh = nxlow + hnew->GetXaxis()->GetBinWidth(jx);
                double overlap_x = std::min(xhigh, nxhigh) - std::max(xlow, nxlow);
                if (overlap_x <= 0) continue;
                double frac_x = overlap_x / xwidth;

                for (int jy = jy_first; jy <= jy_last; ++jy) {
                    double nylow = hnew->GetYaxis()->GetBinLowEdge(jy);
                    double nyhigh = nylow + hnew->GetYaxis()->GetBinWidth(jy);
                    double overlap_y = std::min(yhigh, nyhigh) - std::max(ylow, nylow);
                    if (overlap_y <= 0) continue;
                    double frac_y = overlap_y / ywidth;

                    double frac = frac_x * frac_y;
                    if (frac <= 0) continue;

                    double add = content * frac;
                    int global = hnew->GetBin(jx, jy);
                    hnew->AddBinContent(global, add);

                    double e_contrib = err * frac;
                    if (global >= (int)err2_accum.size()) err2_accum.resize(global + 1, 0.0);
                    err2_accum[global] += e_contrib * e_contrib;
                }
            }
        }
    }

    for (int jx = 1; jx <= nx_new; ++jx) {
        for (int jy = 1; jy <= ny_new; ++jy) {
            int global = hnew->GetBin(jx, jy);
            double e2 = 0.0;
            if (global < (int)err2_accum.size()) e2 = err2_accum[global];
            hnew->SetBinError(global, std::sqrt(e2));
        }
    }

    return hnew;
}

void MakeEfficiencies(const char* accFile = "accepted.root",
                      const char* thrFile = "thrown.root",
                      const char* outFile = "efficiencies.root") {
    TFile *fA = TFile::Open(accFile, "READ"), *fT = TFile::Open(thrFile, "READ");
    if (!fA||fA->IsZombie()||!fT||fT->IsZombie()) {
        Error("MakeEfficiencies","Cannot open input files");
        return;
    }
    TFile* fout = TFile::Open(outFile,"RECREATE");
    TIterator* it = fA->GetListOfKeys()->MakeIterator();
    TKey* key;
    const double NaN = std::numeric_limits<double>::quiet_NaN();

    while ((key = (TKey*)it->Next())) {
        TString name = key->GetName();
        if (!name.BeginsWith("h1") && !name.BeginsWith("h2")) continue;
        TObject *oA = fA->Get(name), *oT = fT->Get(name);
        if (!oT) { Warning("MakeEfficiencies","%s missing", name.Data()); continue; }
        fout->cd();
        if (name.BeginsWith("h1")) {
            TH1D *hA = (TH1D*)oA, *hT = (TH1D*)oT;
            double minA=hA->GetXaxis()->GetXmin(), maxA=hA->GetXaxis()->GetXmax();
            double minT=hT->GetXaxis()->GetXmin(), maxT=hT->GetXaxis()->GetXmax();
            double umin=std::min(minA,minT), umax=std::max(maxA,maxT);
            TH1D *rA=Rebin1D(hA,umin,umax), *rT=Rebin1D(hT,umin,umax);
            TH1D *hE=(TH1D*)rA->Clone(Form("%s_eff", name.Data())); hE->Reset();
            hE->SetTitle(hA->GetTitle()); hE->GetXaxis()->SetTitle(hA->GetXaxis()->GetTitle());
            hE->GetYaxis()->SetTitle("Efficiency");
            hE->SetMinimum(-1.1); hE->SetMaximum(1.1);
            hE->SetMarkerStyle(20); hE->SetMarkerSize(0.8);
            for (int b=1;b<=hE->GetNbinsX();++b) {
                double t=rT->GetBinContent(b), a=rA->GetBinContent(b);
                if (t>0) {
                    double eff = a/t;
                    if (eff >= 0.0 && eff <= 1.0) {
                        double err = std::sqrt(eff*(1-eff)/t);
                        hE->SetBinContent(b, eff);
                        hE->SetBinError(b, err);
                    } else {
                        // invalid efficiency: mark as NaN so it won't be drawn
                        hE->SetBinContent(b, NaN);
                        hE->SetBinError(b, 0.0);
                    }
                } else {
                    // no denominator -> mark undefined as NaN
                    hE->SetBinContent(b, NaN);
                    hE->SetBinError(b, 0.0);
                }
            }
            hE->Write(); delete hE; delete rA; delete rT;
        } else {
            TH2D *hA=(TH2D*)oA, *hT=(TH2D*)oT;
            double minAx=hA->GetXaxis()->GetXmin(), maxAx=hA->GetXaxis()->GetXmax();
            double minTx=hT->GetXaxis()->GetXmin(), maxTx=hT->GetXaxis()->GetXmax();
            double minAy=hA->GetYaxis()->GetXmin(), maxAy=hA->GetYaxis()->GetXmax();
            double minTy=hT->GetYaxis()->GetXmin(), maxTy=hT->GetYaxis()->GetXmax();
            double uminX=std::min(minAx,minTx), umaxX=std::max(maxAx,maxTx);
            double uminY=std::min(minAy,minTy), umaxY=std::max(maxAy,maxTy);
            TH2D *rA=Rebin2D(hA,uminX,umaxX,uminY,umaxY), *rT=Rebin2D(hT,uminX,umaxX,uminY,umaxY);
            TH2D *hE=(TH2D*)rA->Clone(Form("%s_eff", name.Data())); hE->Reset();
            hE->SetTitle(hA->GetTitle());
            hE->GetXaxis()->SetTitle(hA->GetXaxis()->GetTitle());
            hE->GetYaxis()->SetTitle(hA->GetYaxis()->GetTitle());
            hE->GetZaxis()->SetTitle("Efficiency");
            hE->SetMinimum(-1.1); hE->SetMaximum(1.1);
            const Int_t pal=3; Int_t col[3]={kRed,kBlue,kRed}; gStyle->SetPalette(pal,col);
            int nx=hE->GetNbinsX(), ny=hE->GetNbinsY();
            for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) {
                double t=rT->GetBinContent(i,j), a=rA->GetBinContent(i,j);
                if (t>0) {
                    double eff = a/t;
                    if (eff >= 0.0 && eff <= 1.0) {
                        hE->SetBinContent(i,j, eff);
                        // no error plane used for 2D efficiency visualization here
                    } else {
                        hE->SetBinContent(i,j, NaN);
                        hE->SetBinError(i,j, 0.0);
                    }
                } else {
                    hE->SetBinContent(i,j, NaN);
                    hE->SetBinError(i,j, 0.0);
                }
            }
            hE->Write(); delete hE; delete rA; delete rT;
        }
    }
    delete it; fout->Close(); fA->Close(); fT->Close();
    Info("MakeEfficiencies","Efficiencies saved in %s",outFile);
}

