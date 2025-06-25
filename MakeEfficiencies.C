// MakeEfficiencies.C

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

// Rebin TH1D to union of ranges with same bin count, preserving original axis title
TH1D* Rebin1D(const TH1D* h, double newMin, double newMax) {
    int n = h->GetNbinsX();
    TH1D* hnew = new TH1D(Form("%s_reb1d", h->GetName()), h->GetTitle(), n, newMin, newMax);
    hnew->GetXaxis()->SetTitle(h->GetXaxis()->GetTitle());
    for (int i = 1; i <= n; ++i) {
        double c = h->GetBinContent(i);
        double e = h->GetBinError(i);
        double x = h->GetXaxis()->GetBinCenter(i);
        if (x >= newMin && x <= newMax) {
            int b = hnew->FindBin(x);
            double pc = hnew->GetBinContent(b);
            double pe = hnew->GetBinError(b);
            hnew->SetBinContent(b, pc + c);
            hnew->SetBinError(b, std::sqrt(pe*pe + e*e));
        }
    }
    return hnew;
}

// Rebin TH2D to union ranges, preserving original axis titles
TH2D* Rebin2D(const TH2D* h, double minX, double maxX, double minY, double maxY) {
    int nx = h->GetNbinsX(), ny = h->GetNbinsY();
    TH2D* hnew = new TH2D(Form("%s_reb2d", h->GetName()), h->GetTitle(),
                          nx, minX, maxX, ny, minY, maxY);
    hnew->GetXaxis()->SetTitle(h->GetXaxis()->GetTitle());
    hnew->GetYaxis()->SetTitle(h->GetYaxis()->GetTitle());
    for (int i = 1; i <= nx; ++i) for (int j = 1; j <= ny; ++j) {
        double c = h->GetBinContent(i, j);
        double e = h->GetBinError(i, j);
        double x = h->GetXaxis()->GetBinCenter(i);
        double y = h->GetYaxis()->GetBinCenter(j);
        if (x >= minX && x <= maxX && y >= minY && y <= maxY) {
            int bx = hnew->GetXaxis()->FindBin(x);
            int by = hnew->GetYaxis()->FindBin(y);
            double pc = hnew->GetBinContent(bx, by);
            double pe = hnew->GetBinError(bx, by);
            hnew->SetBinContent(bx, by, pc + c);
            hnew->SetBinError(bx, by, std::sqrt(pe*pe + e*e));
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
    while ((key = (TKey*)it->Next())) {
        TString name = key->GetName();
        if (!name.BeginsWith("h1") && !name.BeginsWith("h2")) continue;
        TObject *oA = fA->Get(name), *oT = fT->Get(name);
        if (!oT) { Warning("MakeEfficiencies","%s missing", name.Data()); continue; }
        fout->cd();
        if (name.BeginsWith("h1")) {
            TH1D *hA = (TH1D*)oA, *hT = (TH1D*)oT;
            // find union range
            double minA=hA->GetXaxis()->GetXmin(), maxA=hA->GetXaxis()->GetXmax();
            double minT=hT->GetXaxis()->GetXmin(), maxT=hT->GetXaxis()->GetXmax();
            double umin=std::min(minA,minT), umax=std::max(maxA,maxT);
            TH1D *rA=Rebin1D(hA,umin,umax), *rT=Rebin1D(hT,umin,umax);
            // efficiency
            TH1D *hE=(TH1D*)rA->Clone(name+"_eff"); hE->Reset();
            hE->SetTitle(hA->GetTitle()); hE->GetXaxis()->SetTitle(hA->GetXaxis()->GetTitle());
            hE->GetYaxis()->SetTitle("Efficiency");
            hE->SetMinimum(-1.1); hE->SetMaximum(1.1);
            hE->SetMarkerStyle(20); hE->SetMarkerSize(0.8);
            for (int b=1;b<=hE->GetNbinsX();++b) {
                double t=rT->GetBinContent(b), a=rA->GetBinContent(b);
                double eff=-1, err=0;
                if (t>0) { eff=a/t; err=std::sqrt(eff*(1-eff)/t); if (eff<0||eff>1) eff=-1; }
                hE->SetBinContent(b,eff); hE->SetBinError(b,err);
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
            TH2D *hE=(TH2D*)rA->Clone(name+"_eff"); hE->Reset();
            hE->SetTitle(hA->GetTitle());
            hE->GetXaxis()->SetTitle(hA->GetXaxis()->GetTitle());
            hE->GetYaxis()->SetTitle(hA->GetYaxis()->GetTitle());
            hE->GetZaxis()->SetTitle("Efficiency");
            hE->SetMinimum(-1.1); hE->SetMaximum(1.1);
            const Int_t pal=3; Int_t col[3]={kRed,kBlue,kRed}; gStyle->SetPalette(pal,col);
            int nx=hE->GetNbinsX(), ny=hE->GetNbinsY();
            for(int i=1;i<=nx;++i) for(int j=1;j<=ny;++j) {
                double t=rT->GetBinContent(i,j), a=rA->GetBinContent(i,j);
                double eff=-1;
                if(t>0) { eff=a/t; if(eff<0||eff>1) eff=-1; }
                hE->SetBinContent(i,j,eff);
            }
            hE->Write(); delete hE; delete rA; delete rT;
        }
    }
    delete it; fout->Close(); fA->Close(); fT->Close();
    Info("MakeEfficiencies","Efficiencies saved in %s",outFile);
}

// Usage:
// root -l
// .L MakeEfficiencies.C
// MakeEfficiencies();
