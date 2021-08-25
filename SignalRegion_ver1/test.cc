#include <iostream>
#include <vector>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>

TH1D* get_hist(TString, TString, TString, TString, TString);

void test() 
{

}



TH1D* get_hist(TString sample, TString histkey, TString channel, TString mass_point, TString syst="Central")
{
	TString fkey = "";
	if (sample == "fake")
		fkey = "Outputs/"+channel+"/"+mass_point+"/DATA.root";
	else
		fkey = "Outputs/"+channel+"/"+mass_point+"/"+sample+".root";
	histkey = sample+"/SR/"+syst+"/"+histkey;

	TFile* f = new TFile(fkey);
	TH1D* h = f.Get(histkey);
	h->SetDirectory(nullptr);
	f->Close();

	return h;
}




