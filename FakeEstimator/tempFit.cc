using namespace RooFit;

TH1 *get_hist(TString, TString, TString);

void tempFit(TString path, TString object)
{
	// Initialize samples
	TString DATASTREAM;
	vector<TString> MCSAMPLEs;
	if (object == "electron")
	{
		DATASTREAM = "SingleElectron";
		MCSAMPLEs = {"QCD_EMEnriched", "QCD_bcToE", "W", "DY", "TT", "ST", "VV"};
	}
	else if (object == "muon")
	{
		DATASTREAM = "DoubleMuon";
		MCSAMPLEs = {"QCD_MuEnriched", "W", "DY", "TT", "ST", "VV"};
	}
	else
	{
		cerr << "object should be electron or muon, wrong object" << endl;
		exit(EXIT_FAILURE);
	}

	map<TString, RooHistPdf *> TEMPs;
	map<TString, double> SCALEs;
	map<TString, RooRealVar *> COEFs;

	RooRealVar MT("MT", "MT", 0., 300.);
	TString histkey = path + "/MT";

	// Import data
	TH1 *h_data = get_hist(DATASTREAM, histkey, object);
	RooDataHist *data = new RooDataHist("dh_" + DATASTREAM, "dh_" + DATASTREAM, MT, Import(*h_data));

	// Construct templates
	double scale = 0.;
	for (const auto &sample : MCSAMPLEs)
	{
		TH1 *h = get_hist(sample, histkey, object);
		RooDataHist *dh = new RooDataHist("dh_" + sample, "dh_" + sample, MT, Import(*h));

		RooHistPdf *pdf = new RooHistPdf("temp_" + sample, "temp_" + sample, MT, *dh, 0);
		TEMPs[sample] = pdf;

		scale += h->Integral();
	}

	for (const auto &sample : MCSAMPLEs)
	{
		TH1 *h = get_hist(sample, histkey, object);
		double this_scale = h->Integral() / scale;
		SCALEs[sample] = this_scale;
	}

	// Add all templates
	for (const auto &sample : MCSAMPLEs)
	{
		double this_scale = SCALEs[sample];
		RooRealVar *coef;
		if (sample.Contains("QCD"))
			coef = new RooRealVar("coef_" + sample, "", this_scale, this_scale * 0.2, this_scale * 2.0);
		else
			coef = new RooRealVar("coef_" + sample, "", this_scale, this_scale * 0.9, this_scale * 1.1);
		COEFs[sample] = coef;
	}

	// Make model
	RooAddPdf *model;
	if (object == "electron")
	{
		model = new RooAddPdf("model", "model",
							  RooArgList(*TEMPs["QCD_EMEnriched"], *TEMPs["QCD_bcToE"], *TEMPs["W"], *TEMPs["DY"], *TEMPs["TT"], *TEMPs["ST"], *TEMPs["VV"]),
							  RooArgList(*COEFs["QCD_EMEnriched"], *COEFs["QCD_bcToE"], *COEFs["W"], *COEFs["DY"], *COEFs["TT"], *COEFs["ST"], *COEFs["VV"]),
							  kFALSE);
	}
	else if (object == "muon")
	{
		model = new RooAddPdf("model", "model",
							  RooArgList(*TEMPs["QCD_MuEnriched"], *TEMPs["W"], *TEMPs["DY"], *TEMPs["TT"], *TEMPs["ST"], *TEMPs["VV"]),
							  RooArgList(*COEFs["QCD_MuEnriched"], *COEFs["W"], *COEFs["DY"], *COEFs["TT"], *COEFs["ST"], *COEFs["VV"]),
							  kFALSE);
	}
	else
	{
		cerr << "object should be electron or muon, wrong object" << endl;
		exit(EXIT_FAILURE);
	}

	model->chi2FitTo(*data);
	//model->fitTo(*data);

	RooPlot *frame = MT.frame(Title(""), Bins(60));
	data->plotOn(frame);
	model->plotOn(frame);

	/*	
	TCanvas* cvs = new TCanvas("cvs", "cvs", 800, 600);
	cvs->cd();
	frame->Draw();
	cvs->SaveAs("RooFitResult.png");
	*/
	cout << "chi2 = " << frame->chiSquare() << endl;
	
}

TH1* get_hist(TString sample, TString histkey, TString object)
{
	TFile *f;
	if (object == "electron")
		f = new TFile("Outputs/electron/SkimSglEle__WP90__/SglEle_" + sample + ".root");
	else if (object == "muon")
		f = new TFile("Outputs/muon/SglMu_" + sample + ".root");
	else
		exit(EXIT_FAILURE);

	TH1D *h = (TH1D *)f->Get(histkey);
	h->SetDirectory(nullptr);
	// h->Rebin(5);
	// Rebin with variable bin size
	/*	
	const double xbins[]
		= {0, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 140, 150, 160, 180, 200, 250, 300};
	const unsigned int nbins = 33;
	TH1* h_rebinned = h->Rebin(nbins, "MT_"+sample, xbins);
	h_rebinned->SetDirectory(nullptr);
	f->Close();
	return h_rebinned;
	*/
	return h;
}
