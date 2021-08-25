from ROOT import TFile, TLegend, TH1D
from PlotterBase import PlotterBase
from math import sqrt


class SignalSign(PlotterBase):
    def __init__(self, cvs_params, hist_params, info_params):
        logy = cvs_params['logy']
        grid = cvs_params['grid']
        super().__init__(cvs_type="default", logy=logy, grid=grid)
        self.hist_params = hist_params
        self.info_params = info_params
        self.__set_legend()

    def __set_legend(self):
        self.legend = TLegend(0.12, 0.70, 0.37, 0.89)
        self.legend.SetFillStyle(0)
        self.legend.SetBorderSize(0)

    def get_hists(self, h_sigs, h_bkg):
        # store histograms
        print("INFO: Storing histograms....")
        self.h_sigs = h_sigs
        self.h_bkg = h_bkg

        self.__make_SoverSqrtB()
        self.__decorate_hists()

    def __make_SoverSqrtB(self):
        self.h_targets = {}
        nbins = self.h_bkg.GetXaxis().GetNbins()
        for i in range(1, nbins+1):
            sqrtB = sqrt(self.h_bkg.GetBinContent(i))
            self.h_bkg.SetBinContent(i, sqrtB)

        for name, hist in self.h_sigs.items():
            print(type(hist))
            self.h_targets[name] = hist.Clone("target_" + name)
            self.h_targets[name].Divide(self.h_bkg)

    def __decorate_hists(self):
        x_title = self.hist_params['x_title']
        y_title = self.hist_params['y_title']
        cuts = self.hist_params['cuts']
        nbins = self.h_bkg.GetXaxis().GetNbins()
        maximum = max([hist.GetMaximum() for hist in self.h_targets.values()])

        for name, hist in self.h_targets.items():
            hist.SetStats(0)
            hist.SetLineWidth(2)
            hist.GetXaxis().SetTitle(x_title)
            hist.GetXaxis().SetTitleOffset(0.9)
            hist.GetYaxis().SetTitle(y_title)
            hist.GetYaxis().SetTitleOffset(0.8)
            if self.logy:
                hist.GetYaxis().SetRangeUser(1, maximum*4)
            else:
                hist.GetYaxis().SetRangeUser(0, maximum*1.3)
            for i in range(nbins):
                hist.GetXaxis().SetBinLabel(i+1, cuts[i])
            self.legend.AddEntry(hist, name, "lep")

    def combine(self):
        info = self.info_params['info']
        cms_text = self.info_params['cms_text']
        extra_text = self.info_params['extra_text']

        super().set_canvas()
        super().set_logo()
        super().set_info()
        super().cvs().cd()
        for name, hist in self.h_targets.items():
            hist.Draw("plc&same")
        self.legend.Draw("same")
        super().info().DrawLatexNDC(0.71, 0.91, info)
        super().logo().DrawLatexNDC(0.60, 0.84, cms_text)
        super().extra_logo().DrawLatexNDC(0.60, 0.80, extra_text)


if __name__ == "__main__":
    cvs_params = {"logy": False,
                  "grid": True}
    hist_params = {"x_title": "cuts",
                   "y_title": "S/#sqrt{B}",
                   # "cuts": ["nocut", "metfilter", "3mu", "trigger", "passSafePtCut", "exist_osmu", "Nj_ge2", "Nb_ge1"]
                   "cuts": ["nocut", "metfilter", "1e2mu", "trigger",  "passSafePtCut", "exist_osmu", "Nj_ge2", "Nb_ge1"]
                   }
    info_params = {"info": "L^{int} = 41.5 fb^{-1}",
                   "cms_text": "CMS",
                   "extra_text": "Work on progress"}
    output_path = "Outputs/Preselection/RunLowPtJet__/SignalSign/"

    SKFlat_output = "/data0/HcToWA/Preselection/RunLowPtJet__/2017/"
    signal_points_MHc70 = ['MHc70_MA15', 'MHc70_MA40', 'MHc70_MA65']
    signal_points_MHc100 = ['MHc100_MA15',
                            'MHc100_MA25', 'MHc100_MA60', 'MHc100_MA95']
    signal_points_MHc130 = ['MHc130_MA15', 'MHc130_MA45',
                            'MHc130_MA55', 'MHc130_MA90', 'MHc130_MA125']
    signal_points_MHc160 = ['MHc160_MA15', 'MHc160_MA45',
                            'MHc160_MA75', 'MHc160_MA85', 'MHc160_MA120', 'MHc160_MA155']
    channel = "Pre_3mu"
    h_sigs = {}
    for name in signal_points_MHc70:
        file = TFile(
            SKFlat_output+"SignalSamples/Preselection_TTToHcToWA_AToMuMu_" + name + ".root")
        hist = file.Get("cutflow/" + channel).Clone(name)
        # This is necessary to keep the type of the hist!!!
        hist.SetDirectory(0)
        h_sigs[name] = hist

    file_bkg = TFile(
        SKFlat_output + "TriLepSamples/Preselection_background.root")
    h_bkg = file_bkg.Get("cutflow/" + channel)

    plotter = SignalSign(cvs_params, hist_params, info_params)
    plotter.get_hists(h_sigs, h_bkg)
    plotter.combine()
    plotter.save(output_path + channel + "_MHc70.png")
