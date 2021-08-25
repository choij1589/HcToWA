from ROOT import TFile, THStack, TLegend
from Plotter.PlotterTools.PlotterBase import PlotterBase
#from PlotterBase import PlotterBase


class ObsAndExp(PlotterBase):
    def __init__(self, cvs_params, hist_params, info_params):
        logy = cvs_params['logy']
        grid = cvs_params['grid']
        super().__init__(cvs_type="ratio", logy=logy, grid=grid)
        self.hist_params = hist_params
        self.info_params = info_params
        self.__set_legend()

    def __set_legend(self):
        self.legend = TLegend(0.65, 0.50, 0.93, 0.87)
        self.legend.SetFillStyle(0)
        self.legend.SetBorderSize(0)

    def get_hists(self, hist_obs, hists_exp):
        print("INFO: Storing histograms...")
        self.hist_obs = hist_obs
        self.hists_exp = hists_exp
        self.stack = THStack('stack', '')
        self.syst = None
        self.ratio = None
        self.ratio_syst = None

        self.__decorate_hists()
        self.__make_stack_and_syst()
        self.__make_ratio()

    def __decorate_hists(self):
        # basic settings
        self.hist_obs.SetStats(0)
        for hist in self.hists_exp.values():
            hist.SetStats(0)

        if "rebin" in self.hist_params.keys():
            rebin = self.hist_params['rebin']
            self.hist_obs.Rebin(rebin)
            for hist in self.hists_exp.values():
                hist.Rebin(rebin)

        # x axis
        self.hist_obs.GetXaxis().SetLabelSize(0)
        for hist in self.hists_exp.values():
            hist.GetXaxis().SetLabelSize(0)

        if "x_range" in self.hist_params.keys():
            x_range = self.hist_params['x_range']
            self.hist_obs.GetXaxis().SetRangeUser(x_range[0], x_range[1])
            for hist in self.hists_exp.values():
                hist.GetXaxis().SetRangeUser(x_range[0], x_range[1])

        # y axis
        print("INFO: y axis range set to be maximum of data plot")
        maximum = self.hist_obs.GetMaximum()
        y_title = self.hist_params['y_title']
        self.hist_obs.GetYaxis().SetTitle(y_title)
        self.hist_obs.GetYaxis().SetTitleOffset(1.4)
        self.hist_obs.GetYaxis().SetRangeUser(0, maximum*1.8)
        if self.logy:
            self.hist_obs.GetYaxis().SetRangeUser(1, maximum*30)

        # style
        self.hist_obs.SetMarkerStyle(8)
        self.hist_obs.SetMarkerSize(0.4)
        self.hist_obs.SetMarkerColor(1)

        # add to legend
        self.legend.AddEntry(self.hist_obs, "Data", "lep")
        for name, hist in self.hists_exp.items():
            self.legend.AddEntry(hist, name, "f")

    def __make_stack_and_syst(self):
        for hist in self.hists_exp.values():
            self.stack.Add(hist)

            # for systematics
            if self.syst == None:
                self.syst = hist.Clone("syst")
            else:
                self.syst.Add(hist)

        # stack
        self.stack.Draw()   # need to draw stack first to change axis
        self.stack.GetHistogram().GetXaxis().SetLabelSize(0)

        # syst
        # TODO: Add systematic errors
        self.syst.SetStats(0)
        self.syst.SetFillColorAlpha(12, 0.6)
        self.syst.SetFillStyle(3144)
        self.syst.GetXaxis().SetLabelSize(0)
        self.legend.AddEntry(self.syst, 'stat+syst', 'f')

    def __make_ratio(self):
        ratio_range = self.hist_params['ratio_range']
        x_title = self.hist_params['x_title']

        self.ratio = self.hist_obs.Clone("ratio")
        self.ratio.Divide(self.syst)
        self.ratio_syst = self.ratio.Clone("ratio_syst")

        self.ratio.SetStats(0)
        self.ratio.SetTitle("")

        # x axis
        self.ratio.GetXaxis().SetTitle(x_title)
        self.ratio.GetXaxis().SetTitleSize(0.1)
        self.ratio.GetXaxis().SetTitleOffset(0.8)
        self.ratio.GetXaxis().SetLabelSize(0.08)

        # y axis
        self.ratio.GetYaxis().SetRangeUser(ratio_range[0], ratio_range[1])
        self.ratio.GetYaxis().SetTitle("Obs / Exp")
        self.ratio.GetYaxis().CenterTitle()
        self.ratio.GetYaxis().SetTitleSize(0.1)
        self.ratio.GetYaxis().SetTitleOffset(0.4)
        self.ratio.GetYaxis().SetLabelSize(0.08)

        # ratio_syst
        self.ratio_syst.SetStats(0)
        self.ratio_syst.SetFillColorAlpha(12, 0.6)
        self.ratio_syst.SetFillStyle(3144)

    def combine(self):
        info = self.info_params['info']
        cms_text = self.info_params['cms_text']
        extra_text = self.info_params['extra_text']

        super().set_canvas()
        super().set_logo()
        super().set_info()

        super().pad_up().cd()
        self.hist_obs.Draw("p&hist")
        self.stack.Draw("hist&pfc&same")
        self.syst.Draw("e2&f&same")
        self.hist_obs.Draw("p&hist&same")
        self.hist_obs.Draw("e1&same")
        self.legend.Draw()
        super().info().DrawLatexNDC(0.74, 0.91, info)
        super().logo().DrawLatexNDC(0.15, 0.83, cms_text)
        super().extra_logo().DrawLatexNDC(0.15, 0.78, extra_text)
        super().pad_up().RedrawAxis()

        super().pad_down().cd()
        self.ratio.Draw("p & hist")
        self.ratio_syst.Draw("e2&f&same")
        super().pad_down().RedrawAxis()

        super().cvs().cd()
        super().pad_up().Draw()
        super().pad_down().Draw()


if __name__ == "__main__":
    cvs_params = {"logy": True,
                  "grid": True
                  }
    hist_params = {"x_title": "p_{T}(#mu)",
                   "x_range": [0., 200.],
                   "y_title": "Events",
                   "ratio_title": "Obs/Exp",
                   "ratio_range": [0.5, 1.5]
                   }
    info_params = {"info": "L^{int} = 41.5 fb^{-1}",
                   "cms_text": "CMS",
                   "extra_text": "Work on progress"
                   }
    # Get hitograms
    hist_obs = None
    hists_exp = {}

    skflat_output = "/data0/CRstudy/2017/"
    data = "DoubleMuon"
    #mcsamples = ["DYJets", "TTLL", "fake", "VV", "SingleTop"]
    mcsamples = ["fake", "VV", "SingleTop", "TTLL", "DY"]
    channel = "TT_DiMu"

    path_to_hist = channel + "/muons_tight/1/pt"
    file = TFile(skflat_output+"DATA/CRstudy_"+data+".root")
    hist = file.Get(path_to_hist)
    hist.SetDirectory(0)
    hist_obs = hist
    for name in mcsamples:
        if name == "TTLL":
            file = TFile(skflat_output+"/CRstudy_TTLL_powheg.root")
        else:
            file = TFile(skflat_output+"/CRstudy_"+name+".root")
        hist = file.Get(path_to_hist)
        hist.SetDirectory(0)
        hists_exp[name] = hist

    plotter = ObsAndExp(cvs_params, hist_params, info_params)
    plotter.get_hists(hist_obs, hists_exp)
    plotter.combine()
    plotter.save("DataAndMC.png")
