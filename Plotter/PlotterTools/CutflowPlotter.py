from ROOT import TFile, TLegend, THStack
from PlotterBase import PlotterBase


class CutflowPlotter(PlotterBase):
    def __init__(self, cvs_params, hist_params, info_params):
        logy = cvs_params['logy']
        grid = cvs_params['grid']
        super().__init__(cvs_type='default', logy=logy, grid=grid)
        self.hist_params = hist_params
        self.info_params = info_params
        self.__set_legend()

    def __set_legend(self):
        self.legend = TLegend(0.49, 0.70, 0.90, 0.90)
        self.legend.SetNColumns(2)
        self.legend.SetFillStyle(0)
        self.legend.SetBorderSize(0)

    def get_hists(self, h_sigs, h_bkgs):
        self.h_sigs = {}            # will be normalized
        self.h_bkgs = {}
        self.h_total_bkg = None     # used for systematic and normalization of h_sigs
        self.stack = THStack('stack', "")

        # store histograms
        print("INFO: Storing histograms")
        for name, hist in h_sigs.items():
            self.h_sigs[name] = hist
        for name, hist in h_bkgs.items():
            self.h_bkgs[name] = hist

        self.__make_stack_and_total_bkg()
        self.__normalize_signals()

    def __make_stack_and_total_bkg(self):
        x_title = self.hist_params['x_title']
        y_title = self.hist_params['y_title']
        cuts = self.hist_params['cuts']
        for name, hist in self.h_bkgs.items():
            self.stack.Add(hist)
            self.legend.AddEntry(hist, name, 'f')

            if self.h_total_bkg == None:
                self.h_total_bkg = hist.Clone("bkg")
            else:
                self.h_total_bkg.Add(hist)

        self.stack.Draw()   # need to draw stack first to change axis
        nbins = self.stack.GetHistogram().GetXaxis().GetNbins()
        for i in range(nbins):
            self.stack.GetHistogram().GetXaxis().SetBinLabel(i+1, cuts[i])

        self.h_total_bkg.SetStats(0)
        self.h_total_bkg.SetFillColorAlpha(12, 0.6)
        self.h_total_bkg.SetFillStyle(3144)
        self.h_total_bkg.GetXaxis().SetTitle(x_title)
        self.h_total_bkg.GetXaxis().SetTitleOffset(0.9)
        self.h_total_bkg.GetYaxis().SetTitle(y_title)
        self.h_total_bkg.GetYaxis().SetTitleOffset(1.4)
        maximum = self.h_total_bkg.GetMaximum()
        if self.logy:
            self.h_total_bkg.GetYaxis().SetRangeUser(1, maximum*1000)
        else:
            self.h_total_bkg.GetYaxis().SetRangeUser(0, maximum*1.8)
        for i in range(nbins):
            self.h_total_bkg.GetXaxis().SetBinLabel(i+1, cuts[i])
        self.legend.AddEntry(self.h_total_bkg, "stat error", "f")

    def __normalize_signals(self):
        x_title = self.hist_params['x_title']
        y_title = self.hist_params['y_title']
        cuts = self.hist_params['cuts']
        nbins = self.h_total_bkg.GetXaxis().GetNbins()
        maximum = self.h_total_bkg.GetMaximum()

        for name, hist in self.h_sigs.items():
            scale = self.h_total_bkg.Integral() / hist.Integral()
            hist.Scale(scale)
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
            self.legend.AddEntry(hist, name, 'lep')

    def combine(self):
        info = self.info_params['info']
        cms_text = self.info_params['cms_text']
        extra_text = self.info_params['extra_text']

        super().set_canvas()
        super().set_logo()
        super().set_info()
        super().cvs().cd()
        self.h_total_bkg.Draw()
        self.stack.Draw("pfc&hist&same")
        self.h_total_bkg.Draw("e2&same")
        for name, hist in self.h_sigs.items():
            hist.Draw("plc&same&hist")
        self.legend.Draw("same")
        super().info().DrawLatexNDC(0.71, 0.91, info)
        super().logo().DrawLatexNDC(0.15, 0.84, cms_text)
        super().extra_logo().DrawLatexNDC(0.15, 0.80, extra_text)


if __name__ == "__main__":
    cvs_params = {"logy": True,
                  "grid": True}
    hist_params = {"x_title": "cuts",
                   "y_title": "Events",
                   "cuts": ["nocut", "metfilter", "nleptons", "trigger",  "passSafePtCut", "Nj_ge1", "Nj_ge2", "Nb_ge1", "exist_osmu", "off_Zmass"]
                   }
    info_params = {"info": "L^{int} = 41.5 fb^{-1}",
                   "cms_text": "CMS",
                   "extra_text": "Work on progress"}

    SKFlat_output = "/data0/SigToBkg/SelectionScan/2017/"
    signal_points_MHc70 = ['MHc70_MA15', 'MHc70_MA40', 'MHc70_MA65']
    signal_points_MHc100 = ['MHc100_MA15',
                            'MHc100_MA25', 'MHc100_MA60', 'MHc100_MA95']
    signal_points_MHc130 = ['MHc130_MA15', 'MHc130_MA45',
                            'MHc130_MA55', 'MHc130_MA90', 'MHc130_MA125']
    signal_points_MHc160 = ['MHc160_MA15', 'MHc160_MA45',
                            'MHc160_MA75', 'MHc160_MA85', 'MHc160_MA120', 'MHc160_MA155']
    backgrounds = ['rare', 'ttX', 'VV', 'TTLJ_powheg', 'TTLL_powheg', 'DYJets']
    channel = "3mu"
    h_sigs = {}
    h_bkgs = {}
    for name in signal_points_MHc100:
        file = TFile(
            SKFlat_output+"signal/SigToBkg_TTToHcToWA_AToMuMu_" + name + ".root")
        hist = file.Get("cutflow/" + channel).Clone(name)
        hist.SetDirectory(0)
        h_sigs[name] = hist

    for name in backgrounds:
        print(name)
        file = TFile(
            SKFlat_output+"background/SigToBkg_" + name + ".root")
        hist = file.Get("cutflow/" + channel).Clone(name)
        hist.SetDirectory(0)
        h_bkgs[name] = hist

    plotter = CutflowPlotter(cvs_params, hist_params, info_params)
    plotter.get_hists(h_sigs, h_bkgs)
    plotter.combine()
    plotter.save("test.png")
