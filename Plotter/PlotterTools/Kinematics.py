from ROOT import TFile, TLegend, TH1D, THStack
from Plotter.PlotterTools.PlotterBase import PlotterBase


class Kinematics(PlotterBase):
    def __init__(self, cvs_params, hist_params, info_params):
        logy = cvs_params['logy']
        grid = cvs_params['grid']
        super().__init__(cvs_type="default", logy=logy, grid=grid)
        self.hist_params = hist_params
        self.info_params = info_params
        self.__set_legend()

    def get_hists(self, h_sigs, h_bkgs):
        print("INFO: Storing histograms...")
        print("INFO: Histograms will be automatically normalized")
        self.h_sigs = dict()
        self.h_bkgs = dict()
        self.total = None

        for name, hist in h_sigs.items():
            # hist.Scale(1./hist.Integral())
            self.h_sigs[name] = hist

        for hist in h_bkgs.values():
            if self.total == None:
                self.total = hist.Clone("temp")
            else:
                self.total.Add(hist)
        for name, hist in h_bkgs.items():
            # hist.Scale(1./total.Integral())
            self.h_bkgs[name] = hist

        self.__decorate_hists()
        self.__stack_and_syst()

    def combine(self):
        info = self.info_params['info']
        cms_text = self.info_params['cms_text']
        extra_text = self.info_params['extra_text']

        super().set_canvas()
        super().set_logo()
        super().set_info()

        super().cvs().cd()
        for hist in self.h_sigs.values():
            hist.Draw("hist&l")
        self.stack.Draw("hist&pfc&same")
        self.syst.Draw("e2&f&same")
        self.legend.Draw("same")
        for hist in self.h_sigs.values():
            hist.Draw("hist&l&same")
        super().info().DrawLatexNDC(0.71, 0.91, info)
        super().logo().DrawLatexNDC(0.15, 0.84, cms_text)
        super().extra_logo().DrawLatexNDC(0.15, 0.80, extra_text)

    def __set_legend(self):
        self.legend = TLegend(0.60, 0.60, 0.87, 0.87)
        self.legend.SetFillStyle(0)
        self.legend.SetBorderSize(0)
        self.legend.SetNColumns(2)

    def __decorate_hists(self):
        # basic setttings
        for hist in self.h_sigs.values():
            hist.SetStats(0)
        for hist in self.h_bkgs.values():
            hist.SetStats(0)

        # rebin
        if "rebin" in self.hist_params.keys():
            rebin = self.hist_params["rebin"]
            for hist in self.h_sigs.values():
                hist.Rebin(rebin)
            for hist in self.h_bkgs.values():
                hist.Rebin(rebin)

        # x axis
        if "x_range" in self.hist_params.keys():
            x_range = self.hist_params['x_range']
            for hist in self.h_sigs.values():
                hist.GetXaxis().SetRangeUser(x_range[0], x_range[1])
                hist.GetXaxis().SetTitle(self.hist_params['x_title'])
                # hist.GetXaxis().SetTitleSize(0.04)
                hist.GetXaxis().SetTitleOffset(1.0)
                hist.GetXaxis().SetLabelSize(0.04)
            for hist in self.h_bkgs.values():
                hist.GetXaxis().SetRangeUser(x_range[0], x_range[1])
                hist.GetXaxis().SetTitle(self.hist_params['x_title'])
                # hist.GetXaxis().SetTitleSize(0.04)
                hist.GetXaxis().SetTitleOffset(1.0)
                hist.GetXaxis().SetLabelSize(0.04)

        # y axis and style
        print("INFO: Automatically set y axis range")
        maximum = self.total.GetMaximum()
        for hist in self.h_sigs.values():
            maximum = max(maximum, hist.GetMaximum())
        for hist in self.h_bkgs.values():
            maximum = max(maximum, hist.GetMaximum())
        
        colors = [2, 4, 8, 3, 9]
        idx = 0
        for name, hist in self.h_sigs.items():
            hist.GetYaxis().SetTitle(self.hist_params['y_title'])
            hist.GetYaxis().SetTitleOffset(1.4)
            hist.GetYaxis().SetRangeUser(0, maximum*3)
            if self.logy:
                hist.GetYaxis().SetRangeUser(1, maximum*30)
            hist.SetLineColor(colors[idx])
            hist.SetLineWidth(2)
            idx += 1
            self.legend.AddEntry(hist, name, 'l')
        for name, hist in self.h_bkgs.items():
            hist.GetYaxis().SetTitle(self.hist_params['y_title'])
            hist.GetYaxis().SetTitleOffset(1.4)
            hist.GetYaxis().SetRangeUser(0, maximum*3)
            if self.logy:
                hist.GetYaxis().SetRangeUser(1, maximum*30)
            self.legend.AddEntry(hist, name, 'f')

    def __stack_and_syst(self):
        self.stack = THStack()
        self.syst = None
        for name, hist in self.h_bkgs.items():
            self.stack.Add(hist)
            if self.syst == None:
                self.syst = hist.Clone("syst")
            else:
                self.syst.Add(hist)
        self.stack.Draw()   # need to draw stack first to change axis
        self.stack.GetHistogram().GetXaxis().SetLabelSize(0)

        self.syst.SetFillColorAlpha(12, 0.6)
        self.syst.SetFillStyle(3144)
        self.syst.GetXaxis().SetLabelSize(0)
        self.legend.AddEntry(self.syst, 'stat error', 'f')
