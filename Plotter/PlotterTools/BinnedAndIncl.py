from ROOT import TFile, THStack, TLegend
from PlotterBase import PlotterBase


class BinnedAndIncl(PlotterBase):
    """ Add parameters in hist_params to add more functionalities"""

    def __init__(self, cvs_params, hist_params, info_params):
        logy = cvs_params['logy']
        grid = cvs_params['grid']
        super().__init__(cvs_type="ratio", logy=logy, grid=grid)
        self.hist_params = hist_params
        self.info_params = info_params
        self.__set_legend()

    def __set_legend(self):
        self.legend = TLegend(0.69, 0.60, 0.90, 0.90)

    def get_hists(self, h_incl, hs_binned):
        # Initialize parameters
        self.h_incl = None
        self.hs_binned = {}
        self.stack = THStack('stack', "")
        self.syst = None
        self.ratio = None
        self.ratio_syst = None

        # store histograms
        print("INFO: Storing histograms...")
        print("INFO: histograms automatically normalized to L = 150 fb^-1")
        self.h_incl = self.__rebin(h_incl)
        for name, hist in hs_binned.items():
            self.hs_binned[name] = self.__rebin(hist)

        self.__decorate_hists()
        self.__make_stack_and_syst()
        self.__make_ratio()

    def __rebin(self, hist):
        try:
            # rebin histograms
            if "rebin" in self.hist_params.keys():
                rebin = self.hist_params['rebin']
                hist.Rebin(rebin)
            # set x range
            if "x_range" in self.hist_params.keys():
                x_range = self.hist_params['x_range']
                hist.GetXaxis().SetRangeUser(x_range[0], x_range[1])
            return hist
        except Exception as e:
            print("rebin(): Exception occured! -> " + str(e))

    def __decorate_hists(self):
        print("INFO: y axis range set to be maximum of inclusive plot")
        y_range = self.h_incl.GetMaximum()
        y_title = self.hist_params['y_title']

        # decorate
        self.h_incl.SetStats(0)
        self.h_incl.SetMarkerStyle(8)
        self.h_incl.SetMarkerSize(0.5)
        self.h_incl.SetMarkerColor(1)

        # X axis
        self.h_incl.GetXaxis().SetLabelSize(0)

        # Y axis
        self.h_incl.GetYaxis().SetTitle(y_title)
        # Get enough space to set the legend
        self.h_incl.GetYaxis().SetRangeUser(0., y_range*1.3)
        if self.logy:
            self.h_incl.GetYaxis().SetRangeUser(1, y_range*100)

        for name, hist in self.hs_binned.items():
            hist.GetXaxis().SetLabelSize(0)

        # Add to a legendi
        self.legend.AddEntry(self.h_incl, "Incl", "lep")

    def __make_stack_and_syst(self):
        for name, hist in self.hs_binned.items():
            hist.GetXaxis().SetLabelSize(0)
            self.stack.Add(hist)
            self.legend.AddEntry(hist, name, 'f')

            # combine a binned histograms to draw systematics
            if self.syst == None:
                self.syst = hist.Clone("syst")
            else:
                self.syst.Add(hist)

        self.stack.Draw()   # need to draw stack first to change axis
        self.stack.GetHistogram().GetXaxis().SetLabelSize(0)
        self.syst.SetStats(0)
        self.syst.SetFillColorAlpha(12, 0.6)
        self.syst.SetFillStyle(3144)
        self.syst.GetXaxis().SetLabelSize(0)
        self.legend.AddEntry(self.syst, 'stat error', 'f')

    def __make_ratio(self):
        ratio_range = self.hist_params['ratio_range']
        x_title = self.hist_params['x_title']

        self.ratio = self.h_incl.Clone("ratio")
        self.ratio.Divide(self.syst)
        self.ratio_syst = self.ratio.Clone("ratio_syst")

        self.ratio.SetStats(0)
        self.ratio.SetTitle("")
        # y axis
        self.ratio.GetYaxis().SetRangeUser(ratio_range[0], ratio_range[1])
        self.ratio.GetYaxis().SetTitle("Incl / binned")
        self.ratio.GetYaxis().SetTitleSize(0.08)
        self.ratio.GetYaxis().SetTitleOffset(0.5)
        self.ratio.GetYaxis().SetLabelSize(0.08)
        # x axis
        self.ratio.GetXaxis().SetTitle(x_title)
        self.ratio.GetXaxis().SetTitleSize(0.1)
        self.ratio.GetXaxis().SetTitleOffset(0.8)
        self.ratio.GetXaxis().SetLabelSize(0.08)

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
        self.h_incl.Draw("p&hist")
        self.stack.Draw("hist & pfc & same")
        self.syst.Draw("e2 & f & same")
        self.h_incl.Draw("p&hist&same")
        self.h_incl.Draw("e1 & same")

        self.legend.Draw()
        super().info().DrawLatexNDC(0.6, 0.91, info)
        super().logo().DrawLatexNDC(0.15, 0.83, cms_text)
        super().extra_logo().DrawLatexNDC(0.15, 0.78, extra_text)

        super().pad_down().cd()
        self.ratio.Draw("p & hist")
        self.ratio_syst.Draw("e2&f&same")

        super().cvs().cd()
        super().pad_up().Draw()
        super().pad_down().Draw()


if __name__ == "__main__":
    from Parameters.binned_and_incl_params import params

    def plot_maker(obs, root_files, weights, selector_output, output_path, store_name):
        cvs_params = params[obs]["cvs_params"]
        hist_params = params[obs]['hist_params']
        info_params = params[obs]["info_params"]

        hist_name = obs
        hist_incl = None
        hists_binned = {}
        for file_name in file_names:
            if file_name == "DYm50_012j_nlo_cp5":
                dir_name = "DYm50_cp5_GridToNano"
            elif file_name == "DYm50_0j_nlo_cp5":
                dir_name = "DYm50_0j_nlo_cp5_GridToNano"
            elif file_name == "DYm50_1j_nlo_cp5":
                dir_name = "DYm50_1j_nlo_cp5_GridToNano"
            elif file_name == "DYm50_2j_nlo_cp5":
                dir_name = "DYm50_2j_nlo_cp5_GridToNano"
            else:
                dir_name = file_name

            # combine ee and mm channels
            scale = weights[file_name]*lumi*1000
            hist_path = dir_name + "/" + hist_name
            hist_ee = root_files[file_name].Get(
                hist_path + selector_arg + "_ee")
            hist_mm = root_files[file_name].Get(
                hist_path + selector_arg + "_mm")
            hist = hist_ee.Clone(file_name + selector_arg + "_clone")
            hist.Add(hist_mm)
            hist.Scale(scale)

            key = file_name
            if selector_arg == "":
                file_name += "_dressedLep"
            elif selector_arg == "_prefsr":
                file_name += "_matchedGenJet"
            else:
                file_name += selector_arg

            if "012j" in file_name:
                hist_incl = hist
            else:
                hists_binned[key] = hist

        plotter = BinnedAndIncl(cvs_params, hist_params, info_params)
        plotter.get_hists(hist_incl, hists_binned)
        plotter.combine()
        if selector_arg == "":
            path = output_path + store_name + "_" + obs + "_dressedLep" + ".png"
        elif selector_arg == "_prefsr":
            path = output_path + store_name + "_" + obs + "_matchedGenJet" + ".png"
        else:
             + output_pathstore_name + "_" + obs + selector_arg + ".png"
        plotter.save(path)

    # initial settings
    file_names = ["DYm50_012j_nlo_cp5", "DYm50_0j_nlo_cp5",
                  "DYm50_1j_nlo_cp5", "DYm50_2j_nlo_cp5"]
    weights = {
        "DYm50_012j_nlo_cp5": 2.99030301E-03,
        "DYm50_0j_nlo_cp5": 8.43901018E-04,
        "DYm50_1j_nlo_cp5": 7.35992842E-04,
        "DYm50_2j_nlo_cp5": 4.39098262E-04,
        # "DYm50_012j_nlo_cuep5m1" : 2.73003549E-03,
        # "DYm50_0j_nlo_cuep5m1" : 9.57783732E-04,
        # "DYm50_1j_nlo_cuep5m1" : 7.61277377E-04,
        # "DYm50_2j_nlo_cuep5m1" : 3.93164481E-04
    }
    observables = ["ZMass", "yZ", "ptZ", "phiZ",
                   "ptl1", "ptl2", "etal1", "etal2", "phil1", "phil2", "nLeptons",
                   "ptj1", "ptj2", "etaj1", "etaj2", "phij1", "phij2", "nJets"]
    zboson = ['ZMass', 'yZ', 'ptZ', 'phiZ']
    leptons = ['ptl1', 'ptl2', 'etal1', 'etal2', 'phil1', 'phil2', 'nLeptons']
    jets = ['ptj1', 'ptj2', 'etaj1', 'etaj2', 'phij1', 'phij2', 'nJets']
    lumi = 150.  # fb^-1
    selector_arg = ""
    output_path = "/root/workspace/GenValidation/Plotter/Outputs/DrellYan/CP5/DressedLep/"

    # get histograms
    root_files = {}
    selector_output = "/root/workspace/GenValidation/SelectorOutput/DrellYan/"
    store_name = "DYm50_nlo_cp5"
    for name in file_names:
        this_path = selector_output + name + ".root"
        root_files[name] = TFile(this_path)

    for obs in observables:
        plot_maker(obs, root_files, weights,
                   selector_output, output_path, store_name)
