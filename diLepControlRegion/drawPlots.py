import os
import argparse
from math import pow, sqrt
from ROOT import TFile
from Plotter.plotter import Canvas
from histConfigs import hist_configs

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--era", "-e", default=None, required=True, type=str, help="Era")
parser.add_argument("--var", "-v", default=None, required=True, type=str, help="observable")
parser.add_argument("--out", "-o", default=None, required=True, type=str, help="output path")
args = parser.parse_args()
config = hist_configs[args.var]

# get histograms
channel = "dimuon_drellyan_candidate"
MCs = ["DYJets",
       "TTLL_powheg",
       "SingleTop_sch_Lep", "SingleTop_tW_top_NoFullyHad", "SingleTop_tW_antitop_NoFullyHad",
       "WW_pythia", "WZ_pythia", "ZZ_pythia"]
#systematics = ["Central", ["PileUpCorrUp", "PileUpCorrDown"], ["MuonEnUp", "MuonEnDown"], ["MuonIDSFUp", "MuonIDSFDown"], ["MuonTrigSFUp", "MuonTrigSFDown"]]
#systematics = ["Default"]
systematics = ["MuonIDSFOnly", ["MuonIDSFOnlyUp", "MuonIDSFOnlyDown"]]

f_data = TFile.Open(f"SKFlatOutput/{args.era}/DATA/diLepControlRegion_DoubleMuon.root")
h_data = f_data.Get(f"{channel}/Central/{args.var}"); h_data.SetDirectory(0)
f_data.Close()


MCcoll = {}
for mc in MCs:
    # get histograms
    central, *systs = systematics
    f = TFile.Open(f"SKFlatOutput/{args.era}/diLepControlRegion_{mc}.root")
    h_central = f.Get(f"{channel}/{central}/{args.var}"); h_central.SetDirectory(0)
    h_systs = []
    for syst in systs:
        h_up = f.Get(f"{channel}/{syst[0]}/{args.var}"); h_up.SetDirectory(0)
        h_down = f.Get(f"{channel}/{syst[1]}/{args.var}"); h_down.SetDirectory(0)
        h_systs.append([h_up, h_down])
    f.Close()

    for bin in range(h_central.GetNcells()):
        this_value, this_error = h_central.GetBinContent(bin), h_central.GetBinError(bin)
        this_error = pow(this_error, 2)
        for syst in h_systs:
            this_syst_up = syst[0].GetBinContent(bin) - this_value
            this_syst_down = syst[1].GetBinContent(bin) - this_value
            this_syst = this_syst_up if abs(this_syst_up) > abs(this_syst_down) else this_syst_down
            this_error += pow(this_syst, 2)
        this_error = sqrt(this_error)
        h_central.SetBinError(bin, this_error)
    MCcoll[mc] = h_central

#### Set histogram names
histograms = {}
# data
h_data.SetName("data"); h_data.SetTitle("kBlack")
# DY
h_DY = MCcoll['DYJets'].Clone("DY"); h_DY.SetTitle("kGreen")
# TT
h_TT = MCcoll["TTLL_powheg"].Clone("TT"); h_TT.SetTitle("kBlue")
# ST
h_ST = MCcoll["SingleTop_sch_Lep"].Clone("ST");
h_ST.Add(MCcoll["SingleTop_tW_top_NoFullyHad"])
h_ST.Add(MCcoll["SingleTop_tW_antitop_NoFullyHad"])
h_ST.SetTitle("kMagenta")
# VV
h_VV = MCcoll["WW_pythia"].Clone("VV");
h_VV.Add(MCcoll["WZ_pythia"])
h_VV.Add(MCcoll["ZZ_pythia"])
h_VV.SetTitle("kRed")

histograms["data"] = h_data
histograms["DY"] = h_DY
histograms["TT"] = h_TT
histograms["ST"] = h_ST
histograms["VV"] = h_VV

# Draw plots
c = Canvas()
c.draw_distributions(histograms, config)
c.draw_ratio(config)
c.draw_legend(config)
c.draw_latex(config, args.era)
c.finalize()
c.savefig(f"{args.out}")
