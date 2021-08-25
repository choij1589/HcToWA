import argparse
from ROOT import TFile, TCanvas, TH1D
import numpy as np
from Plotter.PlotterTools.ObsAndExp import ObsAndExp
from Parameters import cvs_params, info_params, param_list

parser = argparse.ArgumentParser(description="selectorArgs")
parser.add_argument("--histkey", default=None,
                    required=True, type=str, help="histkey")
args = parser.parse_args()

# Define global variables
f = TFile.Open("ZFake_3Mu.root")
PROPMTs = ["rare", "conv", "ttX", "VV"]
SYSTs = ["Central", "L1PrefireUp", "L1PrefireDown", "PUReweightUp",
         "PUReweightDown", "IDSFUp", "IDSFDown", "TrigSFUp", "TrigSFDown"]
histkey = args.histkey
hists = dict()


def get_data_hist(histkey):
    base_dir = f.Get("data")
    h = base_dir.Get(histkey)
    h.SetDirectory(0)
    return h


def get_fake_hist(histkey, syst="Central"):
    base_dir = f.Get("fake/"+syst)
    h = base_dir.Get(histkey)
    h.SetDirectory(0)
    return h


def get_hist(mc, histkey, syst="Central"):
    base_dir = f.Get(mc+"/"+syst)
    h = base_dir.Get(histkey)
    h.SetDirectory(0)
    return h


# get data
h_data = get_data_hist(histkey)

# estimate systematics for fake
h_fakes = dict()
h_fakes["Central"] = get_fake_hist(histkey, syst="Central")
h_fakes["Up"] = get_fake_hist(histkey, syst="Up")
h_fakes["Down"] = get_fake_hist(histkey, syst="Down")

h_fake = h_fakes["Central"].Clone("h_fake")
h_fake.SetDirectory(0)
bins = h_fake.GetNbinsX()
for bin in range(bins+1):
    center = h_fake.GetBinContent(bin)
    upper = h_fakes["Up"].GetBinContent(bin) - center
    lower = center - h_fakes["Down"].GetBinContent(bin)
    total = np.sqrt(pow(upper, 2) + pow(lower, 2))
    print(f"bin: {h_fake.GetBinLowEdge(bin)}, err: {total, center}")
    h_fake.SetBinError(bin, total)

hists['fake'] = h_fake

for mc in PROPMTs:
    print(mc)
    # get all systmatics
    temp = dict()
    # get histograms
    for syst in SYSTs:
        temp[syst] = get_hist(mc, histkey, syst)
    h = temp["Central"].Clone("h_"+mc)
    h.SetDirectory(0)
    bins = h.GetNbinsX()
    for bin in range(bins+1):
        center = h.GetBinContent(bin)
        error = pow(h.GetBinError(bin), 2)
        for syst in SYSTs:
            error += pow(temp[syst].GetBinContent(bin) - center, 2)
        error = np.sqrt(error)
        #print(f"{mc}, bin: {bin}, error: {error}")
        h.SetBinError(bin, error)
    hists[mc] = h

hist_params = param_list[histkey]

plotter = ObsAndExp(cvs_params, hist_params, info_params)
plotter.get_hists(h_data, hists)
plotter.combine()
plotter.save("plots/3mu/"+histkey.replace("/", "_")+".png")
