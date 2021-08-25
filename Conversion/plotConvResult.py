import argparse
import numpy as np
from ROOT import TFile
from Plotter.PlotterTools.ObsAndExp import ObsAndExp
from Plotter.Parameters.params_trilep import cvs_params, info_params, muon_params, electron_params, jet_params, ZCand_params, METv_params
from measConvSF import get_hist, get_conv_sf

parser = argparse.ArgumentParser()
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help="1E2Mu or 3Mu")
# parser.add_argument("--region", "-r", default=None, required=True, type=str, help="DY or ZG")
args = parser.parse_args()

# global variables
PROMPTs = ["rare", "ttX", "VV"]
channel = args.channel
# region = args.region

if channel == "1E2Mu":
    histkey_muons = ["muons/1/pt", "muons/1/eta", "muons/1/phi",
                     "muons/2/pt", "muons/2/eta", "muons/2/phi"]
    histkey_electrons = ["electrons/1/pt", "electrons/1/eta", "electrons/1/phi"]
elif channel == "3Mu":
    histkey_muons = ["muons/1/pt", "muons/1/eta", "muons/1/phi",
                     "muons/2/pt", "muons/2/eta", "muons/2/phi",
                     "muons/3/pt", "muons/3/eta", "muons/3/phi"]
    histkey_electrons = []
else:
    raise(AttributeError)

histkey_jets = ["jets/1/pt", "jets/1/eta", "jets/1/phi",
                "jets/2/pt", "jets/2/eta", "jets/2/phi",
                "jets/3/pt", "jets/3/eta", "jets/3/phi",
                "jets/size"]
histkey_ZCand = ["ZCand/pt", "ZCand/eta", "ZCand/phi", "ZCand/mass"]
histkey_METv = ["METv/pt", "METv/phi"]
histkeys = histkey_muons + histkey_electrons + histkey_jets + histkey_ZCand + histkey_METv

def plot_hist(histkey, hist_params):
    h_data = get_hist("DATA", histkey, channel, "DY")
    h_data.Add(get_hist("DATA", histkey, channel, "ZG"))

    hists = dict()
    hists['fake'] = get_hist("fake", histkey, channel, "DY")
    hists['fake'].Add(get_hist("fake", histkey, channel, "ZG"))
    for prompt in PROMPTs:
        hists[prompt] = get_hist(prompt, histkey, channel, "DY")
        hists[prompt].Add(get_hist(prompt, histkey, channel, "ZG"))

    h_DY = get_hist("DY", histkey, channel, "DY")
    DY_sf, DY_err = get_conv_sf(channel, "DY")
    for bin in range(h_DY.GetNbinsX()+1):
        center = h_DY.GetBinContent(bin) * DY_sf
        upper = h_DY.GetBinContent(bin) * (DY_sf + DY_err) - center
        lower = center - h_DY.GetBinContent(bin) * (DY_sf - DY_err)
        err = np.sqrt(np.power(upper, 2) + np.power(lower, 2))
        h_DY.SetBinContent(bin, center)
        h_DY.SetBinError(bin, err)
    hists['conv'] = h_DY

    h_ZG = get_hist("ZG", histkey, channel, "ZG")
    ZG_sf, ZG_err = get_conv_sf(channel, "ZG")
    for bin in range(h_ZG.GetNbinsX()+1):
        center = h_ZG.GetBinContent(bin) * ZG_sf
        upper = h_ZG.GetBinContent(bin) * (ZG_sf + ZG_err) - center
        lower = center - h_ZG.GetBinContent(bin) * (ZG_sf - ZG_err)
        err = np.sqrt(np.power(upper, 2) + np.power(lower, 2))
        h_ZG.SetBinContent(bin, center)
        h_ZG.SetBinError(bin, err)
    hists['conv'].Add(h_ZG)

    plotter = ObsAndExp(cvs_params, hist_params, info_params)
    plotter.get_hists(h_data, hists)
    plotter.combine()
    plotter.save(f"plots/chan{args.channel}_{histkey.replace('/', '_')}.png")

if __name__ == "__main__":
    for histkey in histkey_muons:
        hist_params = muon_params[histkey]
        plot_hist(histkey, hist_params)

    for histkey in histkey_electrons:
        hist_params = electron_params[histkey]
        plot_hist(histkey, hist_params)

    for histkey in histkey_jets:
        hist_params = jet_params[histkey]
        plot_hist(histkey, hist_params)

    for histkey in histkey_ZCand:
        hist_params = ZCand_params[histkey]
        plot_hist(histkey, hist_params)

    for histkey in histkey_METv:
        hist_params = METv_params[histkey]
        plot_hist(histkey, hist_params)
