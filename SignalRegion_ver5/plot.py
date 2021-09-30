import os
import numpy as np
import argparse
from ROOT import TFile
from Plotter.PlotterTools.Kinematics import Kinematics
from Plotter.PlotterTools.ObsAndExp import ObsAndExp
from Plotter.Parameters.params_trilep import cvs_params, info_params
from Plotter.Parameters.params_trilep import muon_params, electron_params, jet_params, dimuon_params, METv_params
from Plotter.Parameters.params_trilep import input_params, score_params
from Conversion.measConvSF import get_conv_sf

# parse arguments, define global variables
parser = argparse.ArgumentParser()
parser.add_argument("--channel", "-c", default=None, required=True, type=str)
parser.add_argument("--region", "-r", default=None, required=True, type=str)
parser.add_argument("--dist", "-d", default=None, required=True, type=str)
parser.add_argument("--mHc", "-m", default=130, type=int)
args = parser.parse_args()

CHANNEL = args.channel
REGION = args.region
DISTRIBUTIONS = args.dist
MHc = args.mHc
DEFAULT_MASS_POINT = "MHc130_MA90"
# MHc: [MAs]
MASS_POINTs = {
        70: [15, 40, 65],
        100: [15, 25, 60, 95],
        130: [15, 45, 55, 90, 125],
        160: [15, 45, 75, 85, 120, 155]
}

def get_hist(sample, channel, mass_point, sub_dir, region, histkey, syst="Central"):
    if sample == "fake":
        fkey = f"Outputs/{channel}/{mass_point}/ROOT/DATA.root"
    else:
        fkey = f"Outputs/{channel}/{mass_point}/ROOT/{sample}.root"
    histkey = f"{sample}/{sub_dir}/{region}/{syst}/{histkey}"

    f = TFile.Open(fkey)
    h = f.Get(histkey)

    try:
        h.SetDirectory(0)
    except Exception as e:
        print(f"null histogram!")
        print(f"fkey: {fkey}")
        print(f"histkey: {histkey}")
        #print(e)
        #raise(KeyError)
        return None
    f.Close()

    return h

def get_signal_hists(histkey, MHc=130):
    hists = dict()
    # devide histkey
    keys = histkey.split("/")
    sub_dir, histkey = keys[0], "/".join(keys[1:])

    # make masspoints
    mass_points = []
    for MA in MASS_POINTs[MHc]:
        mass_points.append(f"MHc{MHc}_MA{MA}")

    # get histograms
    for mass_point in mass_points:
        hists[mass_point] = get_hist(sample=f"TTToHcToWA_AToMuMu_{mass_point}",
                                     channel=CHANNEL,
                                     mass_point=mass_point,
                                     sub_dir=sub_dir,
                                     region=REGION,
                                     histkey=histkey)

    return hists

def get_data_hist(histkey, mass_point=DEFAULT_MASS_POINT):
    keys = histkey.split("/")
    sub_dir, histkey = keys[0], "/".join(keys[1:])

    h_data = get_hist(sample="DATA",
                      channel=CHANNEL,
                      mass_point=mass_point,
                      sub_dir=sub_dir,
                      region=REGION,
                      histkey=histkey)
    return h_data

def get_bkg_hists(histkey, mass_point=DEFAULT_MASS_POINT):
    hists = dict()
    # devide histkey
    keys = histkey.split("/")
    sub_dir, histkey = keys[0], "/".join(keys[1:])
    # fake
    h_fake = get_hist(sample="fake",
                      channel=CHANNEL,
                      mass_point=mass_point,
                      sub_dir=sub_dir,
                      region=REGION,
                      histkey=histkey)
    h_fake_up = get_hist(sample="fake",
                         channel=CHANNEL,
                         mass_point=mass_point,
                         sub_dir=sub_dir,
                         region=REGION, 
                         histkey=histkey,
                         syst="Up")
    h_fake_down = get_hist(sample="fake",
                           channel=CHANNEL,
                           mass_point=mass_point,
                           sub_dir=sub_dir,
                           region=REGION,
                           histkey=histkey,
                           syst="Down")
    for bin in range(h_fake.GetNcells()+1):
        center = h_fake.GetBinContent(bin)
        up = h_fake_up.GetBinContent(bin) - center
        down = h_fake_down.GetBinContent(bin) - center
        error = np.sqrt(np.power(h_fake.GetBinError(bin), 2) + np.power(up, 2) + np.power(down, 2))
        h_fake.SetBinError(bin, error)
    hists['fake'] = h_fake
    del h_fake_up, h_fake_down

    # conversion
    h_conv = None
    h_DY = get_hist(sample="DY",
                    channel=CHANNEL,
                    mass_point=mass_point,
                    sub_dir=sub_dir,
                    region=REGION,
                    histkey=histkey)
    if h_DY != None:
        DY_sf, DY_err = get_conv_sf(CHANNEL, "DY")
        h_DY_center = h_DY.Clone("DY_center"); h_DY_center.Scale(DY_sf)
        h_DY_up = h_DY.Clone("DY_up"); h_DY_up.Scale(DY_sf+DY_err)
        h_DY_down = h_DY.Clone("DY_down"); h_DY_down.Scale(DY_sf-DY_err)
        h_DY.Scale(DY_sf)
        for bin in range(h_DY.GetNcells()+1):
            center = h_DY_center.GetBinContent(bin)
            up = h_DY_up.GetBinContent(bin) - center
            down = h_DY_down.GetBinContent(bin) - center
            error = np.sqrt(np.power(h_DY_center.GetBinError(bin), 2) + np.power(up, 2) + np.power(down, 2))
            h_DY.SetBinError(bin, error)
        h_conv = h_DY.Clone("conv")
        del h_DY, h_DY_center, h_DY_up, h_DY_down

    h_ZG = get_hist(sample="ZG",
                    channel=CHANNEL,
                    mass_point=mass_point,
                    sub_dir=sub_dir,
                    region=REGION,
                    histkey=histkey)
    if h_ZG != None:
        ZG_sf, ZG_err = get_conv_sf(CHANNEL, "ZG")
        h_ZG_center = h_ZG.Clone("DY_center"); h_ZG_center.Scale(ZG_sf)
        h_ZG_up = h_ZG.Clone("DY_up"); h_ZG_up.Scale(ZG_sf+ZG_err)
        h_ZG_down = h_ZG.Clone("DY_down"); h_ZG_down.Scale(ZG_sf-ZG_err)
        h_ZG.Scale(ZG_sf)
        for bin in range(h_ZG.GetNcells()+1):
            center = h_ZG_center.GetBinContent(bin)
            up = h_ZG_up.GetBinContent(bin) - center
            down = h_ZG_down.GetBinContent(bin) - center
            error = np.sqrt(np.power(h_ZG_center.GetBinError(bin), 2) + np.power(up, 2) + np.power(down, 2))
            h_ZG.SetBinError(bin, error)
    if h_conv != None and h_ZG != None:
        h_conv.Add(h_ZG)
    elif h_conv == None and h_ZG != None:
        h_conv = h_ZG.Clone("conv")
    else:
        h_conv = None
    hists['conv'] = h_conv

    # others
    hists['ttX'] = get_hist(sample='ttX',
                           channel=CHANNEL,
                           mass_point=mass_point,
                           sub_dir=sub_dir,
                           region=REGION,
                           histkey=histkey)
    hists['VV'] = get_hist(sample='VV',
                          channel=CHANNEL,
                          mass_point=mass_point,
                          sub_dir=sub_dir,
                          region=REGION,
                          histkey=histkey)
    hists['rare'] = get_hist(sample="rare",
                            channel=CHANNEL,
                            mass_point=mass_point,
                            sub_dir=sub_dir,
                            region=REGION,
                            histkey=histkey)
    # remove null histograms
    out = dict()
    for name, hist in hists.items():
        if hist == None:
            continue
        else:
            out[name] = hist
    del hists

    return out

if __name__ == "__main__":
    # input distributions
    if DISTRIBUTIONS == "inputs":
        base_dir = f"Outputs/plots/{REGION}/inputs"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        for obs in input_params[CHANNEL].keys():
            histkey = f"inputs/{obs}"
            hist_params = input_params[CHANNEL][obs]
            try:
                if REGION == "SR":
                    h_sigs = get_signal_hists(histkey, MHc=130)
                    h_bkgs = get_bkg_hists(histkey)
                    plotter = Kinematics(cvs_params, hist_params, info_params)
                    plotter.get_hists(h_sigs, h_bkgs)
                    plotter.combine()
                    plotter.save(f"{base_dir}/{obs.replace('/', '_')}.png")
                else:
                    h_data = get_data_hist(histkey)
                    h_bkgs = get_bkg_hists(histkey)
                    plotter = ObsAndExp(cvs_params, hist_params, info_params)
                    plotter.get_hists(h_data, h_bkgs)
                    plotter.combine()
                    plotter.save(f"{base_dir}/{obs.replace('/', '_')}.png")
            except Exception as e:
                print(f"Error occured! {REGION}-{obs}")
                print(e)
    elif DISTRIBUTIONS == "scores":
        base_dir = f"Outputs/plots/{REGION}/scores"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        for MA in MASS_POINTs[MHc]:
            mass_point = f"MHc{MHc}_MA{MA}"
            obs = "score_fake"
            hist_params = score_params[obs]
            try:
                if REGION == "SR":
                    h_sigs = dict()
                    h_sigs[mass_point] = get_hist(sample=f"TTToHcToWA_AToMuMu_{mass_point}",
                                         channel=CHANNEL,
                                         mass_point=mass_point,
                                         sub_dir=mass_point,
                                         region=REGION,
                                         histkey=obs)
                    h_bkgs = get_bkg_hists(f"{mass_point}/{obs}", mass_point)
                    plotter = Kinematics(cvs_params, hist_params, info_params)
                    plotter.get_hists(h_sigs, h_bkgs)
                    plotter.combine()
                    plotter.save(f"{base_dir}/{mass_point}-{obs.replace('/', '_')}.png")
                else:
                    h_data = get_data_hist(f"{mass_point}/{obs}", mass_point)
                    h_bkgs = get_bkg_hists(f"{mass_point}/{obs}", mass_point)
                    plotter = ObsAndExp(cvs_params, hist_params, info_params)
                    plotter.get_hists(h_data, h_bkgs)
                    plotter.combine()
                    plotter.save(f"{base_dir}/{mass_point}-{obs.replace('/', '_')}.png")
            except Exception as e:
                print(f"Error occurred! {REGION}-{obs}")
                print(e)

            obs = "score_ttX"
            hist_params = score_params[obs]
            #try:
            if REGION == "SR":
                h_sigs = dict()
                h_sigs[mass_point] = get_hist(sample=f"TTToHcToWA_AToMuMu_{mass_point}",
                                              channel=CHANNEL,
                                              mass_point=mass_point,
                                              sub_dir=mass_point,
                                              region=REGION,
                                              histkey=obs)
                h_bkgs = get_bkg_hists(f"{mass_point}/{obs}", mass_point)
                plotter = Kinematics(cvs_params, hist_params, info_params)
                plotter.get_hists(h_sigs, h_bkgs)
                plotter.combine()
                plotter.save(f"{base_dir}/{mass_point}-{obs.replace('/', '_')}.png")
            else:
                h_data = get_data_hist(f"{mass_point}/{obs}", mass_point)
                h_bkgs = get_bkg_hists(f"{mass_point}/{obs}", mass_point)
                plotter = ObsAndExp(cvs_params, hist_params, info_params)
                plotter.get_hists(h_data, h_bkgs)
                plotter.combine()
                plotter.save(f"{base_dir}/{mass_point}-{obs.replace('/', '_')}.png")
            #except Exception as e:
            #    print(f"Error occurred! {REGION}-{obs}")
            #    print(e)

