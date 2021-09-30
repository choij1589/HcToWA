import argparse
from math import sqrt, log
import numpy as np
import pandas as pd
import ctypes as c
from itertools import product
from ROOT import TFile
from Conversion.measConvSF import get_conv_sf

# parse arguments, define global variables
parser = argparse.ArgumentParser()
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help="channel")
parser.add_argument("--mass_point", "-m", default=None, required=True, type=str, help="signal mass point")
args = parser.parse_args()

CHANNEL = args.channel
MASS_POINT = args.mass_point
mA = float(MASS_POINT.split("_")[1][2:])

"""helper function for histogram"""
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
        print(f"fkey: {fkey}")
        print(f"histkey: {histkey}")
        print(e)
        raise(KeyError)
    f.Close()

    return h

def get_hists(grid="1d"):
    """get histograms for each process"""
    hists = dict()
    if grid not in ["1d", "3d"]:
        print("grid should be 1d or 3d")
        raise(KeyError)
    histkey = "mMM" if grid == "1d" else "fake_ttX_mMM" # else == "3d"

    # signal
    h_sig = get_hist(sample=f"TTToHcToWA_AToMuMu_{MASS_POINT}", 
                     channel=CHANNEL, 
                     mass_point=MASS_POINT, 
                     sub_dir=MASS_POINT, 
                     region="SR",
                     histkey=histkey)
    hists['signal'] = h_sig

    # fake
    h_fake = get_hist(sample="fake",
                      channel=CHANNEL,
                      mass_point=MASS_POINT,
                      sub_dir=MASS_POINT,
                      region="SR",
                      histkey=histkey)
    h_fake_up = get_hist(sample="fake",
                         channel=CHANNEL,
                         mass_point=MASS_POINT,
                         sub_dir=MASS_POINT,
                         region="SR",
                         histkey=histkey,
                         syst="Up")
    h_fake_down = get_hist(sample="fake",
                           channel=CHANNEL,
                           mass_point=MASS_POINT,
                           sub_dir=MASS_POINT,
                           region="SR",
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
    h_DY = get_hist(sample="DY",
                    channel=CHANNEL,
                    mass_point=MASS_POINT,
                    sub_dir=MASS_POINT,
                    region="SR",
                    histkey=histkey)
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

    # conversion
    h_ZG = get_hist(sample="ZG",
                    channel=CHANNEL,
                    mass_point=MASS_POINT,
                    sub_dir=MASS_POINT,
                    region="SR",
                    histkey=histkey)
    ZG_sf, ZG_err = get_conv_sf(CHANNEL, "ZG")
    h_ZG_center = h_ZG.Clone("DY_center"); h_ZG_center.Scale(ZG_sf)
    h_ZG_up = h_ZG.Clone("DY_up"); h_ZG_up.Scale(ZG_sf+ZG_err)
    h_ZG_down = h_ZG.Clone("DY_down"); h_ZG_down.Scale(ZG_sf-ZG_err)
    h_ZG.Scale(ZG_sf)
    for bin in range(h_ZG.GetNbinsX()+1):
        center = h_ZG_center.GetBinContent(bin)
        up = h_ZG_up.GetBinContent(bin) - center
        down = h_ZG_down.GetBinContent(bin) - center
        error = np.sqrt(np.power(h_ZG_center.GetBinError(bin), 2) + np.power(up, 2) + np.power(down, 2))
        h_ZG.SetBinError(bin, error)
    h_conv.Add(h_ZG)
    del h_ZG, h_ZG_center, h_ZG_up, h_ZG_down
    hists['conv'] = h_conv
    
    # others
    hists['ttX'] = get_hist(sample='ttX',
                           channel=CHANNEL,
                           mass_point=MASS_POINT,
                           sub_dir=MASS_POINT,
                           region="SR",
                           histkey=histkey)
    hists['VV'] = get_hist(sample='VV',
                          channel=CHANNEL,
                          mass_point=MASS_POINT,
                          sub_dir=MASS_POINT,
                          region="SR",
                          histkey=histkey)
    hists['rare'] = get_hist(sample="rare",
                            channel=CHANNEL,
                            mass_point=MASS_POINT,
                            sub_dir=MASS_POINT,
                            region="SR",
                            histkey=histkey)
    
    return hists

# function wrappers and callbacks
def get_number_of_evts(name, hist, cuts):
    """get the list of cuts and perform integrals"""
    # 1d grid search using mass window
    if len(cuts) == 1:
        window = round(cuts[0], 1)
        dN_evt = c.c_double(0.)
        N_evt = hist.IntegralAndError(hist.FindBin(mA-window), hist.FindBin(mA+window), dN_evt, "")
    elif len(cuts) == 3:
        f_step, t_step, window = tuple(cuts)
        f_step, t_step = round(f_step, 2), round(t_step, 2)
        window = round(window, 1)
        x_min, x_max = hist.GetXaxis().FindBin(f_step), hist.GetXaxis().FindBin(1.)
        y_min, y_max = hist.GetYaxis().FindBin(t_step), hist.GetYaxis().FindBin(1.)
        z_min, z_max = hist.GetZaxis().FindBin(mA-window), hist.GetZaxis().FindBin(mA+window)
        dN_evt = c.c_double(0.)
        N_evt = hist.IntegralAndError(x_min, x_max, y_min, y_max, z_min, z_max, dN_evt, "")
    else:
        print("Wrong number of cuts")
        raise(KeyError)
    
    results = {name: (N_evt, dN_evt.value)}
    #print(f"process done for {name}")
    return results

def update_results(results):
    global Evt_dict
    name = list(results.keys())[0]
    N_evt, dN_evt = results[name]
    Evt_dict[name] = (N_evt, dN_evt)


if __name__ == "__main__":
    bkgs = ['fake', 'ttX', 'conv', 'VV', 'rare']
    # perform 1d grid search first
    hists = get_hists(grid="1d")
    windows = np.linspace(0.1, 5., 50) # 0.1 GeV step
    final_window = 0.
    final_metric = 0.
    for window in windows:
        Evt_dict = dict()
        for name, hist in hists.items():
            results = get_number_of_evts(name, hist, [window])
            update_results(results)

        metrics = []
        for _ in range(200):
            N_sig = np.random.normal(Evt_dict['signal'][0], Evt_dict['signal'][1])
            N_bkg = 0.
            for bkg in bkgs:
                N_bkg += np.random.normal(Evt_dict[bkg][0], Evt_dict[bkg][1])
            try:
                this_metric = sqrt(2*((N_sig+N_bkg)*log(1+N_sig/N_bkg) - N_sig))
                metrics.append(this_metric)
            except:
                continue
        metric = np.mean(metrics)

        #N_sig, dN_sig = Evt_dict['signal']
        #N_bkg_total, dN_bkg_total = 0., 0.
        #for bkg in bkgs:
        #    N_bkg, dN_bkg = Evt_dict[bkg]
        #    N_bkg_total += N_bkg
        #    dN_bkg_total += dN_bkg


        #try:
        #    # metric = N_sig / np.sqrt(N_bkg_total + np.power(dN_bkg_total, 2))
        #    metric = sqrt(2*((N_sig+N_bkg)*log(1+N_sig/N_bkg) - N_sig))
        #except:
        #    metric = 0.

        if metric > final_metric:
            final_metric = metric
            final_window = window
    print(f"[1d grid search] current best cut: window = {final_window} with metric {final_metric}")
    
    # save results
    Evt_dict = dict()
    for name, hist in hists.items():
        results = get_number_of_evts(name, hist, [final_window])
        update_results(results)
    Evt_dict["window"] = (final_window, 0)
    Evt_dict["metric"] = (final_metric, 0)
    df = pd.DataFrame(Evt_dict)
    df.to_csv(f"Outputs/{CHANNEL}/CSV/Grid_{MASS_POINT}_exact_1D.csv")
    del hists, Evt_dict

    hists = get_hists(grid="3d")
    
    # first generation
    f_space = np.linspace(0., 0.99, 100)	# 0.01 step
    t_space = np.linspace(0., 0.99, 100)	# 0.01 step
    m_space = np.linspace(0.1, 5., 50)	# 0.1 GeV step
    final_f_cut, final_t_cut, final_m_cut = 0., 0., 0.
    final_metric = 0.
    for idx, (f_step, t_step, m_step) in enumerate(product(f_space, t_space, m_space)):
        cuts = [f_step, t_step, m_step]
        Evt_dict = dict()
        for name, hist in hists.items():
            results = get_number_of_evts(name, hist, cuts)
            update_results(results)

        metrics = []
        for _ in range(200):
            N_sig = np.random.normal(Evt_dict['signal'][0], Evt_dict['signal'][1])
            N_bkg = 0.
            for bkg in bkgs:
                N_bkg += np.random.normal(Evt_dict[bkg][0], Evt_dict[bkg][1])
            try:
                this_metric = sqrt(2*((N_sig+N_bkg)*log(1+N_sig/N_bkg) - N_sig))
                metrics.append(this_metric)
            except:
                continue
        metric = np.mean(metrics)

        #N_sig, dN_sig = Evt_dict['signal']
        #N_bkg_total, dN_bkg_total = 0., 0.
        #for bkg in bkgs:
        #    N_bkg, dN_bkg = Evt_dict[bkg]
        #    N_bkg_total += N_bkg
        #    dN_bkg_total += dN_bkg
        #try:
        #    # metric = N_sig / np.sqrt(N_bkg_total + np.power(dN_bkg_total, 2))
        #    metric = sqrt(2*((N_sig+N_bkg)*log(1+N_sig/N_bkg) - N_sig))
        #except:
        #    metric = 0.	# no bkg?

        if metric > final_metric:
            final_metric = metric
            final_f_cut, final_t_cut, final_m_cut = f_step, t_step, m_step

        if idx % 10000 == 0:
            print(f"[3D grid search] progress: {idx/10000}/50")
            print(f"[3D grid search] current best cut: (f_cut, t_cut, m_cut) = ({final_f_cut}, {final_t_cut}, {final_m_cut} with metric {final_metric}")
    print(f"[3D grid search] final best cut: (f_cut, t_cut, m_cut) = ({final_f_cut}, {final_t_cut}, {final_m_cut}) with metric {final_metric}")

    # final iteration
    cuts = [final_f_cut, final_t_cut, final_m_cut]
    # pool = mp.Pool(processes=5)
    Evt_dict = dict()
    for name, hist in hists.items():
        results = get_number_of_evts(name, hist, cuts)
        update_results(results)
        # pool.apply_async(get_number_of_evts, (name, hist, cuts), callback=update_results)
    # pool.close()
    # pool.join()
    Evt_dict["f_cut"] = (final_f_cut, 0)
    Evt_dict["t_cut"] = (final_t_cut, 0)
    Evt_dict["m_cut"] = (final_m_cut, 0)
    Evt_dict["metric"] = (final_metric, 0)
    df = pd.DataFrame(Evt_dict)
    df.to_csv(f"Outputs/{CHANNEL}/CSV/Grid_{MASS_POINT}_exact_3D.csv")
