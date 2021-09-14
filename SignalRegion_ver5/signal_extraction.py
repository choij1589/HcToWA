import argparse
from itertools import product
import multiprocessing as mp
import numpy as np
import pandas as pd
import ctypes as c
from ROOT import TFile
from Conversion.measConvSF import get_conv_sf

parser = argparse.ArgumentParser()
parser.add_argument("--channel",
                    "-c",
                    default=None,
                    required=True,
                    type=str,
                    help="channel")
args = parser.parse_args()
CHANNEL = args.channel
MASS_POINTs = [
    "MHc70_MA15", "MHc70_MA40", "MHc70_MA65", "MHc100_MA15", "MHc100_MA25",
    "MHc100_MA60", "MHc100_MA95", "MHc130_MA15", "MHc130_MA45", "MHc130_MA55",
    "MHc130_MA90", "MHc130_MA125", "MHc160_MA15", "MHc160_MA45", "MHc160_MA75",
    "MHc160_MA85", "MHc160_MA120", "MHc160_MA155"
]


def get_hist(sample,
             channel,
             mass_point,
             sub_dir,
             region,
             histkey,
             syst="Central"):
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
        raise (KeyError)
    f.Close()

    return h


def extract_signal_1d(mass_point):
    signal = f"TTToHcToWA_AToMuMu_{mass_point}"
    sub_dir = mass_point
    histkey = "mMM"
    prompts = ['rare', 'VV', 'ttX']

    bkgs = dict()

    h_sig = get_hist(signal,
                     CHANNEL,
                     mass_point,
                     sub_dir,
                     region="SR",
                     histkey=histkey)
    """fake"""
    h_fake = get_hist("fake",
                      CHANNEL,
                      mass_point,
                      sub_dir,
                      region='SR',
                      histkey=histkey)
    h_fake_up = get_hist("fake",
                         CHANNEL,
                         mass_point,
                         sub_dir,
                         region="SR",
                         histkey=histkey,
                         syst="Up")
    h_fake_down = get_hist("fake",
                           CHANNEL,
                           mass_point,
                           sub_dir,
                           region="SR",
                           histkey=histkey,
                           syst="Down")

    for bin in range(h_fake.GetNbinsX() + 1):
        center = h_fake.GetBinContent(bin)
        up = h_fake_up.GetBinContent(bin) - center
        down = center - h_fake_down.GetBinContent(bin)
        error = np.sqrt(
            np.power(h_fake.GetBinError(bin), 2) + np.power(up, 2) +
            np.power(down, 2))
        h_fake.SetBinError(bin, error)
    bkgs['fake'] = h_fake
    """conversion"""
    h_DY = get_hist("DY",
                    CHANNEL,
                    mass_point,
                    sub_dir,
                    region="SR",
                    histkey=histkey)
    DY_sf, DY_err = get_conv_sf(CHANNEL, "DY")
    h_DY_center = h_DY.Clone("DY_center")
    h_DY_center.Scale(DY_sf)
    h_DY_up = h_DY.Clone("DY_up")
    h_DY_up.Scale(DY_sf + DY_err)
    h_DY_down = h_DY.Clone("DY_down")
    h_DY_down.Scale(DY_sf - DY_err)
    h_DY.Scale(DY_sf)
    for bin in range(h_DY.GetNbinsX() + 1):
        center = h_DY_center.GetBinContent(bin)
        up = h_DY_up.GetBinContent(bin) - center
        down = center - h_DY_down.GetBinContent(bin)
        error = np.sqrt(
            np.power(h_DY_center.GetBinError(bin), 2) + np.power(up, 2) +
            np.power(down, 2))
        h_DY.SetBinError(bin, error)
    h_conv = h_DY.Clone("conv")

    h_ZG = get_hist("ZG",
                    CHANNEL,
                    mass_point,
                    sub_dir,
                    region="SR",
                    histkey=histkey)
    ZG_sf, ZG_err = get_conv_sf(CHANNEL, "ZG")
    h_ZG_center = h_ZG.Clone("ZG_center")
    h_ZG_center.Scale(ZG_sf)
    h_ZG_up = h_ZG.Clone("ZG_up")
    h_ZG_up.Scale(ZG_sf + ZG_err)
    h_ZG_down = h_ZG.Clone("ZG_down")
    h_ZG_down.Scale(ZG_sf - ZG_err)
    h_ZG.Scale(ZG_sf)
    for bin in range(h_ZG.GetNbinsX() + 1):
        center = h_ZG_center.GetBinContent(bin)
        up = h_ZG_up.GetBinContent(bin) - center
        down = center - h_ZG_down.GetBinContent(bin)
        error = np.sqrt(
            np.power(h_ZG_center.GetBinError(bin), 2) + np.power(up, 2) +
            np.power(down, 2))
        h_ZG.SetBinError(bin, error)
    h_conv.Add(h_ZG)
    bkgs['conv'] = h_conv

    # prompts
    for mc in prompts:
        bkgs[mc] = get_hist(mc,
                            CHANNEL,
                            mass_point,
                            sub_dir,
                            region="SR",
                            histkey=histkey)

    # now estimate
    mA = float(mass_point.split("_")[1][2:])
    significance = 0.
    window = 0.
    for delta in np.linspace(0., 5., 50):
        delta = round(delta, 1)
        N_sig, N_bkg = 0., 0.
        N_sig = h_sig.Integral(h_sig.FindBin(mA - delta),
                               h_sig.FindBin(mA + delta))
        for name, hist in bkgs.items():
            error = c.c_double(0.)
            bkg = hist.IntegralAndError(hist.FindBin(mA - delta),
                                        hist.FindBin(mA + delta), error, "")
            #print(f"{name}: {error.value}")
            N_bkg += bkg + np.power(error.value, 2)
        this_sign = N_sig / np.sqrt(N_bkg)
        if this_sign > significance:
            significance, window = this_sign, delta
    # print(f"Maximum significance {significance} with window {window}")

    results = dict()
    results['index'] = mass_point
    results['window'] = window
    results['significance'] = significance

    return results


def extract_signal_3d(mass_point):
    signal = f"TTToHcToWA_AToMuMu_{mass_point}"
    sub_dir = mass_point
    histkey = "fake_ttX_mMM"
    prompts = ['rare', 'VV', 'ttX']

    bkgs = dict()

    h_sig = get_hist(signal,
                     CHANNEL,
                     mass_point,
                     sub_dir,
                     region="SR",
                     histkey=histkey)
    """fake"""
    h_fake = get_hist("fake",
                      CHANNEL,
                      mass_point,
                      sub_dir,
                      region='SR',
                      histkey=histkey)
    h_fake_up = get_hist("fake",
                         CHANNEL,
                         mass_point,
                         sub_dir,
                         region="SR",
                         histkey=histkey,
                         syst="Up")
    h_fake_down = get_hist("fake",
                           CHANNEL,
                           mass_point,
                           sub_dir,
                           region="SR",
                           histkey=histkey,
                           syst="Down")

    for bin in range(h_fake.GetNbinsX() + 1):
        center = h_fake.GetBinContent(bin)
        up = h_fake_up.GetBinContent(bin) - center
        down = center - h_fake_down.GetBinContent(bin)
        error = np.sqrt(
            np.power(h_fake.GetBinError(bin), 2) + np.power(up, 2) +
            np.power(down, 2))
        h_fake.SetBinError(bin, error)
    bkgs['fake'] = h_fake
    """conversion"""
    h_DY = get_hist("DY",
                    CHANNEL,
                    mass_point,
                    sub_dir,
                    region="SR",
                    histkey=histkey)
    DY_sf, DY_err = get_conv_sf(CHANNEL, "DY")
    h_DY_center = h_DY.Clone("DY_center")
    h_DY_center.Scale(DY_sf)
    h_DY_up = h_DY.Clone("DY_up")
    h_DY_up.Scale(DY_sf + DY_err)
    h_DY_down = h_DY.Clone("DY_down")
    h_DY_down.Scale(DY_sf - DY_err)
    h_DY.Scale(DY_sf)
    for bin in range(h_DY.GetNbinsX() + 1):
        center = h_DY_center.GetBinContent(bin)
        up = h_DY_up.GetBinContent(bin) - center
        down = center - h_DY_down.GetBinContent(bin)
        error = np.sqrt(
            np.power(h_DY_center.GetBinError(bin), 2) + np.power(up, 2) +
            np.power(down, 2))
        h_DY.SetBinError(bin, error)
    h_conv = h_DY.Clone("conv")

    h_ZG = get_hist("ZG",
                    CHANNEL,
                    mass_point,
                    sub_dir,
                    region="SR",
                    histkey=histkey)
    ZG_sf, ZG_err = get_conv_sf(CHANNEL, "ZG")
    h_ZG_center = h_ZG.Clone("ZG_center")
    h_ZG_center.Scale(ZG_sf)
    h_ZG_up = h_ZG.Clone("ZG_up")
    h_ZG_up.Scale(ZG_sf + ZG_err)
    h_ZG_down = h_ZG.Clone("ZG_down")
    h_ZG_down.Scale(ZG_sf - ZG_err)
    h_ZG.Scale(ZG_sf)
    for bin in range(h_ZG.GetNbinsX() + 1):
        center = h_ZG_center.GetBinContent(bin)
        up = h_ZG_up.GetBinContent(bin) - center
        down = center - h_ZG_down.GetBinContent(bin)
        error = np.sqrt(
            np.power(h_ZG_center.GetBinError(bin), 2) + np.power(up, 2) +
            np.power(down, 2))
        h_ZG.SetBinError(bin, error)
    h_conv.Add(h_ZG)
    bkgs['conv'] = h_conv

    # prompts
    for mc in prompts:
        bkgs[mc] = get_hist(mc,
                            CHANNEL,
                            mass_point,
                            sub_dir,
                            region="SR",
                            histkey=histkey)

    mA = float(mass_point.split("_")[1][2:])
    significance = 0.
    fake_cut = 0.
    ttX_cut = 0.
    window = 0.
    f_space = np.linspace(0., 1., 100)
    t_space = np.linspace(0., 1., 100)
    m_space = np.linspace(0., 5., 50)

    for f_step, t_step, delta in product(f_space, t_space, m_space):
        f_step, t_step = round(f_step, 2), round(t_step, 2)
        delta = round(delta, 1)
        x_min, x_max = h_sig.GetXaxis().FindBin(
            f_step), h_sig.GetXaxis().FindBin(1.)
        y_min, y_max = h_sig.GetYaxis().FindBin(
            t_step), h_sig.GetYaxis().FindBin(1.)
        z_min, z_max = h_sig.GetZaxis().FindBin(
            mA - delta), h_sig.GetZaxis().FindBin(mA + delta)

        N_sig, N_bkg = 0., 0.
        N_sig = h_sig.Integral(x_min, x_max, y_min, y_max, z_min, z_max)
        for name, hist in bkgs.items():
            error = c.c_double(0.)
            bkg = hist.IntegralAndError(x_min, x_max, y_min, y_max, z_min,
                                        z_max, error, "")
            #print(f"{name}: {error.value}")
            N_bkg += bkg + np.power(error.value, 2)
        if N_bkg == 0.:
            continue
        this_sign = N_sig / np.sqrt(N_bkg)
        if this_sign > significance:
            significance, window = this_sign, delta
            fake_cut, ttX_cut = f_step, t_step

        if delta == 5. and t_step == 1.:
            print(
                f"[{mass_point}] current best significance with f_step {f_step}: {significance} with f_cut {fake_cut}, t_cut {ttX_cut}, window {window}"
            )

    print(
        f"Maximum significance {significance} with window {window}, fake cut {fake_cut}, ttX cut {ttX_cut}"
    )

    results = dict()
    results['index'] = mass_point
    results['fake_cut'] = fake_cut
    results['ttX_cut'] = ttX_cut
    results['window'] = window
    results['significance'] = significance

    return results


# callbacks
def collect_results_1d(results):
    global mass_points
    global windows
    global significances

    mass_points.append(results['index'])
    windows.append(results['window'])
    significances.append(results['significance'])


def collect_results_3d(results):
    global mass_points
    global fake_cuts
    global ttX_cuts
    global windows
    global significances

    mass_points.append(results['index'])
    fake_cuts.append(results['fake_cut'])
    ttX_cuts.append(results['ttX_cut'])
    windows.append(results['window'])
    significances.append(results['significance'])


if __name__ == "__main__":
    #extract_signal_1d("MHc70_MA15")
    mass_points = list()
    windows = list()
    significances = list()

    mp.set_start_method("spawn")
    pool = mp.Pool(processes=4)
    for mass_point in MASS_POINTs:
        pool.apply_async(extract_signal_1d, (mass_point, ),
                         callback=collect_results_1d)
    pool.close()
    pool.join()

    collections = {
        "mass_point": mass_points,
        "window": windows,
        "significance": significances
    }
    # print(collections)
    df = pd.DataFrame(collections)
    df.to_csv(f"Outputs/{CHANNEL}/CSV/significance_1d.csv")
    print(f"INFO:: Saving CSV for 1D signal extraction")

    # 3d
    extract_signal_3d("MHc70_MA15")
    #mass_points = list()
    #fake_cuts = list()
    #ttX_cuts = list()
    #windows = list()
    #significances = list()
    #pool = mp.Pool(processes=4)
    #for mass_point in MASS_POINTs:
    #    pool.apply_async(extract_signal_3d, (mass_point, ),
    #                     callback=collect_results_3d)
    #pool.close()
    #pool.join()

    #collections = {
    #    "mass_point": mass_points,
    #    "fake_cut": fake_cuts,
    #    "ttX_cut": ttX_cuts,
    #    "window": windows,
    #    "significance": significances
    #}
    #df = pd.DataFrame(collections)
    #df.to_csv(f"Outputs/{CHANNEL}/CSV/significance_3d.csv")
    print(f"INFO:: Saving CSV for 3D signal extraction")
    print("END PROCESS")
