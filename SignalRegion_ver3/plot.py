import numpy as np

from ROOT import TFile, TH1D
from Plotter.PlotterTools.Kinematics import Kinematics
from Plotter.PlotterTools.ObsAndExp import ObsAndExp
from Plotter.Parameters.params_trilep import cvs_params, info_params
from Plotter.Parameters.params_trilep import muon_params, electron_params, jet_params, dimuon_params, METv_params
from Plotter.Parameters.params_trilep import input_params, score_params
from Conversion.measConvSF import get_conv_sf

# Global Variables
mass_point = "MHc160_MA155"
signal = f"TTToHcToWA_AToMuMu_{mass_point}"
prompts = ['rare', 'ttX', 'VV']


def get_hist(sample, channel, mass_point, region, histkey, syst="Central"):
    if sample == "fake":
        fkey = f"Outputs/{channel}/{mass_point}/DATA.root"
    else:
        fkey = f"Outputs/{channel}/{mass_point}/{sample}.root"
    histkey = f"{sample}/{region}/{syst}/{histkey}"

    f = TFile.Open(fkey)
    h = f.Get(histkey)
    if type(h) == TH1D:
        h.SetDirectory(0)
    else:
        return None
    f.Close()

    return h


def get_hists(histkey, channel, region):
    signals = dict()
    hists = dict()
    _hists = dict()

    # signal sample
    h_sig = get_hist(signal, channel, mass_point, region, histkey=histkey)
    signals[mass_point] = h_sig

    # data
    h_data = get_hist("DATA", channel, mass_point, region, histkey=histkey)

    # fake
    h_fake = get_hist("fake", channel, mass_point, region, histkey=histkey)
    h_fake_up = get_hist("fake", channel, mass_point, region, histkey=histkey, syst='Up')
    h_fake_down = get_hist("fake", channel, mass_point, region, histkey=histkey, syst='Down')

    for bin in range(h_fake.GetNbinsX()+1):
        center = h_fake.GetBinContent(bin)
        up, down = h_fake_up.GetBinContent(bin) - center, center - h_fake_down.GetBinContent(bin)
        error = np.sqrt(np.power(h_fake.GetBinError(bin), 2) + np.power(up, 2) + np.power(down, 2))
        h_fake.SetBinError(bin, error)
    _hists['fake'] = h_fake

    # conversion
    h_DY = get_hist("DY", channel, mass_point, region, histkey=histkey)
    DY_sf, DY_err = get_conv_sf(channel, 'DY')
    try:
        h_DY_center = h_DY.Clone("DY_center"); h_DY_center.Scale(DY_sf)
        h_DY_up = h_DY.Clone("DY_up"); h_DY_up.Scale(DY_sf + DY_err)
        h_DY_down = h_DY.Clone("DY_down"); h_DY_down.Scale(DY_sf - DY_err)
        h_DY.Scale(DY_sf)
        for bin in range(h_DY.GetNbinsX()+1):
            center = h_DY.GetBinContent(bin)
            up, down = h_DY_down.GetBinContent(bin) - center, center - h_DY_down.GetBinContent(bin)
            error = np.sqrt(np.power(h_DY_center.GetBinError(bin), 2) + np.power(up, 2) + np.power(down, 2))
            h_DY.SetBinError(bin, error)
    except:
        pass

    h_ZG = get_hist("ZG", channel, mass_point, region, histkey=histkey)
    ZG_sf, ZG_err = get_conv_sf(channel, 'ZG')
    try:
        h_ZG_center = h_ZG.Clone("ZG_center"); h_ZG_center.Scale(ZG_sf)
        h_ZG_up = h_ZG.Clone("ZG_up"); h_ZG_up.Scale(ZG_sf + ZG_err)
        h_ZG_down = h_ZG.Clone("ZG_down"); h_ZG_down.Scale(ZG_sf - ZG_err)
        h_ZG.Scale(ZG_sf)
        for bin in range(h_ZG.GetNbinsX()+1):
            center = h_ZG.GetBinContent(bin)
            up, down = h_ZG_down.GetBinContent(bin) - center, center - h_ZG_down.GetBinContent(bin)
            error = np.sqrt(np.power(h_DY_center.GetBinError(bin), 2) + np.power(up, 2) + np.power(down, 2))
            h_ZG.SetBinError(bin, error)
    except:
        pass

    if type(h_DY) == TH1D and type(h_ZG) == TH1D:
        h_conv = h_DY.Clone("conv")
        h_conv.Add(h_ZG)
    elif type(h_DY) == TH1D and type(h_ZG) != TH1D:
        h_conv = h_DY.Clone("conv")
    elif type(h_ZG) == TH1D and type(h_DY) != TH1D:
        h_conv = h_ZG.Clone("conv")
    else:
        h_conv = None
    _hists['conv'] = h_conv

    for mc in prompts:
        _hists[mc] = get_hist(mc, channel, mass_point, region, histkey=histkey)
    
    # check whether all MCs are contained
    for name, hist in _hists.items():
        if type(hist) == TH1D:
            hists[name] = hist
        else:
            continue

    return (signals, h_data, hists)

# loop over variables
for obs in muon_params.keys():
    histkey = obs
    hist_params = muon_params[obs]
    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='SR')
        plotter = Kinematics(cvs_params, hist_params, info_params)
        plotter.get_hists(signals, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/SR/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! SR-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='ZFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/ZFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! ZFake-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='TTFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/TTFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! TTFake-{obs}")
        print(e)

for obs in electron_params.keys():
    histkey = obs
    hist_params = electron_params[obs]
    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='SR')
        plotter = Kinematics(cvs_params, hist_params, info_params)
        plotter.get_hists(signals, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/SR/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! SR-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='ZFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/ZFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! ZFake-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='TTFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/TTFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! TTFake-{obs}")
        print(e)
        
for obs in jet_params.keys():
    histkey = obs
    hist_params = jet_params[obs]
    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='SR')
        plotter = Kinematics(cvs_params, hist_params, info_params)
        plotter.get_hists(signals, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/SR/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! SR-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='ZFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/ZFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! ZFake-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='TTFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/TTFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! TTFake-{obs}")
        print(e)

for obs in dimuon_params.keys():
    histkey = obs
    hist_params = dimuon_params[obs]
    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='SR')
        plotter = Kinematics(cvs_params, hist_params, info_params)
        plotter.get_hists(signals, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/SR/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! SR-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='ZFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/ZFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! ZFake-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='TTFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/TTFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! TTFake-{obs}")
        print(e)

for obs in METv_params.keys():
    histkey = obs
    hist_params = METv_params[obs]
    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='SR')
        plotter = Kinematics(cvs_params, hist_params, info_params)
        plotter.get_hists(signals, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/SR/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! SR-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='ZFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/ZFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! ZFake-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='TTFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/TTFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurre! TTFake-{obs}")
        print(e)

for obs in input_params.keys():
    histkey = obs
    hist_params = input_params[obs]
    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='SR')
        plotter = Kinematics(cvs_params, hist_params, info_params)
        plotter.get_hists(signals, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/SR/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! SR-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='ZFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/ZFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! ZFake-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='TTFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/TTFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurre! TTFake-{obs}")
        print(e)

for obs in score_params.keys():
    histkey = obs
    hist_params = score_params[obs]
    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='SR')
        plotter = Kinematics(cvs_params, hist_params, info_params)
        plotter.get_hists(signals, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/SR/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! SR-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='ZFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/ZFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! ZFake-{obs}")
        print(e)

    try:
        signals, h_data, hists = get_hists(histkey, channel='1E2Mu', region='TTFake')
        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(h_data, hists)
        plotter.combine()
        plotter.save(f"Outputs/1E2Mu/{mass_point}/TTFake/{histkey.replace('/', '_')}.png")
    except Exception as e:
        print(f"Error occurred! TTFake-{obs}")
        print(e)

