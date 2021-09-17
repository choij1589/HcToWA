import argparse
import numpy as np
from ROOT import TFile, TH1D

# Global variables
PROMPTs = ['rare', 'ttX', 'VV']
SYSTs = ["Central", "FakeUp", "FakeDown", "ElectronEnUp", "ElectronEnDown", 
         "ElectronResUp", "ElectronResDown", "MuonEnUp", "MuonEnDown", 
         "JetEnUp", "JetEnDown", "JetResUp", "JetResDown", 
         "L1PrefireUp", "L1PrefireDown", "PUReweightUp", "PUReweightDown"]

def get_hist(sample, histkey, channel, region, syst="Central"):
    base = "/root/workspace/HcToWA/Conversion"
    if sample == "fake":
        fkey = f"{base}/Outputs/{channel}/{region}/DATA.root"
    else:
        fkey = f"{base}/Outputs/{channel}/{region}/{sample}.root" 
    # print(fkey)
    f = TFile.Open(fkey)
    # print(f"{sample}/{syst}/{histkey}")
    h = f.Get(f"{sample}/{syst}/{histkey}")
    h.SetDirectory(0)
    f.Close()

    return h


def get_syst_sf(channel, region, syst="Central"):
    histkey = "ZCand/mass"
    SCALESYSTs = ["Central", "ElectronEnUp", "ElectronEnDown", "ElectronResUp", "ElectronResDown", 
             "MuonEnUp", "MuonEnDown", "JetEnUp", "JetEnDown", "JetResUp", "JetResDown"]

    # get data hist
    if syst in SCALESYSTs:
        h_data = get_hist("DATA", histkey, channel, region, syst)
    else:
        h_data = get_hist("DATA", histkey, channel, region)
    evts_data = h_data.Integral()

    evts_bkg = 0.

    # get fake hist
    if syst == "FakeUp":
        h_fake = get_hist("fake", histkey, channel, region, syst="Up")
    elif syst == "FakeDown":
        h_fake = get_hist("fake", histkey, channel, region, syst="Down")
    else:
        h_fake = get_hist("fake", histkey, channel, region)
    evts_bkg += h_fake.Integral()


    # get mc
    for prompt in PROMPTs:
        if syst in ["Central", "FakeUp", "FakeDown"]:
            h = get_hist(prompt, histkey, channel, region)
        else:
            h = get_hist(prompt, histkey, channel, region, syst)
        evts_bkg += h.Integral()

    if syst in ["Central", "FakeUp", "FakeDown"]:
        h_conv = get_hist(region, histkey, channel, region)
    else:
        h_conv = get_hist(region, histkey, channel, region, syst)

    sf = (evts_data - evts_bkg) / h_conv.Integral()

    return sf

def get_conv_sf(channel, region):
    sf_central = get_syst_sf(channel, region)
    total_err = 0.
    for syst in SYSTs:
        sf_syst = get_syst_sf(channel, region, syst)
        # print(f"{syst}: {sf_syst}")
        total_err += np.power(sf_central - sf_syst, 2)
    total_err = np.sqrt(total_err)
    # print(f"total err: {total_err}")
    return (sf_central, total_err)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", "-c", default=None, required=True, type=str, help="1E2Mu or 3Mu")
    parser.add_argument("--region", "-r", default=None, required=True, type=str, help="DY or ZG")
    args = parser.parse_args()
    region = args.region
    channel = args.channel
    print(get_conv_sf(channel, region))
