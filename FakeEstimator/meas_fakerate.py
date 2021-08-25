import array
from itertools import product
from Scripts.setup import setup
setup()
import numpy as np
from ROOT import TFile, TCanvas, TH1D
from fit import get_coefs
from time import sleep

def get_hist(sample, histkey):
    path = histkey.split("/")[1]
    if "Ele" in path:
        f = TFile.Open(f"Outputs/electron/SkimSglEle__WP90__/SglEle_{sample}.root")
    elif "Mu" in path:
        f = TFile.Open(f"Outputs/muon/SglMu_{sample}.root")
    else:
        print(f"[get_hist] Wrong histkey {histkey}")
        raise(AttributeError)
    # print(histkey) 
    h = f.Get(histkey)
    h.SetDirectory(0)
    f.Close()
    return h

def get_hists(object, path, observable, scaled=False):
    # get histograms
    if object == "electron":
        DATASTREAM = "SingleElectron"
        MCs = ["QCD_EMEnriched", "QCD_bcToE", "W", "DY", "TT", "ST", "VV"]
    elif object == "muon":
        DATASTREAM = "DoubleMuon"
        MCs = ["QCD_MuEnriched", "W", "DY", "TT", "ST", "VV"]
    else:
        print(f"[get_hists] object shoule be electron or muon")
        raise(AttributeError)
    
    # construct histkey
    # e.g.
    # path = "loose/passMu8Path/{syst}/{etabin}"
    # observable = "electrons/1/pt"
    histkey = f"{path}/{observable}"
    
    h_data = get_hist(DATASTREAM, histkey)
    h_mcs = dict()
    for mc in MCs:
        h_mcs[mc] = get_hist(mc, histkey)
        
    if scaled:
        # keys = path.split("/")
        # central_path = f"{keys[0]}/{keys[1]}/Central/{keys[3]}"
        # scales = get_coefs(central_path, object)
        scales = get_coefs(path, object)
        total = sum(scales.values())
        for mc in MCs:
            scale = scales[mc]/total
            scale *= h_data.Integral() / h_mcs[mc].Integral()
            h_mcs[mc].Scale(scale)
            
    return (h_data, h_mcs)
        

def write_fakerate(outpath, object, syst):
    outfile = TFile(outpath, "recreate")
    if object == "electron":
        DATASTREAM = "SingleElectron"
        paths = ['passEle8Path', 'passEle12Path', 'passEle23Path']
        etabins = ['eta0to0p8', 'eta0p8to1p479', 'eta1p479to2p5']
        xbins = array.array('d', [10, 15, 20, 35, 50, 70])
    elif object == "muon":
        DATASTREAM = "DoubleMuon"
        paths = ['passMu8Path', 'passMu17Path']
        etabins = ['eta0to0p9', 'eta0p9to1p6', 'eta1p6to2p4']
        xbins = array.array('d', [10, 15, 20, 25, 35, 50, 70])
    else:
        print(f"[get_fakerate] Wrong path {path}")
        raise(AttributeError)
    
    prompts = ["W", "DY", "TT", "ST", "VV"]
    observable = "ptCorr"
    # fakerates = dict()
    for (path, etabin) in product(paths, etabins):
        h_loose, h_mcs_loose = get_hists(object, f"loose/{path}/{syst}/{etabin}", observable, scaled=True)
        h_tight, h_mcs_tight = get_hists(object, f"tight/{path}/{syst}/{etabin}", observable, scaled=True)

        h_loose = h_loose.Rebin(len(xbins)-1, f"Rebin_{DATASTREAM}_{syst}_loose", xbins)
        h_tight = h_tight.Rebin(len(xbins)-1, f"Rebin_{DATASTREAM}_{syst}_tight", xbins)
        for prompt in prompts:
            h_mcs_loose[prompt] = h_mcs_loose[prompt].Rebin(len(xbins)-1, f"Rebin_{prompt}_{syst}_loose", xbins)
            h_mcs_tight[prompt] = h_mcs_tight[prompt].Rebin(len(xbins)-1, f"Rebin_{prompt}_{syst}_tight", xbins)
        
        for prompt in prompts:
            h_loose.Add(h_mcs_loose[prompt], -1.)
            h_tight.Add(h_mcs_tight[prompt], -1.)

        outfile.cd()
        
        fkey = f"{path}_{syst}_{etabin}"
        h_fakerate = h_tight.Clone(fkey)
        h_fakerate.Divide(h_loose)

        h_fakerate.Write()
        #outfile.Write()
    outfile.Close()

        
if __name__ == "__main__":
    # object = "electron"
    object = "muon"
    systs = ["Central", "FlavorDep", "JetPtCutUp", "JetPtCutDown", "JetEnUp", "JetEnDown", "JetResUp", "JetResDown", "ElectronEnUp", "ElectronEnDown", "ElectronResUp", "ElectronResDown", "MuonEnUp", "MuonEnDown"]
    #systs = ["ElectronEnUp", "ElectronEnDown", "ElectronResUp", "ElectronResDown", "MuonEnUp", "MuonEnDown"]
    for syst in systs:
        outpath = f"Outputs/{object}/fakerate_{syst}.root"
        write_fakerate(outpath, object, syst)
        #for hist in fakerates.values():
        #    hist.Write()

