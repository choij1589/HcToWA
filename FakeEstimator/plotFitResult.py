import os, shutil
import argparse
from ROOT import TFile
from Plotter.PlotterTools.ObsAndExp import ObsAndExp
from fit import get_coefs

parser = argparse.ArgumentParser()
parser.add_argument("--object", "-o", default=None, required=True, type=str, help="electron or muon")
args = parser.parse_args()

cvs_params = {
    "logy": False,
    "grid": False
}
info_params = {
    "info": "Prescaled",
    "cms_text": "CMS",
    "extra_text": "Work in progress"
}
hist_params = {
    "x_title": "p_{T}^{corr}",
    "x_range": [0, 200],
    "y_title": "Events",
    "rebin": 5,
    "ratio_range": [0., 2.0]
}

def get_hist(sample, histkey, syst):
    if "Ele" in histkey:
        f = TFile.Open(f"Outputs/electron/SkimSglEle__WP90__/SglEle_{sample}.root")
    elif "Mu" in histkey:
        f = TFile.Open(f"Outputs/muon/SglMu_{sample}.root")
    keys = histkey.split("/")
    histkey = f"{keys[0]}/{keys[1]}/{syst}/{keys[2]}/{keys[3]}"
    print(histkey)
    h = f.Get(histkey)
    h.SetDirectory(0)
    f.Close()
    return h

#SYSTs = ["Central", "FlavorDep", "JetPtCutUp", "JetPtCutDown", "JetEnUp", "JetEnDown", "JetResUp", "JetResDown", "ElectronEnUp", "ElectronEnDown", "ElectronResUp", "ElectronResDown", "MuonEnUp", "MuonEnDown"]
SYSTs = ["Central"]

if args.object == "electron":
    DATASTREAM = "SingleElectron"
    MCs = ["QCD_EMEnriched", "QCD_bcToE", "W", "DY", "TT", "ST", "VV"]
    paths = ["loose/passEle8Path/eta0to0p8", 
             "loose/passEle8Path/eta0p8to1p479", 
             "loose/passEle8Path/eta1p479to2p5", 
             "loose/passEle12Path/eta0to0p8", 
             "loose/passEle12Path/eta0p8to1p479", 
             "loose/passEle12Path/eta1p479to2p5", 
             "loose/passEle12Path/eta0to0p8", 
             "loose/passEle23Path/eta0p8to1p479", 
             "loose/passEle23Path/eta1p479to2p5",
             "tight/passEle8Path/eta0to0p8",
             "tight/passEle8Path/eta0p8to1p479",
             "tight/passEle8Path/eta1p479to2p5",
             "tight/passEle12Path/eta0to0p8",
             "tight/passEle12Path/eta0p8to1p479",
             "tight/passEle12Path/eta1p479to2p5",
             "tight/passEle12Path/eta0to0p8",
             "tight/passEle23Path/eta0p8to1p479",
             "tight/passEle23Path/eta1p479to2p5"]

if args.object == "muon":
    DATASTREAM = "DoubleMuon"
    MCs = ["QCD_MuEnriched", "W", "DY", "TT", "ST", "VV"]
    paths = ["loose/passMu8Path/eta0to0p9",
             "loose/passMu8Path/eta0p9to1p6",
             "loose/passMu8Path/eta1p6to2p4",
             "loose/passMu17Path/eta0to0p9",
             "loose/passMu17Path/eta0p9to1p6",
             "loose/passMu17Path/eta1p6to2p4",
             "tight/passMu8Path/eta0to0p9",
             "tight/passMu8Path/eta0p9to1p6",
             "tight/passMu8Path/eta1p6to2p4",
             "tight/passMu17Path/eta0to0p9",
             "tight/passMu17Path/eta0p9to1p6",
             "tight/passMu17Path/eta1p6to2p4"]

if __name__ == "__main__":
    # set up directory
    dir_path = f"plots/{args.object}"
    #try:
    #    if os.listdir(dir_path):
    #        print(f"overwriting directory {dir_path}...")
    #        shutil.rmtree(dir_path)
    #except:
    #    pass
    #os.makedirs(dir_path)

    # prefit
    for path in paths:
        histkey = f"{path}/ptCorr"
        for syst in SYSTs:
            h_data = get_hist(DATASTREAM, histkey, syst)
            h_mcs = dict()

            for MC in MCs:
                hist = get_hist(MC, histkey, syst)
                h_mcs[MC] = hist

            plotter = ObsAndExp(cvs_params, hist_params, info_params)
            plotter.get_hists(h_data, h_mcs)
            plotter.combine()
            #path = path.replace("/", "_")
            plotter.save(f"plots/{args.object}/prefit_{path.replace('/', '_')}_ptCorr_{syst}.png")

    # postfit
    for path in paths:
        histkey = f"{path}/ptCorr"
        for syst in SYSTs:
            h_data = get_hist(DATASTREAM, histkey, syst)
            h_mcs = dict()
    
            keys = path.split("/")
            this_path = f"{keys[0]}/{keys[1]}/{syst}/{keys[2]}"
            scales = get_coefs(this_path, args.object)
            for MC in MCs:
                hist = get_hist(MC, histkey, syst)
                # scale = scales[MC]/sum(scales.values()) 
                # scale *= h_data.Integral() / hist.Integral()
                scale = (scales[MC]/sum(scales.values())) * (h_data.Integral() / hist.Integral())
                #print(f"{MC}: {scale}")
                hist.Scale(scale)
                h_mcs[MC] = hist

            plotter = ObsAndExp(cvs_params, hist_params, info_params)
            plotter.get_hists(h_data, h_mcs)
            plotter.combine()
            #path = path.replace("/", "_")
            plotter.save(f"plots/{args.object}/postfit_{path.replace('/', '_')}_ptCorr_{syst}.png")


