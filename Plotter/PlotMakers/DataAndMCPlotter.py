from ROOT import TFile, TH1D
from PlotterTools.ObsAndExp import ObsAndExp
from Parameters.obs_and_exp import param_set, cvs_params, info_params

# Get histograms
hist_obs = None
hists_exp = {}

observables = {
    "DY_dimu": {
        "leptons": ["ptmu1", "ptmu2", "etamu1", "etamu2", "phimu1", "phimu2"],
        "jets": ["ptj1", "ptj2", "ptj3", "etaj1", "etaj2", "etaj3", "phij1", "phij2", "phij3", "nJets"],
        "METv": ["METv", "etaMETv", "phiMETv"],
        "bjets": [],
        "ZCand": ["ptZ", "etaZ", "phiZ", "Zmass"]
    },
    "TT_dimu": {
        "leptons": ["ptmu1", "ptmu2", "etamu1", "etamu2", "phimu1", "phimu2"],
        "jets": ["ptj1", "ptj2", "ptj3", "etaj1", "etaj2", "etaj3", "phij1", "phij2", "phij3", "nJets"],
        "METv": ["METv", "etaMETv", "phiMETv"],
        "bjets": ["ptb1", "ptb2", "ptb3", "etab1", "etab2", "etab3", "phib1", "phib2", "phib3", "nBJets"],
        "ZCand": ["ptZ", "etaZ", "phiZ", "Zmass"]
    },
    "TT_emu": {
        "leptons": ["ptmu1", "pte1", "etamu1", "etae1", "phimu1", "phie1"],
        "jets": ["ptj1", "ptj2", "ptj3", "etaj1", "etaj2", "etaj3", "phij1", "phij2", "phij3", "nJets"],
        "METv": ["METv", "etaMETv", "phiMETv"],
        "bjets": ["ptb1", "ptb2", "ptb3", "etab1", "etab2", "etab3", "phib1", "phib2", "phib3", "nBJets"],
        "ZCand": []
    },
    "Pre_3mu": {
        "leptons": ["ptmu1", "ptmu2", "ptmu3", "etamu1", "etamu2", "etamu3", "phimu1", "phimu2", "phimu3"],
        "jets": ["ptj1", "ptj2", "ptj3", "etaj1", "etaj2", "etaj3", "phij1", "phij2", "phij3", "nJets"],
        "METv": ["METv", "etaMETv", "phiMETv"],
        "bjets": ["ptb1", "ptb2", "ptb3", "etab1", "etab2", "etab3", "phib1", "phib2", "phib3", "nBJets"],
        "ZCand": []
    },
    "Pre_1e2mu": {
        "leptons": ["ptmu1", "ptmu2", "pte1", "etamu1", "etamu2", "etae1", "phimu1", "phimu2", "phie1"],
        "jets": ["ptj1", "ptj2", "ptj3", "etaj1", "etaj2", "etaj3", "phij1", "phij2", "phij3", "nJets"],
        "METv": ["METv", "etaMETv", "phiMETv"],
        "bjets": ["ptb1", "ptb2", "ptb3", "etab1", "etab2", "etab3", "phib1", "phib2", "phib3", "nBJets"],
        "ZCand": []
    }
}


if __name__ == "__main__":
    analyzer = "Preselection"
    syst = "VetoLeptons__"
    data = "DoubleMuon"
    year = "2017"
    mcsamples = ["fake", "VV", "SingleTop", "TTLL", "DY"]
    #mcsamples = ["fake", "rare", "VV", "ttX"]
    channel = "DY_dimu"
    skflat_output = "~/data0/HcToWA/"+analyzer+"/"+syst+"/"+year
    output_path = "Outputs/Preselection/"+syst

    leptons = observables[channel]["leptons"]
    jets = observables[channel]["jets"]
    bjets = observables[channel]["bjets"]
    met = observables[channel]["METv"]
    zcand = observables[channel]['ZCand']
    all_obs = leptons + jets + bjets + met + zcand
    params = param_set[channel]

    file = TFile()
    for obs in all_obs:
        hist_params = params[obs]["hist_params"]
        path_to_hist = channel + params[obs]["path"]
        print(path_to_hist)
        file = TFile(skflat_output+"/DATA/"+analyzer+"_"+data+".root")
        hist = file.Get(path_to_hist)
        hist.SetDirectory(0)
        hist_obs = hist

        for sample in mcsamples:
            if sample == "TTLL":
                file = TFile(skflat_output+"/DiLepSamples/" +
                             analyzer + "_TTLL_powheg.root")
            else:
                file = TFile(skflat_output+"/DiLepSamples/" +
                             analyzer+"_"+sample+".root")
            hist = file.Get(path_to_hist)
            hist.SetDirectory(0)
            hists_exp[sample] = hist

        plotter = ObsAndExp(cvs_params, hist_params, info_params)
        plotter.get_hists(hist_obs, hists_exp)
        plotter.combine()
        save_name = output_path + "/" + channel + "/DataAndMC_" + \
            channel + "_" + obs + "_log.png"
        plotter.save(save_name)
