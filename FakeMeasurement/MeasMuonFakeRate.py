import os
import shutil
from ROOT import TFile, TH2D

# Set flags
FLAG = "MeasMuon__"

# Define global variables
BASE_DIR = "/root/workspace/HcToWA/Samples/FakeMeasurement/2017"
ANALYZER = "FakeMeasurement"
PD = "DoubleMuon"
MCSamples = ["VV", "ST", "DY", "TT", "W"]

TRIGPATHs = ["HLT_Mu8_TrkIsoVVL", "HLT_Mu17_TrkIsoVVL"]
IDs = ["PassLoose", "PassTight"]
SYSTs = ["Central", "PromptNormUp", "PromptNormDown",
         "JetPtCutUp", "JetPtCutDown", "HasBjet"]


def get_scale(trig_path: str, id: str, syst: str) -> float:
    """Prompt normalization variation will be applied as pm 10% conservatively"""
    """This function is to estimate the exact systematic variation for each source"""
    hist_dir = "/DblMu"

    # data
    f_data = TFile.Open(os.path.join(
        BASE_DIR, FLAG, "DATA", ANALYZER+"_"+PD+".root"))
    h_data = f_data.Get(os.path.join(
        hist_dir, trig_path, id, syst, "ZCand", "mass"))
    h_data.SetDirectory(0)
    f_data.Close()

    # mc
    for i, mc in enumerate(MCSamples):
        f = TFile.Open(os.path.join(
            BASE_DIR, FLAG, "MCSamples", ANALYZER+"_"+mc+".root"))
        h = f.Get(os.path.join(hist_dir, trig_path, id, syst, "ZCand", "mass"))
        h.SetDirectory(0)
        if i == 0:
            h_mc = h.Clone("MC_ZMass")
            h_mc.SetDirectory(0)
        else:
            h_mc.Add(h)
        f.Close()

    # return scale factor
    return h_data.Integral() / h_mc.Integral()


def get_scale_with_syst(trig_path: str, id: str) -> float:
    systs = ["Central", "ElectronEnDown", "ElectronEnUp", "ElectronResUp", "ElectronResDown",
             "JetEnUp", "JetEnDown", "JetResUp", "JetResDown", "MuonEnUp", "MuonEnDown"]
    norms = list()
    diffs = list()

    central = get_scale(trig_path, id, "Central")
    for syst in systs:
        norm = get_scale(trig_path, id, syst)
        print(f"{syst}: {norm}")
        norms.append(norm)
        diffs.append(norm - central)

    syst_all = 0.
    for syst, diff in zip(systs, diffs):
        syst_all += pow(diff, 2)
    syst_all = pow(syst_all, 0.5)
    print(f"SF: {central} pm {syst_all}")

    return central, syst_all


def get_fake_rate(scales: dict, syst: str) -> TH2D:
    hist_dir = "/SglMu"
    ptCorr_bin = [10., 15., 20., 30., 50., 70., 100.]
    absEta_bin = [0., 0.9, 1.6, 2.4]

    # data
    f_data = TFile.Open(os.path.join(
        BASE_DIR, FLAG, "DATA", ANALYZER+"_"+PD+".root"))

    h_data_loose = f_data.Get(os.path.join(hist_dir, "PassLoose", syst))
    h_data_tight = f_data.Get(os.path.join(hist_dir, "PassTight", syst))
    if not h_data_loose:
        # no PromptNormUp/Down in the TFile
        h_data_loose = f_data.Get(os.path.join(
            hist_dir, "PassLoose", "Central"))
        h_data_tight = f_data.Get(os.path.join(
            hist_dir, "PassTight", "Central"))
    h_data_loose.SetDirectory(0)
    h_data_tight.SetDirectory(0)
    f_data.Close()

    # MC
    for i, mc in enumerate(MCSamples):
        f = TFile.Open(os.path.join(
            BASE_DIR, FLAG, "MCSamples", ANALYZER+"_"+mc+".root"))
        # get histograms
        h_temp_loose = f.Get(os.path.join(hist_dir, "PassLoose", syst))
        h_temp_tight = f.Get(os.path.join(hist_dir, "PassTight", syst))
        if not h_temp_loose:
            h_temp_loose = f.Get(os.path.join(
                hist_dir, "PassLoose", "Central"))
            h_temp_tight = f.Get(os.path.join(
                hist_dir, "PassTight", "Central"))
        # add to one
        if i == 0:
            h_mc_loose = h_temp_loose.Clone("MC_Loose")
            h_mc_tight = h_temp_tight.Clone("MC_Tight")
            h_mc_loose.SetDirectory(0)
            h_mc_tight.SetDirectory(0)
        else:
            h_mc_loose.Add(h_temp_loose)
            h_mc_tight.Add(h_temp_tight)
        f.Close()

    # Scale MC
    for absEta in absEta_bin:
        for ptCorr in ptCorr_bin:
            scale_loose = 1.
            scale_tight = 1.
            if ptCorr < 30.:
                scale_loose = scales["HLT_Mu8_TrkIsoVVL/PassLoose"]
                scale_tight = scales['HLT_Mu8_TrkIsoVVL/PassTight']
            else:
                scale_loose = scales["HLT_Mu17_TrkIsoVVL/PassLoose"]
                scale_tight = scales["HLT_Mu17_TrkIsoVVL/PassTight"]

            # PromptNormUp/Down
            if syst == "PromptNormUp":
                scale_loose *= 1.1
                scale_tight *= 1.1
            if syst == "PromptNormDown":
                scale_loose *= 0.9
                scale_tight *= 0.9

            # scale loose
            bin_loose = h_mc_loose.FindBin(ptCorr, absEta)
            bin_tight = h_mc_tight.FindBin(ptCorr, absEta)

            content_loose = h_mc_loose.GetBinContent(bin_loose)*scale_loose
            error_loose = h_mc_loose.GetBinError(bin_loose)*scale_loose

            content_tight = h_mc_tight.GetBinContent(bin_tight)*scale_tight
            error_tight = h_mc_tight.GetBinError(bin_tight)*scale_tight

            h_mc_loose.SetBinContent(bin_loose, content_loose)
            h_mc_loose.SetBinError(bin_loose, error_loose)
            h_mc_tight.SetBinContent(bin_tight, content_tight)
            h_mc_tight.SetBinError(bin_tight, error_tight)

    # Get Fake Rate
    h_loose = h_data_loose.Clone("h_loose_"+syst)
    h_loose.Add(h_mc_loose, -1.)
    h_tight = h_data_tight.Clone("h_tight_"+syst)
    h_tight.Add(h_mc_tight, -1.)

    h_fakerate = h_tight.Clone("h_fakerate_"+syst)
    h_fakerate.Divide(h_loose)
    return h_fakerate


if __name__ == "__main__":
    # get normalization factors
    scales = dict()
    for trig in TRIGPATHs:
        for id in IDs:
            key = "/".join([trig, id])
            scales[key] = get_scale(trig, id, "Central")
            print(f"{key}: {scales[key]}")

    # get central fake rate
    ptCorr_bin = [10., 15., 20., 30., 50., 70., 100.]
    absEta_bin = [0., 0.9, 1.6, 2.4]

    fakerates = dict()
    for syst in SYSTs:
        fakerate = get_fake_rate(scales, syst)
        fakerates[syst] = fakerate
        fakerates[syst].SetDirectory(0)

    central = fakerates['Central'].Clone("h_fakerate")
    for absEta in absEta_bin:
        for ptCorr in ptCorr_bin:
            this_bin = fakerates['Central'].FindBin(ptCorr, absEta)

            stat_err = fakerates['Central'].GetBinError(this_bin)
            total_err = pow(stat_err, 2)
            for syst in SYSTs:
                # estimate error
                syst_err = fakerates[syst].GetBinContent(
                    this_bin) - fakerates['Central'].GetBinContent(this_bin)
                total_err += pow(syst_err, 2)
            total_err = pow(total_err, 0.5)

            central.SetBinError(this_bin, total_err)

    f = TFile("fakerate_muon.root", "recreate")
    f.cd()
    for hist in fakerates.values():
        hist.Write()
    central.Write()
    f.Close()
