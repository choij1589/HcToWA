# configure for local setup
# no need for condor
# from Scripts.setup import setup
# setup()

import argparse
import numpy as np
from ROOT import TFile, TH1D
from Scripts.DataFormat import Particle, get_leptons, get_jets
from Scripts.DataFormat import scale_electrons, smear_electrons, scale_muons, scale_jets, smear_jets
from Scripts.HistTools import HistTool

parser = argparse.ArgumentParser()
parser.add_argument("--sample", "-s", default=None, required=True,
                    type=str, help="sample name")
parser.add_argument("--object", "-o", default=None, required=True,
                    type=str, help="electron or muon")
parser.add_argument("--num_core", "-n", default=1, required=True, type=int, help="number of cores to submit")
parser.add_argument("--this_core", "-t", default=None, required=True, type=int, help="the number of core")
parser.add_argument("--flag", "-f", default=None, required=False, type=str, help="user defined flag")
args = parser.parse_args()

# define global variables
DATA = ["SingleElectron", "DoubleMuon"]
MC = ["QCD_EMEnriched", "QCD_bcToE", "QCD_MuEnriched", "W", "DY", "TT", "ST", "VV"]
SAMPLE = args.sample

if args.object == "electron":
    SAMPLE_DIR = "/data6/Users/choij/CMSSW/CMSSW_11_3_0_pre2/src/HcToWA/Samples/FakeEstimator/2017/SkimSglEle__WP90__"
    OUTFILE = f"/data6/Users/choij/CMSSW/CMSSW_11_3_0_pre2/src/HcToWA/FakeEstimator/Outputs/{args.object}/{SAMPLE}/SglEle_WP90_{SAMPLE}_{args.this_core}.root"
    DATASTREAM = "SingleElectron"
    MCSAMPLEs = ["QCD_EMEnriched", "QCD_bcToE", "W", "DY", "TT", "ST", "VV"]
elif args.object == "muon":
    SAMPLE_DIR = "/data6/Users/choij/CMSSW/CMSSW_11_3_0_pre2/src/HcToWA/Samples/FakeEstimator/2017/SkimSglMu__"
    OUTFILE = f"/data6/Users/choij/CMSSW/CMSSW_11_3_0_pre2/src/HcToWA/FakeEstimator/Outputs/{args.object}/{SAMPLE}/SglMu_{SAMPLE}_{args.this_core}.root"
    DATASTREAM = "DoubleMuon"
    MCSAMPLEs = ["QCD_MuEnriched", "W", "DY", "TT", "ST", "VV"]
else:
    print(f"Wrong object {args.object}")
    raise(AttributeError)


SELECTSYSTs = ["Central", "FlavorDep", "JetPtCutUp", "JetPtCutDown"]
PROMPTSYSTs = ["JetEnUp", "JetEnDown", "JetResUp", "JetResDown", "ElectronEnUp",
               "ElectronEnDown", "ElectronResUp", "ElectronResDown", "MuonEnUp", "MuonEnDown"]
if args.flag == "syst":
	SYSTs = SELECTSYSTs + PROMPTSYSTs
else:
	SYSTs = ["Central"]

# Get pileup reweight
f_pileup = TFile.Open(
    "/data6/Users/choij/CMSSW/CMSSW_11_3_0_pre2/src/HcToWA/MetaInfo/2017/PUReweight_2017.root")
h_pileup = f_pileup.Get("PUReweight_2017")
h_pileup.SetDirectory(0)
f_pileup.Close()


# prepare hitogram tools
htool = HistTool(outfile=OUTFILE)

def scale_or_smear(muons, electrons, jets, bjets, syst):
    if syst in SELECTSYSTs:
        pass
    elif syst == "ElectronEnUp":
        scale_electrons(electrons, 'Up')
    elif syst == "ElectronEnDown":
        scale_electrons(electrons, 'Down')
    elif syst == "ElectronResUp":
        smear_electrons(electrons, 'Up')
    elif syst == "ElectronResDown":
        smear_electrons(electrons, "Down")
    elif syst == "MuonEnUp":
        scale_muons(muons, 'Up')
    elif syst == "MuonEnDown":
        scale_muons(muons, 'Down')
    elif syst == "JetEnUp":
        scale_jets(jets, 'Up')
        scale_jets(bjets, 'Up')
    elif syst == "JetEnDown":
        scale_jets(jets, 'Down')
        scale_jets(bjets, 'Down')
    elif syst == "JetResUp":
        smear_jets(jets, 'Up')
        smear_jets(bjets, 'Up')
    elif syst == "JetResDown":
        smear_jets(jets, 'Down')
        smear_jets(bjets, 'Down')
    else:
        print(f"Wrong systematics {syst}")
        raise(AttributeError)


def select(evt, muons, electrons, jets, bjets, syst):
    # NOTE: No Trigger selection!
    # only one lepton
    # at least one jet with Pt > 40.
    # dR(l, j1) > 1.0

    # Assign jets to bjets for mother flaver dependency
    if syst == "FlavorDep":
        jets = bjets

    # define lepton
    if args.object == "electron":
        if not (len(electrons) == 1 and len(muons) == 0):
            return False
        if not (len(jets) > 0):
            return False
        if not (electrons[0].DeltaR(jets[0]) > 1.0):
            return False

        # mother jet momentum dependency
        if syst == "JetPtCutUp":
            if not (jets[0].Pt() > 60.):
                return False
        elif syst == "JetPtCutDown":
            if not (jets[0].Pt() > 30.):
                return False
        else:
            if not (jets[0].Pt() > 40.):
                return False

    elif args.object == "muon":
        if not (len(electrons) == 0 and len(muons) == 1):
            return False
        if not (len(jets) > 0):
            return False
        if not (muons[0].DeltaR(jets[0]) > 1.0):
            return False

        # mother jet momentum dependency
        if syst == "JetPtCutUp":
            if not (jets[0].Pt() > 60.):
                return False
        elif syst == "JetPtCutDown":
            if not (jets[0].Pt() > 20.):
                return False
        else:
            if not (jets[0].Pt() > 40.):
                return False
    else:
        print(f"[Analyzer::select] Wrong object {args.object}")
        raise(AttributeError)

    return True


def get_weight(evt):
    # IMPORTANT: Triggers are prescaled! no trigger lumi
    weight = 1.
    if evt.IsDATA:
        return weight

    weight *= evt.genWeight
    weight *= evt.weights_L1Prefire[0]
    weight *= evt.weights_btag[0]
    weight *= h_pileup.GetBinContent(h_pileup.FindBin(evt.nPileUp))
    
    return weight


def MT(lep, METv):
    Mt = np.sqrt(2*lep.Pt()*METv.Pt()*(1.-np.cos(lep.DeltaPhi(METv))))
    return Mt

def split(evt):
	n_evt = evt.GetEntries()
	dividen = int(n_evt/args.num_core)
	first = dividen*(args.this_core)+1
	if args.this_core == args.num_core-1:
		end = n_evt
	else:
		end = dividen*(args.this_core+1)

	return (first, end)

def main():
    # Get Root file for data
    print(f"Estimating {SAMPLE}...")
    if SAMPLE in DATA:
        fkey = f"{SAMPLE_DIR}/DATA/FakeEstimator_{SAMPLE}.root"
    elif SAMPLE in MC:
        fkey = f"{SAMPLE_DIR}/MCSamples/FakeEstimator_{SAMPLE}.root"
    else:
        raise AttributeError

    f = TFile.Open(fkey)
    evt = f.Get("Events")

    # divide events
    first, end = split(evt)
    print(first, end, evt.GetEntries())
    for syst in SYSTs:
        for i in range(first, end+1):
            evt.GetEntry(i)
            # reconstruct objects
            muons, electrons = get_leptons(evt)
            jets, bjets = get_jets(evt)
            METv = Particle(evt.METv_pt, evt.METv_eta, evt.METv_phi, 0.)
            scale_or_smear(muons, electrons, jets, bjets, syst)

            # event selection
            if not select(evt, muons, electrons, jets, bjets, syst):
                continue

            if args.object == "electron":
                ele = electrons[0]
                ptCorr = ele.Pt()*(1.+max(ele.MiniIso() - 0.1, 0.))
                weight = get_weight(evt)

                # loose ID part
                Mt = MT(ele, METv)
                #if evt.passEle8Path and (10. <= ptCorr and ptCorr < 20.):
                if evt.passEle8Path:
                    trigLumi = 3.973 if not evt.IsDATA else 1.
                    key = f"loose/passEle8Path/{syst}"
                    htool.fill_hist(f"{key}/MT", Mt, weight *
                                    trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/MET", METv.Pt(),
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_electrons(
                        f"{key}/electrons", electrons, weight*trigLumi)
                    htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    if abs(ele.Eta()) < 0.8:
                        key = f"loose/passEle8Path/{syst}/eta0to0p8"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(ele.Eta()) < 1.479:
                        key = f"loose/passEle8Path/{syst}/eta0p8to1p479"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(ele.Eta()) < 2.5:
                        key = f"loose/passEle8Path/{syst}/eta1p479to2p5"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    else:
                        pass

                #if evt.passEle12Path and (20. <= ptCorr and ptCorr < 35.):
                if evt.passEle12Path:
                    trigLumi = 27.699 if not evt.IsDATA else 1.
                    key = f"loose/passEle12Path/{syst}"
                    htool.fill_hist(f"{key}/MT", Mt, weight *
                                    trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/MET", METv.Pt(),
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_electrons(
                        f"{key}/electrons", electrons, weight*trigLumi)
                    htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    if abs(ele.Eta()) < 0.8:
                        key = f"loose/passEle12Path/{syst}/eta0to0p8"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(ele.Eta()) < 1.479:
                        key = f"loose/passEle12Path/{syst}/eta0p8to1p479"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(ele.Eta()) < 2.5:
                        key = f"loose/passEle12Path/{syst}/eta1p479to2p5"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    else:
                        pass

                #if evt.passEle23Path and (35. <= ptCorr and ptCorr <= 70.):
                if evt.passEle23Path:
                    trigLumi = 43.468 if not evt.IsDATA else 1.
                    key = f"loose/passEle23Path/{syst}"
                    htool.fill_hist(f"{key}/MT", Mt, weight *
                                    trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/MET", METv.Pt(),
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_electrons(
                        f"{key}/electrons", electrons, weight*trigLumi)
                    htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    if abs(ele.Eta()) < 0.8:
                        key = f"loose/passEle23Path/{syst}/eta0to0p8"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(ele.Eta()) < 1.479:
                        key = f"loose/passEle23Path/{syst}/eta0p8to1p479"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(ele.Eta()) < 2.5:
                        key = f"loose/passEle23Path/{syst}/eta1p479to2p5"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    else:
                        pass

                if not ele.IsTight():
                    continue

                #if evt.passEle8Path and (10. <= ptCorr and ptCorr < 20.):
                if evt.passEle8Path:
                    trigLumi = 3.973 if not evt.IsDATA else 1.
                    key = f"tight/passEle8Path/{syst}"
                    htool.fill_hist(f"{key}/MT", Mt, weight *
                                    trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/MET", METv.Pt(),
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_electrons(
                        f"{key}/electrons", electrons, weight*trigLumi)
                    htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    if abs(ele.Eta()) < 0.8:
                        key = f"tight/passEle8Path/{syst}/eta0to0p8"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(ele.Eta()) < 1.479:
                        key = f"tight/passEle8Path/{syst}/eta0p8to1p479"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(ele.Eta()) < 2.5:
                        key = f"tight/passEle8Path/{syst}/eta1p479to2p5"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    else:
                        pass

                # if evt.passEle12Path and (20. <= ptCorr and ptCorr < 35.):
                if evt.passEle12Path:
                    trigLumi = 27.699 if not evt.IsDATA else 1.
                    key = f"tight/passEle12Path/{syst}"
                    htool.fill_hist(f"{key}/MT", Mt, weight *
                                    trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/MET", METv.Pt(),
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_electrons(
                        f"{key}/electrons", electrons, weight*trigLumi)
                    htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    if abs(ele.Eta()) < 0.8:
                        key = f"tight/passEle12Path/{syst}/eta0to0p8"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(ele.Eta()) < 1.479:
                        key = f"tight/passEle12Path/{syst}/eta0p8to1p479"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(ele.Eta()) < 2.5:
                        key = f"tight/passEle12Path/{syst}/eta1p479to2p5"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)

                # if evt.passEle23Path and (35. <= ptCorr and ptCorr <= 70.):
                if evt.passEle23Path:
                    trigLumi = 43.468 if not evt.IsDATA else 1.
                    key = f"tight/passEle23Path/{syst}"
                    htool.fill_hist(f"{key}/MT", Mt, weight *
                                    trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/MET", METv.Pt(),
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_electrons(
                        f"{key}/electrons", electrons, weight*trigLumi)
                    htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    if abs(ele.Eta()) < 0.8:
                        key = f"tight/passEle23Path/{syst}/eta0to0p8"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(ele.Eta()) < 1.479:
                        key = f"tight/passEle23Path/{syst}/eta0p8to1p479"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(ele.Eta()) < 2.5:
                        key = f"tight/passEle23Path/{syst}/eta1p479to2p5"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_electrons(
                            f"{key}/electrons", electrons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)

            elif args.object == "muon":
                muon = muons[0]
                ptCorr = muon.Pt()*(1.+max(muon.MiniIso() - 0.1, 0.))
                weight = get_weight(evt)

                # loose ID part
                Mt = MT(muon, METv)
                # if evt.passMu8Path and (10. <= ptCorr and ptCorr < 30.):
                if evt.passMu8Path:
                    trigLumi = 2.8977 if not evt.IsDATA else 1.
                    key = f"loose/passMu8Path/{syst}"
                    htool.fill_hist(f"{key}/MT", Mt, weight *
                                    trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/MET", METv.Pt(),
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_muons(f"{key}/muons", muons, weight*trigLumi)
                    htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    if abs(muon.Eta()) < 0.9:
                        key = f"loose/passMu8Path/{syst}/eta0to0p9"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_muons(
                            f"{key}/muons", muons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(muon.Eta()) < 1.6:
                        key = f"loose/passMu8Path/{syst}/eta0p9to1p6"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_muons(
                            f"{key}/muons", muons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(muon.Eta()) < 2.4:
                        key = f"loose/passMu8Path/{syst}/eta1p6to2p4"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_muons(
                            f"{key}/muons", muons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    else:
                        pass

                # if evt.passMu17Path and (30. <= ptCorr < 70.):
                if evt.passMu17Path:
                    trigLumi = 65.8989 if not evt.IsDATA else 1.
                    key = f"loose/passMu17Path/{syst}"
                    htool.fill_hist(f"{key}/MT", Mt, weight *
                                    trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/MET", METv.Pt(),
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                    weight*trigLumi, 300, 0., 300.)
                    htool.fill_muons(f"{key}/muons", muons, weight*trigLumi)
                    htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    if abs(muon.Eta()) < 0.9:
                        key = f"loose/passMu17Path/{syst}/eta0to0p9"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_muons(
                            f"{key}/muons", muons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(muon.Eta()) < 1.6:
                        key = f"loose/passMu17Path/{syst}/eta0p9to1p6"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_muons(
                            f"{key}/muons", muons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(muon.Eta()) < 2.4:
                        key = f"loose/passMu17Path/{syst}/eta1p6to2p4"
                        htool.fill_hist(f"{key}/MT", Mt,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                        weight*trigLumi, 300, 0., 300.)
                        htool.fill_muons(
                            f"{key}/muons", muons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    else:
                        pass

                if not muon.IsTight():
                    continue

                # if evt.passMu8Path and (10. <= ptCorr and ptCorr < 30.):
                if evt.passMu8Path:
                    trigLumi = 2.8977 if not evt.IsDATA else 1.
                    key = f"tight/passMu8Path/{syst}"
                    htool.fill_hist(f"{key}/MT", Mt, weight*trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/MET", METv.Pt(), weight*trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/ptCorr", ptCorr, weight*trigLumi, 300, 0., 300.)
                    htool.fill_muons(f"{key}/muons", muons, weight*trigLumi)
                    htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    if abs(muon.Eta()) < 0.9:
                        key = f"tight/passMu8Path/{syst}/eta0to0p9"
                        htool.fill_hist(f"{key}/MT", Mt, weight *
                                trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                weight*trigLumi, 300, 0., 300.)
                        htool.fill_muons(f"{key}/muons", muons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(muon.Eta()) < 1.6:
                        key = f"tight/passMu8Path/{syst}/eta0p9to1p6"
                        htool.fill_hist(f"{key}/MT", Mt, weight *
                                trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                weight*trigLumi, 300, 0., 300.)
                        htool.fill_muons(f"{key}/muons", muons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(muon.Eta()) < 2.4:
                        key = f"tight/passMu8Path/{syst}/eta1p6to2p4"
                        htool.fill_hist(f"{key}/MT", Mt, weight *
                                trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                weight*trigLumi, 300, 0., 300.)
                        htool.fill_muons(f"{key}/muons", muons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    else:
                        pass

                # if evt.passMu17Path and (30. <= ptCorr < 70.):
                if evt.passMu17Path:
                    trigLumi = 65.8989 if not evt.IsDATA else 1.
                    key = f"tight/passMu17Path/{syst}"
                    htool.fill_hist(f"{key}/MT", Mt, weight*trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/MET", METv.Pt(),
                            weight*trigLumi, 300, 0., 300.)
                    htool.fill_hist(f"{key}/ptCorr", ptCorr,
                            weight*trigLumi, 300, 0., 300.)
                    htool.fill_muons(f"{key}/muons", muons, weight*trigLumi)
                    htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    if abs(muon.Eta()) < 0.9:
                        key = f"tight/passMu17Path/{syst}/eta0to0p9"
                        htool.fill_hist(f"{key}/MT", Mt, weight *
                                trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                    weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                weight*trigLumi, 300, 0., 300.)
                        htool.fill_muons(f"{key}/muons", muons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(muon.Eta()) < 1.6:
                        key = f"tight/passMu17Path/{syst}/eta0p9to1p6"
                        htool.fill_hist(f"{key}/MT", Mt, weight *
                                trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                weight*trigLumi, 300, 0., 300.)
                        htool.fill_muons(f"{key}/muons", muons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    elif abs(muon.Eta()) < 2.4:
                        key = f"tight/passMu17Path/{syst}/eta1p6to2p4"
                        htool.fill_hist(f"{key}/MT", Mt, weight *
                                trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/MET", METv.Pt(),
                                weight*trigLumi, 300, 0., 300.)
                        htool.fill_hist(f"{key}/ptCorr", ptCorr,
                                    weight*trigLumi, 300, 0., 300.)
                        htool.fill_muons(f"{key}/muons", muons, weight*trigLumi)
                        htool.fill_jets(f"{key}/jets", jets, weight*trigLumi)
                    else:
                        pass

    htool.save()


if __name__ == "__main__":
    main()
    print("End process")
