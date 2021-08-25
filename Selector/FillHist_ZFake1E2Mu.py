import os
import shutil
from itertools import product, combinations
import numpy as np
from ROOT import TFile, TTree, TCanvas, TH1D, TH2D
from Scripts.DataFormat import Particle, get_leptons, get_jets
from Scripts.HistTools import HistTool
from ROOT import EnableImplicitMT
EnableImplicitMT(20)


# define global variables
SAMPLE_DIR = "/root/workspace/HcToWA/Samples/Selector/2017/Skim1E2Mu__"
DATASTREAM = "DoubleMuon"
BKG_LIST = ["rare", "conv", "ttX", "VV", "fake"]
PROMPT_LIST = ["fake_mc", "rare", "conv", "ttX", "VV"]
#PROMPT_LIST = ["ZGToLLG_01J", "conv"]
SYSTS = ["Central", "L1PrefireUp", "L1PrefireDown", "PUReweightUp",
         "PUReweightDown", "IDSFUp", "IDSFDown", "TrigSFUp", "TrigSFDown"]

# Get fake rate and pileup reweight
f_muon = TFile.Open(
    "/root/workspace/HcToWA/MetaInfo/2017/fakerate_muon.root")
f_electron = TFile.Open(
    "/root/workspace/HcToWA/MetaInfo/2017/fakerate_electron.root")
f_pileup = TFile.Open(
    "/root/workspace/HcToWA/MetaInfo/2017/PUReweight_2017.root")

h_muon = f_muon.Get("h_fakerate")
h_electron = f_electron.Get("h_fakerate")
h_pileup = f_pileup.Get("PUReweight_2017")
h_pileup_up = f_pileup.Get("PUReweight_2017_Up")
h_pileup_down = f_pileup.Get("PUReweight_2017_Down")
h_muon.SetDirectory(0)
h_pileup.SetDirectory(0)
h_pileup_up.SetDirectory(0)
h_pileup_down.SetDirectory(0)

f_muon.Close()
f_pileup.Close()

# prepare histogram tools
maker = HistTool(outfile="ZFake_1E2Mu.root")
#maker = HistTool(outfile="out.root")


def select(evt, muons, electrons, jets, bjets, METv):
    # should pass dblmu triggers
    # should pass safe cut
    # exist onshell Z
    # all os pair above 12 GeV
    # no b-jet
    if not (evt.passDblMuTrigs or evt.passEMuTrigs):
        return False

    pass_safecut = ((muons[0].Pt() > 20. and muons[1].Pt() > 10.) or
                    (muons[0].Pt() > 25. and electrons[0].Pt() > 15.) or
                    (electrons[0].Pt() > 25. and muons[0].Pt() > 10.))
    if not pass_safecut:
        return False

    # check os pair
    if not muons[0].Charge() + muons[1].Charge() == 0:
        return False

    ZCand = muons[0] + muons[1]

    if not abs(ZCand.Mass() - 91.2) < 10.:
        return False

    if not len(bjets) == 0:
        return False

    if not (METv.Pt() > 80.):
        return False

    return True


def get_tight_leptons(muons, electrons):
    muons_tight = []
    electrons_tight = []
    for muon in muons:
        if muon.IsTight():
            muons_tight.append(muon)
    for electron in electrons:
        if electron.IsTight():
            electrons_tight.append(electron)

    return muons_tight, electrons_tight


def get_fake_weights(muons, electrons):
    w_fake = -1.
    w_fake_up = -1.
    w_fake_down = -1.
    for muon in muons:
        ptCorr = muon.Pt()*(1.+max(muon.MiniIso()-0.1, 0.))
        ptCorr = min(ptCorr, 69.)
        absEta = abs(muon.Eta())
        this_bin = h_muon.FindBin(ptCorr, absEta)
        fr = h_muon.GetBinContent(this_bin)
        fr_up = fr + h_muon.GetBinError(this_bin)
        fr_down = fr - h_muon.GetBinError(this_bin)

        if muon.IsTight():
            continue
        else:
            w_fake *= -fr/(1.-fr)
            w_fake_up *= -fr_up/(1.-fr_up)
            w_fake_down *= -fr_down/(1.-fr_down)
    for ele in electrons:
        ptCorr = ele.Pt()*(1.+max(ele.MiniIso()-0.1, 0.))
        ptCorr = min(ptCorr, 69.)
        absEta = abs(ele.Eta())
        this_bin = h_electron.FindBin(ptCorr, absEta)
        fr = h_electron.GetBinContent(this_bin)
        fr_up = fr + h_electron.GetBinError(this_bin)
        fr_down = fr - h_electron.GetBinError(this_bin)

        if ele.IsTight():
            continue
        else:
            w_fake *= -fr/(1.-fr)
            w_fake_up *= -fr_up/(1.-fr_up)
            w_fake_down *= -fr_down/(1.-fr_down)

    return w_fake, w_fake_up, w_fake_down


def main():

    # Get root file for data
    fkey = os.path.join(SAMPLE_DIR, "DATA", "Selector_DoubleMuon.root")
    f_dblmu = TFile.Open(fkey)

    # loop
    # DoubleMuon
    events = []
    print(f"Estimating DoubleMuon...")
    for evt in f_dblmu.Events:
        # fill events
        this_evt = (evt.run, evt.event, evt.lumi)
        events.append(this_evt)

        # reconstruct objects
        muons, electrons = get_leptons(evt)
        jets, bjets = get_jets(evt)
        METv = Particle(evt.METv_Pt, evt.METv_eta, evt.METv_phi, 0.)

        # event selection
        if not select(evt, muons, electrons, jets, bjets, METv):
            continue

        # check if pass tight IDs
        muons_tight, electrons_tight = get_tight_leptons(muons, electrons)
        ZCand = muons[0] + muons[1]

        if len(muons_tight) == 2 and len(electrons_tight) == 1:
            maker.fill_hist("data/ZMass", ZCand.Mass(), 1., 200, 0., 200.)
            maker.fill_hist("data/MET", METv.Pt(), 1., 300, 0., 300.)
            maker.fill_hist("data/nPileUp", evt.nPileUp, 1., 100, 0., 100.)
            maker.fill_hist("data/nPV", evt.nPV, 1., 100, 0., 100.)
            maker.fill_muons("data/muons", muons, 1.)
            maker.fill_electrons("data/electrons", electrons, 1.)
            maker.fill_jets("data/jets", jets, 1.)
        else:
            # Central, Up, Down
            w_fake, w_fake_up, w_fake_down = get_fake_weights(muons, electrons)
            maker.fill_hist("fake/Central/ZMass", ZCand.Mass(),
                            w_fake, 30, 75., 105.)
            maker.fill_hist("fake/Central/MET", METv.Pt(),
                            w_fake, 300, 0., 300.)
            maker.fill_hist("fake/Central/nPileUp",
                            evt.nPileUp, w_fake, 100, 0., 100.)
            maker.fill_hist("fake/Central/nPV", evt.nPV, w_fake, 100, 0., 100.)
            maker.fill_muons("fake/Central/muons", muons, w_fake)
            maker.fill_electrons("fake/Central/electrons", electrons, w_fake)
            maker.fill_jets("fake/Central/jets", jets, w_fake)

            maker.fill_hist("fake/Up/ZMass", ZCand.Mass(),
                            w_fake_up, 30, 75., 105.)
            maker.fill_hist("fake/Up/MET", METv.Pt(), w_fake_up, 300, 0., 300.)
            maker.fill_hist("fake/Up/nPileUp", evt.nPileUp,
                            w_fake_up, 100, 0., 100.)
            maker.fill_hist("fake/Up/nPV", evt.nPV, w_fake_up, 100, 0., 100.)
            maker.fill_muons("fake/Up/muons", muons, w_fake_up)
            maker.fill_electrons("fake/Up/electrons", electrons, w_fake_up)
            maker.fill_jets("fake/Up/jets", jets, w_fake_up)

            maker.fill_hist("fake/Down/ZMass", ZCand.Mass(),
                            w_fake_down, 30, 75., 105.)
            maker.fill_hist("fake/Down/MET", METv.Pt(),
                            w_fake_down, 300, 0., 300.)
            maker.fill_hist("fake/Down/nPileUp", evt.nPileUp,
                            w_fake_down, 100, 0., 100.)
            maker.fill_hist("fake/Down/nPV", evt.nPV,
                            w_fake_down, 100, 0., 100.)
            maker.fill_muons("fake/Down/muons", muons, w_fake_down)
            maker.fill_electrons("fake/Down/electrons", electrons, w_fake_down)
            maker.fill_jets("fake/Down/jets", jets, w_fake_down)
    f_dblmu.Close()

    # Get root file for data
    fkey = os.path.join(SAMPLE_DIR, "DATA", "Selector_MuonEG.root")
    f_emu = TFile.Open(fkey)
    # loop
    # MuonEG
    print(f"Estimating MuonEG...")
    for evt in f_emu.Events:
        # fill events
        this_evt = (evt.run, evt.event, evt.lumi)
        if this_evt in events:
            continue
        # reconstruct objects
        muons, electrons = get_leptons(evt)
        jets, bjets = get_jets(evt)
        METv = Particle(evt.METv_Pt, evt.METv_eta, evt.METv_phi, 0.)
        # event selection
        if not select(evt, muons, electrons, jets, bjets, METv):
            continue
        # check if pass tight IDs
        muons_tight, electrons_tight = get_tight_leptons(muons, electrons)
        ZCand = muons[0] + muons[1]
        if len(muons_tight) == 2 and len(electrons_tight) == 1:
            maker.fill_hist("data/ZMass", ZCand.Mass(), 1., 200, 0., 200.)
            maker.fill_hist("data/MET", METv.Pt(), 1., 300, 0., 300.)
            maker.fill_hist("data/nPileUp", evt.nPileUp, 1., 100, 0., 100.)
            maker.fill_hist("data/nPV", evt.nPV, 1., 100, 0., 100.)
            maker.fill_muons("data/muons", muons, 1.)
            maker.fill_electrons("data/electrons", electrons, 1.)
            maker.fill_jets("data/jets", jets, 1.)
        else:
            # Central, Up, Down
            w_fake, w_fake_up, w_fake_down = get_fake_weights(muons, electrons)
            maker.fill_hist("fake/Central/ZMass", ZCand.Mass(),
                            w_fake, 30, 75., 105.)
            maker.fill_hist("fake/Central/MET", METv.Pt(),
                            w_fake, 300, 0., 300.)
            maker.fill_hist("fake/Central/nPileUp",
                            evt.nPileUp, w_fake, 100, 0., 100.)
            maker.fill_hist("fake/Central/nPV", evt.nPV, w_fake, 100, 0., 100.)
            maker.fill_muons("fake/Central/muons", muons, w_fake)
            maker.fill_electrons("fake/Central/electrons", electrons, w_fake)
            maker.fill_jets("fake/Central/jets", jets, w_fake)
            maker.fill_hist("fake/Up/ZMass", ZCand.Mass(),
                            w_fake_up, 30, 75., 105.)
            maker.fill_hist("fake/Up/MET", METv.Pt(), w_fake_up, 300, 0., 300.)
            maker.fill_hist("fake/Up/nPileUp", evt.nPileUp,
                            w_fake_up, 100, 0., 100.)
            maker.fill_hist("fake/Up/nPV", evt.nPV, w_fake_up, 100, 0., 100.)
            maker.fill_muons("fake/Up/muons", muons, w_fake_up)
            maker.fill_electrons("fake/Up/electrons", electrons, w_fake_up)
            maker.fill_jets("fake/Up/jets", jets, w_fake_up)
            maker.fill_hist("fake/Down/ZMass", ZCand.Mass(),
                            w_fake_down, 30, 75., 105.)
            maker.fill_hist("fake/Down/MET", METv.Pt(),
                            w_fake_down, 300, 0., 300.)
            maker.fill_hist("fake/Down/nPileUp", evt.nPileUp,
                            w_fake_down, 100, 0., 100.)
            maker.fill_hist("fake/Down/nPV", evt.nPV,
                            w_fake_down, 100, 0., 100.)
            maker.fill_muons("fake/Down/muons", muons, w_fake_down)
            maker.fill_electrons("fake/Down/electrons", electrons, w_fake_down)
            maker.fill_jets("fake/Down/jets", jets, w_fake_down)
    f_emu.Close()

    # mc
    for mc in PROMPT_LIST:
        print(f"Estimating {mc}...")
        fkey = os.path.join(SAMPLE_DIR, "MCSamples", "Selector_"+mc+".root")
        f = TFile.Open(fkey)

        for evt in f.Events:
            # reconstruct objects
            muons, electrons = get_leptons(evt)
            jets, bjets = get_jets(evt)
            METv = Particle(evt.METv_Pt, evt.METv_eta, evt.METv_phi, 0.)

            # event selection
            if not select(evt, muons, electrons, jets, bjets, METv):
                continue

            # check if pass tight IDs
            muons_tight, electrons_tight = get_tight_leptons(muons, electrons)
            ZCand = muons[0] + muons[1]

            # all tight for mcs
            if not (len(muons_tight) == 2 and len(electrons_tight) == 1):
                continue

            # evaluate weight
            for syst in SYSTS:
                weight = 1.
                weight *= evt.genWeight
                weight *= evt.trigLumi

                # weight variation
                w_L1Prefire = evt.weights_L1Prefire[0]
                w_pileup = h_pileup.GetBinContent(
                    h_pileup.FindBin(evt.nPileUp))
                sf_id = evt.weights_id[0]
                sf_trig = 1. - \
                    (1. - evt.weights_trigs_dblmu[0]) * \
                    (1. - evt.weights_trigs_emu[0])
                sf_btag = evt.weights_btag[0]
                if syst == "L1PrefireUp":
                    w_L1Prefire = evt.weights_L1Prefire[1]
                elif syst == "L1PrefireDown":
                    w_L1Prefire = evt.weights_L1Prefire[2]
                elif syst == "PUReweightUp":
                    w_pileup = h_pileup_up.GetBinContent(
                        h_pileup_up.FindBin(evt.nPileUp))
                elif syst == "PUReweightDown":
                    w_pileup = h_pileup_down.GetBinContent(
                        h_pileup_down.FindBin(evt.nPileUp))
                elif syst == "IDSFUp":
                    sf_id = evt.weights_id[1]
                elif syst == "IDSFDown":
                    sf_id = evt.weights_id[2]
                elif syst == "TrigSFUp":
                    sf_trig = 1. - \
                        (1. - evt.weights_trigs_dblmu[1]) * \
                        (1. - evt.weights_trigs_emu[1])
                elif syst == "TrigSFDown":
                    sf_trig = 1. - \
                        (1. - evt.weights_trigs_dblmu[2]) * \
                        (1. - evt.weights_trigs_emu[2])
                else:
                    pass

                weight *= w_L1Prefire
                weight *= w_pileup
                weight *= sf_id
                weight *= sf_trig
                weight *= sf_btag

                # Fill histograms
                maker.fill_hist(mc+"/"+syst+"/electrontype",
                                electrons[0].LepType(), weight, 20, -10., 10.)
                maker.fill_hist(mc+"/"+syst+"/ZMass", ZCand.Mass(),
                                weight, 200, 0., 200.)
                maker.fill_hist(mc+"/"+syst+"/MET", METv.Pt(),
                                weight, 300, 0., 300.)
                maker.fill_hist(mc+"/"+syst+"/nPileUp",
                                evt.nPileUp, weight, 100, 0., 100.)
                maker.fill_hist(mc+"/"+syst+"/nPV", evt.nPV,
                                weight, 100, 0., 100.)
                maker.fill_muons(mc+"/"+syst+"/muons", muons, weight)
                maker.fill_electrons(
                    mc+"/"+syst+"/electrons", electrons, weight)
                maker.fill_jets(mc+"/"+syst+"/jets", jets, weight)
        f.Close()

    maker.save()


if __name__ == "__main__":
    main()
