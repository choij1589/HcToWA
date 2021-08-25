import os
from itertools import product, combinations
import numpy as np
from ROOT import TFile, TTree, TCanvas, TH1D, TH2D
from Scripts.DataFormat import Particle, get_leptons, get_jets
from Scripts.HistTools import HistTool
from ROOT import EnableImplicitMT
EnableImplicitMT(20)


# define global variables
SAMPLE_DIR = "/root/workspace/HcToWA/Samples/Selector/2017/Skim3Mu__"
DATASTREAM = "DoubleMuon"
BKG_LIST = ["rare", "conv", "ttX", "VV", "fake"]
PROMPT_LIST = ["rare", "conv", "ttX", "VV"]
SYSTS = ["Central", "L1PrefireUp", "L1PrefireDown", "PUReweightUp",
         "PUReweightDown", "IDSFUp", "IDSFDown", "TrigSFUp", "TrigSFDown"]
#SYSTS = ["Central"]

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
maker = HistTool(outfile="ZFake_3Mu.root")


def make_ospairs(muons):
    nMuons = len(muons)
    comb = combinations(range(nMuons), 2)
    zcands = []
    for ele in comb:
        mu1 = muons[ele[0]]
        mu2 = muons[ele[1]]
        if mu1.Charge() + mu2.Charge() == 0:
            cand = mu1 + mu2
            zcands.append(cand)

    return zcands[0], zcands[1]


def assign_zcand(muons):
    zcand1, zcand2 = make_ospairs(muons)
    if abs(zcand1.Mass() - 91.2) < abs(zcand2.Mass() - 91.2):
        return zcand1
    else:
        return zcand2


def select(evt, muons, electrons, jets, bjets):
    # should pass dblmu triggers
    # should pass safe cut
    # exist onshell Z
    # all os pair above 12 GeV
    # no b-jet
    if not evt.passDblMuTrigs:
        return False

    pass_safecut = muons[0].Pt() > 20. and muons[1].Pt() > 10.
    if not pass_safecut:
        return False

    # check os pair
    if not abs(muons[0].Charge() + muons[1].Charge() + muons[2].Charge()) == 1:
        return False

    ZCand1, ZCand2 = make_ospairs(muons)

    if not (ZCand1.Mass() > 12. and ZCand2.Mass() > 12.):
        return False
    if not (abs(ZCand1.Mass() - 91.2) < 10. or abs(ZCand2.Mass() - 91.2) < 10.):
        return False

    if not len(bjets) == 0:
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
        ptCorr = min(ptCorr, 49.)
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
    f_data = TFile.Open(fkey)

    # loop
    # data
    print(f"Estimating {DATASTREAM}")
    for evt in f_data.Events:
        # reconstruct objects
        muons, electrons = get_leptons(evt)
        jets, bjets = get_jets(evt)
        METv = Particle(evt.METv_Pt, evt.METv_eta, evt.METv_phi, 0.)

        # event selection
        if not select(evt, muons, electrons, jets, bjets):
            continue

        # check if pass tight IDs
        muons_tight, electrons_tight = get_tight_leptons(muons, electrons)
        ZCand = assign_zcand(muons)

        if len(muons_tight) == 3:
            maker.fill_hist("data/ZMass", ZCand.Mass(), 1., 30, 75., 105.)
            maker.fill_hist("data/MET", METv.Pt(), 1., 300, 0., 300.)
            maker.fill_hist("data/nPileUp", evt.nPileUp, 1., 100, 0., 100.)
            maker.fill_hist("data/nPV", evt.nPV, 1., 100, 0., 100.)
            maker.fill_muons("data/muons", muons, 1.)
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
            maker.fill_jets("fake/Central/jets", jets, w_fake)

            maker.fill_hist("fake/Up/ZMass", ZCand.Mass(),
                            w_fake_up, 30, 75., 105.)
            maker.fill_hist("fake/Up/MET", METv.Pt(), w_fake_up, 300, 0., 300.)
            maker.fill_hist("fake/Up/nPileUp", evt.nPileUp,
                            w_fake_up, 100, 0., 100.)
            maker.fill_hist("fake/Up/nPV", evt.nPV, w_fake_up, 100, 0., 100.)
            maker.fill_muons("fake/Up/muons", muons, w_fake_up)
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
            maker.fill_jets("fake/Down/jets", jets, w_fake_down)
    f_data.Close()

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
            if not select(evt, muons, electrons, jets, bjets):
                continue

            # check if pass tight IDs
            muons_tight, elctrons_tight = get_tight_leptons(muons, electrons)
            ZCand = assign_zcand(muons)

            # all tight for mcs
            if len(muons_tight) != 3:
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
                sf_trig = evt.weights_trigs_dblmu[0]
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
                    sf_trig = evt.weights_trigs_dblmu[1]
                elif syst == "TrigSFDown":
                    sf_trig = evt.weights_trigs_dblmu[2]
                else:
                    pass

                weight *= w_L1Prefire
                weight *= w_pileup
                weight *= sf_id
                weight *= sf_trig
                weight *= sf_btag

                # Fill histograms
                maker.fill_hist(mc+"/"+syst+"/ZMass", ZCand.Mass(),
                                weight, 30, 75., 105.)
                maker.fill_hist(mc+"/"+syst+"/MET", METv.Pt(),
                                weight, 300, 0., 300.)
                maker.fill_hist(mc+"/"+syst+"/nPileUp",
                                evt.nPileUp, weight, 100, 0., 100.)
                maker.fill_hist(mc+"/"+syst+"/nPV", evt.nPV, weight, 100, 0., 100.)
                maker.fill_muons(mc+"/"+syst+"/muons", muons, weight)
                maker.fill_jets(mc+"/"+syst+"/jets", jets, weight)
        f.Close()

    maker.save()


if __name__ == "__main__":
    main()
