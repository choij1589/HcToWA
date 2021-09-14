import numpy as np
from itertools import combinations
from ROOT import TFile
from Scripts.DataFormat import Particle, scale_electrons, smear_electrons, scale_muons, scale_jets, smear_jets

# get fake rate and pileup reweight
f_muon = TFile.Open(
    "../MetaInfo/2017/fakerate_muon.root")
f_electron = TFile.Open(
    "../MetaInfo/2017/fakerate_electron.root")
f_pileup = TFile.Open(
    "../MetaInfo/2017/PUReweight_2017.root")

h_muon = f_muon.Get("fakerate2D")
h_electron = f_electron.Get("fakerate2D")
h_pileup = f_pileup.Get("PUReweight_2017")
h_pileup_up = f_pileup.Get("PUReweight_2017_Up")
h_pileup_down = f_pileup.Get("PUReweight_2017_Down")
h_muon.SetDirectory(0)
h_electron.SetDirectory(0)
h_pileup.SetDirectory(0)
h_pileup_up.SetDirectory(0)
h_pileup_down.SetDirectory(0)

f_muon.Close()
f_electron.Close()
f_pileup.Close()

def set_global(channel, sample):
    global CHANNEL, SAMPLE
    CHANNEL = channel
    SAMPLE = sample

def select_loosen(evt, muons, electrons, jets, bjets, METv):
    """Event selection for loosen criteria"""
    if CHANNEL == "1E2Mu":
        if not (evt.passDblMuTrigs or evt.passEMuTrigs):
            return False
        
        # check pt condition for the conversion samples
        if SAMPLE == "DY":
           if (muons[0].Pt() > 20. and muons[1].Pt() > 20. and electrons[0].Pt() > 20.):
                return False
        if SAMPLE == "ZG":
            if (muons[0].Pt() < 20. or muons[1].Pt() < 20. or electrons[0].Pt() < 20.):
                return False
        
        if not len(jets) >= 2:
            return False

    elif CHANNEL == "3Mu":
        if not evt.passDblMuTrigs:
            return False

        # check pt condition for the conversion samples
        if SAMPLE == "DY":
            if (muons[0].Pt() > 20. and muons[1].Pt() > 20. and muons[2].Pt() > 20.):
                return False
        if SAMPLE == "ZG":
            if (muons[0].Pt() < 20. or muons[1].Pt() < 20. or muons[2].Pt() < 20.):
                return False

        # jet conditions
        if not len(jets) >= 2:
            return False

    else:
        print(f"Wrong channel {CHANNEL}")
        raise(AttributeError)

    return True


def select(evt, muons, electrons, jets, bjets, METv):
    """Event selection for the signal region"""

    if not select_loosen(evt, muons, electrons, jets, bjets, METv):
        return False

    # 1E2Mu
    # 1. Should pass triggers and safe cut
    # 2. 1e2mu
    # 3. Exist OS muon pair with M(OSSF) > 12 GeV
    # 4. Nj >= 2, Nb >= 1
    if CHANNEL == "1E2Mu":
        pass_safecut = ((muons[0].Pt() > 20. and muons[1].Pt() > 10.) or
                        (muons[0].Pt() > 25. and electrons[0].Pt() > 15.) or
                        (electrons[0].Pt() > 25. and muons[0].Pt() > 10.))
        if not pass_safecut:
            return False

        # jet conditions
        if not len(bjets) >= 1:
            return False
    
    # 3Mu
    # 1. Should pass triggers and safe cut
    # 2. 3mu
    # 3. Exist OS muon pair, all OS muon pairs' mass > 12 GeV
    # 4. Nj >= 2, Nb >= 1
    elif CHANNEL == "3Mu":
        pass_safecut = muons[0].Pt() > 20. and muons[1].Pt() > 10.
        if not pass_safecut:
            return False

        if not len(bjets) >= 1:
            return False

    else:
        print(f"Wrong channel {CHANNEL}")
        raise(AttributeError)

    return True

def make_ospairs(muons):
    if not len(muons) == 3:
        print(f"Wrong muon number {len(muons)}")
        return None

    comb = combinations(range(nMuons), 2)
    OSpairs = []
    for cands in comb:
        mu1 = cands[0]
        mu2 = cands[1]
        if mu1.Charge() + mu2.Charge() == 0:
            OSpair = mu1 + mu2
            OSpairs.append(OSpair)

    OSpairs.sort(key=lambda OS: OS.Mass(), reverse=True)
    return (OSpairs[0], OSpairs[1])

def make_mumujjpairs(ACand, jets, mHc):
    comb = combinations(range(len(jets)), 2)
    mumujjpairs = []
    for cands in comb:
        mumujjpair = Particle(0., 0., 0., 0.)
        mumujjpair += ACand
        mumujjpair += jets[cands[0]]
        mumujjpair += jets[cands[1]]
        mumujjpairs.append(mumujjpair)

    mumujjpairs.sort(key=lambda pair: pair.Mass()-mHc)
    return mumujjpairs   # closest to the mHc

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


def get_prompt_leptons(muons, electrons):
    muons_prompt = []
    electrons_prompt = []
    for muon in muons:
        if muon.LepType() < 0:
            continue
        muons_prompt.append(muon)
    for electron in electrons:
        if electron.LepType() < 0:
            continue
        electrons_prompt.append(electron)

    return muons_prompt, electrons_prompt


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
        ptCorr = min(ptCorr, 49.)
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


def get_weight(evt, wsyst):
    weight = 1.
    weight *= evt.genWeight
    weight *= evt.trigLumi

    # weight variation
    w_L1Prefire = evt.weights_L1Prefire[0]
    w_pileup = h_pileup.GetBinContent(h_pileup.FindBin(evt.nPileUp))
    sf_id = evt.weights_id[0]
    if CHANNEL == "1E2Mu":
        sf_trig = 1. - (1. - evt.weights_trigs_dblmu[0]) * (1. - evt.weights_trigs_emu[0])
    elif CHANNEL == "3Mu":
        sf_trig = evt.weights_trigs_dblmu[0]
    else:
        raise(AttributeError)

    sf_btag = evt.weights_btag[0]
    if wsyst == "L1PrefireUp":
        w_L1Prefire = evt.weights_L1Prefire[1]
    elif wsyst == "L1PrefireDown":
        w_L1Prefire = evt.weights_L1Prefire[2]
    elif wsyst == "PUReweightUp":
        w_pileup = h_pileup_up.GetBinContent(h_pileup_up.FindBin(evt.nPileUp))
    elif wsyst == "PUReweightDown":
        w_pileup = h_pileup_down.GetBinContent(
            h_pileup_down.FindBin(evt.nPileUp))
    elif wsyst == "IDSFUp":
        sf_id = evt.weights_id[1]
    elif wsyst == "IDSFDown":
        sf_id = evt.weights_id[2]
    elif wsyst == "TrigSFUp":
        if CHANNEL == "1E2Mu":
            sf_trig = 1. - (1. - evt.weights_trigs_dblmu[1]) * (1. - evt.weights_trigs_emu[1])
        elif CHANNEL == "3Mu":
            sf_trig = evt.weights_trigs_dblmu[1]
        else:
            raise(AttributeError)
    elif wsyst == "TrigSFDown":
        if CHANNEL == "1E2Mu":
            sf_trig = 1. - (1. - evt.weights_trigs_dblmu[2]) * (1. - evt.weights_trigs_emu[2])
        elif CHANNEL == "3Mu":
            sf_trig = evt.weights_trigs_dblmu[2]
        else:
            raise(AttributeError)
    else:
        pass

    weight *= w_L1Prefire
    weight *= w_pileup
    weight *= sf_id
    weight *= sf_trig
    weight *= sf_btag

    return weight


def scale_or_smear(muons, electrons, jets, bjets, syst):
    if syst == "ElectronEnUp":
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
        pass

