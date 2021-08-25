import argparse
import numpy as np
from itertools import combinations
from ROOT import TFile, TH1D
from Scripts.DataFormat import Particle, get_leptons, get_jets
from Scripts.DataFormat import scale_electrons, smear_electrons, scale_muons, scale_jets, smear_jets
from Scripts.HistTools import HistTool

parser = argparse.ArgumentParser()
parser.add_argument("--sample", "-s", default=None, required=True,
                    type=str, help="sample name")
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help="channel")
args = parser.parse_args()

# define global variables
SAMPLE = args.sample
if args.channel == "1E2Mu":
    DATA = ["MuonEG", "DoubleMuon"]
    BKGs = ["rare", "ttX", "VV", "fake", "conv"]
    MCs = ["DY", "ZG", "rare", "ttX", "VV"]
    SAMPLE_DIR = "/Users/choij/workspace/HcToWA/Samples/Selector/2017/Skim1E2Mu__/"
    OUTFILE = f"/Users/choij/workspace/HcToWA/ZFakeRegion/Outputs/{args.channel}/{SAMPLE}.root"
elif args.channel == "3Mu":
    DATA = ["DoubleMuon"]
    BKGs = ["rake", "ttX", "VV", "fake"]
    MCs = ["DY", "ZG", "rare", "ttX", "VV"]
    SAMPLE_DIR = "/Users/choij/workspace/HcToWA/Samples/Selector/2017/Skim3Mu__/"
    OUTFILE = f"/Users/choij/workspace/HcToWA/ZFakeRegion/Outputs/{args.channel}/{SAMPLE}.root"
else:
    print(f"Wrong channel {args.channel}")
    raise(AttributeError)

SYSTs = ["Central", "ElectronEnUp", "ElectronEnDown", "ElectronResUp", "ElectronResDown",
         "MuonEnUp", "MuonEnDown", "JetEnUp", "JetEnDown", "JetResUp", "JetResDown"]
WEIGHTSYSTs = ["L1PrefireUp", "L1PrefireDown", "PUReweightUp", "PUReweightDown", "TrigSFUp", "TrigSFDown"]

# get fake rate and pileup reweight
f_muon = TFile.Open(
    "/Users/choij/workspace/HcToWA/MetaInfo/2017/fakerate_muon.root")
f_electron = TFile.Open(
    "/Users/choij/workspace/HcToWA/MetaInfo/2017/fakerate_electron.root")
f_pileup = TFile.Open(
    "/Users/choij/workspace/HcToWA/MetaInfo/2017/PUReweight_2017.root")

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

htool = HistTool(outfile=OUTFILE)

def select(evt, muons, electrons, jets, bjets, METv):
    # trigger
    if args.channel == "1E2Mu":
        # Event selection
        # 1. Should pass triggers and safe cut
        # 2. exist OS muon pair
        # 3. N(b) = 0
        # 4. |M(l+l-) - 91.2| < 10 GeV
        # 5. Veto M(l+l-) < 12 GeV
        # 6. For DY, at least one lepton pt < 20 GeV
        #    For ZG, pt of all leptons > 20 GeV
        if not (evt.passDblMuTrigs or evt.passEMuTrigs):
            return False

        pass_safecut = ((muons[0].Pt() > 20. and muons[1].Pt() > 10.) or
                    (muons[0].Pt() > 25. and electrons[0].Pt() > 15.) or
                    (electrons[0].Pt() > 25. and muons[0].Pt() > 10.))
        if not pass_safecut:
            return False

        # check pt condition
        if args.sample == "DY":
            if (muons[0].Pt() > 20. and muons[1].Pt() > 20. and electrons[0].Pt() > 20.):
                return False
        if args.sample == "ZG":
            if (muons[0].Pt() < 20. or muons[1].Pt() < 20. or electrons[0].Pt() < 20.):
                return False

        # check OS pair
        if not muons[0].Charge() + muons[1].Charge() == 0:
            return False

        # mass condition
        ZCand = muons[0] + muons[1]

        if not abs(ZCand.Mass() - 91.2) < 10.:
            return False
        
        if not ZCand.Mass() > 12.:
            return False

        # no bjet
        if not len(bjets) == 0:
            return False

    elif args.channel== "3Mu":
        # Event selection
        # 1. Should pass triggers and safe cut
        # 2. exist OS muon pair
        # 3. N(b) = 0
        # 4. exist |M(ZCand) - 91.2| < 10 GeV
        # 5. veto M(l+l-) < 12 GeV
        # 6. For DY, at least one lepton pt < 20 GeV
        #    For ZG, pt of all leptons > 20 GeV
        if not evt.passDblMuTrigs:
            return False
        
        pass_safecut = muons[0].Pt() > 20. and muons[1].Pt() > 10.
        if not pass_safecut:
            return False

        # check pt condition
        if args.sample == "DY":
            if (muons[0].Pt() > 20. and muons[1].Pt() > 20. and muons[2].Pt() > 20.):
                return False
        if args.sample == "ZG":
            if (muons[0].Pt() < 20. or muons[1].Pt() < 20. or muons[2].Pt() < 20.):
                return False

        # check os pair
        if not abs(muons[0].Charge() + muons[1].Charge() + muons[2].Charge()) == 1:
            return False

        # mass condition
        ZCand1, ZCand2 = make_ospairs(muons)

        if not (abs(ZCand1.Mass() - 91.2) < 10. or abs(ZCand2.Mass() - 91.2) < 10.):
            return False

        if not (ZCand1.Mass() > 12. and ZCand2.Mass() > 12.):
            return False

        # no bjet
        if not len(bjets) == 0:
            return False

    else:
        print("wrong channel")
        raise(AttributeError)

    return True

def make_ospairs(muons):
    nMuons = len(muons)
    comb = combinations(range(nMuons), 2)
    zcands = []
    for elements in comb:
        mu1 = muons[elements[0]]
        mu2 = muons[elements[1]]
        if mu1.Charge() + mu2.Charge() == 0:
            cand = mu1 + mu2
            zcands.append(cand)

    return zcands[0], zcands[1]


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

def get_weight(evt, wsyst):
    weight = 1.
    weight *= evt.genWeight
    weight *= evt.trigLumi

    # weight variation
    w_L1Prefire = evt.weights_L1Prefire[0]
    w_pileup = h_pileup.GetBinContent(h_pileup.FindBin(evt.nPileUp))
    sf_id = evt.weights_id[0]
    if args.channel == "1E2Mu":
        sf_trig = 1. - (1. - evt.weights_trigs_dblmu[0]) * (1. - evt.weights_trigs_emu[0])
    elif args.channel == "3Mu":
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
        if args.channel == "1E2Mu":
            sf_trig = 1. - (1. - evt.weights_trigs_dblmu[1]) * (1. - evt.weights_trigs_emu[1])
        elif args.channel == "3Mu":
            sf_trig = evt.weights_trigs_dblmu[1]
        else:
            raise(AttributeError)
    elif wsyst == "TrigSFDown":
        if args.channel == "1E2Mu":
            sf_trig = 1. - (1. - evt.weights_trigs_dblmu[2]) * (1. - evt.weights_trigs_emu[2])
        elif args.channel == "3Mu":
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

# Temporarily set 1E2Mu channel first
def loop(evt, syst, htool):
    muons, electrons = get_leptons(evt)
    jets, bjets = get_jets(evt)
    METv = Particle(evt.METv_pt, evt.METv_eta, evt.METv_phi, 0.)
    scale_or_smear(muons, electrons, jets, bjets, syst)

    # event selection
    if not select(evt, muons, electrons, jets, bjets, syst):
        return

    if args.channel == "1E2Mu":
        ZCand = muons[0] + muons[1]
    elif args.channel == "3Mu":
        ZCand1, ZCand2 = make_ospairs(muons)
        if abs(ZCand1.Mass() - 91.2) < abs(ZCand2.Mass() - 91.1):
            ZCand = ZCand1
        else:
            ZCand = ZCand2
    else:
        raise(AttributeError)

    # check if pass tight IDs
    muons_tight, electrons_tight = get_tight_leptons(muons, electrons)
    
    # fill
    tight_flag = False
    if args.channel == "1E2Mu":
        if len(muons_tight) == 2 and len(electrons_tight) == 1:
            tight_flag = True
    elif args.channel == "3Mu":
        if len(muons_tight) == 3 and len(electrons_tight) == 0:
            tight_flag = True
    else:
        raise(AttributeError)

    if tight_flag:
        # set weight
        weight = 1.
        if not evt.IsDATA:
            weight = get_weight(evt, syst)
        if SAMPLE in ["DY", "ZG"]:
            muons_prompt, electrons_prompt = get_prompt_leptons(muons, electrons)
            
            prompt_flag = False
            if args.channel == "1E2Mu":
                if (len(muons_prompt) == 2 and len(electrons_prompt) == 1):
                    prompt_flag = True
            elif args.channel == "3Mu":
                if (len(muons_prompt) == 3 and len(electrons_prompt) == 0):
                    prompt_flag = True
            else:
                raise(AttributeError)

            if not prompt_flag:
                return


        htool.fill_object(f"{SAMPLE}/{syst}/ZCand", ZCand, weight)
        htool.fill_object(f"{SAMPLE}/{syst}/METv", METv, weight)
        htool.fill_muons(f"{SAMPLE}/{syst}/muons", muons, weight)
        htool.fill_electrons(f"{SAMPLE}/{syst}/electrons", electrons, weight)
        htool.fill_jets(f"{SAMPLE}/{syst}/jets", jets, weight)
    else:
        if not evt.IsDATA:
            return
        if not syst == "Central":
            return
        # estimate fake contribution
        w_fake, w_fake_up, w_fake_down = get_fake_weights(muons, electrons)
        htool.fill_object(f"fake/Central/ZCand", ZCand, w_fake)
        htool.fill_object(f"fake/Central/METv", METv, w_fake)
        htool.fill_muons(f"fake/Central/muons", muons, w_fake)
        htool.fill_electrons(f"fake/Central/electrons", electrons, w_fake)
        htool.fill_jets(f"fake/Central/jets", jets, w_fake)
        htool.fill_object(f"fake/Up/ZCand", ZCand, w_fake_up)
        htool.fill_object(f"fake/Up/METv", METv, w_fake_up)
        htool.fill_muons(f"fake/Up/muons", muons, w_fake_up)
        htool.fill_electrons(f"fake/Up/electrons", electrons, w_fake_up)
        htool.fill_jets(f"fake/Up/jets", jets, w_fake_up)
        htool.fill_object(f"fake/Down/ZCand", ZCand, w_fake_down)
        htool.fill_object(f"fake/Down/METv", METv, w_fake_down)
        htool.fill_muons(f"fake/Down/muons", muons, w_fake_down)
        htool.fill_electrons(f"fake/Down/electrons", electrons, w_fake_down)
        htool.fill_jets(f"fake/Down/jets", jets, w_fake_down)

        

def main():
    print(f"Estimating {SAMPLE}...")
    if SAMPLE == "DATA" and args.channel == "1E2Mu":
        fkey_dblmu = f"{SAMPLE_DIR}/DATA/Selector_DoubleMuon.root"
        fkey_emu = f"{SAMPLE_DIR}/DATA/Selector_MuonEG.root"
        
        events = dict()
        for syst in SYSTs + WEIGHTSYSTs:
            events[syst] = list()

        f = TFile.Open(fkey_dblmu)
        for syst in SYSTs + WEIGHTSYSTs:
            for evt in f.Events:
                # store events
                this_evt = (evt.run, evt.event, evt.lumi)
                events[syst].append(this_evt)

                loop(evt, syst, htool)
        f.Close()
        f = TFile.Open(fkey_emu)
        for syst in SYSTs + WEIGHTSYSTs:
            for evt in f.Events:
                # check double counting
                this_evt = (evt.run, evt.event, evt.lumi)
                if this_evt in events[syst]:
                    continue

                loop(evt, syst, htool)
    elif SAMPLE == "DATA" and args.channel == "3Mu":
        fkey = f"{SAMPLE_DIR}/DATA/Selector_DoubleMuon.root"
        f = TFile.Open(fkey)
        for syst in SYSTs + WEIGHTSYSTs:
            for evt in f.Events:
                loop(evt, syst, htool)
        f.Close()

    elif SAMPLE in MCs:
        fkey = f"{SAMPLE_DIR}/MCSamples/Selector_{SAMPLE}.root"
        f = TFile.Open(fkey)
        for syst in SYSTs + WEIGHTSYSTs:
            for evt in f.Events:
                loop(evt, syst, htool)
        f.Close()
    else:
        raise AttributeError
    
    htool.save()

if __name__ == "__main__":
    main()
    print("End prosess")
