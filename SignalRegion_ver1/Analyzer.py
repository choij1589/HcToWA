import argparse
import numpy as np
import pandas as pd
from itertools import combinations
from ROOT import TFile, TH1D
from ROOT import EnableImplicitMT
EnableImplicitMT(20)

import torch
from sklearn.preprocessing import MinMaxScaler

from Scripts.DataFormat import Particle, get_leptons, get_jets
from Scripts.DataFormat import scale_electrons, smear_electrons, scale_muons, scale_jets, smear_jets
from Scripts.HistTools import HistTool
from SignalTools import set_global, select, get_tight_leptons, get_prompt_leptons 
from SignalTools import get_weight, get_fake_weights, make_ospairs, make_mumujjpairs, scale_or_smear
from MLTools import DEVICE
from MLTools import DNN

parser = argparse.ArgumentParser()
parser.add_argument("--sample", "-s", default=None, required=True, type=str, help="sample name")
parser.add_argument("--mass", "-m", default=None, required=True, type=str, help="signal mass point")
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help="channel")
args = parser.parse_args()

# define global variables
CHANNEL = args.channel
SAMPLE = args.sample
set_global(CHANNEL, SAMPLE)
mHc = float(args.mass.split("_")[0][3:])
mA = float(args.mass.split("_")[1][2:])
if args.channel == "1E2Mu":
    DATA = ["MuonEG", "DoubleMuon"]
    BKGs = ["rare", "ttX", "VV", "fake", "conv"]
    MCs = ["DY", "ZG", "rare", "ttX", "VV"]
    SAMPLE_DIR = "/root/workspace/HcToWA/Samples/Selector/2017/Skim1E2Mu__/"
    OUTFILE = f"/root/workspace/HcToWA/SignalRegion/Outputs/{args.channel}/{args.mass}/{SAMPLE}.root"
elif args.channel == "3Mu":
    DATA = ["DoubleMuon"]
    BKGs = ["rake", "ttX", "VV", "fake"]
    MCs = ["DY", "ZG", "rare", "ttX", "VV"]
    SAMPLE_DIR = "/root/workspace/HcToWA/Samples/Selector/2017/Skim3Mu__/"
    OUTFILE = f"/root/workspace/HcToWA/SignalRegion/Outputs/{args.channel}/{args.mass}/{SAMPLE}.root"
else:
    print(f"Wrong channel {args.channel}")
    raise(AttributeError)

# SYSTs = ["Central", "ElectronEnUp", "ElectronEnDown", "ElectronResUp", "ElectronResDown",
#         "MuonEnUp", "MuonEnDown", "JetEnUp", "JetEnDown", "JetResUp", "JetResDown"]
# WEIGHTSYSTs = ["L1PrefireUp", "L1PrefireDown", "PUReweightUp", "PUReweightDown", "TrigSFUp", "TrigSFDown"]
SYSTs = ["Central"]
WEIGHTSYSTs = []

htool = HistTool(outfile=OUTFILE)

# make input scaler
SIGNAL = f"TTToHcToWA_AToMuMu_{args.mass}"
signal = pd.read_csv(f"Outputs/{CHANNEL}/CSV/{SIGNAL}.csv", index_col=0)
fake = pd.read_csv(f"Outputs/{CHANNEL}/CSV/fake.csv", index_col=0)
rare = pd.read_csv(f"Outputs/{CHANNEL}/CSV/rare.csv", index_col=0)
VV = pd.read_csv(f"Outputs/{CHANNEL}/CSV/VV.csv", index_col=0)
ttX = pd.read_csv(f"Outputs/{CHANNEL}/CSV/ttX.csv", index_col=0)
DY = pd.read_csv(f"Outputs/{CHANNEL}/CSV/DY.csv", index_col=0)
ZG = pd.read_csv(f"Outputs/{CHANNEL}/CSV/ZG.csv", index_col=0)

signal['label'] = 0
fake['label'] = 1
rare['label'] = 1
VV['label'] = 1
ttX['label'] = 1
DY['label'] = 1
ZG['label'] = 1
sample = pd.concat([signal, fake, rare, VV, ttX, DY, ZG])

scaler = MinMaxScaler()
if args.channel == "1E2Mu":
    features = sample[['mu1_px', 'mu1_py', 'mu1_pz', 'mu1_mass', 'mu1_charge',
                'mu2_px', 'mu2_py', 'mu2_pz', 'mu2_mass', 'mu2_charge',
                'ele_px', 'ele_py', 'ele_pz', 'ele_mass', 'ele_charge',
                'j1_px', 'j1_py', 'j1_pz', 'j1_mass', 'j1_bscore',
                'j2_px', 'j2_py', 'j2_pz', 'j2_mass', 'j2_bscore',
                'dR_mu1mu2', 'dR_j1j2', 'dR_j1ele', 'dR_j2ele',
                'HT', 'MT', 'Nj', 'Nb', 'LT+MET',
                'avg_dRjets', 'avg_bscore']].to_numpy()
elif args.channel == "3Mu":
    features = sample[['MA1', 'MA2', 'Mlll', 'mu1_pt', 'mu2_pt', 'mu3_pt', 'j1_pt', 'j2_pt', 'j1_btagScore', 'j2_btagScore', 'Nj', 'Nb']].to_numpy()

scaler.fit(features)

# Temporarily set 1E2Mu channel first
def loop(evt, syst, htool):
    muons, electrons = get_leptons(evt)
    jets, bjets = get_jets(evt)
    METv = Particle(evt.METv_pt, evt.METv_eta, evt.METv_phi, 0.)
    scale_or_smear(muons, electrons, jets, bjets, syst)

    # event selection
    if not select(evt, muons, electrons, jets, bjets, syst):
        return

    # divide signal and control region
    # Signal Region: OS muon pair wiht M(SFOS) > 12 GeV
    # Control Region: SS muon pair + OS electron
    if CHANNEL == "1E2Mu":
        if muons[0].Charge() + muons[1].Charge() == 0:
            REGION = "SR"
        elif muons[0].Charge() + electrons[0].Charge() == 0:
            REGION = "CR"
        else:
            return

        ACand = muons[0] + muons[1]
        if not ACand.Mass() > 12.:
            return

    elif CHANNEL == "3Mu":
        ACand1, ACand2 = make_ospairs(muons)
        if abs(ACand1.Mass() - mA) < abs(ACand2.Mass() - mA):
            ACand = ACand1
        else:
            ACand = ACand2
    else:
        raise(AttributeError)

    # make feature tensor
    if args.channel == "1E2Mu":
        mu1_px = muons[0].P4().Px()
        mu1_py = muons[0].P4().Py()
        mu1_pz = muons[0].P4().Pz()
        mu1_mass = muons[0].Mass()
        mu1_charge = muons[0].Charge()
        mu2_px = muons[1].P4().Px()
        mu2_py = muons[1].P4().Py()
        mu2_pz = muons[1].P4().Pz()
        mu2_mass = muons[1].Mass()
        mu2_charge = muons[1].Charge()
        ele_px = electrons[0].P4().Px()
        ele_py = electrons[0].P4().Py()
        ele_pz = electrons[0].P4().Pz()
        ele_mass = electrons[0].Mass()
        ele_charge = electrons[0].Charge()
        j1_px = jets[0].P4().Px()
        j1_py = jets[0].P4().Py()
        j1_pz = jets[0].P4().Pz()
        j1_mass = jets[0].Mass()
        j1_bscore = jets[0].BtagScore()
        j2_px = jets[1].P4().Px()
        j2_py = jets[1].P4().Py()
        j2_pz = jets[1].P4().Pz()
        j2_mass = jets[1].Mass()
        j2_bscore = jets[1].BtagScore()
        dR_mu1mu2 = muons[0].DeltaR(muons[1])
        dR_j1j2 = jets[0].DeltaR(jets[1])
        dR_j1ele = jets[0].DeltaR(electrons[0])
        dR_j2ele = jets[1].DeltaR(electrons[0])
        HT = sum([x.Pt() for x in jets])
        MT = np.sqrt(2*electrons[0].Pt()*METv.Pt()*(1. - np.cos(electrons[0].DeltaPhi(METv))))
        Nj = len(jets)
        Nb = len(bjets)
        LTpMET = muons[0].Pt() + muons[1].Pt() + electrons[0].Pt() + METv.Pt()
        comb = combinations(range(len(jets)), 2)
        dRjets = []
        for elements in comb:
            dRjets.append(jets[elements[0]].DeltaR(jets[elements[1]]))
        avg_dRjets = sum(dRjets) / len(dRjets)
        avg_bscore = sum([x.BtagScore() for x in jets]) / len(jets)

        features = np.array([[mu1_px, mu1_py, mu1_pz, mu1_mass, mu1_charge, mu2_px, mu2_py, mu2_pz, mu2_mass, mu2_charge, ele_px, ele_py, ele_pz, ele_mass, ele_charge, j1_px, j1_py, j1_pz, j1_mass, j1_bscore, j2_px, j2_py, j2_pz, j2_mass, j2_bscore, dR_mu1mu2, dR_j1j2, dR_j1ele, dR_j2ele, HT, MT, Nj, Nb, LTpMET, avg_dRjets, avg_bscore]])
    elif args.channel == "3Mu":
        print("3Mu channel is not prepared yet")
        exit()
    features = scaler.transform(features)
    features = torch.FloatTensor(features)

    # get model
    model = DNN(len(features[0]), 2)
    model.load_state_dict(torch.load(f"models/{CHANNEL}/{SIGNAL}.pt"))
    model.eval()

    # get score
    output = model(features).view(2)
    score = output[0].detach().numpy()

    # check if pass tight IDs
    muons_tight, electrons_tight = get_tight_leptons(muons, electrons)

    # fill
    tight_flag = False
    if CHANNEL == "1E2Mu":
        if len(muons_tight) == 2 and len(electrons_tight) == 1:
            tight_flag = True
    elif CHANNEL == "3Mu":
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
            if CHANNEL == "1E2Mu":
                if (len(muons_prompt) == 2 and len(electrons_prompt) == 1):
                    prompt_flag = True
            elif CHANNEL == "3Mu":
                if (len(muons_prompt) == 3 and len(electrons_prompt) == 0):
                    prompt_flag = True
            else:
                raise(AttributeError)

            if not prompt_flag:
                return

        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mA", ACand.Mass(), weight, 3000, 0., 300.)
        htool.fill_object(f"{SAMPLE}/{REGION}/{syst}/ACand", ACand, weight)
        htool.fill_object(f"{SAMPLE}/{REGION}/{syst}/METv", METv, weight)
        htool.fill_muons(f"{SAMPLE}/{REGION}/{syst}/muons", muons, weight)
        htool.fill_electrons(f"{SAMPLE}/{REGION}/{syst}/electrons", electrons, weight)
        htool.fill_jets(f"{SAMPLE}/{REGION}/{syst}/jets", jets, weight)
        htool.fill_jets(f"{SAMPLE}/{REGION}/{syst}/bjets", bjets, weight)
        
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mu1_px", mu1_px, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mu1_py", mu1_py, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mu1_pz", mu1_pz, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mu2_px", mu1_px, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mu2_py", mu2_py, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mu2_pz", mu2_pz, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/ele_px", ele_px, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/ele_py", ele_py, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/ele_pz", ele_pz, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j1_px", j1_px, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j1_py", j1_py, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j1_pz", j1_pz, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j1_mass", j1_mass, weight, 200, 0., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j1_bscore", j1_bscore, weight, 100, 0., 1.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j2_px", j2_px, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j2_py", j2_py, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j2_pz", j2_pz, weight, 400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j2_mass", j2_mass, weight, 200, 0., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j2_bscore", j2_bscore, weight, 100, 0., 1.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/dR_mu1mu2", dR_mu1mu2, weight, 100, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/dR_j1j2", dR_j1j2, weight, 100, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/dR_j1ele", dR_j1ele, weight, 100, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/dR_j2ele", dR_j2ele, weight, 100, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/HT", HT, weight, 1000, 0., 1000.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/MT", MT, weight, 300, 0., 300.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/Nj", Nj, weight, 20, 0., 20.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/Nb", Nb, weight, 10, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/LTpMET", LTpMET, weight, 500, 0., 500.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/avg_dRjets", avg_dRjets, weight, 100, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/avg_bscore", avg_bscore, weight, 100, 0., 1.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/score", score, weight, 100, 0., 1.)
        htool.fill_hist2d(f"{SAMPLE}/{REGION}/{syst}/mA_score", ACand.Mass(), score, weight, 3000, 0., 300., 100, 0., 1.)

    else:
        if (not evt.IsDATA) or (not syst == "Central"):
            return
        # estimate fake contribution
        w_fake, w_fake_up, w_fake_down = get_fake_weights(muons, electrons)
        htool.fill_hist(f"fake/{REGION}/Central/mA", ACand.Mass(), w_fake, 3000, 0., 300.)
        htool.fill_hist(f"fake/{REGION}/Central/MT", MT, w_fake, 300, 0., 300.)
        htool.fill_object(f"fake/{REGION}/Central/ACand", ACand, w_fake)
        htool.fill_object(f"fake/{REGION}/Central/METv", METv, w_fake)
        htool.fill_muons(f"fake/{REGION}/Central/muons", muons, w_fake)
        htool.fill_electrons(f"fake/{REGION}/Central/electrons", electrons, w_fake)
        htool.fill_jets(f"fake/{REGION}/Central/jets", jets, w_fake)
        htool.fill_jets(f"fake/{REGION}/Central/bjets", bjets, w_fake)

        htool.fill_hist(f"fake/{REGION}/Central/mu1_px", mu1_px, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/mu1_py", mu1_py, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/mu1_pz", mu1_pz, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/mu2_px", mu1_px, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/mu2_py", mu2_py, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/mu2_pz", mu2_pz, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/ele_px", ele_px, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/ele_py", ele_py, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/ele_pz", ele_pz, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j1_px", j1_px, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j1_py", j1_py, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j1_pz", j1_pz, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j1_mass", j1_mass, w_fake, 200, 0., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j1_bscore", j1_bscore, w_fake, 100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Central/j2_px", j2_px, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j2_py", j2_py, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j2_pz", j2_pz, w_fake, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j2_mass", j2_mass, w_fake, 200, 0., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j2_bscore", j2_bscore, w_fake, 100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Central/dR_mu1mu2", dR_mu1mu2, w_fake, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/dR_j1j2", dR_j1j2, w_fake, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/dR_j1ele", dR_j1ele, w_fake, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/dR_j2ele", dR_j2ele, w_fake, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/HT", HT, w_fake, 1000, 0., 1000.)
        htool.fill_hist(f"fake/{REGION}/Central/MT", MT, w_fake, 300, 0., 300.)
        htool.fill_hist(f"fake/{REGION}/Central/Nj", Nj, w_fake, 20, 0., 20.)
        htool.fill_hist(f"fake/{REGION}/Central/Nb", Nb, w_fake, 10, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/LTpMET", LTpMET, w_fake, 500, 0., 500.)
        htool.fill_hist(f"fake/{REGION}/Central/avg_dRjets", avg_dRjets, w_fake, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/avg_bscore", avg_bscore, w_fake, 100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Central/score", score, w_fake, 100, 0., 1.)
        htool.fill_hist2d(f"fake/{REGION}/Central/mA_score", ACand.Mass(), score, w_fake, 3000, 0., 300., 100, 0., 1.)

        htool.fill_hist(f"fake/{REGION}/Up/mA", ACand.Mass(), w_fake_up, 3000, 0., 300.)
        htool.fill_hist(f"fake/{REGION}/Up/MT", MT, w_fake_up, 300, 0., 300.)
        htool.fill_object(f"fake/{REGION}/Up/ACand", ACand, w_fake_up)
        htool.fill_object(f"fake/{REGION}/Up/METv", METv, w_fake_up)
        htool.fill_muons(f"fake/{REGION}/Up/muons", muons, w_fake_up)
        htool.fill_electrons(f"fake/{REGION}/Up/electrons", electrons, w_fake_up)
        htool.fill_jets(f"fake/{REGION}/Up/jets", jets, w_fake_up)
        htool.fill_jets(f"fake/{REGION}/Up/bjets", bjets, w_fake_up)

        htool.fill_hist(f"fake/{REGION}/Up/mu1_px", mu1_px, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/mu1_py", mu1_py, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/mu1_pz", mu1_pz, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/mu2_px", mu1_px, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/mu2_py", mu2_py, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/mu2_pz", mu2_pz, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/ele_px", ele_px, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/ele_py", ele_py, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/ele_pz", ele_pz, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j1_px", j1_px, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j1_py", j1_py, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j1_pz", j1_pz, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j1_mass", j1_mass, w_fake_up, 200, 0., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j1_bscore", j1_bscore, w_fake_up, 100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Up/j2_px", j2_px, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j2_py", j2_py, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j2_pz", j2_pz, w_fake_up, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j2_mass", j2_mass, w_fake_up, 200, 0., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j2_bscore", j2_bscore, w_fake_up, 100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Up/dR_mu1mu2", dR_mu1mu2, w_fake_up, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/dR_j1j2", dR_j1j2, w_fake_up, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/dR_j1ele", dR_j1ele, w_fake_up, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/dR_j2ele", dR_j2ele, w_fake_up, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/HT", HT, w_fake_up, 1000, 0., 1000.)
        htool.fill_hist(f"fake/{REGION}/Up/MT", MT, w_fake_up, 300, 0., 300.)
        htool.fill_hist(f"fake/{REGION}/Up/Nj", Nj, w_fake_up, 20, 0., 20.)
        htool.fill_hist(f"fake/{REGION}/Up/Nb", Nb, w_fake_up, 10, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/LTpMET", LTpMET, w_fake_up, 500, 0., 500.)
        htool.fill_hist(f"fake/{REGION}/Up/avg_dRjets", avg_dRjets, w_fake_up, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/avg_bscore", avg_bscore, w_fake_up, 100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Up/score", score, w_fake_up, 100, 0., 1.)
        htool.fill_hist2d(f"fake/{REGION}/Up/mA_score", ACand.Mass(), score, w_fake_up, 3000, 0., 300., 100, 0., 1.)

        htool.fill_hist(f"fake/{REGION}/Down/mA", ACand.Mass(), w_fake_down, 3000, 0., 300.)
        htool.fill_hist(f"fake/{REGION}/Down/MT", MT, w_fake_down, 300, 0., 300.)
        htool.fill_object(f"fake/{REGION}/Down/ACand", ACand, w_fake_down)
        htool.fill_object(f"fake/{REGION}/Down/METv", METv, w_fake_down)
        htool.fill_muons(f"fake/{REGION}/Down/muons", muons, w_fake_down)
        htool.fill_electrons(f"fake/{REGION}/Down/electrons", electrons, w_fake_down)
        htool.fill_jets(f"fake/{REGION}/Down/jets", jets, w_fake_down)
        htool.fill_jets(f"fake/{REGION}/Down/bjets", bjets, w_fake_down)
        htool.fill_hist(f"fake/{REGION}/Down/score", score, w_fake_down, 100, 0., 1.)

        htool.fill_hist(f"fake/{REGION}/Down/mu1_px", mu1_px, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/mu1_py", mu1_py, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/mu1_pz", mu1_pz, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/mu2_px", mu1_px, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/mu2_py", mu2_py, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/mu2_pz", mu2_pz, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/ele_px", ele_px, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/ele_py", ele_py, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/ele_pz", ele_pz, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j1_px", j1_px, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j1_py", j1_py, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j1_pz", j1_pz, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j1_mass", j1_mass, w_fake_down, 200, 0., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j1_bscore", j1_bscore, w_fake_down, 100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Down/j2_px", j2_px, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j2_py", j2_py, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j2_pz", j2_pz, w_fake_down, 400, -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j2_mass", j2_mass, w_fake_down, 200, 0., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j2_bscore", j2_bscore, w_fake_down, 100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Down/dR_mu1mu2", dR_mu1mu2, w_fake_down, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/dR_j1j2", dR_j1j2, w_fake_down, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/dR_j1ele", dR_j1ele, w_fake_down, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/dR_j2ele", dR_j2ele, w_fake_down, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/HT", HT, w_fake_down, 1000, 0., 1000.)
        htool.fill_hist(f"fake/{REGION}/Down/MT", MT, w_fake_down, 300, 0., 300.)
        htool.fill_hist(f"fake/{REGION}/Down/Nj", Nj, w_fake_down, 20, 0., 20.)
        htool.fill_hist(f"fake/{REGION}/Down/Nb", Nb, w_fake_down, 10, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/LTpMET", LTpMET, w_fake_down, 500, 0., 500.)
        htool.fill_hist(f"fake/{REGION}/Down/avg_dRjets", avg_dRjets, w_fake_down, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/avg_bscore", avg_bscore, w_fake_down, 100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Down/score", score, w_fake_down, 100, 0., 1.)
        htool.fill_hist2d(f"fake/{REGION}/Down/mA_score", ACand.Mass(), score, w_fake_down, 3000, 0., 300., 100, 0., 1.)


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

    elif SAMPLE in MCs or "TTToHcToWA" in SAMPLE:
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
