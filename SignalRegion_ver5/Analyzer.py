import argparse
import numpy as np
import pandas as pd
from itertools import combinations
import pickle
from ROOT import TFile

import torch
import torch.nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from Scripts.DataFormat import Particle, get_leptons, get_jets
from Scripts.HistTools import HistTool
from SignalTools import set_global
from SignalTools import get_tight_leptons, get_prompt_leptons
from SignalTools import get_weight, get_fake_weights

parser = argparse.ArgumentParser()
parser.add_argument("--sample", "-s", default=None, required=True, type=str, help="sample name")
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help="channel")
args = parser.parse_args()

# define global variables
CHANNEL = args.channel
SAMPLE = args.sample
MASS_POINTs = [
    "MHc70_MA15", "MHc70_MA40", "MHc70_MA65",
    "MHc100_MA15", "MHc100_MA25", "MHc100_MA60", "MHc100_MA95",
    "MHc130_MA15", "MHc130_MA45", "MHc130_MA55", "MHc130_MA90", "MHc130_MA125",
    "MHc160_MA15", "MHc160_MA45", "MHc160_MA75", "MHc160_MA85", "MHc160_MA120", "MHc160_MA155"
]
set_global(CHANNEL, SAMPLE)

if CHANNEL == "1E2Mu":
    DATA = ["MuonEG", "DoubleMuon"]
    BKGs = ["rare", "ttX", "VV", "fake", "conv"]
    MCs = ["DY", "ZG", "rare", "ttX", "VV"]
    SAMPLE_DIR = "../Samples/Selector/2017/Skim1E2Mu__/"
    OUTFILE = f"Outputs/{CHANNEL}/{SAMPLE}.root"
    features = [
        'mu1_px', 'mu1_py', 'mu1_pz', 'mu1_mass', 'mu1_charge', 'mu2_px',
        'mu2_py', 'mu2_pz', 'mu2_mass', 'mu2_charge', 'ele_px', 'ele_py',
        'ele_pz', 'ele_mass', 'j1_px', 'j1_py', 'j1_pz', 'j1_mass',
        'j1_bscore', 'j2_px', 'j2_py', 'j2_pz', 'j2_mass', 'j2_bscore',
        'dR_mu1mu2', 'dR_mu1ele', 'dR_mu2ele', 'dR_j1ele', 'dR_j2ele',
        'dR_j1j2', 'HT', 'MT', 'LT', 'MET', 'Nj', 'Nb', 'avg_dRjets',
        'avg_bscore'
    ]
elif CHANNEL == "3Mu":
    DATA = ["MuonEG"]
    BKGs = ["rare", "ttX", "VV", "fake", "conv"]
    MCs = ["DY", "ZG", "rare", "ttX", "VV"]
    SAMPLE_DIR = "../Samples/Selector/2017/Skim3Mu__/"
    OUTFILE = f"Outputs/{CHANNEL}/{SAMPLE}.root"
    features = [
        'mu1_px', 'mu1_py', 'mu1_pz', 'mu1_mass', 'mu1_charge', 'mu2_px',
        'mu2_py', 'mu2_pz', 'mu2_mass', 'mu2_charge', 'mu3_px', 'mu3_py',
        'mu3_pz', 'mu3_mass', 'mu3_charge', 'j1_px', 'j1_py', 'j1_pz',
        'j1_mass', 'j1_bscore', 'j2_px', 'j2_py', 'j2_pz', 'j2_mass',
        'j2_bscore', 'dR_mu1mu2', 'dR_mu1mu3', 'dR_mu2mu3', 'dR_j1j2',
        'dR_j1mu1', 'dR_j1mu2', 'dR_j1mu3', 'dR_j2mu1', 'dR_j2mu2', 'dR_j2mu3',
        'HT', 'Nj', 'Nb', 'LT', 'MET', 'avg_dRjets', 'avg_bscore'
    ]
else:
    print(f"discriminators for {CHANNEL} channel is not trained yet")
    raise(AttributeError)

# Save Hyperparametes

SYSTs = ["Central"]
WEIGHTSYSTs = []
htool = HistTool(outfile=OUTFILE)

# load scaler
scalers_fake = dict()
scalers_ttX = dict()
for mass_point in MASS_POINTs:
    scalers_fake[mass_point] = pickle.load(open(f"Outputs/{CHANNEL}/{mass_point}/models/scaler_fake.pkl", 'rb'))
    scalers_ttX[mass_point] = pickle.load(open(f"Outputs/{CHANNEL}/{mass_point}/models/scaler_ttX.pkl", 'rb'))

# load classifiers
# all the hyperparameters for each mass point have been optimized using grid search
# the values are stored in Outputs/{CHANNEL}/CSV/hyper_params.csv
from MLTools import SelfNormDNN

hyper_params = pd.read_csv(f"Outputs/{CHANNEL}/CSV/hyper_params.csv", index_col="mass_point")
clfs_fake = dict()
clfs_ttX = dict()
for mass_point in MASS_POINTs:
    lr_fake, n_hidden_fake = hyper_params.loc[mass_point, 'lr_fake'], hyper_params.loc[mass_point, 'n_hidden_fake']
    lr_ttX, n_hidden_ttX = hyper_params.loc[mass_point, 'lr_ttX'], hyper_params.loc[mass_point, 'n_hidden_ttX']
    clf_fake = SelfNormDNN(len(features), 2, n_hidden_fake)
    clf_ttX = SelfNormDNN(len(features), 2, n_hidden_ttX)
    clf_fake.load_state_dict(
        torch.load(
            f"Outputs/{CHANNEL}/{mass_point}/models/chkpoint_sig_vs_fake_lr-{lr_fake}_n_hidden-{n_hidden_fake}.pt",
		map_location=torch.device('cpu')
        )
    )
    clf_ttX.load_state_dict(
        torch.load(
            f"Outputs/{CHANNEL}/{mass_point}/models/chkpoint_sig_vs_ttX_lr-{lr_ttX}_n_hidden-{n_hidden_ttX}.pt",
		map_location=torch.device('cpu')
        )
    )
    clf_fake.eval()
    clf_ttX.eval()
    clfs_fake[mass_point] = clf_fake
    clfs_ttX[mass_point] = clf_ttX

# Selection
def select(evt, muons, electrons, jets, bjets, METv):
    """Event selection on trilep regions"""
    if CHANNEL == "1E2Mu":
        if not (evt.passDblMuTrigs or evt.passEMuTrigs):
            return False
        
        pass_safecut = ((muons[0].Pt() > 20. and muons[1].Pt() > 10.)
                        or (muons[0].Pt() > 25. and electrons[0].Pt() > 15.)
                        or (electrons[0].Pt() > 25. and muons[0].Pt() > 10.))
        if not pass_safecut:
            return False
        
        # check pt condition for the conversion samples
        if SAMPLE == "DY":
            if (muons[0].Pt() > 20. and muons[1].Pt() > 20. and electrons[0].Pt() > 20.):
                return False
        elif SAMPLE == "ZG":
            if (muons[0].Pt() < 20. or muons[1].Pt() < 20. or electrons[0].Pt() < 20.):
                return False
        else:
            pass
        
        # diMuon resonance
        diMuon = muons[0] + muons[1]
        if not (diMuon.Mass() > 12.):
            return False
        
        # jet conditions
        if not len(jets) >= 2:
            return False
        
        if len(bjets) >= 1:
            if muons[0].Charge() + muons[1].Charge() == 0:
                return "SR"
            elif muons[0].Charge() + electrons[0].Charge() == 0:
                return "TTFake"
            else:
                return False
        else:
            if (muons[0].Charge() + muons[1].Charge() == 0) and abs(diMuon.Mass() - 91.2) < 15.:
                return "ZFake"
            else:
                return False
        
    elif CHANNEL == "3Mu":
        print("Event selection for 3Mu channel is not defined yet")
        raise AttributeError
    else:
        raise AttributeError
    
def loop(evt, clfs_fake, clfs_ttX, syst, htool):
    muons, electrons = get_leptons(evt)
    leptons = muons + electrons
    jets, bjets = get_jets(evt)
    METv = Particle(evt.METv_pt, evt.METv_eta, evt.METv_phi, 0.)
    
    REGION = select(evt, muons, electrons, jets, bjets, syst)
    if not REGION:
        return
    
    # check if pass tight IDs
    muons_tight, electrons_tight = get_tight_leptons(muons, electrons)
    muons_prompt, electrons_prompt = get_prompt_leptons(muons, electrons)
    
    tight_flag = False
    prompt_flag = False
    
    if CHANNEL == "1E2Mu":
        if len(muons_tight) == 2 and len(electrons_tight) == 1:
            tight_flag = True
        if len(muons_prompt) == 2 and len(electrons_prompt) == 1:
            prompt_flag = True
        
        diMuon = muons[0] + muons[1]
        MT = np.sqrt(2 * electrons[0].Pt() * METv.Pt() * (1. - np.cos(electrons[0].DeltaPhi(METv))))
        
        # make input variables
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
        dR_mu1ele = muons[0].DeltaR(electrons[0])
        dR_mu2ele = muons[1].DeltaR(electrons[0])
        dR_j1j2 = jets[0].DeltaR(jets[1])
        dR_j1ele = jets[0].DeltaR(electrons[0])
        dR_j2ele = jets[1].DeltaR(electrons[0])
        comb = combinations(range(len(jets)), 2)

        dRjets = []
        for elements in comb:
            j1 = jets[elements[0]]
            j2 = jets[elements[1]]
            dRjets.append(j1.DeltaR(j2))
        avg_dRjets = sum(dRjets) / len(dRjets)
        avg_bscore = sum([x.BtagScore() for x in jets]) / len(jets)

        HT = sum([x.Pt() for x in jets])
        MT = np.sqrt(2 * electrons[0].Pt() * METv.Pt() *
                     (1. - np.cos(electrons[0].DeltaPhi(METv))))
        LT = sum([x.Pt() for x in leptons])
        MET = METv.Pt()
        Nj = len(jets)
        Nb = len(bjets)

        features = np.array([[
            mu1_px, mu1_py, mu1_pz, mu1_mass, mu1_charge, mu2_px, mu2_py,
            mu2_pz, mu2_mass, mu2_charge, ele_px, ele_py, ele_pz, ele_mass,
            j1_px, j1_py, j1_pz, j1_mass, j1_bscore, j2_px, j2_py, j2_pz,
            j2_mass, j2_bscore, dR_mu1mu2, dR_mu1ele, dR_mu2ele, dR_j1ele,
            dR_j2ele, dR_j1j2, HT, MT, LT, MET, Nj, Nb, avg_dRjets, avg_bscore
        ]])
    else:
        raise (AttributeError)
    
    scores_fake = dict()
    scores_ttX = dict()
    
    for mass_point in MASS_POINTs:
        clf_fake, scaler_fake = clfs_fake[mass_point], scalers_fake[mass_point]
        clf_ttX, scaler_ttX = clfs_ttX[mass_point], scalers_ttX[mass_point]
        
        score_fake = clf_fake(torch.FloatTensor(
            scaler_fake.transform(features))).view(2)[0].detach().numpy()
        score_ttX = clf_ttX(torch.FloatTensor(
            scaler_ttX.transform(features))).view(2)[0].detach().numpy()
        
        scores_fake[mass_point] = score_fake
        scores_ttX[mass_point] = score_ttX
        
    if tight_flag:
        weight = 1.
        if not evt.IsDATA:
            weight = get_weight(evt, syst)
        if SAMPLE in ["DY", "ZG"] and not prompt_flag:
            return

        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mMuMu", diMuon.Mass(),
                        weight, 3000, 0., 300.)
        htool.fill_object(f"{SAMPLE}/{REGION}/{syst}/diMuon", diMuon, weight)
        htool.fill_object(f"{SAMPLE}/{REGION}/{syst}/METv", METv, weight)
        htool.fill_muons(f"{SAMPLE}/{REGION}/{syst}/muons", muons, weight)
        htool.fill_electrons(f"{SAMPLE}/{REGION}/{syst}/electrons", electrons,
                             weight)
        htool.fill_jets(f"{SAMPLE}/{REGION}/{syst}/jets", jets, weight)
        htool.fill_jets(f"{SAMPLE}/{REGION}/{syst}/bjets", bjets, weight)

        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mu1_px", mu1_px, weight,
                        400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mu1_py", mu1_py, weight,
                        400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mu1_pz", mu1_pz, weight,
                        400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mu2_px", mu1_px, weight,
                        400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mu2_py", mu2_py, weight,
                        400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/mu2_pz", mu2_pz, weight,
                        400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/ele_px", ele_px, weight,
                        400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/ele_py", ele_py, weight,
                        400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/ele_pz", ele_pz, weight,
                        400, -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j1_px", j1_px, weight, 400,
                        -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j1_py", j1_py, weight, 400,
                        -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j1_pz", j1_pz, weight, 400,
                        -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j1_mass", j1_mass, weight,
                        200, 0., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j1_bscore", j1_bscore,
                        weight, 100, 0., 1.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j2_px", j2_px, weight, 400,
                        -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j2_py", j2_py, weight, 400,
                        -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j2_pz", j2_pz, weight, 400,
                        -200., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j2_mass", j2_mass, weight,
                        200, 0., 200.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/j2_bscore", j2_bscore,
                        weight, 100, 0., 1.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/dR_mu1mu2", dR_mu1mu2,
                        weight, 100, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/dR_mu1ele", dR_mu1ele,
                        weight, 100, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/dR_mu2ele", dR_mu2ele,
                        weight, 100, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/dR_j1ele", dR_j1ele, weight,
                        100, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/dR_j2ele", dR_j2ele, weight,
                        100, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/dR_j1j2", dR_j1j2, weight,
                        100, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/HT", HT, weight, 1000, 0.,
                        1000.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/MT", MT, weight, 300, 0.,
                        300.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/Nj", Nj, weight, 20, 0.,
                        20.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/Nb", Nb, weight, 10, 0.,
                        10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/MET", METv.Pt(), weight,
                        300, 0., 300.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/avg_dRjets", avg_dRjets,
                        weight, 100, 0., 10.)
        htool.fill_hist(f"{SAMPLE}/{REGION}/{syst}/avg_bscore", avg_bscore,
                        weight, 100, 0., 1.)
        for mass_point in MASS_POINTs:
            score_fake, score_ttX = scores_fake[mass_point], scores_ttX[mass_point]
            htool.fill_hist(f"{SAMPLE}/{mass_point}/{REGION}/{syst}/score_fake", score_fake, weight, 100, 0., 1.)
            htool.fill_hist(f"{SAMPLE}/{mass_point}/{REGION}/{syst}/score_ttX", score_ttX, weight, 100, 0., 1.)
            htool.fill_hist3d(f"{SAMPLE}/{mass_point}/{REGION}/{syst}/fake_ttX_mMM", score_fake, score_ttX, diMuon.Mass(), weight,
                                100, 0., 1., 100, 0., 1., 3000, 0., 300.)
    else:
        if (not evt.IsDATA) or (not syst == "Central"):
            return
        # estimate fake contribution
        w_fake, w_fake_up, w_fake_down = get_fake_weights(muons, electrons)
        htool.fill_hist(f"fake/{REGION}/Central/mMuMu", diMuon.Mass(), w_fake,
                        3000, 0., 300.)
        htool.fill_hist(f"fake/{REGION}/Central/MT", MT, w_fake, 300, 0., 300.)
        htool.fill_object(f"fake/{REGION}/Central/diMuon", diMuon, w_fake)
        htool.fill_object(f"fake/{REGION}/Central/METv", METv, w_fake)
        htool.fill_muons(f"fake/{REGION}/Central/muons", muons, w_fake)
        htool.fill_electrons(f"fake/{REGION}/Central/electrons", electrons,
                             w_fake)
        htool.fill_jets(f"fake/{REGION}/Central/jets", jets, w_fake)
        htool.fill_jets(f"fake/{REGION}/Central/bjets", bjets, w_fake)

        htool.fill_hist(f"fake/{REGION}/Central/mu1_px", mu1_px, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/mu1_py", mu1_py, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/mu1_pz", mu1_pz, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/mu2_px", mu1_px, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/mu2_py", mu2_py, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/mu2_pz", mu2_pz, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/ele_px", ele_px, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/ele_py", ele_py, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/ele_pz", ele_pz, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j1_px", j1_px, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j1_py", j1_py, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j1_pz", j1_pz, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j1_mass", j1_mass, w_fake, 200,
                        0., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j1_bscore", j1_bscore, w_fake,
                        100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Central/j2_px", j2_px, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j2_py", j2_py, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j2_pz", j2_pz, w_fake, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j2_mass", j2_mass, w_fake, 200,
                        0., 200.)
        htool.fill_hist(f"fake/{REGION}/Central/j2_bscore", j2_bscore, w_fake,
                        100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Central/dR_mu1mu2", dR_mu1mu2, w_fake,
                        100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/dR_mu1ele", dR_mu1ele, w_fake,
                        100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/dR_mu2ele", dR_mu2ele, w_fake,
                        100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/dR_j1ele", dR_j1ele, w_fake,
                        100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/dR_j2ele", dR_j2ele, w_fake,
                        100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/dR_j1j2", dR_j1j2, w_fake, 100,
                        0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/HT", HT, w_fake, 1000, 0.,
                        1000.)
        htool.fill_hist(f"fake/{REGION}/Central/MT", MT, w_fake, 300, 0., 300.)
        htool.fill_hist(f"fake/{REGION}/Central/Nj", Nj, w_fake, 20, 0., 20.)
        htool.fill_hist(f"fake/{REGION}/Central/Nb", Nb, w_fake, 10, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/MET", METv.Pt(), w_fake, 300,
                        0., 300.)
        htool.fill_hist(f"fake/{REGION}/Central/avg_dRjets", avg_dRjets,
                        w_fake, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Central/avg_bscore", avg_bscore,
                        w_fake, 100, 0., 1.)
        for mass_point in MASS_POINTs:
            score_fake, score_ttX = scores_fake[mass_point], scores_ttX[mass_point]
            htool.fill_hist(f"fake/{mass_point}/{REGION}/Central/score_fake", score_fake, w_fake, 100, 0., 1.)
            htool.fill_hist(f"fake/{mass_point}/{REGION}/Central/score_ttX", score_ttX, w_fake, 100, 0., 1.)
            htool.fill_hist3d(f"fake/{mass_point}/{REGION}/Central/fake_ttX_mMM", score_fake, score_ttX, diMuon.Mass(), w_fake,
                                100, 0., 1., 100, 0., 1., 3000, 0., 300.)
        
        htool.fill_hist(f"fake/{REGION}/Up/mMuMu", diMuon.Mass(), w_fake_up,
                        3000, 0., 300.)
        htool.fill_hist(f"fake/{REGION}/Up/MT", MT, w_fake_up, 300, 0., 300.)
        htool.fill_object(f"fake/{REGION}/Up/diMuon", diMuon, w_fake_up)
        htool.fill_object(f"fake/{REGION}/Up/METv", METv, w_fake_up)
        htool.fill_muons(f"fake/{REGION}/Up/muons", muons, w_fake_up)
        htool.fill_electrons(f"fake/{REGION}/Up/electrons", electrons,
                             w_fake_up)
        htool.fill_jets(f"fake/{REGION}/Up/jets", jets, w_fake_up)
        htool.fill_jets(f"fake/{REGION}/Up/bjets", bjets, w_fake_up)

        htool.fill_hist(f"fake/{REGION}/Up/mu1_px", mu1_px, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/mu1_py", mu1_py, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/mu1_pz", mu1_pz, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/mu2_px", mu1_px, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/mu2_py", mu2_py, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/mu2_pz", mu2_pz, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/ele_px", ele_px, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/ele_py", ele_py, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/ele_pz", ele_pz, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j1_px", j1_px, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j1_py", j1_py, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j1_pz", j1_pz, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j1_mass", j1_mass, w_fake_up, 200,
                        0., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j1_bscore", j1_bscore, w_fake_up,
                        100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Up/j2_px", j2_px, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j2_py", j2_py, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j2_pz", j2_pz, w_fake_up, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j2_mass", j2_mass, w_fake_up, 200,
                        0., 200.)
        htool.fill_hist(f"fake/{REGION}/Up/j2_bscore", j2_bscore, w_fake_up,
                        100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Up/dR_mu1mu2", dR_mu1mu2, w_fake_up,
                        100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/dR_mu1ele", dR_mu1ele, w_fake_up,
                        100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/dR_mu2ele", dR_mu2ele, w_fake_up,
                        100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/dR_j1ele", dR_j1ele, w_fake_up, 100,
                        0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/dR_j2ele", dR_j2ele, w_fake_up, 100,
                        0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/dR_j1j2", dR_j1j2, w_fake_up, 100,
                        0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/HT", HT, w_fake_up, 1000, 0., 1000.)
        htool.fill_hist(f"fake/{REGION}/Up/MT", MT, w_fake_up, 300, 0., 300.)
        htool.fill_hist(f"fake/{REGION}/Up/Nj", Nj, w_fake_up, 20, 0., 20.)
        htool.fill_hist(f"fake/{REGION}/Up/Nb", Nb, w_fake_up, 10, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/MET", METv.Pt(), w_fake_up, 300, 0.,
                        300.)
        htool.fill_hist(f"fake/{REGION}/Up/avg_dRjets", avg_dRjets, w_fake_up,
                        100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Up/avg_bscore", avg_bscore, w_fake_up,
                        100, 0., 1.)
        for mass_point in MASS_POINTs:
            score_fake, score_ttX = scores_fake[mass_point], scores_ttX[mass_point]
            htool.fill_hist(f"fake/{mass_point}/{REGION}/Up/score_fake", score_fake, w_fake_up, 100, 0., 1.)
            htool.fill_hist(f"fake/{mass_point}/{REGION}/Up/score_ttX", score_ttX, w_fake_up, 100, 0., 1.)
            htool.fill_hist3d(f"fake/{mass_point}/{REGION}/Up/fake_ttX_mMM", score_fake, score_ttX, diMuon.Mass(), w_fake_up,
                                100, 0., 1., 100, 0., 1., 3000, 0., 300.)

        htool.fill_hist(f"fake/{REGION}/Down/mMuMu", diMuon.Mass(),
                        w_fake_down, 3000, 0., 300.)
        htool.fill_hist(f"fake/{REGION}/Down/MT", MT, w_fake_down, 300, 0.,
                        300.)
        htool.fill_object(f"fake/{REGION}/Down/diMuon", diMuon, w_fake_down)
        htool.fill_object(f"fake/{REGION}/Down/METv", METv, w_fake_down)
        htool.fill_muons(f"fake/{REGION}/Down/muons", muons, w_fake_down)
        htool.fill_electrons(f"fake/{REGION}/Down/electrons", electrons,
                             w_fake_down)
        htool.fill_jets(f"fake/{REGION}/Down/jets", jets, w_fake_down)
        htool.fill_jets(f"fake/{REGION}/Down/bjets", bjets, w_fake_down)

        htool.fill_hist(f"fake/{REGION}/Down/mu1_px", mu1_px, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/mu1_py", mu1_py, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/mu1_pz", mu1_pz, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/mu2_px", mu1_px, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/mu2_py", mu2_py, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/mu2_pz", mu2_pz, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/ele_px", ele_px, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/ele_py", ele_py, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/ele_pz", ele_pz, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j1_px", j1_px, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j1_py", j1_py, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j1_pz", j1_pz, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j1_mass", j1_mass, w_fake_down,
                        200, 0., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j1_bscore", j1_bscore,
                        w_fake_down, 100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Down/j2_px", j2_px, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j2_py", j2_py, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j2_pz", j2_pz, w_fake_down, 400,
                        -200., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j2_mass", j2_mass, w_fake_down,
                        200, 0., 200.)
        htool.fill_hist(f"fake/{REGION}/Down/j2_bscore", j2_bscore,
                        w_fake_down, 100, 0., 1.)
        htool.fill_hist(f"fake/{REGION}/Down/dR_mu1mu2", dR_mu1mu2,
                        w_fake_down, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/dR_mu1ele", dR_mu1ele,
                        w_fake_down, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/dR_mu2ele", dR_mu2ele,
                        w_fake_down, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/dR_j1ele", dR_j1ele, w_fake_down,
                        100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/dR_j2ele", dR_j2ele, w_fake_down,
                        100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/dR_j1j2", dR_j1j2, w_fake_down,
                        100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/HT", HT, w_fake_down, 1000, 0.,
                        1000.)
        htool.fill_hist(f"fake/{REGION}/Down/MT", MT, w_fake_down, 300, 0.,
                        300.)
        htool.fill_hist(f"fake/{REGION}/Down/Nj", Nj, w_fake_down, 20, 0., 20.)
        htool.fill_hist(f"fake/{REGION}/Down/Nb", Nb, w_fake_down, 10, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/MET", METv.Pt(), w_fake_down, 300,
                        0., 300.)
        htool.fill_hist(f"fake/{REGION}/Down/avg_dRjets", avg_dRjets,
                        w_fake_down, 100, 0., 10.)
        htool.fill_hist(f"fake/{REGION}/Down/avg_bscore", avg_bscore,
                        w_fake_down, 100, 0., 1.)
        for mass_point in MASS_POINTs:
            score_fake, score_ttX = scores_fake[mass_point], scores_ttX[mass_point]
            htool.fill_hist(f"fake/{mass_point}/{REGION}/Down/score_fake", score_fake, w_fake_down, 100, 0., 1.)
            htool.fill_hist(f"fake/{mass_point}/{REGION}/Down/score_ttX", score_ttX, w_fake_down, 100, 0., 1.)
            htool.fill_hist3d(f"fake/{mass_point}/{REGION}/Down/fake_ttX_mMM", score_fake, score_ttX, diMuon.Mass(), w_fake_down,
                                100, 0., 1., 100, 0., 1., 3000, 0., 300.)
        
def main():
    print(f"Estimating {SAMPLE}...")
    if SAMPLE == "DATA" and CHANNEL == "1E2Mu":
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

                loop(evt, clfs_fake, clfs_ttX, syst, htool)
        f.Close()
        f = TFile.Open(fkey_emu)
        for syst in SYSTs + WEIGHTSYSTs:
            for evt in f.Events:
                # check double counting
                this_evt = (evt.run, evt.event, evt.lumi)
                if this_evt in events[syst]:
                    continue

                loop(evt, clfs_fake, clfs_ttX, syst, htool)
        f.Close()

    elif SAMPLE == "DATA" and args.channel == "3Mu":
        fkey = f"{SAMPLE_DIR}/DATA/Selector_DoubleMuon.root"
        f = TFile.Open(fkey)
        for syst in SYSTs + WEIGHTSYSTs:
            for evt in f.Events:
                loop(evt, clfs_fake, clfs_ttX, syst, htool)
        f.Close()

    elif SAMPLE in MCs or "TTToHcToWA" in SAMPLE:
        fkey = f"{SAMPLE_DIR}/MCSamples/Selector_{SAMPLE}.root"
        f = TFile.Open(fkey)
        for syst in SYSTs + WEIGHTSYSTs:
            for evt in f.Events:
                loop(evt, clfs_fake, clfs_ttX, syst, htool)
        f.Close()
    else:
        raise AttributeError

    htool.save()


if __name__ == "__main__":
    main()
    print("End process")
