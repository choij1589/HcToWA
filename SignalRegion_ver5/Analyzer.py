import os
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
parser.add_argument("--mass", "-m", default=None, required=True, type=str, help="signal mass point")
args = parser.parse_args()

# Define global variables
CHANNEL = args.channel
SAMPLE = args.sample
MASS_POINT = args.mass
set_global(CHANNEL, SAMPLE)

# check output directory
if not os.path.exists(f"Outputs/{CHANNEL}/{MASS_POINT}/ROOT"):
    os.makedirs(f"Outputs/{CHANNEL}/{MASS_POINT}/ROOT")

if CHANNEL == "1E2Mu":
    DATA = ["MuonEG", "DoubleMuon"]
    BKGs = ["rare", "ttX", "VV", "fake", "conv"]
    MCs = ["DY", "ZG", "rare", "ttX", "VV"]
    SAMPLE_DIR = "../Samples/Selector/2017/Skim1E2Mu__/"
    OUTFILE = f"Outputs/{CHANNEL}/{MASS_POINT}/ROOT/{SAMPLE}.root"
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
    OUTFILE = f"Outputs/{CHANNEL}/{MASS_POINT}/ROOT/{SAMPLE}.root"
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

SYSTs = ["Central"]
WEIGHTSYSTs = []
htool = HistTool(outfile=OUTFILE)

# load scalers
scaler_fake = pickle.load(
    open(f"Outputs/{CHANNEL}/{MASS_POINT}/models/scaler_fake.pkl", 'rb')
)
scaler_ttX = pickle.load(
    open(f"Outputs/{CHANNEL}/{MASS_POINT}/models/scaler_ttX.pkl", 'rb')
)

# load classifiers
# all hyperparameters for each mass point have been optimized 
# using the grid search technique
# The values are stored in Outputs/{CHANNEL}/CSV/hyper_params.csv
# TODO: you should also optimize the discriminator for
# kernel initialization & activation functions also
from MLTools import SelfNormDNN
hyper_params = pd.read_csv(
        f"Outputs/{CHANNEL}/CSV/hyper_params.csv", index_col="mass_point"
)
lr_fake, n_hidden_fake = hyper_params.loc[MASS_POINT, 'lr_fake'], hyper_params.loc[MASS_POINT, "n_hidden_fake"]
clf_fake = SelfNormDNN(len(features), 2, n_hidden_fake)
clf_fake.load_state_dict(
    torch.load(
        f"Outputs/{CHANNEL}/{MASS_POINT}/models/chkpoint_sig_vs_fake_lr-{lr_fake}_n_hidden-{n_hidden_fake}.pt",
        map_location=torch.device('cpu')
    )
)
clf_fake.eval()

lr_ttX, n_hidden_ttX = hyper_params.loc[MASS_POINT, 'lr_ttX'], hyper_params.loc[MASS_POINT, 'n_hidden_ttX']
clf_ttX = SelfNormDNN(len(features), 2, n_hidden_ttX)
clf_ttX.load_state_dict(
    torch.load(
        f"Outputs/{CHANNEL}/{MASS_POINT}/models/chkpoint_sig_vs_ttX_lr-{lr_ttX}_n_hidden-{n_hidden_ttX}.pt",
        map_location=torch.device('cpu')
    )
)
clf_ttX.eval()

# helper functions
def get_os_pairs(muons):
    # expect the muon pairs are not all the same sign
    for idx in range(len(muons)):
        if abs(muons[idx].Charge() + muons[(idx+1)%3].Charge()) == 2:
            ssmu1, ssmu2 = muons[idx], muons[(idx+1)%3]
            osmu = muons[(idx+2)%3]
            break
        else:
            continue
    # make pairs
    pair1 = osmu + ssmu1
    pair2 = osmu + ssmu2
    
    return (pair1, pair2)

def select(evt, muons, electrons, jets, bjets, METv):
    """Event selection on trilep regions"""
    if CHANNEL == "1E2Mu":
        if not (evt.passDblMuTrigs or evt.passEMuTrigs):
            return False
        
        if not (len(muons) == 2 and len(electrons) == 1):
            return False 
        
        pass_safecut = ((muons[0].Pt() > 20. and muons[1].Pt() > 10.)
                        or (muons[0].Pt() > 25. and electrons[0].Pt() > 15.)
                        or (electrons[0].Pt() > 25. and muons[0].Pt() > 10.))
        if not pass_safecut:
            return False
        
        # check pt condition for the conversion samples
        if SAMPLE == "DY" and (muons[0].Pt() > 20. and muons[1].Pt() > 20. and electrons[0].Pt() > 20.):
            return False
        if SAMPLE == "ZG" and (muons[0].Pt() < 20. or muons[1].Pt() < 20. or electrons[0].Pt() < 20.):
            return False
        
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
        if not evt.passDblMuTrigs:
            return False
        
        if not (len(muons) == 3):
            return False
        
        pass_safecut = (muons[0].Pt() > 20. and muons[1].Pt() > 10.)
        if not pass_safecut:
            return False
        
        # check pt condition for the conversion samples
        if SAMPLE == "DY" and (muons[0].Pt() > 20. and muons[1].Pt() > 20. and muons[2].Pt() > 20.):
            return False
        if SAMPLE == "ZG" and (muons[0].Pt() < 20. or muons[1].Pt() < 20. or muons[2].Pt() < 20.):
            return False
        
        # check charge condition
        if abs(muons[0].Charge() + muons[1].Charge() + muons[2].Charge()) != 1:
            return False
        
        # make os pairs
        ospair1, ospair2 = get_os_pairs(muons)
        
        if not (ospair1.Mass() > 12. and ospair2.Mass() > 12.):
            return False
        
        # jet conditions
        if not len(jets) >= 2:
            return False
        
        if len(bjets) >= 1:
            return "SR"
        else:
            if abs(ospair1.Mass() - 91.2) < 15. or abs(ospair2.Mass() - 91.2) < 15.:
                return "ZFake"
            else:
                return False    
    else:
        print(f"Wrong channel {CHANNEL}")
        raise(AttributeError)
    
# loop over events, fill histograms
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
    elif CHANNEL == "3Mu":
        if len(muons_tight) == 3 and len(electrons_tight) == 0:
            tight_flag = True
        if len(muons_prompt) == 3 and len(electrons_prompt) == 0:
            prompt_flag = True
            
        ospair1, ospair2 = get_os_pairs(muons)
        
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
        mu3_px = muons[2].P4().Px()
        mu3_py = muons[2].P4().Py()
        mu3_pz = muons[2].P4().Pz()
        mu3_mass = muons[2].Mass()
        mu3_charge = muons[2].Charge()

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
        dR_mu1mu3 = muons[0].DeltaR(muons[2])
        dR_mu2mu3 = muons[1].DeltaR(muons[2])
        dR_j1j2 = jets[0].DeltaR(jets[1])
        dR_j1mu1 = jets[0].DeltaR(muons[0])
        dR_j1mu2 = jets[0].DeltaR(muons[1])
        dR_j1mu3 = jets[0].DeltaR(muons[2])
        dR_j2mu1 = jets[1].DeltaR(muons[0])
        dR_j2mu2 = jets[1].DeltaR(muons[1])
        dR_j2mu3 = jets[1].DeltaR(muons[2])
        comb = combinations(range(len(jets)), 2)

        dRjets = []
        for elements in comb:
            j1 = jets[elements[0]]
            j2 = jets[elements[1]]
            dRjets.append(j1.DeltaR(j2))
        avg_dRjets = sum(dRjets) / len(dRjets)
        avg_bscore = sum([x.BtagScore() for x in jets]) / len(jets)

        HT = sum([x.Pt() for x in jets])
        Nj = len(jets)
        Nb = len(bjets)
        LT = sum([x.Pt() for x in leptons])
        MET = METv.Pt()

        features = np.array([[
            mu1_px, mu1_py, mu1_pz, mu1_mass, mu1_charge, mu2_px,
            mu2_py, mu2_pz, mu2_mass, mu2_charge, mu3_px, mu3_py,
            mu3_pz, mu3_mass, mu3_charge, j1_px, j1_py, j1_pz,
            j1_mass, j1_bscore, j2_px, j2_py, j2_pz, j2_mass,
            j2_bscore, dR_mu1mu2, dR_mu1mu3, dR_mu2mu3, dR_j1j2,
            dR_j1mu1, dR_j1mu2, dR_j1mu3, dR_j2mu1, dR_j2mu2, dR_j2mu3,
            HT, Nj, Nb, LT, MET, avg_dRjets, avg_bscore
        ]])
    else:
        raise (AttributeError)
    
    score_fake = clf_fake(torch.FloatTensor(
        scaler_fake.transform(features))).view(2)[0].detach().numpy()
    score_ttX = clf_ttX(torch.FloatTensor(
        scaler_ttX.transform(features))).view(2)[0].detach().numpy()

    # Now fill histograms
    if CHANNEL == "1E2Mu":
        if tight_flag:
            if SAMPLE in ["DY", "ZG"] and not prompt_flag:
                return
            
            weight = 1. if evt.IsDATA else get_weight(evt, syst)
            
            input_path = f"{SAMPLE}/object/{REGION}/{syst}"
            htool.fill_object(f"{input_path}/METv", METv, weight)
            htool.fill_object(f"{input_path}/diMuon", diMuon, weight)
            htool.fill_muons(f"{input_path}/muons", muons, weight)
            htool.fill_electrons(f"{input_path}/electrons", electrons, weight)
            htool.fill_jets(f"{input_path}/jets", jets, weight)
            htool.fill_jets(f"{input_path}/bjets", bjets, weight)
            
            # fill input variables
            input_path = f"{SAMPLE}/inputs/{REGION}/{syst}"
            htool.fill_hist(f"{input_path}/mu1_px", mu1_px, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_py", mu1_py, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_pz", mu1_pz, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_mass", mu1_mass, weight, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu1_charge", mu1_charge, weight, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/mu2_px", mu1_px, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_py", mu2_py, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_pz", mu2_pz, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_mass", mu2_mass, weight, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu2_charge", mu2_charge, weight, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/ele_px", ele_px, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/ele_py", ele_py, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/ele_pz", ele_pz, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/ele_mass", ele_mass, weight, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/j1_px", j1_px, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_py", j1_py, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_pz", j1_pz, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_mass", j1_mass, weight, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j1_bscore", j1_bscore, weight, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/j2_px", j2_px, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_py", j2_py, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_pz", j2_pz, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_mass", j2_mass, weight, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j2_bscore", j2_bscore, weight, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/dR_mu1mu2", dR_mu1mu2, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu1ele", dR_mu1ele, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu2ele", dR_mu2ele, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1ele", dR_j1ele, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2ele", dR_j2ele, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1j2", dR_j1j2, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/HT", HT, weight, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/MT", MT, weight, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/Nj", Nj, weight, 20, 0., 20.)
            htool.fill_hist(f"{input_path}/Nb", Nb, weight, 10, 0., 10.)
            htool.fill_hist(f"{input_path}/MET", METv.Pt(), weight, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/avg_dRjets", avg_dRjets, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/avg_bscore", avg_bscore, weight, 100, 0., 1.)
            
            # fill mass dependent observables
            input_path = f"{SAMPLE}/{MASS_POINT}/{REGION}/{syst}"
            htool.fill_hist(f"{input_path}/mMM", diMuon.Mass(), weight, 3000, 0., 300.)
            htool.fill_hist(f"{input_path}/score_fake", score_fake, weight, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/score_ttX", score_ttX, weight, 100, 0., 1.)
            htool.fill_hist3d(f"{input_path}/fake_ttX_mMM", score_fake, score_ttX, diMuon.Mass(), weight,
                                100, 0., 1., 100, 0., 1., 3000, 0., 300.)
        else:
            # estimate fake contribution
            if (not evt.IsDATA) or (not syst == "Central"):
                return
            w_fake, w_fake_up, w_fake_down = get_fake_weights(muons, electrons)
            
            input_path = f"fake/object/{REGION}/Central"
            htool.fill_object(f"{input_path}/METv", METv, w_fake)
            htool.fill_object(f"{input_path}/diMuon", diMuon, w_fake)
            htool.fill_muons(f"{input_path}/muons", muons, w_fake)
            htool.fill_electrons(f"{input_path}/electrons", electrons, w_fake)
            htool.fill_jets(f"{input_path}/jets", jets, w_fake)
            htool.fill_jets(f"{input_path}/bjets", bjets, w_fake)
            
            # fill input variables
            input_path = f"fake/inputs/{REGION}/Central"
            htool.fill_hist(f"{input_path}/mu1_px", mu1_px, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_py", mu1_py, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_pz", mu1_pz, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_mass", mu1_mass, w_fake, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu1_charge", mu1_charge, w_fake, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/mu2_px", mu1_px, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_py", mu2_py, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_pz", mu2_pz, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/ele_px", ele_px, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/ele_py", ele_py, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/ele_pz", ele_pz, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_px", j1_px, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_py", j1_py, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_pz", j1_pz, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_mass", j1_mass, w_fake, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j1_bscore", j1_bscore, w_fake, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/j2_px", j2_px, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_py", j2_py, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_pz", j2_pz, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_mass", j2_mass, w_fake, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j2_bscore", j2_bscore, w_fake, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/dR_mu1mu2", dR_mu1mu2, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu1ele", dR_mu1ele, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu2ele", dR_mu2ele, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1ele", dR_j1ele, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2ele", dR_j2ele, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1j2", dR_j1j2, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/HT", HT, w_fake, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/MT", MT, w_fake, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/Nj", Nj, w_fake, 20, 0., 20.)
            htool.fill_hist(f"{input_path}/Nb", Nb, w_fake, 10, 0., 10.)
            htool.fill_hist(f"{input_path}/MET", METv.Pt(), w_fake, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/avg_dRjets", avg_dRjets, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/avg_bscore", avg_bscore, w_fake, 100, 0., 1.)
            
            input_path = f"fake/{MASS_POINT}/{REGION}/Central"
            htool.fill_hist(f"{input_path}/mMM", diMuon.Mass(), w_fake, 3000, 0., 300.)
            htool.fill_hist(f"{input_path}/score_fake", score_fake, w_fake, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/score_ttX", score_ttX, w_fake, 100, 0., 1.)
            htool.fill_hist3d(f"{input_path}/fake_ttX_mMM", score_fake, score_ttX, diMuon.Mass(), w_fake,
                                100, 0., 1., 100, 0., 1., 3000, 0., 300.)
                
            input_path = f"fake/object/{REGION}/Up"
            htool.fill_object(f"{input_path}/METv", METv, w_fake_up)
            htool.fill_object(f"{input_path}/diMuon", diMuon, w_fake_up)
            htool.fill_muons(f"{input_path}/muons", muons, w_fake_up)
            htool.fill_electrons(f"{input_path}/electrons", electrons, w_fake_up)
            htool.fill_jets(f"{input_path}/jets", jets, w_fake_up)
            htool.fill_jets(f"{input_path}/bjets", bjets, w_fake_up)
            
            # fill input variables
            input_path = f"fake/inputs/{REGION}/Up"
            htool.fill_hist(f"{input_path}/mu1_px", mu1_px, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_py", mu1_py, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_pz", mu1_pz, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_mass", mu1_mass, w_fake_up, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu1_charge", mu1_charge, w_fake_up, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/mu2_px", mu1_px, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_py", mu2_py, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_pz", mu2_pz, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/ele_px", ele_px, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/ele_py", ele_py, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/ele_pz", ele_pz, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_px", j1_px, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_py", j1_py, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_pz", j1_pz, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_mass", j1_mass, w_fake_up, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j1_bscore", j1_bscore, w_fake_up, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/j2_px", j2_px, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_py", j2_py, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_pz", j2_pz, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_mass", j2_mass, w_fake_up, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j2_bscore", j2_bscore, w_fake_up, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/dR_mu1mu2", dR_mu1mu2, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu1ele", dR_mu1ele, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu2ele", dR_mu2ele, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1ele", dR_j1ele, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2ele", dR_j2ele, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1j2", dR_j1j2, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/HT", HT, w_fake_up, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/MT", MT, w_fake_up, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/Nj", Nj, w_fake_up, 20, 0., 20.)
            htool.fill_hist(f"{input_path}/Nb", Nb, w_fake_up, 10, 0., 10.)
            htool.fill_hist(f"{input_path}/MET", METv.Pt(), w_fake_up, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/avg_dRjets", avg_dRjets, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/avg_bscore", avg_bscore, w_fake_up, 100, 0., 1.)
                
            input_path = f"fake/{MASS_POINT}/{REGION}/Up"
            htool.fill_hist(f"{input_path}/mMM", diMuon.Mass(), w_fake_up, 3000, 0., 300.)
            htool.fill_hist(f"{input_path}/score_fake", score_fake, w_fake_up, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/score_ttX", score_ttX, w_fake_up, 100, 0., 1.)
            htool.fill_hist3d(f"{input_path}/fake_ttX_mMM", score_fake, score_ttX, diMuon.Mass(), w_fake_up,
                                100, 0., 1., 100, 0., 1., 3000, 0., 300.) 
                
            input_path = f"fake/object/{REGION}/Down"
            htool.fill_object(f"{input_path}/METv", METv, w_fake_down)
            htool.fill_object(f"{input_path}/diMuon", diMuon, w_fake_down)
            htool.fill_muons(f"{input_path}/muons", muons, w_fake_down)
            htool.fill_electrons(f"{input_path}/electrons", electrons, w_fake_down)
            htool.fill_jets(f"{input_path}/jets", jets, w_fake_down)
            htool.fill_jets(f"{input_path}/bjets", bjets, w_fake_down)
            
            # fill input variables
            input_path = f"fake/inputs/{REGION}/Down"
            htool.fill_hist(f"{input_path}/mu1_px", mu1_px, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_py", mu1_py, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_pz", mu1_pz, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_mass", mu1_mass, w_fake_down, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu1_charge", mu1_charge, w_fake_down, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/mu2_px", mu1_px, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_py", mu2_py, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_pz", mu2_pz, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/ele_px", ele_px, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/ele_py", ele_py, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/ele_pz", ele_pz, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_px", j1_px, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_py", j1_py, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_pz", j1_pz, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_mass", j1_mass, w_fake_down, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j1_bscore", j1_bscore, w_fake_down, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/j2_px", j2_px, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_py", j2_py, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_pz", j2_pz, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_mass", j2_mass, w_fake_down, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j2_bscore", j2_bscore, w_fake_down, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/dR_mu1mu2", dR_mu1mu2, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu1ele", dR_mu1ele, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu2ele", dR_mu2ele, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1ele", dR_j1ele, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2ele", dR_j2ele, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1j2", dR_j1j2, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/HT", HT, w_fake_down, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/MT", MT, w_fake_down, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/Nj", Nj, w_fake_down, 20, 0., 20.)
            htool.fill_hist(f"{input_path}/Nb", Nb, w_fake_down, 10, 0., 10.)
            htool.fill_hist(f"{input_path}/MET", METv.Pt(), w_fake_down, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/avg_dRjets", avg_dRjets, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/avg_bscore", avg_bscore, w_fake_down, 100, 0., 1.)
            
            input_path = f"fake/{MASS_POINT}/{REGION}/Down"
            htool.fill_hist(f"{input_path}/mMM", diMuon.Mass(), w_fake_down, 3000, 0., 300.)
            htool.fill_hist(f"{input_path}/score_fake", score_fake, w_fake_down, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/score_ttX", score_ttX, w_fake_down, 100, 0., 1.)
            htool.fill_hist3d(f"{input_path}/fake_ttX_mMM", score_fake, score_ttX, diMuon.Mass(), w_fake_down,
                                100, 0., 1., 100, 0., 1., 3000, 0., 300.)
        
    elif CHANNEL == "3Mu":
        if tight_flag:
            if SAMPLE in ["DY", "ZG"] and not prompt_flag:
                return
            weight = 1. if evt.IsDATA else get_weight(evt, syst)     
            
            # os pairs will be filled after matching
            input_path = f"{SAMPLE}/objects/{REGION}/{syst}"
            htool.fill_object(f"{input_path}/METv", METv, weight)
            htool.fill_muons(f"{input_path}/muons", muons, weight)
            htool.fill_jets(f"{input_path}/jets", jets, weight)
            htool.fill_jets(f"{input_path}/bjets", bjets, weight)
            
            # fill input variables
            input_path = f"{SAMPLE}/inputs/{REGION}/{syst}"
            htool.fill_hist(f"{input_path}/mu1_px", mu1_px, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_py", mu1_py, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_pz", mu1_pz, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_mass",mu1_mass, weight, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu1_charge", mu1_charge, weight, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/mu2_px", mu2_px, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_py", mu2_py, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_pz", mu2_pz, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_mass",mu2_mass, weight, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu2_charge", mu2_charge, weight, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/mu3_px", mu3_px, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu3_py", mu3_py, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu3_pz", mu3_pz, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu3_mass",mu3_mass, weight, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu3_charge", mu3_charge, weight, 4, -2., 2.) 
            htool.fill_hist(f"{input_path}/j1_px", j1_px, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_py", j1_py, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_pz", j1_pz, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_mass", j1_mass, weight, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j1_bscore", j1_bscore, weight, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/j2_px", j2_px, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_py", j2_py, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_pz", j2_pz, weight, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_mass", j2_mass, weight, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j2_bscore", j2_bscore, weight, 100, 0., 1.) 
            htool.fill_hist(f"{input_path}/dR_mu1mu2", dR_mu1mu2, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu1mu3", dR_mu1mu3, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu2mu3", dR_mu2mu3, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1j2", dR_j1j2, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1mu1", dR_j1mu1, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1mu2", dR_j1mu2, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1mu3", dR_j1mu3, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2mu1", dR_j2mu1, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2mu2", dR_j2mu2, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2mu3", dR_j2mu3, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/HT", HT, weight, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/Nj", Nj, weight, 20, 0., 20.)
            htool.fill_hist(f"{input_path}/Nb", Nb, weight, 10, 0., 10.)
            htool.fill_hist(f"{input_path}/LT", LT, weight, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/MET", MET, weight, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/avg_dRjets", avg_dRjets, weight, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/avg_bscore", avg_bscore, weight, 100, 0., 10.)
            
            # variables with mass point dependence
            input_path = f"{SAMPLE}/{MASS_POINT}/{REGION}/{syst}"
            mA = float(MASS_POINT.split("_")[1][2:])
            
            htool.fill_hist(f"{input_path}/score_fake", score_fake, weight, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/score_ttX", score_ttX, weight, 100, 0., 1.)
            if REGION == "ZFake":
                # make Z candidate
                if abs(ospair1.Mass() - 91.2) < abs(ospair2.Mass() - 91.2):
                    ZCand, nZCand = ospair1, ospair2
                else:
                    ZCand, nZCand = ospair2, ospair1
                htool.fill_object(f"{input_path}/ZCand", ZCand, weight)
                htool.fill_object(f"{input_path}/nZCand", nZCand, weight)
            elif REGION == "SR":
                # make A candidate
                if abs(ospair1.Mass() - mA) < abs(ospair2.Mass() - mA):
                    Acand, nAcand = ospair1, ospair2
                else:
                    Acand, nAcand = ospair2, ospair1
                htool.fill_object(f"{input_path}/Acand", Acand, weight)
                htool.fill_object(f"{input_path}/nAcand", nAcand, weight)
                htool.fill_hist3d(f"{input_path}/fake_ttX_mMM", score_fake, score_ttX, Acand.Mass(), weight,
                                    100, 0., 1., 100, 0., 1., 3000, 0., 300.)                    
                # matching ACand in Signal Samples
                if "TTToHcToWA" in SAMPLE:
                    AGen = Particle(evt.A_pt, evt.A_eta, evt.A_phi, evt.A_mass)
                    dRAAcand, dRAnAcand = AGen.DeltaR(Acand), AGen.DeltaR(nAcand)
                    htool.fill_hist(f"{input_path}/dRAAcand", dRAAcand, weight, 100, 0., 10.)
                    htool.fill_hist(f"{input_path}/dRAnAcand", dRAnAcand, weight, 100, 0., 10.)
            
        #fake
        else:
            if (not evt.IsDATA) or (not syst == "Central"):
                return
            w_fake, w_fake_up, w_fake_down = get_fake_weights(muons, electrons)
            
            input_path = f"fake/objects/{REGION}/Central"
            htool.fill_object(f"{input_path}/METv", METv, w_fake)
            htool.fill_muons(f"{input_path}/muons", muons, w_fake)
            htool.fill_jets(f"{input_path}/jets", jets, w_fake)
            htool.fill_jets(f"{input_path}/bjets", bjets, w_fake)
            
            # fill input variables
            input_path = f"fake/inputs/{REGION}/Central"
            htool.fill_hist(f"{input_path}/mu1_px", mu1_px, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_py", mu1_py, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_pz", mu1_pz, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_mass",mu1_mass, w_fake, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu1_charge", mu1_charge, w_fake, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/mu2_px", mu2_px, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_py", mu2_py, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_pz", mu2_pz, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_mass",mu2_mass, w_fake, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu2_charge", mu2_charge, w_fake, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/mu3_px", mu3_px, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu3_py", mu3_py, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu3_pz", mu3_pz, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu3_mass",mu3_mass, w_fake, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu3_charge", mu3_charge, w_fake, 4, -2., 2.) 
            htool.fill_hist(f"{input_path}/j1_px", j1_px, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_py", j1_py, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_pz", j1_pz, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_mass", j1_mass, w_fake, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j1_bscore", j1_bscore, w_fake, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/j2_px", j2_px, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_py", j2_py, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_pz", j2_pz, w_fake, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_mass", j2_mass, w_fake, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j2_bscore", j2_bscore, w_fake, 100, 0., 1.) 
            htool.fill_hist(f"{input_path}/dR_mu1mu2", dR_mu1mu2, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu1mu3", dR_mu1mu3, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu2mu3", dR_mu2mu3, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1j2", dR_j1j2, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1mu1", dR_j1mu1, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1mu2", dR_j1mu2, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1mu3", dR_j1mu3, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2mu1", dR_j2mu1, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2mu2", dR_j2mu2, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2mu3", dR_j2mu3, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/HT", HT, w_fake, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/Nj", Nj, w_fake, 20, 0., 20.)
            htool.fill_hist(f"{input_path}/Nb", Nb, w_fake, 10, 0., 10.)
            htool.fill_hist(f"{input_path}/LT", LT, w_fake, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/MET", MET, w_fake, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/avg_dRjets", avg_dRjets, w_fake, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/avg_bscore", avg_bscore, w_fake, 100, 0., 10.)
            
            # variables with mass point dependence
            input_path = f"fake/{MASS_POINT}/{REGION}/Central"
            mA = float(MASS_POINT.split("_")[1][2:])
            
            htool.fill_hist(f"{input_path}/score_fake", score_fake, w_fake, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/score_ttX", score_ttX, w_fake, 100, 0., 1.)
            if REGION == "ZFake":
                # make Z candidate
                if abs(ospair1.Mass() - 91.2) < abs(ospair2.Mass() - 91.2):
                    ZCand, nZCand = ospair1, ospair2
                else:
                    ZCand, nZCand = ospair2, ospair1
                htool.fill_object(f"{input_path}/ZCand", ZCand, w_fake)
                htool.fill_object(f"{input_path}/nZCand", nZCand, w_fake)
            elif REGION == "SR":
                # make A candidate
                if abs(ospair1.Mass() - mA) < abs(ospair2.Mass() - mA):
                    Acand, nAcand = ospair1, ospair2
                else:
                    Acand, nAcand = ospair2, ospair1
                htool.fill_object(f"{input_path}/Acand", Acand, w_fake)
                htool.fill_object(f"{input_path}/nAcand", nAcand, w_fake)
                htool.fill_hist3d(f"{input_path}/fake_ttX_mMM", score_fake, score_ttX, Acand.Mass(), w_fake,
                                    100, 0., 1., 100, 0., 1., 3000, 0., 300.)
                    
            # Up
            input_path = f"fake/objects/{REGION}/Up"
            htool.fill_object(f"{input_path}/METv", METv, w_fake_up)
            htool.fill_muons(f"{input_path}/muons", muons, w_fake_up)
            htool.fill_jets(f"{input_path}/jets", jets, w_fake_up)
            htool.fill_jets(f"{input_path}/bjets", bjets, w_fake_up)
            
            # fill input variables
            input_path = f"fake/inputs/{REGION}/Up"
            htool.fill_hist(f"{input_path}/mu1_px", mu1_px, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_py", mu1_py, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_pz", mu1_pz, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_mass",mu1_mass, w_fake_up, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu1_charge", mu1_charge, w_fake_up, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/mu2_px", mu2_px, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_py", mu2_py, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_pz", mu2_pz, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_mass",mu2_mass, w_fake_up, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu2_charge", mu2_charge, w_fake_up, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/mu3_px", mu3_px, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu3_py", mu3_py, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu3_pz", mu3_pz, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu3_mass",mu3_mass, w_fake_up, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu3_charge", mu3_charge, w_fake_up, 4, -2., 2.) 
            htool.fill_hist(f"{input_path}/j1_px", j1_px, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_py", j1_py, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_pz", j1_pz, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_mass", j1_mass, w_fake_up, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j1_bscore", j1_bscore, w_fake_up, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/j2_px", j2_px, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_py", j2_py, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_pz", j2_pz, w_fake_up, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_mass", j2_mass, w_fake_up, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j2_bscore", j2_bscore, w_fake_up, 100, 0., 1.) 
            htool.fill_hist(f"{input_path}/dR_mu1mu2", dR_mu1mu2, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu1mu3", dR_mu1mu3, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu2mu3", dR_mu2mu3, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1j2", dR_j1j2, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1mu1", dR_j1mu1, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1mu2", dR_j1mu2, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1mu3", dR_j1mu3, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2mu1", dR_j2mu1, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2mu2", dR_j2mu2, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2mu3", dR_j2mu3, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/HT", HT, w_fake_up, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/Nj", Nj, w_fake_up, 20, 0., 20.)
            htool.fill_hist(f"{input_path}/Nb", Nb, w_fake_up, 10, 0., 10.)
            htool.fill_hist(f"{input_path}/LT", LT, w_fake_up, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/MET", MET, w_fake_up, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/avg_dRjets", avg_dRjets, w_fake_up, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/avg_bscore", avg_bscore, w_fake_up, 100, 0., 10.)
            
            # variables with mass point dependence
            input_path = f"fake/{MASS_POINT}/{REGION}/Up"
            mA = float(MASS_POINT.split("_")[1][2:])
            
            htool.fill_hist(f"{input_path}/score_fake", score_fake, w_fake_up, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/score_ttX", score_ttX, w_fake_up, 100, 0., 1.)
            if REGION == "ZFake":
                # make Z candidate
                if abs(ospair1.Mass() - 91.2) < abs(ospair2.Mass() - 91.2):
                    ZCand, nZCand = ospair1, ospair2
                else:
                    ZCand, nZCand = ospair2, ospair1
                htool.fill_object(f"{input_path}/ZCand", ZCand, w_fake_up)
                htool.fill_object(f"{input_path}/nZCand", nZCand, w_fake_up)
            elif REGION == "SR":
                # make A candidate
                if abs(ospair1.Mass() - mA) < abs(ospair2.Mass() - mA):
                    Acand, nAcand = ospair1, ospair2
                else:
                    Acand, nAcand = ospair2, ospair1
                htool.fill_object(f"{input_path}/Acand", Acand, w_fake_up)
                htool.fill_object(f"{input_path}/nAcand", nAcand, w_fake_up)
                htool.fill_hist3d(f"{input_path}/fake_ttX_mMM", score_fake, score_ttX, Acand.Mass(), w_fake_up,
                                    100, 0., 1., 100, 0., 1., 3000, 0., 300.)                

            # Down
            input_path = f"fake/objects/{REGION}/Down"
            htool.fill_object(f"{input_path}/METv", METv, w_fake_down)
            htool.fill_muons(f"{input_path}/muons", muons, w_fake_down)
            htool.fill_jets(f"{input_path}/jets", jets, w_fake_down)
            htool.fill_jets(f"{input_path}/bjets", bjets, w_fake_down)
            
            # fill input variables
            input_path = f"fake/inputs/{REGION}/Down"
            htool.fill_hist(f"{input_path}/mu1_px", mu1_px, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_py", mu1_py, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_pz", mu1_pz, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu1_mass",mu1_mass, w_fake_down, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu1_charge", mu1_charge, w_fake_down, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/mu2_px", mu2_px, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_py", mu2_py, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_pz", mu2_pz, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu2_mass",mu2_mass, w_fake_down, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu2_charge", mu2_charge, w_fake_down, 4, -2., 2.)
            htool.fill_hist(f"{input_path}/mu3_px", mu3_px, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu3_py", mu3_py, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu3_pz", mu3_pz, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/mu3_mass",mu3_mass, w_fake_down, 100, 0., 100.)
            htool.fill_hist(f"{input_path}/mu3_charge", mu3_charge, w_fake_down, 4, -2., 2.) 
            htool.fill_hist(f"{input_path}/j1_px", j1_px, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_py", j1_py, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_pz", j1_pz, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j1_mass", j1_mass, w_fake_down, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j1_bscore", j1_bscore, w_fake_down, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/j2_px", j2_px, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_py", j2_py, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_pz", j2_pz, w_fake_down, 400, -200., 200.)
            htool.fill_hist(f"{input_path}/j2_mass", j2_mass, w_fake_down, 200, 0., 200.)
            htool.fill_hist(f"{input_path}/j2_bscore", j2_bscore, w_fake_down, 100, 0., 1.) 
            htool.fill_hist(f"{input_path}/dR_mu1mu2", dR_mu1mu2, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu1mu3", dR_mu1mu3, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_mu2mu3", dR_mu2mu3, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1j2", dR_j1j2, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1mu1", dR_j1mu1, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1mu2", dR_j1mu2, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j1mu3", dR_j1mu3, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2mu1", dR_j2mu1, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2mu2", dR_j2mu2, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/dR_j2mu3", dR_j2mu3, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/HT", HT, w_fake_down, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/Nj", Nj, w_fake_down, 20, 0., 20.)
            htool.fill_hist(f"{input_path}/Nb", Nb, w_fake_down, 10, 0., 10.)
            htool.fill_hist(f"{input_path}/LT", LT, w_fake_down, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/MET", MET, w_fake_down, 1000, 0., 1000.)
            htool.fill_hist(f"{input_path}/avg_dRjets", avg_dRjets, w_fake_down, 100, 0., 10.)
            htool.fill_hist(f"{input_path}/avg_bscore", avg_bscore, w_fake_down, 100, 0., 10.)
            
            # variables with mass point dependence
            input_path = f"fake/{MASS_POINT}/{REGION}/Down"
            mA = float(MASS_POINT.split("_")[1][2:])
            
            htool.fill_hist(f"{input_path}/score_fake", score_fake, w_fake_down, 100, 0., 1.)
            htool.fill_hist(f"{input_path}/score_ttX", score_ttX, w_fake_down, 100, 0., 1.)
            if REGION == "ZFake":
                # make Z candidate
                if abs(ospair1.Mass() - 91.2) < abs(ospair2.Mass() - 91.2):
                    ZCand, nZCand = ospair1, ospair2
                else:
                    ZCand, nZCand = ospair2, ospair1
                htool.fill_object(f"{input_path}/ZCand", ZCand, w_fake_down)
                htool.fill_object(f"{input_path}/nZCand", nZCand, w_fake_down)
            elif REGION == "SR":
                # make A candidate
                if abs(ospair1.Mass() - mA) < abs(ospair2.Mass() - mA):
                    Acand, nAcand = ospair1, ospair2
                else:
                    Acand, nAcand = ospair2, ospair1
                htool.fill_object(f"{input_path}/Acand", Acand, w_fake_down)
                htool.fill_object(f"{input_path}/nAcand", nAcand, w_fake_down)
                htool.fill_hist3d(f"{input_path}/fake_ttX_mMM", score_fake, score_ttX, Acand.Mass(), w_fake_down,
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

                loop(evt, clf_fake, clf_ttX, syst, htool)
        f.Close()
        f = TFile.Open(fkey_emu)
        for syst in SYSTs + WEIGHTSYSTs:
            for evt in f.Events:
                # check double counting
                this_evt = (evt.run, evt.event, evt.lumi)
                if this_evt in events[syst]:
                    continue

                loop(evt, clf_fake, clf_ttX, syst, htool)
        f.Close()

    elif SAMPLE == "DATA" and CHANNEL == "3Mu":
        fkey = f"{SAMPLE_DIR}/DATA/Selector_DoubleMuon.root"
        f = TFile.Open(fkey)
        for syst in SYSTs + WEIGHTSYSTs:
            for evt in f.Events:
                loop(evt, clf_fake, clf_ttX, syst, htool)
        f.Close()

    elif SAMPLE in MCs or "TTToHcToWA" in SAMPLE:
        fkey = f"{SAMPLE_DIR}/MCSamples/Selector_{SAMPLE}.root"
        f = TFile.Open(fkey)
        for syst in SYSTs + WEIGHTSYSTs:
            for evt in f.Events:
                loop(evt, clf_fake, clf_ttX, syst, htool)
        f.Close()
    else:
        raise(AttributeError)
    htool.save()

if __name__ == "__main__":
    main()
    print("End process")
