import argparse
from itertools import combinations
import numpy as np
import pandas as pd
from ROOT import TFile
from Scripts.DataFormat import Particle, get_leptons, get_jets
from Scripts.HistTools import HistTool
from SignalTools import set_global, select_loosen, get_tight_leptons, get_prompt_leptons, get_weight


parser = argparse.ArgumentParser()
parser.add_argument("--sample", "-s", default=None, required=True,
                    type=str, help="sample name")
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help="channel")
#parser.add_argument("--mass", "-m", default=None, required=True, type=str, help="signal mass point")
args = parser.parse_args()

# define global variables
SAMPLE = args.sample
CHANNEL = args.channel
set_global(CHANNEL, SAMPLE)
# mHc = float(args.mass.split("_")[0][3:])
# mA = float(args.mass.split("_")[1][2:])

if CHANNEL == "1E2Mu":
    DATA = ["MuonEG", "DoubleMuon"]
    BKGs = ["rare", "ttX", "VV", "fake", "conv"]
    MCs = ["DY", "ZG", "rare", "ttX", "VV"]
    SAMPLE_DIR = "/root/workspace/HcToWA/Samples/Selector/2017/Skim1E2Mu__/"
    # OUTFILE = f"/root/workspace/HcToWA/SignalRegion/Outputs/{CHANNEL}/{args.mass}/{SAMPLE}.root"
elif CHANNEL == "3Mu":
    DATA = ["DoubleMuon"]
    BKGs = ["rake", "ttX", "VV", "fake"]
    MCs = ["DY", "ZG", "rare", "ttX", "VV"]
    SAMPLE_DIR = "/root/workspace/HcToWA/Samples/Selector/2017/Skim3Mu__/"
    # OUTFILE = f"/root/workspace/HcToWA/SignalRegion/Outputs/{CHANNEL}/{args.mass}/{SAMPLE}.root"
else:
    print(f"Wrong channel {CHANNEL}")
    raise(AttributeError)

# features to write in the CSV file
if CHANNEL == "1E2Mu":
    features = ['run', 'event', 'lumi',
                'mu1_px', 'mu1_py', 'mu1_pz', 'mu1_mass', 'mu1_charge',
                'mu2_px', 'mu2_py', 'mu2_pz', 'mu2_mass', 'mu2_charge',
                'ele_px', 'ele_py', 'ele_pz', 'ele_mass', 'ele_charge',
                'j1_px', 'j1_py', 'j1_pz', 'j1_mass', 'j1_bscore',
                'j2_px', 'j2_py', 'j2_pz', 'j2_mass', 'j2_bscore',
                'dR_mu1mu2', 'dR_mu1ele', 'dR_mu2ele', 'dR_j1j2', 'dR_j1ele', 'dR_j2ele',
                'HT', 'MT', 'Nj', 'Nb', 'LT', 'MET',
                'avg_dRjets', 'avg_bscore', 'weight']
elif CHANNEL == "3Mu":
    features = ['run', 'event', 'lumi',
                'mu1_px', 'mu1_py', 'mu1_pz', 'mu1_mass', 'mu1_charge',
                'mu2_px', 'mu2_py', 'mu2_pz', 'mu2_mass', 'mu2_charge',
                'mu3_px', 'mu3_py', 'mu3_pz', 'mu3_mass', 'mu3_charge',
                'j1_px', 'j1_py', 'j1_pz', 'j1_mass', 'j1_bscore',
                'j2_px', 'j2_py', 'j2_pz', 'j2_mass', 'j2_bscore',
                'dR_mu1mu2', 'dR_mu1mu3', 'dR_mu2mu3', 'dR_j1j2',
                'dR_j1mu1', 'dR_j1mu2', 'dR_j1mu3',
                'dR_j2mu1', 'dR_j2mu2', 'dR_j2mu3',
                'HT', 'Nj', 'Nb', 'LT', 'MET',
                'avg_dRjets', 'avg_bscore', 'weight']
else:
    raise(AttributeError)

# make dictionary to store features
data = dict()
for feature in features:
    data[feature] = list()
#htool = HistTool(outfile=OUTFILE)

def loop(evt, syst, data):
    muons, electrons = get_leptons(evt)
    jets, bjets = get_jets(evt)
    METv = Particle(evt.METv_pt, evt.METv_eta, evt.METv_phi, 0.)

    # event selection
    if not select_loosen(evt, muons, electrons, jets, bjets, syst):
        return

    # check if pass tight IDs
    muons_tight, electrons_tight = get_tight_leptons(muons, electrons)

    # tight & prompt flags
    tight_flag = False
    if CHANNEL == "1E2Mu":
        if len(muons_tight) == 2 and len(electrons_tight) == 1:
            tight_flag = True
    elif CHANNEL == "3Mu":
        if len(muons_tight) == 3 and len(electrons_tight) == 0:
            tight_flag = True
    else:
        raise(AttributeError)

    if not tight_flag:
        return
       
    prompt_flag = False
    muons_prompt, electrons_prompt = get_prompt_leptons(muons, electrons)
    if CHANNEL == "1E2Mu":
        if (len(muons_prompt) == 2 and len(electrons_prompt) == 1):
            prompt_flag = True
    elif CHANNEL == "3Mu":
        if (len(muons_prompt) == 3 and len(electrons_prompt) == 0):
            prompt_flag = True
    else:
        raise(AttributeError)

    if SAMPLE in ["DY", "ZG"] and not prompt_flag:
        return
    if SAMPLE == "fake" and prompt_flag:
        return

    # set weight
    weight = 1.
    if not evt.IsDATA:
        weight = get_weight(evt, syst)
    
    leptons = electrons + muons

    # make event array
    if CHANNEL == "1E2Mu":
        MT = np.sqrt(2*electrons[0].Pt()*METv.Pt()*(1. - np.cos(electrons[0].DeltaPhi(METv))))
        data['run'].append(evt.run)
        data['event'].append(evt.event)
        data['lumi'].append(evt.lumi)
        data['mu1_px'].append(muons[0].P4().Px())
        data['mu1_py'].append(muons[0].P4().Py())
        data['mu1_pz'].append(muons[0].P4().Pz())
        data['mu1_mass'].append(muons[0].Mass())
        data['mu1_charge'].append(muons[0].Charge())
        data['mu2_px'].append(muons[1].P4().Px())
        data['mu2_py'].append(muons[1].P4().Py())
        data['mu2_pz'].append(muons[1].P4().Pz())
        data['mu2_mass'].append(muons[1].Mass())
        data['mu2_charge'].append(muons[1].Charge())
        data['ele_px'].append(electrons[0].P4().Px())
        data['ele_py'].append(electrons[0].P4().Py())
        data['ele_pz'].append(electrons[0].P4().Pz())
        data['ele_mass'].append(electrons[0].Mass())
        data['ele_charge'].append(electrons[0].Charge())

        data['j1_px'].append(jets[0].P4().Px())
        data['j1_py'].append(jets[0].P4().Py())
        data['j1_pz'].append(jets[0].P4().Pz())
        data['j1_mass'].append(jets[0].Mass())
        data['j1_bscore'].append(jets[0].BtagScore())
        data['j2_px'].append(jets[1].P4().Px())
        data['j2_py'].append(jets[1].P4().Py())
        data['j2_pz'].append(jets[1].P4().Pz())
        data['j2_mass'].append(jets[1].Mass())
        data['j2_bscore'].append(jets[1].BtagScore())

        data['dR_mu1mu2'].append(muons[0].DeltaR(muons[1]))
        data['dR_mu1ele'].append(muons[0].DeltaR(electrons[0]))
        data['dR_mu2ele'].append(muons[1].DeltaR(electrons[0]))
        data['dR_j1j2'].append(jets[0].DeltaR(jets[1]))
        data['dR_j1ele'].append(jets[0].DeltaR(electrons[0]))
        data['dR_j2ele'].append(jets[1].DeltaR(electrons[0]))
        comb = combinations(range(len(jets)), 2)
        dRjets = []
        for elements in comb:
            j1 = jets[elements[0]]
            j2 = jets[elements[1]]
            dRjets.append(j1.DeltaR(j2))
        data['avg_dRjets'].append(sum(dRjets)/len(dRjets))
        data['avg_bscore'].append(sum([x.BtagScore() for x in jets])/len(jets))

        data['HT'].append(sum([x.Pt() for x in jets]))
        data['MT'].append(MT)
        data['LT'].append(sum([x.Pt() for x in leptons]))
        data['MET'].append(METv.Pt())
        data['Nj'].append(len(jets))
        data['Nb'].append(len(bjets))
        data['weight'].append(weight)
    elif CHANNEL == "3Mu":
        data['run'].append(evt.run)
        data['event'].append(evt.event)
        data['lumi'].append(evt.lumi)
        data['mu1_px'].append(muons[0].P4().Px())
        data['mu1_py'].append(muons[0].P4().Py())
        data['mu1_pz'].append(muons[0].P4().Pz())
        data['mu1_mass'].append(muons[0].Mass())
        data['mu1_charge'].append(muons[0].Charge())
        data['mu2_px'].append(muons[1].P4().Px())
        data['mu2_py'].append(muons[1].P4().Py())
        data['mu2_pz'].append(muons[1].P4().Pz())
        data['mu2_mass'].append(muons[1].Mass())
        data['mu2_charge'].append(muons[1].Charge())
        data['mu3_px'].append(muons[2].P4().Px())
        data['mu3_py'].append(muons[2].P4().Py())
        data['mu3_pz'].append(muons[2].P4().Pz())
        data['mu3_mass'].append(muons[2].Mass())
        data['mu3_charge'].append(muons[2].Charge())

        data['j1_px'].append(jets[0].P4().Px())
        data['j1_py'].append(jets[0].P4().Py())
        data['j1_pz'].append(jets[0].P4().Pz())
        data['j1_mass'].append(jets[0].Mass())
        data['j1_bscore'].append(jets[0].BtagScore())
        data['j2_px'].append(jets[1].P4().Px())
        data['j2_py'].append(jets[1].P4().Py())
        data['j2_pz'].append(jets[1].P4().Pz())
        data['j2_mass'].append(jets[1].Mass())
        data['j2_bscore'].append(jets[1].BtagScore())

        data['dR_mu1mu2'].append(muons[0].DeltaR(muons[1]))
        data['dR_mu1mu3'].append(muons[0].DeltaR(muons[2]))
        data['dR_mu2mu3'].append(muons[1].DeltaR(muons[2]))
        data['dR_j1j2'].append(jets[0].DeltaR(jets[1]))
        data['dR_j1mu1'].append(jets[0].DeltaR(muons[0]))
        data['dR_j1mu2'].append(jets[0].DeltaR(muons[1]))
        data['dR_j1mu3'].append(jets[0].DeltaR(muons[2]))
        data['dR_j2mu1'].append(jets[1].DeltaR(muons[0]))
        data['dR_j2mu2'].append(jets[1].DeltaR(muons[1]))
        data['dR_j2mu3'].append(jets[1].DeltaR(muons[2]))
        comb = combinations(range(len(jets)), 2)
        dRjets = []
        for elements in comb:
            j1 = jets[elements[0]]
            j2 = jets[elements[1]]
            dRjets.append(j1.DeltaR(j2))
        data['avg_dRjets'].append(sum(dRjets)/len(dRjets))
        data['avg_bscore'].append(sum([x.BtagScore() for x in jets])/len(jets))

        data['HT'].append(sum([x.Pt() for x in jets]))
        data['LT'].append(sum([x.Pt() for x in leptons]))
        data['MET'].append(METv.Pt())
        data['Nj'].append(len(jets))
        data['Nb'].append(len(bjets))
        data['weight'].append(weight)
    else:
        raise(AttributeError)

if __name__ == "__main__":
    print(f"Estimating {SAMPLE}...")
    if SAMPLE == "DATA" and CHANNEL == "1E2Mu":
        fkey_dblmu = f"{SAMPLE_DIR}/DATA/Selector_DoubleMuon.root"
        fkey_emu = f"{SAMPLE_DIR}/DATA/Selector_MuonEG.root"

        # DoubleMuon
        f = TFile.Open(fkey_dblmu)
        events = list()
        for evt in f.Events:
            this_evt = (evt.run, evt.event, evt.lumi)
            events.append(this_evt)

            loop(evt, "Central", data)
        f.Close()
        
        # MuonEG
        f = TFile.Open(fkey_emu)
        for evt in f.Events:
            this_evt = (evt.run, evt.event, evt.lumi)
            if this_evt in events:
                continue
            loop(evt, "Central", data)
        f.Close()

    elif SAMPLE == "DATA" and args.channel == "3Mu":
        fkey = f"{SAMPLE_DIR}/DATA/Selector_DoubleMuon.root"
        f = TFile.Open(fkey)
        for evt in f.Events:
            loop(evt, "Central", data)
        f.Close()

    elif SAMPLE == "fake":
        fkey = f"{SAMPLE_DIR}/MCSamples/Selector_DY.root"
        f = TFile.Open(fkey)
        for evt in f.Events:
            loop(evt, "Central", data)
        f.Close()

        fkey = f"{SAMPLE_DIR}/MCSamples/Selector_TTLL.root"
        f = TFile.Open(fkey)
        for evt in f.Events:
            loop(evt, "Central", data)
        f.Close()

    elif SAMPLE in MCs or "TTToHcToWA" in SAMPLE:
        fkey = f"{SAMPLE_DIR}/MCSamples/Selector_{SAMPLE}.root"
        f = TFile.Open(fkey)
        for evt in f.Events:
            loop(evt, "Central", data)
        f.Close()
    else:
        raise(AttributeError)

    #htool.save()
    df = pd.DataFrame(data, columns=features)
    df.to_csv(f"Outputs/{CHANNEL}/CSV/{SAMPLE}.csv")
