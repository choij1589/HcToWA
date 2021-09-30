# train signal vs fake, signal vs ttX classifier
# for every mass point
# This script should be run in KNU server
# to use multi-GPU support
import warnings

warnings.filterwarnings(action='ignore')

import argparse
from itertools import product
import multiprocessing as mp
import pickle 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from ROOT import TFile, TH1D

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from MLTools import DEVICE, EPOCHS, BATCH_SIZE
from MLTools import MyDataset
from MLTools import SelfNormDNN
from MLTools import EarlyStopping
from MLTools import train, evaluate, predict

parser = argparse.ArgumentParser()
parser.add_argument("--mass",
                    "-m",
                    default=None,
                    required=True,
                    type=str,
                    help="signal mass point")
parser.add_argument("--channel",
                    "-c",
                    default=None,
                    required=True,
                    type=str,
                    help="channel")
parser.add_argument("--target",
                    "-t",
                    default=None,
                    required=True,
                    type=str,
                    help="target process to discriminate")
parser.add_argument("--dry_run",
                    "-d",
                    default=False,
                    action="store_true",
                    help="dry run")
args = parser.parse_args()

# global variables
MASS_POINT = args.mass
SIGNAL = f"TTToHcToWA_AToMuMu_{MASS_POINT}"
CHANNEL = args.channel
TARGET = args.target

# hyperparameters
learning_rates = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
hidden_nodes = [64, 128, 192, 256]

# reduce parameters for dry run
if args.dry_run:
    EPOCHS = 300
    learning_rates = [0.1]
    hidden_layers = [64]


class DataManager:
    def __init__(self, sig, bkg, features):
        self.sig = sig
        self.bkg = bkg
        self.features = features
        self.class_weights = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

        self.sig['label'] = 0
        self.bkg['label'] = 1

    def get_dataloaders(self):
        print("get dataloaders for signal and background samples")
        # split samples to train / validation / test sample
        sig_cut1 = int(len(self.sig) * 0.65)
        sig_cut2 = int(len(self.sig) * 0.8)
        bkg_cut1 = int(len(self.bkg) * 0.65)
        bkg_cut2 = int(len(self.bkg) * 0.8)
        train_sig, val_sig, test_sig = self.sig[:sig_cut1], self.sig[
            sig_cut1:sig_cut2], self.sig[sig_cut2:]
        train_bkg, val_bkg, test_bkg = self.bkg[:bkg_cut1], self.bkg[
            bkg_cut1:bkg_cut2], self.bkg[bkg_cut2:]
        self.__update_class_weight(train_sig, train_bkg)

        train_sample = shuffle(pd.concat([train_sig, train_bkg]))
        val_sample = shuffle(pd.concat([val_sig, val_bkg]))
        test_sample = shuffle(pd.concat([test_sig, test_bkg]))

        X_train, self.y_train = train_sample[
            self.features].to_numpy(), train_sample[['label']].to_numpy()
        X_val, self.y_val = val_sample[self.features].to_numpy(), val_sample[[
            'label'
        ]].to_numpy()
        X_test, self.y_test = test_sample[
            self.features].to_numpy(), test_sample[['label']].to_numpy()

        scaler = StandardScaler()
        scaler.fit(X_train)
        # save scaler to use in the final analyzer
        pickle.dump(scaler, open(f'Outputs/{CHANNEL}/{MASS_POINT}/models/scaler_{TARGET}.pkl', 'wb'))
        
        self.X_train, self.X_val, self.X_test = scaler.transform(
            X_train), scaler.transform(X_val), scaler.transform(X_test)

        train_loader = DataLoader(MyDataset(self.X_train, self.y_train),
                                  batch_size=BATCH_SIZE,
                                  num_workers=0,
                                  pin_memory=True,
                                  shuffle=True)
        val_loader = DataLoader(MyDataset(self.X_val, self.y_val),
                                batch_size=BATCH_SIZE,
                                num_workers=0,
                                pin_memory=True,
                                shuffle=False)
        test_loader = DataLoader(MyDataset(self.X_test, self.y_test),
                                 batch_size=BATCH_SIZE,
                                 num_workers=0,
                                 pin_memory=True,
                                 shuffle=False)

        return (train_loader, val_loader, test_loader)

    def __update_class_weight(self, signal, bkg):
        class_weights = list()
        class_weights.append(len(signal))
        class_weights.append(len(bkg))
        class_weights = [x / (len(signal) + len(bkg)) for x in class_weights]
        class_weights = [1 / x for x in class_weights]
        class_weights = torch.FloatTensor(class_weights)
        self.class_weights = class_weights


# load dataset
# note that it will be copied in the subprocesses
print(f"Loading datasets for {MASS_POINT}...")
signal = shuffle(pd.read_csv(f"Outputs/{CHANNEL}/CSV/{SIGNAL}.csv", index_col=0))
fake = shuffle(pd.read_csv(f"Outputs/{CHANNEL}/CSV/fake.csv", index_col=0))
ttX = shuffle(pd.read_csv(f"Outputs/{CHANNEL}/CSV/ttX.csv", index_col=0))
rare = shuffle(pd.read_csv(f"Outputs/{CHANNEL}/CSV/rare.csv", index_col=0))
VV = shuffle(pd.read_csv(f"Outputs/{CHANNEL}/CSV/VV.csv", index_col=0))
DY = shuffle(pd.read_csv(f"Outputs/{CHANNEL}/CSV/DY.csv", index_col=0))
ZG = shuffle(pd.read_csv(f"Outputs/{CHANNEL}/CSV/ZG.csv", index_col=0))
others = pd.concat([rare, VV, DY, ZG])

if TARGET == "fake":
    bkg = fake
elif TARGET == "ttX":
    bkg = ttX
elif TARGET == "others":
    bkg = others
else:
    print(f"wrong target {TARGET}")
    raise (AttributeError)

if CHANNEL == "1E2Mu":
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
    print("wrong channel")
    raise (AttributeError)


def optimize(lr, n_hidden):
    print(f"training with learning rate {lr}, nodes {n_hidden}")
    sig_sub = signal.copy()
    bkg_sub = bkg.copy()
    manager = DataManager(sig_sub, bkg_sub, features)
    train_loader, val_loader, test_loader = manager.get_dataloaders()
    class_weights = manager.class_weights

    model = SelfNormDNN(len(features), 2, n_hidden=n_hidden).to(DEVICE)
    # model.share_memory()
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.3)
    early_stopping = EarlyStopping(
        patience=5,
        path=
        f"Outputs/{CHANNEL}/{MASS_POINT}/models/chkpoint_sig_vs_{TARGET}_lr-{lr}_n_hidden-{n_hidden}.pt"
    )

    epochs = 0
    history = dict()
    history['loss'] = list()
    history['acc'] = list()
    history['val_loss'] = list()
    history['val_acc'] = list()

    for epoch in range(1, EPOCHS + 1):
        loss, acc = train(model, train_loader, optimizer, class_weights, epoch)
        scheduler.step()
        test_loss, test_acc = evaluate(model, val_loader, class_weights, epoch)

        history['loss'].append(loss)
        history['acc'].append(acc)
        history['val_loss'].append(test_loss)
        history['val_acc'].append(test_acc)

        early_stopping(test_loss, model)
        epochs = epoch
        if early_stopping.early_stop:
            print(f"Early stopping in epoch {epoch}")
            print(f"final training acc: {acc}, test acc: {test_acc}")
            break

    # train / validation loss and accuracy
    epochs = np.arange(1, epochs + 1)
    plt.figure(figsize=(24, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['loss'], label="train loss")
    plt.plot(epochs, history['val_loss'], label="validation loss")
    plt.xlabel("epoch")
    plt.legend(loc='best')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['acc'], label="train accuracy")
    plt.plot(epochs, history['val_acc'], label="validation accuracy")
    plt.xlabel("epoch")
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig(
        f"Outputs/{CHANNEL}/{MASS_POINT}/figures/training_sig_vs_{TARGET}_learning_rate_{lr}_hidden_layers_{n_hidden}.png"
    )

    # Confusion matrix and ROC curve
    X_train, X_valid, X_test = manager.X_train, manager.X_val, manager.X_test
    y_train, y_valid, y_test = manager.y_train, manager.y_val, manager.y_test

    pred_test = predict(model, X_test)
    confusion_matrix = metrics.confusion_matrix(y_test, pred_test)

    pred_prob_train = predict(model, X_train, prob=True)
    pred_prob_valid = predict(model, X_valid, prob=True)
    pred_prob_test = predict(model, X_test, prob=True)

    score_train = TH1D(
        f"score_train_sig_vs_{TARGET}_lr-{lr}_n_hidden-{n_hidden}", "", 100,
        0., 1.)
    score_test = TH1D(
        f"score_test_sig_vs_{TARGET}_lr-{lr}_n_hidden-{n_hidden}", "", 100, 0.,
        1.)
    score_train.SetDirectory(0)
    score_test.SetDirectory(0)

    for score in pred_prob_train:
        score_train.Fill(score)
    for score in pred_prob_test:
        score_test.Fill(score)

    fpr_train, tpr_train, th_train = metrics.roc_curve(y_train,
                                                       pred_prob_train,
                                                       pos_label=0)
    fpr_valid, tpr_valid, th_valid = metrics.roc_curve(y_valid,
                                                       pred_prob_valid,
                                                       pos_label=0)
    fpr_test, tpr_test, th_test = metrics.roc_curve(y_test,
                                                    pred_prob_test,
                                                    pos_label=0)
    auc_train = metrics.auc(fpr_train, tpr_train)
    auc_valid = metrics.auc(fpr_valid, tpr_valid)
    auc_test = metrics.auc(fpr_test, tpr_test)

    # KS score using scipy
    ksprob_scipy = stats.kstest(pred_prob_train, pred_prob_test).pvalue
    ksprob_root = score_train.KolmogorovTest(score_test, option='X')
    
    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title(
        f"Confusion Matrix\nlearning rate:{lr}\nhidden layers:{n_hidden}")
    ax1 = sns.heatmap(confusion_matrix,
                      annot=True,
                      fmt='d',
                      square=False,
                      xticklabels=['signal', 'background'],
                      yticklabels=['signal', 'background'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title(f'ROC curve\nks-prob: {round(ksprob_scipy, 3)} ~ {round(ksprob_root, 3)}')
    ax2.plot(tpr_train,
             1 - fpr_train,
             'r--',
             label=f'train roc, AUC={round(auc_train, 3)}')
    ax2.plot(tpr_valid,
             1 - fpr_valid,
             'b--',
             label=f'valid roc, AUC={round(auc_valid, 3)}')
    ax2.plot(tpr_test,
             1 - fpr_test,
             'c--',
             label=f'test roc, AUC={round(auc_test, 3)}')
    ax2.legend(loc='best')
    ax2.set_xlabel('signal efficiency')
    ax2.set_ylabel('background rejection')

    plt.savefig(
        f"Outputs/{CHANNEL}/{MASS_POINT}/figures/roc_sig_vs_{TARGET}_learning_rate_{lr}_hidden_layers_{n_hidden}.png"
    )
    results = {"index": f"lr-{lr}_hidden-{n_hidden}",
			   "score_train": score_train,
			   "score_test": score_test,
			   "auc_train": auc_train,
			   "auc_valid": auc_valid,
			   "auc_test": auc_test,
			   "ksprob_scipy": ksprob_scipy,
			   "ksprob_root": ksprob_root}

    return results

def collect_scores(results):
    global rtfile
    global h_params
    global auc_trains
    global auc_valids
    global auc_tests
    global ksprobs_scipy
    global ksprobs_root
    rtfile.cd()
    results["score_train"].Write()
    results["score_test"].Write()
    h_params.append(results["index"])
    auc_trains.append(results["auc_train"])
    auc_valids.append(results["auc_valid"])
    auc_tests.append(results["auc_test"])
    ksprobs_scipy.append(results["ksprob_scipy"])
    ksprobs_root.append(results["ksprob_root"])


if __name__ == "__main__":
    rtfile = TFile(f"Outputs/{CHANNEL}/{MASS_POINT}/scores_vs_{TARGET}.root",
                   "recreate")
    h_params = list()
    auc_trains = list()
    auc_valids = list()
    auc_tests = list()
    ksprobs_scipy = list()
    ksprobs_root = list()

    mp.set_start_method("spawn")
    pool = mp.Pool(processes=6)

    for (lr, n_hidden) in product(learning_rates, hidden_nodes):
        pool.apply_async(optimize, (
            lr,
            n_hidden,
        ), callback=collect_scores)
    pool.close()
    pool.join()

    rtfile.Close()

    frame = {"index": h_params, "auc_train": auc_trains, 
			"auc_valid": auc_valids, "auc_test": auc_tests,
			"ksprob_scipy": ksprobs_scipy, "ksprob_root": ksprobs_root}

    df = pd.DataFrame(frame)
    df.to_csv(f"Outputs/{CHANNEL}/{MASS_POINT}/metrics_vs_{TARGET}.csv")
