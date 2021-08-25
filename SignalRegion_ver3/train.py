import warnings

from numpy.lib.histograms import _histogramdd_dispatcher

warnings.filterwarnings(action='ignore')

import argparse
from itertools import product
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

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
                    help='signal mass point')
parser.add_argument("--channel",
                    "-c",
                    default=None,
                    required=True,
                    type=str,
                    help='channel')
args = parser.parse_args()

# global variables
MASS_POINT = args.mass
SIGNAL = f"TTToHcToWA_AToMuMu_{MASS_POINT}"
CHANNEL = args.channel
learning_rates = [
    1e-4, 5e-4, 1e-3, 3e-3, 6e-3, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.4
]
hidden_layers = [512, 216, 128]


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
        self.X_train, self.X_val, self.X_test = scaler.transform(
            X_train), scaler.transform(X_val), scaler.transform(X_test)

        train_loader = DataLoader(MyDataset(self.X_train, self.y_train),
                                  batch_size=BATCH_SIZE,
                                  num_workers=4,
                                  shuffle=True)
        val_loader = DataLoader(MyDataset(self.X_val, self.y_val),
                                batch_size=BATCH_SIZE,
                                num_workers=4,
                                shuffle=False)
        test_loader = DataLoader(MyDataset(self.X_test, self.y_test),
                                 batch_size=BATCH_SIZE,
                                 num_workers=4,
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
print(f"Loading datasets for {MASS_POINT}...")
signal = pd.read_csv(f"Outputs/{CHANNEL}/CSV/{SIGNAL}.csv", index_col=0)
fake = pd.read_csv(f"Outputs/{CHANNEL}/CSV/fake.csv", index_col=0)
ttX = pd.read_csv(f"Outputs/{CHANNEL}/CSV/ttX.csv", index_col=0)
rare = pd.read_csv(f"Outputs/{CHANNEL}/CSV/rare.csv", index_col=0)
VV = pd.read_csv(f"Outputs/{CHANNEL}/CSV/VV.csv", index_col=0)
DY = pd.read_csv(f"Outputs/{CHANNEL}/CSV/DY.csv", index_col=0)
ZG = pd.read_csv(f"Outputs/{CHANNEL}/CSV/ZG.csv", index_col=0)
others = pd.concat([rare, VV, DY, ZG])

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
    print("features for 3Mu are not defined yet")
    raise (AttributeError)
else:
    print("wrong channel")
    raise (AttributeError)

# signal vs fake
manager = DataManager(signal, fake, features)
print("training signal vs fake...")
train_loader, val_loader, test_loader = manager.get_dataloaders()
class_weights = manager.class_weights
auc_max = 0.
for lr, n_hidden in product(learning_rates, hidden_layers):
    print(f"training with learning rate {lr}, hidden layers {n_hidden}")
    model = SelfNormDNN(len(features), 2, n_hidden=n_hidden,
                        batch_norm=False).to(DEVICE)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    early_stopping = EarlyStopping(
        patience=5,
        path=
        f"Outputs/{CHANNEL}/{MASS_POINT}/models/chkpoint_sig_vs_fake_lr_{lr}_n_hidden_{n_hidden}.pt"
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
    epochs = np.arange(1, epochs + 1)

    X_train, X_test = manager.X_train, manager.X_test
    y_train, y_test = manager.y_train, manager.y_test

    pred_test = predict(model, X_test)
    pred_prob_train = predict(model, X_train, prob=True)
    pred_prob_test = predict(model, X_test, prob=True)
    cm = metrics.confusion_matrix(y_test, pred_test)
    fpr_train, tpr_train, thresh_train = metrics.roc_curve(y_train,
                                                           pred_prob_train,
                                                           pos_label=0)
    fpr, tpr, thresh = metrics.roc_curve(y_test, pred_prob_test, pos_label=0)
    auc_train = metrics.auc(fpr_train, tpr_train)
    auc = metrics.auc(fpr, tpr)
    print(f"train auc:{round(auc_train, 3)}, test auc:{round(auc, 3)}")
    print(f"auc difference: {round((auc_train-auc)/auc_train*100, 2)}%\n")
    if auc > auc_max:
        auc_max = auc

    # no need to save if train & test auc differs more than 3%
    if abs(auc_train - auc) / auc > 0.03:
        continue

    # train / validation loss and accuracy
    if auc_max == auc:
        torch.save(
            model.state_dict(),
            f"Outputs/{CHANNEL}/{MASS_POINT}/models/chkpoint_sig_vs_fake.pt")
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
            f"Outputs/{CHANNEL}/{MASS_POINT}/figures/training_sig_vs_fake.png")

        plt.figure(figsize=(14, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title(
            f"Confusion Matrix\nlearning rate:{lr}\nhidden layers:{n_hidden}")
        ax1 = sns.heatmap(cm,
                          annot=True,
                          fmt='d',
                          square=False,
                          xticklabels=['signal', 'background'],
                          yticklabels=['signal', 'background'])
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title(
            f'ROC curve\nAUC(train):{round(auc_train, 3)}\nAUC(test):{round(auc, 3)}'
        )
        ax2.plot(tpr_train, 1 - fpr_train, 'r--', label='train roc')
        ax2.plot(tpr, 1 - fpr, 'b--', label='test roc')
        ax2.legend(loc='best')
        ax2.set_xlabel('signal efficiency')
        ax2.set_ylabel('background rejection')

        plt.savefig(
            f"Outputs/{CHANNEL}/{MASS_POINT}/figures/roc_sig_vs_fake.png")

# signal vs ttX
manager = DataManager(signal, ttX, features)
print("training signal vs ttX...")
train_loader, val_loader, test_loader = manager.get_dataloaders()
class_weights = manager.class_weights
auc_max = 0.
for lr, n_hidden in product(learning_rates, hidden_layers):
    print(f"training with learning rate {lr}, hidden layers {n_hidden}")
    model = SelfNormDNN(len(features), 2, n_hidden=n_hidden,
                        batch_norm=False).to(DEVICE)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    early_stopping = EarlyStopping(
        patience=5,
        path=
        f"Outputs/{CHANNEL}/{MASS_POINT}/models/chkpoint_sig_vs_ttX_lr_{lr}_n_hidden_{n_hidden}.pt"
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
    epochs = np.arange(1, epochs + 1)

    X_train, X_test = manager.X_train, manager.X_test
    y_train, y_test = manager.y_train, manager.y_test

    pred_test = predict(model, X_test)
    pred_prob_train = predict(model, X_train, prob=True)
    pred_prob_test = predict(model, X_test, prob=True)
    cm = metrics.confusion_matrix(y_test, pred_test)
    fpr_train, tpr_train, thresh_train = metrics.roc_curve(y_train,
                                                           pred_prob_train,
                                                           pos_label=0)
    fpr, tpr, thresh = metrics.roc_curve(y_test, pred_prob_test, pos_label=0)
    auc_train = metrics.auc(fpr_train, tpr_train)
    auc = metrics.auc(fpr, tpr)
    print(f"train auc:{round(auc_train, 3)}, test auc:{round(auc, 3)}")
    print(f"auc difference: {round((auc_train-auc)/auc_train*100, 2)}%\n")
    if auc > auc_max:
        auc_max = auc

    # no need to save if train & test auc differs more than 3%
    if abs(auc_train - auc) / auc > 0.03:
        continue

    # train / validation loss and accuracy
    if auc_max == auc:
        torch.save(
            model.state_dict(),
            f"Outputs/{CHANNEL}/{MASS_POINT}/models/chkpoint_sig_vs_ttX.pt")
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
            f"Outputs/{CHANNEL}/{MASS_POINT}/figures/training_sig_vs_ttX.png")

        plt.figure(figsize=(14, 6))
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title(
            f"Confusion Matrix\nlearning rate:{lr}\nhidden layers:{n_hidden}")
        ax1 = sns.heatmap(cm,
                          annot=True,
                          fmt='d',
                          square=False,
                          xticklabels=['signal', 'background'],
                          yticklabels=['signal', 'background'])
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title(
            f'ROC curve\nAUC(train):{round(auc_train, 3)}\nAUC(test):{round(auc, 3)}'
        )
        ax2.plot(tpr_train, 1 - fpr_train, 'r--', label='train roc')
        ax2.plot(tpr, 1 - fpr, 'b--', label='test roc')
        ax2.legend(loc='best')
        ax2.set_xlabel('signal efficiency')
        ax2.set_ylabel('background rejection')

        plt.savefig(
            f"Outputs/{CHANNEL}/{MASS_POINT}/figures/roc_sig_vs_ttX.png")