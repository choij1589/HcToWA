import warnings
warnings.filterwarnings(action="ignore")

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from MLTools import MyDataset, train_test_split, DNN, ConvNN, train, evaluate
from MLTools import DEVICE, BATCH_SIZE

# Initialization
parser = argparse.ArgumentParser()
parser.add_argument("--mass", "-m", default=None, required=True, type=str, help="signal mass point")
parser.add_argument("--channel", "-c", default=None, required=True, type=str, help="channel")
args = parser.parse_args()

EPOCHS = 150
SIGNAL = f"TTToHcToWA_AToMuMu_{args.mass}" 
CHANNEL = args.channel

print(f"training modle for {SIGNAL}...")

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

class_weights = list()
class_weights.append(len(signal))
class_weights.append(len(sample) - len(signal))
class_weights = [x/len(sample) for x in class_weights]
class_weights = [1/x for x in class_weights]
#class_weights.append(class_weights[1])
#class_weights.append(class_weights[1])
#class_weights.append(class_weights[1])
class_weights = torch.FloatTensor(class_weights)
print(class_weights)

scaler = MinMaxScaler()

if CHANNEL == "1E2Mu":
    #conv1 = sample[['mu1_px', 'mu1_py', 'mu1_pz', 'mu1_mass', 'mu2_px', 'mu2_py', 'mu2_pz', 'mu2_mass', 'mu1_charge', 'mu2_charge', 'dR_mu1mu2']].to_numpy()
    #conv2 = sample[['ele_px', 'ele_py', 'ele_pz', 'ele_mass', 'j1_px', 'j1_py', 'j1_pz', 'j1_mass', 'j1_bscore', 'dR_j1ele']].to_numpy()
    #conv3 = sample[['ele_px', 'ele_py', 'ele_pz', 'ele_mass', 'j2_px', 'j2_py', 'j2_pz', 'j2_mass', 'j2_bscore', 'dR_j2ele']].to_numpy()
    #conv4 = sample[['j1_px', 'j1_py', 'j1_pz', 'j1_mass', 'j2_px', 'j2_py', 'j2_pz', 'j2_mass', 'j1_bscore', 'j2_bscore', 'dR_j1j2']].to_numpy()
    #res = sample[['Nj', 'Nb', 'MT', 'HT', 'LT+MET', 'avg_dRjets', 'avg_bscore']].to_numpy()
    #features = np.concatenate((conv1, conv2, conv3, conv4, res), axis=1)

    features = sample[['mu1_px', 'mu1_py', 'mu1_pz', 'mu1_mass', 'mu1_charge',
                'mu2_px', 'mu2_py', 'mu2_pz', 'mu2_mass', 'mu2_charge',
                'ele_px', 'ele_py', 'ele_pz', 'ele_mass', 'ele_charge',
                'j1_px', 'j1_py', 'j1_pz', 'j1_mass', 'j1_bscore',
                'j2_px', 'j2_py', 'j2_pz', 'j2_mass', 'j2_bscore',
                'dR_mu1mu2', 'dR_j1j2', 'dR_j1ele', 'dR_j2ele',
                'HT', 'MT', 'Nj', 'Nb', 'LT+MET',
                'avg_dRjets', 'avg_bscore']].to_numpy()

elif CHANNEL == "3Mu":
    features = sample[['MA1', 'MA2',  "Mlll", "mu1_pt", "mu2_pt", "mu3_pt", "j1_pt", "j2_pt", "j1_btagScore", "j2_btagScore", "Nj", "Nb"]].to_numpy()

scaler.fit(features)
targets = sample[['label']].to_numpy()
weights = sample[['weight']].to_numpy()
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(features, targets, weights, train_size=0.75, shuffle=True)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

train_dataset = MyDataset(X_train, y_train, w_train)
test_dataset = MyDataset(X_test, y_test, w_test)

# already shuffled in spliting stage, no need to shuffle
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)

model = DNN(len(features[0]), 2).to(DEVICE)
#model = ConvNN(len(res[0]), 2).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
history = dict()
history['loss'] = []
history['acc'] = []
history['val_loss'] = []
history['val_acc'] = []
#history['sign'] = []
#history['val_sign'] = []
for epoch in range(1, EPOCHS+1):
    loss, accuracy = train(model, train_loader, optimizer, class_weights, epoch)
    scheduler.step()
    test_loss, test_accuracy = evaluate(model, test_loader, class_weights, epoch)
    history['loss'].append(loss)
    history['acc'].append(accuracy)
    #history['sign'].append(sign)
    history['val_loss'].append(test_loss)
    history['val_acc'].append(test_accuracy)
    #history['val_sign'].append(test_sign)

torch.save(model.state_dict(), f"models/{CHANNEL}/{SIGNAL}.pt")

epochs = np.arange(1, EPOCHS+1)

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

plt.savefig(f"Outputs/{CHANNEL}/figures/{SIGNAL}.png")
