import warnings

warnings.filterwarnings(action='ignore')

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Global variables
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIGNAL = f"TTToHcToWA_AToMuMu_{args.mass}"
CHANNEL = args.channel
BATCH_SIZE = 4096
EPOCHS = 70
N_FOLD = 3


# classes and helper functions
class MyDataset(Dataset):
    def __init__(self, features, targets):
        super(MyDataset, self).__init__()
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx])
        y = torch.LongTensor(self.targets[idx])
        return x, y


class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, output_size)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = F.dropout(x, p=0.5)
        x = F.softmax(self.fc4(x), dim=1)
        return x


def train(model, train_loader, optimizer, class_weights, epoch):
    model.train()
    train_loss = 0.
    correct = 0
    class_weights = class_weights.to(DEVICE)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)

        target = target.view(len(target))
        class_weights = class_weights.view(len(class_weights))
        optimizer.zero_grad()

        output = model(data)
        loss = F.cross_entropy(output, target, class_weights)
        loss.backward()
        optimizer.step()
        train_loss += loss

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= (len(train_loader.dataset) / BATCH_SIZE)
    train_loss = train_loss.cpu().detach().numpy()
    train_accuracy = 100. * correct / len(train_loader.dataset)

    print(
        f"[{epoch}] Train Loss: {train_loss}, Train Accuracy: {train_accuracy}%"
    )
    return (train_loss, train_accuracy)


def evaluate(model, test_loader, class_weights, epoch):
    model.eval()
    test_loss = 0.
    correct = 0
    class_weights = class_weights.to(DEVICE)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            target = target.view(len(target))
            output = model(data)
            test_loss += F.cross_entropy(output,
                                         target,
                                         class_weights,
                                         reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print(
        f"[{epoch}] Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%\n")
    return (test_loss, test_accuracy)


def predict(model, features):
    model.to('cpu')
    model.eval()
    predictions = list()
    with torch.no_grad():
        for idx in range(len(features)):
            feature = (torch.FloatTensor(features[idx])).unsqueeze(0)
            prediction = model(feature)
            predictions.append(np.argmax(prediction[0]).numpy())
    return np.array(predictions)


def predict_prob(model, features):
    model.to('cpu')
    model.eval()
    predictions = list()
    with torch.no_grad():
        for idx in range(len(features)):
            feature = (torch.FloatTensor(features[idx])).unsqueeze(0)
            prediction = model(feature)
            predictions.append(prediction[0][0].numpy())

    return np.array(predictions)


def kFold(sample, n_split=3):
    """Divide sample and return as a list of sub samples"""
    sub_samples = []
    fold = len(sample) // n_split
    for idx in range(n_split):
        if idx != n_split - 1:
            sub_sample = sample.iloc[fold * idx:fold * (idx + 1)]
        else:
            sub_sample = sample.iloc[fold * idx:]
        sub_samples.append(sub_sample)

    return sub_samples


def train_test_split(sample, split_idx, label):
    train_list = list()
    for idx in range(len(sample)):
        if idx == split_idx:
            continue
        else:
            train_list.append(sample[idx].copy())
    train_sample = pd.concat(train_list)
    test_sample = sample[split_idx].copy()

    train_sample['label'] = label
    test_sample['label'] = label

    return (train_sample, test_sample)


def get_class_weights(signal, bkg):
    class_weights = list()
    class_weights.append(len(signal))
    class_weights.append(len(bkg))
    class_weights = [x / (len(signal) + len(bkg)) for x in class_weights]
    class_weights = [1 / x for x in class_weights]
    class_weights = torch.FloatTensor(class_weights)
    return class_weights


def kFoldTest(signal, bkg, idx, bkg_name):
    if idx == N_FOLD:
        train_signal, test_signal = pd.concat(signal), pd.concat(signal)
        train_bkg, test_bkg = pd.concat(bkg), pd.concat(bkg)
        train_signal['label'] = 0
        test_signal['label'] = 0
        train_bkg['label'] = 1
        test_bkg['label'] = 1
    else:
        train_signal, test_signal = train_test_split(signal, idx, label=0)
        train_bkg, test_bkg = train_test_split(bkg, idx, label=1)

    print(f"traing signal vs {bkg_name}")
    # define class weights
    train_sample = pd.concat([train_signal, train_bkg])
    test_sample = pd.concat([test_signal, test_bkg])
    class_weights = get_class_weights(train_signal, train_bkg)
    print(class_weights)

    # extract features and labels
    X_train, X_test = train_sample[features].to_numpy(
    ), test_sample[features].to_numpy()
    y_train, y_test = train_sample[['label'
                                    ]].to_numpy(), test_sample[['label'
                                                                ]].to_numpy()

    scaler = MinMaxScaler()
    scaler.fit(train_sample[features].to_numpy())
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    train_dataset, test_dataset = MyDataset(X_train, y_train), MyDataset(
        X_test, y_test)
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              num_workers=4,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             num_workers=4,
                             shuffle=True)

    model = DNN(len(features), 2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    history = dict()
    history['loss'] = list()
    history['acc'] = list()
    history['val_loss'] = list()
    history['val_acc'] = list()

    for epoch in range(1, EPOCHS + 1):
        loss, acc = train(model, train_loader, optimizer, class_weights, epoch)
        scheduler.step()
        test_loss, test_acc = evaluate(model, test_loader, class_weights,
                                       epoch)
        history['loss'].append(loss)
        history['acc'].append(acc)
        history['val_loss'].append(test_loss)
        history['val_acc'].append(test_acc)

    if idx == N_FOLD:
        model_name = f"{SIGNAL}_vs_{bkg_name}.pt"
    else:
        model_name = f"{SIGNAL}_vs_{bkg_name}_{idx}.pt"
    torch.save(model.state_dict(), f"models/{CHANNEL}/{model_name}")

    pred_train = predict(model, X_train)
    pred_test = predict(model, X_test)
    pred_prob_train = predict_prob(model, X_train)
    pred_prob_test = predict_prob(model, X_test)
    cm = metrics.confusion_matrix(y_test, pred_test)
    fpr_train, tpr_train, thresh_train = metrics.roc_curve(y_train,
                                                           pred_prob_train,
                                                           pos_label=0)
    fpr, tpr, thresh = metrics.roc_curve(y_test, pred_prob_test, pos_label=0)
    auc_train = metrics.auc(fpr_train, tpr_train)
    auc = metrics.auc(fpr, tpr)
    print(auc_train, auc)

    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title("Confusion Matrix")
    ax1 = sns.heatmap(cm,
                      annot=True,
                      fmt='d',
                      square=False,
                      xticklabels=['signal', 'background'],
                      yticklabels=['signal', 'background'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')

    ax2 = plt.subplot(1, 2, 2)
    ax2.set_title(f'ROC curve\nAUC:{round(auc, 3)}')
    ax2.plot(tpr_train, 1 - fpr_train, 'r--', label='train roc')
    ax2.plot(tpr, 1 - fpr, 'b--', label='test roc')
    ax2.legend(loc='best')
    ax2.set_xlabel('signal efficiency')
    ax2.set_ylabel('background rejection')

    if idx == N_FOLD:
        save_name = f"cm_and_roc_signal_vs_{bkg_name}.png"
    else:
        save_name = f"cm_and_roc_signal_vs_{bkg_name}_{idx}.png"
    plt.savefig(f"Outputs/{CHANNEL}/{args.mass}/{save_name}")


# load dataset
signal = kFold(
    shuffle(pd.read_csv(f"Outputs/{CHANNEL}/CSV/{SIGNAL}.csv", index_col=0)),
    N_FOLD)
fake = kFold(
    shuffle(pd.read_csv(f"Outputs/{CHANNEL}/CSV/fake.csv", index_col=0)),
    N_FOLD)
ttX = kFold(
    shuffle(pd.read_csv(f"Outputs/{CHANNEL}/CSV/ttX.csv", index_col=0)),
    N_FOLD)

rare = pd.read_csv(f"Outputs/{CHANNEL}/CSV/rare.csv", index_col=0)
VV = pd.read_csv(f"Outputs/{CHANNEL}/CSV/VV.csv", index_col=0)
DY = pd.read_csv(f"Outputs/{CHANNEL}/CSV/DY.csv", index_col=0)
ZG = pd.read_csv(f"Outputs/{CHANNEL}/CSV/ZG.csv", index_col=0)
others = shuffle(pd.concat([rare, VV, DY, ZG]))
print(len(others))
others = kFold(others, N_FOLD)

# train 3 classifiers: signal vs fake, signal vs ttX, signal vs others
# will check confusion matrix and ROC curve
features = [
    'mu1_px', 'mu1_py', 'mu1_pz', 'mu1_mass', 'mu1_charge', 'mu2_px', 'mu2_py',
    'mu2_pz', 'mu2_mass', 'mu2_charge', 'ele_px', 'ele_py', 'ele_pz',
    'ele_mass', 'j1_px', 'j1_py', 'j1_pz', 'j1_mass', 'j1_bscore', 'j2_px',
    'j2_py', 'j2_pz', 'j2_mass', 'j2_bscore', 'dR_mu1mu2', 'dR_mu1ele',
    'dR_mu2ele', 'dR_j1ele', 'dR_j2ele', 'dR_j1j2', 'HT', 'MT', 'LT', 'MET',
    'Nj', 'Nb', 'avg_dRjets', 'avg_bscore'
]

# train with 3 fold
for idx in range(1, N_FOLD + 1):
    print(f"Fold {idx}")
    kFoldTest(signal, fake, idx - 1, "fake")
    kFoldTest(signal, ttX, idx - 1, "ttX")
    kFoldTest(signal, others, idx - 1, "others")
kFoldTest(signal, fake, N_FOLD, "fake")
kFoldTest(signal, ttX, N_FOLD, "ttX")
kFoldTest(signal, others, N_FOLD, "others")
