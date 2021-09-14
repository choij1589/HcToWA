import warnings

warnings.filterwarnings(action='ignore')

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
print(f"using {DEVICE}")
BATCH_SIZE = 2048
EPOCHS = 300
N_FOLD = 4


# DataLoader
class MyDataset(Dataset):
    def __init__(self, features, targets):
        super(MyDataset, self).__init__()
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.features[idx])
        y = torch.LongTensor(self.targets[idx])
        return (X, y)


class DataManager:
    def __init__(self, signal_fold, bkg_fold, features):
        self.signal_fold = signal_fold
        self.bkg_fold = bkg_fold
        self.features = features
        self.idx = None
        self.class_weights = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None

    def get_dataloaders(self, idx):
        print("Assume N_FOLD = 4")
        self.idx = idx
        train_sig = pd.concat([
            self.signal_fold[(idx + 1) % N_FOLD],
            self.signal_fold[(idx + 2) % N_FOLD],
            self.signal_fold[(idx + 3) % 4]
        ])
        test_sig = self.signal_fold[idx]
        train_bkg = pd.concat([
            self.bkg_fold[(idx + 1) % N_FOLD],
            self.bkg_fold[(idx + 2) % N_FOLD],
            self.bkg_fold[(idx + 3) % N_FOLD]
        ])
        test_bkg = self.bkg_fold[idx]
        train_sig['label'] = 0
        test_sig['label'] = 0
        train_bkg['label'] = 1
        test_bkg['label'] = 1

        train_sig, val_sig = self.__train_val_split(train_sig)
        train_bkg, val_bkg = self.__train_val_split(train_bkg)

        train_sample = shuffle(pd.concat([train_sig, train_bkg]))
        val_sample = shuffle(pd.concat([val_sig, val_bkg]))
        test_sample = shuffle(pd.concat([test_sig, test_bkg]))

        self.__update_class_weights(train_sig, train_bkg)

        X_train, X_val, X_test = train_sample[self.features].to_numpy(
        ), val_sample[self.features].to_numpy(), test_sample[
            self.features].to_numpy()
        y_train, y_val, y_test = train_sample[[
            'label'
        ]].to_numpy(), val_sample[['label'
                                   ]].to_numpy(), test_sample[['label'
                                                               ]].to_numpy()

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(
            X_val), scaler.transform(X_test)

        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        train_dataset, val_dataset, test_dataset = MyDataset(
            X_train, y_train), MyDataset(X_val,
                                         y_val), MyDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  num_workers=4,
                                  shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                num_workers=4,
                                shuffle=False)
        test_loader = DataLoader(test_dataset,
                                 batch_size=BATCH_SIZE,
                                 num_workers=4,
                                 shuffle=False)

        return (train_loader, val_loader, test_loader)

    def __train_val_split(self, sample, p=0.8):
        length = int(len(sample) * p)
        train = sample[:length]
        val = sample[length:]

        return (train, val)

    def __update_class_weights(self, signal, bkg):
        class_weights = list()
        class_weights.append(len(signal))
        class_weights.append(len(bkg))
        class_weights = [x / (len(signal) + len(bkg)) for x in class_weights]
        class_weights = [1 / x for x in class_weights]
        class_weights = torch.FloatTensor(class_weights)
        self.class_weights = class_weights


# Models
class SimpleDNN(nn.Module):
    def __init__(self, input_size, output_size, activation='relu'):
        super(SimpleDNN, self).__init__()
        self.activation = activation
        self.fc1 = nn.Linear(input_size, 256, bias=True)
        self.fc2 = nn.Linear(256, 256, bias=True)
        self.fc3 = nn.Linear(256, 256, bias=True)
        self.fc4 = nn.Linear(256, output_size, bias=True)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        torch.nn.init.kaiming_normal_(self.fc1.weight)
        torch.nn.init.kaiming_normal_(self.fc2.weight)
        torch.nn.init.kaiming_normal_(self.fc3.weight)
        torch.nn.init.kaiming_normal_(self.fc4.weight)

    def forward(self, x):
        if self.activation == 'relu':
            x = F.relu(self.fc1(x))
            x = self.bn1(x)
            x = F.dropout(x, p=0.5)
            x = F.relu(self.fc2(x))
            x = self.bn2(x)
            x = F.dropout(x, p=0.5)
            x = F.relu(self.fc3(x))
            x = self.bn3(x)
            x = F.dropout(x, p=0.5)

        elif self.activation == 'elu':
            x = F.elu(self.fc1(x))
            x = self.bn1(x)
            x = F.dropout(x, p=0.5)
            x = F.elu(self.fc2(x))
            x = self.bn2(x)
            x = F.dropout(x, p=0.5)
            x = F.elu(self.fc3(x))
            x = self.bn3(x)
            x = F.dropout(x, p=0.5)

        else:
            print(f"Wrong activation {self.activation}")
            raise (AttributeError)

        return F.softmax(self.fc4(x), dim=1)


class SelfNormDNN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 n_hidden=256,
                 batch_norm=False):
        super(SelfNormDNN, self).__init__()
        # LeCun initialization is default for PyTorch
        self.batch_norm = batch_norm
        self.fc1 = nn.Linear(input_size, n_hidden, bias=True)
        self.fc2 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.fc3 = nn.Linear(n_hidden, n_hidden, bias=True)
        self.fc4 = nn.Linear(n_hidden, output_size, bias=True)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.bn3 = nn.BatchNorm1d(n_hidden)

    def forward(self, x):
        if self.batch_norm:
            x = F.selu(self.fc1(x))
            x = self.bn1(x)
            x = F.alpha_dropout(x, p=0.5)
            x = F.selu(self.fc2(x))
            x = self.bn2(x)
            x = F.alpha_dropout(x, p=0.5)
            x = F.selu(self.fc3(x))
            x = self.bn3(x)
            x = F.alpha_dropout(x, p=0.5)
        else:
            x = F.selu(self.fc1(x))
            x = F.alpha_dropout(x, p=0.5)
            x = F.selu(self.fc2(x))
            x = F.alpha_dropout(x, p=0.5)
            x = F.selu(self.fc3(x))
            x = F.alpha_dropout(x, p=0.5)

        return F.softmax(self.fc4(x), dim=1)


# helper functions
class EarlyStopping:
    def __init__(self,
                 patience=7,
                 verbose=False,
                 delta=0,
                 path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train(model, train_loader, optimizer, class_weights, epoch):
    model.train()
    train_loss = 0.
    correct = 0
    class_weights = class_weights.to(DEVICE)
    for data, target in train_loader:
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

    print(f"[{epoch}] Train Loss: {train_loss}, Train Acc: {train_accuracy}")
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

    print(f"[{epoch}] Test Loss: {test_loss}, Test Acc: {test_accuracy}\n")
    return (test_loss, test_accuracy)


def predict(model, features, prob=False):
    model.to('cpu')
    model.eval()
    predictions = list()
    with torch.no_grad():
        for idx in range(len(features)):
            feature = (torch.FloatTensor(features[idx])).unsqueeze(0)
            prediction = model(feature)
            if prob:
                predictions.append(prediction[0][0].numpy())
            else:
                predictions.append(np.argmax(prediction[0]).numpy())

    return np.array(predictions)
