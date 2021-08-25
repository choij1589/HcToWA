import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8192

class MyDataset(Dataset):
    def __init__(self, features, targets, weights):
        super(MyDataset, self).__init__()
        self.features = features
        self.targets = targets
        self.weights = weights

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.features[idx])
        y = torch.LongTensor(self.targets[idx])
        w = torch.FloatTensor(self.weights[idx])
        return x, y, w


def train_test_split(features, targets, weights, train_size=0.75, shuffle=True):
    # shuffle first
    if shuffle:
        shuf_features = list()
        shuf_targets = list()
        shuf_weights = list()
        shuf_idx = np.arange(len(targets))
        np.random.shuffle(shuf_idx)
        for idx in shuf_idx:
            shuf_features.append(features[idx])
            shuf_targets.append(targets[idx])
            shuf_weights.append(weights[idx])
        features = shuf_features
        targets = shuf_targets
        weights = shuf_weights

    split = int(len(targets)*train_size)
    train_features, test_features = features[:split], features[split:]
    train_targets, test_targets = targets[:split], targets[split:]
    train_weights, test_weights = weights[:split], weights[split:]

    return (train_features, test_features, train_targets, test_targets, train_weights, test_weights)


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
        x = F.relu(self.fc4(x))
        return F.softmax(x, dim=1)

class ConvNN(nn.Module):
    def __init__(self, res_size, output_size):
        super(ConvNN, self).__init__()
        #self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=True)
        self.conv1 = nn.Conv1d(1, 8, 2)
        self.conv2 = nn.Conv1d(1, 8, 2)
        self.conv3 = nn.Conv1d(1, 8, 2)
        self.conv4 = nn.Conv1d(1, 8, 2)
        self.pool = nn.MaxPool1d(2)
    
        self.fc1 = nn.Linear(151, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, conv1, conv2, conv3, conv4, res):
        x1 = F.relu(self.conv1(conv1))
        x1 = self.pool(x1)
        x2 = F.relu(self.conv2(conv2))
        x2 = self.pool(x2)
        x3 = F.relu(self.conv3(conv3))
        x3 = self.pool(x3)
        x4 = F.relu(self.conv4(conv4))
        x4 = self.pool(x4)
        x = torch.cat((x1, x2, x3, x4), 2)
        x = x.view([-1, x.shape[1]*x.shape[2]])
        x = torch.cat((x, res), 1)
     
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=0.5)
        x = F.softmax(self.fc3(x))
        return x

def train(model, train_loader, optimizer, class_weights, epoch):
    model.train()
    train_loss = 0.
    correct = 0
    #N_sig = 0.
    #N_bkg = 0.
    for batch_idx, (data, target, weight) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        #conv1 = data[:, :11]
        #conv2 = data[:, 11:21]
        #conv3 = data[:, 21:31]
        #conv4 = data[:, 31:42]
        #conv1 = conv1.view([-1, 1, 11])
        #conv2 = conv2.view([-1, 1, 10])
        #conv3 = conv3.view([-1, 1, 10])
        #conv4 = conv4.view([-1, 1, 11])
        #res = data[:, 42:]

        #conv1, conv2, conv3, conv4, res = conv1.to(DEVICE), conv2.to(DEVICE), conv3.to(DEVICE), conv4.to(DEVICE), res.to(DEVICE)
        #target = target.to(DEVICE)
        
        class_weights = class_weights.to(DEVICE)
        target = target.view(len(target))
        class_weights = class_weights.view(len(class_weights))
        optimizer.zero_grad()

        output = model(data)
        #output = model(conv1, conv2, conv3, conv4, res)
        loss = F.cross_entropy(output, target, class_weights)
        loss.backward()
        optimizer.step()
        train_loss += loss

        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)/BATCH_SIZE
    train_loss = train_loss.cpu()
    train_loss = train_loss.detach().numpy()
    train_accuracy = 100.*correct / len(train_loader.dataset)

    #N_sig, N_bkg = N_sig.cpu(), N_bkg.cpu()
    #significance = N_sig / np.sqrt(N_bkg)
    print(f'[{epoch}] Train Loss: {train_loss}, Train Accuracy: {train_accuracy}%')
    #print(f'[{epoch}] Train significance: {significance}')
    return (train_loss, train_accuracy)


def evaluate(model, test_loader, class_weights, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    #N_sig = 0.
    #N_bkg = 0.
    with torch.no_grad():
        for data, target, weight in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            #conv1 = data[:, :11]
            #conv2 = data[:, 11:21]
            #conv3 = data[:, 21:31]
            #conv4 = data[:, 31:42]
            #conv1 = conv1.view([-1, 1, 11])
            #conv2 = conv2.view([-1, 1, 10])
            #conv3 = conv3.view([-1, 1, 10])
            #conv4 = conv4.view([-1, 1, 11])
            #res = data[:, 42:]
            #conv1, conv2, conv3, conv4, res = conv1.to(DEVICE), conv2.to(DEVICE), conv3.to(DEVICE), conv4.to(DEVICE), res.to(DEVICE)
            #target = target.to(DEVICE)
            
            class_weights = class_weights.to(DEVICE)
            target = target.view(len(target))
            output = model(data)
            #output = model(conv1, conv2, conv3, conv4, res)
            test_loss += F.cross_entropy(output, target, class_weights, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.*correct / len(test_loader.dataset)

    print(f'[{epoch}] Test Loss: {test_loss}, Test Accuracy: {test_accuracy}%\n')
    return (test_loss, test_accuracy)
