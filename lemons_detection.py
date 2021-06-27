import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import OrdinalEncoder

from dataloader import TrainData, TestData
from lemons_net import LemonsNet
from metrics import binary_accuracy


def encode_inputs(x):
    '''
    Encodes the inpput because of categorical values
    :param x: Input
    :return: Encoded input
    '''
    oe = OrdinalEncoder()
    oe.fit(x)
    x_enc = oe.transform(x)
    return x_enc


def scale_inputs(x_train, x_test):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


df = pd.read_csv("/home/baha/codes/lemonsData/data_lemons.csv")
df = df.dropna()
x = df.drop('IsBadBuy', axis=1).values
y = df['IsBadBuy'].values

x_encoded = encode_inputs(x)
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=20)
print('Train', x_train.shape, y_train.shape)
print('Test', x_test.shape, y_test.shape)

#
x_train, x_test = scale_inputs(x_train, x_test)


# defining model hyperparameters
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001

train_data = TrainData(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
test_data = TestData(torch.FloatTensor(x_test))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=1)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LemonsNet()
model.to(device)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model.train()
for e in range(1, EPOCHS + 1):
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.unsqueeze(1))
        acc = binary_accuracy(y_pred, y_batch.unsqueeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')


y_pred_list = []
model.eval()
with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

confusion_matrix(y_test, y_pred_list)
print(classification_report(y_test, y_pred_list))


