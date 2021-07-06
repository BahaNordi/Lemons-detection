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
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn import metrics
from torch.optim.lr_scheduler import MultiStepLR

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
LEARNING_RATE = 0.01
train_label = df['IsBadBuy'][:len(y_train)]
train_label_ids = torch.tensor([label for label in train_label], dtype=torch.long)
labels = TensorDataset(train_label_ids)

#class weighting
labels_unique, counts = np.unique(train_label, return_counts=True)
print("Unique labels : {}".format(labels_unique))
class_weights = [sum(counts) / c for c in counts] #(#{class_0}, #(class_1}]
weights = [class_weights[e] for e in train_label]
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(train_label))

train_data = TrainData(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
test_data = TestData(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
train_loader = DataLoader(dataset=train_data, sampler=sampler, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LemonsNet()
model.to(device)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
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
        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} | Acc: {epoch_acc / len(train_loader):.3f}')

model.eval()
all_predictions = torch.LongTensor()
all_hard_predictions = torch.FloatTensor()
all_labels = torch.LongTensor()
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        y_test_pred = model(x_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        all_hard_predictions = torch.cat([all_hard_predictions, y_pred_tag], dim=0)
        all_predictions = torch.cat([all_predictions, y_test_pred], dim=0)
        all_labels = torch.cat([all_labels, y_batch], dim=0)

confusion_matrix(all_labels, all_hard_predictions)
fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_predictions, pos_label=1)
# ap = metrics.average_precision_score(all_labels, all_predictions)
AUC = metrics.auc(fpr, tpr)
print(classification_report(all_labels, all_hard_predictions))
print("AUC on test set", AUC)


'''
Steps I have taken to improve the results and code clarity:

- Definitely commented the code and followed Pep8 standard quality

- The dataset seems to have Nan values. Currently, I simply removed the datapoints that included Nan. A better
way would have been replacement by the median of that feature accoss the training scale_inputs

- The dataset is heavily imbalanced. Given more time, I would devised augmentation strategies to increase the minority 
glass. Additionally I would have changed the weight of the loss function to consider this imbalance.

- Since the dataset is heavily imbalanced using accuracy to evaluate the model is not a wise decision. I would have used
area under the ROC curve as a better measure to evaluate the performance of my model.

_ I had no time to tune any hyper parameters but currently have a binary accuracy of 90.6% on the training set & 60% on 
the test set whhich signals heavy overfitting. I would use better regularization techniques such as
L2 and early stopping to overcome this issue.

'''