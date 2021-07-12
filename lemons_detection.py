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
from tqdm.notebook import tqdm


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


def scale_inputs(x_train, x_val):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    return x_train, x_val


def bin_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc


df = pd.read_csv("/home/baha/codes/lemonsData/data_lemons.csv")
df = df.dropna()
x = df.drop('IsBadBuy', axis=1).values
y = df['IsBadBuy'].values
count = df['IsBadBuy'].value_counts()
sns.countplot(x='IsBadBuy', data=df)
plt.show()


x_encoded = encode_inputs(x)
x_train, x_val, y_train, y_val = train_test_split(x_encoded, y, test_size=0.2, random_state=20)
print('Train', x_train.shape, y_train.shape)
print('Test', x_val.shape, y_val.shape)

#
x_train, x_val = scale_inputs(x_train, x_val)


# defining model hyperparameters
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.001
# train_label = df['IsBadBuy'][:len(y_train)]
train_label_ids = torch.tensor([label for label in y_train], dtype=torch.long)
labels = TensorDataset(train_label_ids)

#class weighting
labels_unique, counts = np.unique(y_train, return_counts=True)
print("Unique labels : {}".format(labels_unique))
class_weights = [sum(counts) / c for c in counts] #(#{class_0}, #(class_1}]
weights = [class_weights[e]/np.sum(class_weights) for e in y_train]
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(y_train))

train_data = TrainData(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
val_data = TestData(torch.FloatTensor(x_val), torch.FloatTensor(y_val))
train_loader = DataLoader(dataset=train_data, sampler=sampler, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LemonsNet()
model.to(device)
print(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)
model.train()

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}

print("Begin training.")

for e in tqdm(range(1, EPOCHS + 1)):
    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0

    model.train()
    for X_train_batch, y_train_batch in train_loader:
        X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
        optimizer.zero_grad()

        y_train_pred = model(X_train_batch)

        train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
        train_acc = bin_acc(y_train_pred, y_train_batch.unsqueeze(1))

        train_loss.backward()
        optimizer.step()

        train_epoch_loss += train_loss.item()
        train_epoch_acc += train_acc.item()

    # VALIDATION
    with torch.no_grad():

        val_epoch_loss = 0
        val_epoch_acc = 0

        model.eval()
        all_predictions = torch.LongTensor()
        all_hard_predictions = torch.FloatTensor()
        all_labels = torch.LongTensor()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_pred = model(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))
            val_acc = bin_acc(y_val_pred, y_val_batch.unsqueeze(1))

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()

            loss_stats['train'].append(train_epoch_loss / len(train_loader))
            loss_stats['val'].append(val_epoch_loss / len(val_loader))
            accuracy_stats['train'].append(train_epoch_acc / len(train_loader))
            accuracy_stats['val'].append(val_epoch_acc / len(val_loader))

            y_val_pred_auc = torch.sigmoid(y_val_pred)
            y_pred_tag = torch.round(y_val_pred_auc)
            all_hard_predictions = torch.cat([all_hard_predictions, y_pred_tag], dim=0)
            all_predictions = torch.cat([all_predictions, y_val_pred_auc], dim=0)
            all_labels = torch.cat([all_labels, y_val_batch], dim=0)

    print(f'Epoch {e + 0:03}: | Train Loss: '
          f'{train_epoch_loss / len(train_loader):.5f} | Val Loss: '
          f'{val_epoch_loss / len(val_loader):.5f} | Train Acc: '
          f'{train_epoch_acc / len(train_loader):.3f}| Val Acc: '
          f'{val_epoch_acc / len(val_loader):.3f}')


confusion_matrix(all_labels, all_hard_predictions)
fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_predictions, pos_label=1)
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