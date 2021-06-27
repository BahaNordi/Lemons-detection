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


# read data
df = pd.read_csv("/home/baha/codes/lemonsData/data_lemons.csv")
df = df.dropna()
x = df.drop('IsBadBuy', axis=1).values
y = df['IsBadBuy'].values

# Train Test Split
x_encoded = encode_inputs(x)
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=20)
print('Train', x_train.shape, y_train.shape)
print('Test', x_test.shape, y_test.shape)

