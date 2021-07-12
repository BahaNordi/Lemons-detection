import torch.nn as nn


class LemonsNet(nn.Module):
    def __init__(self):
        super(LemonsNet, self).__init__()
        self.l1 = nn.Linear(14, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 64)
        self.l_out = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.relu(self.l1(x))
        x = self.bn1(x)
        x = self.relu(self.l2(x))
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.relu(self.l3(x))
        x = self.bn3(x)
        x = self.dropout(x)

        out = self.l_out(x)

        return out
