import torch.nn as nn
import torch.nn.functional as F

def flatten(x, start_dim=1, end_dim=-1):
    return x.flatten(start_dim=start_dim, end_dim=end_dim)

class tcFNN(nn.Module):
    def __init__(self, input_size, hidden_size, ConvSize, ConvSizeOut, num_states):
        super().__init__()
        # assign layer objects to class attributes
        self.bn0 = nn.BatchNorm1d(input_size)
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(input_size * ConvSizeOut)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size)
        self.do1 = nn.Dropout(p=0.5)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.do2 = nn.Dropout(p=0.5)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.do3 = nn.Dropout(p=0.5)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_states)
        self.bn5 = nn.BatchNorm1d(num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x, BadChannels=()):
        # forward always defines connectivity
        x[:, BadChannels, :] = 0
        x = self.bn0(x)
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(self.bn1(flatten(x)))
        x = F.relu(self.bn2(self.do1(self.fc1(x))))
        x = F.relu(self.bn3(self.do2(self.fc2(x))))
        x = F.relu(self.bn4(self.do3(self.fc3(x))))
        scores = (self.bn5(self.fc4(x)) - self.bn5.bias)/self.bn5.weight

        return scores

# WillseyNet with no batchnorm or dropout
class noreg_tcFNN(nn.Module):
    def __init__(self, input_size, hidden_size, ConvSize, ConvSizeOut, num_states):
        super().__init__()
        # assign layer objects to class attributes
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x, BadChannels=()):
        # forward always defines connectivity
        x[:, BadChannels, :] = 0
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(flatten(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        scores = self.fc4(x)
        return scores

class tcFNN_nobn(nn.Module):
    def __init__(self, input_size, hidden_size, ConvSize, ConvSizeOut, num_states):
        super().__init__()
        # assign layer objects to class attributes
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.do2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.do3 = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(hidden_size, num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x, BadChannels=()):
        # forward always defines connectivity
        x[:, BadChannels, :] = 0
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(flatten(x))
        x = F.relu(self.do1(self.fc1(x)))
        x = F.relu(self.do2(self.fc2(x)))
        x = F.relu(self.do3(self.fc3(x)))
        scores = self.fc4(x)

        return scores

class tcFNN_nodp(nn.Module):
    def __init__(self, input_size, hidden_size, ConvSize, ConvSizeOut, num_states):
        super().__init__()
        # assign layer objects to class attributes
        self.bn0 = nn.BatchNorm1d(input_size)
        self.cn1 = nn.Conv1d(ConvSize, ConvSizeOut, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(input_size * ConvSizeOut)
        self.fc1 = nn.Linear(input_size * ConvSizeOut, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.bn4 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_states)
        self.bn5 = nn.BatchNorm1d(num_states)

        # nn.init package contains convenient initialization methods
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
        nn.init.kaiming_normal_(self.cn1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='relu')
        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc4.bias)

    def forward(self, x, BadChannels=()):
        # forward always defines connectivity
        x[:, BadChannels, :] = 0
        x = self.bn0(x)
        x = self.cn1(x.permute(0, 2, 1))
        x = F.relu(self.bn1(flatten(x)))
        x = F.relu(self.bn2(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        x = F.relu(self.bn4(self.fc3(x)))
        scores = (self.bn5(self.fc4(x)) - self.bn5.bias)/self.bn5.weight

        return scores