import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self, char_embed_vec, N_w, D_c, num_category):
        super(MyModel, self).__init__()
        self.char_embed_vec = char_embed_vec

        self.conv_1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=(2, N_w * D_c), stride=1)
        self.conv_2 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=(3, N_w * D_c), stride=1)
        self.conv_3 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=(4, N_w * D_c), stride=1)

        self.bn = nn.BatchNorm1d(100)

        self.fc_1 = nn.Linear(300, 100)
        self.fc_2 = nn.Linear(100, num_category)  # Output classifier layer

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.maxpool_1 = nn.MaxPool1d(kernel_size=19)
        self.maxpool_2 = nn.MaxPool1d(kernel_size=18)
        self.maxpool_3 = nn.MaxPool1d(kernel_size=17)

    def forward(self, inputs):
        # First Conv block
        out_1 = self.conv_1(inputs).squeeze()
        out_1 = self.bn(out_1)
        out_1 = self.relu(out_1)
        out_1 = self.maxpool_1(out_1).squeeze()

        # Second Conv block
        out_2 = self.conv_2(inputs).squeeze()
        out_2 = self.bn(out_2)
        out_2 = self.relu(out_2)
        out_2 = self.maxpool_2(out_2).squeeze()

        # Third Conv block
        out_3 = self.conv_3(inputs).squeeze()
        out_3 = self.bn(out_3)
        out_3 = self.relu(out_3)
        out_3 = self.maxpool_3(out_3).squeeze()

        # concatenation of three conv outputs
        out = torch.cat([out_1, out_2, out_3], dim=1)

        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        out = self.softmax(out)

        return out