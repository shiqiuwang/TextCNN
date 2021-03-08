import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):

    def __init__(self):
        super(TextCNN, self).__init__()
        self.conv3 = nn.Conv2d(1, 256, (3, 100), stride=1)
        self.conv4 = nn.Conv2d(1, 256, (4, 100), stride=1)
        self.conv5 = nn.Conv2d(1, 256, (5, 100), stride=1)
        self.Max3_pool = nn.MaxPool2d((39 - 3 + 1, 1))  # (kernel_size,stride)
        self.Max4_pool = nn.MaxPool2d((39 - 4 + 1, 1))
        self.Max5_pool = nn.MaxPool2d((39 - 5 + 1, 1))
        self.linear1 = nn.Linear(768, 6)

    def forward(self, x):
        batch = x.shape[0]
        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)

        # project the features to the labels
        x = self.linear1(x)
        x = x.view(-1, 6)

        return x
