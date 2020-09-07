import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                     #nn.ReLU(),
                                     #nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                     #nn.ReLU(),
                                     #nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = nn.Sequential(nn.Linear(28 * 28 * 1, 1024),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(1024, 10))

    def forward(self, x):
        #x = self.conv1(x)
        x = x.view(-1, 28 * 28 * 1)
        x = self.dense(x)
        return x
