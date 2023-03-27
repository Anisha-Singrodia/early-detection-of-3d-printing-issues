import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A simple CNN with 2 convolutional layers and 2 fully-connected layers.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super().__init__()
        # Input = 3 x 224 x 224, Output = 16 x 224 x 224
        self.conv_layer1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # Input = 16 x 224 x 224, Output = 16 x 224 x 224
        self.conv_layer2 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        # Input = 16 x 224 x 224, Output = 16, 112, 112
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Input = 16, 112, 112, Output = 32, 112, 112
        self.conv_layer3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        # Input = 32, 112, 112, Output = 32, 112, 112
        self.conv_layer4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )
        # Input = 32, 112, 112, Output = 32, 56, 56
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(8192, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        self.soft = nn.Softmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward method
        """
        out = self.conv_layer1(x)
        # print(out.shape)
        # out = self.relu1(out)
        # out = self.conv_layer2(out)
        # print(out.shape)
        # out = self.relu1(out)
        out = self.max_pool1(out)
        
        # out = self.conv_layer3(out)
        # out = self.relu1(out)
        # out = self.conv_layer4(out)
        
        # out = self.relu1(out)
        # out = self.max_pool2(out)
        out = out.reshape(out.size(0), -1)
        # print(out.shape)
        # out = nn.Dropout(0.25)(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        # print(out.shape)
        # out = self.soft(out)
        # print(out.shape)
        # exit()
        return out