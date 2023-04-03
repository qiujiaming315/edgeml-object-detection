import torch
import torch.nn as nn
from torch.utils.data import Dataset

"""Dataloader and NN model for offloading reward estimation."""


class EdgeDetectionDataset(Dataset):
    def __init__(self, inputs, labels):
        self.labels = labels
        self.inputs = inputs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        label = self.labels[idx]
        x = torch.from_numpy(x).to(torch.float32)
        label = torch.tensor([label], dtype=torch.float32)
        return x, label


class EdgeDetectionNet(nn.Module):
    def __init__(self, channels, kernels, pools, linear, resize=True):
        """
        Build a convolutional neural network to predict offloading reward using feature maps from the weak detector.
        :param channels: a list with the number of (input and output) channel for each convolutional layer.
                         If an empty list is given, build a network without convolutional layers.
        :param kernels: a list with the kernel size for each convolutional layer.
        :param pools: a boolean list that specifies whether each convolutional layer should followed by a pooling layer.
        :param linear: a list that specifies the number of (input and output) features in each linear (fully-connected)
                       layer. If an empty list is given, build a fully-convolutional network that ends with
                       global average pooling.
        :param resize: whether the feature maps are resized to the same shape.
        """
        super(EdgeDetectionNet, self).__init__()
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_stacks, self.linear_stacks = nn.ModuleList(), nn.ModuleList()
        self.resize = resize
        # Construct convolutional and linear stacks.
        assert len(channels) > 1 or len(linear) > 1, \
            "Invalid CNN architecture. Please add at least 1 convolutional or linear layer."
        if len(channels) > 1:
            for in_channel, out_channel, kernel_size, pool in zip(channels[:-1], channels[1:], kernels, pools):
                self.conv_stacks.append(self.conv_stack(in_channel, out_channel, kernel_size, pool))
        if len(linear) > 1:
            last = [False] * (len(linear) - 1)
            last[-1] = True
            for in_feature, out_feature, l in zip(linear[:-1], linear[1:], last):
                self.linear_stacks.append(self.linear_stack(in_feature, out_feature, l))

    def conv_stack(self, in_channels, out_channels, kernel_size, pool):
        """
        Build a convolutional layer with relu activator, batch normalization, and (optional) max pooling.
        :param in_channels: number of channels in the input feature map.
        :param out_channels: number of channels in the output feature map.
        :param kernel_size: size of kernel in the convolutional layer.
        :param pool: if max pooling is applied.
        :return: the constructed convolutional stack.
        """
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
        nn.init.kaiming_uniform_(conv_layer.weight)
        modules = [conv_layer]
        if self.resize:
            modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.LeakyReLU(0.01))
        modules.append(nn.Dropout(0.4))
        if pool:
            modules.append(self.pool)
        conv = nn.Sequential(*modules)
        return conv

    def linear_stack(self, in_features, out_features, last=False):
        """
        Build a linear (fully-connected) layer with relu activator, and (optional) dropout.
        :param in_features: number of features in the input.
        :param out_features: number of channels in the output.
        :param last: whether it is the last layer.
        :return: the constructed linear stack.
        """
        linear_layer = nn.Linear(in_features, out_features)
        torch.nn.init.kaiming_uniform_(linear_layer.weight)
        modules = [linear_layer]
        if not last:
            if self.resize:
                modules.append(nn.BatchNorm1d(out_features))
            modules.append(nn.LeakyReLU(0.01))
            modules.append(nn.Dropout(0.2))
        linear = nn.Sequential(*modules)
        return linear

    def forward(self, x):
        """Forward function."""
        for conv in self.conv_stacks:
            x = conv(x)
        if not self.resize:
            # Use average pooling if feature maps have different shapes.
            x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.flatten(x)
        for linear in self.linear_stacks:
            x = linear(x)
        return x
