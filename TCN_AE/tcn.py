import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class ConvEncoder(nn.Module):
    def __init__(self, input_size):
        super(ConvEncoder, self).__init__()
        self.input_size = input_size

        layers = []
        self.layer_1 = weight_norm(nn.Conv1d(input_size, 32, 3, stride=1))
        self.layer_2 = nn.CELU()
        self.layer_3 = nn.MaxPool1d(2)
        self.layer_4 = weight_norm(nn.Conv1d(32,16,3,stride=1))
        self.layer_5 = nn.MaxPool1d(2)
        self.layer_6 = nn.CELU()
        self.init_weights()

    def init_weights(self):
        self.layer_1.weight.data.normal_(0, 0.01)
        self.layer_4.weight.data.normal_(0, 0.01)
        # self.layer_7.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        # x = self.layer_4(x)
        # x = self.layer_5(x)
        # x = self.layer_6(x)
        x = self.layer_5(x)
        x = self.layer_6(x)

        return x


class ConvDecoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size


        self.layer_1 = weight_norm(nn.ConvTranspose1d(self.input_size,self.output_size//2,3,stride=1))
        # layers.append(nn.ReLU())
        self.layer_2 = nn.CELU()
        self.layer_3 = nn.Upsample(scale_factor=2)
        self.layer_4 = weight_norm(nn.ConvTranspose1d(self.output_size//2, self.output_size, 3, stride=1))
        self.layer_5 = nn.CELU()
        self.layer_6 = nn.Upsample(scale_factor=2)
        # layers.append(nn.Conv1d(16, 64, 3, stride=1))
        # layers.append(nn.ReLU())
        # layers.append(nn.Upsample(scale_factor=2))
        # self.layer_4 = weight_norm(nn.ConvTranspose1d(32,self.output_size,3,stride=1))
        # self.layer_5 = nn.CELU()
        self.layer_7 = weight_norm(nn.ConvTranspose1d(self.output_size, self.output_size, 3, stride=1))
        self.layer_8 = nn.CELU()
        self.init_weights()

    def init_weights(self):
        self.layer_1.weight.data.normal_(0, 0.01)
        self.layer_4.weight.data.normal_(0, 0.01)
        self.layer_7.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        return x


class ConvAutoencoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvAutoencoder, self).__init__()
        self.input_size = input_size
        self.fc_size = 512
        self.core_length = 992
        self.encoder = ConvEncoder(input_size)
        self.fc = nn.Linear(self.core_length, self.fc_size)
        self.decoder_fc = nn.Linear(self.fc_size, self.core_length)

        self.decoder = ConvDecoder(16, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x_shape = x.shape
        x = x.view(x.shape[0], np.prod(x_shape[1:]))
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = self.decoder_fc(x)
        x = nn.ReLU()(x)
        x = x.view(x_shape)
        x = self.decoder(x)
        return x


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
