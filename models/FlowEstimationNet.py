import torch
import torch.nn as nn

# Define the flow encoder
class FlowEncoder(nn.Module):
    def __init__(self):
        super(FlowEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        return x

# Define the flow decoder
class FlowDecoder(nn.Module):
    def __init__(self):
        super(FlowDecoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.leaky_relu(self.deconv1(x))
        x = self.leaky_relu(self.deconv2(x))
        x = self.leaky_relu(self.deconv3(x))
        x = self.leaky_relu(self.deconv4(x))
        x = self.deconv5(x)
        return x

# Combining the Flow Encoder and Flow Decoder into the Flow Estimation Module
class FlowEstimationModule(nn.Module):
    def __init__(self, in_channels=3):
        super(FlowEstimationModule, self).__init__()
        self.flow_encoder = FlowEncoder(in_channels)
        self.flow_decoder = FlowDecoder()

    def forward(self, x):
        x = self.flow_encoder(x)
        x = self.flow_decoder(x)
        forward_flow, backward_flow = torch.chunk(x, chunks=2, dim=1)
        return forward_flow, backward_flow

# Example usage:
# Assuming input tensor x with appropriate dimensions (batch_size, in_channels, height, width)
flow_estimation_module = FlowEstimationModule(in_channels=3)
forward_flow, backward_flow = flow_estimation_module(x)
