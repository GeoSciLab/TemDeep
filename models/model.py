import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        out += residual
        out = self.relu(out)

        return out

class FieldPredictionNet(nn.Module):
    def __init__(self):
        super(FieldPredictionNet, self).__init__()

        # Define a more complex network architecture with residual blocks and ConvTranspose2d
        self.encoder_conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3)
        self.residual_block1 = ResidualBlock(64, 64)
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.residual_block2 = ResidualBlock(128, 128)
        self.encoder_conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.residual_block3 = ResidualBlock(256, 256)
        self.encoder_conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.residual_block4 = ResidualBlock(512, 512)

        self.decoder_deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.decoder_deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.decoder_deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decoder_deconv4 = nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1)

    def forward(self, flow):
        enc_conv1_out = F.relu(self.encoder_conv1(flow))
        residual_block1_out = self.residual_block1(enc_conv1_out)
        enc_conv2_out = F.relu(self.encoder_conv2(residual_block1_out))
        residual_block2_out = self.residual_block2(enc_conv2_out)
        enc_conv3_out = F.relu(self.encoder_conv3(residual_block2_out))
        residual_block3_out = self.residual_block3(enc_conv3_out)
        enc_conv4_out = F.relu(self.encoder_conv4(residual_block3_out))
        residual_block4_out = self.residual_block4(enc_conv4_out)

        dec_deconv1_out = F.relu(self.decoder_deconv1(residual_block4_out))
        dec_deconv2_out = F.relu(self.decoder_deconv2(dec_deconv1_out))
        dec_deconv3_out = F.relu(self.decoder_deconv3(dec_deconv2_out))
        dec_deconv4_out = self.decoder_deconv4(dec_deconv3_out)

        return dec_deconv4_out

# Create an instance of the FieldPredictionNet
net = FieldPredictionNet()

# Generate some dummy input flow data
input_flow = torch.randn(1, 2, 256, 256)

# Pass the input flow through the network
output_flow = net(input_flow)

print(output_flow.shape)  # Check the shape of the output
