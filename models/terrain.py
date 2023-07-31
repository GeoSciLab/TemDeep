import torch
import torch.nn as nn

class CTIM(nn.Module):
    def __init__(self):
        super(CTIM, self).__init__()

        # First convolution layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolution layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # First Conv -> BatchNorm -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second Conv -> BatchNorm -> ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

# Create an instance of the CTIM
ctim = CTIM()

# Generate some dummy input data (Elevation)
input_data = torch.randn(1, 1, 256, 256)

# Pass the input data through the CTIM
output_data = ctim(input_data)

print(output_data.shape)  # Check the shape of the output
