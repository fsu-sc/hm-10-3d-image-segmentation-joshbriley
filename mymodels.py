import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

# Padding-based skip connection matching
def match_size_and_concat(upsampled, bypass):
    diff_d = bypass.shape[2] - upsampled.shape[2]
    diff_h = bypass.shape[3] - upsampled.shape[3]
    diff_w = bypass.shape[4] - upsampled.shape[4]

    upsampled = F.pad(upsampled, [
        diff_w // 2, diff_w - diff_w // 2,
        diff_h // 2, diff_h - diff_h // 2,
        diff_d // 2, diff_d - diff_d // 2
    ])

    return torch.cat([upsampled, bypass], dim=1)

class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet3D, self).__init__()
        self.encoder1 = DoubleConv3D(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2)

        self.encoder2 = DoubleConv3D(16, 32)
        self.pool2 = nn.MaxPool3d(2)

        self.encoder3 = DoubleConv3D(32, 64)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = DoubleConv3D(64, 128)

        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv3D(128, 64)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv3D(64, 32)

        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv3D(32, 16)

        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.upconv3(bottleneck)
        dec3 = self.decoder3(match_size_and_concat(dec3, enc3))

        dec2 = self.upconv2(dec3)
        dec2 = self.decoder2(match_size_and_concat(dec2, enc2))

        dec1 = self.upconv1(dec2)
        dec1 = self.decoder1(match_size_and_concat(dec1, enc1))

        return self.final_conv(dec1)

if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    summary(model, (1, 64, 64, 64))

    from torch.utils.tensorboard import SummaryWriter
    dummy_input = torch.randn(1, 1, 64, 64, 64).to(device)
    writer = SummaryWriter("runs/unet3d")
    writer.add_graph(model, dummy_input)
    writer.close()
