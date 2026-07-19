import torch
from torch import nn
from torchinfo import summary

class Inception(nn.Module):
    def __init__(self, in_channels, r1, r2, r3, r4):
        super().__init__()
        self.ReLU = nn.ReLU()
        #   路径一
        self.c1_1 = nn.Conv2d(in_channels= in_channels, out_channels=r1, kernel_size=1, stride=1)

        #   路径二
        self.c2_1 = nn.Conv2d(in_channels=in_channels, out_channels=r2[0], kernel_size=1, stride=1)
        self.c2_2 = nn.Conv2d(in_channels=r2[0], out_channels=r2[1], kernel_size=3, stride=1, padding=1)

        #   路径三
        self.c3_1 = nn.Conv2d(in_channels=in_channels, out_channels=r3[0], kernel_size=1, stride=1)
        self.c3_2 = nn.Conv2d(in_channels=r3[0], out_channels=r3[1], kernel_size=5, stride=1, padding=2)

        #   路径四
        self.c4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.c4_2 = nn.Conv2d(in_channels=in_channels, out_channels=r4, kernel_size=1, stride=1)

    def forward(self, x):
        Res1 = self.ReLU(self.c1_1(x))
        Res2 = self.ReLU(self.c2_2(self.ReLU(self.c2_1(x))))
        Res3 = self.ReLU(self.c3_2(self.ReLU(self.c3_1(x))))
        Res4 = self.ReLU(self.c4_2(self.c4_1(x)))

        return torch.cat((Res1, Res2, Res3, Res4), dim=1)

class GoogLeNet(nn.Module):
    def __init__(self, Inception):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (128, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (16, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024,2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


if __name__ == '__main__':
    device = torch.device('mps'if torch.backends.mps.is_available() else 'cpu')
    model = GoogLeNet(Inception).to(device)
    summary(model, input_size=(1, 1, 224, 224))