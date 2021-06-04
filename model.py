import torch.nn as nn


class MiniTinySleepNet(nn.Module):
    def __init__(self):
        super(MiniTinySleepNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=50, stride=6)
        self.max_pool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.dropout1 = nn.Dropout(p=0.5)
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1),
        )
        self.max_pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1280, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.dropout1(x)
        x = self.convs(x)
        x = self.max_pool2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary
    model = MiniTinySleepNet()
    summary(model, (2, 3000), device="cpu")