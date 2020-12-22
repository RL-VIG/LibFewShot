from torch import nn


class Conv64F(nn.Module):
    """
        Four convolutional blocks network, each of which consists of a Covolutional layer,
        a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
        Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

        Input:  3 * 84 *84
        Output: 64 * 5 * 5
    """
    def __init__(self, is_flatten=False):
        super(Conv64F, self).__init__()

        self.is_flatten = is_flatten
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        out = self.features(x)
        if self.is_flatten:
            out = out.view(out.size(0), -1)

        return out


class Conv64FLeakyReLU(nn.Module):
    """
        Four convolutional blocks network, each of which consists of a Covolutional layer,
        a Batch Normalizaiton layer, a LeakyReLU layer and a Maxpooling layer.
        Used in the original DN4 and CovaMNet: https://github.com/WenbinLee/DN4.git.

        Input:  3 * 84 *84
        Output: 64 * 21 * 21
    """
    def __init__(self, is_flatten=False):
        super(Conv64FLeakyReLU, self).__init__()

        self.is_flatten = is_flatten
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        out = self.features(x)
        if self.is_flatten:
            out = out.view(out.size(0), -1)

        return out
