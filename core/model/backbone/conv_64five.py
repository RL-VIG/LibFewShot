from torch import nn


class Conv64Five(nn.Module):
    """
        Four convolutional blocks network, each of which consists of a Covolutional layer,
        a Batch Normalizaiton layer, a ReLU layer and a Maxpooling layer.
        Used in the original ProtoNet: https://github.com/jakesnell/prototypical-networks.git.

        Input:  3 * 84 *84
        Output: 64 * 5 * 5
    """

    def __init__(self, is_flatten=False, is_feature=False, drop_prob=0.1):
        super(Conv64Five, self).__init__()

        self.is_flatten = is_flatten
        self.is_feature = is_feature
        self.drop_prob = drop_prob

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.ReLU(),
            nn.Dropout(p=self.drop_prob),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.ReLU(),
            nn.Dropout(p=self.drop_prob),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.ReLU(),
            nn.Dropout(p=self.drop_prob),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.ReLU(),
            nn.Dropout(p=self.drop_prob),
            nn.MaxPool2d(kernel_size=2, stride=2), )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5),
            nn.ReLU(),
            nn.Dropout(p=self.drop_prob),
            nn.MaxPool2d(kernel_size=2, stride=2), )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        if self.is_flatten:
            out5 = out5.view(out5.size(0), -1)

        if self.is_feature:
            return out1, out2, out3, out4, out5

        return out5
