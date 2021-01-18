from torch import nn
from torch.nn import functional as F


class LabelSmoothCELoss(nn.Module):
    def __init__(self, smoothing):
        super(LabelSmoothCELoss, self).__init__()

        self.smoothing = smoothing

    def forward(self, output, target):
        log_prob = F.log_softmax(output, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
