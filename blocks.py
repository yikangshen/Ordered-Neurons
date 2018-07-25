import torch
import torch.nn as nn
import torch.nn.functional as F


def stick_breaking(logits):
    e = F.sigmoid(logits)
    z = (1 - e).cumprod(dim=1)
    p = torch.cat([e.narrow(1, 0, 1), e[:, 1:] * z[:, :-1]], dim=1)

    return p


def softmax(x, mask=None):
    max_x, _ = x.max(dim=-1, keepdim=True)
    e_x = torch.exp(x - max_x)
    if not (mask is None):
        e_x = e_x * mask
    out = e_x / (e_x.sum(dim=-1, keepdim=True) + 1e-8)

    return out


class AttentionDropout(nn.Module):
    def __init__(self, p=0.5):
        super(AttentionDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p

    def forward(self, input):
        if self.training and self.p > 0:
            p = input.data.new(input.size()).zero_() + (1 - self.p)
            mask = torch.bernoulli(p)
            output = input * mask  # bsz, nslots
            output = output / (output.sum(dim=1, keepdim=True) + 1e-8)
        else:
            output = input
        return output


class TimeNorm(nn.Module):

    def __init__(self, features, eps=1e-6, mu=0.99):
        super(TimeNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
        self.step = 0
        # self.mean = 0
        self.register_buffer('mean', torch.zeros(features))
        self.mu = mu

    def forward(self, x):
        if self.training:
            self.step += 1
            self.mean = self.mu * self.mean + (1 - self.mu) * x.mean(dim=0).data
            mean = self.mean / (1 - self.mu ** self.step)
        else:
            mean = self.mean
        return self.gamma * (x - mean) + self.beta