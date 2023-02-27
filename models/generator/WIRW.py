import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class sa_layer(nn.Module):###zheg jiushi SA layer de daima
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, n_feats, groups=4):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, n_feats // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, n_feats // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, n_feats // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, n_feats // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(n_feats // (2 * groups), n_feats // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out
class WIRW(nn.Module):
    def __init__(
            self, n_feats, wn=lambda x: torch.nn.utils.weight_norm(x), act=nn.ReLU(True)):
        super(WIRW, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        body = []
        body.append(
            wn(nn.Conv2d(n_feats , n_feats*2 , kernel_size=1, padding=0)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(n_feats*2, n_feats//2, kernel_size=1, padding=0)))
        body.append(
            wn(nn.Conv2d(n_feats//2, n_feats, kernel_size=3, padding=1)))

        self.body = nn.Sequential(*body)
        self.SAlayer = sa_layer(n_feats)

    def forward(self, x):
        y = self.res_scale(self.SAlayer(self.body(x))) + self.x_scale(x)
        return y