import torch
import torch.nn as nn
from math import sqrt
from torch.nn import init

def make_model(args, parent=False):
    return IRFRN(args)



class FRCU(nn.Module):
    def __init__(self, channels):
        super(FRCU, self).__init__()
        #
        self.residual_encode = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(channels)
        ])
        #
        self.residual_decode = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(channels)
        ])

    def forward(self, x):
        encode = self.residual_encode(x) + x
        decode = self.residual_decode(encode) + encode
        return decode

class IRFRN(nn.Module):
    def __init__(self, args):
        super(IRFRN, self).__init__()
        self.n_frcus = 8
        n_colors = 3
        n_feats = 64
        kernel_size = 3
        scale = 1

        # feature shallow extraction
        self.head = nn.Sequential(*[
            nn.Conv2d(n_colors, n_feats * 4, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats * 4),
            nn.Conv2d(n_feats * 4, n_feats, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats)
        ])

        #
        self.hbd_flow = nn.Sequential(*[
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats),
            nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2),
            nn.PReLU(n_feats)
        ])

        #
        self.convs = nn.ModuleList()
        for i in range(self.n_frcus):
            self.convs.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2))

        self.activate = nn.ModuleList()
        for i in range(self.n_frcus):
            self.activate.append(nn.PReLU(n_feats))

        #
        self.frcus = nn.ModuleList()
        for i in range(self.n_frcus):
            self.frcus.append(FRCU(n_feats))

        # reconstruction
        self.tail = nn.Conv2d(n_feats * (self.n_frcus+1), n_colors, kernel_size, padding=kernel_size // 2)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, l):
        lbd = self.head(l)

        out = []
        hbd_sets = []

        residual = self.hbd_flow(lbd) + lbd
        out.append(residual)

        for i in range(self.n_frcus):
            r = self.frcus[i](residual)
            residual_m = residual + r
            residual = self.convs[i](residual_m)
            residual = self.activate[i](residual)
            out.append(residual)


        predict = self.tail(torch.cat(out, dim=1))
        predict = predict + l
        return predict