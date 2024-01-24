import abc

import torch as th
from torch import nn


def get_non_linear(args):
    if args.non_linear == "gelu":
        non_linear = nn.GELU()
    elif args.non_linear == "selu":
        non_linear = nn.SELU()
    else:
        raise ValueError(f"Unknown non_linear:{args.non_linear}")
    return non_linear


class Linear3D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        assert x.ndim == 3
        B, L, C = x.shape
        result = self.linear(x.reshape(-1, C))
        return result.reshape(B, L, -1)


class BaseSTEmbedding(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.time_period = (th.tensor((365, 30.5, 7, 1)) * 24).to(device)

    @abc.abstractmethod
    def forward(self, s_t):
        pass


class SimpleSTEmbedding(BaseSTEmbedding):
    def forward(self, s_t: th.Tensor):
        assert (s_t.ndim == 3) and (s_t.shape[2] == 3)
        s, t = s_t[:, :, :2], s_t[:, :, -1]
        t_sin = th.sin(2 * th.pi * t.unsqueeze(2) * (1 / self.time_period))
        t_cos = th.cos(2 * th.pi * t.unsqueeze(2) * (1 / self.time_period))
        result = th.concat([s, t_sin, t_cos], dim=2)
        return result


class LinearSTEmbedding(BaseSTEmbedding):
    def __init__(self, s_embed_dim, t_embed_dim, device):
        super().__init__(device)
        self.s_linear = Linear3D(2, s_embed_dim)
        self.t_linear = Linear3D(8, t_embed_dim)

    def forward(self, s_t: th.Tensor):
        assert (s_t.ndim == 3) and (s_t.shape[2] == 3)
        s, t = s_t[:, :, :2], s_t[:, :, -1]
        s_embed = self.s_linear(s)
        t_sin = th.sin(2 * th.pi * t.unsqueeze(2) * (1 / self.time_period))
        t_cos = th.cos(2 * th.pi * t.unsqueeze(2) * (1 / self.time_period))
        t_embed = self.t_linear(th.concat([t_sin, t_cos], dim=2))
        result = th.concat([s_embed, t_embed], dim=2)
        return result


class PolySTEmbedding(BaseSTEmbedding):
    def forward(self, s_t: th.Tensor):
        assert (s_t.ndim == 3) and (s_t.shape[2] == 3)
        s, t = s_t[:, :, :2], s_t[:, :, -1]
        s_embed = th.concat([s**i for i in range(3)], dim=2)
        t_sin = th.sin(2 * th.pi * t.unsqueeze(2) * (1 / self.time_period))
        t_cos = th.cos(2 * th.pi * t.unsqueeze(2) * (1 / self.time_period))
        t_embed = th.concat([t_sin, t_cos], dim=2)
        result = th.concat([s_embed, t_embed], dim=2)
        return result


def get_st_embedding(args, device):
    st_embed_block = {
        "simple": SimpleSTEmbedding(device),
        "linear": LinearSTEmbedding(args.s_embed_dim, args.t_embed_dim, device),
        "poly": PolySTEmbedding(device)

    }[args.st_embed]
    return st_embed_block

