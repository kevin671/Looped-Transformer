import torch
import torch.nn as nn

from enum import StrEnum


# given time t, the hypernetwork generates a set of parameters for the transformer
class HyperNetwork(nn.Module):
    def __init__(self, config):
        super(HyperNetwork, self).__init__()
        self.config = config
        self.time_embed = nn.Sequential(
            nn.Linear(1, config.hidden_t_dim * 4),
            nn.LeakyReLU(),
            nn.Linear(config.hidden_t_dim * 4, config.hidden_t_dim),
        )
        self.hypernet = nn.Sequential(
            nn.Linear(config.hidden_t_dim, config.hypernet_hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(
                config.hypernet_hidden_dim, config.hidden_dim * 3 * config.hidden_dim
            ),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = self.time_embed(t)
        print(t.shape)
        return self.hypernet(t).reshape(
            self.config.hidden_dim * 3, self.config.hidden_dim
        )


# Lie-Trotter splitting scheme with the Euler’s method
class LieTrotterSplittingScheme(nn.Module):
    def __init__(self, transformer, config):
        super(LieTrotterSplittingScheme, self).__init__()
        self.transformer = transformer
        self.hypernet = HyperNetwork(config)
        self.h = config.step_size

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        qkv = self.hypernet(t)
        print(qkv.shape)
        x += self.transformer.attn(x, qkv) * self.h
        x += self.transformer.mlp(x) * self.h
        return x


class KolmogorovArnoldNetSheme(nn.Module):
    def __init__(self, transformer, config):
        super(KolmogorovArnoldNetSheme, self).__init__()
        self.transformer = transformer
        self.hypernet = HyperNetwork(config)
        self.h = config.step_size

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        qkv = self.hypernet(t)
        x += self.transformer.mlp(x) * self.h
        x += self.transformer.attn(x, qkv) * self.h
        return x


# Strang-Marchuk splitting scheme
class StrangMarchukSplittingScheme(nn.Module):
    def __init__(self, transformer, config):
        super(StrangMarchukSplittingScheme, self).__init__()
        self.transformer = transformer
        self.hypernet = HyperNetwork(config)
        self.h = config.step_size

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        qkv = self.hypernet(t)
        x += 0.5 * self.transformer.mlp(x) * self.h
        x += self.transformer.attn(x, qkv) * self.h
        x += 0.5 * self.transformer.mlp(x) * self.h
        return x


class TrotterAdditiveSplittingScheme(nn.Module):
    def __init__(self, transformer, config):
        super(TrotterAdditiveSplittingScheme, self).__init__()
        self.transformer = transformer
        self.hypernet = HyperNetwork(config)
        self.h = config.step_size

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        qkv = self.hypernet(t)
        x1 = self.transformer.attn(x, qkv=qkv) * self.h * 2
        x2 = self.transformer.mlp(x) * self.h * 2
        return x + (x1 + x2) / 2

    def step(self, x: torch.Tensor, t: torch.Tensor):
        qkv = self.hypernet(t)
        x1 = self.transformer.attn(x, qkv=qkv) * self.h
        x2 = self.transformer.mlp(x) * self.h
        return x1 + x2


class StrangAdditiveSplittingScheme(nn.Module):
    def __init__(self, transformer, config):
        super(StrangMarchukSplittingScheme, self).__init__()
        self.transformer = transformer
        self.hypernet = HyperNetwork(config)
        self.h = config.step_size

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        qkv = self.hypernet(t)
        x1 = self.transformer.attn(x, qkv=qkv) * self.h
        x1 = self.transformer.mlp(x1) * self.h

        x2 = self.transformer.mlp(x) * self.h
        x2 = self.transformer.attn(x2, qkv=qkv) * self.h
        return x + (x1 + x2) / 2

    def step(self, x: torch.Tensor, t: torch.Tensor):
        qkv = self.hypernet(t)
        x1 = self.transformer.attn(x, qkv=qkv) * self.h
        x1 = self.transformer.mlp(x1) * self.h

        x2 = self.transformer.mlp(x) * self.h
        x2 = self.transformer.attn(x2, qkv=qkv) * self.h
        return x1 + x2


def build_scheme(scheme: str, transformer, config):
    if scheme == "Lie-Trotter":
        return LieTrotterSplittingScheme(transformer, config)
    elif scheme == "Strang-Marchuk":
        return StrangMarchukSplittingScheme(transformer, config)
    elif scheme == "Kolmogorov-Arnold":
        return KolmogorovArnoldNetSheme(transformer, config)
    elif scheme == "Trotter-Additive":
        return TrotterAdditiveSplittingScheme(transformer, config)
    elif scheme == "Strang-Additive":
        return StrangAdditiveSplittingScheme(transformer, config)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


class Scheme(StrEnum):
    LIE_TROTTER = "Lie-Trotter"
    KOLMOGOROV_ARNOLD = "Kolmogorov-Arnold"
    STRANG_MARCHUK = "Strang-Marchuk"
    TROTTER_ADDITIVE = "Trotter-Additive"
    STRANG_ADDITIVE = "Strang-Additive"
