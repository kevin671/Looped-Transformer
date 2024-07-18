import torch
import torch.nn as nn

from enum import StrEnum


# Lie-Trotter splitting scheme with the Euler’s method
class LieTrotterSplittingScheme(nn.Module):
    def __init__(self, transformer):
        super(LieTrotterSplittingScheme, self).__init__()
        self.transformer = transformer

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.transformer.attn(x)
        x = self.transformer.mlp(x)
        return x


class KolmogorovArnoldNetSheme(nn.Module):
    def __init__(self, transformer):
        super(KolmogorovArnoldNetSheme, self).__init__()
        self.transformer = transformer

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.transformer.mlp(x)
        x = self.transformer.attn(x)
        return x


class TrotterAdditiveSplittingScheme(nn.Module):
    def __init__(self, transformer):
        super(LieTrotterSplittingScheme, self).__init__()
        self.transformer = transformer

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x1 = self.transformer.attn(x)
        x2 = self.transformer.mlp(x)
        return (x1 + x2) / 2


# Strang-Marchuk splitting scheme
class StrangMarchukSplittingScheme(nn.Module):
    def __init__(self, transformer):
        super(StrangMarchukSplittingScheme, self).__init__()
        self.transformer = transformer

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = 0.5 * self.transformer.attn(x)
        x = self.transformer.mlp(x)
        x = 0.5 * self.transformer.attn(x)
        return x


class StrangAdditiveSplittingScheme(nn.Module):
    def __init__(self, transformer):
        super(StrangMarchukSplittingScheme, self).__init__()
        self.transformer = transformer

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x1 = self.transformer.attn(x)
        x1 = self.transformer.mlp(x1)

        x2 = self.transformer.mlp(x)
        x2 = self.transformer.attn(x2)
        return (x1 + x2) / 2


def build_scheme(scheme: str, transformer):
    if scheme == "Lie-Trotter":
        return LieTrotterSplittingScheme(transformer)
    elif scheme == "Strang-Marchuk":
        return StrangMarchukSplittingScheme(transformer)
    elif scheme == "Kolmogorov-Arnold":
        return KolmogorovArnoldNetSheme(transformer)
    elif scheme == "Trotter-Additive":
        return TrotterAdditiveSplittingScheme(transformer)
    elif scheme == "Strang-Additive":
        return StrangAdditiveSplittingScheme(transformer)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


class Scheme(StrEnum):
    LIE_TROTTER = "Lie-Trotter"
    KOLMOGOROV_ARNOLD = "Kolmogorov-Arnold"
    STRANG_MARCHUK = "Strang-Marchuk"
    TROTTER_ADDITIVE = "Trotter-Additive"
    STRANG_ADDITIVE = "Strang-Additive"
