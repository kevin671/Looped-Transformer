import torch
import torch.nn as nn


class LoRAHyperNetwork(nn.Module):
    def __init__(self, h_dim: int, rank: int, t_dim: int):
        super(LoRAHyperNetwork, self).__init__()

        self.wa = nn.Parameter(torch.Tensor(rank, h_dim, t_dim))
        self.ba = nn.Parameter(torch.Tensor(rank, h_dim))

        self.wb = nn.Parameter(torch.Tensor(h_dim, rank, t_dim))
        self.bb = nn.Parameter(torch.Tensor(h_dim, rank))

    def forward(self, temb: torch.Tensor):
        lora_a = torch.einsum("rht,ht->rh", self.wa, temb) + self.ba
        lora_b = torch.einsum("hrh,rht->ht", self.wb, temb) + self.bb
        return lora_a, lora_b
