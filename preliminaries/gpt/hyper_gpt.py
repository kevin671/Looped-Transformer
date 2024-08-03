# %%
# https://nn.labml.ai/hypernetworks/hyper_lstm.html
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import GPTBase
from model import LayerNorm


class HyperLayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, config):
        super().__init__()
        # self.weight = nn.Parameter(torch.ones(ndim))
        # self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

        self.z_w = nn.Embedding(config.n_layer, config.n_z)
        self.z_b = nn.Embedding(config.n_layer, config.n_z)

        self.d_w = nn.Linear(config.n_z, config.n_embd)
        self.d_b = nn.Linear(config.n_z, config.n_embd)

    def forward(self, input, i_layer):
        # return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
        z_w = self.z_w(i_layer)
        z_b = self.z_b(i_layer)
        d_w = self.d_w(z_w)
        d_b = self.d_b(z_b)
        return F.layer_norm(input, d_w.shape, d_w, d_b, 1e-5)


@dataclass
class HyperGPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    hyper_size: int = 256
    n_z: int = 256


class HyperMLP(GPTBase):

    def __init__(self, config):
        super().__init__()
        # self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        # self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.z_fc = nn.Embedding(config.n_layer, config.n_z)
        self.z_fc_b = nn.Embedding(config.n_layer, config.n_z)

        self.z_proj = nn.Embedding(config.n_layer, config.n_z)
        self.z_proj_b = nn.Embedding(config.n_layer, config.n_z)

        self.d_fc = nn.Linear(config.n_z, config.n_embd * 4)
        self.d_fc_b = nn.Linear(config.n_z, config.n_embd * 4)
        self.d_proj = nn.Linear(config.n_z, config.n_embd)
        self.d_proj_b = nn.Linear(config.n_z, config.n_embd)

        self.w_fc = nn.Parameter(torch.zeros(config.n_embd, config.n_embd * 4))
        self.w_proj = nn.Parameter(torch.zeros(config.n_embd, config.n_embd * 4))

        # self.layer_norm = nn.ModuleList([nn.LayerNorm(config.n_embd) for _ in range(4)])

    def forward(self, x, i_layer):
        z_fc = self.z_fc(i_layer)
        z_fc_b = self.z_fc_b(i_layer)
        z_proj = self.z_proj(i_layer)
        z_proj_b = self.z_proj_b(i_layer)

        d_fc = self.d_fc(z_fc)  # shape (n_embd * 4)
        d_proj = self.d_proj(z_proj)  # shape (n_embd * 4)

        # x = self.c_fc(x)
        x = d_fc * torch.einsum("ij,bti->btj", self.w_fc, x) + self.d_fc_b(z_fc_b)
        x = self.gelu(x)
        # x = self.c_proj(x)
        x = d_proj * torch.einsum("ij,btj->bti", self.w_proj, x) + self.d_proj_b(
            z_proj_b
        )
        x = self.dropout(x)
        return x


class HyperCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.z_attn = nn.Embedding(config.n_layer, config.n_z * 3)
        self.z_attn_b = nn.Embedding(config.n_layer, config.n_z * 3)
        # output projection
        # self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.z_proj = nn.Embedding(config.n_layer, config.n_z)
        self.z_proj_b = nn.Embedding(config.n_layer, config.n_z)

        self.d_attn = nn.ModuleList(
            [nn.Linear(config.n_z, config.n_embd) for _ in range(3)]
        )
        self.d_attn_b = nn.ModuleList(
            [nn.Linear(config.n_z, config.n_embd) for _ in range(3)]
        )
        self.d_proj = nn.Linear(config.n_z, config.n_embd)
        self.d_proj_b = nn.Linear(config.n_z, config.n_embd)

        self.w_attn = nn.ParameterList(
            [nn.Parameter(torch.randn(config.n_embd, config.n_embd)) for _ in range(3)]
        )
        self.w_proj = nn.Parameter(torch.randn(config.n_embd, config.n_embd))

        # self.attn_layer_norm = nn.LayerNorm(config.n_embd)

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x, i_layer):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        z_attn = self.z_attn(i_layer).chunk(3, dim=-1)
        z_attn_b = self.z_attn_b(i_layer).chunk(3, dim=-1)

        qkv = []
        for i in range(3):
            d_attn = self.d_attn[i](z_attn[i])
            qkv.append(
                d_attn * torch.einsum("ic,btc->bti", self.w_attn[i], x)
                + self.d_attn_b[i](z_attn_b[i])
            )
        q, k, v = qkv

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        # y = self.resid_dropout(self.c_proj(y))
        z_proj = self.z_proj(i_layer)
        z_proj_b = self.z_proj_b(i_layer)

        d_proj = self.d_proj(z_proj)
        y = d_proj * torch.einsum("ic,btc->bti", self.w_proj, y) + self.d_proj_b(
            z_proj_b
        )
        return y


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = HyperLayerNorm(config)
        self.attn = HyperCausalSelfAttention(config)
        self.ln_2 = HyperLayerNorm(config)
        self.mlp = HyperMLP(config)

    def forward(self, x, i_layer):
        x = x + self.attn(self.ln_1(x, i_layer), i_layer)
        x = x + self.mlp(self.ln_2(x, i_layer), i_layer)
        return x

    def derivative(self, x):
        return self.attn(self.ln_1(x)) + self.mlp(self.ln_2(x))


class HyperGPT(GPTBase):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=Block(config),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for i_layer in range(self.config.n_layer):
            i_layer = torch.tensor(i_layer, device=device)
            x = self.transformer.h(x, i_layer)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


# %%
if __name__ == "__main__":
    import os

    import numpy as np

    device = "cpu"
    device_type = "cpu"
    data_dir = "/work/gg45/g45004/Looped-Transformer/nanoGPT/data/wikitext-103"

    def get_batch(split="train", batch_size=2, block_size=1024):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == "train":
            data = np.memmap(
                os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
            )
        else:
            data = np.memmap(
                os.path.join(data_dir, "validation.bin"), dtype=np.uint16, mode="r"
            )
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack(
            [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
                for i in ix
            ]
        )
        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    x, y = get_batch()

    config = HyperGPTConfig()
    model = HyperGPT(config)

    logits, loss = model(x, y)

# %%
