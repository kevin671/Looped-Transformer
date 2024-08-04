# %%
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import GPTBase
from model import Block, CausalSelfAttention, LayerNorm


@dataclass
class HyperGPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_layer: int = 1
    n_loops: int = 100
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    use_input_injection: bool = False
    n_truncated: int = 100  # 10
    t_dim: int = 160  # 320
    hypernet_dim: int = 384

    # share_attn: bool = True
    # share_mlp: bool = False
    # share_normalization: bool = False  # not use


class SinusoidalPositionEmbeddings(nn.Module):
    # (batch_size, 1) -> (batch_size, 1, dim)
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]  # (batch_size, 1, dim)
        # embeddings = time * embeddings  # (batch_size, dim)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(config.t_dim),
            nn.Linear(config.t_dim, config.t_dim * 4),
            nn.SiLU(),
            nn.Linear(config.t_dim * 4, config.t_dim * 4),
            nn.SiLU(),
            nn.Linear(config.t_dim * 4, config.n_embd),
        )

    def forward(self, time):
        return self.time_mlp(time)


class HyperNetworkMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, config.hypernet_dim),
            nn.GELU(),
            nn.Linear(
                config.hypernet_dim,
                config.n_embd * config.n_embd * 8 + 5 * config.n_embd,
            ),
            nn.Dropout(config.dropout),
        )

    def forward(self, time):
        config = self.config
        weight = self.net(time)
        # split weight
        fc_w = weight[: config.n_embd * config.n_embd * 4].reshape(
            config.n_embd * 4, config.n_embd
        )
        fc_b = weight[
            config.n_embd * config.n_embd * 4 : config.n_embd * config.n_embd * 4
            + config.n_embd * 4
        ]
        proj_w = weight[
            config.n_embd * config.n_embd * 4
            + config.n_embd * 4 : config.n_embd * config.n_embd * 4
            + config.n_embd * 4
            + config.n_embd * config.n_embd * 4
        ].reshape(config.n_embd, config.n_embd * 4)
        proj_b = weight[
            config.n_embd * config.n_embd * 4
            + config.n_embd * 4
            + config.n_embd * config.n_embd * 4 :
        ]
        # print(fc_w.shape, fc_b.shape, proj_w.shape, proj_b.shape)
        return fc_w, fc_b, proj_w, proj_b


class HyperMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        # self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        self.time_mlp = TimeEmbedding(config)
        self.weight_mlp = HyperNetworkMLP(config)

    def forward(self, x, idx):
        t = self.time_mlp(idx).squeeze(0)
        # print(t.shape) # torch.Size([128])
        # delta_fc_w, delta_fc_b, delta_proj_w, delta_proj_b = self.weight_mlp(t)
        fc_w, fc_b, proj_w, proj_b = self.weight_mlp(t)
        # print(fc_w[:1])
        # x = self.c_fc(x)
        # x = self.gelu(x)
        # x = self.c_proj(x)
        # x = self.dropout(x)
        x = F.linear(t, fc_w, fc_b)
        x = self.gelu(x)
        x = F.linear(x, proj_w, proj_b)
        x = self.dropout(x)
        return x


class HyperBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = HyperMLP(config)

    def forward(self, x, idx):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x), idx)
        return x


# %%
class HyperGPT(GPTBase):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_loops = config.n_loops

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([HyperBlock(config) for _ in range(config.n_layer)]),
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
        input_emb = tok_emb + pos_emb
        x = self.transformer.drop(input_emb)

        for idx in range(self.n_loops):

            idx_emb = torch.tensor([idx], device=device)
            # t_emb = self.time_mlp(idx_emb).squeeze(0) # (1, 768) -> (768)

            if idx < self.n_loops - self.config.n_truncated:
                with torch.no_grad():
                    for block in self.transformer.h:
                        if self.config.use_input_injection:
                            x = block(x + input_emb, idx_emb)
                        else:
                            x = block(x, idx_emb)
            else:
                for block in self.transformer.h:
                    if self.config.use_input_injection:
                        x = block(x + input_emb, idx_emb)
                    else:
                        x = block(x, idx_emb)

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


class EmbeddingLoopedGPT(GPTBase):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_loops = config.n_loops

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
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

        self.time_mlp = TimeEmbedding(config)

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
        input_emb = tok_emb + pos_emb
        x = self.transformer.drop(input_emb)

        for idx in range(self.n_loops):

            idx_emb = torch.ones((b, 1), device=device) * idx
            t_emb = self.time_mlp(idx_emb)
            # print(t_emb.shape)  # torch.Size([bs, 1, 768])
            x = x + t_emb

            if idx < self.n_loops - self.config.n_truncated:
                with torch.no_grad():
                    for block in self.transformer.h:
                        if self.config.use_input_injection:
                            x = block(x + input_emb)
                        else:
                            x = block(x)
            else:
                for block in self.transformer.h:
                    if self.config.use_input_injection:
                        x = block(x + input_emb)
                    else:
                        x = block(x)

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
    data_dir = (
        "/work/gg45/g45004/Looped-Transformer/preliminaries/gpt/data/wikitext-103"
    )

    def get_batch(split="train", batch_size=10, block_size=1024):
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

    config = HyperGPTConfig(n_loops=100)
    # model = EmbeddingLoopedGPT(config)
    model = HyperGPT(config)

    x, y = get_batch("train")
    # print(x.shape, y.shape)  # torch.Size([2, 1024]) torch.Size([2, 1024])
    # print(x, y)
    # input = torch.randint(0, 50257, (1, 12))
    output, _ = model(x, y)

# %%
