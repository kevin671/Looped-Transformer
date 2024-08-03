# %%
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import GPTBase
from hypernet import HyperNetworkAttnLoRA, HyperNetworkMLPLoRA


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class DynamicCausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        # self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
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

    def forward(self, x, qkv_w, qkv_b, o_w, o_b):
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = F.linear(x, qkv_w, qkv_b).split(self.n_embd, dim=2)
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
        y = self.resid_dropout(F.linear(y, o_w, o_b))
        return y


class DynamicMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        # self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, fc_w, fc_b, proj_w, proj_b):
        # x = self.c_fc(x)
        x = F.linear(x, fc_w, fc_b)
        x = self.gelu(x)
        # x = self.c_proj(x)
        x = F.linear(x, proj_w, proj_b)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = DynamicCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        # self.mlp = DynamicMLP(config)
        self.mlp = DynamicMLP(config)

    def forward(self, x, qkv_w, qkv_b, o_w, o_b, fc_w, fc_b, proj_w, proj_b):
        x = x + self.attn(self.ln_1(x), qkv_w=qkv_w, qkv_b=qkv_b, o_w=o_w, o_b=o_b)
        x = x + self.mlp(self.ln_2(x), fc_w, fc_b, proj_w, proj_b)
        return x


@dataclass
class DynamicLoopedGPTConfig:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    n_loops: int = 100
    n_layer: int = 1
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )


class DynamicLoopedGPT(GPTBase):
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
                # h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
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

        self.qkv_w = nn.Parameter(torch.randn(config.n_embd, 3 * config.n_embd))
        self.qkv_b = nn.Parameter(torch.zeros(3 * config.n_embd))
        self.o_w = nn.Parameter(torch.randn(config.n_embd, config.n_embd))
        self.o_b = nn.Parameter(torch.zeros(config.n_embd))
        self.fc_w = nn.Parameter(torch.randn(config.n_embd, config.n_embd))
        self.fc_b = nn.Parameter(torch.zeros(config.n_embd))
        self.proj_w = nn.Parameter(torch.randn(config.n_embd, config.n_embd))
        self.proj_b = nn.Parameter(torch.zeros(config.n_embd))
        # nn.init.kaiming_normal_(self.qkv_w)
        # nn.init.kaiming_normal_(self.o_w)
        # nn.init.kaiming_normal_(self.fc_w)
        # nn.init.kaiming_normal_(self.proj_w)
        scaling_std = 0.02  # / (config.n_layer * config.n_loops)
        # print(f"scaling std: {scaling_std}")
        nn.init.normal_(self.qkv_w, mean=0.0, std=scaling_std)
        nn.init.normal_(self.o_w, mean=0.0, std=scaling_std)
        nn.init.normal_(self.fc_w, mean=0.0, std=scaling_std)
        nn.init.normal_(self.proj_w, mean=0.0, std=scaling_std)

        t_dim = 512
        r = 1
        self.hypernet_attn = HyperNetworkAttnLoRA(
            t_dim=t_dim, n_embd=config.n_embd, r=r
        )
        # self.hypernet_attn_bias = HyperNetworkBias(t_dim=t_dim, n_embd=3 * config.n_embd)
        self.hypernet_mlp = HyperNetworkMLPLoRA(t_dim=t_dim, n_embd=config.n_embd, r=r)
        # self.hypernet_mlp_bias = HyperNetworkBias(t_dim=t_dim, n_embd=config.n_embd)

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, idx, targets=None, debug=False):
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
        # for block in self.transformer.h:
        # x = block(x)
        debug_data = []
        for t in range(self.n_loops):
            lora_qkv_w, lora_o_w = self.hypernet_attn(torch.tensor(t, device=device))
            lora_fc_w, lora_proj_w = self.hypernet_mlp(torch.tensor(t, device=device))
            """
            print(f"{self.qkv_w[:10, :10]=}")
            print(f"{lora_qkv_w[:10, :10]=}")
            self.qkv_w = nn.Parameter(self.qkv_w + lora_qkv_w)
            print(f"{self.qkv_w[:10, :10]=}")
            self.o_w = nn.Parameter(self.o_w + lora_o_w)
            self.fc_w = nn.Parameter(self.fc_w + lora_fc_w)
            self.proj_w = nn.Parameter(self.proj_w + lora_proj_w)

            x = self.transformer.h(
                x,
                self.qkv_w.transpose(0, 1),
                self.qkv_b,
                self.o_w,
                self.o_b,
                self.fc_w,
                self.fc_b,
                self.proj_w,
                self.proj_b,
            )
            """

            qkv_w = nn.Parameter(self.qkv_w + lora_qkv_w)
            o_w = nn.Parameter(self.o_w + lora_o_w)
            fc_w = nn.Parameter(self.fc_w + lora_fc_w)
            proj_w = nn.Parameter(self.proj_w + lora_proj_w)

            # print(x[0, 0, :10])
            # print(f"{self.qkv_w[:10, :10]=}")
            # print(f"{lora_qkv_w[:10, :10]=}")
            # print(f"{qkv_w[:10, :10]=}")

            x = self.transformer.h(
                x,
                qkv_w.transpose(0, 1),
                self.qkv_b,
                o_w,
                self.o_b,
                fc_w,
                self.fc_b,
                proj_w,
                self.proj_b,
            )
            if debug:
                debug_data.append(x)

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

        if debug:
            return logits, loss, debug_data

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

    config = DynamicLoopedGPTConfig()
    model = DynamicLoopedGPT(config)

    ckpt = torch.load(
        "/work/gg45/g45004/Looped-Transformer/nanoGPT/out/dynamic_looped_1_layer_100_loop_wikitext-103/ckpt.pt",
        map_location="cpu",
    )
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

    output, _, debugs = model(x, debug=True)
    print(output[0])  # torch.Size([1, 50257])
    # print(len(debugs))  # 200
    # print(debugs[0].shape)  # torch.Size([1, 12, 768])

    # show the norm of the tensor at each loop
    # norms = [torch.linalg.norm(d).item() for d in debugs]
    # print as figure
    # import matplotlib.pyplot as plt

    # plt.plot(norms)
    # plt.xlabel("n_loops")
    # plt.ylabel("Norm")
    # plt.title("Norms of the tensor at each loop")
    # plt.show()
# %%


# %%
