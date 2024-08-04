# %%
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import GPTBase
from model import Block, LayerNorm


@dataclass
class LoopedGPTConfig:
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
    use_input_injection: bool = False
    n_truncated: int = 100  # 10
    t_dim: int = 320


class LoopedGPT(GPTBase):
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

    config = LoopedGPTConfig(n_loops=200)
    model = LoopedGPT(config)

    """
    ckpt = torch.load(
        "/work/gg45/g45004/Looped-Transformer/nanoGPT/out/looped_1_layer_200_loop_wikitext-103/ckpt.pt",
        map_location="cpu",
    )
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    """

    x, y = get_batch("train")
    # print(x.shape, y.shape)  # torch.Size([2, 1024]) torch.Size([2, 1024])
    # print(x, y)
    # input = torch.randint(0, 50257, (1, 12))
    output, _ = model(x, y)

# %%
