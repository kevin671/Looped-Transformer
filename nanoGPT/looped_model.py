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
    input_injection: bool = False


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
        input_emb = tok_emb + pos_emb
        x = self.transformer.drop(input_emb)

        debug_data = []
        for _ in range(self.n_loops):
            for block in self.transformer.h:
                if self.config.input_injection:
                    x = block(x + input_emb)
                else:
                    x = block(x)
                if debug:
                    print(x[0, 0, :10])
                    # print(x.shape)  # torch.Size([2, 1024, 768])
                    x = self.transformer.ln_f(x)
                    logits = self.lm_head(x)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-1,
                    )
                    # print(logits.shape)  # torch.Size([2, 1024, 50257])
                    # print(torch.argmax(logits, dim=-1).shape)  # torch.Size([2, 1024])
                    debug_data.append(loss.detach().cpu())

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

    @torch.no_grad()
    def picard_forward(
        self,
        idx,
        targets=None,
        parallel: int = 100,
        tolerance: float = 1e-3,
    ):
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

        # Picard iteration alogrithm
        assert self.config.n_layer == 1, "n_layer must be 1"

        device = x.device
        batch_size, seq_len, hidden_dim = x.shape

        assert batch_size == 1, "batch size must be 1"

        stats_pass_count = 0
        stats_flop_count = 0

        # timesteps = torch.arange(
        #    0, num_loops * len(self.transformer.looped_block.layers), device=device
        # )
        timesteps = torch.arange(0, self.n_loops, device=device)
        parallel = min(parallel, len(timesteps))

        begin_idx = 0
        end_idx = parallel
        latents_time_evolution_buffer = torch.stack([x] * (len(timesteps) + 1))

        scaled_tolerance = tolerance**2

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        # start.record()

        while begin_idx < len(timesteps):
            print(f"{begin_idx=}, {end_idx=}", flush=True)
            parallel_len = end_idx - begin_idx

            block_latents = latents_time_evolution_buffer[begin_idx:end_idx]
            t_vec = timesteps[begin_idx:end_idx]

            # if add_inputs_embeds:
            #    # num_layers間隔でinputs_embedsを足す
            #    block_latents[:: len(self.layers)] += inputs_embeds

            model_output = torch.zeros_like(block_latents)
            for i, t in enumerate(t_vec):
                # model_output[i] = self.transformer.looped_block(block_latents[i], t)
                model_output[i] = self.transformer.h[0].derivative(block_latents[i])

            delta = model_output.reshape(parallel_len, 1, seq_len, hidden_dim)
            cumulative_delta = torch.cumsum(delta, dim=0)
            # print(f"{cumulative_delta.shape=}", flush=True)

            block_latents_new = (
                latents_time_evolution_buffer[begin_idx][None,] + cumulative_delta
            )  # (parallel_len, seq_len, hidden_dim)
            cur_error_vec = (
                block_latents_new - block_latents
            )  # (parallel_len, 1, seq_len, hidden_dim)
            cur_error = (
                torch.linalg.norm(cur_error_vec, dim=-1).pow(2).mean(dim=-1).squeeze(1)
            )  # (parallel_len,)

            print(f"{cur_error=}", flush=True)
            # find the first index of the vector error_ratio that is greater than error tolerance
            # we can shift the window for the next iteration up to this index

            # pad with a large number at last to avoid the case where all errors are below tolerance
            cur_error = torch.cat(
                [
                    cur_error,
                    torch.tensor(
                        [
                            1e6,
                        ],
                        device=device,
                    ),
                ]
            )
            ind = torch.argmax((cur_error > scaled_tolerance).int()).item()

            # compute the new begin and end idxs for the window
            new_begin_idx = begin_idx + min(1 + ind, parallel)
            new_end_idx = min(new_begin_idx + parallel, len(timesteps))

            # store the computed latents for the current window in the global buffer
            latents_time_evolution_buffer[begin_idx + 1 : end_idx + 1] = (
                block_latents_new
            )
            # initialize the new sliding window latents with the end of the current window,
            # should be better than random initialization
            latents_time_evolution_buffer[
                end_idx : new_end_idx + 1
            ] = latents_time_evolution_buffer[end_idx][
                None,
            ]

            begin_idx = new_begin_idx
            end_idx = new_end_idx

            stats_pass_count += 1
            stats_flop_count += parallel_len

        x = latents_time_evolution_buffer[-1]

        # end.record()

        # Waits for everything to finish running
        # torch.cuda.synchronize()

        # print(f"elapsed time: {start.elapsed_time(end)} ms")

        stats = {
            "pass_count": stats_pass_count,
            "flops_count": stats_flop_count,
            # "time": start.elapsed_time(end),
        }
        print(stats)

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

    x, y = get_batch("train")
    # print(x.shape, y.shape)  # torch.Size([2, 1024]) torch.Size([2, 1024])
    # print(x, y)
    # input = torch.randint(0, 50257, (1, 12))
    output, _, debugs = model(x, y, debug=True)
    # print(output[0])  # torch.Size([1, 50257])
    # print(len(debugs))  # 200
    # print(debugs[0].shape)  # torch.Size([1, 12, 768])

    # print(debugs)

    # show the norm of the tensor at each loop
    # norms = [torch.linalg.norm(d).item() for d in debugs]
    # print as figure
    import matplotlib.pyplot as plt

    plt.plot(debugs)
    # plt.xlabel("n_loops")
    # plt.ylabel("Norm")
    # plt.title("Norms of the tensor at each loop")
    plt.show()
    # output = model.picard_forward(input)

# %%
