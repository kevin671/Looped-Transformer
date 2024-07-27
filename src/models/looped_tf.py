# %%
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from looped_tf_block import MLP, MultiheadAttention, LayerNorm, PreNormLayer
from einops import repeat

from scheme import (
    Scheme,
    LieTrotterSplittingScheme,
    StrangMarchukSplittingScheme,
    build_scheme,
)


@dataclass
class LoopedConfig:
    sheme: Scheme = Scheme.LIE_TROTTER  # STRANG_MARCHUK
    picard_sheme: Scheme = Scheme.TROTTER_ADDITIVE  # STRANG_ADDITIV
    use_time_embed: bool = True
    block_size: int = 128
    vocab_size: int = 1024
    n_layer: int = 1
    n_head: int = 4
    hidden_dim: int = 64
    n_reg_tokens: int = 0
    bias: bool = True
    dropout: float = 0.0
    hidden_t_dim: int = 16
    hypernet_hidden_dim: int = 64
    step_size: float = 0.1


class LoopedTransformer(nn.Module):

    def __init__(self, config: LoopedConfig):
        super().__init__()
        # assert config.vocab_size is not None
        # assert config.block_size is not None
        self.config = config

        # n_layer == 1
        assert config.n_layer == 1

        self.hidden_t_dim = config.hidden_dim * 4

        time_embed_dim = self.hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_t_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, config.hidden_dim),
        )

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.hidden_dim),
                wpe=nn.Embedding(config.block_size, config.hidden_dim),
                drop=nn.Dropout(config.dropout),
                mlp=PreNormLayer(MLP(config), config.hidden_dim, config.bias),
                attn=PreNormLayer(
                    MultiheadAttention(config),
                    config.hidden_dim,
                    config.bias,
                ),
                ln_f=LayerNorm(config.hidden_dim, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        if config.n_reg_tokens > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(config.n_reg_tokens, config.hidden_dim)
            )
        else:
            self.register_tokens = None  # Optionally set to None or skip entirely

        self.solver = (
            StrangMarchukSplittingScheme(self.transformer, config)
            if config.sheme == Scheme.STRANG_MARCHUK
            else LieTrotterSplittingScheme(self.transformer, config)
        )
        # LieTrotterSplittingScheme(self.transformer)

        self.picard_solver = build_scheme(config.picard_sheme, self.transformer, config)

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        inputs_idx=None,
        inputs_embeds=None,
        targets=None,
        num_loops=1000,
        rm_pos_embd=False,
        add_inputs_embeds=False,
        output_intermediate=False,
    ):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        assert (inputs_idx is not None) ^ (
            inputs_embeds is not None
        ), "Either inputs_idx or inputs_embeds must be not None"

        if inputs_idx is not None:
            device = inputs_idx.device
            batch, seq_len = inputs_idx.size()
            assert (
                seq_len <= self.config.block_size
            ), f"Cannot forward sequence of length {seq_len}, block size is only {self.config.block_size}"
            tok_emb = self.transformer.wte(inputs_idx)

        if inputs_embeds is not None:
            device = inputs_embeds.device
            batch, seq_len, _ = inputs_embeds.shape
            tok_emb = inputs_embeds

        # print(f"{tok_emb.shape=}")

        # repeat register token
        if self.register_tokens is not None:
            r = repeat(self.register_tokens, "n d -> b n d", b=batch)
            x = torch.cat(
                [r, tok_emb], dim=1
            )  # (batch, seq_len + n_reg_tokens, hidden_dim)
        else:
            x = tok_emb

        position_ids = torch.arange(
            0, seq_len + self.config.n_reg_tokens, dtype=torch.long, device=device
        )  # shape (t)
        pos_emb = self.transformer.wpe(
            position_ids
        )  # position embeddings of shape (1, t, hidden_dim)
        if rm_pos_embd:
            pos_emb = torch.zeros_like(pos_emb, device=device)

        x += pos_emb

        # print(f"{x.shape=}")

        # loop solver
        for t in range(num_loops):
            x = self.transformer.drop(x)
            x = self.solver(
                x,
                torch.tensor(t, device=device, dtype=torch.float32).expand(batch, 1),
            )

        x = self.transformer.ln_f(x)
        return x

        # start.record()
        """
        embeds = []
        for t in range(num_loops):
            num_block_layers = len(self.transformer.looped_block.layers)
            for t in range(num_block_layers):
                x = x + self.transformer.looped_block(x, t)
                if add_inputs_embeds and t > 0 and t % num_block_layers == 0:
                    x = x + inputs_embeds
            if output_intermediate:
                embeds.append(x)

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

        # end.record()
        # torch.cuda.synchronize()
        # print(f"elapsed time: {start.elapsed_time(end)} ms")
        # if output_intermediate:
        #    return x, embeds
        # return x

        return logits, loss
        """

    # https://github.com/AndyShih12/paradigms/blob/master/paradigms/stablediffusion_paradigms.py
    @torch.no_grad()
    def picard_forward(
        self,
        x,
        num_loops: int = 1000,
        parallel: int = 100,
        tolerance: float = 0.1,
        # position_ids=None,
        # rm_pos_embd=False,
        # add_inputs_embeds=False,
        output_intermediate=False,
    ):
        # assert not add_inputs_embeds, "add_inputs_embeds is not supported"

        # device = inputs_embeds.device
        device = x.device
        batch_size, seq_len, hidden_dim = x.shape

        assert batch_size == 1, "batch size must be 1"

        stats_pass_count = 0
        stats_flop_count = 0

        # timesteps = torch.arange(
        #    0, num_loops * len(self.transformer.looped_block.layers), device=device
        # )
        timesteps = torch.arange(0, num_loops, device=device)
        parallel = min(parallel, len(timesteps))

        begin_idx = 0
        end_idx = parallel
        latents_time_evolution_buffer = torch.stack([x] * (len(timesteps) + 1))

        scaled_tolerance = tolerance**2

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        while begin_idx < len(timesteps):
            parallel_len = end_idx - begin_idx

            block_latents = latents_time_evolution_buffer[begin_idx:end_idx]
            t_vec = timesteps[begin_idx:end_idx]

            # if add_inputs_embeds:
            #    # num_layers間隔でinputs_embedsを足す
            #    block_latents[:: len(self.layers)] += inputs_embeds

            model_output = torch.zeros_like(block_latents)
            for i, t in enumerate(t_vec):
                # model_output[i] = self.transformer.looped_block(block_latents[i], t)
                model_output[i] = self.picard_solver(block_latents[i], t)

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

        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()

        print(f"elapsed time: {start.elapsed_time(end)} ms")

        stats = {
            "pass_count": stats_pass_count,
            "flops_count": stats_flop_count,
            "time": start.elapsed_time(end),
        }
        print(stats)

        if output_intermediate:
            return x, latents_time_evolution_buffer[:: len()]

        return x


def generate_func(data_points, seq_len, dim):
    # random noise

    x = torch.randn(data_points, seq_len, dim)
    y = torch.randn(data_points, seq_len, dim)
    return x, y


def train(x, model, epochs=100, batch_size=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    l1 = torch.nn.L1Loss()
    for epoch in range(epochs):
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i : i + batch_size]
            y_batch = x[i : i + batch_size]
            optimizer.zero_grad()
            y_hat = model(inputs_embeds=x_batch, num_loops=10)
            loss = l1(y_hat, y_batch)
            loss.backward(retain_graph=True)
            optimizer.step()
        print(f"Epoch {epoch} Loss {loss.item()}")
    return model


def test(y, model):
    l1 = torch.nn.L1Loss()
    y_hat = model(inputs_embeds=y, num_loops=10)
    loss = l1(y_hat, y)
    print(f"Test Loss {loss.item()}")
    return y_hat


def plot(x, y, model):
    import matplotlib.pyplot as plt

    y_hat = model(inputs_embeds=x, num_loops=10)

    plt.plot(y[0, :, 0].detach().numpy(), label="y")
    plt.plot(y_hat[0, :, 0].detach().numpy(), label="y_hat")
    plt.legend()
    plt.show()


# %%
if __name__ == "__main__":
    config = LoopedConfig(
        sheme=Scheme.LIE_TROTTER,
        use_time_embed=True,
        block_size=128,
        vocab_size=128,
        n_layer=1,
        n_head=4,
        hidden_dim=64,
        n_reg_tokens=0,
        bias=True,
        dropout=0.0,
        hidden_t_dim=16,
        hypernet_hidden_dim=64,
        step_size=0.1,
    )

    batch_size = 5
    seq_len = 10
    num_loops = 1000

    inputs_embeds = torch.rand(batch_size, seq_len, config.hidden_dim)
    model = LoopedTransformer(config)

    with torch.no_grad():
        out = model(inputs_embeds=inputs_embeds, num_loops=num_loops)
        print("output of forward pass: ", out.shape)

        # paradigm forward
        # out = model.picard_forward(
        #    inputs_embeds, num_loops=num_loops, parallel=100, tolerance=0.1
        # )
    # print("output of paradigms forward pass: ", out[0][0])
    # num_loops = 5000, parallel=100, tolerance=0.1, pass_count=778!

# %%
