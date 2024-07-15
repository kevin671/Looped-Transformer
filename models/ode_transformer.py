import torch
import torch.nn as nn

from .hypernetworks import LoRAHyperNetwork
from dataclasses import dataclass
from .transformer_block import CausalSelfAttention, LayerNorm, MLP, PreNormLayer


# Lie-Trotter splitting scheme with the Euler’s method
class LieTrotterSplittingScheme(nn.Module):
    def __init__(self, transformer):
        super(LieTrotterSplittingScheme, self).__init__()
        self.transformer = transformer

    def forward(self, x: torch.Tensor):
        x = self.transformer.attn(x)
        x = self.transformer.mlp(x)
        return x


class KolmogorovArnoldNetSheme(nn.Module):
    def __init__(self, transformer):
        super(KolmogorovArnoldNetSheme, self).__init__()
        self.transformer = transformer

    def forward(self, x: torch.Tensor):
        x = self.transformer.mlp(x)
        x = self.transformer.attn(x)
        return x


class TrotterAdditiveSplittingScheme(nn.Module):
    def __init__(self, transformer):
        super(LieTrotterSplittingScheme, self).__init__()
        self.transformer = transformer

    def forward(self, x: torch.Tensor):
        x1 = self.transformer.attn(x)
        x2 = self.transformer.mlp(x)
        return (x1 + x2) / 2


# Strang-Marchuk splitting scheme
class StrangMarchukSplittingScheme(nn.Module):
    def __init__(self, transformer):
        super(StrangMarchukSplittingScheme, self).__init__()
        self.transformer = transformer

    def forward(self, x: torch.Tensor):
        x = 0.5 * self.transformer.attn(x)
        x = self.transformer.mlp(x)
        x = 0.5 * self.transformer.attn(x)
        return x


class StrangAdditiveSplittingScheme(nn.Module):
    def __init__(self, transformer):
        super(StrangMarchukSplittingScheme, self).__init__()
        self.transformer = transformer

    def forward(self, x: torch.Tensor):
        x1 = self.transformer.attn(x)
        x1 = self.transformer.mlp(x1)

        x2 = self.transformer.mlp(x)
        x2 = self.transformer.attn(x2)
        return (x1 + x2) / 2


@dataclass
class ODETransformerConfig:
    block_size: int = 128
    vocab_size: int = 1024
    n_layer: int = 5
    n_head: int = 4
    n_embd: int = 64
    # n_reg_tokens: int = 4
    bias: bool = True
    dropout: float = 0.0
    lora_rank: int = 2
    temb_dim: int = 64
    solver_scheme: str = "Lie-Trotter"  # "Strang-Marchuk"


# ODE transformer
class ODETransformer(nn.Module):
    def __init__(self, config: ODETransformerConfig):
        super(ODETransformer, self).__init__()

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                mlp=PreNormLayer(MLP(config), config.n_embd, config.bias),
                attn=PreNormLayer(
                    CausalSelfAttention(config),
                    config.n_embd,
                    config.bias,
                    # LoRACausalSelfAttention(config), config.n_embd, config.bias
                ),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        if config.lora_rank > 0:
            self.lora_hypernetwork = LoRAHyperNetwork(
                config.n_embd, config.lora_rank, config.temb_dim
            )

        self.solver = (
            LieTrotterSplittingScheme(self.transformer)
            if config.solver_scheme == "Lie-Trotter"
            else StrangMarchukSplittingScheme(self.transformer)
        )

    # TODO(fix)
    def forward(self, x: torch.Tensor, temb: torch.Tensor):
        # lora_a, lora_b = self.lora_hypernetwork(temb)
        return self.solver(x)

    # TODO(fix)
    # https://github.com/AndyShih12/paradigms/blob/master/paradigms/stablediffusion_paradigms.py
    @torch.no_grad()
    def paradigms_forward(
        self,
        x,
        num_loops: int = 1000,
        parallel: int = 100,
        tolerance: float = 0.1,
        position_ids=None,
        rm_pos_embd=False,
        add_inputs_embeds=False,
        output_intermediate=False,
    ):
        assert not add_inputs_embeds, "add_inputs_embeds is not supported"

        device = inputs_embeds.device
        batch_size, seq_len, n_embd = x.shape

        assert batch_size == 1, "batch size must be 1"

        stats_pass_count = 0
        stats_flop_count = 0

        timesteps = torch.arange(
            0, num_loops * len(self.transformer.looped_block.layers), device=device
        )
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
                model_output[i] = self.transformer.looped_block(block_latents[i], t)

            delta = model_output.reshape(parallel_len, 1, seq_len, n_embd)
            cumulative_delta = torch.cumsum(delta, dim=0)
            # print(f"{cumulative_delta.shape=}", flush=True)

            block_latents_new = (
                latents_time_evolution_buffer[begin_idx][None,] + cumulative_delta
            )  # (parallel_len, seq_len, n_embd)
            cur_error_vec = (
                block_latents_new - block_latents
            )  # (parallel_len, 1, seq_len, n_embd)
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
