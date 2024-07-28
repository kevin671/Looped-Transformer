# %%
import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # embeddings = time[:, None] * embeddings[None, :]
        embeddings = time * embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def lora_transform(tensor, n_embd, r, scaling):
    lora_A, lora_B = tensor.chunk(2, dim=-1)
    lora_A = lora_A.view(n_embd, r)
    lora_B = lora_B.view(r, n_embd)
    return lora_A @ lora_B * scaling


class HyperNetworkMLPLoRA(nn.Module):  # for attn
    def __init__(self, t_dim, n_embd, r=1, bias=False):
        super().__init__()
        out_n_embd = n_embd * 2 * 2
        self.n_embd = n_embd  # 768
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        self.weight_mlp = nn.Sequential(
            nn.Linear(t_dim, r * out_n_embd * 4),
            nn.SiLU(),
            nn.Linear(r * out_n_embd * 4, r * out_n_embd),
        )

        self.r = r
        self.lora_alpha = 1.0
        self.scaling = self.lora_alpha / self.r

    def forward(self, time):
        t = self.time_mlp(time)
        fc, proj = self.weight_mlp(t).chunk(2, dim=-1)
        fc = lora_transform(fc, self.n_embd, self.r, self.scaling)
        proj = lora_transform(proj, self.n_embd, self.r, self.scaling)
        return fc, proj


class HyperNetworkAttnLoRA(nn.Module):  # for attn
    def __init__(self, t_dim, n_embd, r=1, bias=False):
        super().__init__()
        out_n_embd = n_embd * 2 * 4  # for attn q,k,v, o
        self.n_embd = n_embd  # 768
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        self.weight_mlp = nn.Sequential(
            nn.Linear(t_dim, r * out_n_embd * 4),
            nn.SiLU(),
            nn.Linear(r * out_n_embd * 4, r * out_n_embd),
        )

        self.r = r
        self.lora_alpha = 1.0
        self.scaling = self.lora_alpha / self.r

    def forward(self, time):
        t = self.time_mlp(time)
        weight = self.weight_mlp(t)
        q, k, v, o = weight.chunk(
            4, dim=-1
        )  # [r * n_embd * 2], [r * n_embd * 2], [r * n_embd * 2], [r * n_embd * 2]
        q_transformed = lora_transform(q, self.n_embd, self.r, self.scaling)
        k_transformed = lora_transform(k, self.n_embd, self.r, self.scaling)
        v_transformed = lora_transform(v, self.n_embd, self.r, self.scaling)
        o_transformed = lora_transform(o, self.n_embd, self.r, self.scaling)

        return (
            torch.cat([q_transformed, k_transformed, v_transformed], dim=-1),
            o_transformed,
        )


# %%
# time = torch.tensor(1.0)
# model = HyperNetworkMLPLoRA(128, 768, r=2)
# out, proj = model(time)
# print(out, proj)
# print(out.shape, proj.shape)

# model = HyperNetworkAttnLoRA(512, 768, r=2)
# out, o = model(time)
# print(out.shape, o.shape)


# %%
class HyperNetwork(nn.Module):
    def __init__(self, dim, n_embd, bias=False):
        super().__init__()
        self.n_embd = n_embd  # 768
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        # self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        self.attn_w_end = n_embd * 3 * n_embd
        self.attn_b_start = self.attn_w_end
        self.attn_b_end = self.attn_b_start + n_embd
        self.attn_proj_w_start = self.attn_b_end
        self.attn_proj_w_end = self.attn_proj_w_start + n_embd * n_embd
        self.attn_proj_b_start = self.attn_proj_w_end
        self.attn_proj_b_end = self.attn_proj_b_start + n_embd
        self.fc_w_start = self.attn_proj_b_end
        self.fc_w_end = self.fc_w_start + 4 * n_embd * n_embd
        self.fc_b_start = self.fc_w_end
        self.fc_b_end = self.fc_b_start + 4 * n_embd
        self.proj_w_start = self.fc_b_end
        self.proj_w_end = self.proj_w_start + n_embd * n_embd
        self.proj_b_start = self.proj_w_end
        self.proj_b_end = self.proj_b_start + n_embd

        out_dim = (
            n_embd * 3 * n_embd
            + n_embd * n_embd
            + n_embd * 4 * n_embd
            + 4 * n_embd * n_embd
        )
        # if bias:
        out_dim += 3 * n_embd + n_embd + 4 * n_embd + n_embd

        n_deconv = math.ceil((math.log2(out_dim / time_dim)))
        self.hypernet = nn.ModuleList(
            [
                nn.Sequential(
                    nn.SiLU(),
                    # nn.BatchNorm1d(time_dim * (2**i)),
                    nn.ConvTranspose1d(1, 1, 3, padding=0, stride=2),
                )
                for i in range(n_deconv)
            ]
        )

    def forward(self, time):
        # output n_embd * 3 * .n_embd + n_embd * 4 * n_embd * 2
        t = self.time_mlp(time)  # [time_dim]
        weights = t.unsqueeze(0).unsqueeze(0)  # [1, 1, time_dim]
        for layer in self.hypernet:
            weights = layer(weights)
        weights = weights.squeeze(0).squeeze(0)  # [out_dim]

        return {
            "attn_w": weights[: self.attn_w_end].reshape(self.n_embd, 3 * self.n_embd),
            "attn_b": weights[self.attn_b_start : self.attn_b_end],
            "attn_proj_w": weights[
                self.attn_proj_w_start : self.attn_proj_w_end
            ].reshape(self.n_embd, self.n_embd),
            "attn_proj_b": weights[self.attn_proj_b_start : self.attn_proj_b_end],
            "fc_w": weights[self.fc_w_start : self.fc_w_end].reshape(
                4 * self.n_embd, self.n_embd
            ),
            "fc_b": weights[self.fc_b_start : self.fc_b_end],
            "proj_w": weights[self.proj_w_start : self.proj_w_end].reshape(
                self.n_embd, self.n_embd
            ),
            "proj_b": weights[self.proj_b_start : self.proj_b_end],
        }


# %%
# time = torch.tensor(1.0)

# embeddings = SinusoidalPositionEmbeddings(512)
# print(embeddings(time).shape)

# hypernet = HyperNetwork(512, 768)
# out = hypernet(time)

# for k, v in out.items():
#    print(k, v.shape)

# %%
