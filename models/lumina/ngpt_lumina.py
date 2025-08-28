import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from torch.nn import RMSNorm

class ReparamLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.eps = eps

    def forward(self, x):
        # Normalize each row of weights
        w = self.weight / (self.weight.norm(dim=1, keepdim=True) + self.eps)
        return F.linear(x, w, self.bias)

def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def normalize_linear_columns_(W: torch.Tensor, eps: float = 1e-12):
    # W: [out, in] ; normalize columns (in-dim)
    W.div_(W.norm(dim=0, keepdim=True).clamp_min_(eps))

def normalize_module_params_(m: nn.Module):
    for mod in m.modules():
        if isinstance(mod, ReparamLinear):
            normalize_linear_columns_(mod.weight)

def precompute_freqs_cis(
    dim: List[int],
    end: List[int],
    theta: float = 10000.0,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with
    given dimensions.

    This function calculates a frequency tensor with complex exponentials
    using the given dimension 'dim' and the end index 'end'. The 'theta'
    parameter scales the frequencies. The returned tensor contains complex
    values in complex64 data type.

    Args:
        dim (list): Dimension of the frequency tensor.
        end (list): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation.
            Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex
            exponentials.
    """
    freqs_cis = []
    for i, (d, e) in enumerate(zip(dim, end)):
        freqs = 1.0 / (
            theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d)
        )
        timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
        freqs = torch.outer(timestep, freqs).float()
        freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(
            torch.complex64
        )  # complex64
        freqs_cis.append(freqs_cis_i)

    return freqs_cis


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def apply_rotary_emb(
    x_in: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency
    tensor.

    This function applies rotary embeddings to the given query 'xq' and
    key 'xk' tensors using the provided frequency tensor 'freqs_cis'. The
    input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors
    contain rotary embeddings and are returned as real tensors.

    Args:
        x_in (torch.Tensor): Query or Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex
            exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor
            and key tensor with rotary embeddings.
    """
    if freqs_cis is None:
        return x_in
    with torch.autocast(enabled=False, device_type="cuda"):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            ReparamLinear(
                frequency_embedding_size,
                hidden_size,
                bias=True,
            ),
            nn.SiLU(),
            ReparamLinear(
                hidden_size,
                hidden_size,
                bias=True,
            ),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(self.mlp[0].weight.dtype))
        return t_emb

class JointAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: Optional[int], qk_norm: bool):
        super().__init__()
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_local_heads = n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = dim // n_heads

        self.qkv = ReparamLinear(dim, (n_heads + self.n_kv_heads + self.n_kv_heads) * self.head_dim, bias=False)
        nn.init.xavier_uniform_(self.qkv.weight)

        self.out = ReparamLinear(n_heads * self.head_dim, dim, bias=False)
        nn.init.xavier_uniform_(self.out.weight)

        # learnable per-dim rescalers for q and k
        self.s_q = nn.Parameter(torch.ones(self.head_dim))
        self.s_k = nn.Parameter(torch.ones(self.head_dim))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        dtype = x.dtype

        xq, xk, xv = torch.split(
            self.qkv(x),
            [self.n_local_heads * self.head_dim,
             self.n_local_kv_heads * self.head_dim,
             self.n_local_kv_heads * self.head_dim],
            dim=-1,
        )
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE
        xq = apply_rotary_emb(xq, freqs_cis=freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis=freqs_cis)

        # nGPT: per-vector L2 norm + learned rescale
        xq = l2_normalize(xq) * self.s_q
        xk = l2_normalize(xk) * self.s_k
        xq, xk = xq.to(dtype), xk.to(dtype)

        # nGPT: use sqrt(d) scale
        softmax_scale = math.sqrt(self.head_dim)

        n_rep = self.n_local_heads // self.n_local_kv_heads
        if n_rep >= 1:
            xk = xk.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            xv = xv.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        output = F.scaled_dot_product_attention(
            xq.permute(0, 2, 1, 3),
            xk.permute(0, 2, 1, 3),
            xv.permute(0, 2, 1, 3),
            attn_mask=x_mask.bool().view(bsz, 1, 1, seqlen).expand(-1, self.n_local_heads, seqlen, -1),
            scale=softmax_scale,
        ).permute(0, 2, 1, 3).to(dtype)

        return self.out(output.flatten(-2))



class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ReparamLinear(dim, hidden_dim, bias=False)
        self.w2 = ReparamLinear(hidden_dim, dim, bias=False)
        self.w3 = ReparamLinear(dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w2.weight)
        nn.init.xavier_uniform_(self.w3.weight)

        # nGPT scalers
        self.s_u = nn.Parameter(torch.ones(hidden_dim))
        self.s_v = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        u = self.w1(x)
        u = u * self.s_u
        h = F.relu6(u) * self.w3(x)
        h = h * self.s_v
        return self.w2(h)

class JointTransformerBlock(nn.Module):
    def __init__(self, layer_id, dim, n_heads, n_kv_heads, multiple_of, ffn_dim_multiplier, norm_eps, qk_norm, modulation=True):
        super().__init__()
        self.dim = dim
        self.attention = JointAttention(dim, n_heads, n_kv_heads, qk_norm)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=4 * dim, multiple_of=multiple_of, ffn_dim_multiplier=ffn_dim_multiplier)
        self.layer_id = layer_id
        self.modulation = modulation

        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                ReparamLinear(min(dim, 1024), 4 * dim, bias=True),
            )
            nn.init.zeros_(self.adaLN_modulation[1].weight)
            nn.init.zeros_(self.adaLN_modulation[1].bias)

        # nGPT: per-dim eigen learning-rate (alphas)
        self.alpha_attn = nn.Parameter(torch.full((dim,), 0.05))
        self.alpha_mlp  = nn.Parameter(torch.full((dim,), 0.05))

        self.use_compiled = False
        self.modulate = torch.compile(modulate) if self.use_compiled else modulate

    def forward(self, x, x_mask, freqs_cis, adaln_input: Optional[torch.Tensor] = None):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)

            # Attention step on the sphere
            h = x
            hA = self.attention(self.modulate(h, scale_msa), x_mask, freqs_cis)
            x = l2_normalize(h + self.alpha_attn.unsqueeze(0).unsqueeze(0) * (hA - h))
            x = x + gate_msa.unsqueeze(1).tanh() * 0  # keep gate param in graph

            # MLP step on the sphere
            h = x
            hM = self.feed_forward(self.modulate(h, scale_mlp))
            x = l2_normalize(h + self.alpha_mlp.unsqueeze(0).unsqueeze(0) * (hM - h))
            x = x + gate_mlp.unsqueeze(1).tanh() * 0
        else:
            # Same but without AdaLN modulate
            h = x
            hA = self.attention(h, x_mask, freqs_cis)
            x = l2_normalize(h + self.alpha_attn.unsqueeze(0).unsqueeze(0) * (hA - h))

            h = x
            hM = self.feed_forward(h)
            x = l2_normalize(h + self.alpha_mlp.unsqueeze(0).unsqueeze(0) * (hM - h))
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.linear = ReparamLinear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ReparamLinear(min(hidden_size, 1024), hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        self.use_compiled = False
        self.modulate = torch.compile(modulate) if self.use_compiled else modulate

    def forward(self, x, c):
        scale = self.adaLN_modulation(c)
        x = self.modulate(x, scale)
        return self.linear(x)


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 10000.0,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (1, 512, 512),
    ):
        super().__init__()
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cis = precompute_freqs_cis(
            self.axes_dims, self.axes_lens, theta=self.theta
        )

    def __call__(self, ids: torch.Tensor):
        self.freqs_cis = [freqs_cis.to(ids.device) for freqs_cis in self.freqs_cis]
        result = []
        for i in range(len(self.axes_dims)):
            index = (
                ids[:, :, i : i + 1]
                .repeat(1, 1, self.freqs_cis[i].shape[-1])
                .to(torch.int64)
            )
            result.append(
                torch.gather(
                    self.freqs_cis[i].unsqueeze(0).repeat(index.shape[0], 1, 1),
                    dim=1,
                    index=index,
                )
            )
        return torch.cat(result, dim=-1)


class Lumina(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        dim: int = 4096,
        n_layers: int = 32,
        n_refiner_layers: int = 2,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        qk_norm: bool = False,
        cap_feat_dim: int = 5120,
        axes_dims: List[int] = (16, 56, 56),
        axes_lens: List[int] = (1, 512, 512),
        use_fast = False
    ) -> None:
        super().__init__()
        if use_fast:
            torch.backends.cuda.cudnn_sdp_enabled = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            torch.set_float32_matmul_precision("medium")
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size

        self.x_embedder = ReparamLinear(
            in_features=patch_size * patch_size * in_channels,
            out_features=dim,
            bias=True,
        )
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0.0)

        self.noise_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(min(dim, 1024))
        self.t_to_token = ReparamLinear(min(dim, 1024), dim, bias=True)
        nn.init.xavier_uniform_(self.t_to_token.weight);
        nn.init.zeros_(self.t_to_token.bias)

        self.cap_embedder = nn.Sequential(
            ReparamLinear(cap_feat_dim, dim, bias=True),
        )
        nn.init.trunc_normal_(self.cap_embedder[0].weight, std=0.02)
        nn.init.zeros_(self.cap_embedder[0].bias)
        nn.init.zeros_(self.cap_embedder[0].bias)

        self.layers = nn.ModuleList(
            [
                JointTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    multiple_of,
                    ffn_dim_multiplier,
                    norm_eps,
                    qk_norm,
                )
                for layer_id in range(n_layers)
            ]
        )
        self.final_layer = FinalLayer(dim, patch_size, self.out_channels)

        assert (dim // n_heads) == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.rope_embedder = RopeEmbedder(axes_dims=axes_dims, axes_lens=axes_lens)
        self.dim = dim
        self.n_heads = n_heads
    def patchify_and_embed(self, x: List[torch.Tensor] | torch.Tensor, cap_feats: torch.Tensor, cap_mask: torch.Tensor, t_vec: torch.Tensor):
        bsz = len(x)
        pH = pW = self.patch_size
        device = x[0].device

        # prepend 1 time token to context
        time_token = self.t_to_token(t_vec).unsqueeze(1)         # [B,1,dim]
        cap_feats = self.cap_embedder(cap_feats)                  # [B,Lc,dim]
        cap_feats = torch.cat([time_token, cap_feats], dim=1)     # [B,1+Lc,dim]
        cap_mask = torch.cat([torch.ones(bsz, 1, dtype=torch.bool, device=device), cap_mask], dim=1)

        l_effective_cap_len = cap_mask.sum(dim=1).tolist()
        img_sizes = [(img.size(1), img.size(2)) for img in x]
        l_effective_img_len = [(H // pH) * (W // pW) for (H, W) in img_sizes]

        max_seq_len = max((cap_len + img_len for cap_len, img_len in zip(l_effective_cap_len, l_effective_img_len)))
        max_cap_len = max(l_effective_cap_len)
        max_img_len = max(l_effective_img_len)

        position_ids = torch.zeros(bsz, max_seq_len, 3, dtype=torch.int32, device=device)

        for i in range(bsz):
            cap_len = l_effective_cap_len[i]                         # includes time token
            img_len = l_effective_img_len[i]
            H, W = img_sizes[i]
            H_tokens, W_tokens = H // pH, W // pW

            position_ids[i, :cap_len, 0] = torch.arange(cap_len, dtype=torch.int32, device=device)  # 0..cap_len-1
            position_ids[i, cap_len: cap_len + img_len, 0] = cap_len

            row_ids = torch.arange(H_tokens, dtype=torch.int32, device=device).view(-1, 1).repeat(1, W_tokens).flatten()
            col_ids = torch.arange(W_tokens, dtype=torch.int32, device=device).view(1, -1).repeat(H_tokens, 1).flatten()
            position_ids[i, cap_len: cap_len + img_len, 1] = row_ids
            position_ids[i, cap_len: cap_len + img_len, 2] = col_ids

        freqs_cis = self.rope_embedder(position_ids)

        # build cap/img freqs
        cap_freqs_cis_shape = list(freqs_cis.shape); cap_freqs_cis_shape[1] = max_cap_len
        cap_freqs_cis = torch.zeros(*cap_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)
        img_freqs_cis_shape = list(freqs_cis.shape); img_freqs_cis_shape[1] = max_img_len
        img_freqs_cis = torch.zeros(*img_freqs_cis_shape, device=device, dtype=freqs_cis.dtype)

        for i in range(bsz):
            cap_len = l_effective_cap_len[i]
            img_len = l_effective_img_len[i]
            cap_freqs_cis[i, :cap_len] = freqs_cis[i, :cap_len]
            img_freqs_cis[i, :img_len] = freqs_cis[i, cap_len: cap_len + img_len]

        # refine context
        for layer in self.context_refiner:
            if self.training:
                cap_feats = ckpt.checkpoint(layer, cap_feats, cap_mask, cap_freqs_cis)
            else:
                cap_feats = layer(cap_feats, cap_mask, cap_freqs_cis)

        # patchify images -> embed -> refine noise
        flat_x = []
        for i in range(bsz):
            img = x[i]; C, H, W = img.size()
            img = (img.view(C, H // pH, pH, W // pW, pW).permute(1, 3, 2, 4, 0).flatten(2).flatten(0, 1))
            flat_x.append(img)
        x = flat_x
        padded_img_embed = torch.zeros(bsz, max_img_len, x[0].shape[-1], device=device, dtype=x[0].dtype)
        padded_img_mask = torch.zeros(bsz, max_img_len, dtype=torch.bool, device=device)
        for i in range(bsz):
            padded_img_embed[i, : l_effective_img_len[i]] = x[i]
            padded_img_mask[i, : l_effective_img_len[i]] = True

        padded_img_embed = self.x_embedder(padded_img_embed)
        for layer in self.noise_refiner:
            if self.training:
                padded_img_embed = ckpt.checkpoint(layer, padded_img_embed, padded_img_mask, img_freqs_cis, t_vec)
            else:
                padded_img_embed = layer(padded_img_embed, padded_img_mask, img_freqs_cis, t_vec)

        # concat context + image
        mask = torch.zeros(bsz, max_seq_len, dtype=torch.bool, device=device)
        padded_full_embed = torch.zeros(bsz, max_seq_len, self.dim, device=device, dtype=padded_img_embed.dtype)
        for i in range(bsz):
            cap_len = l_effective_cap_len[i]; img_len = l_effective_img_len[i]
            mask[i, : cap_len + img_len] = True
            padded_full_embed[i, :cap_len] = cap_feats[i, :cap_len]
            padded_full_embed[i, cap_len: cap_len + img_len] = padded_img_embed[i, :img_len]
        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis

    def unpatchify(
        self,
        x: torch.Tensor,
        img_size: List[Tuple[int, int]],
        cap_size: List[int],
        return_tensor=False,
    ) -> List[torch.Tensor]:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        pH = pW = self.patch_size
        imgs = []
        for i in range(x.size(0)):
            H, W = img_size[i]
            begin = cap_size[i]
            end = begin + (H // pH) * (W // pW)
            imgs.append(
                x[i][begin:end]
                .view(H // pH, W // pW, pH, pW, self.out_channels)
                .permute(4, 0, 2, 1, 3)
                .flatten(3, 4)
                .flatten(1, 2)
            )

        if return_tensor:
            imgs = torch.stack(imgs, dim=0)
        return imgs

    @property
    def device(self):
        # Get the device of the module (assumes all parameters are on the same device)
        return next(self.parameters()).device

    def set_use_compiled(self):
        for name, module in self.named_modules():
            # Check if the module has the 'use_compiled' attribute
            if hasattr(module, "use_compiled"):
                print(f"Setting 'use_compiled' to True in module: {name}")
                setattr(module, "use_compiled", True)

    def forward(self, x, t, cap_feats, cap_mask):
        t_vec = self.t_embedder(t)                     # [B, min(dim,1024)]
        adaln_input = t_vec
        x_is_tensor = isinstance(x, torch.Tensor)
        x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(x, cap_feats, cap_mask, t_vec)
        freqs_cis = freqs_cis.to(x.device)

        n_layers = len(self.layers)
        dropped_info = None

        for i, layer in enumerate(self.layers):
            use_rope = (i % 2 == 0)
            if self.training and i == 1 and n_layers >= 4:
                B, N, _ = x.shape
                n_tokens_to_keep = N // 2
                perm = torch.stack([torch.randperm(N, device=x.device) for _ in range(B)])
                keep_indices = perm[:, :n_tokens_to_keep].sort(dim=1).values
                drop_indices = perm[:, n_tokens_to_keep:].sort(dim=1).values

                dropped_x = torch.gather(x, 1, drop_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                dropped_mask = torch.gather(mask, 1, drop_indices)
                dropped_freqs_cis = torch.gather(freqs_cis, 1, drop_indices.unsqueeze(-1).expand(-1, -1, freqs_cis.shape[-1]))
                dropped_info = (dropped_x, dropped_mask, dropped_freqs_cis, keep_indices, drop_indices)

                x = torch.gather(x, 1, keep_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
                mask = torch.gather(mask, 1, keep_indices)
                freqs_cis = torch.gather(freqs_cis, 1, keep_indices.unsqueeze(-1).expand(-1, -1, freqs_cis.shape[-1]))

            if self.training and i == n_layers - 2 and dropped_info is not None:
                dropped_x, dropped_mask, dropped_freqs_cis, keep_idx, drop_idx = dropped_info
                N_orig = keep_idx.shape[1] + drop_idx.shape[1]

                full_x = torch.empty((x.size(0), N_orig, x.size(-1)), dtype=x.dtype, device=x.device)
                full_mask = torch.empty((mask.size(0), N_orig), dtype=mask.dtype, device=mask.device)
                full_freqs = torch.empty((freqs_cis.size(0), N_orig, freqs_cis.size(-1)),
                                         dtype=freqs_cis.dtype, device=freqs_cis.device)
                full_x[:, keep_idx[0]] = x
                full_x[:, drop_idx[0]] = dropped_x
                full_mask[:, keep_idx[0]] = mask
                full_mask[:, drop_idx[0]] = dropped_mask
                full_freqs[:, keep_idx[0]] = freqs_cis
                full_freqs[:, drop_idx[0]] = dropped_freqs_cis

                x, mask, freqs_cis = full_x, full_mask, full_freqs
                dropped_info = None

            freqs = freqs_cis if use_rope else None
            x = ckpt.checkpoint(layer, x, mask, freqs, adaln_input) if self.training else layer(x, mask, freqs, adaln_input)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, img_size, cap_size, return_tensor=x_is_tensor)
        return x

    # convenience: call after optimizer.step()
    def normalize_params_(self):
        normalize_module_params_(self)


    def forward_with_cfg(
            self,
            x,
            t,
            cap_feats,
            cap_mask,
            cfg_scale,
            cfg_trunc=100,
            renorm_cfg=1
    ):

        """
        Forward pass of NextDiT, but also batches the unconditional forward pass
        for classifier-free guidance.
        """
        # # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        if t[0] < cfg_trunc:
            combined = torch.cat([half, half], dim=0) # [2, 16, 128, 128]
            model_out = self.forward(combined, t, cap_feats, cap_mask) # [2, 16, 128, 128]
            # For exact reproducibility reasons, we apply classifier-free guidance on only
            # three channels by default. The standard approach to cfg applies it to all channels.
            # This can be done by uncommenting the following line and commenting-out the line following that.
            eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            if float(renorm_cfg) > 0.0:
                ori_pos_norm = torch.linalg.vector_norm(cond_eps
                        , dim=tuple(range(1, len(cond_eps.shape))), keepdim=True
                )
                max_new_norm = ori_pos_norm * float(renorm_cfg)
                new_pos_norm = torch.linalg.vector_norm(
                        half_eps, dim=tuple(range(1, len(half_eps.shape))), keepdim=True
                    )
                if new_pos_norm >= max_new_norm:
                    half_eps = half_eps * (max_new_norm / new_pos_norm)
        else:
            combined = half
            model_out = self.forward(combined, t[:len(x) // 2], cap_feats[:len(x) // 2], cap_mask[:len(x) // 2])
            eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
            half_eps = eps

        output = torch.cat([half_eps, half_eps], dim=0)
        return output
    def parameter_count(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def Lumina_2b(**kwargs):
    return Lumina(
        patch_size=2,
        in_channels=16,
        dim=768,
        n_layers=12,
        n_heads=8,
        n_kv_heads=4,
        axes_dims=[32, 32, 32],
        axes_lens=[300, 512, 512],
        qk_norm=True,
        cap_feat_dim=640,
        **kwargs,
    )
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model and move to device
    model = Lumina_2b(n_refiner_layers=2).to(device)
    model.train()
    print(f"Model parameters: {count_parameters(model)/1e6:.2f}M")

    # Dummy input
    batch_size = 2
    C, H, W = 16, 64, 64  # in_channels=16 and H, W should be divisible by patch_size=2
    cap_len = 40
    cap_feat_dim = 640

    # Create dummy inputs
    x = torch.randn(batch_size, C, H, W, device=device, requires_grad=True)
    t = torch.randint(0, 1000, (batch_size,), device=device)  # scalar timesteps
    cap_feats = torch.randn(batch_size, cap_len, cap_feat_dim, device=device)
    cap_mask = torch.ones(batch_size, cap_len, dtype=torch.bool, device=device)

    # Forward pass
    output = model(x, t, cap_feats, cap_mask)
    print(f"Output shape: {output.shape}")  # Should be (B, C, H, W)

    # Dummy loss
    target = torch.randn_like(output)
    loss = F.mse_loss(output, target)
    print(f"Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()
    print("Backward pass successful.")