import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


# --- Utility Modules (Unchanged) ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        xn = self.norm(x)
        a, _ = self.attn(xn, xn, xn)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


# --- New Combined Proxy Block ---
class CombinedProxyBlock(nn.Module):
    """
    A single block that performs cross-attention, self-attention, and MLP.
    The flow is: cross -> self -> mlp, with residual connections and norms.
    """

    def __init__(self, query_dim, kv_dim, num_heads):
        super().__init__()
        # Cross-Attention part
        self.norm_q_cross = RMSNorm(query_dim)
        self.norm_kv_cross = RMSNorm(kv_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, kdim=kv_dim, vdim=kv_dim,
                                                batch_first=True)

        # Self-Attention part
        self.norm_self = RMSNorm(query_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, batch_first=True)

        # Feed-Forward Network part
        self.norm_ffn = RMSNorm(query_dim)
        self.ffn = nn.Sequential(nn.Linear(query_dim, query_dim * 4), nn.SiLU(), nn.Linear(query_dim * 4, query_dim))

    def forward(self, queries, kv):
        # 1. Cross-Attention (queries attend to kv)
        q_norm = self.norm_q_cross(queries)
        kv_norm = self.norm_kv_cross(kv)
        cross_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm)
        queries = queries + cross_out

        # 2. Self-Attention (queries attend to themselves)
        q_norm = self.norm_self(queries)
        self_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + self_out

        # 3. Feed-Forward Network
        q_norm = self.norm_ffn(queries)
        ffn_out = self.ffn(q_norm)
        queries = queries + ffn_out

        return queries


# --- Modified Global Compressor ---
class GlobalNerfCompressor(nn.Module):
    """
    s: [B, L, D]
    returns:
      s_rec: [B, L, D]
      proxies: [B, P, D]
    """

    def __init__(self, hidden_s, num_proxy=32, enc_layers=8, proxy_layers=4, dec_layers=4, num_heads=8):
        super().__init__()
        D = hidden_s
        self.num_proxy = num_proxy

        # optional encoder over s (process patches)
        self.enc_blocks = nn.ModuleList([SelfAttentionBlock(D, num_heads) for _ in range(enc_layers)])

        # learned proxy queries (one set per image)
        self.proxy_queries = nn.Parameter(torch.randn(1, num_proxy, D))

        # --- MODIFIED PART ---
        # A single stack of blocks that combines cross-attention, self-attention, and MLP for proxies.
        self.proxy_blocks = nn.ModuleList(
            [CombinedProxyBlock(query_dim=D, kv_dim=D, num_heads=num_heads) for _ in range(proxy_layers)])
        # --- END MODIFIED PART ---

        self.vq = VectorQuantize(
            dim=D,  # Assuming D is compatible with VQ dim, e.g., 256
            codebook_size=2048,
            decay=0.8,
            commitment_weight=1.
        )

        # learned recon queries per-patch (to decode proxies back to L items)
        self.recon_queries = nn.Parameter(torch.randn(1, 1, D))
        self.recon_expand = nn.Linear(D, D)

        # cross-attention decoder: recon_queries attend to proxies
        # Note: I'm reusing your original CrossAttentionBlock for the decoder as it wasn't part of the request.
        self.dec_blocks = nn.ModuleList([
            CombinedProxyBlock(query_dim=D, kv_dim=D, num_heads=num_heads)
            for _ in range(dec_layers)
        ])

        # small projection heads
        self.proxy_out = nn.Linear(D, D)
        self.recon_out = nn.Linear(D, D)

    def forward(self, s):
        B, L, D = s.shape
        x = s
        for blk in self.enc_blocks:
            x = blk(x)

        # --- MODIFIED PART ---
        # Process proxies using the new combined blocks
        proxies = self.proxy_queries.expand(B, -1, -1).contiguous()
        for blk in self.proxy_blocks:
            proxies = blk(proxies, x)  # Pass both proxies and encoded context 'x'
        # --- END MODIFIED PART ---

        proxies = self.proxy_out(proxies)
        proxies, indices, commit_loss = self.vq(proxies)

        pos = torch.arange(L, device=s.device).unsqueeze(0).unsqueeze(-1).float()
        pos = pos / max(1, L - 1)
        recon = self.recon_queries.expand(B, L, -1).contiguous()
        recon = recon + self.recon_expand(recon) * pos
        q = recon

        for blk in self.dec_blocks:
            q = blk(q, proxies)
        s_rec = self.recon_out(q)

        return s_rec, proxies, commit_loss, indices

    def reconstruct_from_proxies(self, proxies, seq_len):
        B, P, D = proxies.shape
        pos = torch.arange(seq_len, device=proxies.device).unsqueeze(0).unsqueeze(-1).float()
        pos = pos / max(1, seq_len - 1)

        recon = self.recon_queries.expand(B, seq_len, -1).contiguous()
        recon = recon + self.recon_expand(recon) * pos

        q = recon
        for blk in self.dec_blocks:
            q = blk(q, proxies)
        s_rec = self.recon_out(q)

        return s_rec

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import List, Optional

    class LayerNorm(nn.Module):
        """LayerNorm that supports two data formats: channels_last (default) or channels_first."""

        def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
            self.eps = eps
            self.data_format = data_format
            if self.data_format not in ["channels_last", "channels_first"]:
                raise NotImplementedError
            self.normalized_shape = (normalized_shape,)

        def forward(self, x):
            if self.data_format == "channels_last":
                return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            elif self.data_format == "channels_first":
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x

    class ConvNeXtBlock(nn.Module):
        """ConvNeXt Block with depthwise conv, layer norm, and MLP."""

        def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
            super().__init__()
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
            self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
            self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(4 * dim, dim)
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        def forward(self, x):
            input = x
            x = self.dwconv(x)
            x = self.norm(x)
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

            x = input + self.drop_path(x)
            return x

    class DropPath(nn.Module):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

        def __init__(self, drop_prob=None):
            super(DropPath, self).__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()  # binarize
            output = x.div(keep_prob) * random_tensor
            return output

    class ConvNeXtEncoder(nn.Module):
        """ConvNeXt Encoder that progressively downsamples the input."""

        def __init__(self,
                     in_channels=3,
                     dims=[96, 192, 384, 768],
                     depths=[3, 3, 9, 3],
                     drop_path_rate=0.):
            super().__init__()

            self.downsample_layers = nn.ModuleList()
            # Stem layer
            stem = nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
            self.downsample_layers.append(stem)

            # Downsampling layers between stages
            for i in range(3):
                downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            self.stages = nn.ModuleList()
            dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            cur = 0
            for i in range(4):
                stage = nn.Sequential(
                    *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j])
                      for j in range(depths[i])]
                )
                self.stages.append(stage)
                cur += depths[i]

        def forward(self, x):
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)
            return x  # Return only the final encoded features

    class ConvNeXtDecoder(nn.Module):
        """ConvNeXt Decoder that progressively upsamples to reconstruct the input - NO SKIP CONNECTIONS."""

        def __init__(self,
                     out_channels=3,
                     dims=[768, 384, 192, 96],
                     depths=[3, 9, 3, 3],
                     drop_path_rate=0.):
            super().__init__()

            self.upsample_layers = nn.ModuleList()
            # Upsample layers between stages
            for i in range(3):
                upsample_layer = nn.Sequential(
                    nn.ConvTranspose2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                    LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first")
                )
                self.upsample_layers.append(upsample_layer)

            self.stages = nn.ModuleList()
            dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            cur = 0
            for i in range(4):
                # All stages are now simple ConvNeXt blocks without skip connection handling
                stage = nn.Sequential(
                    *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j])
                      for j in range(depths[i])]
                )
                self.stages.append(stage)
                cur += depths[i]

            # Final reconstruction layer
            self.final_upsample = nn.ConvTranspose2d(dims[-1], dims[-1], kernel_size=4, stride=4)
            self.final_conv = nn.Conv2d(dims[-1], out_channels, kernel_size=1)

        def forward(self, x):
            # Start with the encoded features (bottleneck)

            # First stage at bottleneck level
            x = self.stages[0](x)

            # Progressive upsampling stages
            for i in range(1, 4):
                x = self.upsample_layers[i - 1](x)
                x = self.stages[i](x)

            # Final reconstruction
            x = self.final_upsample(x)
            x = self.final_conv(x)
            return x

    class ConvNeXtAutoencoder(nn.Module):
        """Complete ConvNeXt-based Autoencoder WITHOUT skip connections."""

        def __init__(self,
                     in_channels=3,
                     out_channels=None,
                     encoder_dims=[96, 192, 384, 768],
                     encoder_depths=[3, 3, 9, 3],
                     decoder_depths=None,
                     drop_path_rate=0.1):
            super().__init__()

            if out_channels is None:
                out_channels = in_channels

            if decoder_depths is None:
                decoder_depths = encoder_depths[::-1]  # Reverse of encoder depths

            decoder_dims = encoder_dims[::-1]  # Reverse of encoder dims

            self.encoder = ConvNeXtEncoder(
                in_channels=in_channels,
                dims=encoder_dims,
                depths=encoder_depths,
                drop_path_rate=drop_path_rate
            )

            self.decoder = ConvNeXtDecoder(
                out_channels=out_channels,
                dims=decoder_dims,
                depths=decoder_depths,
                drop_path_rate=drop_path_rate
            )

        def forward(self, x):
            encoded_features = self.encoder(x)
            reconstructed = self.decoder(encoded_features)
            return reconstructed

        def encode(self, x):
            """Get encoded representation (bottleneck features)."""
            return self.encoder(x)

        def decode(self, encoded_features):
            """Decode from encoded features."""
            return self.decoder(encoded_features)

    # Example usage and training setup
    def create_convnext_autoencoder(input_size=(3, 224, 224),
                                    variant='tiny'):
        """Create a ConvNeXt autoencoder with different size variants."""

        variants = {
            'tiny': {'dims': [96, 192, 384, 768], 'depths': [3, 3, 9, 3]},
            'small': {'dims': [96, 192, 384, 768], 'depths': [3, 3, 27, 3]},
            'base': {'dims': [128, 256, 512, 1024], 'depths': [3, 3, 27, 3]},
            'large': {'dims': [192, 384, 768, 1536], 'depths': [3, 3, 27, 3]}
        }

        config = variants[variant]

        model = ConvNeXtAutoencoder(
            in_channels=input_size[0],
            encoder_dims=config['dims'],
            encoder_depths=config['depths'],
            drop_path_rate=0.1
        )

        return model

    if __name__ == "__main__":
        # Test the autoencoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        model = create_convnext_autoencoder(variant='tiny').to(device)

        # Test input
        x = torch.randn(4, 3, 224, 224).to(device)  # Batch of 4 images

        print(f"Input shape: {x.shape}")

        # Forward pass
        with torch.no_grad():
            reconstructed = model(x)
            encoded = model.encode(x)

        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Encoded (bottleneck) shape: {encoded.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Compute reconstruction loss
        mse_loss = F.l1_loss(reconstructed, x)
        print(f"Reconstruction L1 Loss: {mse_loss.item():.6f}")


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block with depthwise conv, layer norm, and MLP."""

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt Encoder that progressively downsamples the input."""

    def __init__(self,
                 in_channels=3,
                 dims=[96, 192, 384, 768],
                 depths=[3, 3, 9, 3],
                 drop_path_rate=0.):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        # Stem layer
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # Downsampling layers between stages
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j])
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x  # Return only the final encoded features


class ConvNeXtDecoder(nn.Module):
    """ConvNeXt Decoder that progressively upsamples to reconstruct the input - NO SKIP CONNECTIONS."""

    def __init__(self,
                 out_channels=3,
                 dims=[768, 384, 192, 96],
                 depths=[3, 9, 3, 3],
                 drop_path_rate=0.):
        super().__init__()

        self.upsample_layers = nn.ModuleList()
        # Upsample layers between stages
        for i in range(3):
            upsample_layer = nn.Sequential(
                nn.ConvTranspose2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first")
            )
            self.upsample_layers.append(upsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            # All stages are now simple ConvNeXt blocks without skip connection handling
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j])
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Final reconstruction layer
        self.final_upsample = nn.ConvTranspose2d(dims[-1], dims[-1], kernel_size=4, stride=4)
        self.final_conv = nn.Conv2d(dims[-1], out_channels, kernel_size=1)

    def forward(self, x):
        # Start with the encoded features (bottleneck)

        # First stage at bottleneck level
        x = self.stages[0](x)

        # Progressive upsampling stages
        for i in range(1, 4):
            x = self.upsample_layers[i - 1](x)
            x = self.stages[i](x)

        # Final reconstruction
        x = self.final_upsample(x)
        x = self.final_conv(x)
        return x


class ConvNeXtAutoencoder(nn.Module):
    """Complete ConvNeXt-based Autoencoder WITHOUT skip connections."""

    def __init__(self,
                 in_channels=3,
                 out_channels=None,
                 encoder_dims=[96, 192, 384, 768],
                 encoder_depths=[3, 3, 9, 3],
                 decoder_depths=None,
                 drop_path_rate=0.1):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        if decoder_depths is None:
            decoder_depths = encoder_depths[::-1]  # Reverse of encoder depths

        decoder_dims = encoder_dims[::-1]  # Reverse of encoder dims

        self.encoder = ConvNeXtEncoder(
            in_channels=in_channels,
            dims=encoder_dims,
            depths=encoder_depths,
            drop_path_rate=drop_path_rate
        )

        self.decoder = ConvNeXtDecoder(
            out_channels=out_channels,
            dims=decoder_dims,
            depths=decoder_depths,
            drop_path_rate=drop_path_rate
        )

    def forward(self, x):
        encoded_features = self.encoder(x)
        reconstructed = self.decoder(encoded_features)
        return reconstructed

    def encode(self, x):
        """Get encoded representation (bottleneck features)."""
        return self.encoder(x)

    def decode(self, encoded_features):
        """Decode from encoded features."""
        return self.decoder(encoded_features)


# Example usage and training setup
def create_convnext_autoencoder(input_size=(3, 224, 224),
                                variant='tiny'):
    """Create a ConvNeXt autoencoder with different size variants."""

    variants = {
        'tiny': {'dims': [96, 192, 384, 768], 'depths': [3, 3, 9, 3]},
        'small': {'dims': [96, 192, 384, 768], 'depths': [3, 3, 27, 3]},
        'base': {'dims': [128, 256, 512, 1024], 'depths': [3, 3, 27, 3]},
        'large': {'dims': [192, 384, 768, 1536], 'depths': [3, 3, 27, 3]}
    }

    config = variants[variant]

    model = ConvNeXtAutoencoder(
        in_channels=input_size[0],
        encoder_dims=config['dims'],
        encoder_depths=config['depths'],
        drop_path_rate=0.1
    )

    return model


class ConvNeXtWithGlobalCompressor(nn.Module):
    def __init__(self, ae_model: ConvNeXtAutoencoder, compressor: GlobalNerfCompressor):
        super().__init__()
        self.ae_model = ae_model
        self.compressor = compressor
        self.proj_in = nn.Linear(768, 256)
        self.proj_out = nn.Linear(256, 768)

    def forward(self, x):
        # Step 1: Encode using ConvNeXt
        encoder_features = self.ae_model.encoder(x)
        bottleneck = encoder_features  # [B, C, H, W]
        B, C, H, W = bottleneck.shape

        # Step 2: Flatten spatial dimensions for GlobalNerfCompressor
        seq = bottleneck.view(B, C, H * W).permute(0, 2, 1).contiguous()  # [B, L, D] with L=H*W, D=C
        seq = self.proj_in(seq)
        # Step 3: Compress with GlobalNerfCompressor
        s_rec, proxies, commit_loss, indices = self.compressor(seq)
        s_rec = self.proj_out(s_rec)
        # Step 4: Optionally reshape reconstructed sequence back to feature map
        s_rec_map = s_rec.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # Step 5: Decode with ConvNeXt decoder (skip connections can still be used)
        encoder_features = s_rec_map  # replace bottleneck with compressed version
        reconstructed = self.ae_model.decoder(encoder_features)
        #        return s_rec, proxies, commit_loss, indices

        return reconstructed, proxies, commit_loss, indices

    def reconstruct_using_proxy(self, x, sizes):
        B, C, H, W = sizes
        s_rec = self.compressor.reconstruct_from_proxies(x, (H // 32) * (W // 32))
        s_rec = self.proj_out(s_rec)
        s_rec_map = s_rec.permute(0, 2, 1).contiguous().view(B, 768, H // 32, W // 32)
        reconstructed = self.ae_model.decoder(s_rec_map)
        return reconstructed