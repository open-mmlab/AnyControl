import torch
from torch import nn
from einops import rearrange


class FourierEmbedder(object):
    def __init__(self, num_freqs=64, temperature=100):
        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** (torch.arange(num_freqs) / num_freqs)  

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append(torch.sin(freq*x))
            out.append(torch.cos(freq*x))
        return torch.cat(out, cat_dim)


class FourierColorEmbedder(nn.Module):
    def __init__(self, in_dim=180, out_dim=768, num_tokens=4, fourier_freqs=4, temperature=100, scale=100):
        super().__init__()
        self.in_dim = in_dim  
        self.out_dim = out_dim
        self.fourier_freqs = fourier_freqs
        self.num_tokens = num_tokens

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs, temperature=temperature)
        self.in_dim *= (fourier_freqs * 2)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, out_dim*self.num_tokens),
        )

        self.null_features = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.scale = scale

    def forward(self, x, mask=None):
        if x.ndim == 3:
            assert x.size(1) == 1
            x = x.squeeze(1)
        bs = x.shape[0]
        if mask is None:
            mask = torch.ones(bs, 1, device=x.device)
        x = self.fourier_embedder(x * self.scale) 
        x = mask * x + (1-mask) * self.null_features.view(1,-1)
        x = self.mlp(x).view(bs, self.num_tokens, self.out_dim)  # B*1*C
        return x


class GlobalAdapter(nn.Module):

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, context_tokens=4,
            color_in_dim=180, color_num_tokens=4, color_fourier_freqs=4, color_temperature=100, color_scale=100):

        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.context_tokens = context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        self.color_embed = FourierColorEmbedder(color_in_dim, cross_attention_dim, color_num_tokens, color_fourier_freqs, color_temperature, color_scale)

    def forward(self, x, x_color, *args, **kwargs):
        context_tokens = self.proj(x).reshape(-1, self.context_tokens, self.cross_attention_dim)
        context_tokens = self.norm(context_tokens)
        color_tokens = self.color_embed(x_color)
        return context_tokens, color_tokens
