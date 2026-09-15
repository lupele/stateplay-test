"""Annotation-free state rollout and clean-video transition supervision."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _mlp(in_dim, hidden_dim, out_dim, dropout=0.0):
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.key_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = _mlp(dim, int(dim * mlp_ratio), dim, dropout)

    def forward(self, queries, source):
        update, _ = self.attention(self.query_norm(queries), self.key_norm(source),
                                   source, need_weights=False)
        queries = queries + update
        return queries + self.mlp(self.norm(queries))


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _mlp(dim, int(dim * mlp_ratio), dim, dropout)

    def forward(self, x):
        normalized = self.norm1(x)
        update, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + update
        return x + self.mlp(self.norm2(x))


class LatentStateInitializer(nn.Module):
    def __init__(self, visual_hidden_dim=3072, num_tokens=4, dim=256,
                 num_layers=2, num_heads=8, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.input_projection = nn.Linear(visual_hidden_dim, dim)
        self.queries = nn.Parameter(torch.randn(num_tokens, dim) / math.sqrt(dim))
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, first_frame_tokens):
        source = self.input_projection(first_frame_tokens)
        state = self.queries.unsqueeze(0).expand(source.shape[0], -1, -1)
        for block in self.blocks:
            state = block(state, source)
        return self.norm(state)


class LatentStateTransition(nn.Module):
    def __init__(self, dim=256, action_dim=11, num_layers=2, num_heads=8,
                 mlp_ratio=4.0, dropout=0.0, max_frames=64):
        super().__init__()
        self.action_embedding = nn.Linear(action_dim, dim)
        self.step_embedding = nn.Embedding(max_frames, dim)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.delta = nn.Linear(dim, dim)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.norm = nn.LayerNorm(dim)

    def forward(self, G_i, action_i, step_idx=None):
        state = G_i
        if step_idx is not None:
            if not 0 <= step_idx < self.step_embedding.num_embeddings:
                raise ValueError("Rollout exceeds latent_state_max_frames")
            state = state + self.step_embedding.weight[step_idx]
        action = self.action_embedding(action_i.to(G_i.dtype)).unsqueeze(1)
        tokens = torch.cat((state, action), dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(G_i + self.delta(tokens[:, :G_i.shape[1]]))


class LatentStateRollout(nn.Module):
    def __init__(self, visual_hidden_dim=3072, num_tokens=4, dim=256, action_dim=11,
                 initializer_layers=2, transition_layers=2, num_heads=8,
                 mlp_ratio=4.0, dropout=0.0, max_frames=64):
        super().__init__()
        self.initializer = LatentStateInitializer(
            visual_hidden_dim, num_tokens, dim, initializer_layers, num_heads, mlp_ratio, dropout)
        self.transition = LatentStateTransition(
            dim, action_dim, transition_layers, num_heads, mlp_ratio, dropout, max_frames)
        self.output_projection = nn.Linear(dim, visual_hidden_dim)
        self.max_frames = max_frames

    def forward(self, first_frame_tokens, action_seq):
        if action_seq.ndim != 3 or not 1 <= action_seq.shape[1] <= self.max_frames:
            raise ValueError("action_seq must be [B,f,K] with 1 <= f <= max_frames")
        state = self.initializer(first_frame_tokens)
        states = [state]
        for i in range(action_seq.shape[1] - 1):
            state = self.transition(state, action_seq[:, i], step_idx=i)
            states.append(state)
        low = torch.stack(states, dim=1)
        return low, self.output_projection(low.flatten(1, 2))


class LatentStateToVisualAttention(nn.Module):
    def __init__(self, visual_hidden_dim=3072, dim=256, num_heads=8):
        super().__init__()
        self.video_norm = nn.LayerNorm(visual_hidden_dim)
        self.state_norm = nn.LayerNorm(visual_hidden_dim)
        self.q = nn.Linear(visual_hidden_dim, dim)
        self.k = nn.Linear(visual_hidden_dim, dim)
        self.v = nn.Linear(visual_hidden_dim, dim)
        self.out = nn.Linear(dim, visual_hidden_dim)
        self.num_heads = num_heads
        if dim % num_heads:
            raise ValueError("Latent attention dim must be divisible by num_heads")
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, video_tokens, latent_state_tokens):
        def heads(x):
            return x.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        q = heads(self.q(self.video_norm(video_tokens)))
        k = heads(self.k(self.state_norm(latent_state_tokens)))
        v = heads(self.v(latent_state_tokens))
        update = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
        return video_tokens + self.gate * self.out(update)


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, levels=(8, 8, 8)):
        super().__init__()
        if not levels or any(int(level) < 2 for level in levels):
            raise ValueError("FSQ levels must be integers >= 2")
        self.vocab_size = math.prod(levels)
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.long))
        self.register_buffer("basis", torch.tensor([
            math.prod(levels[i + 1:]) for i in range(len(levels))], dtype=torch.long))

    def forward(self, u):
        if u.shape[-1] != self.levels.numel():
            raise ValueError("FSQ input dimension must equal number of levels")
        bounded = u.float().tanh()
        bins = ((bounded + 1) * (self.levels - 1) / 2).round().long()
        centers = 2 * bins.float() / (self.levels - 1) - 1
        quantized = bounded + (centers - bounded).detach()
        indices = (bins * self.basis).sum(dim=-1)
        return quantized.to(u.dtype), indices


class GameTransitionTokenizer(nn.Module):
    def __init__(self, visual_hidden_dim=3072, num_tokens=4, dim=256,
                 num_layers=2, num_heads=8, mlp_ratio=4.0, dropout=0.0,
                 fsq_levels=(8, 8, 8)):
        super().__init__()
        self.input_projection = nn.Linear(visual_hidden_dim, dim)
        self.queries = nn.Parameter(torch.randn(num_tokens, dim) / math.sqrt(dim))
        # Pair roles make previous/next order visible to otherwise permutation-invariant attention.
        self.pair_embedding = nn.Parameter(torch.randn(2, dim) / math.sqrt(dim))
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(num_layers)
        ])
        self.to_codes = nn.Linear(dim, len(fsq_levels))
        self.quantizer = FiniteScalarQuantizer(fsq_levels)
        self.from_codes = nn.Linear(len(fsq_levels), dim)
        self.reconstruction_head = _mlp(2 * dim, int(dim * mlp_ratio), dim, dropout)

    def forward(self, clean_visual_tokens, latent_frames=None):
        x = clean_visual_tokens
        if x.ndim == 3:
            if latent_frames is None or latent_frames < 2 or x.shape[1] % latent_frames:
                raise ValueError("Supply latent_frames >= 2 dividing the clean token sequence")
            x = x.unflatten(1, (latent_frames, -1))
        if x.ndim != 4 or x.shape[1] < 2:
            raise ValueError("Clean tokens must describe at least two frames: [B,f,P,C]")
        if latent_frames is not None and latent_frames != x.shape[1]:
            raise ValueError("latent_frames disagrees with clean tokens")
        low = self.input_projection(x)
        batch, frames, _, dim = low.shape
        pairs = torch.cat((low[:, :-1] + self.pair_embedding[0],
                           low[:, 1:] + self.pair_embedding[1]), dim=2).flatten(0, 1)
        queries = self.queries.unsqueeze(0).expand(batch * (frames - 1), -1, -1)
        for block in self.blocks:
            queries = block(queries, pairs)
        quantized, indices = self.quantizer(self.to_codes(queries))
        embs = self.from_codes(quantized).reshape(batch, frames - 1, -1, dim)
        pooled = low.mean(dim=2)
        target = (pooled[:, 1:] - pooled[:, :-1]).detach()
        prediction = self.reconstruction_head(torch.cat((pooled[:, :-1], embs.mean(dim=2)), dim=-1))
        return {"z_indices": indices.reshape(batch, frames - 1, -1), "z_embs": embs,
                "loss_tok": F.smooth_l1_loss(prediction.float(), target.float()),
                "pred_delta": prediction, "target_delta": target}


class LatentTransitionPredictor(nn.Module):
    def __init__(self, dim=256, action_dim=11, vocab_size=512, mlp_ratio=4.0):
        super().__init__()
        self.action_embedding = nn.Linear(action_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = _mlp(dim, int(dim * mlp_ratio), vocab_size)

    def forward(self, G_seq_low, action_seq):
        if G_seq_low.shape[:2] != action_seq.shape[:2]:
            raise ValueError("State and action sequences must share [B,f]")
        action = self.action_embedding(action_seq[:, :-1].to(G_seq_low.dtype)).unsqueeze(2)
        return self.mlp(self.norm(G_seq_low[:, :-1] + action))
