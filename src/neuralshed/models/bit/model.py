"""BIT, ported from ``bit/`` with the read-in moved ahead of the patcher.

Upstream order is smooth -> patchify -> per-dataset embed. That fixes the input width at
``patch_width`` for the whole batch, because the depthwise smoother is built with
``groups=512`` and the patchifier with ``p2=512``, both of which run before any per-dataset
layer gets to see the tensor. A per-dataset adapter therefore cannot absorb a different channel
count, and the 512 in the config is a data fact masquerading as a hyper-parameter.

Here::

    SpikeCounts (B, C_spike, T) ─ Linear(C_spike → D) ─┐
    BandPower   (B, C_sbp,   T) ─ Linear(C_sbp   → D) ─┴─ sum ─→ (B, T, D)  [BitReadIn]
                                       + participant embedding + region embedding
                                                     │
                              Conv1d(D → D, kernel = stride = patch_bins)     [BitPatcher]
                                                     │  (B, P, D)
                              transformer, causal mask AND padding mask       [BitTrunk]
                                                     │
                              Linear(D → n_classes), log_softmax              [CTC head]
                           or Linear(D → C), no softmax             [dense regression head]

Which head the stack ends in is the config's business: a collection declaring ``target_stream``
predicts a dense ``(B, P, C)`` target instead of characters, because a kinematic is not a
distribution.

Raw channel count now only reaches ``BitReadIn``. Adding a recording adds one projection per
stream and leaves the trunk untouched. Projections are also ``C -> D`` rather than ``5C -> D``,
roughly a fifth of the parameters, since they run before patching rather than after.

There is one projection per **layout**, and one layout is one stream, so which projection a
tensor goes through is decided by the layout key the manifest resolved for it rather than by a
position in a list.

Three upstream defects are fixed rather than carried over:

1. **No padding mask.** Upstream's only mask is the causal one. With ``causal=False``, which
   its own config comment recommends for offline accuracy, nothing masks padding and every
   query attends to zero-filled frames. Here :class:`BitTrunk` always takes a key-padding mask.
2. **Silent routing fallback.** ``_route_and_embed`` routed everything through
   ``dataset_keys[0]`` whenever routing was empty or unmatched, so a misconfigured run trained
   on the wrong adapter without complaint. Here an unknown adapter raises.
3. **Unimplemented per-dataset heads.** ``per_dataset_heads=True`` was accepted and ignored,
   ``forward`` always using the first key's head. The head is shared here, and there is no
   flag that promises otherwise.

Gaussian smoothing is deliberately absent. It belongs upstream of the model as a serializable
wrapper extractor, in physical time, applied after resampling.
"""

import math
import typing as tp

import torch
from einops import rearrange
from torch import nn

from neuralshed.models.bit.config import BitConfig

if tp.TYPE_CHECKING:
    from neuralshed.manifest import CollectionManifest
    from neuralshed.models.bit.preprocess import BitPreprocessingState


def sanitize(key: str) -> str:
    """Adapter key as an ``nn.Module`` name (``.`` is the submodule separator)."""
    return key.replace(".", "_").replace("/", "__")


# ===========================================================================
# Read-in
# ===========================================================================


class BitReadIn(nn.Module):
    """One linear projection per input layout, mapping that stream's channels into width D.

    One layout is one stream, so a recording carrying spike counts and band power has two
    projections whose outputs are summed. A recording with no band power simply has no second
    projection, which is why a missing stream is an absent tensor rather than a block of zeros
    the trunk would have to learn to ignore.

    Participant and region embeddings are zero-initialized. A default ``N(0, 1)`` embedding
    would swamp unit-variance normalized features and park CTC in the all-blank solution for
    many epochs.

    Fitted normalization is applied here, from buffers, so the statistics travel inside the
    checkpoint and cannot silently go missing between fitting and inference.
    """

    def __init__(self, config: BitConfig):
        super().__init__()
        self.layout_keys = list(config.layout_keys)
        self.streams = dict(config.layout_streams)
        self.primary = config.primary_stream
        dim = config.dim

        self.project = nn.ModuleDict(
            {
                sanitize(key): nn.Linear(config.layout_channels[key], dim)
                for key in self.layout_keys
            }
        )
        if not any(s == self.primary for s in self.streams.values()):
            raise ValueError(
                f"no layout reads {self.primary!r}; the collection declares no primary "
                "neural input, so there is nothing for the trunk to read"
            )

        self.norm = nn.LayerNorm(dim)
        self.participant = (
            nn.Embedding(config.n_participants, dim)
            if config.use_participant_embedding
            else None
        )
        self.region = (
            nn.Embedding(config.n_regions, dim) if config.use_region_embedding else None
        )
        for emb in (self.participant, self.region):
            if emb is not None:
                nn.init.zeros_(emb.weight)
        self._normalized_keys: set[str] = set()

    def load_preprocessing(self, state: "BitPreprocessingState") -> None:
        """Register fitted statistics as buffers, so they are saved with the weights."""
        for name, tensor in state.as_buffers().items():
            self.register_buffer(f"stat_{name}", tensor, persistent=True)
        self._normalized_keys = set(state.stats)

    def _normalize(self, x: torch.Tensor, layout_key: str) -> torch.Tensor:
        stem = f"stat_{sanitize(layout_key)}"
        mean = getattr(self, f"{stem}__mean", None)
        if mean is None:
            return x
        std = getattr(self, f"{stem}__std")
        return (x - mean) / std

    def forward(
        self,
        arrays: dict[str, torch.Tensor],
        layouts: dict[str, str],
        participant_idx: torch.Tensor | None = None,
        region_idx: torch.Tensor | None = None,
        normalized: bool = False,
    ) -> torch.Tensor:
        """``(B, C, T)`` per stream -> ``(B, T, D)``, summed across the streams present.

        ``layouts`` maps each stream's event class to the layout key it was recorded under,
        which is what selects its projection. ``normalized=True`` says the caller already
        z-scored the arrays, which is how masked pretraining avoids normalizing twice.
        """
        if self.primary not in arrays:
            raise KeyError(
                f"batch has no {self.primary!r} input; it carries {sorted(arrays)}"
            )
        h = None
        for event_type, x in arrays.items():
            layout_key = layouts.get(event_type)
            if layout_key is None:
                raise KeyError(
                    f"batch supplies {event_type!r} but names no layout for it; the trial "
                    "and the manifest disagree about which streams this recording has"
                )
            name = sanitize(layout_key)
            if name not in self.project:
                raise KeyError(
                    f"no projection for layout {layout_key!r}; the model was built from a "
                    f"manifest with {len(self.layout_keys)} input layouts and this is not one"
                )
            if not normalized:
                x = self._normalize(x, layout_key)
            projected = self.project[name](x.transpose(1, 2))  # (B, T, D)
            h = projected if h is None else h + projected

        h = self.norm(h)
        cond = None
        if self.participant is not None and participant_idx is not None:
            cond = self.participant(participant_idx)
        if self.region is not None and region_idx is not None:
            reg = self.region(region_idx)
            cond = reg if cond is None else cond + reg
        return h if cond is None else h + cond.unsqueeze(1)


# ===========================================================================
# Patcher
# ===========================================================================


class BitPatcher(nn.Module):
    """Non-overlapping temporal patches over the D-wide representation.

    Upstream reshaped ``(time x raw features)`` into ``5 * 512`` flat tokens, which is why raw
    width had to equal ``patch_width`` exactly rather than merely divide it: a width that was a
    multiple would have folded the extra feature block into the *token* axis and scrambled time
    order. Running after the read-in makes this a plain strided conv, and width stops mattering.
    """

    def __init__(self, dim: int, patch_bins: int):
        super().__init__()
        self.patch_bins = patch_bins
        self.proj = nn.Conv1d(dim, dim, kernel_size=patch_bins, stride=patch_bins)
        self.norm = nn.LayerNorm(dim)

    def out_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Patch count per item. Ceiling, matching the right-pad to a whole patch."""
        return torch.div(
            input_lengths + self.patch_bins - 1, self.patch_bins, rounding_mode="floor"
        ).clamp(min=1)

    def pool_target(
        self, target: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Average a bin-rate target into patches: ``(B, C, T) -> (B, P, C)``.

        The model predicts once per patch, so a dense target sampled on the input's clock has
        to be brought to the same rate before it can be subtracted from a prediction. The
        average runs over each patch's *valid* bins only: a trial whose length is not a whole
        number of patches has a last partial patch, and counting its padding would pull that
        patch's target toward zero. Patches past an item's own length come out zero and are
        masked by ``out_lengths`` in the loss.

        The result comes out in the head's ``(B, P, C)`` layout rather than the batch's
        ``(B, C, T)`` one, so the loss subtracts it from a prediction without a transpose.
        """
        b, c, t = target.shape
        pad = (-t) % self.patch_bins
        if pad:
            target = nn.functional.pad(target, (0, pad))
        total = target.shape[-1]
        valid = (
            torch.arange(total, device=target.device)[None]
            < input_lengths.to(target.device)[:, None]
        )
        patches = total // self.patch_bins
        counts = valid.view(b, patches, self.patch_bins).sum(-1).clamp(min=1)
        sums = (
            (target * valid.unsqueeze(1)).view(b, c, patches, self.patch_bins).sum(-1)
        )
        return (sums / counts.unsqueeze(1)).transpose(1, 2)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, T, D) -> (B, P, D)
        pad = (-h.shape[1]) % self.patch_bins
        if pad:
            h = nn.functional.pad(h, (0, 0, 0, pad))
        return self.norm(self.proj(h.transpose(1, 2)).transpose(1, 2))


# ===========================================================================
# Trunk
# ===========================================================================


class Attention(nn.Module):
    """Multi-head attention with T5-style relative position bias and a combined mask.

    ``padding_mask`` is ``True`` on valid patches. It is combined with the causal mask instead
    of replacing it, which is the fix for upstream attending freely over padding whenever
    ``causal=False``.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        max_rel_dist: int = 200,
        use_relative_bias: bool = True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if not (heads == 1 and dim_head == dim)
            else nn.Identity()
        )
        self.max_rel_dist = max_rel_dist
        self.use_relative_bias = use_relative_bias
        if use_relative_bias:
            self.rel_pos_bias = nn.Embedding(2 * max_rel_dist - 1, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.norm(x)
        q, k, v = (
            rearrange(t, "b n (h d) -> b h n d", h=self.heads)
            for t in self.to_qkv(x).chunk(3, dim=-1)
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if self.use_relative_bias:
            n = x.size(1)
            i = torch.arange(n, device=x.device).unsqueeze(1)
            j = torch.arange(n, device=x.device).unsqueeze(0)
            rel = (
                (i - j).clamp(-self.max_rel_dist + 1, self.max_rel_dist - 1)
                + self.max_rel_dist
                - 1
            )
            dots = dots + self.rel_pos_bias(rel).squeeze(-1)[None, None]

        if mask is not None:
            dots = dots.masked_fill(~mask, torch.finfo(dots.dtype).min)

        attn = self.dropout(self.attend(dots))
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.to_out(out)


class FFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BitTrunk(nn.Module):
    """Shared transformer encoder. Nothing here depends on channel count or dataset."""

    def __init__(self, config: BitConfig):
        super().__init__()
        self.causal = config.causal
        self.norm = nn.LayerNorm(config.dim)
        self.layers = nn.ModuleList(
            nn.ModuleList(
                [
                    Attention(
                        config.dim,
                        config.heads,
                        config.dim_head,
                        config.dropout,
                        max_rel_dist=config.max_rel_dist,
                        use_relative_bias=config.t5_style_pos,
                    ),
                    FFN(config.dim, config.mlp_dim_ratio * config.dim, config.dropout),
                ]
            )
            for _ in range(config.depth)
        )

    def build_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        """``(B, P)`` valid-patch mask -> ``(B, 1, P, P)`` allowed-attention mask.

        A fully padded row would produce an all-``-inf`` softmax row and NaN gradients, so
        padded queries are allowed to attend to themselves. Their outputs are discarded
        downstream by the CTC input lengths.
        """
        b, p = padding_mask.shape
        allowed = padding_mask[:, None, None, :].expand(b, 1, p, p)
        if self.causal:
            i = torch.arange(p, device=padding_mask.device)
            allowed = allowed & (i[None, None, :, None] >= i[None, None, None, :])
        self_attend = torch.eye(p, dtype=torch.bool, device=padding_mask.device)
        return allowed | self_attend[None, None]

    def forward(
        self, x: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        mask = None if padding_mask is None else self.build_mask(padding_mask)
        for attn, ffn in self.layers:
            x = attn(x, mask=mask) + x
            x = ffn(x) + x
        return self.norm(x)


# ===========================================================================
# Whole model
# ===========================================================================


class BitCrossEncoder(nn.Module):
    """Read-in -> patcher -> trunk -> head.

    ``forward`` takes a batch dict as produced by ``collate_bit``. Which head it ends in is the
    config's business: a text collection gets log-probabilities of shape ``(B, P, n_classes)``,
    and a collection declaring ``target_stream`` gets raw predictions ``(B, P, C)`` with no
    softmax, because a velocity is not a distribution. Emission lengths come from
    :meth:`out_lengths` either way, which is the same arithmetic the patcher uses, so neither
    loss sees a length the trunk did not produce.
    """

    def __init__(self, config: BitConfig):
        super().__init__()
        self.config = config
        self.readin = BitReadIn(config)
        self.patcher = BitPatcher(config.dim, config.patch_bins)
        self.trunk = BitTrunk(config)
        self.input_dropout = nn.Dropout(config.input_dropout)
        self.head = nn.Linear(config.dim, config.head_width)
        self.mask_token = nn.Parameter(
            torch.zeros(config.dim)
            if config.mask_token_zeros
            else torch.randn(config.dim) * 0.02
        )
        nn.init.zeros_(self.head.bias)
        if not config.is_regression:
            # bias against the blank so training leaves the all-blank collapse promptly
            self.head.bias.data[0] = -3.0

    @classmethod
    def from_manifest(
        cls, manifest: "CollectionManifest", n_classes: int, **overrides: tp.Any
    ) -> "BitCrossEncoder":
        return cls(BitConfig.from_manifest(manifest, n_classes, **overrides))

    def load_preprocessing(self, state: "BitPreprocessingState") -> None:
        self.readin.load_preprocessing(state)
        # The target's statistics register here rather than on the read-in, because it is the
        # head that has to undo them. Buffers travel in the checkpoint either way, so a scale
        # cannot go missing between fitting and inference.
        for name, tensor in state.target_buffers().items():
            self.register_buffer(f"target_stat_{name}", tensor, persistent=True)

    def out_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return self.patcher.out_lengths(input_lengths)

    def _target_stats(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
        """This batch's fitted target scale, or ``None`` if none was fitted for it.

        A batch is one recording, so one layout key answers for every item in it. The key rides
        in ``layouts`` beside the input streams' keys; the target itself is not in ``arrays``,
        so the read-in never sees it.
        """
        layout_key = batch["layouts"].get(self.config.target_stream)
        if layout_key is None:
            return None
        stem = f"target_stat_{sanitize(layout_key)}"
        mean = getattr(self, f"{stem}__mean", None)
        if mean is None:
            return None
        # Stored as ``(C, 1)`` to divide a ``(C, T)`` array; the head works in ``(B, P, C)``.
        return mean.squeeze(-1), getattr(self, f"{stem}__std").squeeze(-1)

    def normalize_target(self, target: torch.Tensor, batch: dict) -> torch.Tensor:
        """A pooled target in source units -> the z-score space the model trains in."""
        stats = self._target_stats(batch)
        if stats is None:
            return target
        mean, std = stats
        return (target - mean) / std

    def denormalize(self, prediction: torch.Tensor, batch: dict) -> torch.Tensor:
        """A prediction -> the target's source units, so a metric reads in real units."""
        stats = self._target_stats(batch)
        if stats is None:
            return prediction
        mean, std = stats
        return prediction * std + mean

    def apply_time_mask(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SpecAugment-style patch masking, training only. Masks land inside each item's own
        valid length, so padding is never "masked" and the augmentation strength does not
        depend on how long the longest item in the batch happened to be."""
        b, p, _ = x.shape
        cfg = self.config
        device = x.device
        span = (cfg.max_mask_pct * lengths.float()).clamp(min=1.0)
        mask = torch.zeros(b, p, dtype=torch.bool, device=device)
        arange = torch.arange(p, device=device)[None]
        for _ in range(cfg.num_masks):
            width = (torch.rand(b, device=device) * span).floor().long()
            start = (
                (torch.rand(b, device=device) * (lengths - width).clamp(min=1).float())
                .floor()
                .long()
            )
            mask |= (arange >= start[:, None]) & (arange < (start + width)[:, None])
        out = torch.where(mask.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        return out, mask

    def forward(self, batch: dict) -> torch.Tensor:
        """CTC log-probabilities ``(B, P, n_classes)``, or a dense prediction ``(B, P, C)``."""
        h = self.readin(
            batch["arrays"],
            batch["layouts"],
            batch.get("participant_idx"),
            batch.get("region_idx"),
        )
        # Zero the read-in output past each item's own length, before patching.
        # The read-in has a bias and a LayerNorm, so a zero-padded input bin comes out
        # *non*-zero. Without this, an item's last partial patch is filled with whatever
        # the batch padding produced, and the same trial gives different logits depending
        # on what it was batched with.
        input_lengths = batch["input_lengths"].to(h.device)
        valid = torch.arange(h.shape[1], device=h.device)[None] < input_lengths[:, None]
        h = h * valid.unsqueeze(-1).to(h.dtype)

        x = self.patcher(h)  # (B, P, D)
        lengths = self.out_lengths(batch["input_lengths"]).to(x.device)
        if self.training and self.config.max_mask_pct > 0 and self.config.num_masks > 0:
            x, _ = self.apply_time_mask(x, lengths)
        x = self.input_dropout(x)
        padding_mask = (
            torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        )
        x = self.trunk(x, padding_mask=padding_mask)
        out = self.head(x)
        return out if self.config.is_regression else out.log_softmax(-1)

    def n_parameters(self) -> dict[str, int]:
        """Parameter count split into the layout-specific part and the shared part, which is
        the number the read-in-before-patcher change is meant to move."""
        readin = sum(p.numel() for p in self.readin.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"readin": readin, "shared": total - readin, "total": total}


def build_bit_model(
    manifest: "CollectionManifest",
    vocab: tp.Any = None,
    state: "BitPreprocessingState | None" = None,
    **overrides: tp.Any,
) -> BitCrossEncoder:
    """The target API: a manifest and a vocabulary in, a ready model out.

    ``vocab=None`` is how a dense-target collection builds: it has no characters to predict, so
    it passes ``target_stream=`` instead and the head's width comes off the manifest.
    """
    n_classes = 0 if vocab is None else vocab.n_classes
    model = BitCrossEncoder.from_manifest(manifest, n_classes, **overrides)
    if state is not None:
        model.load_preprocessing(state)
    return model


def gaussian_kernel(sigma_bins: float, width_bins: int) -> torch.Tensor:
    """Normalized 1-D Gaussian, kept here only so the level-2 smoothing extractor and any
    upstream comparison agree on the kernel. The model itself does not smooth."""
    x = torch.arange(width_bins, dtype=torch.float32) - (width_bins - 1) / 2
    k = torch.exp(-((x / sigma_bins) ** 2) / 2) / (sigma_bins * math.sqrt(2 * math.pi))
    return k / k.sum()


__all__ = [
    "BitReadIn",
    "BitPatcher",
    "BitTrunk",
    "Attention",
    "FFN",
    "BitCrossEncoder",
    "build_bit_model",
    "gaussian_kernel",
    "sanitize",
]
