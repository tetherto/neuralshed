"""The extensible brain-to-text model: per-adapter read-in -> shared backbone -> heads.

    SpikeCounts (B, C_spike, T) ─┐
                                 ├─[read-in]─> (B, T, D) ─┐
    BandPower   (B, C_sbp,  T) ──┘  (one per stream)      ├─ + region emb + participant emb
                                                           ▼
                                    shared conv (stride 2) + BiGRU backbone
                                                           │  (B, T', 2H)
                                      ┌────────────────────┴────────────────────┐
                                      ▼                                          ▼
                      CTC head -> (B, T', n_classes)    recon head -> (B, T', C_spike)
                      (supervised text)                 (self-supervised: predict masked bins)

Everything left of the backbone is *layout-specific* (one small linear projection per stream,
plus a region and participant embedding). Everything from the backbone rightward is *shared*, so
pooling data trains one common representation. Adding a recording means adding one projection
per stream and one embedding row, leaving the backbone intact.

This is the GRU baseline. It and the BIT transformer in :mod:`neuralshed.models.bit` build
from the same :class:`~neuralshed.manifest.CollectionManifest` and consume the same
``BitBatch``, so a CER difference between them is a difference of models rather than of data
pipelines.

The backbone downsamples time by ``stride`` (like the reference decoders). CTC emission lengths
and the reconstruction target are therefore computed at the encoder frame rate (T'), not the raw
bin rate -- the same choice masked speech models (wav2vec/HuBERT) make.
"""

import typing as tp

import torch
from torch import nn
from torch.nn import functional as F

if tp.TYPE_CHECKING:
    from neuralshed.manifest import CollectionManifest
    from neuralshed.models.bit.preprocess import BitPreprocessingState


class SessionReadIn(nn.Module):
    """One linear projection per input layout, mapping that stream's channels into width D.

    One layout is one stream, so a recording with spike counts and band power gets two
    projections whose outputs are summed, and a recording with neither gets none. Region and
    participant embeddings are added so the backbone knows each clip's provenance.

    Built from a :class:`~neuralshed.manifest.CollectionManifest`, the same table BIT's
    read-in builds from, so both decoders see one layout list resolved from one events frame.

    Fitted normalization is applied here, from buffers, exactly as ``BitReadIn`` does it. It has
    to happen inside the model rather than on the trial arrays, because the bundle is shared:
    z-scoring the arrays in place would normalize them for whichever model ran first and leave
    the fitted statistics unable to travel with a checkpoint.
    """

    def __init__(
        self,
        manifest: "CollectionManifest",
        d_model: int,
        use_participant: bool = True,
    ):
        super().__init__()
        from neuralshed.models.bit.model import sanitize

        layouts = manifest.array_layouts
        self.project = nn.ModuleDict(
            {
                sanitize(s.layout_key): nn.Linear(int(s.channels or 0), d_model)
                for s in layouts
            }
        )
        # stabilizes the projected features
        self.norm = nn.LayerNorm(d_model)
        self.region = nn.Embedding(max(1, manifest.n_regions), d_model)
        self.participant = (
            nn.Embedding(manifest.n_participants, d_model) if use_participant else None
        )
        # zero-init conditioning so it starts as a no-op (a default N(0,1) embedding would swamp
        # the unit-variance normalized features and stall CTC in the all-blank solution)
        nn.init.zeros_(self.region.weight)
        if self.participant is not None:
            nn.init.zeros_(self.participant.weight)

    def load_preprocessing(self, state: "BitPreprocessingState") -> None:
        """Register fitted statistics as buffers, so they are saved with the weights."""
        for name, tensor in state.as_buffers().items():
            self.register_buffer(f"stat_{name}", tensor, persistent=True)

    def _normalize(self, x, layout_key: str):
        from neuralshed.models.bit.model import sanitize

        stem = f"stat_{sanitize(layout_key)}"
        mean = getattr(self, f"{stem}__mean", None)
        if mean is None:
            return x
        return (x - mean) / getattr(self, f"{stem}__std")

    def forward(
        self,
        arrays,
        layouts,
        participant_idx,
        region_idx,
        normalized: bool = False,
    ) -> torch.Tensor:
        from neuralshed.models.bit.model import sanitize

        h = None
        for event_type, x in arrays.items():
            layout_key = layouts[event_type]
            name = sanitize(layout_key)
            if name not in self.project:
                raise KeyError(
                    f"no projection for layout {layout_key!r}; the model and the batch were "
                    "built from different manifests"
                )
            if not normalized:
                x = self._normalize(x, layout_key)
            projected = self.project[name](x.transpose(1, 2))
            h = projected if h is None else h + projected
        h = self.norm(h)
        cond = self.region(region_idx)
        if self.participant is not None:
            cond = cond + self.participant(participant_idx)
        return h + cond.unsqueeze(1)  # broadcast provenance over time


class Backbone(nn.Module):
    """Shared temporal encoder: two strided convs then a bidirectional GRU.

    ``strides`` gives the per-conv temporal stride; the total downsample is their product (default
    ``(2, 2)`` = 4x, matching the reference decoders). Downsampling is what makes the long attempted
    -speech / handwriting windows (hundreds of bins) cheap enough to train on CPU.
    """

    def __init__(
        self,
        d_model: int,
        hidden: int,
        layers: int = 2,
        kernel: int = 5,
        strides: tuple[int, int] = (2, 2),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.kernel, self.strides, self.pad = kernel, strides, kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel, stride=strides[0], padding=self.pad),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel, stride=strides[1], padding=self.pad),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            d_model,
            hidden,
            layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.out_dim = 2 * hidden

    def out_length(self, length: torch.Tensor) -> torch.Tensor:
        """Emission length after the two strided convs."""
        for s in self.strides:
            length = (
                torch.div(length + 2 * self.pad - self.kernel, s, rounding_mode="floor")
                + 1
            )
        return length

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B, T, D) -> (B, T', 2H)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        return self.gru(h)[0]


class BrainToTextCTC(nn.Module):
    """Per-session read-in + shared backbone + a CTC text head and a masked-reconstruction head.

    ``frequency`` is carried as a construct parameter (the bin rate the inputs were resampled to);
    it is metadata here, but keeping it on the model documents what the projections expect.
    """

    PRIMARY = "SpikeCounts"

    def __init__(
        self,
        manifest: "CollectionManifest",
        n_classes: int,
        d_model: int = 192,
        hidden: int = 192,
        gru_layers: int = 2,
        strides: tuple[int, int] = (2, 2),
        dropout: float = 0.3,
        use_participant: bool = True,
        frequency: float = 50.0,
    ):
        super().__init__()
        from neuralshed.models.bit.model import sanitize

        self.frequency = frequency
        self.readin = SessionReadIn(manifest, d_model, use_participant=use_participant)
        self.backbone = Backbone(
            d_model, hidden, gru_layers, strides=strides, dropout=dropout
        )
        self.ctc_head = nn.Linear(self.backbone.out_dim, n_classes)
        # bias the CTC head against the blank (id 0) so training leaves the all-blank collapse
        # immediately rather than sitting in it for many epochs
        nn.init.zeros_(self.ctc_head.bias)
        self.ctc_head.bias.data[0] = -3.0
        # one head per primary-stream layout, reconstructing that layout's own channels
        self.recon_heads = nn.ModuleDict(
            {
                sanitize(s.layout_key): nn.Linear(
                    self.backbone.out_dim, int(s.channels or 0)
                )
                for s in manifest.array_layouts
                if s.event_type == self.PRIMARY
            }
        )

    @classmethod
    def from_manifest(
        cls,
        manifest: "CollectionManifest",
        n_classes: int,
        state: "BitPreprocessingState | None" = None,
        **kwargs: tp.Any,
    ) -> "BrainToTextCTC":
        """Mirror of ``build_bit_model``, so a parity run builds both models the same way.

        Pass ``bundle.state`` unless you know the inputs are already on one scale. Without it
        this trains on raw counts and raw µV² side by side, which is what the retired loader's
        in-place z-scoring used to hide.
        """
        model = cls(manifest, n_classes, **kwargs)
        if state is not None:
            model.load_preprocessing(state)
        return model

    def load_preprocessing(self, state: "BitPreprocessingState") -> None:
        self.readin.load_preprocessing(state)

    def out_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Valid CTC emission length per item at the encoder frame rate."""
        return self.backbone.out_length(input_lengths).clamp(min=1)

    # -- shared trunk -------------------------------------------------------
    def _embed(self, batch: dict, normalized: bool = False) -> torch.Tensor:
        """``batch`` is a ``BitBatch.as_dict()``, the same object BIT's forward takes.

        ``normalized=True`` says the arrays were already z-scored by the caller, which is how
        :meth:`masked_reconstruct` avoids normalizing twice.
        """
        return self.readin(
            batch["arrays"],
            batch["layouts"],
            batch["participant_idx"],
            batch["region_idx"],
            normalized=normalized,
        )

    def normalized_arrays(self, batch: dict) -> dict:
        """The batch's arrays after this model's fitted statistics."""
        layouts = batch["layouts"]
        return {
            stream: self.readin._normalize(value, layouts[stream])
            for stream, value in batch["arrays"].items()
        }

    # -- supervised: neural -> log-probs over characters --------------------
    def forward(self, batch: dict) -> torch.Tensor:
        """Return CTC log-probabilities, shape ``(B, T', n_classes)``."""
        feats = self.backbone(self._embed(batch))
        return self.ctc_head(feats).log_softmax(-1)

    # -- self-supervised: predict the masked neural activity ----------------
    def masked_reconstruct(self, batch: dict, mask: torch.Tensor):
        """Hide contiguous spans of the input, then predict the neural activity that was hidden.

        The masked bins are zeroed *before* the read-in, so the model must infer them from
        temporal context. The reconstruction target is the recording's own primary-stream
        channels pooled to the encoder frame rate T' (a fixed target -- no representation
        collapse). The loss (in the
        training module) is an MSE over the masked frames only. Returns
        ``(prediction, target, frame_mask)`` at rate T'.

        Both the masking and the target are in **normalized** space. Zero is the channel mean
        there, so a masked bin is genuinely uninformative; zeroing a raw count instead plants an
        out-of-distribution value, and an MSE against raw counts is dominated by whichever
        channels happen to have the largest scale.
        """
        from neuralshed.models.bit.model import sanitize

        head = sanitize(batch["layouts"][self.PRIMARY])
        arrays = self.normalized_arrays(batch)
        target_source = arrays[self.PRIMARY].detach()
        m = mask.unsqueeze(1)  # (B, 1, T)
        masked = {
            **batch,
            "arrays": {k: v.masked_fill(m, 0.0) for k, v in arrays.items()},
        }
        feats = self.backbone(self._embed(masked, normalized=True))  # (B, T', 2H)
        n_frames = feats.shape[1]
        target = F.adaptive_avg_pool1d(target_source, n_frames).transpose(
            1, 2
        )  # (B, T', C_spike)
        frame_mask = (
            F.adaptive_max_pool1d(mask.float().unsqueeze(1), n_frames).squeeze(1) > 0.5
        )  # (B, T')
        return self.recon_heads[head](feats), target, frame_mask


__all__ = ["SessionReadIn", "Backbone", "BrainToTextCTC"]
