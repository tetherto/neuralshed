"""BIT configuration, derived from a resolved manifest rather than hand-listed.

Upstream (``bit/bit_cross_encoder_config.py``) carries ``patch_size=(5, 512)`` and a literal
``dataset_keys`` list. Both encode data facts into the model file: 512 is a channel count, and
the key list has to be edited whenever a collection changes. Here the patch is stated in
**milliseconds** and the layouts come from the manifest, so the same config text describes a
192-channel 100 Hz recording and a 512-channel 50 Hz one.

This is the model-specific end of the manifest. Turning a layout table into one model's layer
shapes is that model's business, and it is the only part of BIT that knows a manifest exists.
"""

import dataclasses
import typing as tp

if tp.TYPE_CHECKING:
    from neuralshed.manifest import CollectionManifest


@dataclasses.dataclass(frozen=True)
class BitConfig:
    """Trunk hyper-parameters plus the layout table.

    ``layout_channels`` is flat, one channel count per layout, because one layout is one
    stream. A recording with spike counts and band power contributes two entries, and the
    read-in builds a projection for each and sums them.

    ``patch_ms`` is the patch duration in physical time. ``patch_bins`` is derived per rate, so
    100 ms is 5 bins at 50 Hz and 10 bins at 100 Hz, and the trunk sees the same time span
    either way.
    """

    n_classes: int  # includes the CTC blank
    layout_keys: tuple[str, ...]
    layout_channels: dict[str, int]  # layout_key -> channels
    layout_streams: dict[str, str]  # layout_key -> event class it reads
    n_participants: int = 1
    n_regions: int = 1

    #: A dense regression target: the event class the head predicts, and its channel width.
    #: Both ``None`` for a text collection, whose head predicts ``n_classes`` characters
    #: instead. They cannot be listed beside ``n_classes`` because a defaulted field cannot
    #: precede an undefaulted one.
    target_stream: str | None = None
    target_channels: int | None = None

    frequency: float = 50.0
    patch_ms: float = 100.0

    dim: int = 384
    depth: int = 7
    heads: int = 4
    dim_head: int = 64
    mlp_dim_ratio: int = 4
    dropout: float = 0.4
    input_dropout: float = 0.2
    t5_style_pos: bool = True
    max_rel_dist: int = 200

    max_mask_pct: float = 0.075
    num_masks: int = 20
    mask_token_zeros: bool = False

    causal: bool = False
    use_participant_embedding: bool = True
    use_region_embedding: bool = True

    def __post_init__(self) -> None:
        if (self.target_stream is None) != (self.target_channels is None):
            raise ValueError(
                "target_stream and target_channels are set together; got "
                f"{self.target_stream!r} and {self.target_channels!r}"
            )
        if self.target_channels is None:
            if self.n_classes < 2:
                raise ValueError(
                    f"n_classes must include the blank, got {self.n_classes}"
                )
        elif self.target_channels < 1:
            raise ValueError(
                f"target_channels must be positive, got {self.target_channels}"
            )
        if not self.layout_keys:
            raise ValueError("config needs at least one layout key")
        missing = set(self.layout_keys) - set(self.layout_channels)
        if missing:
            raise ValueError(f"layout keys with no channel count: {sorted(missing)}")
        if self.patch_bins < 1:
            raise ValueError(
                f"patch_ms={self.patch_ms} at {self.frequency} Hz rounds to "
                f"{self.patch_bins} bins; raise patch_ms or the sampling rate"
            )

    @property
    def patch_bins(self) -> int:
        """Patch length in input bins. The model's temporal stride."""
        return int(round(self.patch_ms * self.frequency / 1000.0))

    @property
    def is_regression(self) -> bool:
        """True when the head predicts a dense target rather than characters."""
        return self.target_channels is not None

    @property
    def head_width(self) -> int:
        """Output width of the head: classes for text, channels for a dense target."""
        return self.n_classes if self.target_channels is None else self.target_channels

    @property
    def primary_stream(self) -> str:
        """The stream every recording is required to have, and the reconstruction target."""
        return "SpikeCounts"

    @classmethod
    def from_manifest(
        cls, manifest: "CollectionManifest", n_classes: int, **overrides: tp.Any
    ) -> "BitConfig":
        """Build a config whose layout table matches the manifest exactly.

        The read-in is then guaranteed to have a projection for every stream the data can
        produce, which is the correspondence upstream's silent fallback to ``dataset_keys[0]``
        was hiding.
        """
        layouts = manifest.array_layouts
        if not layouts:
            raise ValueError("manifest has no dense input streams to read in")
        if "frequency" not in overrides:
            rates = {
                round(rate, 6)
                for rate in (s.effective_rate(manifest.target_rate) for s in layouts)
                if rate is not None
            }
            if len(rates) > 1:
                raise ValueError(
                    f"manifest spans {sorted(rates)} Hz; declare a target rate on the "
                    "collection or pass frequency= explicitly"
                )
            overrides["frequency"] = rates.pop()
        target_stream = overrides.get("target_stream")
        if target_stream is not None and "target_channels" not in overrides:
            widths = {
                int(s.channels or 0)
                for s in manifest.layouts
                if s.is_array and s.event_type == target_stream
            }
            if len(widths) != 1:
                raise ValueError(
                    f"target stream {target_stream!r} resolves to {sorted(widths)} channels "
                    "across the manifest; one head predicts one target space"
                )
            overrides["target_channels"] = widths.pop()
        return cls(
            n_classes=n_classes,
            layout_keys=tuple(s.layout_key for s in layouts),
            layout_channels={s.layout_key: int(s.channels or 0) for s in layouts},
            layout_streams={s.layout_key: s.event_type for s in layouts},
            n_participants=manifest.n_participants,
            n_regions=max(1, manifest.n_regions),
            **overrides,
        )


__all__ = ["BitConfig"]
