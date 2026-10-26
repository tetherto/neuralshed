"""From a collection to loaders a CTC decoder can train on.

    bundle = BitDataBundle.from_collection("speech-text")
    model  = build_bit_model(bundle.manifest, bundle.vocab, bundle.state)
    trainer.fit(model, bundle.train_loader(), bundle.val_loader())

The pipeline is: run the collection's step chain, resolve a manifest, preflight it, segment
every ``target_is_valid`` trial at its own duration, drop the trials CTC cannot align, fit
normalization on the training split only, and hand back loaders whose batches are homogeneous
in recording.

This module is the *task* end of the pipeline. The manifest in
:mod:`neuralshed.manifest` is generic; encoding text into CTC targets and dropping trials
too short to emit them is specific to this decoder, and lives here.

Streams stay separate all the way through rather than being concatenated into one wide array.
Concatenating would make a missing ``BandPower`` indistinguishable from a block of real zeros,
and would tie the channel ordering to the order events happen to appear in the frame.
"""

import dataclasses
import typing as tp

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import neuralset as ns
from neuralshed import units
from neuralshed.manifest import (
    DEFAULT_GROUP_BY,
    DEFAULT_KEY_TEMPLATE,
    CollectionManifest,
)
from neuralshed.models.bit.config import BitConfig
from neuralshed.models.bit.preprocess import (
    BitPreprocessingState,
    NormalizationPolicy,
)
from neuralshed.models.data import SessionBatchSampler
from neuralshed.text import CharVocab, ctc_min_input_length
from neuralshed.transforms import (
    MEMBER_COLUMN,
    SPLIT_COLUMN,
    TARGET_FLAG,
    TARGET_TEXT,
    valid_targets,
)

#: The stream every recording must carry for a trial to exist.
PRIMARY_STREAM = "SpikeCounts"

#: What ``ids`` holds on a trial whose target is a dense array rather than text.
_NO_IDS = np.empty(0, dtype=np.int64)


@dataclasses.dataclass
class BitTrial:
    """One segmented trial, before batching.

    ``layouts`` maps each stream's event class to the layout key it was recorded under. A trial
    therefore references several layouts, one per stream, and one recording (``group_key``).
    """

    group_key: str
    layouts: dict[str, str]  # event class -> layout key
    member: str
    participant: str
    arrays: dict[str, np.ndarray]  # event class -> (C, T) float32
    ids: np.ndarray  # (L,) int64 CTC target
    text: str
    split: str

    participant_idx: int = 0
    region_idx: int = 0
    #: A dense ``(C, T)`` target on the input's own clock, for collections decoding a
    #: kinematic instead of text. ``ids`` and ``text`` are empty when this is set.
    target: np.ndarray | None = None

    @property
    def n_frames(self) -> int:
        return self.arrays[PRIMARY_STREAM].shape[1]


@dataclasses.dataclass
class BitBatch:
    """A padded batch drawn from one recording. ``as_dict`` is what the model consumes."""

    arrays: dict[str, torch.Tensor]  # event class -> (B, C, T)
    layouts: dict[str, str]  # event class -> layout key
    group_key: str
    input_lengths: torch.Tensor  # (B,) valid bins before padding
    target_ids: torch.Tensor  # (B, S) right-padded with the blank
    target_lengths: torch.Tensor  # (B,)
    participant_idx: torch.Tensor
    region_idx: torch.Tensor
    members: list[str]
    texts: list[str]
    #: ``(B, C, T)`` dense target, padded like the inputs and masked by ``input_lengths``.
    #: ``None`` for a text collection, whose target is ``target_ids``.
    target: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self._dict: dict | None = None  # not a field, so `replace` rebuilds it

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, key: str) -> tp.Any:
        """Read like a dict, so a model's ``batch["arrays"]`` takes this object as-is.

        Both decoders consume this one type. That is the point of there being one loader: a
        parity run cannot drift on batch construction if there is only one way to build a batch.
        """
        return self.as_dict()[key]

    def get(self, key: str, default: tp.Any = None) -> tp.Any:
        return self.as_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.as_dict()

    def keys(self):
        """With :meth:`__getitem__`, makes ``{**batch}`` work, which is how the masked
        reconstruction path swaps in a modified ``arrays`` without rebuilding the batch.
        """
        return self.as_dict().keys()

    @property
    def flat_targets(self) -> torch.Tensor:
        """Targets concatenated, the form ``nn.CTCLoss`` wants."""
        return torch.cat(
            [row[:n] for row, n in zip(self.target_ids, self.target_lengths)]
        )

    def as_dict(self) -> dict:
        if self._dict is not None:
            return self._dict
        self._dict = {
            "arrays": self.arrays,
            "layouts": self.layouts,
            "group_key": self.group_key,
            "input_lengths": self.input_lengths,
            "participant_idx": self.participant_idx,
            "region_idx": self.region_idx,
            "targets": self.flat_targets,
            "target_ids": self.target_ids,
            "target_lengths": self.target_lengths,
            "texts": self.texts,
            "members": self.members,
            "target": self.target,
        }
        return self._dict

    def to(self, device) -> "BitBatch":
        move = lambda t: t.to(device)  # noqa: E731
        return dataclasses.replace(
            self,
            arrays={k: move(v) for k, v in self.arrays.items()},
            input_lengths=move(self.input_lengths),
            target_ids=move(self.target_ids),
            target_lengths=move(self.target_lengths),
            participant_idx=move(self.participant_idx),
            region_idx=move(self.region_idx),
            target=None if self.target is None else move(self.target),
        )


def collate_bit(batch: tp.Sequence[BitTrial], blank: int = 0) -> BitBatch:
    """Right-zero-pad each stream to a common T; right-pad targets with the blank.

    Every item must come from one recording, which ``SessionBatchSampler(key_attr="group_key")``
    guarantees. It is asserted rather than assumed because a mixed batch would train silently,
    which is exactly the upstream failure mode this port set out to remove.
    """
    groups = {t.group_key for t in batch}
    if len(groups) != 1:
        raise ValueError(f"a batch must come from one recording, got {sorted(groups)}")
    group_key = groups.pop()
    layouts = batch[0].layouts

    n_frames = max(t.n_frames for t in batch)
    n_target = max(len(t.ids) for t in batch)
    b = len(batch)

    arrays: dict[str, torch.Tensor] = {}
    for stream in batch[0].arrays:
        channels = batch[0].arrays[stream].shape[0]
        out = torch.zeros(b, channels, n_frames)
        for i, t in enumerate(batch):
            a = t.arrays[stream]
            if a.shape[0] != channels:
                raise ValueError(
                    f"recording {group_key!r} mixes {channels} and {a.shape[0]} "
                    f"{stream} channels within one batch"
                )
            out[i, :, : a.shape[1]] = torch.from_numpy(a)
        arrays[stream] = out

    target_ids = torch.full((b, n_target), blank, dtype=torch.long)
    for i, t in enumerate(batch):
        target_ids[i, : len(t.ids)] = torch.from_numpy(t.ids)

    target = None
    if batch[0].target is not None:
        # Padded to the same T as the inputs, not to its own maximum: a dense target shares
        # the input's clock, so one `input_lengths` masks both.
        channels = batch[0].target.shape[0]
        target = torch.zeros(b, channels, n_frames)
        for i, t in enumerate(batch):
            target[i, :, : t.target.shape[1]] = torch.from_numpy(t.target)

    tensor = lambda vals: torch.tensor(vals, dtype=torch.long)  # noqa: E731
    return BitBatch(
        arrays=arrays,
        layouts=dict(layouts),
        group_key=group_key,
        input_lengths=tensor([t.n_frames for t in batch]),
        target_ids=target_ids,
        target_lengths=tensor([len(t.ids) for t in batch]),
        participant_idx=tensor([t.participant_idx for t in batch]),
        region_idx=tensor([t.region_idx for t in batch]),
        members=[t.member for t in batch],
        texts=[t.text for t in batch],
        target=target,
    )


class BitTrialDataset(Dataset):
    """A flat view over :class:`BitTrial` objects for one split."""

    def __init__(self, trials: tp.Sequence[BitTrial], split: str | None = None):
        self.trials = [t for t in trials if split is None or t.split == split]

    def __len__(self) -> int:
        return len(self.trials)

    def __getitem__(self, i: int) -> BitTrial:
        return self.trials[i]


class BitDataBundle:
    """Everything a training run needs, assembled once and reused across experiments."""

    def __init__(
        self,
        trials: list[BitTrial],
        manifest: CollectionManifest,
        vocab: CharVocab,
        state: BitPreprocessingState,
        events: pd.DataFrame,
        dropped: dict[str, int],
        patch_bins: int,
        already_normalized: frozenset[str] = frozenset(),
        target_stream: str | None = None,
    ):
        self.trials = trials
        self.manifest = manifest
        self.vocab = vocab
        self.state = state
        self.events = events
        self.dropped = dropped
        self.patch_bins = patch_bins
        self.already_normalized = frozenset(already_normalized)
        self.target_stream = target_stream

    def __repr__(self) -> str:
        head = (
            f"target={self.target_stream}"
            if self.target_stream is not None
            else f"n_classes={self.vocab.n_classes}"
        )
        return (
            f"BitDataBundle({len(self.trials)} trials {self.split_counts()}, "
            f"{len(self.manifest)} layouts, {head})"
        )

    def split_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.trials:
            out[t.split] = out.get(t.split, 0) + 1
        return dict(sorted(out.items()))

    def subset(
        self,
        *,
        participants: tp.Iterable[str] | None = None,
        members: tp.Iterable[str] | None = None,
        max_trials_per_participant: int | None = None,
        seed: int = 0,
    ) -> "BitDataBundle":
        """A new bundle over some of the trials, with the manifest filtered to match.

        This is what an isolated-versus-pooled comparison needs: one participant's trials and a
        manifest holding only their layouts, so the isolated model gets the same read-in the
        pooled model gives that participant and nothing else.

        Layout keys are preserved rather than recomputed, so both runs name the same layouts and
        their results are comparable. ``max_trials_per_participant`` caps each person's
        contribution, drawn round-robin across their recordings so the cap does not silently
        favour whichever session is longest. Fitted statistics carry over unchanged, since they
        are keyed by layout, so a subset never refits on less data than the parent.
        """
        keep_p = set(participants) if participants is not None else None
        keep_m = set(members) if members is not None else None

        selected = [
            t
            for t in self.trials
            if (keep_p is None or t.participant in keep_p)
            and (keep_m is None or t.member in keep_m)
        ]
        if max_trials_per_participant is not None:
            selected = _cap_per_participant(selected, max_trials_per_participant, seed)
        if not selected:
            raise ValueError("subset selected no trials")

        manifest = self.manifest.filter(
            groups={t.group_key for t in selected},
            participants=keep_p,
            members=keep_m,
        )
        p_idx = manifest.participant_index
        r_idx = manifest.region_index
        trials = [
            dataclasses.replace(
                t,
                participant_idx=p_idx.get(t.participant, 0),
                region_idx=r_idx.get(
                    manifest.layout_for(t.group_key, PRIMARY_STREAM).region, 0
                ),
            )
            for t in selected
        ]
        return BitDataBundle(
            trials,
            manifest,
            self.vocab,
            self.state,
            self.events,
            self.dropped,
            self.patch_bins,
            self.already_normalized,
            self.target_stream,
        )

    def config(self, **overrides: tp.Any) -> BitConfig:
        """A config for this bundle, refusing any override that changes the temporal stride.

        ``patch_bins`` decided which trials were dropped as too short for their own target. A
        config that patches more coarsely would put those trials back below the CTC bound, where
        ``zero_infinity=True`` turns them into zero gradients rather than an error. Re-run
        ``from_collection`` with the new ``patch_bins`` instead.
        """
        n_classes = self.vocab.n_classes
        if self.target_stream is not None:
            # A dense target has no characters to predict, so the head's width comes off the
            # manifest instead and n_classes is left unused.
            overrides.setdefault("target_stream", self.target_stream)
            n_classes = 0
        config = BitConfig.from_manifest(self.manifest, n_classes, **overrides)
        if config.patch_bins != self.patch_bins:
            raise ValueError(
                f"this bundle dropped trials at patch_bins={self.patch_bins}, but "
                f"patch_ms={config.patch_ms} at {config.frequency} Hz gives "
                f"{config.patch_bins}; rebuild the bundle with "
                f"patch_bins={config.patch_bins} rather than overriding it here"
            )
        return config

    def loader(
        self,
        split: str | None,
        batch_size: int = 32,
        shuffle: bool | None = None,
        seed: int = 0,
    ) -> DataLoader:
        ds = BitTrialDataset(self.trials, split=split)
        if not len(ds):
            raise ValueError(f"split {split!r} has no trials")
        if shuffle is None:
            shuffle = split == "train"
        sampler = SessionBatchSampler(
            ds, batch_size=batch_size, shuffle=shuffle, seed=seed, key_attr="group_key"
        )
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_bit)

    def train_loader(self, batch_size: int = 32, seed: int = 0) -> DataLoader:
        return self.loader("train", batch_size=batch_size, seed=seed)

    def val_loader(self, batch_size: int = 32) -> DataLoader:
        return self.loader("val", batch_size=batch_size, shuffle=False)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_collection(
        cls,
        collection: "str | ns.Step",
        *,
        inputs: tp.Sequence[str] = ("SpikeCounts", "BandPower"),
        targets: tp.Sequence[str] = ("Sentence",),
        frequency: float | tp.Literal["native"] = "native",
        target_unit: units.TargetUnit = "Hz",
        key_template: tp.Sequence[str] = DEFAULT_KEY_TEMPLATE,
        group_by: tp.Sequence[str] = DEFAULT_GROUP_BY,
        charset_version: str = "v1",
        target_stream: str | None = None,
        regions: tp.Mapping[str, str] | None = None,
        participants: tp.Mapping[str, str] | None = None,
        already_normalized: tp.Iterable[str] = (),
        policy: NormalizationPolicy = NormalizationPolicy(),
        patch_bins: int | None = None,
        manifest: CollectionManifest | None = None,
        validate: bool = True,
        verbose: bool = True,
    ) -> "BitDataBundle":
        """Run a collection end to end.

        ``manifest`` accepts an already-resolved manifest, which is how a stored one is used
        instead of being rebuilt. When omitted it is resolved here from the same events, by the
        same code, so the two routes cannot drift.

        ``patch_bins`` is the model's temporal stride, needed here because a trial too short to
        emit its own target must be dropped before training rather than discovered as a silent
        zero gradient at step 500.

        ``frequency`` is recorded on the manifest as its target rate, which is what every
        stream is then seen at, so ``patch_bins`` follows the requested rate rather than a
        member's native one and a 100 ms patch stays 100 ms. ``"native"`` declares no target
        and leaves each stream at its own rate.

        ``target_unit`` harmonizes the neural streams across collections that bin at different
        widths. ``"native"`` leaves every stream as its source stored it.

        A target derived from other streams (a velocity, the direction to a cued target) is
        a ``DeriveStream`` step in the collection, not an argument here: by the time a
        bundle is built it is a published stream like any other.
        """
        from neuralshed.models.bit.validate import validate_collection_for_bit

        events = _run_collection(collection)
        if manifest is None:
            manifest = CollectionManifest.from_events(
                events,
                inputs=inputs,
                targets=targets,
                key_template=key_template,
                group_by=group_by,
                target_rate=None if frequency == "native" else float(frequency),
                regions=regions,
                participants=participants,
            )
        vocab = CharVocab.canonical(charset_version)

        if validate:
            report = validate_collection_for_bit(
                events,
                manifest=manifest,
                vocab=vocab,
                patch_bins=patch_bins,
                target_stream=target_stream,
            )
            report.raise_if_failed()
            if verbose:
                print(report)

        if patch_bins is None:
            patch_bins = BitConfig.from_manifest(manifest, vocab.n_classes).patch_bins

        trials, dropped = _extract_trials(
            events,
            manifest,
            vocab,
            frequency=frequency,
            target_unit=target_unit,
            patch_bins=patch_bins,
            target_stream=target_stream,
        )
        if not trials:
            raise ValueError("no trials survived extraction")

        state = BitPreprocessingState.fit(
            trials, manifest, policy, already_normalized=already_normalized
        )
        if verbose:
            print(
                f"bundle: {len(trials)} trials, {len(manifest)} layouts, "
                f"{manifest.n_participants} participants, dropped={dropped}"
            )
        return cls(
            trials,
            manifest,
            vocab,
            state,
            events,
            dropped,
            patch_bins,
            frozenset(already_normalized),
            target_stream,
        )


def _cap_per_participant(trials: list[BitTrial], cap: int, seed: int) -> list[BitTrial]:
    """Keep at most ``cap`` trials per participant, drawn round-robin across their recordings.

    Round-robin rather than a flat sample, so a participant recorded across a long session and
    a short one does not end up represented almost entirely by the long one.
    """
    rng = np.random.default_rng(seed)
    by_person: dict[str, dict[str, list[BitTrial]]] = {}
    for trial in trials:
        by_person.setdefault(trial.participant, {}).setdefault(
            trial.group_key, []
        ).append(trial)

    kept: list[BitTrial] = []
    for buckets in by_person.values():
        for group in buckets.values():
            rng.shuffle(group)
        queues = [list(g) for g in buckets.values()]
        taken = 0
        while taken < cap and any(queues):
            for queue in queues:
                if not queue or taken >= cap:
                    continue
                kept.append(queue.pop())
                taken += 1
    return kept


def _run_collection(collection: "str | ns.Step") -> pd.DataFrame:
    from neuralshed import studies, transforms  # noqa: F401 — registers the classes

    if isinstance(collection, str):
        from neuralshed.database import collections as db_collections

        chain = db_collections.load_chain(collection_name=collection)
    else:
        chain = collection
    return chain.run().reset_index(drop=True)


def _extractor_for(event_type: str, frequency: float | str, target_unit: str):
    """The extractor that reads one dense stream, chosen by its event class.

    Dispatching on the class rather than hard-coding the neural pair is what lets a behavioral
    stream join a collection without a second extraction path. ``target_unit`` applies to the
    neural streams only; behavioral series carry no convertible unit.
    """
    from neuralset.events.etypes import Event

    from neuralshed import events as _events  # noqa: F401 — registers our classes
    from neuralshed.extractors import (
        BehavioralSeriesExtractor,
        NeuralFeatureExtractor,
    )

    cls = Event._CLASSES[event_type]
    common = dict(frequency=frequency, event_types=event_type, allow_missing=True)
    if issubclass(cls, _events.BehavioralSeries):
        return BehavioralSeriesExtractor(**common, aggregation="cat")
    return NeuralFeatureExtractor(**common, aggregation="cat", target_unit=target_unit)


def _extract_trials(
    events: pd.DataFrame,
    manifest: CollectionManifest,
    vocab: CharVocab,
    *,
    frequency: float | tp.Literal["native"],
    target_unit: units.TargetUnit,
    patch_bins: int,
    target_stream: str | None = None,
) -> tuple[list[BitTrial], dict[str, int]]:
    """Segment every target trial at its own duration and read its target.

    ``target_stream`` names a dense event class to read as the target, for a collection
    decoding a kinematic. It comes out of the same ``Segmenter`` window as the inputs, so it
    is aligned to them bin for bin, and the CTC-specific steps (encoding text, dropping
    trials too short to spell it) are skipped. Left at ``None`` the target is the text.

    One shared ``Segmenter`` per recording, because extractor output width is fixed by the
    recording and a single segmenter across recordings would produce ragged arrays. Trials are
    read straight from the extractors rather than through ``SegmentDataset.collate_fn``, which
    pads every segment to a common duration and would erase the variable lengths CTC needs.
    """
    targets = valid_targets(events)
    # Required rather than defaulted: an absent split would silently make every trial a
    # training trial, and normalization would then be fitted on the whole collection.
    if SPLIT_COLUMN not in targets.columns:
        raise KeyError(
            f"events have no {SPLIT_COLUMN!r} column; add a split transform "
            "(AssignHashSplit or neuralset's SklearnSplit) to the collection chain"
        )
    blank = targets[SPLIT_COLUMN].isna() | (targets[SPLIT_COLUMN].astype(str) == "")
    if blank.any():
        raise ValueError(
            f"{int(blank.sum())} of {len(targets)} target rows have no {SPLIT_COLUMN}; "
            "the split transform ran but did not reach every target"
        )
    dropped = {"no_target": 0, "too_short": 0, "empty_window": 0, "no_signal": 0}
    trials: list[BitTrial] = []
    p_idx = manifest.participant_index
    r_idx = manifest.region_index
    group_of = targets.apply(lambda r: _group_key(r, manifest.group_by), axis=1)

    for group_key in manifest.groups:
        rows = targets[group_of == group_key]
        if rows.empty:
            continue
        streams = {
            event: spec
            for event, spec in manifest.streams_of(group_key).items()
            if spec.is_array and spec.role in ("input", "both")
        }
        if PRIMARY_STREAM not in streams:
            continue
        layouts = {event: spec.layout_key for event, spec in streams.items()}
        primary = streams[PRIMARY_STREAM]

        timelines = set(rows["timeline"].astype(str))
        scope = events[events["timeline"].astype(str).isin(timelines)]
        keep_index = set(rows.index)
        seg_events = scope[
            scope.index.isin(keep_index)
            | ~scope[TARGET_FLAG].fillna(False).astype(bool)
        ]

        extractors = {
            event: _extractor_for(event, frequency, target_unit)
            for event in sorted(streams)
        }
        if target_stream is not None:
            target_spec = manifest.streams_of(group_key).get(target_stream)
            if target_spec is None:
                continue
            # The target's layout key rides in `layouts` beside the input streams' keys. The
            # read-in iterates `arrays`, which the target is popped out of, so the extra entry
            # reaches only the head, which needs it to pick the scale fitted for this session.
            layouts[target_stream] = target_spec.layout_key
            extractors[target_stream] = _extractor_for(
                target_stream, frequency, target_unit
            )
        ds = ns.dataloader.Segmenter(
            extractors=extractors,
            trigger_query=f"{TARGET_FLAG} == True",
            duration=None,
        ).apply(seg_events)
        ds.prepare()

        trigger_index = ds.triggers.index.tolist()
        for i, seg in enumerate(ds.segments):
            row = events.loc[trigger_index[i]]
            arrays = {
                event: ex(
                    seg.ns_events,
                    start=seg.start,
                    duration=seg.duration,
                    trigger=seg.trigger,
                )
                .numpy()
                .astype(np.float32)
                for event, ex in extractors.items()
            }
            target_array = arrays.pop(target_stream) if target_stream else None
            spikes = arrays[PRIMARY_STREAM]
            if spikes.shape[1] == 0:
                dropped["empty_window"] += 1
                continue
            # `allow_missing=True` keeps one bad trigger from killing the run, but it returns
            # a zero-filled window rather than raising. Left in, that window is a real text
            # target paired with silence. Every channel zero across the whole window is that
            # case, not data.
            if not spikes.any():
                dropped["no_signal"] += 1
                continue

            if target_stream is not None:
                # A window the target stream does not cover is the kinematic equivalent of
                # an unencodable sentence: a real input paired with no target.
                if target_array.shape[1] != spikes.shape[1]:
                    dropped["no_target"] += 1
                    continue
                text, ids = "", _NO_IDS
            else:
                text = str(row[TARGET_TEXT])
                ids = vocab.encode(text)
                if len(ids) == 0:
                    dropped["no_target"] += 1
                    continue
                n_patches = max(1, -(-spikes.shape[1] // patch_bins))
                if n_patches < ctc_min_input_length(ids):
                    dropped["too_short"] += 1
                    continue

            trials.append(
                BitTrial(
                    group_key=group_key,
                    layouts=layouts,
                    member=primary.member,
                    participant=primary.participant,
                    arrays=arrays,
                    ids=ids,
                    text=text,
                    split=str(row[SPLIT_COLUMN]),
                    participant_idx=p_idx.get(primary.participant, 0),
                    region_idx=r_idx.get(primary.region, 0),
                    target=target_array,
                )
            )
    return trials, dropped


def _group_key(row: tp.Any, group_by: tp.Sequence[str]) -> str:
    """The recording a row belongs to, built the same way the manifest built its groups."""
    member = row.get(MEMBER_COLUMN) or row.get("study") or "?"
    context = {
        "member": member,
        "study": row.get("study") or member,
        "session": row.get("session"),
    }
    return "/".join(str(context[c]) for c in group_by)


__all__ = [
    "PRIMARY_STREAM",
    "BitTrial",
    "BitBatch",
    "collate_bit",
    "BitTrialDataset",
    "BitDataBundle",
]
