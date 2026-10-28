"""Preflight: fail on a collection BIT cannot train on, before anything expensive runs.

Every check here catches a failure that is otherwise silent. Label-free trials train against an
empty target, a rate mismatch between members trains one member at the wrong time scale, a
coarse layout key merges recordings whose channel *k* means two different electrodes, a trial
shorter than its own target contributes an infinite CTC loss that ``zero_infinity=True`` turns
into a zero gradient. None of these raise on their own, and all of them make the loss curve
look plausible.

Preflight comes in two halves, because they cost very different amounts:

:func:`validate_manifest` is **structural**. It reads the layout table alone, so it runs in
milliseconds against a stored manifest with no data drive mounted. That is what a collection
builder can show while somebody is still assembling members.

:func:`validate_collection_for_bit` adds the **data** half: targets, vocabulary coverage,
splits and CTC feasibility, all of which need the events frame.

    from neuralshed.models.bit import validate_collection_for_bit, validate_manifest

    manifest = collections.get_manifest(collection_name="speech-text")
    print(validate_manifest(manifest))                    # offline
    print(validate_collection_for_bit("speech-text"))     # full

Report style follows ``database/validate.py``: errors fail, warnings inform.
"""

import dataclasses
import typing as tp

import pandas as pd

from neuralshed.manifest import CollectionManifest
from neuralshed.models.bit.bundle import PRIMARY_STREAM
from neuralshed.text import CharVocab, ctc_min_input_length
from neuralshed.transforms import (
    MEMBER_COLUMN,
    SPLIT_COLUMN,
    SPLITS,
    TARGET_FLAG,
    TARGET_TEXT,
    valid_targets,
)


@dataclasses.dataclass
class BitReport:
    """Outcome of a preflight. Truth if there are no errors."""

    errors: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)
    facts: dict[str, tp.Any] = dataclasses.field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def __str__(self) -> str:
        head = "preflight OK" if self.ok else f"preflight FAILED ({len(self.errors)})"
        lines = [head]
        for key, value in self.facts.items():
            lines.append(f"  {key}: {value}")
        for e in self.errors:
            lines.append(f"  ERROR   {e}")
        for w in self.warnings:
            lines.append(f"  warning {w}")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError(
                "collection is not BIT-trainable:\n"
                + "\n".join(f"  - {e}" for e in self.errors)
            )


def validate_collection_for_bit(
    collection: "str | pd.DataFrame | tp.Any",
    *,
    manifest: CollectionManifest | None = None,
    vocab: CharVocab | None = None,
    patch_bins: int | None = None,
    target_stream: str | None = None,
) -> BitReport:
    """Check a collection's events (or a collection name / step) against BIT's requirements.

    ``target_stream`` names a dense target, which moves what can go wrong: there is no text to
    encode and no CTC bound to satisfy, so those checks are replaced by one on the target
    stream's own width.
    """
    events = _as_events(collection)
    vocab = vocab or CharVocab.canonical()
    report = BitReport()

    required = (TARGET_FLAG,) if target_stream else (TARGET_FLAG, TARGET_TEXT)
    for column in required:
        if column not in events.columns:
            report.errors.append(
                f"events have no {column!r} column; the collection chain is missing "
                "StandardizeText/ValidateTarget"
            )
    if report.errors:
        return report

    targets = valid_targets(events)
    report.facts["events"] = len(events)
    report.facts["targets"] = len(targets)
    if targets.empty:
        report.errors.append("no rows with target_is_valid=True; nothing to train on")
        return report

    if manifest is None:
        manifest = CollectionManifest.from_events(
            events,
            inputs=("SpikeCounts", "BandPower"),
            targets=(target_stream,) if target_stream else ("Sentence",),
        )

    _check_manifest(manifest, report)
    _check_members(events, targets, report)
    if target_stream is None:
        _check_targets(targets, vocab, report)
    else:
        _check_target_stream(manifest, target_stream, report)
    _check_splits(events, targets, report)
    if patch_bins is not None and target_stream is None:
        _check_ctc_feasibility(targets, manifest, vocab, patch_bins, report)
    return report


def validate_manifest(
    manifest: CollectionManifest, *, require_stream: str | None = PRIMARY_STREAM
) -> BitReport:
    """The structural half, from the layout table alone.

    Everything checked here is a fact the manifest already carries: illegal merges, merges that
    could not be verified, layouts spanning several rates, recordings with no target stream,
    recordings missing the stream this decoder needs. No events, no data drive, no extraction.

    ``require_stream`` is the one task-specific input. A CTC speech decoder needs
    ``SpikeCounts``; pass ``None`` to check only what is true of any model.
    """
    report = BitReport()
    _check_manifest(manifest, report, require_stream=require_stream)
    return report


def _as_events(collection: "str | pd.DataFrame | tp.Any") -> pd.DataFrame:
    if isinstance(collection, pd.DataFrame):
        return collection
    from neuralshed.models.bit.bundle import _run_collection

    return _run_collection(collection)


def _check_members(
    events: pd.DataFrame,
    targets: pd.DataFrame,
    report: BitReport,
) -> None:
    if MEMBER_COLUMN not in events.columns:
        report.warnings.append(
            f"no {MEMBER_COLUMN!r} column; per-member rules cannot be applied and two "
            "members running the same Study class are indistinguishable"
        )
        return
    missing = events[MEMBER_COLUMN].isna().sum()
    if missing:
        report.errors.append(
            f"{missing} rows have no {MEMBER_COLUMN}; a member frame was concatenated "
            "outside CombineStudies"
        )
    counts = targets[MEMBER_COLUMN].astype(str).value_counts().to_dict()
    report.facts["targets per member"] = counts
    # A member with no targets never reaches `counts` at all, so it has to be found by
    # differencing against the members present in the events.
    silent = sorted(set(events[MEMBER_COLUMN].dropna().astype(str)) - set(counts))
    if silent:
        report.errors.append(
            f"members contribute no targets: {silent}; they cost extraction time and train "
            "nothing, so either their ValidateTarget condition is wrong or they do not belong "
            "in this collection"
        )


def _check_targets(
    targets: pd.DataFrame,
    vocab: CharVocab,
    report: BitReport,
) -> None:
    text = targets[TARGET_TEXT].astype(str)
    empty = int((text.str.len() == 0).sum())
    if empty:
        report.errors.append(
            f"{empty} target rows have empty {TARGET_TEXT}; label-free trials must not be "
            "marked target_is_valid (card_2024's test split is the usual source)"
        )
    missing = vocab.coverage(text.tolist())
    if missing:
        report.errors.append(
            f"vocabulary {vocab.version!r} cannot encode {sorted(missing)}; the collection's "
            "StandardizeText scheme and the model's charset disagree, and those characters "
            "would be dropped from the target silently"
        )
    report.facts["vocabulary"] = f"{vocab.version} ({vocab.n_classes} classes)"


def _check_target_stream(
    manifest: CollectionManifest,
    target_stream: str,
    report: BitReport,
) -> None:
    """A dense target has no vocabulary, so what is left to check is its width.

    One head predicts one target space, so recordings disagreeing about how many channels the
    target has is not a mixture the model can represent.
    """
    layouts = [
        s for s in manifest.layouts if s.is_array and s.event_type == target_stream
    ]
    if not layouts:
        report.errors.append(
            f"no recording declares {target_stream!r} as a dense target stream; it is either "
            "not in the collection's declared targets or not a data event"
        )
        return
    widths = sorted({int(s.channels or 0) for s in layouts})
    if len(widths) > 1:
        report.errors.append(
            f"{target_stream!r} is {widths} channels wide across recordings; one head cannot "
            "predict target spaces of different widths"
        )
    missing = manifest.recordings_missing_targets()
    if missing:
        report.errors.append(f"recordings carrying inputs but no target: {missing}")
    report.facts["target"] = (
        f"{target_stream} ({widths} channels, {len(layouts)} layouts)"
    )


def _check_splits(
    events: pd.DataFrame,
    targets: pd.DataFrame,
    report: BitReport,
) -> None:
    if SPLIT_COLUMN not in events.columns:
        report.errors.append(
            f"events have no {SPLIT_COLUMN!r} column; the collection chain is missing "
            "a split transform (e.g. SklearnSplit), so normalization would be fitted on everything"
        )
        return
    split = targets[SPLIT_COLUMN].astype(str)
    counts = split.value_counts().to_dict()
    report.facts["split"] = counts
    unknown = sorted(set(counts) - set(SPLITS))
    if unknown:
        report.errors.append(f"split values outside {SPLITS}: {unknown}")
    if counts.get("train", 0) == 0:
        report.errors.append("no training trials")
    if counts.get("val", 0) == 0:
        report.warnings.append("no validation trials; CER cannot be reported")

    # Leakage: one trial must not appear in two splits. Timeline plus start is its identity.
    identity = targets["timeline"].astype(str) + "@" + targets["start"].astype(str)
    per_trial = pd.DataFrame({"id": identity, "split": split}).drop_duplicates()
    duplicated = per_trial[per_trial["id"].duplicated(keep=False)]
    if not duplicated.empty:
        report.errors.append(
            f"{duplicated['id'].nunique()} trials are assigned to more than one split"
        )


def _check_manifest(
    manifest: CollectionManifest,
    report: BitReport,
    *,
    require_stream: str | None = PRIMARY_STREAM,
) -> None:
    report.facts["layouts"] = len(manifest)
    report.facts["recordings"] = len(manifest.groups)
    report.facts["participants"] = manifest.participants
    report.facts["inputs"] = manifest.inputs
    report.facts["targets"] = manifest.targets
    if manifest.regions:
        report.facts["regions"] = manifest.regions

    for message in manifest.conflicts():
        report.errors.append(message)
    for message in manifest.warnings():
        report.warnings.append(message)

    rates = sorted(
        {
            round(rate, 6)
            for rate in (
                s.effective_rate(manifest.target_rate) for s in manifest.array_layouts
            )
            if rate is not None
        }
    )
    report.facts["rates (Hz)"] = rates
    if len(rates) > 1:
        report.errors.append(
            f"layouts span {rates} Hz; declare a target rate on the collection, or the same "
            "patch duration means a different time span per member"
        )

    silent = manifest.recordings_missing_targets()
    if silent:
        report.errors.append(
            f"{len(silent)} recordings carry input streams but no target stream: "
            f"{silent[:5]}{'...' if len(silent) > 5 else ''}; they cost extraction time and "
            "train nothing"
        )

    if require_stream is not None:
        without = [
            group
            for group in manifest.groups
            if require_stream not in manifest.streams_of(group)
        ]
        if without:
            report.errors.append(
                f"{len(without)} recordings have no {require_stream}: {without[:5]}"
            )
    # Inputs are wanted-if-present, so a recording carrying fewer of them is legal. It is still
    # worth saying out loud, because that read-in trains on less than the others do.
    declared = {s.event_type for s in manifest.array_layouts}
    partial = [
        group
        for group in manifest.groups
        if {e for e, s in manifest.streams_of(group).items() if s.is_array} != declared
    ]
    if partial and len(partial) != len(manifest.groups):
        report.warnings.append(
            f"{len(partial)} of {len(manifest.groups)} recordings carry only some of "
            f"{sorted(declared)}; those read-ins run on fewer streams than the rest"
        )


def _check_ctc_feasibility(
    targets: pd.DataFrame,
    manifest: CollectionManifest,
    vocab: CharVocab,
    patch_bins: int,
    report: BitReport,
) -> None:
    """Trials whose patched emission length cannot cover their own target.

    The bound is ``len(ids) + n_repeats``, not ``len(ids)``: CTC needs a blank between two
    identical adjacent labels, so "coffee" costs one extra frame and "all" one more.

    Patch count is a ceiling, matching the patcher's right-pad to a whole patch and the drop
    rule in ``_extract_trials``. Flooring here would warn about trials extraction then keeps.
    """
    infeasible = 0
    checked = 0
    for _, row in targets.iterrows():
        duration = float(row.get("duration") or 0.0)
        if duration <= 0:
            continue
        try:
            layout = manifest.layout_for(
                _group_key(row, manifest.group_by), PRIMARY_STREAM
            )
        except KeyError:
            continue
        rate = layout.effective_rate(manifest.target_rate)
        if not rate:
            continue
        n_bins = int(duration * rate)
        n_patches = max(1, -(-n_bins // patch_bins))
        if n_patches < ctc_min_input_length(vocab.encode(str(row[TARGET_TEXT]))):
            infeasible += 1
        checked += 1
    report.facts["ctc feasibility"] = f"{checked - infeasible}/{checked} trials"
    if infeasible:
        report.warnings.append(
            f"{infeasible} trials are too short to emit their target at "
            f"patch_bins={patch_bins} and will be dropped during extraction"
        )


def _group_key(row: tp.Any, group_by: tp.Sequence[str]) -> str:
    from neuralshed.models.bit.bundle import _group_key as build

    return build(row, group_by)


__all__ = ["BitReport", "validate_collection_for_bit", "validate_manifest"]
