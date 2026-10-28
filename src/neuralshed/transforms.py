"""neuroai ``Step`` primitives neuralshed adds on top of neuralset.

neuralset's step graph is linear: ``Study`` is a source, every ``EventsTransform`` is
one-in/one-out, and ``Chain`` runs them in order. There is no fan-in, so pooling several
studies has no configurable form. :class:`CombineStudies` is that missing node.

A collection recipe is a ``Chain`` whose head is the fan-in and whose final transform assigns
validated data layouts. :class:`neuralshed.collection.Collection` enforces that structure::

    {"name": "Collection", "steps": {
        "CombineStudies": {"name": "CombineStudies", "chains": {
            "Card2024Brain2Text": [{"name": "Card2024Brain2Text", "query": "session_index < 5"}],
            "Fan2023Handwriting_DummyTransform": [{"name": "Fan2023Handwriting"}, {"name": "DummyTransform"}],
        }},
        "DummyTransform": {"name": "DummyTransform"},
        "AssignDataLayouts": {"name": "AssignDataLayouts", "inputs": ["SpikeCounts"]},
    }}

Per-study transforms are steps inside a member chain; per-collection transforms are steps
after the fan-in. Both register under their name via neuralset's ``name`` discriminator, so a
stored config resolves once this module is imported.

:class:`StandardizeText`, :class:`ValidateTarget`, :class:`AssignSourceSplit`,
:class:`AssignHashSplit` and :class:`AssignDataLayouts` are generic, dataset-agnostic
transforms. None knows anything about BIT. A member needing a different rule puts its own copy
in that member's chain rather than these carrying a per-member dict, since the ``Chain``
structure *is* the per-member mechanism. :class:`AssignSourceSplit` is the clearest case: its
mapping is per member by construction, because it runs before the fan-in.

neuralset's ``SklearnSplit`` also works, but splits whole values of one column: with
``split_by="timeline"`` a dataset that stores one timeline per *session* puts that entire
session in one split. :class:`AssignHashSplit` keys on several columns instead, so a trial can
be finer than a timeline.
"""

import json
import typing as tp
import urllib.parse
import warnings

import neuralset as ns
import numpy as np
import pandas as pd
from neuralset.events import utils as ns_utils
from neuralset.events.transforms.utils import DeterministicSplitter

#: Column naming which collection member a row came from. Written by :class:`CombineStudies`.
MEMBER_COLUMN = "collection_member"

#: The boolean column :class:`ValidateTarget` writes and the segmenter triggers on.
TARGET_FLAG = "target_is_valid"
#: The normalized-text column :class:`StandardizeText` writes.
TARGET_TEXT = "canonical_text"
#: The split column, written by whichever neuralset split transform the chain uses.
SPLIT_COLUMN = "split"

#: Columns written by :class:`AssignDataLayouts`.
DATA_LAYOUT_KEY = "data_layout_key"
DATA_ROLE = "data_role"
#: Optional source feature width carried by dense event rows.
N_FEATURES = "n_features"

SPLITS = ("train", "val", "test")


def _study_name(member: ns.Step) -> str:
    """The member's study class name (a bare Study, or the head of its Chain)."""
    head = member[0] if isinstance(member, ns.Chain) else member
    return type(head).__name__


class CombineStudies(ns.base.Step):
    """Fan-in: run each member chain and concatenate their events into one frame.

    ``chains`` maps a member key to a ``Study`` or a ``Chain`` headed by one. Members are
    concatenated in a canonical order (by study class name, then key) rather than in dict
    order, so the output depends only on what the config *says*, matching the cache uid,
    which sorts the keys too.

    Every row gets a ``collection_member`` column naming its key. ``study`` alone cannot
    stand in for it: two members can run the same ``Study`` class under different queries,
    and downstream rules are per member. It is set on each frame *before* the concat because
    ``standardize_events`` only back-fills its own five invariants, so a column present on
    one frame and absent on another would come out ``NaN``.
    """

    CACHE_TYPE: tp.ClassVar[str | None] = "ValidatedParquet"  # preserves str dtypes

    chains: dict[str, ns.Step]

    def _ordered(self) -> list[tuple[str, ns.Step]]:
        return sorted(self.chains.items(), key=lambda kv: (_study_name(kv[1]), kv[0]))

    def _run(self, *args: tp.Any) -> pd.DataFrame:
        frames = []
        for key, member in self._ordered():
            frame = member.run()
            if MEMBER_COLUMN in frame.columns and (frame[MEMBER_COLUMN] != key).any():
                raise ValueError(
                    f"member {key!r} already carries a conflicting {MEMBER_COLUMN}: "
                    f"{sorted(set(frame[MEMBER_COLUMN]) - {key})}"
                )
            frames.append(frame.assign(**{MEMBER_COLUMN: key}))
        out = pd.concat(frames, ignore_index=True)
        return ns_utils.standardize_events(out, auto_fill=False)


class DummyTransform(ns.EventsTransform):
    """No-op passthrough, to exercise the collection builder before real transforms land."""

    note: str = ""

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        return events


# ===========================================================================
# Generic preprocessing transforms
# ===========================================================================


class StandardizeText(ns.EventsTransform):
    """Normalize one text column into another, for every row that has text.

    ``scheme`` names the normalizer (see :data:`neuralshed.text.TEXT_SCHEMES`). The scheme
    is the same fact as the model's output alphabet: ``"v1"`` keeps exactly the characters a
    ``CharVocab.canonical("v1")`` can encode, so text cleaned here always encodes without
    silently dropping characters. Rows with no text (a neural stream) get an empty string
    rather than ``NaN``, so a later ``dest != ''`` condition is well-defined everywhere.

    Generic: it knows nothing about targets or BIT. Different members normalize the same way,
    so this normally runs once after the fan-in rather than per member.
    """

    source: str = "text"
    dest: str = "canonical_text"
    scheme: str = "v1"

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        from neuralshed.text import TEXT_SCHEMES

        if self.scheme not in TEXT_SCHEMES:
            raise ValueError(
                f"{type(self).__name__}: unknown scheme {self.scheme!r}; "
                f"have {sorted(TEXT_SCHEMES)}"
            )
        normalize = TEXT_SCHEMES[self.scheme]
        out = events.copy()
        source = (
            out[self.source]
            if self.source in out.columns
            else pd.Series(index=out.index, dtype=object)
        )
        out[self.dest] = source.map(
            lambda v: "" if v is None or pd.isna(v) else normalize(v)
        )
        return out


class DeriveStream(ns.EventsTransform):
    """Append a stream computed from other streams of the same timeline.

    ``operation`` names an entry of :data:`neuralshed.derivations.DERIVATIONS`, and
    ``sources`` lists the event types it reads, in the order it reads them
    (``("CursorPositionSeries", "TargetPositionSeries")`` for ``"intention"``). One row is
    appended per timeline that has all of them, carrying a loader that reads the sources and
    applies the operation, so the derived array is computed when it is read and over the whole
    timeline rather than inside a trial window.

    This is a *derivation*: it produces a new quantity, which is the dataset's business and a
    model never undoes. Normalization, which rescales one quantity for one network and is
    reversed before scoring, is the model's business and lives with the model.

    A timeline missing any source is left alone, since a member publishing no cursor position
    simply has no velocity to derive. Two candidate rows of one source type in one timeline is
    an error: which cursor to differentiate is a decision for a ``QueryEvents`` step upstream,
    not a silent pick here.
    """

    operation: str
    sources: tuple[str, ...]
    emit: str = ""
    feature: str = ""

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        from neuralshed.derivations import DERIVATIONS

        if self.operation not in DERIVATIONS:
            raise ValueError(
                f"{type(self).__name__}: unknown operation {self.operation!r}; "
                f"have {sorted(DERIVATIONS)}"
            )
        spec = DERIVATIONS[self.operation]
        if len(self.sources) != spec.n_sources:
            raise ValueError(
                f"{type(self).__name__}: {self.operation!r} reads {spec.n_sources} "
                f"stream(s), got {list(self.sources)}"
            )
        out = events.copy()
        rows = [
            self._derived_row(picked, spec)
            for _, frame in out.groupby("timeline", sort=False)
            if (picked := self._sources_in(frame)) is not None
        ]
        if not rows:
            raise ValueError(
                f"{type(self).__name__}: no timeline carries all of {list(self.sources)}"
            )
        return ns_utils.standardize_events(
            pd.concat([out, pd.DataFrame(rows)], ignore_index=True), auto_fill=False
        )

    def _sources_in(self, frame: pd.DataFrame) -> list[pd.Series] | None:
        picked: list[pd.Series] = []
        for event_type in self.sources:
            match = frame[frame["type"].astype(str) == event_type]
            if match.empty:
                return None
            if len(match) > 1:
                raise ValueError(
                    f"{type(self).__name__}: timeline "
                    f"{frame['timeline'].iloc[0]!r} has {len(match)} {event_type} rows; "
                    "filter down to one before deriving from it"
                )
            picked.append(match.iloc[0])
        return picked

    def _derived_row(self, picked: list[pd.Series], spec: tp.Any) -> dict[str, tp.Any]:
        head = picked[0]
        feature = self.feature or f"{self.operation}_{head.get('feature', '')}".rstrip(
            "_"
        )
        row = head.to_dict()
        row.update(
            {
                "type": self.emit or spec.emits,
                "feature": feature,
                "unit": spec.unit(str(head.get("unit", ""))),
                "kind": self.operation,
                "processing": "; ".join(
                    part
                    for part in (
                        str(head.get("processing", "")),
                        f"derived: {spec.summary} ({', '.join(self.sources)})",
                    )
                    if part
                ),
                "filepath": self._loader_json(picked),
            }
        )
        if spec.dims is not None:  # an operation that does not keep its sources' width
            row["dims"] = tuple(spec.dims)
            row["n_features"] = len(spec.dims)
        return row

    def _loader_json(self, picked: list[pd.Series]) -> str:
        sources = [str(row["filepath"]) for row in picked]
        try:
            head = json.loads(sources[0])
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{type(self).__name__}: {self.sources[0]} stores its array at "
                f"{sources[0]!r}; deriving needs a SpecialLoader-backed stream"
            ) from error
        payload = {
            "cls": head["cls"],
            "timeline": head["timeline"],
            "method": "_load_derived",
            "kwargs": {
                "operation": self.operation,
                "sources": sources,
                "rate": float(picked[0]["frequency"]),
            },
        }
        if "cls_kwargs" in head:
            payload["cls_kwargs"] = head["cls_kwargs"]
        return json.dumps(payload, sort_keys=True)


class ValidateTarget(ns.EventsTransform):
    """Mark which rows are valid decode targets, as a boolean column.

    ``condition`` is a pandas expression evaluated as a *mask* over the frame (``df.eval``),
    the marking counterpart of neuralset's ``QueryEvents``, which filters with the same
    grammar. It writes a boolean ``dest`` column rather than dropping rows, because the
    segmenter needs the trigger row and the neural-data rows to sit in the same frame. The
    condition can reference a column an earlier step wrote, so ``canonical_text != ''`` works
    once :class:`StandardizeText` has run.

    Generic: the trigger column name is the model side's concern, defaulted to
    ``target_is_valid``. A member whose rule differs puts its own ``ValidateTarget`` in its
    chain rather than this class carrying a per-member dict.
    """

    condition: str = "type == 'Sentence' and canonical_text != ''"
    dest: str = "target_is_valid"

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        out = events.copy()
        try:
            mask = out.eval(self.condition, engine="python")
        except Exception as e:
            raise ValueError(
                f"{type(self).__name__}: condition {self.condition!r} failed to evaluate "
                f"({e}); it must be a pandas expression over existing columns"
            ) from e
        if not isinstance(mask, pd.Series) or mask.dtype != bool:
            raise ValueError(
                f"{type(self).__name__}: condition {self.condition!r} must evaluate to a "
                "boolean column, e.g. 'duration >= 1.0'"
            )
        out[self.dest] = mask
        return out


class AssignSourceSplit(ns.EventsTransform):
    """Map a source's own split labels onto ``train`` / ``val`` / ``test``.

    ``mapping`` translates the values of ``source``. It has no default because the published
    vocabularies disagree (``train``/``val``/``test``, ``train``/``eval``, FALCON's
    ``held-in-calib``/``held-out-calib``), so every member states its own in its own chain::

        AssignSourceSplit(mapping={"held-in-calib": "train", "held-out-calib": "test"})

    A value the mapping does not name becomes ``""``, as does a row with no source label at
    all. That covers both leaving a label out on purpose (card_2024's ``test``, whose
    sentences were never released) and the rows a partition simply does not reach
    (willet_2023's excluded blocks). What happens to those rows is the next step's call, not
    this one's.
    """

    mapping: dict[str, str]
    source: str = "original_split"
    dest: str = "split"

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        if self.source not in events.columns:
            raise KeyError(
                f"{type(self).__name__}: no column {self.source!r} to map from"
            )
        if unknown := sorted(set(self.mapping.values()) - set(SPLITS)):
            raise ValueError(
                f"{type(self).__name__}: mapping sends labels to {unknown}, which are not "
                f"splits; have {list(SPLITS)}"
            )
        out = events.copy()
        out[self.dest] = out[self.source].map(
            lambda v: "" if v is None or pd.isna(v) else self.mapping.get(v, "")
        )
        return out


class AssignHashSplit(ns.EventsTransform):
    """Assign a deterministic train/val/test split by hashing a stable per-trial key.

    The split is a property of the *trial*, and what counts as a trial depends on how the
    source stored its events:

    - One timeline per trial (example: card_2024): the timeline alone identifies the trial.
    - One timeline per recording, block or session, with many trials inside it (example:
      willet_2023): the trial is the target event itself, and the key must be finer than the
      timeline or every trial of a recording lands in one split.

    ``by=["timeline", "start"]`` covers both. Rows sharing all ``by`` values get the same
    split, so a per-trial timeline's rows (all at ``start=0``) stay together while the
    trials of a shared timeline, each at its own start, split individually. Unlike
    ``SklearnSplit(split_by="timeline")``, this never forces a whole recording into one
    bucket.

    ``event_type`` scopes which rows count as trials.

    Uses neuralset's ``DeterministicSplitter`` (a stable hash of the key), so the assignment is
    independent of row order and reproducible across runs.
    """

    by: list[str] = ["timeline", "start"]
    ratios: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}
    seed: float = 0.0
    dest: str = "split"
    event_type: str = ""

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        needed = list(self.by) + (["type"] if self.event_type else [])
        missing = [c for c in needed if c not in events.columns]
        if missing:
            raise KeyError(f"{type(self).__name__}: no column(s) {missing} to key on")
        chosen = (
            events["type"] == self.event_type
            if self.event_type
            else pd.Series(True, index=events.index)
        )
        out = events.copy()
        out[self.dest] = ""
        if not chosen.any():
            return out
        splitter = DeterministicSplitter(ratios=self.ratios, seed=self.seed)
        key = events.loc[chosen, self.by].astype(str).agg("\x1f".join, axis=1)
        cache = {k: splitter(k) for k in key.unique()}
        out.loc[chosen, self.dest] = key.map(cache)
        return out


class AssignDataLayouts(ns.EventsTransform):
    """Assign collection roles and validate explicit model-facing data layouts.

    ``inputs`` and ``targets`` accept any registered event class. A class present in both
    receives role ``"both"``. Every declared event receives a ``data_layout_key`` built from
    ``group_by`` plus event type. The key can route input adapters, target adapters, output
    heads, and losses; it is not limited to channel-based data.

    Layout identity follows only the columns explicitly named by ``group_by``. Required dense
    Study-authored ``n_features`` and channel-name headers validate a requested merge but do
    not silently split it. A user who wants feature width in the identity can explicitly add
    ``"n_features"`` to ``group_by``. Native frequency remains a source fact because an
    extractor may resample without changing the channel adapter.

    Grouping applicability is resolved per event type. A column missing for every row of one
    type is omitted from that type's identity with one warning; a partially populated column
    raises because it would create ambiguous layouts. Keys encode ``column=value`` components
    so different applicable-column subsets cannot collide positionally.
    """

    inputs: tuple[str, ...]
    targets: tuple[str, ...] = ()
    group_by: tuple[str, ...] = (
        "participant_id",
        MEMBER_COLUMN,
        "session",
    )

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.inputs:
            raise ValueError(
                "AssignDataLayouts.inputs must contain at least one event type"
            )
        for name, values in (("inputs", self.inputs), ("targets", self.targets)):
            if len(values) != len(set(values)):
                raise ValueError(
                    f"AssignDataLayouts.{name} contains duplicates: {values}"
                )
        if not self.group_by:
            raise ValueError(
                "AssignDataLayouts.group_by must contain at least one column"
            )
        if len(self.group_by) != len(set(self.group_by)):
            raise ValueError(
                f"AssignDataLayouts.group_by contains duplicates: {self.group_by}"
            )
        if any(not isinstance(column, str) or not column for column in self.group_by):
            raise TypeError("AssignDataLayouts.group_by must contain non-empty strings")
        if "type" in self.group_by:
            raise ValueError(
                "AssignDataLayouts.group_by must not contain 'type'; it is added automatically"
            )
        derived = {DATA_LAYOUT_KEY, DATA_ROLE}
        if overlap := derived.intersection(self.group_by):
            raise ValueError(
                "AssignDataLayouts.group_by cannot use derived columns: "
                f"{sorted(overlap)}"
            )
        for event_type in set(self.inputs) | set(self.targets):
            _event_class(event_type)

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        missing = {"timeline", "type"} - set(events.columns)
        if missing:
            raise KeyError(
                f"AssignDataLayouts events are missing columns: {sorted(missing)}"
            )
        if events.empty:
            raise ValueError("AssignDataLayouts cannot annotate an empty events frame")

        out = events.copy()
        out[DATA_LAYOUT_KEY] = ""
        out[DATA_ROLE] = ""

        roles = self._roles()
        selected = out["type"].astype(str).isin(roles)
        if not selected.any():
            raise ValueError(
                "AssignDataLayouts found none of its declared event types: "
                f"{sorted(roles)}"
            )
        out.loc[selected, DATA_ROLE] = out.loc[selected, "type"].astype(str).map(roles)

        dense_types = {
            event_type
            for event_type in roles
            if _is_dense_event_class(_event_class(event_type))
        }
        keys: dict[str, tuple[tuple[str, str], ...]] = {}
        event_types = out.loc[selected, "type"].astype(str).drop_duplicates()
        for event_type in event_types:
            type_rows = out.loc[selected & out["type"].astype(str).eq(event_type)]
            applicable = self._applicable_group_by(type_rows, event_type)
            grouped: dict[tuple[tuple[str, str], ...], list[tp.Any]] = {}
            for index, row in type_rows.iterrows():
                identity = tuple(
                    (column, str(row[column])) for column in applicable
                ) + (("type", event_type),)
                grouped.setdefault(identity, []).append(index)

            for identity, indexes in grouped.items():
                rows = out.loc[indexes]
                key = _data_layout_key(identity)
                previous = keys.setdefault(key, identity)
                if previous != identity:
                    raise ValueError(
                        f"data layout key collision between {previous!r} and {identity!r}"
                    )
                if event_type in dense_types:
                    _validate_layout_features(rows, event_type=event_type, key=key)
                out.loc[indexes, DATA_LAYOUT_KEY] = key

        self._validate_assigned(out, selected)
        return out

    def _applicable_group_by(
        self,
        rows: pd.DataFrame,
        event_type: str,
    ) -> tuple[str, ...]:
        applicable: list[str] = []
        for column in self.group_by:
            values = (
                rows[column]
                if column in rows.columns
                else pd.Series(pd.NA, index=rows.index, dtype=object)
            )
            missing = values.isna() | values.astype(str).eq("")
            if missing.all():
                warnings.warn(
                    f"AssignDataLayouts omitted group_by column {column!r} for event "
                    f"type {event_type!r} because all values are missing",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            if missing.any():
                raise ValueError(
                    f"AssignDataLayouts group_by column {column!r} is only partially "
                    f"populated for event type {event_type!r}: "
                    f"{int(missing.sum())} of {len(rows)} row(s) are missing"
                )
            applicable.append(column)
        return tuple(applicable)

    def _roles(self) -> dict[str, str]:
        inputs, targets = set(self.inputs), set(self.targets)
        return {
            event_type: (
                "both"
                if event_type in inputs and event_type in targets
                else "input"
                if event_type in inputs
                else "target"
            )
            for event_type in inputs | targets
        }

    def _validate_assigned(
        self,
        events: pd.DataFrame,
        selected: pd.Series,
    ) -> None:
        assigned = events.loc[selected]
        if (
            assigned[DATA_ROLE].isna().any()
            or assigned[DATA_ROLE].astype(str).eq("").any()
        ):
            raise ValueError("AssignDataLayouts left declared rows without data_role")
        missing_layout = assigned[DATA_LAYOUT_KEY].isna() | assigned[
            DATA_LAYOUT_KEY
        ].astype(str).eq("")
        if missing_layout.any():
            raise ValueError(
                "AssignDataLayouts left declared rows without data_layout_key"
            )

        for key, rows in assigned.groupby(DATA_LAYOUT_KEY, sort=False):
            for column in ("type", DATA_ROLE):
                if rows[column].nunique(dropna=False) != 1:
                    raise ValueError(
                        f"data layout {key!r} has conflicting {column} values"
                    )

        recording_columns = ["timeline"]
        if MEMBER_COLUMN in events.columns:
            recording_columns.insert(0, MEMBER_COLUMN)
        recordings = {
            tuple(str(value) for value in values)
            for values in events.loc[:, recording_columns].itertuples(
                index=False, name=None
            )
        }
        input_rows = events[DATA_ROLE].isin(("input", "both"))
        with_input = {
            tuple(str(value) for value in values)
            for values in events.loc[input_rows, recording_columns].itertuples(
                index=False, name=None
            )
        }
        if missing := sorted(recordings - with_input):
            preview = missing[:3]
            suffix = " ..." if len(missing) > len(preview) else ""
            raise ValueError(
                f"{len(missing)} recording group(s) have no declared input data: "
                f"{preview!r}{suffix}"
            )


class RescaleNeural(ns.EventsTransform):
    """Rewrite dense neural rows so they read rescaled, without touching data.

    Each row in ``event_types`` gets its loader replaced by one that fits
    per-channel ``(mean, std)`` over pooled stat bins and divides at read
    time, in the Study's ``_load_rescaled`` (:mod:`neuralshed.scaling`).

    - ``stat_row_query`` is a pandas mask selecting the sibling rows whose
      ``[start, start + duration)`` windows feed the statistics. Empty fits the whole stored array.
    - ``trim_leading_bins`` drops k leading bins from every stat window. Default ``0``.
    - ``guard`` picks the zero-std treatment: ``"eps"`` or ``"zero_to_one"``;
    formulas in :mod:`neuralshed.scaling`.

    Statistics are per timeline, pooled from that timeline's own rows.
    """

    event_types: tuple[str, ...] = ("SpikeCounts", "BandPower")
    stat_row_query: str = ""
    trim_leading_bins: int = 0
    guard: tp.Literal["eps", "zero_to_one"] = "eps"

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.event_types:
            raise ValueError(f"{type(self).__name__}.event_types must be non-empty")
        if self.trim_leading_bins < 0:
            raise ValueError(
                f"{type(self).__name__}.trim_leading_bins must be >= 0, got "
                f"{self.trim_leading_bins}"
            )

    def _run(self, events: pd.DataFrame) -> pd.DataFrame:
        from neuralshed import scaling

        missing = {
            "timeline",
            "type",
            "start",
            "duration",
            "frequency",
            "filepath",
        } - set(events.columns)
        if missing:
            raise KeyError(
                f"{type(self).__name__}: events are missing columns: {sorted(missing)}"
            )

        out = events.copy()
        rewritten = 0
        for timeline, frame in events.groupby("timeline", sort=False):
            windows = self._stat_windows(timeline, frame)
            for index in frame.index[frame["type"].astype(str).isin(self.event_types)]:
                raw = str(out.at[index, "filepath"])
                try:
                    json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{type(self).__name__}: {frame.at[index, 'type']} stores its "
                        f"array at {raw!r}; rescaling needs a SpecialLoader-backed "
                        "stream"
                    ) from error
                rate = float(out.at[index, "frequency"])
                n_bins = int(round(float(out.at[index, "duration"]) * rate))
                rows = [
                    [
                        int(round(a * rate)) + self.trim_leading_bins,
                        int(round(b * rate)),
                    ]
                    for a, b in windows
                ]
                if not scaling.pooled_columns(n_bins, rows).size:
                    raise ValueError(
                        f"{type(self).__name__}: timeline {timeline!r} has no stat bins "
                        f"in range for query {self.stat_row_query!r} "
                        f"(trim_leading_bins={self.trim_leading_bins})"
                    )
                out.at[index, "filepath"] = scaling.rescaled_loader_json(
                    raw, stat_rows=rows, guard=self.guard
                )
                rewritten += 1
        if not rewritten:
            raise ValueError(
                f"{type(self).__name__}: no rows of types {list(self.event_types)} to "
                "rescale in this frame"
            )
        return out

    def _stat_windows(
        self, timeline: tp.Any, frame: pd.DataFrame
    ) -> list[tuple[float, float]]:
        """The ``(start, end)`` seconds of every stat window of one timeline.

        An empty query means the whole stored array, encoded as no windows
        for :func:`neuralshed.scaling.pooled_columns`.
        """
        if not self.stat_row_query:
            return []
        try:
            mask = frame.eval(self.stat_row_query, engine="python")
        except Exception as error:
            raise ValueError(
                f"{type(self).__name__}: stat_row_query {self.stat_row_query!r} failed "
                f"on timeline {timeline!r} ({error}); it must be a pandas expression "
                "over existing columns"
            ) from error
        if not isinstance(mask, pd.Series) or mask.dtype != bool:
            raise ValueError(
                f"{type(self).__name__}: stat_row_query {self.stat_row_query!r} must "
                "evaluate to a boolean column, e.g. \"type == 'Sentence'\""
            )
        selected = frame[mask & ~frame["type"].astype(str).isin(self.event_types)]
        if selected.empty:
            raise ValueError(
                f"{type(self).__name__}: stat_row_query {self.stat_row_query!r} matched "
                f"no stat rows on timeline {timeline!r}"
            )
        return [
            (float(row["start"]), float(row["start"]) + float(row["duration"]))
            for _, row in selected.iterrows()
        ]


def _event_class(event_type: str) -> type:
    """Resolve one declared NeuroAI event class."""
    from neuralset.events.etypes import Event

    from neuralshed import events as _events  # noqa: F401 — register local events

    event_class = Event._CLASSES.get(event_type)
    if event_class is None:
        raise KeyError(f"AssignDataLayouts received unknown event type {event_type!r}")
    return event_class


def _is_dense_event_class(event_class: type) -> bool:
    from neuralshed.events import DenseArrayEvent

    return issubclass(event_class, DenseArrayEvent)


def _data_layout_key(identity: tuple[tuple[str, str], ...]) -> str:
    """Human-readable, collision-safe encoding of named layout identity components."""
    return "/".join(
        f"{urllib.parse.quote(column, safe='')}={urllib.parse.quote(value, safe='')}"
        for column, value in identity
    )


def _validate_layout_features(
    rows: pd.DataFrame,
    *,
    event_type: str,
    key: str,
) -> None:
    """Validate known source feature counts/order without deriving layout identity."""
    known_features: set[int] = set()
    if N_FEATURES in rows.columns:
        declared = rows[N_FEATURES].dropna()
        numeric = pd.to_numeric(declared, errors="coerce")
        invalid = numeric.isna() | (numeric <= 0) | (numeric % 1 != 0)
        if invalid.any():
            raise ValueError(
                f"data layout {key!r} ({event_type}) has invalid n_features values"
            )
        known_features = {int(value) for value in numeric}

    headers = [
        names
        for value in rows.get("ch_names", pd.Series(index=rows.index, dtype=object))
        if (names := _channel_names(value)) is not None
    ]
    header_features = {len(names) for names in headers}
    observed_features = known_features | header_features
    if len(observed_features) > 1:
        raise ValueError(
            f"data layout {key!r} ({event_type}) mixes known n_features "
            f"{sorted(observed_features)}; include {N_FEATURES!r} in group_by to split "
            "them explicitly"
        )
    if len(set(headers)) > 1:
        raise ValueError(f"data layout {key!r} ({event_type}) mixes channel orderings")


def _channel_names(value: tp.Any) -> tuple[str, ...] | None:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        return tuple(part for part in value.split(",") if part) or None
    if isinstance(value, (tuple, list, np.ndarray)):
        return tuple(str(part) for part in value) or None
    return None


def valid_targets(events: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """The rows :class:`ValidateTarget` marked, optionally restricted to one split.

    A missing flag means *not a target*, and has to be spelled out: ``bool(np.nan)`` is
    ``True``, and ``np.nan`` is what a member without its own :class:`ValidateTarget`
    contributes once :class:`CombineStudies` concatenates the frames. Reading the flag
    directly selected that member's rows as targets while correctly rejecting the rows a
    member had validated and refused.
    """
    if TARGET_FLAG not in events.columns:
        raise KeyError(
            f"events have no {TARGET_FLAG!r} column; the collection is missing ValidateTarget"
        )
    column = events[TARGET_FLAG]
    out = events[column.where(column.notna(), False).astype(bool)]
    if split is not None:
        if SPLIT_COLUMN not in out.columns:
            raise KeyError(
                f"events have no {SPLIT_COLUMN!r} column; the collection is missing a split "
                "transform (e.g. neuralset's SklearnSplit)"
            )
        out = out[out[SPLIT_COLUMN].astype(str) == split]
    return out


def member_of(row: tp.Any, default: str = "?") -> str:
    """A row's collection member, falling back to its study class when the column is absent
    (an events frame built outside :class:`CombineStudies`)."""
    for field in (MEMBER_COLUMN, "study"):
        value = getattr(row, field, None) if not hasattr(row, "get") else row.get(field)
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            return str(value)
    return default


__all__ = [
    "CombineStudies",
    "DummyTransform",
    "StandardizeText",
    "ValidateTarget",
    "AssignSourceSplit",
    "AssignHashSplit",
    "AssignDataLayouts",
    "RescaleNeural",
    "valid_targets",
    "member_of",
    "MEMBER_COLUMN",
    "TARGET_FLAG",
    "TARGET_TEXT",
    "SPLIT_COLUMN",
    "SPLITS",
    "DATA_LAYOUT_KEY",
    "DATA_ROLE",
    "N_FEATURES",
]
