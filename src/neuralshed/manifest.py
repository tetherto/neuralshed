"""What a collection is made of, resolved from its events.

A collection is a pile of recordings from different labs, people and machines. Before any model
can be built over it, some boring questions have to be answered for every recording in it: how
many channels, which streams, at what rate, from whom. The manifest is the written answer, one
row per **stream**.

It exists because ``nn.Linear(256, dim)`` needs the 256 before the layer is created, and there
is no single answer: card_2024 is 256 channels wide and fan_2023 is 192, both feeding one trunk.
Its second job matters more. The data side stamps every trial with a layout key and the model
side builds one adapter per key from this same table, so the two cannot drift apart, which is
what lets a read-in refuse an unknown key instead of quietly routing to the wrong adapter.

Nothing here is model-specific or task-specific. Turning a manifest into one model's layer
shapes belongs to that model; deciding whether a target is decodable belongs to the task.

**Declared versus resolved.** A manifest holds only *resolved* facts, the ones nobody chose:
this session has 256 spike channels at 50 Hz. Facts somebody *decided* (which members, which
event classes, which key template, whether a member arrives already normalized) live in the
recipe and are inputs to :meth:`CollectionManifest.from_events`, not fields on its output. The
boundary is what makes the manifest disposable: it can always be rebuilt, and a stale one is a
cache miss rather than a lost decision.

**One layout is one stream.** ``event_type`` is part of the layout key, so a recording carrying
spike counts and band power produces two rows, each with its own channel count and rate. A
recording with no band power simply has no such row. Absence is the representation, so a missing
stream can never be confused with a block of real zeros.

**Key states intent, signature states legality.** The key template says how you *want*
recordings grouped (``session`` optional, so one adapter can span a participant's whole study).
Whether that grouping is *allowed* is a separate question, answered by :meth:`conflicts` over
the things one adapter cannot represent: channel count, effective rate, stream kind, channel
ordering.

    manifest = CollectionManifest.from_events(
        events, inputs=("SpikeCounts", "BandPower"), targets=("Sentence",)
    )
"""

import dataclasses
import hashlib
import json
import typing as tp

import pandas as pd

from neuralshed.transforms import MEMBER_COLUMN

#: Bumped when resolution logic changes in a way that makes older stored manifests wrong.
#: A reader whose version is newer than a stored blob's rebuilds rather than trusting it.
BUILDER_VERSION = 1

Kind = tp.Literal["data", "text", "categorical"]
Role = tp.Literal["input", "target", "both"]

#: Key components a template may use. ``participant`` and ``event_type`` are mandatory: dropping
#: the first merges two people into one adapter, dropping the second breaks one-layout-one-stream.
KEY_COMPONENTS = ("participant", "study", "member", "session", "event_type")
REQUIRED_KEY_COMPONENTS = ("participant", "event_type")

#: Coarse to fine, so a sorted manifest reads as related rows sitting together.
DEFAULT_KEY_TEMPLATE = ("participant", "study", "session", "event_type")

#: What identifies one recording, i.e. which streams co-occur and can be assembled into a trial.
DEFAULT_GROUP_BY = ("member", "session")


@dataclasses.dataclass(frozen=True)
class LayoutSpec:
    """One stream of one recording (or of several, when the key merges them).

    ``channels`` and ``native_rate`` are ``None`` for anything that is not a dense array. A
    ``Sentence`` target has neither, and asking a text stream for its channel count is a
    category error rather than a missing value.
    """

    layout_key: str
    group_key: str  # only meaningful when the key is at least as fine as the grouping
    event_type: str
    kind: Kind
    role: Role
    participant: str
    study: str
    member: str
    groups: tuple[str, ...]  # every recording this layout serves
    channels: int | None = None
    native_rate: float | None = None
    ch_names: tuple[str, ...] | None = None
    region: str | None = None  # anatomy, resolved later

    @property
    def is_array(self) -> bool:
        """Whether this stream has a channel count and a rate at all."""
        return self.kind == "data"

    def effective_rate(self, target_rate: float | None) -> float | None:
        """The rate this stream is seen at, which is the target rate once one is declared.

        Before modality-aware resampling exists nobody declares one, so this is the native rate
        and mismatched rates block a merge. Once resampling is real a declared target makes
        this uniform by construction and the check goes quietly vacuous. Same rule either side.
        """
        return target_rate if target_rate is not None else self.native_rate


class CollectionManifest:
    """The layout table for a collection, plus the lookups models and extraction need.

    Build with :meth:`from_events`. Integer indices are deliberately **not** fields on
    :class:`LayoutSpec` and are not serialized: they are positions in a sorted list, so a
    filtered manifest would renumber them and a checkpoint's "embedding row 2" would stop
    meaning the same thing. Names are the identity; :attr:`layout_index` and friends are a
    per-run view each model records for itself.
    """

    def __init__(
        self,
        layouts: tp.Sequence[LayoutSpec],
        *,
        key_template: tp.Sequence[str] = DEFAULT_KEY_TEMPLATE,
        group_by: tp.Sequence[str] = DEFAULT_GROUP_BY,
        target_rate: float | None = None,
        conflicts: tp.Sequence[str] = (),
        warnings: tp.Sequence[str] = (),
        recipe_digest: str | None = None,
        built_at: str | None = None,
        builder_version: int = BUILDER_VERSION,
    ):
        if not layouts:
            raise ValueError("a manifest needs at least one layout")
        self.layouts = sorted(layouts, key=lambda s: s.layout_key)
        self.key_template = tuple(key_template)
        self.group_by = tuple(group_by)
        self.target_rate = target_rate
        self._conflicts = list(conflicts)
        self._warnings = list(warnings)
        self.recipe_digest = recipe_digest
        self.built_at = built_at
        self.builder_version = builder_version

        self.by_key = {s.layout_key: s for s in self.layouts}
        #: ``(group_key, event_type) -> layout_key``, how a trial finds the adapter for a stream.
        self.routing = {
            (group, s.event_type): s.layout_key
            for s in self.layouts
            for group in s.groups
        }
        self.participants = sorted({s.participant for s in self.layouts})
        self.studies = sorted({s.study for s in self.layouts})
        self.members = sorted({s.member for s in self.layouts})
        self.regions = sorted({s.region for s in self.layouts if s.region})
        self.groups = sorted({g for s in self.layouts for g in s.groups})

    def __len__(self) -> int:
        return len(self.layouts)

    def __repr__(self) -> str:
        return (
            f"CollectionManifest({len(self.layouts)} layouts over {len(self.groups)} "
            f"recordings, {len(self.participants)} participants, "
            f"inputs={self.inputs}, targets={self.targets})"
        )

    # -- views ---------------------------------------------------------------
    @property
    def inputs(self) -> list[str]:
        """Event classes this collection reads, in sorted order."""
        return sorted(
            {s.event_type for s in self.layouts if s.role in ("input", "both")}
        )

    @property
    def targets(self) -> list[str]:
        return sorted(
            {s.event_type for s in self.layouts if s.role in ("target", "both")}
        )

    @property
    def array_layouts(self) -> list[LayoutSpec]:
        """Input layouts a read-in has to build a projection for."""
        return [s for s in self.layouts if s.is_array and s.role in ("input", "both")]

    @property
    def layout_index(self) -> dict[str, int]:
        return {s.layout_key: i for i, s in enumerate(self.array_layouts)}

    @property
    def participant_index(self) -> dict[str, int]:
        return {p: i for i, p in enumerate(self.participants)}

    @property
    def region_index(self) -> dict[str, int]:
        return {r: i for i, r in enumerate(self.regions)}

    @property
    def n_participants(self) -> int:
        return len(self.participants)

    @property
    def n_regions(self) -> int:
        return len(self.regions)

    def layout_for(self, group_key: str, event_type: str) -> LayoutSpec:
        """The layout serving one stream of one recording."""
        key = self.routing.get((group_key, event_type))
        if key is None:
            raise KeyError(
                f"no layout for {event_type!r} in recording {group_key!r} "
                f"({len(self.groups)} recordings known)"
            )
        return self.by_key[key]

    def streams_of(self, group_key: str) -> dict[str, LayoutSpec]:
        """Every stream one recording carries, keyed by event class."""
        return {
            event: self.by_key[key]
            for (group, event), key in self.routing.items()
            if group == group_key
        }

    def conflicts(self) -> list[str]:
        """Layouts whose own recordings disagreed, one message each.

        Empty by construction while ``session`` is in the key. Non-empty means a coarser key
        merged recordings differing in something one adapter cannot represent.
        """
        return list(self._conflicts)

    def warnings(self) -> list[str]:
        """Merges that could not be *verified*, as opposed to merges known to be wrong."""
        return list(self._warnings)

    def recordings_missing_targets(self) -> list[str]:
        """Recordings carrying input streams but no target stream.

        These contribute nothing to training. Without this they surface as trials that
        mysteriously fail to appear during extraction, after the expensive part has run.
        """
        out = []
        for group in self.groups:
            streams = self.streams_of(group)
            has_input = any(s.role in ("input", "both") for s in streams.values())
            has_target = any(s.role in ("target", "both") for s in streams.values())
            if has_input and not has_target:
                out.append(group)
        return sorted(out)

    def filter(
        self,
        *,
        participants: tp.Iterable[str] | None = None,
        members: tp.Iterable[str] | None = None,
        groups: tp.Iterable[str] | None = None,
    ) -> "CollectionManifest":
        """A manifest over part of this one, keeping every layout key unchanged.

        Keys are preserved rather than recomputed, so an isolated run and a pooled run refer to
        the same layouts by the same names and their results can be compared. Only the derived
        integer views differ, which is exactly why they are derived.
        """
        keep_p = set(participants) if participants is not None else None
        keep_m = set(members) if members is not None else None
        keep_g = set(groups) if groups is not None else None

        selected = []
        for s in self.layouts:
            if keep_p is not None and s.participant not in keep_p:
                continue
            if keep_m is not None and s.member not in keep_m:
                continue
            kept_groups = tuple(g for g in s.groups if keep_g is None or g in keep_g)
            if not kept_groups:
                continue
            selected.append(dataclasses.replace(s, groups=kept_groups))
        if not selected:
            raise ValueError("filter selected no layouts")
        return CollectionManifest(
            selected,
            key_template=self.key_template,
            group_by=self.group_by,
            target_rate=self.target_rate,
            conflicts=self._conflicts,
            warnings=self._warnings,
            recipe_digest=self.recipe_digest,
            built_at=self.built_at,
            builder_version=self.builder_version,
        )

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict:
        """The stored form. Plain JSON types only, and no derived integer indices."""
        return {
            "builder_version": self.builder_version,
            "key_template": list(self.key_template),
            "group_by": list(self.group_by),
            "target_rate": self.target_rate,
            "recipe_digest": self.recipe_digest,
            "built_at": self.built_at,
            "conflicts": list(self._conflicts),
            "warnings": list(self._warnings),
            "layouts": [_layout_to_dict(s) for s in self.layouts],
        }

    @classmethod
    def from_dict(cls, blob: dict) -> "CollectionManifest":
        version = int(blob.get("builder_version", 0))
        if version != BUILDER_VERSION:
            raise ValueError(
                f"manifest was built by version {version}, this code is {BUILDER_VERSION}; "
                "rebuild it rather than trusting a blob produced under different rules"
            )
        return cls(
            [_layout_from_dict(d) for d in blob["layouts"]],
            key_template=blob.get("key_template", DEFAULT_KEY_TEMPLATE),
            group_by=blob.get("group_by", DEFAULT_GROUP_BY),
            target_rate=blob.get("target_rate"),
            conflicts=blob.get("conflicts", []),
            warnings=blob.get("warnings", []),
            recipe_digest=blob.get("recipe_digest"),
            built_at=blob.get("built_at"),
            builder_version=version,
        )

    def to_frame(self) -> pd.DataFrame:
        """The manifest as a table, which is how it is meant to be read."""
        return pd.DataFrame([_layout_to_dict(s) for s in self.layouts])

    # -- construction --------------------------------------------------------
    @classmethod
    def from_events(
        cls,
        events: pd.DataFrame,
        *,
        inputs: tp.Sequence[str],
        targets: tp.Sequence[str] = (),
        key_template: tp.Sequence[str] = DEFAULT_KEY_TEMPLATE,
        group_by: tp.Sequence[str] = DEFAULT_GROUP_BY,
        target_rate: float | None = None,
        participants: tp.Mapping[str, str] | None = None,
        regions: tp.Mapping[str, str] | None = None,
        recipe_digest: str | None = None,
        built_at: str | None = None,
    ) -> "CollectionManifest":
        """Resolve a collection's layouts from the events its chain produced.

        ``inputs`` and ``targets`` are neuroai event class names, the vocabulary the catalog
        already uses. Inputs are *wanted-if-present*, so a member carrying only ``SpikeCounts``
        joins a ``(SpikeCounts, BandPower)`` collection with one input layout instead of being
        rejected. Targets are required, and a recording missing one is reported by
        :meth:`recordings_missing_targets` rather than silently dropped.

        ``participants`` maps a member to a participant id, which is what merges one person
        across datasets. Without it, participant falls back to the event's own ``subject``,
        already namespaced by study class.

        Layout comes from the events, never from the catalog's per-dataset channel count, which
        is documented as *the predominant stored layout*. That summary is wrong for jude_2026_speech
        (256- and 384-channel sessions whose column-to-area mapping differs) and kunz_2025 (128
        and 256 channels, four bin sizes, one Study), and wrong in a way that raises nothing.
        """
        template = _check_template(key_template)
        wanted = _roles(inputs, targets)
        participants = dict(participants or {})
        regions = dict(regions or {})

        rows = _resolve_rows(events, wanted, participants, regions, group_by)
        if not rows:
            raise ValueError(
                f"no recording carries any of the declared streams {sorted(wanted)}; "
                "either the collection's members are wrong or the declared event classes are"
            )

        keyed: dict[str, list[dict]] = {}
        for row in rows:
            keyed.setdefault(_layout_key(row, template), []).append(row)

        layouts, conflicts, warnings = [], [], []
        for key, group_rows in sorted(keyed.items()):
            head = group_rows[0]
            groups = tuple(sorted({r["group_key"] for r in group_rows}))
            disagreements = _disagreements(group_rows, target_rate)
            if disagreements:
                conflicts.append(
                    f"layout {key!r} merges {len(groups)} recordings that disagree on "
                    + "; ".join(disagreements)
                )
            # Only dense streams have an ordering to disagree about. A text target merging
            # across recordings is not an unverified merge, it is a stream with no columns.
            if (
                head["kind"] == "data"
                and len(groups) > 1
                and not any(r["ch_names"] for r in group_rows)
            ):
                warnings.append(
                    f"layout {key!r} merges {len(groups)} recordings whose channel ordering "
                    "could not be verified: no stream carries a ch_names header, so a "
                    "reordering between them would not be detected"
                )
            layouts.append(
                LayoutSpec(
                    layout_key=key,
                    group_key=head["group_key"] if len(groups) == 1 else key,
                    event_type=head["event_type"],
                    kind=head["kind"],
                    role=head["role"],
                    participant=head["participant"],
                    study=head["study"],
                    member=head["member"],
                    groups=groups,
                    channels=head["channels"],
                    native_rate=head["native_rate"],
                    ch_names=head["ch_names"],
                    region=head["region"],
                )
            )
        return cls(
            layouts,
            key_template=template,
            group_by=group_by,
            target_rate=target_rate,
            conflicts=conflicts,
            warnings=warnings,
            recipe_digest=recipe_digest,
            built_at=built_at,
        )


# ===========================================================================
# Recipe identity
# ===========================================================================


def recipe_digest(**parts: tp.Any) -> str:
    """A stable digest of everything that changes what a manifest would resolve to.

    This is the *only* staleness signal a stored manifest has, so it must cover the member
    chains and their queries, the declared event classes, the key template, the grouping and
    the target rate. Leave the key template out and a manifest keyed one way can be used under
    a recipe that now says another, with nothing complaining.

    It deliberately does not cover the data on disk. Sessions added after a build are not
    noticed, and picking them up needs an explicit rebuild.
    """
    canonical = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ===========================================================================
# Resolution internals
# ===========================================================================


def _check_template(template: tp.Sequence[str]) -> tuple[str, ...]:
    out = tuple(template)
    unknown = [c for c in out if c not in KEY_COMPONENTS]
    if unknown:
        raise ValueError(
            f"unknown key components {unknown}; have {list(KEY_COMPONENTS)}"
        )
    missing = [c for c in REQUIRED_KEY_COMPONENTS if c not in out]
    if missing:
        raise ValueError(
            f"key template is missing {missing}. 'participant' is required or two people "
            "would share one adapter; 'event_type' is required or one layout would cover "
            "several streams with different channel counts"
        )
    return out


def _roles(inputs: tp.Sequence[str], targets: tp.Sequence[str]) -> dict[str, Role]:
    """Event class -> the role this collection gives it."""
    out: dict[str, Role] = {}
    for name in inputs:
        out[name] = "input"
    for name in targets:
        out[name] = "both" if name in out else "target"
    if not out:
        raise ValueError("a collection must declare at least one input event class")
    return out


def kind_of(event_type: str) -> Kind:
    """Whether a stream is a dense array, text, or a categorical marker.

    Read off the neuroai class hierarchy, which is the same distinction the catalog stores as
    ``event_types.category``. Deriving it here keeps the manifest free of a database.
    """
    from neuralset.events.etypes import (
        BaseDataEvent,
        BaseText,
        CategoricalEvent,
        Event,
    )

    from neuralshed import events as _events  # noqa: F401 — registers our classes

    cls = Event._CLASSES.get(event_type)
    if cls is None:
        raise KeyError(
            f"unknown event class {event_type!r}; it is not registered with neuroai, so the "
            "collection declares a stream nothing can produce"
        )
    if issubclass(cls, BaseDataEvent):
        return "data"
    if issubclass(cls, BaseText):
        return "text"
    if issubclass(cls, CategoricalEvent):
        return "categorical"
    raise TypeError(f"{event_type!r} is not a data, text or categorical event")


def _resolve_rows(
    events: pd.DataFrame,
    wanted: dict[str, Role],
    participants: tp.Mapping[str, str],
    regions: tp.Mapping[str, str],
    group_by: tp.Sequence[str],
) -> list[dict]:
    """One dict per (recording, declared stream) actually present in the events."""
    kinds = {name: kind_of(name) for name in wanted}
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for _, group in events.groupby(events["timeline"].astype(str)):
        member = _one(group, MEMBER_COLUMN, default=_one(group, "study", "?"))
        study = str(_one(group, "study", default=member))
        session = str(_one(group, "session", default="?"))
        subject = str(_one(group, "subject", default=member))
        context = {"member": member, "study": study, "session": session}
        group_key = "/".join(str(context[c]) for c in group_by)

        for event_type, role in wanted.items():
            streams = group[group["type"].astype(str) == event_type]
            if streams.empty:
                continue
            # One row per recording, not per event: a willet_2023 block timeline holds tens
            # of Sentence events and they describe one stream between them.
            if (group_key, event_type) in seen:
                continue
            seen.add((group_key, event_type))

            channels = rate = names = None
            if kinds[event_type] == "data":
                channels, names = _array_layout(streams, event_type)
                rate = float(streams["frequency"].astype(float).iloc[0])
            rows.append(
                {
                    "group_key": group_key,
                    "event_type": event_type,
                    "kind": kinds[event_type],
                    "role": role,
                    "participant": participants.get(member, subject),
                    "study": study,
                    "member": member,
                    "session": session,
                    "channels": channels,
                    "native_rate": rate,
                    "ch_names": names,
                    "region": regions.get(member),
                }
            )
    return rows


def _array_layout(
    streams: pd.DataFrame, event_type: str
) -> tuple[int, tuple[str, ...] | None]:
    """Channel count and, when stored, channel ordering for one dense stream.

    The count comes from the ``ch_names`` header when present and otherwise from reading the
    array, which is why it reflects the real layout rather than the catalog's summary. Ordering
    rides along because two recordings can share a count while column *k* is a different
    electrode in each. ``None`` means the header was absent, which is currently the norm.
    """
    resolved = [_channels_of(row) for _, row in streams.iterrows()]
    counts = {n for n, _ in resolved if n}
    if not counts:
        raise ValueError(f"{event_type} streams report no channels")
    if len(counts) > 1:
        raise ValueError(
            f"one recording has {event_type} streams of different widths: {sorted(counts)}"
        )
    names = {n for _, n in resolved if n is not None}
    return counts.pop(), (names.pop() if len(names) == 1 else None)


def _channels_of(row: pd.Series) -> tuple[int, tuple[str, ...] | None]:
    header = row.get("ch_names")
    if isinstance(header, str) and header:
        names = tuple(header.split(","))
        return len(names), names
    from neuralset.events.etypes import Event

    event = Event.from_dict({**row.to_dict(), "type": row["type"]})
    return int(event.read().shape[0]), None


def _layout_key(row: dict, template: tp.Sequence[str]) -> str:
    return "/".join(str(row[component]) for component in template)


#: What two recordings must agree on to share one layout, beyond channel ordering. ``region`` is
#: here because a layout carries one, so merging across regions would mislabel the second.
_SIGNATURE_FIELDS = ("channels", "kind", "region")


def _disagreements(group_rows: list[dict], target_rate: float | None) -> list[str]:
    """Why these recordings cannot share one layout. Empty when they can."""
    out = []
    for field in _SIGNATURE_FIELDS:
        values = {r[field] for r in group_rows}
        if len(values) > 1:
            out.append(f"{field} {sorted(str(v) for v in values)}")
    if target_rate is None:
        rates = {r["native_rate"] for r in group_rows}
        if len(rates) > 1:
            out.append(
                f"native rate {sorted(str(r) for r in rates)} with no target rate declared "
                "to reconcile them"
            )
    # Only compare orderings that exist. A recording whose width came from reading the array
    # has none to offer, and calling that a disagreement would reject merely undocumented data.
    known = {r["ch_names"] for r in group_rows if r["ch_names"]}
    if len(known) > 1:
        out.append(f"{len(known)} distinct channel orderings")
    return out


def _one(group: pd.DataFrame, column: str, default: tp.Any = None) -> tp.Any:
    """The single value of ``column`` across a recording, or ``default`` when absent.

    Raises on a genuine disagreement: a timeline spanning two members or two subjects is a
    broken frame, and silently taking the first row would hide it.
    """
    if column not in group.columns:
        return default
    values = {
        v
        for v in group[column].tolist()
        if v is not None and not (isinstance(v, float) and pd.isna(v)) and v != ""
    }
    if not values:
        return default
    if len(values) > 1:
        raise ValueError(
            f"recording has conflicting {column!r} values: {sorted(values)}"
        )
    return values.pop()


def _layout_to_dict(spec: LayoutSpec) -> dict:
    out = dataclasses.asdict(spec)
    out["groups"] = list(spec.groups)
    out["ch_names"] = list(spec.ch_names) if spec.ch_names else None
    return out


def _layout_from_dict(blob: dict) -> LayoutSpec:
    data = dict(blob)
    data["groups"] = tuple(data.get("groups", ()))
    names = data.get("ch_names")
    data["ch_names"] = tuple(names) if names else None
    return LayoutSpec(**data)


__all__ = [
    "BUILDER_VERSION",
    "DEFAULT_GROUP_BY",
    "DEFAULT_KEY_TEMPLATE",
    "KEY_COMPONENTS",
    "CollectionManifest",
    "LayoutSpec",
    "kind_of",
    "recipe_digest",
]
