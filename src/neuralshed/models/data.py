"""Seed collection recipes, and the sampler both decoders batch with."""

import typing as tp

import numpy as np
from torch.utils.data import Sampler

import neuralset as ns
from neuralset.events.transforms import QueryEvents
from neuralshed import events as _events  # noqa: F401 — registers custom events
from neuralshed.collection import Collection
from neuralshed.datasets.utils import dataset_dir
from neuralshed.transforms import (
    AssignDataLayouts,
    AssignHashSplit,
    CombineStudies,
    DeriveStream,
    StandardizeText,
    ValidateTarget,
)

# ===========================================================================
# Seed collections
# ===========================================================================

#: Member key -> (study class, ``neuralshed.datasets`` subpackage, drive-relative path).
#: Importing the subpackage is what registers the study class with neuralset.
SPEECH_TEXT_MEMBERS: dict[str, tuple[str, str, str]] = {
    "card_2024": ("Card2024Brain2Text", "card_2024", "card_2024/data"),
    "willet_2023": ("Willett2023Sentences", "willet_2023", "willet_2023/data"),
    "fan_2023": ("Fan2023Handwriting", "fan_2023", "fan_2023/data"),
}

#: What the seed collection is *for*: the event classes it reads and the one it decodes.
#: Inputs are wanted-if-present (fan_2023 has no band power and still joins), targets required.
SPEECH_TEXT_INPUTS: tuple[str, ...] = ("SpikeCounts", "BandPower")
SPEECH_TEXT_TARGETS: tuple[str, ...] = ("Sentence",)

#: Coarse implant site per member, for the region embedding.
SPEECH_TEXT_REGIONS: dict[str, str] = {
    "card_2024": "speech_precentral",
    "willet_2023": "speech_precentral",
    "fan_2023": "handknob_motor",
}

#: The same person across datasets shares one embedding. Refined in
#: ``neuralshed.datasets.participants``.
SPEECH_TEXT_PARTICIPANTS: dict[str, str] = {
    "card_2024": "braingate2/T15",
    "willet_2023": "braingate2/T12",
    "fan_2023": "braingate2/T5",
}

#: Members arriving already z-scored by the source, which normalization must leave alone.
SPEECH_TEXT_ALREADY_NORMALIZED: frozenset[str] = frozenset({"card_2024"})

#: Member key -> (study class, ``neuralshed.datasets`` subpackage, drive-relative path).
MOTOR_CURSOR_MEMBERS: dict[str, tuple[str, str, str]] = {
    "karpowicz_2025": (
        "Karpowicz2025CursorStability",
        "karpowicz_2025",
        "karpowicz_2025/data",
    ),
    "wilson_2025": (
        "Wilson2025CursorRecalibration",
        "wilson_2025",
        "wilson_2025/data",
    ),
    "singer_2025": ("Singer2025CursorControl", "singer_2025", "singer_2025/data"),
    "deo_2024": ("Deo2024Bimanual", "deo_2024", "deo_2024/data"),
}

#: Event filters one member needs and the others cannot run, because they reference that
#: member's own columns. ``deo_2024`` publishes a right and a left cursor per block, and a
#: 2-channel target is the right one: the arrays are in the left hand knob, where right-hand
#: tuning is the strong one. Rows without an ``effector`` keep it. Open-loop blocks are the
#: ones whose cursor follows the computer's trajectory rather than a decoder. Left-hand-only
#: trials (``code == 2``) stop being triggers, since the right cursor is still during them.
#: The part of deo's filter that holds whatever blocks are being read.
DEO_RIGHT_CURSOR: str = "effector != 'left' and not (type == 'Action' and code == 2)"

#: ``singer_2025`` drops the simultaneous-speech blocks, where he was attempting a word while
#: driving the cursor, so the neural features carry speech the cursor never reflects. The
#: computer-assistance filter the collection also wants is missing: assist lives only inside
#: `CursorAssistSeries`, per bin, and no timeline attribute summarizes it, so it cannot be
#: written as an event query. 30 of the 51 blocks survive this filter and 15 of those are
#: assist-free throughout (the 9 grid blocks and 6 of the 9 day-468 radial8 ones).
MOTOR_CURSOR_MEMBER_QUERIES: dict[str, str] = {
    "deo_2024": f"{DEO_RIGHT_CURSOR} and loop_type == 'open_loop'",
    "singer_2025": "speech_condition == 'none'",
}

#: deo's closed-loop blocks, for the comparison that needs a decoder in the loop. Trials are
#: dropped where the 10 s timeout ran out, which is a failure the release does not label,
#: and ``closed_loop_recal`` stays out because its assistance is never logged.
MOTOR_CURSOR_MEMBER_QUERIES_CLOSED_LOOP: dict[str, str] = {
    "deo_2024": (
        f"{DEO_RIGHT_CURSOR} and loop_type == 'closed_loop'"
        " and not (type == 'Action' and duration > 9.9)"
    ),
}

#: Inputs are wanted-if-present, as in the speech collection. The target is velocity rather
#: than position: it is what these three decode online, and position is its integral, so
#: predicting both scores the same information twice while position carries the session drift.
MOTOR_CURSOR_INPUTS: tuple[str, ...] = ("SpikeCounts", "BandPower")
MOTOR_CURSOR_TARGETS: tuple[str, ...] = ("DecodedCursorVelocitySeries",)

#: The declared targets for the intention variant, a stream `DeriveStream` appends.
#: `DecodedCursorVelocitySeries` is the online decoder's own output, so a model trained on
#: it partly relearns that decoder. The cued target is task-defined and observed, and the
#: unit vector from the cursor to it is dimensionless, so this variant also needs no scale
#: to reconcile the members' units.
MOTOR_CURSOR_INTENT_TARGETS: tuple[str, ...] = ("CursorIntentionSeries",)

#: Derived streams this collection can append, name -> the event types the operation reads in
#: the order it reads them. ``"velocity"`` emits `CursorVelocitySeries`, how the cursor
#: actually moved, which is *not* `DecodedCursorVelocitySeries`, what a decoder commanded:
#: where both exist they differ, so the two keep separate types.
MOTOR_CURSOR_DERIVATIONS: dict[str, tuple[str, ...]] = {
    "intention": ("CursorPositionSeries", "TargetPositionSeries"),
    "velocity": ("CursorPositionSeries",),
    "aligned_velocity": ("CursorPositionSeries", "TargetPositionSeries"),
    "residual_velocity": ("CursorPositionSeries", "TargetPositionSeries"),
}

#: Coarse implant site per member, sharing the speech collection's vocabulary so a combined
#: collection would index the same region embedding.
MOTOR_CURSOR_REGIONS: dict[str, str] = {
    "karpowicz_2025": "handknob_motor",
    "wilson_2025": "handknob_motor",
    "singer_2025": "speech_precentral",
    "deo_2024": "handknob_motor",
}

#: ``deo_2024`` and ``wilson_2025`` are both T5, so they share one participant embedding.
MOTOR_CURSOR_PARTICIPANTS: dict[str, str] = {
    "karpowicz_2025": "braingate2/T11",
    "wilson_2025": "braingate2/T5",
    "singer_2025": "braingate2/T15",
    "deo_2024": "braingate2/T5",
}

#: All three publish raw threshold crossings, so none is exempt from normalization.
MOTOR_CURSOR_ALREADY_NORMALIZED: frozenset[str] = frozenset()

#: One trigger row is one cursor movement. All three members write ``action_kind`` on their
#: ``Action`` rows, so the same condition selects trials in every member.
MOTOR_CURSOR_TRIGGER: str = "type == 'Action' and action_kind == 'cursor_control'"


def build_study(
    member: str,
    query: str | None = None,
    members: tp.Mapping[str, tuple[str, str, str]] = SPEECH_TEXT_MEMBERS,
) -> ns.Step:
    """One member's Study step, built from a member map."""
    if member not in members:
        raise KeyError(f"unknown member {member!r}; have {sorted(members)}")
    import importlib

    study_class, module, path = members[member]
    mod = importlib.import_module(f"neuralshed.datasets.{module}.study")
    return getattr(mod, study_class)(path=dataset_dir(path), query=query)


def speech_text_collection(
    members: tp.Sequence[str] = ("card_2024", "willet_2023"),
    *,
    query: str | None = None,
    ratios: tp.Mapping[str, float] | None = None,
    seed: float = 0.0,
) -> Collection:
    """The neural-to-text collection as a runnable step chain, equivalent to a saved one.

    This is a *seed fixture*, not the source of truth. A collection in the database is, and
    ``database.collections.load_chain`` returns the same shape of object. It exists
    because the saved collections predate the transforms below, so a notebook that wants
    a trainable collection today has to compose one.

    ``query`` is applied to every member, which is how the notebooks keep themselves light
    (``"session_index < 2"``). Splitting keys on ``["timeline", "start"]`` so that a member
    storing one timeline per session still splits per trial.
    """
    return Collection(
        steps={
            "CombineStudies": CombineStudies(
                chains={m: build_study(m, query) for m in members}
            ),
            "StandardizeText": StandardizeText(),
            "ValidateTarget": ValidateTarget(),
            "AssignHashSplit": AssignHashSplit(
                by=["timeline", "start"],
                ratios=dict(ratios or {"train": 0.8, "val": 0.1, "test": 0.1}),
                seed=seed,
            ),
            "AssignDataLayouts": AssignDataLayouts(
                inputs=SPEECH_TEXT_INPUTS,
                targets=SPEECH_TEXT_TARGETS,
            ),
        }
    )


def collection_identities(
    members: tp.Sequence[str] = ("card_2024", "willet_2023"),
) -> dict[str, tp.Any]:
    """The ``regions`` / ``participants`` / ``already_normalized`` arguments for a member set.

    These are per-member facts a collection does not yet carry itself (level 6's recipe
    column), so they have to be passed to ``BitDataBundle.from_collection`` explicitly. Keeping
    them next to the recipe stops two notebooks from disagreeing about who T15 is.
    """
    members = list(members)
    return {
        "regions": {m: SPEECH_TEXT_REGIONS[m] for m in members},
        "participants": {m: SPEECH_TEXT_PARTICIPANTS[m] for m in members},
        "already_normalized": {
            m for m in members if m in SPEECH_TEXT_ALREADY_NORMALIZED
        },
    }


def motor_cursor_collection(
    members: tp.Sequence[str] = ("karpowicz_2025", "wilson_2025", "singer_2025"),
    *,
    query: str | None = None,
    derive: tp.Sequence[str] = (),
    member_queries: tp.Mapping[str, str] | None = None,
    ratios: tp.Mapping[str, float] | None = None,
    split_by: tp.Sequence[str] = ("timeline", "start"),
    seed: float = 0.0,
) -> ns.Chain:
    """The cursor-kinematics collection as a runnable step chain.

    The same shape as :func:`speech_text_collection` with the text steps removed.
    :class:`~neuralshed.transforms.StandardizeText` has nothing to clean when the target
    is a dense array, and :class:`~neuralshed.transforms.ValidateTarget` needs no new
    class: its condition is a pandas expression, so marking cursor trials is a different
    condition rather than a different transform.

    ``query`` is applied to every member's Study. A member listed in
    :data:`MOTOR_CURSOR_MEMBER_QUERIES` also gets its own ``QueryEvents`` step.

    ``split_by`` keys the split. The default splits per trial; ``("timeline",)`` splits whole
    recordings instead, which is the honest test when trials from one block are not
    independent.

    ``member_queries`` replaces :data:`MOTOR_CURSOR_MEMBER_QUERIES` wholesale, for reading a
    different slice of a member (deo's closed-loop blocks rather than its open-loop ones).

    ``derive`` names entries of :data:`MOTOR_CURSOR_DERIVATIONS` to append as streams, after
    the fan-in so every member gets the same treatment. Deriving here rather than at
    extraction is what makes the derived target part of the collection: it shows up in the
    events, in the manifest and in the preflight report like any published stream.
    """
    unknown = [name for name in derive if name not in MOTOR_CURSOR_DERIVATIONS]
    if unknown:
        raise KeyError(
            f"unknown derivation(s) {unknown}; have {sorted(MOTOR_CURSOR_DERIVATIONS)}"
        )

    queries = (
        MOTOR_CURSOR_MEMBER_QUERIES if member_queries is None else dict(member_queries)
    )

    def member_chain(member: str) -> ns.Step:
        study = build_study(member, query, MOTOR_CURSOR_MEMBERS)
        if member not in queries:
            return study
        return ns.Chain(steps=[study, QueryEvents(query=queries[member])])

    return ns.Chain(
        steps={
            "CombineStudies": CombineStudies(
                chains={m: member_chain(m) for m in members}
            ),
            **{
                f"DeriveStream_{name}": DeriveStream(
                    operation=name, sources=MOTOR_CURSOR_DERIVATIONS[name]
                )
                for name in derive
            },
            "ValidateTarget": ValidateTarget(condition=MOTOR_CURSOR_TRIGGER),
            "AssignHashSplit": AssignHashSplit(
                by=list(split_by),
                ratios=dict(ratios or {"train": 0.8, "val": 0.1, "test": 0.1}),
                seed=seed,
            ),
        }
    )


def motor_cursor_identities(
    members: tp.Sequence[str] = ("karpowicz_2025", "wilson_2025", "singer_2025"),
) -> dict[str, tp.Any]:
    """The ``regions`` / ``participants`` / ``already_normalized`` arguments for a member set."""
    members = list(members)
    return {
        "regions": {m: MOTOR_CURSOR_REGIONS[m] for m in members},
        "participants": {m: MOTOR_CURSOR_PARTICIPANTS[m] for m in members},
        "already_normalized": {
            m for m in members if m in MOTOR_CURSOR_ALREADY_NORMALIZED
        },
    }


# ===========================================================================
# Batch sampler
# ===========================================================================


class SessionBatchSampler(Sampler[list[int]]):
    """Yield batches whose items all come from one recording (fixed channel count per batch).

    ``key_attr`` names the trial attribute batches are homogeneous in. Both decoders pass
    ``group_key``, the recording a trial belongs to.

    Grouping by recording is safe but conservative. When a coarse layout key deliberately shares
    one projection across several recordings, those trials *could* batch together and this will
    still keep them apart, so batches stay smaller than they need to be. The eventual fix is to
    key on the trial's set of layout keys, which reduces to this in the default case.
    """

    def __init__(
        self,
        dataset: tp.Sequence,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        key_attr: str = "group_key",
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self.key_attr = key_attr
        items = getattr(dataset, "trials", dataset)
        self.by_session: dict[int, list[int]] = {}
        for i, t in enumerate(items):
            self.by_session.setdefault(getattr(t, key_attr), []).append(i)

    def __iter__(self) -> tp.Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        batches: list[list[int]] = []
        for idxs in self.by_session.values():
            order = rng.permutation(idxs) if self.shuffle else np.array(idxs)
            for i in range(0, len(order), self.batch_size):
                batches.append(order[i : i + self.batch_size].tolist())
        if self.shuffle:
            rng.shuffle(batches)  # interleave sessions across the epoch
        return iter(batches)

    def __len__(self) -> int:
        return sum(
            (len(v) + self.batch_size - 1) // self.batch_size
            for v in self.by_session.values()
        )


__all__ = [
    "SPEECH_TEXT_MEMBERS",
    "SPEECH_TEXT_INPUTS",
    "SPEECH_TEXT_TARGETS",
    "SPEECH_TEXT_REGIONS",
    "SPEECH_TEXT_PARTICIPANTS",
    "SPEECH_TEXT_ALREADY_NORMALIZED",
    "MOTOR_CURSOR_MEMBERS",
    "MOTOR_CURSOR_MEMBER_QUERIES",
    "MOTOR_CURSOR_MEMBER_QUERIES_CLOSED_LOOP",
    "MOTOR_CURSOR_DERIVATIONS",
    "MOTOR_CURSOR_INPUTS",
    "MOTOR_CURSOR_TARGETS",
    "MOTOR_CURSOR_INTENT_TARGETS",
    "MOTOR_CURSOR_REGIONS",
    "MOTOR_CURSOR_PARTICIPANTS",
    "MOTOR_CURSOR_ALREADY_NORMALIZED",
    "MOTOR_CURSOR_TRIGGER",
    "build_study",
    "speech_text_collection",
    "collection_identities",
    "motor_cursor_collection",
    "motor_cursor_identities",
    "SessionBatchSampler",
]
