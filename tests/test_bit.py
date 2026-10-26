"""Unit checks for the BIT collection pipeline.

These target the failure modes that produce no error on their own: a padding mask that does not
actually mask, a CTC feasibility bound that ignores repeated labels, statistics fitted on data
they should not have seen, and a config that does not round-trip.

    conda activate tether && pytest tests/test_bit.py
"""

import dataclasses
import pathlib

import numpy as np
import pandas as pd
import pytest
import torch

from neuralshed.manifest import (
    BUILDER_VERSION,
    CollectionManifest,
    LayoutSpec,
    kind_of,
    recipe_digest,
)
from neuralshed.models.bit import (
    BitConfig,
    BitCrossEncoder,
    BitPreprocessingState,
    BitTrial,
    NormalizationPolicy,
    collate_bit,
    validate_collection_for_bit,
    validate_manifest,
)
from neuralshed.models.bit.bundle import PRIMARY_STREAM
from neuralshed.text import (
    CANONICAL_CHARSET_V1,
    CharVocab,
    ctc_min_input_length,
    n_repeated_labels,
    normalize_text,
)
from neuralshed.transforms import (
    AssignHashSplit,
    AssignSourceSplit,
    StandardizeText,
    ValidateTarget,
    SPLITS,
    TARGET_FLAG,
    valid_targets,
)

# ===========================================================================
# Fixtures
# ===========================================================================

N_SPIKE, N_SBP, RATE = 8, 8, 50.0

#: Two recordings' worth of layout keys, in the shape ``CollectionManifest`` produces:
#: ``participant/study/session/event_type``. Recording "b" carries no band power, which is how
#: fan_2023 looks next to card_2024.
LAYOUTS_A = {"SpikeCounts": "P/m/a/SpikeCounts", "BandPower": "P/m/a/BandPower"}
LAYOUTS_B = {"SpikeCounts": "P/m/b/SpikeCounts"}
GROUP_A, GROUP_B = "m/a", "m/b"


def _config(**overrides) -> BitConfig:
    base = dict(
        n_classes=29,
        layout_keys=tuple(LAYOUTS_A.values()) + tuple(LAYOUTS_B.values()),
        layout_channels={
            LAYOUTS_A["SpikeCounts"]: N_SPIKE,
            LAYOUTS_A["BandPower"]: N_SBP,
            LAYOUTS_B["SpikeCounts"]: 4,
        },
        layout_streams={
            LAYOUTS_A["SpikeCounts"]: "SpikeCounts",
            LAYOUTS_A["BandPower"]: "BandPower",
            LAYOUTS_B["SpikeCounts"]: "SpikeCounts",
        },
        n_participants=2,
        n_regions=1,
        frequency=RATE,
        patch_ms=100.0,
        dim=32,
        depth=2,
        heads=2,
        dim_head=8,
        dropout=0.0,
        input_dropout=0.0,
        causal=False,
    )
    base.update(overrides)
    return BitConfig(**base)


def _trial(split="train", group_key=GROUP_A, n_frames=40, text="hello", scale=1.0):
    rng = np.random.default_rng(abs(hash((split, group_key, n_frames, text))) % 2**32)
    layouts = LAYOUTS_A if group_key == GROUP_A else LAYOUTS_B
    channels = {"SpikeCounts": N_SPIKE, "BandPower": N_SBP}
    arrays = {
        stream: (rng.normal(size=(channels[stream], n_frames)) * scale).astype(
            np.float32
        )
        for stream in layouts
    }
    return BitTrial(
        group_key=group_key,
        layouts=dict(layouts),
        member="m",
        participant="P",
        arrays=arrays,
        ids=CharVocab.canonical().encode(text),
        text=text,
        split=split,
        participant_idx=0,
        region_idx=0,
    )


def _fixture_manifest(groups=(GROUP_A,), participants=None) -> CollectionManifest:
    """A manifest matching the ``_trial`` fixture, built from specs rather than from events.

    Building it directly is the point: the manifest is a plain table, so a test that needs one
    of a particular shape should state that shape instead of reverse-engineering events.
    """
    widths = {
        LAYOUTS_A["SpikeCounts"]: N_SPIKE,
        LAYOUTS_A["BandPower"]: N_SBP,
        LAYOUTS_B["SpikeCounts"]: 4,
    }
    participants = participants or {}
    specs = []
    for group in groups:
        person = participants.get(group, "P")
        for stream, key in (LAYOUTS_A if group == GROUP_A else LAYOUTS_B).items():
            specs.append(
                LayoutSpec(
                    layout_key=key,
                    group_key=group,
                    event_type=stream,
                    kind="data",
                    role="input",
                    participant=person,
                    study="m",
                    member="m",
                    groups=(group,),
                    channels=widths[key],
                    native_rate=RATE,
                )
            )
        specs.append(
            LayoutSpec(
                layout_key=f"P/{group}/Sentence",
                group_key=group,
                event_type="Sentence",
                kind="text",
                role="target",
                participant=person,
                study="m",
                member="m",
                groups=(group,),
            )
        )
    return CollectionManifest(specs)


def _manifest(events, **kwargs) -> CollectionManifest:
    """The manifest a text-decoding collection resolves from these events."""
    kwargs.setdefault("inputs", ("SpikeCounts", "BandPower"))
    kwargs.setdefault("targets", ("Sentence",))
    return CollectionManifest.from_events(events, **kwargs)


def _events(
    n_trials=4, member="card", session="s1", with_sbp=True, split="train", subject=None
):
    """A minimal events frame shaped like a collection's output, with real ch_names headers
    so the manifest can resolve layout without touching a filesystem."""
    rows = []
    for i in range(n_trials):
        timeline = f"{member}/{session}/t{i}"
        common = dict(
            timeline=timeline,
            session=session,
            subject=subject or f"Study/{member.upper()}",
            study="Study",
            collection_member=member,
            start=0.0,
            duration=2.0,
            frequency=RATE,
        )
        rows.append(
            dict(
                type="SpikeCounts",
                n_features=N_SPIKE,
                ch_names=",".join(f"tx:{c}" for c in range(N_SPIKE)),
                unit="counts/bin",
                normalization="raw",
                **common,
            )
        )
        if with_sbp:
            rows.append(
                dict(
                    type="BandPower",
                    n_features=N_SBP,
                    ch_names=",".join(f"sbp:{c}" for c in range(N_SBP)),
                    unit="uV^2",
                    normalization="raw",
                    **common,
                )
            )
        rows.append(
            dict(
                type="Sentence",
                text=f"trial number {i}",
                original_split=split,
                **common,
            )
        )
    return pd.DataFrame(rows)


def _prepared_events(**kwargs) -> pd.DataFrame:
    """The full pre-model chain: normalize text, mark targets, assign a split.

    The split step stands in for a neuralset split transform (SklearnSplit); here it just
    copies each target's published ``original_split``, so fixtures stay deterministic.
    """
    events = _events(**kwargs)
    events = StandardizeText()._run(events)
    events = ValidateTarget()._run(events)
    events["split"] = ""
    valid = events[TARGET_FLAG].astype(bool)
    events.loc[valid, "split"] = events.loc[valid, "original_split"]
    return events


# ===========================================================================
# Vocabulary and CTC feasibility (items 1, 6)
# ===========================================================================


def test_canonical_charset_is_frozen():
    vocab = CharVocab.canonical()
    assert vocab.n_classes == len(CANONICAL_CHARSET_V1) + 1 == 29
    assert vocab.chars[0] == " " and vocab.BLANK == 0
    # id order is part of the contract: a checkpoint's head rows mean these characters
    assert vocab.encode("a b").tolist() == [3, 1, 4]


def test_canonical_vocab_does_not_depend_on_the_collection():
    assert CharVocab.canonical() == CharVocab.canonical()
    fitted = CharVocab.from_texts(["hello"])
    assert fitted.n_classes != CharVocab.canonical().n_classes


def test_coverage_reports_unencodable_characters():
    assert CharVocab.canonical().coverage(["hello"]) == {}
    assert CharVocab(["a"]).coverage(["abc"]) == {"b": 1, "c": 1}


def test_coverage_detects_a_scheme_charset_mismatch():
    """The check `coverage` exists for: a collection whose StandardizeText scheme lets through
    characters the model's charset does not hold. It only works because coverage takes the
    text as-is; re-normalizing with v1 would clean the digit away and report full coverage.
    """
    canonical_text = "flight 7 boards"  # what a digit-keeping scheme would emit
    assert CharVocab.canonical().coverage([canonical_text]) == {"7": 1}
    assert CharVocab.canonical().coverage([normalize_text(canonical_text)]) == {}


@pytest.mark.parametrize(
    "text,repeats",
    [("abc", 0), ("all", 1), ("coffee", 2), ("aaa", 2), ("", 0)],
)
def test_ctc_min_input_length_counts_repeats(text, repeats):
    ids = CharVocab.canonical().encode(text)
    assert n_repeated_labels(ids) == repeats
    assert ctc_min_input_length(ids) == len(ids) + repeats


def test_ctc_bound_admits_exactly_the_feasible_lengths():
    """A target at its minimum length must be alignable; one frame shorter must not be.

    zero_infinity=True turns an infeasible alignment into a zero gradient rather than an
    error, so this bound is the only thing standing between a bad trial and silent training.
    """
    vocab = CharVocab.canonical()
    ids = vocab.encode("coffee")
    n_min = ctc_min_input_length(ids)
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    for n_frames, feasible in ((n_min, True), (n_min - 1, False)):
        log_probs = torch.full((n_frames, 1, vocab.n_classes), -1.0).log_softmax(-1)
        value = loss(
            log_probs,
            torch.from_numpy(ids)[None],
            torch.tensor([n_frames]),
            torch.tensor([len(ids)]),
        )
        assert torch.isfinite(value).item() is feasible


# ===========================================================================
# Events transforms (items 2, 3)
# ===========================================================================


def test_standardize_text_normalizes_and_fills_empty():
    out = StandardizeText()._run(_events(n_trials=2))
    sentences = out[out.type == "Sentence"]
    assert set(sentences["canonical_text"]) == {
        "trial number"
    }  # lower-cased, unchanged
    # rows with no text get an empty string, not NaN, so a later `!= ''` is well-defined
    assert (out[out.type == "SpikeCounts"]["canonical_text"] == "").all()


def test_validate_target_marks_by_condition():
    out = ValidateTarget()._run(StandardizeText()._run(_events(n_trials=2)))
    targets = valid_targets(out)
    assert len(targets) == 2
    assert not out[out.type == "SpikeCounts"][TARGET_FLAG].any()


def test_validate_target_rejects_empty_text():
    """card_2024's test split has no sentence label. With the default condition, an empty
    normalized text fails `canonical_text != ''`, so it never becomes a target."""
    events = _events(n_trials=2)
    events.loc[events.type == "Sentence", "text"] = ["", "real words"]
    out = ValidateTarget()._run(StandardizeText()._run(events))
    targets = valid_targets(out)
    assert len(targets) == 1 and targets["canonical_text"].iloc[0] == "real words"


def test_validate_target_rejects_a_non_boolean_condition():
    with pytest.raises(ValueError, match="must evaluate to a boolean column"):
        ValidateTarget(condition="duration")._run(_events(n_trials=1))


def test_hash_split_is_deterministic_and_order_independent():
    events = _events(n_trials=40)
    step = AssignHashSplit()
    a = step._run(events).set_index(["timeline", "start", "type"])["split"]
    b = step._run(events.sample(frac=1.0, random_state=1)).set_index(
        ["timeline", "start", "type"]
    )["split"]
    assert a.sort_index().equals(b.sort_index())
    assert set(a) <= set(SPLITS)


def test_hash_split_keeps_rows_of_one_trial_together():
    """Rows sharing every `by` value get one split. In the per-trial layout (card_2024) that
    is a whole timeline, so a trial's neural rows and its target never disagree."""
    out = AssignHashSplit(by=["timeline"])._run(_events(n_trials=20))
    assert (out.groupby("timeline").split.nunique() == 1).all()


def test_hash_split_can_split_below_the_timeline():
    """willet_2023 stores one timeline per *session* with hundreds of sentences inside it.
    Keying on the timeline alone would drop the whole session into one split; adding `start`
    makes each sentence its own trial."""
    rows = [
        dict(
            type="Sentence",
            timeline="one-session",
            session="s",
            start=float(i),
            duration=1.0,
            text=f"sentence {i}",
        )
        for i in range(200)
    ]
    events = pd.DataFrame(rows)

    assert AssignHashSplit(by=["timeline"])._run(events).split.nunique() == 1
    fine = AssignHashSplit(by=["timeline", "start"])._run(events)
    assert set(fine.split) == set(SPLITS)


def test_hash_split_requires_its_key_columns():
    with pytest.raises(KeyError, match=r"no column\(s\) \['block'\]"):
        AssignHashSplit(by=["block"])._run(_events(n_trials=1))


def test_hash_split_can_scope_itself_to_one_event_type():
    """Scoped to the target class, the hash lands on the same rows AssignSourceSplit reaches,
    and the whole-session streams of a session-per-timeline source keep out of it."""
    events = _events(n_trials=40)
    out = AssignHashSplit(event_type="Sentence")._run(events)
    assert set(out[out.type == "Sentence"]["split"]) <= set(SPLITS)
    assert set(out[out.type != "Sentence"]["split"]) == {""}


def test_hash_split_scoping_does_not_change_the_assignment():
    """`event_type` masks which rows are written, it does not re-key the hash."""
    events = _events(n_trials=40)
    every = AssignHashSplit()._run(events)
    scoped = AssignHashSplit(event_type="Sentence")._run(events)
    targets = events.type == "Sentence"
    assert every.loc[targets, "split"].equals(scoped.loc[targets, "split"])


def test_hash_split_can_scope_to_a_stream_for_a_member_with_no_targets():
    """A label-free member has no target rows at all, so the trial is a window over the
    stream and the split has to land there."""
    events = _events(n_trials=20).query("type != 'Sentence'")
    out = AssignHashSplit(by=["timeline"], event_type="SpikeCounts")._run(events)
    assert set(out[out.type == "SpikeCounts"]["split"]) <= set(SPLITS)
    assert set(out[out.type == "BandPower"]["split"]) == {""}


def test_hash_split_leaves_everything_blank_when_no_row_matches():
    out = AssignHashSplit(event_type="Character")._run(_events(n_trials=2))
    assert set(out["split"]) == {""}


def test_source_split_maps_a_members_own_vocabulary():
    """FALCON names its splits held-in-calib / held-in-minival / held-out-calib and stamps one
    on a whole session, so the member states that mapping itself rather than the transform
    guessing at it."""
    events = _events(n_trials=3)
    labels = dict(zip(sorted(events.timeline.unique()), SPLITS))
    events["original_split"] = events.timeline.map(
        {
            tl: {
                "train": "held-in-calib",
                "val": "held-in-minival",
                "test": "held-out-calib",
            }[s]
            for tl, s in labels.items()
        }
    )
    out = AssignSourceSplit(
        mapping={
            "held-in-calib": "train",
            "held-in-minival": "val",
            "held-out-calib": "test",
        }
    )._run(events)
    assert dict(zip(out.timeline, out.split)) == labels


def test_source_split_blanks_labels_the_mapping_omits():
    """card_2024's `test` sentences were never released, so a member leaves that label out of
    its mapping and those rows come through unassigned rather than as a fourth split."""
    events = _events(n_trials=2, split="test")
    out = AssignSourceSplit(mapping={"train": "train", "val": "val"})._run(events)
    assert set(out["split"]) == {""}


def test_source_split_blanks_rows_with_no_label():
    """willet_2023 labels its Sentence rows and leaves the whole-session neural streams NaN,
    since a stream spanning both splits belongs to neither."""
    out = AssignSourceSplit(mapping={"train": "train"})._run(_events(n_trials=2))
    assert set(out[out.type == "Sentence"]["split"]) == {"train"}
    assert set(out[out.type != "Sentence"]["split"]) == {""}


def test_source_split_rejects_a_destination_that_is_not_a_split():
    with pytest.raises(ValueError, match=r"sends labels to \['eval'\]"):
        AssignSourceSplit(mapping={"train": "train", "eval": "eval"})._run(
            _events(n_trials=1)
        )


def test_source_split_requires_its_source_column():
    events = _events(n_trials=1).drop(columns=["original_split"])
    with pytest.raises(KeyError, match="no column 'original_split'"):
        AssignSourceSplit(mapping={"train": "train"})._run(events)


def test_validate_target_condition_can_reference_any_column():
    """The condition is a general pandas expression, not a fixed rule — a member with a
    different rule just uses a different ValidateTarget in its own chain."""
    out = ValidateTarget(condition="type == 'Sentence' and duration > 5.0")._run(
        _events(n_trials=2)  # 2.0 s trials
    )
    assert not out[TARGET_FLAG].any()


def test_combine_studies_writes_member_provenance():
    """Two members of the same Study class are indistinguishable without this column, and
    standardize_events does not back-fill it, so it has to be set before the concat."""
    import neuralset as ns
    from neuralshed.transforms import CombineStudies

    class _Fake(ns.base.Step):
        member: str = "a"

        def _run(self, *args):
            return _events(n_trials=1, member="ignored").drop(
                columns=["collection_member"]
            )

    out = CombineStudies(chains={"card": _Fake(), "willet": _Fake(member="b")})._run()
    assert set(out["collection_member"]) == {"card", "willet"}
    assert out["collection_member"].notna().all()


@pytest.mark.parametrize("missing", [np.nan, None, pd.NA])
def test_a_missing_target_flag_is_not_a_target(missing):
    """`bool(np.nan)` is True, so reading the flag directly selected rows nobody validated.

    This is the shape `CombineStudies` produces when one member carries its own ValidateTarget
    and another does not, which is the documented way to vary the rule per member. The member
    that correctly rejected its row was dropped; the member that never ran was kept.
    """
    events = pd.DataFrame(
        {
            "type": ["Sentence", "Sentence"],
            "canonical_text": ["rejected", "never validated"],
            TARGET_FLAG: [False, missing],
        }
    )
    assert valid_targets(events)["canonical_text"].tolist() == []


def test_valid_targets_still_selects_real_targets():
    events = pd.DataFrame(
        {
            "type": ["Sentence"] * 3,
            "canonical_text": ["yes", "no", "gap"],
            TARGET_FLAG: [True, False, np.nan],
        }
    )
    assert valid_targets(events)["canonical_text"].tolist() == ["yes"]


# ===========================================================================
# Manifest (item 4)
# ===========================================================================


def test_manifest_resolves_one_row_per_stream():
    manifest = _manifest(_prepared_events(n_trials=2))
    by_stream = {s.event_type: s for s in manifest.layouts}
    assert set(by_stream) == {"SpikeCounts", "BandPower", "Sentence"}
    assert by_stream["SpikeCounts"].channels == N_SPIKE
    assert by_stream["BandPower"].native_rate == RATE
    assert manifest.inputs == ["BandPower", "SpikeCounts"]
    assert manifest.targets == ["Sentence"]


def test_a_text_target_has_no_channels_or_rate():
    """`channels` and `rate` are properties of a dense array, not of every declared stream.
    Asking a Sentence for its channel count is a category error, not a missing value."""
    sentence = next(
        s for s in _manifest(_prepared_events()).layouts if s.kind == "text"
    )
    assert sentence.channels is None and sentence.native_rate is None
    assert sentence.role == "target" and not sentence.is_array


def test_kind_comes_from_the_event_class_hierarchy():
    assert kind_of("SpikeCounts") == "data"
    assert kind_of("Sentence") == "text"
    assert kind_of("Stimulus") == "categorical"
    with pytest.raises(KeyError, match="unknown event class"):
        kind_of("NotAnEventClass")


def test_a_missing_input_stream_is_an_absent_row():
    """fan_2023 has no band power. That has to be an absent layout rather than a zero block, or
    the trunk has to learn to ignore channels that were never recorded."""
    manifest = _manifest(_prepared_events(n_trials=2, with_sbp=False))
    assert {s.event_type for s in manifest.layouts} == {"SpikeCounts", "Sentence"}
    assert manifest.inputs == ["SpikeCounts"]  # wanted-if-present, not required


def test_manifest_keys_per_session_by_default():
    events = pd.concat(
        [_prepared_events(session="s1"), _prepared_events(session="s2")],
        ignore_index=True,
    )
    fine = _manifest(events)
    assert len(fine.groups) == 2 and len(fine) == 6  # 3 streams x 2 recordings

    coarse = _manifest(events, key_template=("participant", "study", "event_type"))
    assert len(coarse) == 3
    assert coarse.layout_for("card/s2", "SpikeCounts") is coarse.layout_for(
        "card/s1", "SpikeCounts"
    )


def test_key_template_requires_participant_and_event_type():
    """Dropping participant merges two people into one projection; dropping event_type puts
    two streams of different widths into one layout. Neither is recoverable downstream.
    """
    events = _prepared_events()
    for bad in [("study", "session"), ("participant", "session")]:
        with pytest.raises(ValueError, match="key template is missing"):
            _manifest(events, key_template=bad)
    with pytest.raises(ValueError, match="unknown key components"):
        _manifest(events, key_template=("participant", "event_type", "nonsense"))


def test_coarse_layout_key_reports_width_conflicts():
    """jude_2026_speech changes its column-to-area mapping between 256- and 384-channel sessions.
    A key without session would merge them, so the conflict has to be reported, not tolerated.
    """
    wide = _prepared_events(session="s2")
    wide.loc[wide.type == "SpikeCounts", "ch_names"] = ",".join(
        f"tx:{c}" for c in range(N_SPIKE * 2)
    )
    events = pd.concat([_prepared_events(session="s1"), wide], ignore_index=True)

    assert _manifest(events).conflicts() == []
    conflicts = _manifest(
        events, key_template=("participant", "study", "event_type")
    ).conflicts()
    assert len(conflicts) == 1
    assert "channels" in conflicts[0] and "16" in conflicts[0]


def test_coarse_layout_key_reports_reordered_channels():
    """Two recordings can share a channel *count* while column k is a different electrode in
    each. Comparing counts alone called that compatible and fed both through one projection.
    """
    shuffled = _prepared_events(session="s2")
    shuffled.loc[shuffled.type == "SpikeCounts", "ch_names"] = ",".join(
        f"tx:{c}" for c in reversed(range(N_SPIKE))
    )
    events = pd.concat([_prepared_events(session="s1"), shuffled], ignore_index=True)

    conflicts = _manifest(
        events, key_template=("participant", "study", "event_type")
    ).conflicts()
    assert len(conflicts) == 1 and "channel orderings" in conflicts[0]


def test_a_merge_with_no_headers_to_compare_warns_rather_than_passing(tmp_path):
    """The ordering check can only fire where a ch_names header exists, and no Study writes
    one today. Silence from the check is therefore not evidence of agreement, so a merge it
    could not verify says so out loud."""
    events = pd.concat(
        [
            _on_disk_events(tmp_path / "a", session="s1"),
            _on_disk_events(tmp_path / "b", session="s2"),
        ],
        ignore_index=True,
    ).drop(
        columns=["ch_names"]
    )  # real arrays, so widths come from reading them
    manifest = _manifest(events, key_template=("participant", "study", "event_type"))
    assert manifest.conflicts() == []
    assert any("could not be verified" in w for w in manifest.warnings())
    # a text layout has no columns to reorder, so it must not be reported as unverified
    assert not any("Sentence" in w for w in manifest.warnings())


def test_participants_stay_separate_unless_the_key_merges_them():
    """One Study recording two people (kunz_2025's shape). Participant is mandatory in the key
    precisely so this merge cannot be requested by accident."""
    events = pd.concat(
        [
            _prepared_events(session="s1", subject="Study/T12"),
            _prepared_events(session="s2", subject="Study/T15"),
        ],
        ignore_index=True,
    )
    assert _manifest(events).participants == ["Study/T12", "Study/T15"]
    coarse = _manifest(events, key_template=("participant", "study", "event_type"))
    assert coarse.participants == ["Study/T12", "Study/T15"]
    assert len(coarse) == 6  # participant keeps them apart even without session


def test_manifest_separates_the_identities():
    events = pd.concat(
        [
            _prepared_events(member="card", session="s1"),
            _prepared_events(member="wairagkar", session="s2"),
        ],
        ignore_index=True,
    )
    manifest = _manifest(
        events,
        participants={"card": "braingate2/T15", "wairagkar": "braingate2/T15"},
        regions={"card": "speech", "wairagkar": "speech"},
    )
    # two recordings, one participant, one region: layout varies independently of identity
    assert len(manifest.groups) == 2
    assert (manifest.n_participants, manifest.n_regions) == (1, 1)
    assert len(manifest.array_layouts) == 4  # 2 streams x 2 recordings


def test_rate_blocks_a_merge_only_until_a_target_rate_is_declared():
    """Rate is layout-defining while nothing reconciles it. Once the recipe declares a target
    rate, resampling makes it uniform by construction and the check goes vacuous."""
    fast = _prepared_events(session="s2")
    fast.loc[:, "frequency"] = 100.0
    events = pd.concat([_prepared_events(session="s1"), fast], ignore_index=True)
    coarse = ("participant", "study", "event_type")

    assert any(
        "native rate" in c for c in _manifest(events, key_template=coarse).conflicts()
    )
    assert _manifest(events, key_template=coarse, target_rate=50.0).conflicts() == []


def test_recordings_without_a_target_stream_are_reported():
    """A session with neural data and no sentences contributes nothing. Caught here it is one
    named recording; caught at extraction it is a trial count that quietly fails to add up.
    """
    events = _prepared_events(n_trials=2)
    manifest = CollectionManifest.from_events(
        events[events.type != "Sentence"],
        inputs=("SpikeCounts",),
        targets=("Sentence",),
    )
    assert manifest.recordings_missing_targets() == ["card/s1"]
    assert _manifest(events).recordings_missing_targets() == []


def test_manifest_round_trips_through_json():
    """It is stored as a JSON blob on the collection, so it has to survive the trip exactly."""
    import json

    manifest = _manifest(_prepared_events(n_trials=2), participants={"card": "bg2/T15"})
    blob = json.loads(json.dumps(manifest.to_dict()))
    assert CollectionManifest.from_dict(blob).to_dict() == manifest.to_dict()


def test_a_blob_from_another_builder_version_is_refused():
    """A stored manifest is a second source of truth about the data. When the code that reads
    it is not the code that wrote it, rebuilding beats trusting."""
    blob = _manifest(_prepared_events()).to_dict()
    blob["builder_version"] = BUILDER_VERSION + 1
    with pytest.raises(ValueError, match="rebuild it"):
        CollectionManifest.from_dict(blob)


def test_integer_indices_are_not_part_of_the_stored_form():
    """Indices are positions in a sorted list. Storing them would make a filtered manifest
    renumber a checkpoint's embedding rows underneath it."""
    manifest = _manifest(_prepared_events(n_trials=2))
    blob = manifest.to_dict()
    assert not any("idx" in key or "index" in key for key in blob["layouts"][0])
    assert manifest.participant_index == {"Study/CARD": 0}


def test_filter_keeps_layout_keys_stable():
    """An isolated run and a pooled run must name the same layouts, or their results cannot be
    compared."""
    events = pd.concat(
        [
            _prepared_events(session="s1", subject="Study/T12"),
            _prepared_events(session="s2", subject="Study/T15"),
        ],
        ignore_index=True,
    )
    manifest = _manifest(events)
    one = manifest.filter(participants=["Study/T15"])
    assert one.participants == ["Study/T15"]
    assert set(s.layout_key for s in one.layouts) <= set(
        s.layout_key for s in manifest.layouts
    )
    assert len(manifest.layouts) == 6  # parent untouched


def test_the_recipe_digest_covers_the_key_template():
    """It is the only staleness signal a stored manifest has. Leave the template out and a
    manifest keyed one way is used under a recipe that now says another."""
    base = dict(chain={"members": ["card"]}, inputs=["SpikeCounts"])
    fine = recipe_digest(
        **base, key_template=["participant", "study", "session", "event_type"]
    )
    coarse = recipe_digest(**base, key_template=["participant", "study", "event_type"])
    assert fine != coarse
    assert fine == recipe_digest(
        **base, key_template=["participant", "study", "session", "event_type"]
    )


# ===========================================================================
# Normalization (item 7)
# ===========================================================================


SPIKE_A = LAYOUTS_A["SpikeCounts"]


def test_statistics_are_fitted_on_training_trials_only():
    """The leakage test from the plan: removing the validation trials must not change a
    single fitted value."""
    manifest = _fixture_manifest()
    train = [_trial("train", text=f"train {i}") for i in range(3)]
    val = [_trial("val", text=f"val {i}", scale=100.0) for i in range(3)]

    with_val = BitPreprocessingState.fit(train + val, manifest)
    without_val = BitPreprocessingState.fit(train, manifest)
    assert with_val.fingerprint() == without_val.fingerprint()


def test_already_normalized_members_are_left_alone():
    """card_2024 arrives z-scored within session. This is a *declared* fact, so it is passed
    in from the recipe rather than read off the manifest, which holds only resolved facts.
    """
    state = BitPreprocessingState.fit(
        [_trial("train")], _fixture_manifest(), already_normalized={"m"}
    )
    assert state.stats == {}
    x = np.ones((N_SPIKE, 5), np.float32)
    assert np.array_equal(state.transform(SPIKE_A, x), x)


def test_zscore_actually_normalizes():
    trials = [_trial("train", text=f"t {i}", scale=7.0) for i in range(4)]
    state = BitPreprocessingState.fit(trials, _fixture_manifest())
    out = np.concatenate(
        [state.transform(SPIKE_A, t.arrays[PRIMARY_STREAM]) for t in trials], axis=1
    )
    assert np.allclose(out.mean(axis=1), 0.0, atol=1e-4)
    assert np.allclose(out.std(axis=1), 1.0, atol=1e-3)


def test_statistics_are_keyed_by_layout_not_by_a_pair():
    """A layout is a stream, so its key already names the event class. One entry per layout."""
    state = BitPreprocessingState.fit([_trial("train")], _fixture_manifest())
    assert set(state.stats) == set(LAYOUTS_A.values())
    assert state.get(SPIKE_A).mean.shape == (N_SPIKE, 1)


def test_state_survives_a_buffer_round_trip():
    state = BitPreprocessingState.fit(
        [_trial("train")], _fixture_manifest(), NormalizationPolicy()
    )
    buffers = state.as_buffers()
    assert buffers and all("." not in name for name in buffers)
    assert all(torch.is_tensor(t) for t in buffers.values())


# ===========================================================================
# Collate and model (items 5, 8)
# ===========================================================================


def test_collate_pads_inputs_and_targets():
    batch = collate_bit(
        [_trial(n_frames=40, text="abc"), _trial(n_frames=25, text="ab")]
    )
    assert batch.arrays[PRIMARY_STREAM].shape == (2, N_SPIKE, 40)
    assert batch.input_lengths.tolist() == [40, 25]
    # padding is zeros, not a repeat of the last frame
    assert torch.count_nonzero(batch.arrays[PRIMARY_STREAM][1, :, 25:]) == 0
    assert batch.flat_targets.numel() == batch.target_lengths.sum()


def test_collate_refuses_a_batch_from_two_recordings():
    """Batches come from one recording because each stream is padded to one channel count and
    routed through one projection. Mixing them would train the wrong projection silently.
    """
    with pytest.raises(ValueError, match="must come from one recording"):
        collate_bit([_trial(group_key=GROUP_A), _trial(group_key=GROUP_B)])


def test_patcher_lengths_match_the_tensor_it_produces():
    model = BitCrossEncoder(_config())
    for n_frames in (1, 4, 5, 6, 40, 41):
        h = torch.zeros(1, n_frames, model.config.dim)
        assert model.patcher(h).shape[1] == int(
            model.out_lengths(torch.tensor([n_frames]))
        )


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("n_frames", [20, 21, 23, 37])
def test_padding_does_not_change_the_valid_logits(causal, n_frames):
    """The upstream defect this port fixes. With causal=False upstream has no mask at all, so
    a padded batch and its unpadded single-item equivalent disagree on the valid frames.

    ``n_frames`` deliberately includes lengths that are *not* whole patches. Those exercise a
    second, subtler path: the read-in has a bias and a LayerNorm, so a zero-padded input bin
    comes out non-zero, and unless the read-in output is masked the item's last partial patch
    gets filled from the batch padding. Real trials are almost never a whole number of
    patches, so an all-multiples test suite misses this entirely.
    """
    model = BitCrossEncoder(_config(causal=causal)).eval()
    short = _trial(n_frames=n_frames, text="abc")
    long = _trial(n_frames=60, text="abcdef")

    with torch.no_grad():
        padded = model(collate_bit([short, long]).as_dict())
        alone = model(collate_bit([short]).as_dict())
    n_valid = int(model.out_lengths(torch.tensor([short.n_frames])))
    assert torch.allclose(padded[0, :n_valid], alone[0, :n_valid], atol=1e-5)


def test_readin_output_is_zeroed_past_each_items_length():
    """Stated directly, so the cause is visible when the test above fails."""
    model = BitCrossEncoder(_config()).eval()
    batch = collate_bit(
        [_trial(n_frames=23, text="abc"), _trial(n_frames=60, text="abcdef")]
    ).as_dict()

    with torch.no_grad():
        h = model.readin(
            batch["arrays"],
            batch["layouts"],
            batch["participant_idx"],
            batch["region_idx"],
        )
    # left alone, the read-in maps zero padding to something non-zero
    assert h[0, 23:].abs().max() > 1e-3
    valid = torch.arange(h.shape[1])[None] < batch["input_lengths"][:, None]
    assert (h * valid.unsqueeze(-1))[0, 23:].abs().max() == 0.0


def test_fully_padded_rows_do_not_produce_nan():
    model = BitCrossEncoder(_config()).eval()
    batch = collate_bit([_trial(n_frames=60, text="abcdef")]).as_dict()
    batch["input_lengths"] = torch.tensor([1])  # everything after patch 0 is padding
    with torch.no_grad():
        assert torch.isfinite(model(batch)).all()


def test_read_in_absorbs_different_channel_counts_under_one_trunk():
    """The point of moving the read-in ahead of the patcher: two widths, one shared trunk,
    no model edit. Upstream cannot do this at all, since patchify fixes width batch-globally.
    """
    model = BitCrossEncoder(_config())
    wide = collate_bit([_trial(group_key=GROUP_A, n_frames=30)]).as_dict()

    narrow_trial = _trial(group_key=GROUP_B, n_frames=30)
    narrow_trial.arrays = {PRIMARY_STREAM: np.zeros((4, 30), np.float32)}
    narrow = collate_bit([narrow_trial]).as_dict()

    with torch.no_grad():
        a, b = model(wide), model(narrow)
    assert a.shape == b.shape
    trunk_params = {id(p) for p in model.trunk.parameters()}
    assert trunk_params and not (
        trunk_params & {id(p) for p in model.readin.parameters()}
    )


def test_unknown_layout_raises_instead_of_falling_back():
    """Upstream routed anything unmatched through dataset_keys[0], so a misconfigured run
    trained on the wrong projection and reported nothing."""
    model = BitCrossEncoder(_config())
    batch = collate_bit([_trial(group_key=GROUP_A)]).as_dict()
    batch["layouts"] = {**batch["layouts"], PRIMARY_STREAM: "does/not/exist"}
    with pytest.raises(KeyError, match="no projection for layout"):
        model(batch)


def test_a_batch_naming_no_layout_for_a_stream_raises():
    """A trial carries a stream the manifest does not know about: the two disagree, and
    guessing which projection to use is how upstream trained on the wrong one."""
    model = BitCrossEncoder(_config())
    batch = collate_bit([_trial(group_key=GROUP_A)]).as_dict()
    batch["layouts"] = {PRIMARY_STREAM: LAYOUTS_A[PRIMARY_STREAM]}  # BandPower dropped
    with pytest.raises(KeyError, match="names no layout for it"):
        model(batch)


def test_a_recording_without_the_primary_stream_is_refused():
    model = BitCrossEncoder(_config())
    batch = collate_bit([_trial(group_key=GROUP_A)]).as_dict()
    batch["arrays"] = {"BandPower": batch["arrays"]["BandPower"]}
    with pytest.raises(KeyError, match="no 'SpikeCounts' input"):
        model(batch)


def test_config_round_trips_and_derives_patch_bins_from_time():
    config = _config(patch_ms=100.0, frequency=50.0)
    assert config.patch_bins == 5
    assert BitConfig(**dataclasses_asdict(config)) == config
    # the same physical patch at double the rate is twice as many bins
    assert _config(patch_ms=100.0, frequency=100.0).patch_bins == 10


def test_config_from_manifest_rejects_mixed_rates():
    fast = _prepared_events(session="s2")
    fast.loc[:, "frequency"] = 100.0
    events = pd.concat([_prepared_events(session="s1"), fast], ignore_index=True)
    with pytest.raises(ValueError, match="spans .* Hz"):
        BitConfig.from_manifest(_manifest(events), n_classes=29)


def test_config_only_builds_projections_for_dense_inputs():
    """A Sentence target is a layout too, but there is nothing to project."""
    config = BitConfig.from_manifest(_manifest(_prepared_events()), n_classes=29)
    assert set(config.layout_streams.values()) == {"SpikeCounts", "BandPower"}


def test_model_from_manifest_covers_every_input_layout():
    manifest = _manifest(
        pd.concat(
            [_prepared_events(session="s1"), _prepared_events(session="s2")],
            ignore_index=True,
        )
    )
    model = BitCrossEncoder.from_manifest(manifest, n_classes=29, dim=16, depth=1)
    from neuralshed.models.bit.model import sanitize

    assert set(model.readin.project) == {
        sanitize(s.layout_key) for s in manifest.array_layouts
    }


def dataclasses_asdict(config: BitConfig) -> dict:
    import dataclasses

    return {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}


# ===========================================================================
# Preflight (item 9)
# ===========================================================================


def test_preflight_passes_a_well_formed_collection():
    events = _prepared_events(n_trials=6)
    report = validate_collection_for_bit(events, patch_bins=5)
    assert report.ok, report


def test_preflight_fails_without_the_transforms():
    report = validate_collection_for_bit(_events())
    assert not report.ok
    assert "ValidateTarget" in report.errors[0]


def test_preflight_fails_on_a_missing_split_column():
    # targets marked, but no split transform ran
    events = ValidateTarget()._run(StandardizeText()._run(_events(n_trials=2)))
    report = validate_collection_for_bit(events)
    assert not report.ok
    assert any("split transform" in e for e in report.errors)


def test_preflight_fails_on_mixed_rates():
    fast = _prepared_events(session="s2")
    fast.loc[:, "frequency"] = 100.0
    events = pd.concat([_prepared_events(session="s1"), fast], ignore_index=True)
    report = validate_collection_for_bit(events)
    assert not report.ok
    assert any("Hz" in e for e in report.errors)


def test_preflight_fails_on_uncodable_characters():
    events = _prepared_events(n_trials=2)
    report = validate_collection_for_bit(events, vocab=CharVocab(["a"], version="tiny"))
    assert not report.ok
    assert any("cannot encode" in e for e in report.errors)


def test_preflight_reports_infeasible_trials_before_extraction():
    events = _prepared_events(n_trials=2)
    events.loc[events.type == "Sentence", "canonical_text"] = "a" * 200
    report = validate_collection_for_bit(events, patch_bins=5)
    assert any("too short to emit" in w for w in report.warnings)


def test_preflight_fails_when_a_member_contributes_no_targets():
    """A member whose rows are all rejected never appears in the per-member target counts, so
    it has to be found by differencing against the members present in the events."""
    good = _prepared_events(member="card", session="s1")
    silent = _prepared_events(member="willet", session="s2")
    silent.loc[silent.type == "Sentence", TARGET_FLAG] = False
    report = validate_collection_for_bit(pd.concat([good, silent], ignore_index=True))
    assert not report.ok
    assert any("contribute no targets" in e and "willet" in e for e in report.errors)


def test_feasibility_uses_the_same_ceiling_as_extraction():
    """101 bins over 5-bin patches is 21 emissions, not 20. Flooring here would warn about a
    trial extraction goes on to keep, because the patcher right-pads the last partial patch.
    """
    events = _prepared_events(n_trials=1)
    events["duration"] = 101 / RATE
    events.loc[events.type == "Sentence", "canonical_text"] = "abcdefghijklmnopqrstu"
    assert (
        ctc_min_input_length(CharVocab.canonical().encode("abcdefghijklmnopqrstu"))
        == 21
    )

    report = validate_collection_for_bit(events, patch_bins=5)
    assert report.facts["ctc feasibility"] == "1/1 trials"
    assert not any("too short to emit" in w for w in report.warnings)


def test_structural_preflight_needs_only_the_manifest():
    """The half a collection builder can run while somebody is still assembling members: no
    events, no data drive, no extraction. A stored manifest is enough."""
    manifest = _manifest(_prepared_events(n_trials=2))
    assert validate_manifest(manifest).ok

    # and it still catches the structural failures, from the table alone
    events = _prepared_events(n_trials=2)
    report = validate_manifest(
        CollectionManifest.from_dict(
            _manifest(
                events[events.type != "Sentence"], targets=("Sentence",)
            ).to_dict()
        )
    )
    assert not report.ok
    assert any("no target stream" in e for e in report.errors)


def test_structural_preflight_can_drop_the_task_specific_requirement():
    """Needing SpikeCounts is this decoder's rule, not the manifest's."""
    events = _prepared_events(n_trials=2, with_sbp=True)
    manifest = CollectionManifest.from_events(
        events, inputs=("BandPower",), targets=("Sentence",)
    )
    assert not validate_manifest(manifest).ok  # no SpikeCounts anywhere
    assert validate_manifest(manifest, require_stream=None).ok


def test_report_raises_with_every_error_listed():
    report = validate_collection_for_bit(_events())
    with pytest.raises(ValueError, match="not BIT-trainable"):
        report.raise_if_failed()


# ===========================================================================
# End to end
# ===========================================================================


def _on_disk_events(tmp_path, n_trials=4, n_bins=300, text="hello world", session="s1"):
    """An events frame backed by real ``.npy`` files, so the extractors actually read arrays.

    Everything above this point works on frames alone; segmentation is the one stage that
    needs a filesystem, and it is where the window arithmetic lives.
    """
    from neuralset.events import utils as ns_utils

    tmp_path = pathlib.Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_trials):
        timeline = f"card/{session}/t{i}"
        common = dict(
            timeline=timeline,
            session=session,
            subject="T15",
            study="Study",
            collection_member="card",
        )
        for etype, name, unit in (
            ("SpikeCounts", "tx", "counts/bin"),
            ("BandPower", "sbp", "uV^2"),
        ):
            path = tmp_path / f"t{i}_{name}.npy"
            np.save(path, np.random.rand(N_SPIKE, n_bins).astype(np.float32))
            rows.append(
                dict(
                    type=etype,
                    start=0.0,
                    duration=n_bins / RATE,
                    frequency=RATE,
                    filepath=str(path),
                    n_features=N_SPIKE,
                    ch_names=",".join(f"{name}:{c}" for c in range(N_SPIKE)),
                    unit=unit,
                    normalization="raw",
                    **common,
                )
            )
        rows.append(
            dict(
                type="Sentence",
                start=0.2,
                duration=4.0,
                text=text,
                original_split="train",
                **common,
            )
        )
    events = ns_utils.standardize_events(pd.DataFrame(rows), auto_fill=False)
    events = StandardizeText()._run(events)
    events = ValidateTarget()._run(events)
    events["split"] = ""
    valid = events[TARGET_FLAG].astype(bool)
    events.loc[valid, "split"] = events.loc[valid, "original_split"]
    return events


def test_extraction_windows_each_trial_at_its_own_duration(tmp_path):
    from neuralshed.models.bit.bundle import _extract_trials

    events = _on_disk_events(tmp_path)
    manifest = _manifest(events)
    trials, dropped = _extract_trials(
        events,
        manifest,
        CharVocab.canonical(),
        frequency="native",
        target_unit="Hz",
        patch_bins=5,
    )
    assert len(trials) == 4 and not any(dropped.values())
    # the 4.0 s trigger window at 50 Hz, not the 6.0 s neural stream behind it
    assert trials[0].arrays[PRIMARY_STREAM].shape == (N_SPIKE, 200)
    assert set(trials[0].arrays) == {"SpikeCounts", "BandPower"}
    assert trials[0].split == "train"


def test_extraction_requires_a_split_column(tmp_path):
    """Without this the split defaulted to "train", so a collection missing its split
    transform trained on everything and fitted normalization on everything."""
    from neuralshed.models.bit.bundle import _extract_trials

    events = _on_disk_events(tmp_path)
    manifest = _manifest(events)
    args = (manifest, CharVocab.canonical())
    kwargs = dict(frequency="native", target_unit="Hz", patch_bins=5)

    with pytest.raises(KeyError, match="split"):
        _extract_trials(events.drop(columns=["split"]), *args, **kwargs)

    partial = events.copy()
    partial.loc[partial[TARGET_FLAG].astype(bool).idxmax(), "split"] = ""
    with pytest.raises(ValueError, match="did not reach every target"):
        _extract_trials(partial, *args, **kwargs)


def test_a_declared_target_rate_carries_through_to_patch_bins():
    """``from_collection``'s ``frequency`` reaches ``patch_bins`` through the manifest's target
    rate, so a 100 ms patch stays 100 ms after a resample instead of silently doubling.
    """
    events = _prepared_events()
    native = BitConfig.from_manifest(_manifest(events), n_classes=29)
    retimed = BitConfig.from_manifest(
        _manifest(events, target_rate=2 * RATE), n_classes=29
    )
    assert native.frequency == RATE and native.patch_bins == 5
    assert retimed.frequency == 2 * RATE and retimed.patch_bins == 10


def test_subset_filters_the_manifest_and_reindexes_embeddings():
    """An isolated model must get a manifest holding only that participant's layouts, so its
    embedding tables are not sized for people it never sees. The layout *keys* stay as they
    were, so the isolated run and the pooled run name the same things."""
    from neuralshed.models.bit.bundle import BitDataBundle

    people = {GROUP_A: "T12", GROUP_B: "T15"}
    manifest = _fixture_manifest(groups=(GROUP_A, GROUP_B), participants=people)
    trials = [
        dataclasses.replace(_trial(group_key=g), participant=p, participant_idx=i)
        for i, (g, p) in enumerate(people.items())
        for _ in range(3)
    ]
    bundle = BitDataBundle(
        trials,
        manifest,
        CharVocab.canonical(),
        BitPreprocessingState({}, NormalizationPolicy()),
        pd.DataFrame(),
        {},
        5,
    )

    one = bundle.subset(participants=["T15"])
    assert {t.participant for t in one.trials} == {"T15"}
    assert {t.participant_idx for t in one.trials} == {0}  # packed from zero
    assert set(one.manifest.by_key) < set(
        manifest.by_key
    )  # keys preserved, not rebuilt
    assert len(bundle.trials) == 6  # parent untouched


def test_subset_caps_each_participant():
    from neuralshed.models.bit.bundle import _cap_per_participant

    trials = [
        dataclasses.replace(_trial(group_key=g), participant=p)
        for g, p in ((GROUP_A, "T12"), (GROUP_B, "T12"), (GROUP_A, "T15"))
        for _ in range(10)
    ]
    kept = _cap_per_participant(trials, cap=6, seed=0)

    per_person: dict[str, int] = {}
    for t in kept:
        per_person[t.participant] = per_person.get(t.participant, 0) + 1
    assert per_person == {"T12": 6, "T15": 6}
    # T12 spans two recordings; the cap is split across them rather than taken from one
    t12 = [t for t in kept if t.participant == "T12"]
    assert len({t.group_key for t in t12}) == 2


def test_bundle_config_refuses_a_stride_changing_override():
    """`patch_bins` decided which trials were dropped as too short. Patching more coarsely
    afterwards puts them back below the CTC bound, where zero_infinity hides them."""
    from neuralshed.models.bit.bundle import BitDataBundle
    from neuralshed.models.bit.preprocess import BitPreprocessingState

    events = _prepared_events(n_trials=2)
    manifest = _manifest(events)
    bundle = BitDataBundle(
        trials=[],
        manifest=manifest,
        vocab=CharVocab.canonical(),
        state=BitPreprocessingState({}, NormalizationPolicy()),
        events=events,
        dropped={},
        patch_bins=5,
    )
    assert bundle.config().patch_bins == 5
    assert bundle.config(dim=16).patch_bins == 5  # unrelated overrides still work
    with pytest.raises(ValueError, match="rebuild the bundle"):
        bundle.config(patch_ms=200.0)


def test_trials_too_short_for_their_target_are_dropped(tmp_path):
    from neuralshed.models.bit.bundle import _extract_trials

    # 4.0 s at 50 Hz over 5-bin patches is 40 emissions; 60 characters cannot fit
    events = _on_disk_events(tmp_path, text="a bcde " * 10)
    manifest = _manifest(events)
    trials, dropped = _extract_trials(
        events,
        manifest,
        CharVocab.canonical(),
        frequency="native",
        target_unit="Hz",
        patch_bins=5,
    )
    assert trials == [] and dropped["too_short"] == 4


def test_both_decoders_consume_one_batch():
    """The point of retiring build_pooled_dataset: BIT and the GRU baseline are built from one
    manifest and fed one batch, so a CER gap between them is a model difference rather than a
    difference in vocabulary, splits or normalization."""
    from neuralshed.models import BrainToTextCTC
    from neuralshed.models.bit import build_bit_model

    manifest = _fixture_manifest()
    vocab = CharVocab.canonical()
    batch = collate_bit([_trial(), _trial(text="world")])

    bit = build_bit_model(manifest, vocab, dim=16, depth=1, heads=2, dim_head=8)
    gru = BrainToTextCTC.from_manifest(
        manifest, vocab.n_classes, d_model=16, hidden=8, gru_layers=1
    )

    loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    for model in (bit, gru):
        log_probs = model(batch)  # the same object, nothing in between
        assert log_probs.shape[0] == 2 and log_probs.shape[-1] == vocab.n_classes
        value = loss(
            log_probs.permute(1, 0, 2),
            batch.flat_targets,
            model.out_lengths(batch.input_lengths),
            batch.target_lengths,
        )
        assert torch.isfinite(value), type(model).__name__


def test_the_gru_baseline_applies_fitted_normalization():
    """The retired loader z-scored trial arrays in place, so `BrainToTextCTC` never normalized
    anything itself. With the arrays now left raw, it has to apply the bundle's statistics or it
    trains on card's z-scored features and willett's raw µV² at the same time."""
    from neuralshed.models import BrainToTextCTC

    manifest = _fixture_manifest()
    trials = [_trial(scale=500.0)]  # far off unit variance
    state = BitPreprocessingState.fit(trials, manifest)
    batch = collate_bit(trials)

    raw = BrainToTextCTC.from_manifest(manifest, 29, d_model=16, hidden=8, gru_layers=1)
    normed = BrainToTextCTC.from_manifest(
        manifest, 29, state=state, d_model=16, hidden=8, gru_layers=1
    )
    normed.load_state_dict(
        raw.state_dict(), strict=False
    )  # same weights, only stats differ

    raw.eval(), normed.eval()
    with torch.no_grad():
        assert not torch.allclose(
            raw(batch), normed(batch)
        ), "statistics were not applied"

    # and the values it applies are the same ones BIT applies
    h = normed.readin._normalize(batch["arrays"][PRIMARY_STREAM], SPIKE_A)
    expected = state.get(SPIKE_A).apply(batch["arrays"][PRIMARY_STREAM].numpy())
    assert np.allclose(h.numpy(), expected, atol=1e-5)


def test_bit_batch_reads_like_a_dict():
    """Models index the batch with []; `to()` must not hand back a stale cached view."""
    batch = collate_bit([_trial()])
    assert "arrays" in batch and batch["arrays"][PRIMARY_STREAM].shape[0] == 1
    assert batch.get("nope") is None
    assert batch["targets"].tolist() == batch.flat_targets.tolist()
    assert dict(**batch)["input_lengths"] is batch["input_lengths"]  # {**batch} works
    assert batch.to("cpu").as_dict() is not batch.as_dict()  # `replace` drops the cache


def test_masked_pretraining_targets_normalized_space():
    """Moving normalization into the model silently changed the pretext task twice: the MSE
    target became raw counts, and masking to 0.0 stopped meaning "the channel mean". Both make
    the objective scale-dependent, and pretraining measurably hurt fine-tuning until fixed.
    """
    from neuralshed.models import BrainToTextCTC, span_mask

    manifest = _fixture_manifest()
    trials = [_trial(scale=400.0) for _ in range(2)]
    state = BitPreprocessingState.fit(trials, manifest)
    batch = collate_bit(trials)
    gru = BrainToTextCTC.from_manifest(
        manifest, 29, state, d_model=16, hidden=8, gru_layers=1
    )
    mask = span_mask(batch.input_lengths, batch["arrays"][PRIMARY_STREAM].shape[-1])
    _, target, _ = gru.masked_reconstruct(batch, mask)

    raw_scale = batch["arrays"][PRIMARY_STREAM].std().item()
    assert raw_scale > 50, raw_scale  # the inputs really are far off unit variance
    assert target.std().item() < 5.0, "target is in raw space, not normalized space"


def test_masked_pretraining_runs_on_the_shared_batch():
    """The self-supervised path reads the primary stream and its layout off the same batch."""
    from neuralshed.models import BrainToTextCTC, span_mask

    manifest = _fixture_manifest()
    batch = collate_bit([_trial(), _trial()])
    gru = BrainToTextCTC.from_manifest(
        manifest, CharVocab.canonical().n_classes, d_model=16, hidden=8, gru_layers=1
    )
    mask = span_mask(batch.input_lengths, batch["arrays"][PRIMARY_STREAM].shape[-1])
    pred, target, frame_mask = gru.masked_reconstruct(batch, mask)
    assert pred.shape == target.shape and frame_mask.any()


def test_trials_with_no_neural_signal_are_dropped(tmp_path):
    """`allow_missing=True` returns a zero-filled window instead of raising when a trigger's
    span has no neural events, which pairs a real text target with silence."""
    from neuralshed.models.bit.bundle import _extract_trials

    events = _on_disk_events(tmp_path)
    manifest = _manifest(events)
    silent = events.copy()
    # push one trigger past the end of its neural stream, the shape the warning describes
    idx = silent[silent[TARGET_FLAG].astype(bool)].index[0]
    silent.loc[idx, "start"] = 500.0

    trials, dropped = _extract_trials(
        silent,
        manifest,
        CharVocab.canonical(),
        frequency="native",
        target_unit="Hz",
        patch_bins=5,
    )
    assert dropped["empty_window"] + dropped["no_signal"] == 1
    assert len(trials) == 3
    assert all(t.arrays[PRIMARY_STREAM].any() for t in trials)


def test_end_to_end_produces_a_finite_ctc_loss(tmp_path):
    """Events to loss, through every stage. A finite value means the emission lengths the
    model reports and the ones CTC is given are the same arithmetic."""
    from neuralshed.models.bit import build_bit_model
    from neuralshed.models.bit.bundle import _extract_trials

    events = _on_disk_events(tmp_path)
    manifest = _manifest(events)
    vocab = CharVocab.canonical()
    trials, _ = _extract_trials(
        events, manifest, vocab, frequency="native", target_unit="Hz", patch_bins=5
    )
    state = BitPreprocessingState.fit(trials, manifest)
    model = build_bit_model(
        manifest, vocab, state, dim=16, depth=1, heads=2, dim_head=8
    )

    batch = collate_bit(trials)
    log_probs = model(batch.as_dict())
    assert log_probs.shape == (4, 40, vocab.n_classes)

    lengths = model.out_lengths(batch.input_lengths)
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=False)(
        log_probs.permute(1, 0, 2), batch.flat_targets, lengths, batch.target_lengths
    )
    assert torch.isfinite(loss)
