"""Logical-stream selection for neuralshed dense-array extractors."""

import neuralset as ns
import numpy as np
import pytest
import torch

from neuralshed.events import BandPower, SpikeCounts
from neuralshed.extractors import NeuralFeatureExtractor, WhisperLabelEncoder


def _event(tmp_path, *, value, brain_region="", cls=SpikeCounts):
    data = np.full((2, 8), value, dtype=np.float32)
    path = tmp_path / f"{cls.__name__}_{brain_region or 'unset'}_{value}.npy"
    np.save(path, data)
    common = dict(
        timeline="recording",
        subject="P1",
        feature="threshold_crossings",
        brain_region=brain_region,
        start=0.0,
        duration=2.0,
        frequency=4.0,
        n_features=2,
        unit="counts/bin",
        normalization="raw",
        filepath=str(path),
    )
    if cls is BandPower:
        common.update(feature="spike_band_power", unit="uV^2")
    return cls(**common)


def test_brain_region_is_optional_event_metadata(tmp_path):
    assert _event(tmp_path, value=1).brain_region == ""
    assert _event(tmp_path, value=2, brain_region="6d").brain_region == "6d"


def test_one_extractor_output_per_parallel_logical_stream(tmp_path):
    events = [
        _event(tmp_path, value=1, brain_region="6d"),
        _event(tmp_path, value=2, brain_region="6v"),
        _event(tmp_path, value=3, brain_region="44"),
        _event(tmp_path, value=4, brain_region="M1"),
    ]
    extractors = {
        region: NeuralFeatureExtractor(
            event_types="SpikeCounts",
            aggregation="single",
            frequency="native",
            target_unit="native",
            query=f"brain_region == '{region}'",
        )
        for region in ("6d", "6v", "44", "M1")
    }

    for extractor in extractors.values():
        extractor.prepare(events)

    outputs = {
        region: extractor(events, start=0.5, duration=1.0)
        for region, extractor in extractors.items()
    }

    assert set(outputs) == {"6d", "6v", "44", "M1"}
    for expected, region in enumerate(("6d", "6v", "44", "M1"), start=1):
        assert outputs[region].shape == (2, 4)
        assert torch.equal(outputs[region], torch.full((2, 4), float(expected)))


def test_query_survives_neuralset_config_round_trip(tmp_path):
    events = [
        _event(tmp_path, value=1, brain_region="6d"),
        _event(tmp_path, value=2, brain_region="6v"),
    ]
    extractor = NeuralFeatureExtractor(
        event_types="SpikeCounts",
        query="brain_region == '6v'",
    )

    restored = ns.BaseExtractor(**extractor.model_dump())
    restored.prepare(events)

    assert isinstance(restored, NeuralFeatureExtractor)
    assert restored.query == "brain_region == '6v'"
    assert torch.equal(
        restored(events, start=0.0, duration=1.0),
        torch.full((2, 4), 2.0),
    )


def test_prepare_rejects_a_query_field_not_shared_by_all_candidate_types(tmp_path):
    events = [
        _event(tmp_path, value=1, brain_region="6d"),
        _event(tmp_path, value=2, brain_region="6d", cls=BandPower),
    ]
    extractor = NeuralFeatureExtractor(
        event_types=("SpikeCounts", "BandPower"),
        query="brain_region == '6d' or method == 'threshold_crossing'",
        aggregation="cat",
    )

    with pytest.raises(ValueError, match="unknown field 'method'"):
        extractor.prepare(events)


def test_prepare_rejects_a_query_that_matches_no_candidate(tmp_path):
    events = [_event(tmp_path, value=1, brain_region="6d")]
    extractor = NeuralFeatureExtractor(
        event_types="SpikeCounts",
        query="brain_region == 'M1'",
    )

    with pytest.raises(ValueError, match="matched none of the 1 event"):
        extractor.prepare(events)


def test_query_cannot_be_combined_with_trigger_aggregation():
    with pytest.raises(ValueError, match="bypassing selection"):
        NeuralFeatureExtractor(
            event_types="SpikeCounts",
            aggregation="trigger",
            query="brain_region == '6d'",
        )


# ---------------------------------------------------------------------------
# Channel picks
# ---------------------------------------------------------------------------


def _three_channel_event(tmp_path):
    """One (3, 8) event whose rows are labeled 0.0, 10.0, 20.0."""
    data = np.stack([np.full(8, 10.0 * i) for i in range(3)]).astype(np.float32)
    path = tmp_path / "three_channel.npy"
    np.save(path, data)
    return SpikeCounts(
        timeline="recording",
        subject="P1",
        feature="threshold_crossings",
        start=0.0,
        duration=2.0,
        frequency=4.0,
        n_features=3,
        unit="counts/bin",
        normalization="raw",
        filepath=str(path),
    )


def test_channels_select_reorder_and_duplicate_rows_by_position(tmp_path):
    event = _three_channel_event(tmp_path)
    extractor = NeuralFeatureExtractor(event_types="SpikeCounts", channels=(2, 0, 0))

    ta = extractor._preprocess(event)

    assert np.all(ta.data == np.stack([np.full(8, v) for v in (20.0, 0.0, 0.0)]))
    # the header names each output row by its SOURCE position, not its output index
    assert (
        ta.header["ch_names"]
        == "threshold_crossings:2,threshold_crossings:0,threshold_crossings:0"
    )


def test_channels_none_keeps_every_row_and_the_plain_header(tmp_path):
    event = _three_channel_event(tmp_path)
    extractor = NeuralFeatureExtractor(event_types="SpikeCounts")

    ta = extractor._preprocess(event)

    assert ta.data.shape == (3, 8)
    assert ta.header["ch_names"] == ",".join(
        f"threshold_crossings:{i}" for i in range(3)
    )


def test_channels_rejects_positions_outside_the_event(tmp_path):
    event = _three_channel_event(tmp_path)
    extractor = NeuralFeatureExtractor(event_types="SpikeCounts", channels=(0, 99))

    with pytest.raises(ValueError, match=r"channels \[99\] are outside the 3 channels"):
        extractor._preprocess(event)


def test_channels_rejects_empty_and_negative_picks():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="must not be empty"):
        NeuralFeatureExtractor(event_types="SpikeCounts", channels=())
    with pytest.raises(ValidationError, match="non-negative"):
        NeuralFeatureExtractor(event_types="SpikeCounts", channels=(0, -1))


def test_channels_default_keeps_the_cache_uid_and_a_pick_changes_it(tmp_path):
    event = _three_channel_event(tmp_path)
    plain = NeuralFeatureExtractor(event_types="SpikeCounts")
    picked = NeuralFeatureExtractor(event_types="SpikeCounts", channels=(1, 2))
    other = NeuralFeatureExtractor(event_types="SpikeCounts", channels=(0, 1))
    plain.prepare([event])
    picked.prepare([event])
    other.prepare([event])

    # exca drops defaults from the uid, so pre-existing channel=None caches survive
    assert "channels" not in plain.infra.uid()
    # but two different picks must never share a cached read
    assert picked.infra.uid() != other.infra.uid()


# ---------------------------------------------------------------------------
# Trial-anchored pairing and leading trim
# ---------------------------------------------------------------------------


def _ramp_event(tmp_path, cls=SpikeCounts):
    """A (2, 8) bin-resolved event: row 0 holds bins 0..7, row 1 holds 100..107."""
    data = np.stack([np.arange(8), 100 + np.arange(8)]).astype(np.float32)
    path = tmp_path / f"ramp_{cls.__name__}.npy"
    np.save(path, data)
    common = dict(
        timeline="recording",
        subject="P1",
        feature="threshold_crossings",
        start=0.0,
        duration=2.0,
        frequency=4.0,
        n_features=2,
        unit="counts/bin",
        normalization="mean_subtracted",
        filepath=str(path),
    )
    if cls is BandPower:
        common.update(feature="spike_band_power", unit="uV^2")
    return cls(**common)


def _trial_trigger(start=0.25, duration=1.5, pair_start=None):
    from neuralset.events import etypes

    extra = {} if pair_start is None else {"pair_start": pair_start}
    return etypes.Sentence(
        timeline="recording",
        text="x",
        start=start,
        duration=duration,
        extra=extra,
    )


def test_pair_from_window_start_sums_spikes_per_pair(tmp_path):
    # window [0.25, 1.75) s = bins 1..6, anchored at the trigger start: pairs
    # (1,2),(3,4),(5,6). Sums are x2 because a pair bin is twice as wide
    # (counts/bin unit conversion); bin 0 and the trailing bin 6 stay out.
    event = _ramp_event(tmp_path)
    extractor = NeuralFeatureExtractor(frequency=2.0, pair_trigger_field="start")

    out = extractor([event], start=0.25, duration=1.5, trigger=_trial_trigger()).numpy()

    assert out.tolist() == [
        [(1 + 2) * 2, (3 + 4) * 2, (5 + 6) * 2],
        [(101 + 102) * 2, (103 + 104) * 2, (105 + 106) * 2],
    ]


def test_pair_from_earlier_anchor_straddles_and_drops_the_remainder(tmp_path):
    # the Kunz phase: anchor (pair_start) at bin 0, window starting at bin 1.
    # The first pair (0,1) reaches one bin before the window; a pair that
    # would end at the first bin AFTER the window (6,7) is dropped.
    event = _ramp_event(tmp_path)
    extractor = NeuralFeatureExtractor(frequency=2.0, pair_trigger_field="pair_start")

    out = extractor(
        [event],
        start=0.25,
        duration=1.5,
        trigger=_trial_trigger(pair_start=0.0),
    ).numpy()

    assert out.tolist() == [
        [(0 + 1) * 2, (2 + 3) * 2, (4 + 5) * 2],
        [(100 + 101) * 2, (102 + 103) * 2, (104 + 105) * 2],
    ]


def test_pairing_averages_band_power_instead_of_summing(tmp_path):
    # uV^2 is non-convertible, so band power pairs average with no unit factor
    event = _ramp_event(tmp_path, cls=BandPower)
    extractor = NeuralFeatureExtractor(frequency=2.0, pair_trigger_field="start")

    out = extractor([event], start=0.25, duration=1.5, trigger=_trial_trigger()).numpy()

    assert out.tolist() == [
        [(1 + 2) / 2, (3 + 4) / 2, (5 + 6) / 2],
        [(101 + 102) / 2, (103 + 104) / 2, (105 + 106) / 2],
    ]


def test_pairing_ignores_the_nominal_window_tail(tmp_path):
    # the trial rows carry a nominal duration (819 x 0.01 s); a 50 Hz buffer
    # sizes itself with round(8.19 * 50) = 410 bins and would zero-pad a
    # phantom bin past the 409 whole pairs — pairing emits whole pairs only
    event = _ramp_event(tmp_path)
    extractor = NeuralFeatureExtractor(frequency=2.0, pair_trigger_field="start")

    out = extractor(
        [event],
        start=0.25,
        duration=1.75,
        trigger=_trial_trigger(start=0.25, duration=1.75),
    ).numpy()

    # bins 1..7 requested, pairs (1,2),(3,4),(5,6): bin 7 is the trailing odd bin
    assert out.tolist() == [
        [(1 + 2) * 2, (3 + 4) * 2, (5 + 6) * 2],
        [(101 + 102) * 2, (103 + 104) * 2, (105 + 106) * 2],
    ]


def test_pairing_grows_the_assembly_to_the_emitted_pairs(tmp_path):
    # 2.25 s at 2 Hz is 4 nominal bins, but the window reaches native bin 10 and
    # pairs anchored at bin 0 leave 5 whole pairs (the 5th zero-padded off the
    # event end): the assembly must cover them, not stop at the nominal 4
    event = _ramp_event(tmp_path)
    extractor = NeuralFeatureExtractor(frequency=2.0, pair_trigger_field="pair_start")

    out = extractor(
        [event],
        start=0.25,
        duration=2.25,
        trigger=_trial_trigger(start=0.25, duration=2.25, pair_start=0.0),
    ).numpy()

    assert out.tolist() == [
        [(0 + 1) * 2, (2 + 3) * 2, (4 + 5) * 2, (6 + 7) * 2, 0],
        [(100 + 101) * 2, (102 + 103) * 2, (104 + 105) * 2, (106 + 107) * 2, 0],
    ]


def test_pairing_needs_an_integer_ratio_and_a_resolvable_anchor(tmp_path):
    from pydantic import ValidationError

    event = _ramp_event(tmp_path)
    with pytest.raises(ValidationError, match="numeric frequency"):
        NeuralFeatureExtractor(pair_trigger_field="start", frequency="native")

    odd = NeuralFeatureExtractor(frequency=1.5, pair_trigger_field="start")
    with pytest.raises(ValueError, match="integer factor >= 2"):
        odd([event], start=0.0, duration=2.0, trigger=_trial_trigger(start=0.0))

    missing = NeuralFeatureExtractor(frequency=2.0, pair_trigger_field="pair_start")
    with pytest.raises(ValueError, match="did not resolve to a time"):
        missing([event], start=0.25, duration=1.5, trigger=_trial_trigger())


def test_pair_mode_keeps_default_caches_and_splits_its_own(tmp_path):
    event = _ramp_event(tmp_path)
    plain = NeuralFeatureExtractor()
    paired = NeuralFeatureExtractor(frequency=2.0, pair_trigger_field="start")

    # exca drops defaults from the uid, so pre-existing grid caches survive
    assert "pair_trigger_field" not in plain.infra.uid()
    # a paired read caches a native-rate payload: it must never share an entry
    assert "pair_trigger_field" in paired.infra.uid()
    # trim is applied per window (like `offset`): never in the uid, no re-read
    assert (
        "trim_leading_bins"
        not in NeuralFeatureExtractor(trim_leading_bins=1).infra.uid()
    )


def test_pair_mode_prepare_populates_shape_without_pairing(tmp_path):
    # neuralset's prepare() dummy window is triggered by the neural event itself
    # (no anchor field), gets the native-rate pair-mode cache back, and must
    # still assemble its channel-shape probe without a frequency mismatch
    event = _ramp_event(tmp_path)
    extractor = NeuralFeatureExtractor(frequency=2.0, pair_trigger_field="pair_start")

    extractor.prepare([event])

    assert extractor._effective_frequency == 2.0
    # pairing afterwards still works with the cache prepare warmed
    out = extractor(
        [event], start=0.25, duration=1.5, trigger=_trial_trigger(pair_start=0.0)
    ).numpy()
    assert out.shape == (2, 3)


def test_trim_leading_drops_first_bins_and_keeps_the_end(tmp_path):
    # the Willet competition window: study window minus its leading bin
    event = _ramp_event(tmp_path)
    extractor = NeuralFeatureExtractor(frequency="native", trim_leading_bins=1)

    out = extractor(
        [event], start=0.0, duration=2.0, trigger=_trial_trigger(start=0.0)
    ).numpy()

    assert np.array_equal(out, event.read()[:, 1:])


def test_trim_leading_rejects_over_trim_and_negative_counts(tmp_path):
    from pydantic import ValidationError

    event = _ramp_event(tmp_path)
    over = NeuralFeatureExtractor(frequency="native", trim_leading_bins=8)
    with pytest.raises(ValueError, match="leaves nothing"):
        over([event], start=0.0, duration=2.0, trigger=_trial_trigger(start=0.0))
    with pytest.raises(ValidationError, match=">= 0"):
        NeuralFeatureExtractor(trim_leading_bins=-1)


# ---------------------------------------------------------------------------
# Phoneme targets
# ---------------------------------------------------------------------------


def _sentence(phonemes: str):
    from neuralset.events import etypes

    return etypes.Event.from_dict(
        dict(
            type="Sentence",
            timeline="recording",
            subject="P1",
            start=0.0,
            duration=2.0,
            text="the cat",
            phonemes=phonemes,
        )
    )


def test_canonical_phoneme_set_is_frozen():
    from neuralshed.text import ARPABET_V1, PhonemeVocab

    vocab = PhonemeVocab.canonical()
    assert vocab.n_classes == len(ARPABET_V1) + 1 == 41
    assert vocab.BLANK == 0 and vocab.symbols[-1] == "|"
    # id order is part of the contract: AA is 1, | is 40
    assert vocab.encode("AA | ZH").tolist() == [1, 40, 39]
    assert vocab.decode([1, 40, 39]) == "AA | ZH"


def test_phoneme_vocab_rejects_unknown_symbols():
    from neuralshed.text import PhonemeVocab

    vocab = PhonemeVocab.canonical()
    assert vocab.coverage(["AA XX", "YY"]) == {"XX": 1, "YY": 1}
    with pytest.raises(KeyError, match="XX"):
        vocab.encode("AA XX")


def test_phoneme_encoder_reads_the_extra_column_and_raises_on_unknown():
    from neuralshed.extractors import PhonemeEncoder

    enc = PhonemeEncoder()
    out = enc.get_static(_sentence("DH AH | K AE T |"))
    assert out.dtype == torch.int64
    assert out.tolist() == [10, 3, 40, 20, 2, 31, 40]
    with pytest.raises(KeyError):
        enc.get_static(_sentence("DH AH | K AE T XX"))


def _cached_whisper_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.WhisperTokenizer.from_pretrained(
            "openai/whisper-tiny.en",
            language="en",
            task="transcribe",
            local_files_only=True,
        )
    except Exception:
        pytest.skip("the pinned Whisper tokenizer is not cached locally")


def test_whisper_label_encoder_strips_prefix_token_and_keeps_end_token():
    tokenizer = _cached_whisper_tokenizer()
    encoder = WhisperLabelEncoder()
    sentence = ns.events.etypes.Sentence(
        start=1.5, timeline="recording", duration=1.0, text="the cat sat on the mat"
    )

    ids = encoder.get_static(sentence)

    assert ids.dtype == torch.int64
    assert ids.ndim == 1
    assert ids[:3].tolist() == [50258, 50358, 50362]
    reference = tokenizer(
        "the cat sat on the mat",
        truncation=True,
        max_length=448,
        return_tensors="pt",
    )["input_ids"][0]
    assert torch.equal(ids, reference[1:].to(torch.int64))
    assert ids.numel() == reference.numel() - 1
    assert int(ids[-1]) == int(tokenizer.eos_token_id)


def test_whisper_label_encoder_truncates_at_configured_length():
    _cached_whisper_tokenizer()
    encoder = WhisperLabelEncoder()
    sentence = ns.events.etypes.Sentence(
        start=0.0, timeline="recording", duration=60.0, text=" ".join(["speech"] * 600)
    )

    ids = encoder.get_static(sentence)
    assert ids.numel() == 447
