"""neuralset ``Study`` for Kunz et al. 2025, inner speech in motor cortex.

Four BrainGate2 participants (T12/T15/T16/T17, severe dysarthria/anarthria), 64-channel
Utah arrays in the left precentral gyrus (and, for T12, area 44), performing
instructed-delay verbal-behavior, sentence-production, counting and sequence-recall tasks
to probe inner (imagined) speech.

A source ``.mat`` concatenates the blocks of one task on one day, and neural activity
between blocks was not recorded, so **one block is one** :term:`timeline` (243 over the 41
timeline files; the 7 per-behavior ``isolatedVerbalBehaviors`` subsets are dropped in
favour of their combined ``*_allBehaviors`` file). ``utils.block_segments`` reads the
blocks from ``blockNum`` and, for the seqRecall files that ship none, from the checked-in
``utils.RECOVERED_SEQ_BLOCK_STOPS`` (positional block ids). Per block:

* two pre-binned neural-feature streams, ``SpikeCounts`` (``threshold_crossings``,
  binnedTX) and ``BandPower`` (``spike_band_power``, spikePow), each
  ``(n_channels, n_bins)`` at the file's bin rate, every stored column, arrays in
  ``utils.ARRAY_LAYOUTS`` order. SpikeCounts first, so ``aggregation="cat"`` reproduces
  the paper's native [binnedTX | spikePow] order. Three T12 seqRecall files store signed
  counts under a ``uint8`` tag and are read as ``int8`` (noted in ``processing``); the one
  file whose ``spikePow`` is a byte copy of ``binnedTX`` carries ``valid=False`` on its
  ``BandPower`` rows;
* per trial, the task periods (``period``): a ``Stimulus`` over the **delay** (the cue,
  ``stimulus_kind`` ``word_cue`` with ``code`` 1-7 into ``utils.WORDS``, ``sentence_cue``,
  ``movement_cue``, ``speech_delay_cue`` for spoken sequences, ``counting_grid`` or
  ``rest_cue``), a one-bin go-cue ``Stimulus`` (``speech_go_cue`` / ``movement_go_cue``,
  none in the self-paced counting task), the cued unit over the **go** period, and an
  ``Intertrial`` over any gap before the next trial of the block (``description``
  ``"return period"`` in the T12 lines and memory files, ``"intertrial interval"``
  elsewhere). The unit is a ``Word`` (``text``, canonical ``behavior``, ``modality``
  heard, read or imagined), a ``Sentence`` (``text`` with the chitty ``*`` spelled out as
  ``utils.KEYWORD``, ``keyword``, ``behavior``), or an ``Action`` (``action_kind``
  ``joystick`` / ``drawing`` / ``speech`` for the seqRecall go periods with the file-local
  ``code``, ``speech`` with ``num_targets`` for the spoken count, ``rest`` with ``code=0``
  / ``DO_NOTHING`` for the rest trials of every family). Passive trials (isolated listening
  and reading blocks, T12's interleaved listening trials) have no cue or go marker: their
  word sits over the source's delay epoch (``period="presentation"``) and the
  ``Intertrial`` runs from the source's go start to the next trial. Every trial row carries
  ``trial_id`` (row of the file's epochs), ``period``, ``task_condition`` (``imagined`` on
  the delay of the verbal-memory strategy, ``na`` on the other cues and on intertrials),
  ``cue_text`` (verbatim source cue) and ``pair_start`` (the trial's delay start in
  seconds from the block start — the anchor the published builder paired its spike and
  power bins from, so consumers can re-pair per trial). seqRecall and counting rows carry
  the delay ``strategy``.

``block``, ``date``, ``post_implant_day``, ``experiment``, ``vocab`` (``50w`` / ``125k``
for the sentence families, ``na`` elsewhere), ``loop_type``, ``original_split`` (``train``
/ ``eval`` for the sentence families, ``""`` elsewhere) and ``mean_subtracted`` ride on
the timeline. The five T16 seqRecall files are read at 40 ms against a stored 20
(``utils.T16_SEQ_BIN_MS``), noted in ``processing``. Times are relative to the block start
and come from the bin epochs, so cue windows align to the feature bins. The out-of-scope
``decodedSentenceText`` / ``sentenceFeatures`` are never read.
"""

import typing as tp

import numpy as np
import pandas as pd
from neuralset.events import study

from neuralshed.datasets import participants
from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401, registers the custom event types
    BandPower,
    Intertrial,
    SpikeCounts,
)

from . import utils

# The blocks the ``_info`` query selects, one per emission path: chitty, T12 interleaved
# (listening as presentation), a T12 passive isolated block, sentenceData, audio-cued speech
# recall, a uint8-tagged joystick block, a T12 lines block (return periods), counting, a T16
# passive isolated block (intertrials to the next trial) and a T16 seqRecall block on the
# inferred 40 ms clock.
_INFO_BLOCKS = (
    ("t12.2024.12.06_chittychittybangbang_50words", 1),
    ("t12.2024.04.11_interleavedVerbalBehaviors_raw", 5),
    ("t12.2023.08.15_allBehaviors_raw", 4),
    ("t12.2023.11.28_attempted_50words", 1),
    ("t12.2023.09.28_binnedTX_spikePow_20ms_speak", 1),
    ("t12.2023.05.16_binnedTX_spikePow_20ms_joystickSeq", 1),
    ("t12.2023.09.19_binnedTX_spikePow_20ms_lineSequenes", 1),
    ("t15.2024.12.15_conjunctive_counting", 6),
    ("t16.2024.03.04_allBehaviors_raw", 19),
    ("t16.2024.04.08_binnedTX_spikePow_20ms", 1),
)


class Kunz2025InnerSpeech(NeuralshedStudy):
    """Kunz et al. 2025, inner speech in motor cortex (BrainGate2 T12/T15/T16/T17).

    One timeline per recorded block (243 across 41 files, 11 task types, 4 participants).
    Each emits two neural streams (``SpikeCounts`` + ``BandPower``, 128/256 ch, 25-100 Hz)
    and, per trial, a delay ``Stimulus``, a go-cue ``Stimulus``, the cued unit
    (``Word``/``Sentence``/``Action``) over the go period and an ``Intertrial`` over any gap
    before the next trial.
    """

    dataset_id: tp.ClassVar[str] = "kunz_2025"
    aliases: tp.ClassVar[tuple[str, ...]] = ("kunz_2025", "inner-speech-2025")
    bibtex: tp.ClassVar[str] = """
@article{kunz2025innerspeech,
  title = {Inner speech in motor cortex and implications for speech neuroprostheses},
  author = {Kunz, Erin M. and Abramovich Krasa, Benyamin and Kamdar, Foram and Avansino, Donald T. and Hahn, Nick and Yoon, Seonghyun and Singh, Akansha and Nason-Tomaszewski, Samuel R. and Card, Nicholas S. and Jude, Justin J. and Jacques, Brandon G. and Bechefsky, Payton H. and Iacobacci, Carrina and Hochberg, Leigh R. and Rubin, Daniel B. and Williams, Ziv M. and Brandman, David M. and Stavisky, Sergey D. and AuYong, Nicholas and Pandarinath, Chethan and Druckmann, Shaul and Henderson, Jaimie M. and Willett, Francis R.},
  journal = {Cell},
  volume = {188},
  number = {17},
  pages = {4658–4673.e17},
  year = {2025},
  month = aug,
  doi = {10.1016/j.cell.2025.06.015},
  url = {https://doi.org/10.1016/j.cell.2025.06.015},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.gf1vhhn1j"
    licence: tp.ClassVar[str] = "CC0-1.0"  # Dryad data deposit (CC0)
    description: tp.ClassVar[str] = (
        "Instructed-delay inner-speech tasks (seven verbal behaviors, sentence production, "
        "conjunctive counting, sequence recall) by four BrainGate2 participants with severe "
        "dysarthria/anarthria, 64-channel Utah arrays in the left precentral gyrus (and T12's "
        "left inferior frontal gyrus). Per-file binned threshold crossings + spike-band "
        "power (128/256 ch) with per-trial word / sentence / movement-sequence cues. 69 "
        "files, 243 block timelines (Kunz et al. 2025, Cell 188)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("scipy",)
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=243,
        num_subjects=4,
        query=" or ".join(
            f"(session == '{session}' and block == {block})"
            for session, block in _INFO_BLOCKS
        ),
        # 10 x 2 neural, 602 Stimulus, 251 Intertrial, 227 Action, 140 Word, 80 Sentence
        num_events_in_query=1320,
        event_types_in_query={
            "SpikeCounts",
            "BandPower",
            "Stimulus",
            "Action",
            "Word",
            "Sentence",
            "Intertrial",
        },
        frequency=50.0,  # the modal rate (149 of 243 timelines), others 25, 30.3, 40, 100 Hz
    )

    # -- data access ------------------------------------------------------
    def _download(self, overwrite: bool = False) -> None:
        """Fetch the Dryad deposit into ``self.path`` (12 assets: one .zip per task type
        + README.md, ~10.8 GB extracted into the task subfolders). Needs a Dryad API token
        (``NEURALFETCH_DRYAD_TOKEN``).
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="kunz_2025",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.gf1vhhn1j",
        ).download(overwrite=overwrite)

    def _abspath(self, timeline: dict[str, tp.Any]):
        return self.path / timeline["path"]

    def _segment(self, timeline: dict[str, tp.Any]) -> utils.BlockSegment:
        for seg in utils.block_segments(self._abspath(timeline)):
            if seg.block == timeline["block"]:
                return seg
        raise ValueError(f"block {timeline['block']} not in {timeline['path']}")

    # -- timelines --------------------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for p in utils.list_timeline_files(self.path):
            subject = utils.participant_of_path(p)
            date = utils.session_date(p)
            exp = utils.experiment_of_path(p)
            for seg in utils.block_segments(p):
                yield dict(
                    subject=subject,
                    session=utils.session_label(p),  # file stem, matches sessions.label
                    date=date,
                    post_implant_day=participants.post_implant_day(
                        f"braingate2/{subject}", date
                    ),
                    path=str(p.relative_to(self.path)),
                    block=seg.block,
                    experiment=exp,
                    vocab=utils.sentence_vocab(p),
                    loop_type=utils.loop_type(p),
                    original_split=utils.original_split(p),
                    mean_subtracted=utils.is_mean_subtracted(p),
                )

    # -- neural streams ---------------------------------------------------
    def _load_stream(self, timeline: dict[str, tp.Any], feature: str) -> np.ndarray:
        """SpecialLoader target: one neural stream over the block, ``(n_ch, n_bins)``."""
        seg = self._segment(timeline)
        return utils.get_neural_stream(
            self._abspath(timeline), feature, seg.start_bin, seg.stop_bin
        )

    def _neural_rows(
        self,
        timeline: dict[str, tp.Any],
        rate: float,
        n_bins: int,
    ) -> list[dict[str, tp.Any]]:
        p = self._abspath(timeline)
        ms = timeline["mean_subtracted"]
        subject = timeline["subject"]
        # The paper states the per-block refit for T15's sentence sessions only.
        refit = subject == "T15" and timeline["experiment"] == "sentenceData"
        chain = (
            f"250 Hz non-causal high-pass ({utils.HIGHPASS_DELAY_MS[subject]} ms delay); "
            "linear regression referencing "
            + (
                "refit after every block"
                if refit
                else "fit on the session's initial diagnostic or rest block"
            )
        )
        if ms:
            chain += "; mean subtracted per block"
        if utils.bin_size_is_inferred(p):
            chain += (
                f"; {utils.T16_SEQ_BIN_MS} ms bins inferred, the file stores binSize=20"
            )
        processing = {"threshold_crossings": chain, "spike_band_power": chain}
        if utils.reinterpreted_uint8(p):
            note = "; uint8 class tag reinterpreted as int8"
            processing["threshold_crossings"] += note
        reason = utils.band_power_invalid_reason(p)
        flags = {"valid": False, "valid_description": reason} if reason else {}

        def _row(*, type_: str, feature: str, unit: str, **extra) -> dict[str, tp.Any]:
            return dict(
                type=type_,
                feature=feature,
                n_features=utils.n_channels(p),
                unit=unit,
                normalization="mean_subtracted" if ms else "raw",
                processing=processing[feature],
                start=0.0,
                duration=n_bins / rate,
                frequency=rate,
                filepath=study.SpecialLoader(
                    method=self._load_stream,
                    timeline=timeline,
                    feature=feature,
                ).to_json(),
                **extra,
            )

        # SpikeCounts first, then BandPower: native [binnedTX | spikePow] under cat.
        return [
            _row(
                type_="SpikeCounts",
                feature="threshold_crossings",
                unit="counts/bin",
                method="threshold_crossing",
                threshold=f"{utils.THRESHOLD_SD[subject]} x SD",
            ),
            _row(
                type_="BandPower",
                feature="spike_band_power",
                unit="uV^2",
                lower_freq=250.0,
                upper_freq=5000.0,  # corpus convention: the source only high-passes
                **flags,
            ),
        ]

    # -- trial events -----------------------------------------------------
    @staticmethod
    def _unit_row(family: str, tr: tp.Any) -> dict[str, tp.Any]:
        """The cued unit of one trial: text, or an ``Action`` for movements, counts and rest."""
        rest = dict(type="Action", action_kind="rest", code=0, description="DO_NOTHING")
        if family == utils.FAMILY_WORD_CUE:
            if not tr.word:  # Word.text must be non-empty
                return dict(behavior=tr.behavior, **rest)
            return dict(
                type="Word",
                language="en",
                text=tr.word,
                modality=tr.text_modality,
                behavior=tr.behavior,
            )
        if family == utils.FAMILY_SENTENCE:
            return dict(
                type="Sentence",
                language="en",
                text=tr.sentence_text,
                modality=tr.text_modality,
                keyword=bool(tr.keyword),
                behavior=tr.behavior,
                phonemes=utils.CUE_PHONEMES.cues.get(tr.sentence_text),
            )
        if family == utils.FAMILY_COUNTING:  # the spoken report of the silent count
            return dict(
                type="Action",
                action_kind="speech",
                description=f"count {int(tr.num_targets)}",
                code=int(tr.num_targets),
                num_targets=int(tr.num_targets),
            )
        if tr.rest:
            return rest
        return dict(
            type="Action",
            action_kind=tr.action_kind,
            description=tr.cue_text,
            code=int(tr.cue_id),  # index into this file's cueList
        )

    @staticmethod
    def _delay_row(family: str, experiment: str, tr: tp.Any) -> dict[str, tp.Any]:
        """The cue shown (or played) over the delay period."""
        row = dict(type="Stimulus", modality="visual", description=tr.cue_text)
        if family == utils.FAMILY_COUNTING:
            row.update(
                stimulus_kind="counting_grid",
                description=f"count {int(tr.num_targets)}",
                code=int(tr.num_targets),
                num_targets=int(tr.num_targets),
            )
        elif family == utils.FAMILY_SEQUENCE:
            speech = experiment == "seqRecallSpeech"
            row.update(
                stimulus_kind="speech_delay_cue" if speech else "movement_cue",
                modality=tr.stimulus_modality,
                code=int(tr.cue_id),
            )
        elif family == utils.FAMILY_WORD_CUE:
            code = utils.WORDS.index(tr.word) + 1 if tr.word else 0  # 1-based, 0 = rest
            row.update(stimulus_kind="word_cue", code=code)
        else:
            row.update(stimulus_kind="sentence_cue")
        rest = getattr(tr, "rest", False) or getattr(tr, "behavior", "") == utils.REST
        if rest:
            row.update(stimulus_kind="rest_cue", code=0, description="DO_NOTHING")
        return row

    @staticmethod
    def _go_marker(family: str, experiment: str) -> dict[str, tp.Any]:
        movement = family == utils.FAMILY_SEQUENCE and experiment != "seqRecallSpeech"
        return dict(
            type="Stimulus",
            # the lines task used an audible go cue, the others a colour change
            modality="audio" if experiment.endswith("3ElementLines") else "visual",
            description="go cue",
            stimulus_kind="movement_go_cue" if movement else "speech_go_cue",
            task_condition="na",
        )

    def _trial_rows(
        self,
        timeline: dict[str, tp.Any],
        trials: pd.DataFrame,
        bin_s: float,
        origin: int,
    ) -> list[dict]:
        """Per trial: delay cue, go marker, the unit over the go period, then an
        ``Intertrial`` over any gap before the next trial of the block.

        A passive trial has its unit over the source's delay epoch (``period="presentation"``),
        no cue or go marker, and an ``Intertrial`` from the source's go start to the next trial.
        """
        family = utils.family_of_path(timeline["path"])
        experiment = timeline["experiment"]
        returns = (
            timeline["subject"] == "T12" and experiment in utils.RETURN_PERIOD_FILES
        )
        gap = dict(
            type="Intertrial",
            description="return period" if returns else "intertrial interval",
            period="intertrial",
            task_condition="na",
        )

        def span(a: int, b: int) -> dict[str, float]:
            return dict(start=(a - origin) * bin_s, duration=(b - a) * bin_s)

        rows: list[dict[str, tp.Any]] = []
        for tr in trials.itertuples(index=False):
            ids = dict(trial_id=int(tr.trial_id), cue_text=tr.cue_text)
            if family == utils.FAMILY_SEQUENCE:
                ids["strategy"] = tr.strategy
            elif family == utils.FAMILY_COUNTING:
                ids["strategy"] = "uninstructed"
            unit = dict(task_condition=tr.task_condition, **self._unit_row(family, tr))
            if unit.get("action_kind") == "rest":
                unit["task_condition"] = "na"
            delay = (tr.delay_start_bin, tr.delay_stop_bin)
            go = (tr.go_start_bin, tr.go_stop_bin)
            # the published builder concatenated delay+go epochs, paired spikes/power
            # from the delay start, then dropped the leading delay pairs; stamp that
            # anchor so an extractor can reproduce the pairing phase per trial
            ids["pair_start"] = (delay[0] - origin) * bin_s
            if tr.passive:
                rows.append(dict(**unit, **span(*delay), period="presentation", **ids))
                # T16 stores a one- or two-bin go stub, so its gap runs to the next trial
                if tr.next_start_bin - go[0] > 2:
                    rows.append(dict(**gap, **span(go[0], tr.next_start_bin), **ids))
                continue
            verbal = ids.get("strategy") == "verbal" and not tr.rest
            rows.append(
                dict(
                    **self._delay_row(family, experiment, tr),
                    task_condition="imagined" if verbal else "na",
                    **span(*delay),
                    period="delay",
                    **ids,
                )
            )
            if (
                family != utils.FAMILY_COUNTING
            ):  # the count report starts on a button press
                rows.append(
                    dict(
                        **self._go_marker(family, experiment),
                        **span(go[0], go[0] + 1),
                        period="go",
                        **ids,
                    )
                )
            rows.append(dict(**unit, **span(*go), period="go", **ids))
            if tr.next_start_bin > go[1]:
                rows.append(dict(**gap, **span(go[1], tr.next_start_bin), **ids))
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        p = self._abspath(timeline)
        meta = utils.load_meta(str(p))
        rate = utils.rate_hz(meta)
        seg = self._segment(timeline)
        trials = utils.get_trials(p, meta)
        trials = trials[trials.block == seg.block]
        if trials.empty:
            raise ValueError(f"{p}: block {seg.block} holds no trial")
        rows = self._neural_rows(timeline, rate, seg.stop_bin - seg.start_bin)
        rows += self._trial_rows(timeline, trials, 1.0 / rate, seg.start_bin)
        return pd.DataFrame(rows)
