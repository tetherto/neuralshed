"""neuralset ``Study`` for Jude et al. 2026, intended speech in anarthria (T17).

BrainGate2 participant T17 (advanced ALS, anarthria, locked-in), six 64-channel Utah
arrays (384 ch; areas 6v/55b/6d) in left precentral gyrus, attempting or intending
speech at the orofacial, phoneme, word and sentence levels across 14 sessions.

Each ``.mat`` block is a continuous recording with many cued trials inside it, so one
block is one :term:`timeline` (121 catalogued task blocks; the 19 files in
``utils.EXCLUDED_BLOCKS``, reference, diagnostic, aborted and unreadable blocks, are
skipped). Per block the Study emits:

* two neural-feature streams over the whole block, ``SpikeCounts``
  (``threshold_crossings``) and ``BandPower`` (``spike_band_power``), each
  ``(n_ch, n_bins)`` at 100 Hz (``n_ch`` read from the file: 384, or 256 on
  2024.04.15/16/17), channels permuted from file order into ``utils.ARRAY_ORDER`` (one
  block of 64 per array: ventral 6v, dorsal 6v, ventral 55b, dorsal 55b, ventral 6d,
  dorsal 6d) so a column is the same electrode in every timeline. SpikeCounts first, so
  ``aggregation="cat"`` reproduces the source's [threshold_crossings | spike_band_power]
  feature order;
* per trial, a ``Stimulus`` for the on-screen cue over the delay (trial start to go),
  a one-bin ``Stimulus`` go marker, and the cued unit over the go window: ``Phoneme``
  (ARPAbet target of the cued syllable, ``utils.PHONEME_CUES``), ``Word``, ``Sentence``
  (``text``, ``modality="read"``), or an ``Action`` for the orofacial movements, the
  four ``Say`` syllables and the rest cue. ``code`` is the dataset-wide class id
  (``utils.CUE_CODES``), ``trial_id`` the 0-based trial within the block, ``success``
  the source's paused and timed-out flags. The two trials the recording cuts short
  carry ``valid=False``;
* ``Intertrial`` rows over the gaps the source clocks delimit: between trials of the
  closed-loop blocks (decoder output and text-to-speech playback), and the 2 s before
  each cue of the phased session.

The phased session (2024.05.23) emits the unit three times per trial, over the
``read``, ``imagined`` and ``attempted`` phases (``task_condition`` and ``phase``),
with a delay ``Stimulus`` and a go marker before the imagined and attempted phases.

``task_family``, ``vocab`` (50w, 125k or na), ``loop_type`` (open or closed), ``date``
and ``post_implant_day`` ride the timeline. Event windows are neural-bin indices
(nearest bin to each NSP marker), so they align exactly to the 100 Hz feature grid. The
30 kHz microphone audio is anarthric and is surfaced via ``utils.get_audio`` rather than
as an event.
"""

import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401, registers the custom event types
    BandPower,
    Intertrial,
    SpikeCounts,
)

from . import utils

_SUBJECT = "T17"
_RATE_HZ = float(utils.FS_NEURAL)  # 100 Hz (10 ms bins)
_BIN_S = 1.0 / _RATE_HZ
_THRESHOLD_SD = f"{utils.THRESHOLD_SD} x SD"
_SPIKE_BAND_HZ = (250.0, 5000.0)
# Gaps shorter than this between a trial end and the next trial start are clock jitter
# (open-loop blocks abut by 0-10 ms), not an inter-trial interval.
_MIN_INTERTRIAL_S = 0.1
# A trial whose end marker lies this far past the last neural bin was cut short.
_TRUNCATION_TOLERANCE_S = 0.05

# What the acquisition chain did before binning (STAR Methods and the deposit README).
_FILTER = (
    "non-causal fourth-order 250 Hz-5 kHz Butterworth (scipy filtfilt) on the 30 kHz "
    "signal; linear regression referencing per array, coefficients fit on the "
    "session's initial reference block"
)
_PROCESSING = {
    "threshold_crossings": (
        f"{_FILTER}; thresholds at -3.5 x SD per electrode, fit on the referenced "
        "reference-block data; crossings counted per 10 ms bin"
    ),
    "spike_band_power": (
        f"{_FILTER}; sum of squared voltages per 10 ms bin (the deposit README says "
        "mean); values capped at 50,000 uV^2"
    ),
}

_UNIT_TYPES = {
    "phoneme": "Phoneme",
    "word": "Word",
    "sentence": "Sentence",
    "orofacial": "Action",
}


def _unit_type(task_family: str) -> str:
    """neuralset event type of a task family's cued unit."""
    for key, type_ in _UNIT_TYPES.items():
        if key in task_family:
            return type_
    raise ValueError(f"unknown task family {task_family!r}")


def _task_attrs(task_family: str) -> dict[str, str]:
    """``vocab`` and ``loop_type`` of a task family."""
    if "50w" in task_family or task_family == "word":
        vocab = "50w"
    elif "125k" in task_family:
        vocab = "125k"
    else:
        vocab = "na"
    loop = "closed" if task_family.endswith("_cl") else "open"
    return {"vocab": vocab, "loop_type": loop}


class Jude2026IntendedSpeech(NeuralshedStudy):
    """Jude et al. 2026, intended/attempted speech iBCI (BrainGate2 T17, anarthria).

    One timeline per catalogued block (121 across 14 sessions). Each emits two neural
    streams (``SpikeCounts`` + ``BandPower``, 384/256 ch, 100 Hz), per trial a cue
    ``Stimulus``, a go marker and the cued unit (``Phoneme``/``Word``/``Sentence``, or
    ``Action`` for movements and rest), plus ``Intertrial`` gaps; the phased session
    adds the read/imagined/attempted contrast on ``task_condition``.
    """

    dataset_id: tp.ClassVar[str] = "jude_2026_speech"
    aliases: tp.ClassVar[tuple[str, ...]] = ("jude_2026_speech", "anarthria-speech-bci")
    bibtex: tp.ClassVar[str] = """
@article{jude2026anarthria,
  title = {Decoding intended speech with an intracortical brain-computer interface in a person with long-standing anarthria and locked-in syndrome},
  author = {Jude, Justin J. and Haro, Stephanie and Levi-Aharoni, Hadar and Hashimoto, Hiroaki and Acosta, Alexander J. and Card, Nicholas S. and Wairagkar, Maitreyee and Brandman, David M. and Stavisky, Sergey D. and Williams, Ziv M. and Cash, Sydney S. and Simeral, John D. and Hochberg, Leigh R. and Rubin, Daniel B.},
  journal = {Cell Reports},
  volume = {45},
  number = {4},
  pages = {117162},
  year = {2026},
  month = apr,
  doi = {10.1016/j.celrep.2026.117162},
  url = {https://doi.org/10.1016/j.celrep.2026.117162},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.vq83bk481"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Cued intended/attempted speech by BrainGate2 participant T17 (ALS, anarthria, "
        "locked-in), six 64-channel Utah arrays (384 ch; areas 6v/55b/6d) in left "
        "precentral gyrus. Per-block 10 ms-binned threshold crossings + spike-band power "
        "(384 or 256 ch, 100 Hz) with per-trial orofacial/phoneme/word/sentence cues and a "
        "phased read/internal/attempted-speech task. 14 sessions, 121 task blocks "
        "(Jude et al. 2026, Cell Reports 45, 117162)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("scipy",)

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=121,
        num_subjects=1,
        # one block per task family, the early, 256-channel and late layouts, the truncated
        # phoneme block, both short-sentence phased blocks and a late closed-loop block
        query=(
            "timeline_index in [0, 3, 9, 14, 23, 27, 40, 41, 61, 64, 92, 93, 94, 95, "
            "96, 111]"
        ),
        num_events_in_query=3544,  # 16 x 2 neural, 1940 Stimulus, 1182 units, 352 Intertrial
        event_types_in_query={
            "SpikeCounts",
            "BandPower",
            "Stimulus",
            "Phoneme",
            "Word",
            "Sentence",
            "Action",
            "Intertrial",
        },
        frequency=_RATE_HZ,
    )

    # -- data access ------------------------------------------------------
    def _download(self, overwrite: bool = False) -> None:
        """Fetch the Dryad deposit (``Data.zip`` -> ``t17/``, ``README.md``) into ``self.path``.

        Without ``overwrite``, an extracted tree that passes ``utils.verify_tree`` (README
        digest against Dryad, every catalogued block file present and readable) is
        accepted as downloaded and the success marker records what was verified. Any
        other state transfers the deposit, which needs ``NEURALFETCH_DRYAD_TOKEN``.
        """
        import datetime as dt

        from neuralshed.datasets.dryad import Dryad, latest_files

        dryad = Dryad(
            study="jude_2026_speech",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi=utils.DRYAD_DOI,
            include=list(utils.DRYAD_FILES),
        )
        if not overwrite:
            digest = latest_files(utils.DRYAD_DOI).get("README.md", (0, ""))[1]
            problems = utils.verify_tree(self.path, readme_digest=digest)
            if not problems:
                n_blocks = sum(len(b) for b in utils.expected_blocks().values())
                dryad.get_success_file().write_text(
                    f"verified {dt.date.today().isoformat()}: README.md sha-256 matches "
                    f"Dryad, {n_blocks} block files present in 14 sessions, all but "
                    "t17.2024.05.23 block 7 (EXCLUDED_BLOCKS, unreadable) open\n"
                )
                return
            print("jude_2026_speech tree incomplete, fetching:", *problems, sep="\n  ")
        dryad.download(overwrite=overwrite)

    @property
    def _t17(self):
        return self.path / "t17"

    # -- timelines + events ----------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for session in utils.list_sessions(self._t17):
            fam_map = utils.SESSION_BLOCKS.get(session, {})
            for block, _path in utils.list_blocks(session, self._t17):
                fam = fam_map.get(block)
                if fam is None:
                    if utils.exclusion_reason(session, block) is None:
                        raise ValueError(
                            f"{session} block {block} is neither in SESSION_BLOCKS "
                            "nor in EXCLUDED_BLOCKS"
                        )
                    continue
                attrs = _task_attrs(fam)
                yield dict(
                    subject=_SUBJECT,
                    session=session,
                    date=utils.session_date(session),
                    post_implant_day=utils.post_implant_day(session),
                    block=block,
                    task_family=fam,
                    vocab=attrs["vocab"],
                    loop_type=attrs["loop_type"],
                )

    def _load_features(self, timeline: dict[str, tp.Any], stream: str) -> np.ndarray:
        """SpecialLoader target: one stream over the block as ``(n_ch, n_bins)``, Study order."""
        session = timeline["session"]
        path = utils.block_path(session, timeline["block"], self._t17)
        key = (
            "binned_neural_threshold_crossings"
            if stream == "threshold_crossings"
            else "binned_neural_spike_band_power"
        )
        d = utils.read_block(path, fields=[key])
        data = utils.to_study_order(np.asarray(d[key]), session)
        return np.ascontiguousarray(data.T.astype(np.float32))

    def _neural_rows(
        self, timeline: dict[str, tp.Any], path, n_bins: int
    ) -> list[dict[str, tp.Any]]:
        n_ch = utils.n_channels(path)
        expected = utils.session_n_channels(timeline["session"])
        if n_ch != expected:
            raise ValueError(f"{path}: {n_ch} channels, expected {expected}")

        def _row(*, type_: str, feature: str, **extra) -> dict[str, tp.Any]:
            return dict(
                type=type_,
                feature=feature,
                n_features=n_ch,
                start=0.0,
                duration=n_bins * _BIN_S,
                frequency=_RATE_HZ,
                normalization="raw",
                processing=_PROCESSING[feature],
                filepath=study.SpecialLoader(
                    method=self._load_features, timeline=timeline, stream=feature
                ).to_json(),
                **extra,
            )

        return [
            _row(
                type_="SpikeCounts",
                feature="threshold_crossings",
                method="threshold_crossing",
                threshold=_THRESHOLD_SD,
                unit="counts/bin",
            ),
            _row(
                type_="BandPower",
                feature="spike_band_power",
                lower_freq=_SPIKE_BAND_HZ[0],
                upper_freq=_SPIKE_BAND_HZ[1],
                unit="uV^2",
            ),
        ]

    # -- per-trial rows ---------------------------------------------------
    @staticmethod
    def _span(a: int, b: int) -> dict[str, float]:
        return dict(start=a * _BIN_S, duration=max(0, b - a) * _BIN_S)

    @staticmethod
    def _unit_row(
        type_: str, cue: str, task_condition: str, modality: str
    ) -> dict[str, tp.Any]:
        """The cued unit: a text event, or an ``Action`` for movements and rest."""
        row: dict[str, tp.Any] = dict(
            type=type_, description=cue, task_condition=task_condition
        )
        if cue in utils.REST_CUES:
            row.update(type="Action", action_kind="rest", task_condition="na")
        elif type_ == "Action":
            kind = "speech" if cue in utils.SAY_CUES else "orofacial_movement"
            if cue not in utils.SAY_CUES and cue not in utils.OROFACIAL_CUES:
                raise ValueError(f"unknown orofacial cue {cue!r}")
            row["action_kind"] = kind
        else:
            text = utils.PHONEME_CUES[cue] if type_ == "Phoneme" else cue
            phonemes = text if type_ == "Phoneme" else utils.CUE_PHONEMES[cue]
            row.update(text=text, phonemes=phonemes, language="en", modality=modality)
        if row["type"] == "Action":
            row["code"] = utils.cue_code(cue)
        return row

    @staticmethod
    def _cue_stimulus(cue: str, kind: str) -> dict[str, tp.Any]:
        return dict(
            type="Stimulus",
            modality="visual",
            description=cue,
            stimulus_kind=kind,
            code=utils.cue_code(cue),
        )

    @staticmethod
    def _go_marker(task_family: str) -> dict[str, tp.Any]:
        kind = "movement_go_cue" if "orofacial" in task_family else "speech_go_cue"
        return dict(
            type="Stimulus",
            modality="visual",
            description="go cue",
            stimulus_kind=kind,
            code=utils.NO_CODE,
        )

    @staticmethod
    def _cue_kind(task_family: str, cue: str) -> str:
        if cue in utils.REST_CUES:
            return "rest_cue"
        if cue in utils.SAY_CUES:
            return "syllable_cue"
        if "orofacial" in task_family:
            return "movement_cue"
        return f"{_unit_type(task_family).lower()}_cue"

    def _trial_rows(
        self,
        task_family: str,
        trials: pd.DataFrame,
        bin_times: np.ndarray,
    ) -> list[dict[str, tp.Any]]:
        """Cue ``Stimulus``, go marker and unit per trial, ``Intertrial`` between trials."""
        type_ = _unit_type(task_family)
        last_bin_ns = float(bin_times[-1])
        rows: list[dict[str, tp.Any]] = []
        for i in range(len(trials)):
            tr = trials.iloc[i]
            cue = str(tr["cue"])
            if not cue:
                raise ValueError(f"trial {i} has an empty cue")
            ids = dict(trial_id=i)
            start = utils.bin_index(bin_times, tr["trial_start_ns"])
            go = utils.bin_index(bin_times, tr["go_cue_ns"])
            end = utils.bin_index(bin_times, tr["trial_end_ns"])
            rows.append(
                dict(
                    **self._cue_stimulus(cue, self._cue_kind(task_family, cue)),
                    **self._span(start, go),
                    **ids,
                )
            )
            rows.append(
                dict(**self._go_marker(task_family), **self._span(go, go + 1), **ids)
            )
            unit = dict(
                **self._unit_row(type_, cue, "attempted", "read"),
                **self._span(go, end),
                **ids,
                success=not (bool(tr["paused"]) or bool(tr["timed_out"])),
            )
            cut_s = (float(tr["trial_end_ns"]) - last_bin_ns) / 1e9
            if cut_s > _TRUNCATION_TOLERANCE_S:
                unit["valid"] = False
                unit["valid_description"] = (
                    f"recording ends {cut_s:.2f} s before the trial end marker"
                )
            rows.append(unit)
            if i + 1 < len(trials):
                nxt = utils.bin_index(bin_times, trials.iloc[i + 1]["trial_start_ns"])
                if (nxt - end) * _BIN_S >= _MIN_INTERTRIAL_S:
                    rows.append(
                        dict(
                            type="Intertrial",
                            description="decoder output, text-to-speech playback and "
                            "inter-trial interval",
                            **self._span(end, nxt),
                            **ids,
                        )
                    )
        return rows

    def _phased_rows(
        self,
        task_family: str,
        trials: pd.DataFrame,
        bin_times: np.ndarray,
    ) -> list[dict[str, tp.Any]]:
        """Per trial: Intertrial, read unit, then delay, go and unit for imagined and attempted."""
        type_ = _unit_type(task_family)
        rows: list[dict[str, tp.Any]] = []
        for i in range(len(trials)):
            tr = trials.iloc[i]
            cue = str(tr["cue"])
            ids = dict(trial_id=i)
            success = not (bool(tr["paused"]) or bool(tr["timed_out"]))

            def at(marker: str) -> int:
                return utils.bin_index(bin_times, tr[f"{marker}_ns"])

            rows.append(
                dict(
                    type="Intertrial",
                    description="pre-trial interval",
                    **self._span(at("intertrial_start"), at("cue_presentation")),
                    **ids,
                )
            )
            rows.append(
                dict(
                    **self._unit_row(type_, cue, "read", "read"),
                    **self._span(at("cue_presentation"), at("imagined_delay")),
                    **ids,
                    phase="read",
                    success=success,
                )
            )
            for phase, delay, go, end, modality in (
                (
                    "imagined",
                    "imagined_delay",
                    "imagined_go_period",
                    "vocalized_delay_start",
                    "imagined",
                ),
                (
                    "attempted",
                    "vocalized_delay_start",
                    "vocalized_go_cue",
                    "trial_end",
                    "read",
                ),
            ):
                rows.append(
                    dict(
                        **self._cue_stimulus(cue, "speech_delay_cue"),
                        **self._span(at(delay), at(go)),
                        **ids,
                        phase=phase,
                    )
                )
                rows.append(
                    dict(
                        **self._go_marker(task_family),
                        **self._span(at(go), at(go) + 1),
                        **ids,
                        phase=phase,
                    )
                )
                rows.append(
                    dict(
                        **self._unit_row(type_, cue, phase, modality),
                        **self._span(at(go), at(end)),
                        **ids,
                        phase=phase,
                        success=success,
                    )
                )
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        path = utils.block_path(timeline["session"], timeline["block"], self._t17)
        bin_times = utils.get_bin_times_ns(path)
        trials = utils.get_trials(path)
        fam = timeline["task_family"]
        rows = self._neural_rows(timeline, path, len(bin_times))
        if fam.startswith("phased"):
            rows += self._phased_rows(fam, trials, bin_times)
        else:
            rows += self._trial_rows(fam, trials, bin_times)
        return pd.DataFrame(rows)
