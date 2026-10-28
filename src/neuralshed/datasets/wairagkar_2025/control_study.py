"""neuralset ``Study`` for the Wairagkar et al. 2025 **control experiment** (T15).

Targets the **`t15_control_experiment`** release (the small sibling of
``t15_neural_data``): one session (**``t15_day482``**, post-implant day 482) of
attempted speech over a **50-word vocabulary**, stored as 10 block files
(``t15_day482_block00.mat`` ... ``block09``) of **24 trials each**.

Unlike the full release, each block file holds all 24 trials of one recording
block, and the neural arrays are the **same online pipeline** as
``t15_neural_data`` (same variables, 10 ms bins, 256 channels) — emitted here
**verbatim**. Each block also ships T15's residual
speech from a stethoscopic microphone (30 kHz) and a 6-axis IMU (~242 Hz), plus
a ``brain2voice_*.h5`` decoder trained on the *other* blocks with this one held
out. None of those three reach the model; they are the online system's output
or extra-modal signals, so ``utils`` exposes accessors
(``get_stethoscope``, ``get_imu``, ``BlockRef.decoder_path``) instead of events.

Only each trial's **go** period is stored (no inter-trial timing), so each
trial is one :term:`timeline` (240 total), mirroring the sibling study:

* ``BandPower`` + ``SpikeCounts`` (256 ch, 100 Hz; ``aggregation="cat"`` gives
  a ``[spike-band power | threshold crossings]`` 512-d order);
* one ``Sentence`` — the cued sentence (text **stripped**: the shipped strings
  pad with trailing spaces) carrying ``phonemes``, the cue's dictionary
  pronunciation from ``assets/cue_phonemes.json`` (``utils.build_cue_phonemes``;
  all 240 cues of this release are covered).

The timeline identity is ``session`` / ``block`` (0–9, from ``block_num``) /
``trial_id`` (0–23 within the block). All 240 trials share post-implant day
482, so ``block``, not ``post_implant_day``, is the only grouping finer than the
session.

Lives in its own module because ``tests/test_study.py`` pins exactly one Study
class per dataset ``study.py``.
"""

import datetime as dt
import typing as tp
from pathlib import Path

import numpy as np
import pandas as pd
from neuralset.events import study

from neuralshed.datasets import participants
from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401 — registers the custom event types
    BandPower,
    SpikeCounts,
)

from . import utils
from .study import (
    _PROCESSING,
    _RATE_HZ,
    _SPIKE_BAND_HZ,
    _SUBJECT,
    Wairagkar2025Brain2Voice,
)

_BIN_SIZE_S = utils.NEURAL_BIN_SIZE_MS / 1000  # 0.01 s
_PARTICIPANT_ID = "braingate2/T15"  # datasets/participants.py


class Wairagkar2025ControlExperiment(NeuralshedStudy):
    """Wairagkar et al. 2025 control experiment — attempted speech, 50-word vocabulary.

    One timeline per trial (240 across 10 blocks of one session, ``t15_day482``).
    Each emits the two shipped neural-feature streams (``BandPower`` +
    ``SpikeCounts``, 256 ch, 100 Hz, verbatim) and a ``Sentence`` cue with the
    ``phonemes`` target. The stethoscope mic, IMU and per-block ``.h5``
    decoders are ``utils``-side only.
    """

    dataset_id: tp.ClassVar[str] = "wairagkar_2025"
    aliases: tp.ClassVar[tuple[str, ...]] = (
        "wairagkar_2025_control",
        "brain-to-voice-2025-control",
    )
    # same paper and Dryad release as the full study
    bibtex: tp.ClassVar[str] = Wairagkar2025Brain2Voice.bibtex
    url: tp.ClassVar[str] = Wairagkar2025Brain2Voice.url
    licence: tp.ClassVar[str] = Wairagkar2025Brain2Voice.licence
    description: tp.ClassVar[str] = (
        "Attempted cued speech over a 50-word vocabulary by BrainGate2 "
        "participant T15, control experiment release: one session "
        "(t15_day482), 10 blocks of 24 trials, pre-binned spike-band power + "
        "threshold crossings (256 ch, 100 Hz) with per-trial sentence cues and "
        "their ARPAbet phoneme targets; also ships a 30 kHz stethoscopic "
        "microphone and a 6-axis IMU per trial (utils-side)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("scipy",)

    # one session, 10 blocks x 24 trials; the query spans both neural streams
    # and the cue (every trial carries all three), measured on the full release
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=240,
        num_subjects=1,
        query="block == 0 and trial_id in [0, 1]",
        num_events_in_query=6,
        event_types_in_query={"BandPower", "SpikeCounts", "Sentence"},
        frequency=_RATE_HZ,
    )

    # -- data access ------------------------------------------------------
    def _download(self, overwrite: bool = False) -> None:
        """The control release arrives with the sibling study's Dryad download
        (31 GB extracted, one folder two subfolders); fetch it there.
        """
        # ``path`` points at this study's class-name folder; the download
        # belongs to the main study folder next to it.
        Wairagkar2025Brain2Voice(path=self.path.parent).download(overwrite=overwrite)

    def _data_dir(self) -> Path:
        """Locate the block-file folder across the three layouts a caller may pass.

        1. ``<study folder>/t15_control_experiment`` — a farm whose class-name
           folder holds the release subfolder (the usual neuralset layout);
        2. the class-name folder **is** the block-file folder (a farm symlink
           pointing straight at ``.../t15_control_experiment``);
        3. ``<root>/Wairagkar2025ControlExperiment`` (nonexistent — the base
           appends the class name) with the release next to it inside the main
           study's download folder, the real-world layout.
        """
        candidates = (
            self.path / utils.CONTROL_SUBDIR,
            self.path,
            self.path.parent / Wairagkar2025Brain2Voice.__name__ / utils.CONTROL_SUBDIR,
        )
        for candidate in candidates:
            if candidate.exists() and list(candidate.glob("t15_day*_block*.mat")):
                return candidate
        raise FileNotFoundError(
            f"no t15_control_experiment block files found at {candidates[0]!r}, "
            f"{candidates[1]!r} or {candidates[2]!r}; the release arrives with "
            f"Wairagkar2025Brain2Voice.download()"
        )

    def _ref(self, timeline: tp.Mapping[str, tp.Any]) -> utils.BlockRef:
        """Rebuild a block reference from a (JSON-safe) timeline dict."""
        session, block = str(timeline["session"]), int(timeline["block"])
        return utils.BlockRef(
            session=session,
            block=block,
            path=self._data_dir() / f"{session}_block{block:02d}.mat",
        )

    # -- timelines + events ----------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        implant = participants.implant_date(_PARTICIPANT_ID)
        for ref in utils.list_control_blocks(self._data_dir()):
            day = int(ref.session[len("t15_day") :])
            date = (implant + dt.timedelta(days=day)).isoformat()
            description = utils.get_block_description(ref)
            for trial, sentence in enumerate(utils.get_control_sentences(ref)):
                # whole-trial labels ride the timeline (broadcast to every row)
                yield dict(
                    subject=_SUBJECT,
                    session=ref.session,
                    date=date,
                    post_implant_day=day,
                    block=ref.block,  # 0-9, the only grouping finer than the session
                    block_description=description,
                    trial_id=trial,  # position within the block file
                )

    def _load_binned(
        self, timeline: dict[str, tp.Any], trial: int, mat_key: str
    ) -> np.ndarray:
        """SpecialLoader target: one neural-feature stream as ``(256, n_bins)``,
        the stored array verbatim (float32, transposed like the sibling study)."""
        ref = self._ref(timeline)
        nd = utils._load_control_block(str(ref.path))
        if float(nd["neural_bin_size_ms"]) != utils.NEURAL_BIN_SIZE_MS:
            raise ValueError(
                f"{ref.file_stem}: neural_bin_size_ms={nd['neural_bin_size_ms']}"
            )
        arr = np.asarray(np.atleast_1d(nd[mat_key])[trial])
        if arr.ndim != 2 or arr.shape[1] != utils.N_ELECTRODES:
            raise ValueError(f"{ref.file_stem} {mat_key}: shape {arr.shape}")
        return np.ascontiguousarray(arr.astype(np.float32).T)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ref = self._ref(timeline)
        trial = int(timeline["trial_id"])
        # n_bins from the stored spike-band-power trial (the streams are aligned)
        n_bins = int(utils.get_control_features(ref, trial, "spikepow_trials").shape[0])
        duration_s = n_bins * _BIN_SIZE_S
        sentence = utils.get_control_sentences(ref)[trial]

        rows: list[dict[str, tp.Any]] = []
        # spike-band power first, so a downstream `aggregation="cat"` yields
        # a [spikepow | threshcross] 512-d order
        rows.append(
            dict(
                type="BandPower",
                feature="spike_band_power",
                n_features=utils.N_ELECTRODES,
                lower_freq=_SPIKE_BAND_HZ[0],
                upper_freq=_SPIKE_BAND_HZ[1],
                unit="uV^2",
                normalization="z_scored",
                processing=_PROCESSING["spike_band_power"],
                start=0.0,
                duration=duration_s,
                frequency=_RATE_HZ,
                filepath=study.SpecialLoader(
                    method=self._load_binned,
                    timeline=timeline,
                    trial=trial,
                    mat_key="spikepow_trials",
                ).to_json(),
            )
        )
        rows.append(
            dict(
                type="SpikeCounts",
                feature="threshold_crossings",
                n_features=utils.N_ELECTRODES,
                method="threshold_crossing",
                threshold="-4.5 x RMS",
                unit="counts/bin",
                normalization="z_scored",
                processing=_PROCESSING["threshold_crossings"],
                start=0.0,
                duration=duration_s,
                frequency=_RATE_HZ,
                filepath=study.SpecialLoader(
                    method=self._load_binned,
                    timeline=timeline,
                    trial=trial,
                    mat_key="threshcross_trials",
                ).to_json(),
            )
        )
        # cued sentence (spans the whole go period) with its phoneme target
        if sentence:
            rows.append(
                dict(
                    type="Sentence",
                    start=0.0,
                    duration=duration_s,
                    text=sentence,
                    language="en",
                    # assets/cue_phonemes.json (utils.build_cue_phonemes); keys
                    # and this text are both stripped, so all 240 cues hit.
                    phonemes=utils.CUE_PHONEMES.cues.get(sentence),
                )
            )
        return pd.DataFrame(rows)
