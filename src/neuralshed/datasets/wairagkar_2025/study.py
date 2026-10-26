"""neuralset ``Study`` for Wairagkar et al. 2025 brain-to-voice — attempted speech (T15).

Targets the **`t15_neural_data`** release (the official, full dataset): BrainGate2
participant T15 (ALS, severe dysarthria), 4×64-channel Utah arrays (256 ch) in left
ventral precentral gyrus, attempting to speak cued sentences from an **open vocabulary**
across **37 sessions / 8,489 trials**.

Only each trial's **"go" period** is stored, segmented per trial, with **no inter-trial
timing**. So each trial is one :term:`timeline` (8,489 total). For each we emit:

* two pre-binned neural-feature events — ``BandPower`` (``spike_band_power``) and
  ``SpikeCounts`` (``threshold_crossings``), each ``(256, n_bins)`` at 100 Hz;
  concatenate them (``aggregation="cat"``) for the decoder's 512-d input;
* one :class:`~neuralset.events.etypes.Sentence` — the cued sentence over the go period
  (the whole-trial labels ``original_split``, ``trial_description``, ``task_condition`` and
  ``benchmark`` ride the timeline, broadcast to every event row);
* one :class:`~neuralset.events.etypes.Word` per cue word — onset/offset from the 100 Hz
  ``speech_indices`` (attempted-speech word timings); and
* two :class:`~neuralshed.events.AcousticFeatures` — the **LPCNet acoustic target**
  (20-dim, time-aligned to the neural bins) in its word-aligned and syllable-aligned
  variants, ``feature="lpcnet_word_aligned"`` / ``"lpcnet_syllable_aligned"``. Select
  one with an extractor ``query``. The paper's decoders trained on the syllable-aligned
  target on every session but the first.

The closed-loop synthesized audio (``brain2voice_pred_audio_16kHz``) and per-session causal
``.h5`` decoders ship for the 9 evaluation sessions but are the *online system's output*,
not decode inputs/targets, so they are surfaced via ``utils`` rather than as events. The
small 50-word ``t15_control_experiment`` release is documented in ``README.md`` only.
"""

import datetime as dt
import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

from neuralshed.datasets import participants
from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401 — registers the custom event types
    AcousticFeatures,
    BandPower,
    SpikeCounts,
)

from . import utils

_SUBJECT = "T15"
_PARTICIPANT_ID = "braingate2/T15"  # datasets/participants.py
_BIN_SIZE_S = utils.NEURAL_BIN_SIZE_MS / 1000  # 0.01 s
_RATE_HZ = float(utils.FS_NEURAL)  # 100 Hz, also the speech-index rate
_SPIKE_BAND_HZ = (250.0, 5000.0)
# what the online pipeline did beyond the z-score that `normalization` names
_LRR = "linear regression referencing"
_NORM = (
    "log-transformed before normalization; rolling 10 s statistics; "
    "causally smoothed with a 1.5 s sigmoid kernel"
)
_PROCESSING = {
    "threshold_crossings": f"{_LRR}; {_NORM}",
    "spike_band_power": f"{_LRR}; clipped at 50,000 uV^2; {_NORM}",
}
# the two shipped LPCNet targets: (feature name, source key suffix, processing)
_LPCNET_VARIANTS = (
    (
        "lpcnet_word_aligned",
        "word",
        "offline text-to-speech of the cue, time-stretched to the attempted-speech "
        "word onsets and offsets; original LPCNet feature range",
    ),
    (
        "lpcnet_syllable_aligned",
        "syl",
        "offline text-to-speech of the cue, time-stretched to the attempted-speech "
        "syllable onsets and offsets; original LPCNet feature range",
    ),
)


class Wairagkar2025Brain2Voice(NeuralshedStudy):
    """Wairagkar et al. 2025 brain-to-voice — attempted speech (BrainGate2 T15, full release).

    One timeline per trial (8,489 across 37 sessions). Each emits two neural-feature
    streams (``BandPower`` + ``SpikeCounts``, 256 ch, 100 Hz), a ``Sentence`` cue, per-word
    ``Word`` events, and the LPCNet ``AcousticFeatures`` decode target (20-dim, 100 Hz) in
    its word-aligned and syllable-aligned variants.
    """

    dataset_id: tp.ClassVar[str] = "wairagkar_2025"
    aliases: tp.ClassVar[tuple[str, ...]] = ("wairagkar_2025", "brain-to-voice-2025")
    bibtex: tp.ClassVar[str] = """
@article{wairagkar2025voice,
  title = {An instantaneous voice-synthesis neuroprosthesis},
  author = {Wairagkar, Maitreyee and Card, Nicholas S. and Singer-Clark, Tyler and Hou, Xianda and Iacobacci, Carrina and Miller, Lee M. and Hochberg, Leigh R. and Brandman, David M. and Stavisky, Sergey D.},
  journal = {Nature},
  volume = {644},
  number = {8075},
  pages = {145–153},
  year = {2025},
  month = jun,
  doi = {10.1038/s41586-025-09127-3},
  url = {https://doi.org/10.1038/s41586-025-09127-3},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.2280gb64f"
    licence: tp.ClassVar[str] = "CC0-1.0"  # Dryad data release (CC0)
    description: tp.ClassVar[str] = (
        "Attempted open-vocabulary cued speech by BrainGate2 participant T15 (ALS, severe "
        "dysarthria), 4*64-channel Utah arrays in left ventral precentral gyrus. Pre-binned "
        "spike-band power + threshold crossings (256 ch, 100 Hz) with per-trial sentence "
        "cues, word/syllable speech timings, and the time-aligned LPCNet acoustic target "
        "(brain-to-voice). 37 sessions, 8,489 trials (Wairagkar et al. 2025, t15_neural_data)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("scipy",)

    # 13 trials spanning all seven trial_description values, mimed speech, both benchmark
    # tags and the two single-word cues (t15_day025 trials 50 and 80)
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=8489,
        num_subjects=1,
        query=(
            "(session == 't15_day025' and trial_id in [0, 50, 80]) "
            "or (session == 't15_day027' and trial_id == 0) "
            "or (session == 't15_day110' and trial_id in [0, 11]) "
            "or (session == 't15_day172' and trial_id in [0, 1, 9]) "
            "or (session == 't15_day174' and trial_id == 163) "
            "or (session == 't15_day179' and trial_id == 163) "
            "or (session == 't15_day186' and trial_id == 249) "
            "or (session == 't15_day195' and trial_id == 304)"
        ),
        num_events_in_query=141,
        event_types_in_query={
            "BandPower",
            "SpikeCounts",
            "Sentence",
            "Word",
            "AcousticFeatures",
        },
        frequency=_RATE_HZ,
    )

    # -- data access ------------------------------------------------------
    def _download(self, overwrite: bool = False) -> None:
        """Fetch the Dryad release into ``self.path`` (3 files, 31 GB extracted):
        ``t15_neural_data.zip`` -> ``t15_neural_data/`` (the 26 GB working set),
        ``t15_control_experiment.zip`` -> ``t15_control_experiment/``, and ``README.md``.
        Needs a Dryad API token (``NEURALFETCH_DRYAD_TOKEN``); point ``path`` at
        ``dataset_dir("wairagkar_2025/data")`` so the study folder settles at
        ``.../wairagkar_2025/data/Wairagkar2025Brain2Voice``.
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="wairagkar_2025",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.2280gb64f",
        ).download(overwrite=overwrite)

    def _data_dir(self):
        return self.path / utils.DATA_SUBDIR

    def _ref(self, session: str) -> utils.SessionRef:
        return utils.SessionRef(
            day=int(session[len("t15_day") :]), dir=self._data_dir() / session
        )

    # -- timelines + events ----------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        implant = participants.implant_date(_PARTICIPANT_ID)
        # test_idx.mat indexes trials globally (0-based, sorted-day order)
        ti = utils.load_test_idx(self._data_dir())
        benchmark = {int(i): "video" for i in ti["video"]}
        benchmark.update(
            {int(i): "random" for i in ti["random"]}
        )  # random wins on overlap
        g = 0
        for ref in utils.list_sessions(self._data_dir()):
            date = (implant + dt.timedelta(days=ref.day)).isoformat()
            for trial, descr in enumerate(utils.get_trial_descriptions(ref)):
                # whole-trial labels ride the timeline (broadcast to every event row)
                yield dict(
                    subject=_SUBJECT,
                    session=ref.session,
                    date=date,
                    post_implant_day=ref.day,
                    original_split=utils.split_of(descr),  # coarse "train" / "eval"
                    trial_description=descr,  # the source's verbatim label
                    task_condition="mimed" if "mimed" in descr.lower() else "attempted",
                    benchmark=benchmark.get(g, ""),
                    trial_id=trial,  # position in the session file
                )
                g += 1

    def _load_binned(
        self, timeline: dict[str, tp.Any], trial: int, mat_key: str
    ) -> np.ndarray:
        """SpecialLoader target: one neural feature stream as ``(256, n_bins)``."""
        nd = utils._load_neural(str(self._ref(timeline["session"]).neural_path))
        if float(nd["neural_bin_size_ms"]) != utils.NEURAL_BIN_SIZE_MS:
            raise ValueError(
                f"{timeline['session']}: neural_bin_size_ms={nd['neural_bin_size_ms']}"
            )
        arr = np.asarray(np.atleast_1d(nd[mat_key])[trial])
        if arr.ndim != 2 or arr.shape[1] != utils.N_ELECTRODES:
            raise ValueError(f"{timeline['session']} {mat_key}: shape {arr.shape}")
        return np.ascontiguousarray(arr.astype(np.float32).T)

    def _load_acoustic_features(
        self, timeline: dict[str, tp.Any], trial: int, alignment: str
    ) -> np.ndarray:
        """SpecialLoader: LPCNet acoustic features as ``(20, n_bins)`` (the decode target)."""
        feats = utils.get_target_feats(self._ref(timeline["session"]), trial, alignment)
        return np.ascontiguousarray(np.asarray(feats, dtype=np.float32).T)

    def _word_events(
        self,
        ref: utils.SessionRef,
        trial: int,
        cue: str,
    ) -> list[dict[str, tp.Any]]:
        """One ``Word`` row per cue word, timed from the 100 Hz speech indices."""
        words = cue.split()
        # squeeze_me flattens a single-word [1 x 2] matrix to (2,), hence atleast_2d
        idx = np.atleast_2d(utils.get_word_indices(ref, trial))
        if idx.shape != (len(words), 2):
            raise ValueError(
                f"{ref.session} trial {trial}: word_indices {idx.shape} for "
                f"{len(words)} cue words"
            )
        out = []
        for word, (onset, offset) in zip(words, idx):
            out.append(
                dict(
                    type="Word",
                    start=float(onset) / _RATE_HZ,
                    duration=max(0.0, float(offset - onset) / _RATE_HZ),
                    text=word,
                    sentence=cue,
                    language="en",
                )
            )
        return out

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ref = self._ref(timeline["session"])
        trial = timeline["trial_id"]
        # n_bins from the (smaller) target file — it is aligned to the neural bins.
        n_bins = int(utils.get_target_feats(ref, trial, "word").shape[0])
        duration_s = n_bins * _BIN_SIZE_S
        cue = utils.get_cues(ref)[trial]

        rows: list[dict[str, tp.Any]] = []
        # neural: two pre-binned feature streams. spike_band_power first so a downstream
        # `aggregation="cat"` yields the paper's [spikepow | threshcross] 512-d order.
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
        # cued sentence (spans the whole go period)
        if cue:
            rows.append(
                dict(
                    type="Sentence",
                    start=0.0,
                    duration=duration_s,
                    text=cue,
                    language="en",
                )
            )
            rows += self._word_events(ref, trial, cue)
        # LPCNet acoustic target, both alignments; pick one with an extractor `query`
        for feature, alignment, processing in _LPCNET_VARIANTS:
            rows.append(
                dict(
                    type="AcousticFeatures",
                    feature=feature,
                    start=0.0,
                    duration=duration_s,
                    frequency=_RATE_HZ,
                    dims=utils.LPCNET_DIMS,
                    n_features=len(utils.LPCNET_DIMS),
                    unit="LPCNet",
                    normalization="raw",
                    processing=processing,
                    filepath=study.SpecialLoader(
                        method=self._load_acoustic_features,
                        timeline=timeline,
                        trial=trial,
                        alignment=alignment,
                    ).to_json(),
                )
            )
        return pd.DataFrame(rows)
