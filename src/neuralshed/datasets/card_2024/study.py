import typing as tp
import numpy as np
import pandas as pd
from neuralset.events import study

from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401 — registers the custom event types
    BandPower,
    SpikeCounts,
)
from . import utils

_SUBJECT = "T15"
_BIN_SIZE_S = utils.NEURAL_BIN_SIZE_MS / 1000  # 0.02 s
_RATE_HZ = float(utils.FS_NEURAL)  # 50 Hz
_THRESHOLD_RMS = "-4.5 x RMS"
# NEJM supplementary appendix S1.04 and S2.01.
# The +10 SD clip is not documented there but the max is exactly 10.0 in every session.
_SPIKE_BAND_HZ = (250.0, 5000.0)
_LRR = "linear regression referencing per 64-electrode array, refit after every block"
_ZSCORE = (
    "z-scoring statistics from the speech epochs of the preceding 20 trials within session"
    "; clipped at +10 SD after z-scoring"
)
_PROCESSING = {
    "threshold_crossings": f"{_LRR}; {_ZSCORE}",
    "spike_band_power": f"{_LRR}; clipped at 12,500 uV^2 at 1 ms before binning; {_ZSCORE}",
}
_FEATURE_WIDTHS = {
    "threshold_crossings": utils.N_THRESHOLD_CROSSINGS,
    "spike_band_power": utils.N_SPIKE_BAND_POWER,
}


class Card2024Brain2Text(NeuralshedStudy):
    """Card et al. 2024 brain-to-text — attempted cued speech (BrainGate2 T15, copy task).

    One timeline per trial (10,948 across 45 sessions). Each emits two neural-feature
    streams (``SpikeCounts`` + ``BandPower``, 256 ch, 50 Hz) and — on train/val — a
    ``Sentence`` cue carrying the orthographic text + ARPAbet phoneme target.
    """

    dataset_id: tp.ClassVar[str] = "card_2024"
    aliases: tp.ClassVar[tuple[str, ...]] = ("card_2024", "nejm-brain-to-text")
    bibtex: tp.ClassVar[str] = """
@article{card2024speech,
  title = {An accurate and rapidly calibrating speech neuroprosthesis},
  author = {Card, Nicholas S. and Wairagkar, Maitreyee and Iacobacci, Carrina and Hou, Xianda and Singer-Clark, Tyler and Willett, Francis R. and Kunz, Erin M. and Fan, Chaofei and Vahdati Nia, Maryam and Deo, Darrel R. and Srinivasan, Aparna and Choi, Eun Young and Glasser, Matthew F. and Hochberg, Leigh R. and Henderson, Jaimie M. and Shahlaie, Kiarash and Stavisky, Sergey D. and Brandman, David M.},
  journal = {New England Journal of Medicine},
  volume = {391},
  number = {7},
  pages = {609–618},
  year = {2024},
  month = aug,
  doi = {10.1056/NEJMoa2314132},
  url = {https://doi.org/10.1056/NEJMoa2314132},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.dncjsxm85"
    licence: tp.ClassVar[str] = "CC0-1.0"  # Dryad data deposit (CC0)
    description: tp.ClassVar[str] = (
        "Attempted (vocalized or silent) cued speech by BrainGate2 participant T15 (ALS, "
        "severe dysarthria), 4*64-channel Utah arrays in left precentral (speech motor) "
        "cortex. Pre-binned threshold crossings + spike-band power (256 ch each, 50 Hz) with "
        "per-trial sentence cues and ARPAbet phoneme targets (brain-to-text). 45 sessions, "
        "10,948 trials; test labels held out for the Brain-to-Text '25 challenge "
        "(Card et al. 2024, t15_copyTask)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("h5py",)

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=10948,
        num_subjects=1,
        query="session == 't15.2023.08.13'",  # one session with train, val and test
        num_events_in_query=1219,  # 418 timelines, Sentence absent on the 35 test ones
        event_types_in_query={"SpikeCounts", "BandPower", "Sentence"},
        frequency=_RATE_HZ,
    )

    # -- data access ------------------------------------------------------
    def _download(self, overwrite: bool = False) -> None:
        """Fetch the in-scope copy-task files from Dryad into ``self.path``.

        Skips the optional pretrained-RNN baseline and the out-of-scope
        personal-use pkl. Needs a Dryad API token (``NEURALFETCH_DRYAD_TOKEN``).
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="card_2024",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.dncjsxm85",
            exclude=["t15_pretrained_rnn_baseline.zip", "t15_personalUse.pkl"],
        ).download(overwrite=overwrite)

    def _ref(self, session: str) -> utils.SessionRef:
        return utils.SessionRef(
            session=session, dir=utils.sessions_dir(self.path) / session
        )

    # -- timelines + events ----------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for ref in utils.list_sessions(self.path):
            for split in ref.splits:  # "train", "val", "test"
                for t in utils.list_trials(ref, split):
                    yield dict(
                        subject=_SUBJECT,
                        session=ref.session,
                        date=ref.date,
                        post_implant_day=ref.post_implant_day,
                        original_split=split,  # paper's split
                        block=t.block,  # recording block
                        trial_id=t.trial,  # source trial_num, unique within (session, block)
                        task_condition="attempted",
                    )

    def _load_features(self, timeline: dict[str, tp.Any], stream: str) -> np.ndarray:
        """SpecialLoader target: one neural stream as ``(256, n_bins)``.

        ``stream`` is ``"threshold_crossings"`` or ``"spike_band_power"`` — the column
        block sliced from the 512-wide ``input_features`` array.
        """
        feats = utils.read_input_features(
            self._ref(timeline["session"]),
            timeline["original_split"],
            timeline["block"],
            timeline["trial_id"],
        )
        a, b = utils.feature_channel_layout()[stream]
        return np.ascontiguousarray(feats[:, a:b].T.astype(np.float32))

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ref = self._ref(timeline["session"])
        split = timeline["original_split"]
        lab = utils.read_labels(ref, split, timeline["block"], timeline["trial_id"])
        duration_s = lab["n_bins"] * _BIN_SIZE_S

        def _neural(
            *,
            type_: str,
            feature: str,
            unit: str,
            **extra,
        ) -> dict[str, tp.Any]:
            return dict(
                type=type_,
                feature=feature,
                n_features=_FEATURE_WIDTHS[feature],
                unit=unit,
                normalization="z_scored",
                processing=_PROCESSING[feature],
                start=0.0,
                duration=duration_s,
                frequency=_RATE_HZ,
                filepath=study.SpecialLoader(
                    method=self._load_features, timeline=timeline, stream=feature
                ).to_json(),
                **extra,
            )

        # threshold crossings first, then spike-band power → native [0:256 | 256:512] order.
        rows: list[dict[str, tp.Any]] = [
            _neural(
                type_="SpikeCounts",
                feature="threshold_crossings",
                unit="counts/bin",
                method="threshold_crossing",
                threshold=_THRESHOLD_RMS,
            ),
            _neural(
                type_="BandPower",
                feature="spike_band_power",
                unit="uV^2",
                lower_freq=_SPIKE_BAND_HZ[0],
                upper_freq=_SPIKE_BAND_HZ[1],
            ),
        ]

        # cued sentence + phoneme target (train/val only; test labels are held out)
        sentence = str(lab["attrs"].get("sentence_label", "")).strip()
        if sentence and "seq_class_ids" in lab:
            seq_len = int(lab["attrs"].get("seq_len", 0)) or None
            rows.append(
                dict(
                    type="Sentence",
                    start=0.0,
                    duration=duration_s,
                    text=sentence,
                    language="en",
                    modality="read",
                    phonemes=utils.decode_phonemes(lab["seq_class_ids"], seq_len),
                )
            )
        return pd.DataFrame(rows)
