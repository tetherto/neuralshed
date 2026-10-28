"""neuroai ``Study`` for Fan et al. 2023, the CORP handwriting recordings of participant T5.

Attempted handwriting of prompted English sentences (two 96-channel Utah arrays in the hand
knob). The Study reads the 21 days of the Dryad deposit (seed days, and the recalibration
and no-recalibration blocks of the online days) and the 5 later days that only FALCON H2
(DANDI 000950) releases. The FALCON files of the 21 earlier days are copies of Dryad trials
and are not read. The released features cover each trial's go period, one clip per trial,
so each trial is one timeline with:

* one ``SpikeCounts`` event, the trial's threshold-crossing counts (192 ch, 50 Hz), read
  through a ``SpecialLoader`` as ``(192, n_bins)``;
* one ``Sentence`` event over the same span, the cue T5 attempted to write, with the online
  decoder's outputs on the closed-loop Dryad days.
"""

import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

from neuralshed.datasets import participants
from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import SpikeCounts  # noqa: F401, registers the event type

from . import utils


class Fan2023Handwriting(NeuralshedStudy):
    """Fan et al. 2023: one year of attempted handwriting by BrainGate2 participant T5.

    One timeline per trial over 26 days, each with one ``SpikeCounts`` clip and the
    ``Sentence`` over the same span.
    """

    dataset_id: tp.ClassVar[str] = "fan_2023"
    aliases: tp.ClassVar[tuple[str, ...]] = (
        "fan_2023",
        "CORP",
        "fan_h2",
        "FALCON Benchmark H2",
        "FALCON-H2",
    )
    bibtex: tp.ClassVar[str] = """
@inproceedings{fan2023plugandplay,
  title = {Plug-and-Play Stability for Intracortical Brain-Computer Interfaces: A One-Year Demonstration of Seamless Brain-to-Text Communication},
  author = {Fan, Chaofei and Hahn, Nick and Kamdar, Foram and Avansino, Donald and Wilson, Guy H. and Hochberg, Leigh and Shenoy, Krishna V. and Henderson, Jaimie M. and Willett, Francis R.},
  booktitle = {Advances in Neural Information Processing Systems},
  volume = {36},
  pages = {42258--42270},
  year = {2023},
  url = {https://proceedings.neurips.cc/paper_files/paper/2023/hash/83a14a36de4502bac5b580db36e81858-Abstract-Conference.html},
}
@misc{fan2023dryad,
  title = {Data for: Plug and play stability for intracortical brain-computer interfaces: A one-year demonstration of seamless brain-to-text communication},
  author = {Fan, Chaofei and others},
  publisher = {Dryad},
  year = {2023},
  doi = {10.5061/dryad.hqbzkh1p6},
}
@inproceedings{karpowicz2024falcon,
  title = {Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark},
  author = {Karpowicz, Brianna M. and Ye, Joel and Fan, Chaofei and Tostado-Marcos, Pablo and Rizzoglio, Fabio and Washington, Clay and Scodeler, Thiago and de Lucena, Diogo and Nason-Tomaszewski, Samuel R. and Mender, Matthew J. and Ma, Xuan and Arneodo, Ezequiel Matias and Hochberg, Leigh R. and Chestek, Cynthia A. and Henderson, Jaimie M. and Gentner, Timothy Q. and Gilja, Vikash and Miller, Lee E. and Rouse, Adam G. and Gaunt, Robert A. and Collinger, Jennifer L. and Pandarinath, Chethan},
  booktitle = {Advances in Neural Information Processing Systems (Datasets and Benchmarks Track)},
  volume = {37},
  year = {2024},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html},
}
@misc{fan2024dandi000950,
  title = {FALCON Benchmark H2: Human Handwriting iBCI},
  author = {Fan, Chaofei and Hahn, Nick and Kamdar, Foram and Avansino, Donald and Wilson, Guy and Hochberg, Leigh and Shenoy, Krishna V and Henderson, Jaime and Willett, Frank},
  publisher = {DANDI Archive},
  year = {2024},
  doi = {10.48324/dandi.000950/0.241029.1403},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.hqbzkh1p6"
    licence: tp.ClassVar[str] = "CC0-1.0 (Dryad), CC-BY-4.0 (DANDI)"
    description: tp.ClassVar[str] = (
        "Attempted handwriting of prompted English sentences by iBCI participant T5 "
        "(C4 spinal cord injury) with two 96-channel Utah arrays in the hand knob, over "
        "26 days in a year. Per-trial go-period threshold-crossing counts (192 ch, 50 Hz) "
        "with the cued sentence, open-loop seed days and closed-loop recalibration days "
        "(CORP Dryad deposit, plus the FALCON H2 held-out days from DANDI 000950)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("dandi", "h5py", "scipy")

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1330,  # 1,315 Dryad trials + 15 FALCON held-out trials
        num_subjects=1,
        # an open-loop seed day, both block kinds of an online day, a held-out day and a flagged one
        query=(
            "session in ['t5.2022.05.18', 't5.2022.12.15', 't5.2023.04.17', "
            "'t5.2023.10.09']"
        ),
        num_events_in_query=190,
        event_types_in_query={"SpikeCounts", "Sentence"},
        frequency=utils.BIN_RATE_HZ,
    )

    # -- data access ------------------------------------------------------
    def _download(self, overwrite: bool = False) -> None:
        from neuralfetch.download import Dandi

        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="fan_2023",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi=utils.DRYAD_DOI,
        ).download(overwrite=overwrite)
        # folder="" mirrors the dandiset under self.path: 000950/sub-*/...nwb
        Dandi(
            study=utils.DANDISET_ID,
            version=utils.VERSION,
            dset_dir=self.path,
            folder="",
        ).download(overwrite=overwrite)

    # -- timelines + events ----------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        """One timeline per trial of every source file."""
        for src in utils.list_sources(self.path):
            trials = utils.read_source(self.path, src["path"])
            n = len(trials["tx_feats"])
            date = src["date"]
            flag = (
                {"valid": False, "valid_description": utils.INVALID_DAYS[date]}
                if date in utils.INVALID_DAYS
                else {}
            )
            for trial_id, block in enumerate(trials["blocks"]):
                if src["path"].endswith(".nwb"):
                    split = "held-out-calib"
                    corp = ""  # not on Dryad
                else:
                    corp = utils.corp_split(src["block_kind"], trial_id, n)
                    if src["block_kind"] == "no_recalibration":
                        split = ""  # FALCON did not release these blocks
                    else:
                        split = utils.falcon_split(trial_id, n)
                yield dict(
                    subject=utils.SUBJECT,
                    session=utils.session_label(date),
                    date=date,
                    post_implant_day=participants.post_implant_day(
                        "braingate2/T5", date
                    ),
                    task="attempted_handwriting",
                    task_condition="attempted",
                    block_kind=src["block_kind"],
                    loop_type=src["loop_type"],
                    original_split=split,
                    corp_split=corp,
                    block=int(block),
                    trial_id=trial_id,
                    path=src["path"],
                    **flag,
                )

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        trials = utils.read_source(self.path, timeline["path"])
        i = timeline["trial_id"]
        duration = trials["tx_feats"][i].shape[0] / utils.BIN_RATE_HZ
        cue = trials["sentences"][i]
        decoded = {
            key: utils.decode_cue(trials[key][i])
            for key in ("rnn_decoded", "lm_decoded")
            if key in trials
        }
        rows = [
            dict(
                type="SpikeCounts",
                feature="threshold_crossings",
                n_features=utils.N_CHANNELS,
                method="threshold_crossing",
                threshold=utils.THRESHOLD,
                unit="counts/bin",
                normalization="raw",
                processing=utils.PROCESSING,
                start=0.0,
                duration=duration,
                frequency=utils.BIN_RATE_HZ,
                filepath=study.SpecialLoader(
                    method=self._load_neural, timeline=timeline
                ).to_json(),
            ),
            dict(
                type="Sentence",
                start=0.0,
                duration=duration,
                text=utils.decode_cue(cue),
                language="en",
                modality="written",
                cue_raw=cue,  # the source encoding (> space, ~ period)
                **decoded,
            ),
        ]
        return pd.DataFrame(rows)

    # SpecialLoaders -------------------------------------------------
    def _load_neural(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        """SpecialLoader target: the trial's ``tx_feats`` as ``(192, n_bins)``."""
        trials = utils.read_source(self.path, timeline["path"])
        return np.ascontiguousarray(
            trials["tx_feats"][timeline["trial_id"]].T, dtype=np.float32
        )
