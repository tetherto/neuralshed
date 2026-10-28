"""neuralset ``Study`` for Willett et al. 2021, brain-to-text via attempted handwriting.

Subject T5 (BrainGate2, C4 spinal-cord injury), two 96-channel Utah arrays (192 channels)
in the hand knob of the left precentral gyrus, attempting to handwrite single characters,
straight lines or full sentences under an instructed-delay paradigm.

Each source ``.mat`` splices 2-11 recording blocks whose clock re-zeros at every seam, so
one block is one :term:`timeline` (130 blocks in 21 files). ``session``, ``task`` (the
file: ``singleLetters``, ``straightLines`` or ``sentences``), ``block``, ``date``,
``post_implant_day``, ``loop_type`` and, on sentence blocks, ``sentence_condition`` ride
the timeline. Per block the Study emits:

* one ``SpikeCounts`` stream (``threshold_crossings``, -3.5 x RMS), ``(192, n_bins)`` at
  100 Hz, over the whole block;
* per trial (``trial_id``, the 0-based position in the file's trial arrays): a
  ``Stimulus`` over the delay (``period="delay"``, ``stimulus_kind`` ``character_cue``,
  ``movement_cue``, ``sentence_cue`` or ``rest_cue``), a one-bin go marker
  (``movement_go_cue``) and the attempted unit over the go period (``period="go"``): a
  ``Character``, a straight-line ``Action`` (``action_kind="drawing"``), a rest
  ``Action`` (``action_kind="rest"``) or a ``Sentence``.

``code`` is dataset-wide (``utils.condition_code``): rest 0, characters 1-31, lines 32-79.
Character and line go periods end at the next trial's delay onset (1.0 s and 1.5 s), the
last of a block after the same length. A sentence's go period runs to the head turn that
ended it. A sentence the source excludes keeps its rows, with ``valid=False`` on the
``Sentence``. The one trial without a recoverable intended text (``t5.2020.01.15`` trial
142) keeps its delay cue and go marker and has no ``Sentence``. Only the published
train/test partition is read from ``RNNTrainingSteps``, as ``original_split`` on each
``Sentence``.
"""

import typing as tp

import numpy as np
import pandas as pd
from neuralset.events import study

from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401, registers the custom event types
    Character,
    SpikeCounts,
)

from . import utils

_SUBJECT = "T5"
_BIN_S = utils.BIN_SIZE_S
# Go-period length in bins for the last character or line trial of a block, which has no
# next delay onset to end at.
_GO_BINS = {"singleLetters": 100, "straightLines": 150}
# Which of the two published partitions (utils.PARTITIONS) becomes ``original_split``.
# HeldOutBlocks holds out whole blocks, so a held-out sentence shares no block statistics
# with training. HeldOutTrials interleaves the two inside a block.
_PARTITION = "HeldOutBlocks"

# What the acquisition chain did before binning (Supplementary Methods 1.2 and the
# deposit readme).
_PROCESSING = (
    "0.3 Hz-7.5 kHz analog filter, 30 kHz sampling; common average reference per array; "
    "non-causal 250-3000 Hz bandpass; -3.5 x RMS thresholds set per session; "
    "crossings counted per 10 ms bin"
)
_GO_MARKER = dict(
    type="Stimulus",
    modality="visual",
    description="go cue, square turns green",
    stimulus_kind="movement_go_cue",
    task_condition="na",
    period="go",
)


class Willett2021Handwriting(NeuralshedStudy):
    """Willett et al. 2021, brain-to-text via attempted handwriting (BrainGate2 T5).

    One timeline per recording block (130). Each emits one neural stream (``SpikeCounts``
    threshold crossings, 192 ch, 100 Hz) and per trial a delay cue, a go marker and the
    attempted ``Character``, ``Action`` or ``Sentence``.
    """

    dataset_id: tp.ClassVar[str] = "willet_2021"
    aliases: tp.ClassVar[tuple[str, ...]] = ("willet_2021", "handwritingBCI")
    bibtex: tp.ClassVar[str] = """
@article{willett2021handwriting,
  title = {High-performance brain-to-text communication via handwriting},
  author = {Willett, Francis R. and Avansino, Donald T. and Hochberg, Leigh R. and Henderson, Jaimie M. and Shenoy, Krishna V.},
  journal = {Nature},
  volume = {593},
  number = {7858},
  pages = {249–254},
  year = {2021},
  month = may,
  doi = {10.1038/s41586-021-03506-2},
  url = {https://doi.org/10.1038/s41586-021-03506-2},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.wh70rxwmv"
    licence: tp.ClassVar[str] = "CC0-1.0"  # Dryad data deposit (CC0)
    description: tp.ClassVar[str] = (
        "Attempted handwriting of single characters, straight lines and full sentences by "
        "BrainGate2 participant T5 (spinal-cord injury), recorded from two 96-channel Utah "
        "arrays in the hand knob of the left precentral gyrus. Binned threshold-crossing "
        "counts (192 ch, 100 Hz) with per-trial delay cues, go markers and the attempted "
        "character, line or sentence (Willett et al. 2021, Nature 593)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("scipy",)

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=130,
        num_subjects=1,
        # pilot characters with rest, lines, an open- and a closed-loop copy block, a
        # memory block and the free-write block holding the textless trial 142
        query=(
            "(session == 't5.2019.05.08' and task == 'singleLetters' and block == 5)"
            " or (task == 'straightLines' and block == 4)"
            " or (session == 't5.2019.12.11' and task == 'sentences' and block in [9, 10])"
            " or (session == 't5.2020.01.15' and task == 'sentences' and block in [2, 17])"
        ),
        num_events_in_query=866,  # 6 SpikeCounts + 287 trials x (delay, go) + 286 units
        event_types_in_query={
            "SpikeCounts",
            "Stimulus",
            "Character",
            "Action",
            "Sentence",
        },
        frequency=utils.RATE_HZ,
    )

    # -- data access ------------------------------------------------------
    def _download(self, overwrite: bool = False) -> None:
        """Fetch the Dryad deposit into ``self.path`` (needs a Dryad API token,
        ``NEURALFETCH_DRYAD_TOKEN``). Point ``path`` at ``dataset_dir("willet_2021/data")``
        so the study folder settles at ``.../willet_2021/data/Willett2021Handwriting``.
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="willet_2021",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.wh70rxwmv",
        ).download(overwrite=overwrite)

    def _root(self):
        return self.path / "handwritingBCIData"

    def _mat_path(self, timeline: dict[str, tp.Any]):
        return (
            self._root() / "Datasets" / timeline["session"] / f"{timeline['task']}.mat"
        )

    # -- timelines --------------------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for session in utils.list_sessions(self._root()):
            for task, path in utils.list_task_files(session, self._root()).items():
                condition_of = {}
                if task == "sentences":
                    d = utils.load_variables(
                        path, ["sentenceBlockNums", "sentenceCondition"]
                    )
                    blocks = np.atleast_1d(d["sentenceBlockNums"]).astype(int)
                    conditions = np.atleast_1d(d["sentenceCondition"]).astype(str)
                    for block in set(blocks.tolist()):
                        (condition,) = set(conditions[blocks == block])
                        condition_of[block] = str(condition)
                block_list = utils.load_variables(path, ["blockList"])["blockList"]
                for block in np.atleast_1d(block_list).astype(int).tolist():
                    timeline = dict(
                        subject=_SUBJECT,
                        session=session,
                        task=task,
                        block=block,
                        date=utils.session_date(session),
                        post_implant_day=utils.post_implant_day(session),
                        loop_type="open",
                    )
                    if task == "sentences":
                        condition = condition_of[block]
                        timeline.update(
                            sentence_condition=condition,
                            loop_type="closed"
                            if condition.startswith("CL")
                            else "open",
                        )
                    yield timeline

    # -- neural stream ----------------------------------------------------
    def _load_neural(self, timeline: dict[str, tp.Any]) -> np.ndarray:
        """SpecialLoader target: one block's threshold crossings as ``(192, n_bins)``."""
        path = self._mat_path(timeline)
        block_nums = utils.load_variables(path, ["blockNumsTimeSeries"])
        b0, b1 = utils.block_bins(block_nums["blockNumsTimeSeries"], timeline["block"])
        return np.ascontiguousarray(utils.read_neural(str(path))[b0:b1].T)

    def _neural_row(
        self, timeline: dict[str, tp.Any], n_bins: int
    ) -> dict[str, tp.Any]:
        return dict(
            type="SpikeCounts",
            feature="threshold_crossings",
            n_features=utils.N_CHANNELS,
            method="threshold_crossing",
            threshold="-3.5 x RMS",
            unit="counts/bin",
            normalization="raw",
            processing=_PROCESSING,
            start=0.0,
            duration=n_bins * _BIN_S,
            frequency=utils.RATE_HZ,
            filepath=study.SpecialLoader(
                method=self._load_neural, timeline=timeline
            ).to_json(),
        )

    # -- per-trial rows ---------------------------------------------------
    @staticmethod
    def _span(a: int, b: int) -> dict[str, float]:
        return dict(start=a * _BIN_S, duration=(b - a) * _BIN_S)

    @staticmethod
    def _cued_rows(cue: str) -> tuple[dict[str, tp.Any], dict[str, tp.Any]]:
        """Delay cue and attempted unit of a character, line or rest trial."""
        code = utils.condition_code(cue)
        delay = dict(
            type="Stimulus",
            modality="visual",
            description=cue,
            code=code,
            task_condition="na",
            period="delay",
        )
        if cue == utils.REST_CUE:
            unit = dict(
                type="Action",
                action_kind="rest",
                code=code,
                description=cue,
                task_condition="na",
            )
            return dict(delay, stimulus_kind="rest_cue"), unit
        if cue in utils.CHAR_LIST:
            unit = dict(
                type="Character",
                text=utils.CHAR_CLASS_TO_GLYPH[cue],
                char_class=cue,
                code=code,
                language="en",
                modality="written",
                task_condition="attempted",
            )
            return dict(delay, stimulus_kind="character_cue"), unit
        _, distance, angle = cue.split("_")
        unit = dict(
            type="Action",
            action_kind="drawing",
            code=code,
            description=cue,
            distance=distance,
            angle_index=int(angle),
            task_condition="attempted",
        )
        return dict(delay, stimulus_kind="movement_cue"), unit

    def _cue_trial_rows(
        self, data: dict, task: str, block: int, b0: int, n_bins: int
    ) -> list[dict[str, tp.Any]]:
        trials = [
            (i, tr)
            for i, tr in enumerate(utils.extract_single_letter_trials(data))
            if tr["block_num"] == block
        ]
        rows: list[dict[str, tp.Any]] = []
        for k, (i, tr) in enumerate(trials):
            # 1-based onset bins in the file grid -> 0-based bins in the block
            delay = tr["delay_onset_bin"] - 1 - b0
            go = tr["go_onset_bin"] - 1 - b0
            if k + 1 < len(trials):
                end = trials[k + 1][1]["delay_onset_bin"] - 1 - b0
            else:
                end = go + _GO_BINS[task]
            if not 0 <= delay < go < end <= n_bins:
                raise ValueError(f"{task} trial {i} does not fit block {block}")
            delay_row, unit_row = self._cued_rows(tr["cue"])
            ids = dict(trial_id=i)
            rows.append(dict(**delay_row, **self._span(delay, go), **ids))
            rows.append(dict(**_GO_MARKER, **self._span(go, go + 1), **ids))
            rows.append(dict(**unit_row, **self._span(go, end), period="go", **ids))
        return rows

    def _sentence_rows(
        self, data: dict, session: str, block: int, b0: int, n_bins: int
    ) -> list[dict[str, tp.Any]]:
        # The partition indexes trials by position in the session's sentence arrays.
        split_of = utils.load_partition(_PARTITION, self._root()).get(session, {})
        rows: list[dict[str, tp.Any]] = []
        for i, tr in enumerate(utils.extract_sentence_trials(data)):
            if tr["block_num"] != block:
                continue
            delay = tr["delay_onset_bin"] - 1 - b0
            go = tr["go_onset_bin"] - 1 - b0
            end = go + tr["num_time_bins"]
            if not 0 <= delay < go < end <= n_bins:
                raise ValueError(f"sentence {i} does not fit block {block}")
            ids = dict(trial_id=i)
            rows.append(
                dict(
                    type="Stimulus",
                    modality="visual",
                    description=utils.decode_text(tr["sentence_prompt"]),
                    stimulus_kind="sentence_cue",
                    task_condition="na",
                    period="delay",
                    **self._span(delay, go),
                    **ids,
                )
            )
            rows.append(dict(**_GO_MARKER, **self._span(go, go + 1), **ids))
            text = utils.decode_text(tr["intended_text"]).strip()
            if not text:  # no recoverable intended text, so no Sentence
                continue
            flags = {}
            if tr["excluded"]:
                flags = dict(
                    valid=False, valid_description="excluded by the source authors"
                )
            rows.append(
                dict(
                    type="Sentence",
                    text=text,
                    language="en",
                    modality="written",
                    task_condition="attempted",
                    period="go",
                    excluded=tr["excluded"],
                    original_split=split_of.get(i, ""),
                    **flags,
                    **self._span(go, end),
                    **ids,
                )
            )
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        task, block = timeline["task"], timeline["block"]
        path = self._mat_path(timeline)
        utils.n_time_bins(path)  # checks the channel count
        data = utils.load_trial_variables(path, task)
        b0, b1 = utils.block_bins(data["blockNumsTimeSeries"], block)

        rows: list[dict[str, tp.Any]] = [self._neural_row(timeline, b1 - b0)]
        if task == "sentences":
            rows += self._sentence_rows(data, timeline["session"], block, b0, b1 - b0)
        else:
            rows += self._cue_trial_rows(data, task, block, b0, b1 - b0)
        return pd.DataFrame(rows)
