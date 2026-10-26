"""neuralset ``Study`` for Willett et al. 2023, attempted sentence production (T12).

Subject T12 (BrainGate2, ALS), four 64-channel Utah arrays (two in area 6v, two in
area 44), attempted speech of cued sentences under an instructed-delay paradigm.

One timeline per recording block (306 blocks across the 24 ``sentences/*.mat``
sessions). The session files concatenate blocks recorded minutes to an hour apart
with re-zeroed clocks, so a block is the largest span on one continuous timebase.
Whole-block labels ride the timeline dict: ``session``, ``date``,
``post_implant_day``, ``block``, ``block_type``, ``loop_type``, ``original_split``
(the competition partition), ``task_condition`` (``attempted`` vocalized or
``mimed``), and ``valid`` / ``valid_description`` for blocks whose released neural
data are known to be bad (``utils.INVALID_BLOCKS``).

Per timeline the Study emits two whole-block neural-feature events, binned
threshold crossings (``SpikeCounts``, ``tx2``, -4.5 x RMS) and spike-band power
(``BandPower``, ``spikePow``), each ``(256, n_bins)`` at 50 Hz, and per trial one
``Stimulus`` for the sentence cue over the delay period (``stimulus_kind``
``sentence_cue``), one ``Stimulus`` for the go cue (``speech_go_cue``, one bin)
and one ``Sentence`` spanning the go period.

The neural rows are in array order, not file order. The source matrices interleave
the four arrays (only the two superior arrays are contiguous), so ``_load_binned``
permutes the 256 columns with ``utils.CHANNEL_ORDER`` into one contiguous block of
64 per array, in the order ``utils.ARRAY_ORDER`` (6v superior, 6v inferior, 44
superior, 44 inferior), which is the layout the catalog's ``channel_start`` /
``channel_stop`` describe.

Scope is sentences only (see ``datasets/willet_2023/README.md``): the
``diagnosticBlocks``, ``tuningTasks`` and ``competitionData`` directories and every
decoder-output field are out of scope. The threshold variants ``tx3`` and ``tx4``
are reachable through ``_load_binned`` but never emitted. The ``SpikeCounts``
event emits ``tx2`` (the paper's online decoder and the corpus convention) by
default; pass ``threshold_variant="tx1"`` to emit ``tx1`` instead — the feature
the authors' released baseline decodes from, together with ``spikePow`` of the
128 area 6v channels.
"""

import typing as tp

import numpy as np
import pandas as pd
from neuralset.events import study

from neuralshed.datasets import participants
from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401, registers the neural-feature types
    BandPower,
    SpikeCounts,
)

from . import utils

_SUBJECT = "T12"
_BIN_SIZE_S = utils.BIN_SIZE_S
_RATE_HZ = utils.SAMPLING_RATE_HZ

# What the online chain did before binning (supplement 1.4 and the deposit readmes).
# The readmes give the high-pass cutoff as 250 Hz, the supplement as 300 Hz.
_FILTER = (
    "non-causal 250 Hz high-pass with a 4 ms delay; linear regression referencing "
    "refit per session from its diagnostic block"
)
_PROCESSING = {
    "threshold_crossings": "thresholds at {threshold}, also refit per session",
    "spike_band_power": f"{_FILTER}; mean of squared voltages per bin, unclipped",
}
# The source states the high-pass only; 5 kHz is the spike-band upper edge the corpus
# declares (``events.BandPower``).
_SPIKE_BAND_HZ = (250.0, 5000.0)

# The two released threshold-crossing variants and their thresholds (see
# ``utils.SIGNAL_TYPES``). ``tx2`` is the corpus convention; ``tx1`` is what the
# authors' released baseline decodes from.
_TX_THRESHOLDS = {"tx1": "-3.5 x RMS", "tx2": "-4.5 x RMS"}


def _feature_specs(
    threshold_variant: str,
) -> tuple[tuple[str, dict[str, tp.Any]], ...]:
    """(.mat key, event-row fields) emitted per block for one threshold variant."""
    threshold = _TX_THRESHOLDS[threshold_variant]
    return (
        (
            threshold_variant,
            dict(
                type="SpikeCounts",
                feature="threshold_crossings",
                n_features=256,
                method="threshold_crossing",
                threshold=threshold,
                # uint8 and all-integer in every released session (per-session max 11-53).
                unit="counts/bin",
                normalization="raw",
                processing=f"{_FILTER}; "
                + _PROCESSING["threshold_crossings"].format(threshold=threshold),
            ),
        ),
        (
            "spikePow",
            dict(
                type="BandPower",
                feature="spike_band_power",
                n_features=256,
                lower_freq=_SPIKE_BAND_HZ[0],
                upper_freq=_SPIKE_BAND_HZ[1],
                unit="uV^2",
                normalization="raw",
                processing=_PROCESSING["spike_band_power"],
            ),
        ),
    )


_TRIAL_VARIABLES = [
    "delayTrialEpochs",
    "goTrialEpochs",
    "sentences",
    "sentenceDurations",
    "blockNum",
]


class Willett2023Sentences(NeuralshedStudy):
    """Willett et al. 2023 speech neuroprosthesis, attempted sentence production (T12).

    One timeline per recording block (306). Each emits two whole-block neural-feature
    events (``SpikeCounts`` threshold crossings + ``BandPower`` spike-band power, 256 ch,
    50 Hz) and one ``Sentence`` event per trial (attempted speech over the go period).

    ``threshold_variant`` selects which threshold-crossing stream is emitted:
    ``"tx2"`` (default, the corpus convention) or ``"tx1"`` (the authors' released
    baseline feature). The two variants never mix in one catalog — pick per Study.
    """

    dataset_id: tp.ClassVar[str] = "willet_2023"
    threshold_variant: tp.Literal["tx1", "tx2"] = "tx2"
    aliases: tp.ClassVar[tuple[str, ...]] = ("willet_2023", "Willett2023Speech")
    bibtex: tp.ClassVar[str] = """
@article{willett2023speech,
  title = {A high-performance speech neuroprosthesis},
  author = {Willett, Francis R. and Kunz, Erin M. and Fan, Chaofei and Avansino, Donald T. and Wilson, Guy H. and Choi, Eun Young and Kamdar, Foram and Glasser, Matthew F. and Hochberg, Leigh R. and Druckmann, Shaul and Shenoy, Krishna V. and Henderson, Jaimie M.},
  journal = {Nature},
  volume = {620},
  number = {7976},
  pages = {1031–1036},
  year = {2023},
  month = aug,
  doi = {10.1038/s41586-023-06377-x},
  url = {https://doi.org/10.1038/s41586-023-06377-x},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.x69p8czpq"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Attempted speech of cued sentences by BrainGate2 participant T12 (ALS), "
        "recorded from 4x64-channel Utah arrays in ventral precentral gyrus (area 6v) "
        "and inferior frontal gyrus (area 44). Binned threshold crossings + spike-band "
        "power (256 ch, 50 Hz) "
        "with per-trial sentence cues (Willett et al. 2023, sentences subset)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("scipy",)

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=306,
        num_subjects=1,
        # one vocal open-loop day and one mimed day with closed-loop blocks
        query="session in ['t12.2022.04.28', 't12.2022.06.23']",
        num_events_in_query=2668,  # 870 x (Sentence + 2 Stimulus) + 29 x 2 neural
        event_types_in_query={"SpikeCounts", "BandPower", "Sentence", "Stimulus"},
        frequency=_RATE_HZ,
    )

    # -- data access ------------------------------------------------------
    def _cls_kwargs(self) -> dict[str, tp.Any]:
        """Declare ``threshold_variant`` as the real class parameter it is.

        The neuralset base refuses any non-default class parameter
        ("Class parameters are not yet supported") because it cannot tell
        whether a parameter selects a subset or changes the emitted data's
        identity. For Willett it changes identity: the variant picks which of
        the two released threshold-crossing streams (tx1/tx2) the Study
        emits, so a tx1 timeline must never share a timeline string (or a
        dumped cache entry) with the tx2 one it would otherwise look
        identical to. Like the base's ``exclude_defaults`` semantics, the
        default variant keeps contributing nothing, so catalogs built with
        tx2 keep their existing identities.
        """
        default = Willett2023Sentences.model_fields["threshold_variant"].default
        if self.threshold_variant == default:
            return {}
        return {"threshold_variant": self.threshold_variant}

    def _download(self, overwrite: bool = False) -> None:
        """Fetch the in-scope ``sentences.tar.gz`` subset from Dryad into ``self.path``.

        Skips the out-of-scope diagnosticBlocks / tuningTasks / competitionData /
        derived / languageModel archives (tens of GB). Needs a Dryad API token
        (``NEURALFETCH_DRYAD_TOKEN``).
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="willet_2023",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.x69p8czpq",
            include=["*sentences*"],  # sentences.tar.gz + sentences_readme.txt
        ).download(overwrite=overwrite)

    def _file(self, session: str):
        return utils.sentences_file(self.path, session)

    # -- timelines + events ----------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for session in utils.list_sessions(self.path):
            d = utils.load_variables(
                self._file(session), ["blockList", "blockTypes", "speakingMode"]
            )
            # "attempted speaking" (vocalized) vs "attempted nonvocal speaking" (mouthed)
            mimed = "nonvocal" in str(d["speakingMode"]).lower()
            split_by_block = utils.competition_split_by_block(session)
            bad_blocks, reason = utils.INVALID_BLOCKS.get(session, ((), ""))
            date = utils.session_date(session)
            for block, block_type in utils.list_blocks(d):
                yield dict(
                    subject=_SUBJECT,
                    session=session,
                    date=date,
                    post_implant_day=participants.post_implant_day(
                        utils.PARTICIPANT_ID, date
                    ),
                    block=block,
                    block_type=block_type,  # e.g. "OL Open WebText", "CL Chang Stage 2"
                    loop_type="closed" if block_type.startswith("CL") else "open",
                    original_split=split_by_block.get(block, ""),
                    task_condition="mimed" if mimed else "attempted",
                    valid=block not in bad_blocks,
                    valid_description=reason if block in bad_blocks else "",
                )

    def _block_bins(self, timeline: dict[str, tp.Any]) -> tuple[int, int]:
        d = utils.load_variables(self._file(timeline["session"]), ["blockNum"])
        return utils.block_bins(d["blockNum"], timeline["block"])

    def _load_binned(self, timeline: dict[str, tp.Any], mat_key: str) -> np.ndarray:
        """SpecialLoader target: one feature stream of one block as ``(256, n_bins)``."""
        b0, b1 = self._block_bins(timeline)
        data = utils.read_stream(str(self._file(timeline["session"])), mat_key)
        block = utils.to_array_order(data[b0:b1])
        return np.ascontiguousarray(block.T, dtype=np.float32)

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        d = utils.load_variables(self._file(timeline["session"]), _TRIAL_VARIABLES)
        b0, b1 = utils.block_bins(d["blockNum"], timeline["block"])
        duration_s = (b1 - b0) * _BIN_SIZE_S

        rows: list[dict[str, tp.Any]] = []
        for mat_key, spec in _feature_specs(self.threshold_variant):
            loader = study.SpecialLoader(
                method=self._load_binned, timeline=timeline, mat_key=mat_key
            ).to_json()
            rows.append(
                dict(
                    start=0.0,
                    duration=duration_s,
                    frequency=_RATE_HZ,
                    filepath=loader,
                    **spec,
                )
            )

        delay = np.atleast_2d(np.asarray(d["delayTrialEpochs"])).astype(int)  # 1-based
        go = np.atleast_2d(np.asarray(d["goTrialEpochs"])).astype(int)  # 1-based incl.
        sentences = np.atleast_1d(np.asarray(d["sentences"], dtype=object))
        durations = np.atleast_1d(np.asarray(d["sentenceDurations"])).astype(int)
        for i in range(go.shape[0]):
            delay_start = int(delay[i, 0]) - 1  # 0-based bins in the session grid
            go_start = int(go[i, 0]) - 1
            if not b0 <= go_start < b1:
                continue  # trial belongs to another block
            text = str(sentences[i]).strip()
            if not text:
                raise ValueError(
                    f"{timeline['session']} trial {i} has an empty sentence"
                )
            trial_id = i  # 0-based row in the session's trial arrays
            rows.append(
                dict(
                    type="Stimulus",
                    start=(delay_start - b0) * _BIN_SIZE_S,
                    duration=(go_start - delay_start) * _BIN_SIZE_S,
                    modality="visual",
                    description=text,
                    stimulus_kind="sentence_cue",
                    trial_id=trial_id,
                )
            )
            rows.append(
                dict(
                    type="Stimulus",
                    start=(go_start - b0) * _BIN_SIZE_S,
                    duration=_BIN_SIZE_S,
                    modality="visual",
                    description="go cue, square turns green",
                    stimulus_kind="speech_go_cue",
                    trial_id=trial_id,
                )
            )
            rows.append(
                dict(
                    type="Sentence",
                    start=(go_start - b0) * _BIN_SIZE_S,
                    duration=int(durations[i]) * _BIN_SIZE_S,
                    text=text,
                    language="en",
                    modality="read",
                    trial_id=trial_id,
                    # the cue's dictionary pronunciation from
                    # assets/cue_phonemes.json (utils.build_cue_phonemes)
                    phonemes=utils.CUE_PHONEMES.cues.get(text),
                )
            )
        return pd.DataFrame(rows)
