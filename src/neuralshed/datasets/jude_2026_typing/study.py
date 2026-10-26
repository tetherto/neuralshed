"""neuralset ``Study`` for Jude et al. 2026, bimanual QWERTY typing iBCI (T17 + T18).

Two BrainGate2 participants attempt individual finger movements mapped to keys:
**T17** (ALS, six 64-channel Utah arrays in left precentral gyrus, areas 6v/55b/6d) and
**T18** (spinal-cord injury, six arrays in area 6d, four left and two right hemisphere).
19 sessions, 165 ``.mat`` files.

Each ``.mat`` block is a recording with many cued trials inside it, so one block is one
:term:`timeline` (160 catalogued task blocks; the files in ``utils.EXCLUDED_BLOCKS`` and
``utils.DUPLICATE_FILES`` are skipped). Per block the Study emits:

* two neural-feature streams over the whole block, ``SpikeCounts``
  (``threshold_crossings``) and ``BandPower`` (``spike_band_power``), each
  ``(384, n_bins)`` at 100 Hz. SpikeCounts first, so ``aggregation="cat"`` reproduces the
  source's [threshold_crossings | spike_band_power] feature order;
* per trial (``trial_id``, 0-based within the block, and ``period``): a ``Stimulus`` for
  the on-screen cue over the delay (trial start to go; none in free typing, which is
  uncued), a one-bin go marker, and the attempted unit over the go window: an ``Action``
  for a finger movement (``code``, ``utils.CUE_CODES``, shared by a movement and the key
  it types; rest is ``action_kind="rest"``, code 0) or a ``Sentence`` (``text``,
  ``modality`` ``typed`` or ``written``). Units carry ``task_condition="attempted"`` and
  ``success`` (the source's paused and timed-out flags);
* ``Intertrial`` rows over the gaps of 0.1 s or more between trials, except where the
  recording pauses (``utils.clock_breaks``).

``block``, ``task_family``, ``loop_type`` (open or closed), ``date`` and
``post_implant_day`` ride the timeline. Event windows are neural-bin indices (nearest bin to
each NSP marker), so they align exactly to the 100 Hz feature grid. The 30 kHz microphone
audio (T17 only) is anarthric and is surfaced via ``utils.get_audio`` rather than as an
event.
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

_RATE_HZ = float(utils.FS_NEURAL)  # 100 Hz (10 ms bins)
_BIN_S = 1.0 / _RATE_HZ
_THRESHOLD_SD = f"{utils.THRESHOLD_SD} x SD"
_SPIKE_BAND_HZ = (250.0, 5000.0)
# Gaps shorter than this between a trial end and the next trial start are clock jitter,
# not an inter-trial interval.
_MIN_INTERTRIAL_S = 0.1

# What the acquisition chain did before binning (paper Methods and the deposit README).
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


class Jude2026Typing(NeuralshedStudy):
    """Jude et al. 2026, bimanual QWERTY typing iBCI (BrainGate2 T17 + T18).

    One timeline per catalogued block (160 across 19 sessions). Each emits two neural
    streams (``SpikeCounts`` + ``BandPower``, 384 ch, 100 Hz), per trial a delay cue
    ``Stimulus``, a go marker and the attempted unit (``Action`` for finger movements and
    rest, ``Sentence`` for typed and handwritten sentences), plus ``Intertrial`` gaps.
    """

    dataset_id: tp.ClassVar[str] = "jude_2026_typing"
    aliases: tp.ClassVar[tuple[str, ...]] = ("jude_2026_typing", "typingBCI")
    bibtex: tp.ClassVar[str] = """
@article{jude2026typing,
  title = {Restoring rapid natural bimanual typing with a neuroprosthesis after paralysis},
  author = {Jude, Justin J. and Levi-Aharoni, Hadar and Acosta, Alexander J. and Allcroft, Shane B. and Nicolas, Claire and Lacayo, Bayardo E. and Card, Nicholas S. and Wairagkar, Maitreyee and Levin, Alisa D. and Brandman, David M. and Stavisky, Sergey D. and Willett, Francis R. and Williams, Ziv M. and Simeral, John D. and Hochberg, Leigh R. and Rubin, Daniel B.},
  journal = {Nature Neuroscience},
  year = {2026},
  month = mar,
  doi = {10.1038/s41593-026-02218-y},
  url = {https://doi.org/10.1038/s41593-026-02218-y},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.cz8w9gjjk"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Bimanual QWERTY typing decoded from attempted finger movements by BrainGate2 "
        "participants T17 (ALS, six arrays left precentral gyrus 6d/6v/55b) and T18 (spinal "
        "cord injury, six arrays area 6d, four left + two right hemisphere). Per-block 10 ms-"
        "binned threshold crossings + spike-band power (384 ch, 100 Hz) with per-trial "
        "sentence cues (calibration / sentence-copy / free-typing / handwriting) and isolated "
        "finger-movement cues. 19 sessions, 160 task blocks (Jude et al. 2026, Nature Neuroscience)."
    )
    requirements: tp.ClassVar[tuple[str, ...]] = ("scipy",)

    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=160,
        num_subjects=2,
        # Both participants' movement sweeps and the T18 key-cue sweep, calibration (the
        # block deposited twice), sentence copy (T17's 256-feature day and a block the Dryad
        # tables omit), free typing (a recording pause, the empty cue), and handwriting (two
        # pauses, the closed-loop block).
        query=(
            "(session == 't17.2024.04.24' and block == 2)"
            " or (session == 't18.2024.12.03' and block == 2)"
            " or (session == 't18.2024.12.04' and block in [2, 7])"
            " or (session == 't17.2024.04.25' and block == 1)"
            " or (session == 't17.2024.06.03' and block == 12)"
            " or (session == 't18.2025.01.22' and block == 4)"
            " or (session == 't17.2024.07.22' and block == 5)"
            " or (session == 't18.2025.02.04' and block == 5)"
            " or (session == 't18.2025.04.01' and block in [3, 13])"
        ),
        num_events_in_query=1192,  # 11 x 2 neural, 593 Stimulus, 273 Intertrial, 186 Action, 118 Sentence
        event_types_in_query={
            "SpikeCounts",
            "BandPower",
            "Stimulus",
            "Action",
            "Sentence",
            "Intertrial",
        },
        frequency=_RATE_HZ,
    )

    # -- data access ------------------------------------------------------
    def _download(self, overwrite: bool = False) -> None:
        """Fetch the Dryad deposit (``Data.zip`` -> ``Data/``, ``README.md``) into ``self.path``.

        Returns at once when the success marker exists and ``overwrite`` is off.
        Otherwise an extracted tree that passes ``utils.verify_tree`` (README
        digest against Dryad, every catalogued, excluded and duplicate block file present
        and readable) is accepted as downloaded and the success marker records what was
        verified. Any other state transfers the deposit, which needs
        ``NEURALFETCH_DRYAD_TOKEN``, and is verified afterwards.
        """
        import datetime as dt

        from neuralshed.datasets.dryad import Dryad, latest_files

        dryad = Dryad(
            study="jude_2026_typing",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi=utils.DRYAD_DOI,
            include=list(utils.DRYAD_FILES),
        )
        if not overwrite and dryad.get_success_file().exists():
            return
        digest = latest_files(utils.DRYAD_DOI).get("README.md", (0, ""))[1]
        if not overwrite:
            problems = utils.verify_tree(self.path, readme_digest=digest)
            if not problems:
                expected = utils.expected_blocks()
                n_blocks = sum(map(len, expected.values()))
                n_dup = sum(map(len, utils.DUPLICATE_FILES.values()))
                dryad.get_success_file().write_text(
                    f"verified {dt.date.today().isoformat()}: README.md sha-256 matches "
                    f"Dryad, {n_blocks} block files and {n_dup} duplicate present and "
                    f"readable in {len(expected)} sessions\n"
                )
                return
            print("jude_2026_typing tree incomplete, fetching:", *problems, sep="\n  ")
        dryad.download(overwrite=overwrite)
        if problems := utils.verify_tree(self.path, readme_digest=digest):
            raise RuntimeError(
                "jude_2026_typing tree still incomplete after download (an extracted "
                "Data/ is not re-fetched, pass overwrite=True):\n  "
                + "\n  ".join(problems)
            )

    @property
    def _blocks_root(self):
        return self.path / "Data"

    def _block_path(self, timeline: dict[str, tp.Any]):
        return utils.block_path(
            timeline["session"], timeline["block"], self._blocks_root
        )

    # -- timelines + events ----------------------------------------------
    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for session in utils.list_sessions(self._blocks_root):
            for block, _path in utils.list_blocks(session, self._blocks_root):
                fam = utils.task_of(session, block)
                if fam == "unknown":
                    if utils.exclusion_reason(session, block) is None:
                        raise ValueError(
                            f"{session} block {block} is neither in SESSION_BLOCKS "
                            "nor in EXCLUDED_BLOCKS"
                        )
                    continue
                yield dict(
                    subject=utils.subject_of(session),
                    session=session,
                    date=utils.session_date(session),
                    post_implant_day=utils.post_implant_day(session),
                    block=block,
                    task_family=fam,
                    loop_type=utils.loop_type(session, block),
                )

    def _load_features(self, timeline: dict[str, tp.Any], stream: str) -> np.ndarray:
        """SpecialLoader target: one neural stream over the whole block as ``(n_ch, n_bins)``."""
        key = (
            "binned_neural_threshold_crossings"
            if stream == "threshold_crossings"
            else "binned_neural_spike_band_power"
        )
        d = utils.read_block(self._block_path(timeline), fields=[key])
        return np.ascontiguousarray(np.asarray(d[key]).T.astype(np.float32))

    def _neural_rows(
        self, timeline: dict[str, tp.Any], path, n_bins: int
    ) -> list[dict[str, tp.Any]]:
        n_ch = utils.n_channels(path)
        if n_ch != utils.N_CHANNELS:
            raise ValueError(f"{path}: {n_ch} channels, expected {utils.N_CHANNELS}")

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
    def _delay_row(task_family: str, cue: str) -> dict[str, tp.Any]:
        """The on-screen cue over the delay period."""
        row = dict(
            type="Stimulus",
            modality="visual",
            description=cue,
            task_condition="na",
            period="delay",
        )
        if task_family != "finger_movement":
            return dict(row, stimulus_kind="sentence_cue")
        code = utils.cue_code(cue)
        return dict(
            row, stimulus_kind="rest_cue" if code == 0 else "movement_cue", code=code
        )

    @staticmethod
    def _unit_row(task_family: str, cue: str) -> dict[str, tp.Any]:
        """The attempted unit over the go period: a finger movement, rest or a sentence."""
        if task_family == "finger_movement":
            code = utils.cue_code(cue)
            if code == 0:
                return dict(
                    type="Action",
                    action_kind="rest",
                    code=0,
                    description=utils.MOVEMENTS[0],
                    task_condition="na",
                )
            return dict(
                type="Action",
                action_kind="finger_movement",
                code=code,
                description=utils.MOVEMENTS[code],
                task_condition="attempted",
            )
        written = task_family == "handwriting"
        return dict(
            type="Sentence",
            text=cue,
            language="en",
            modality="written" if written else "typed",
            task_condition="attempted",
        )

    def _trial_rows(
        self,
        task_family: str,
        trials: pd.DataFrame,
        bin_times: np.ndarray,
    ) -> list[dict[str, tp.Any]]:
        """Per trial: delay cue, go marker and unit, then an ``Intertrial`` before the next."""
        breaks = utils.clock_breaks(bin_times)

        def crosses_break(a: int, b: int) -> bool:
            return bool(np.any((breaks >= a) & (breaks < b - 1)))

        go_marker = dict(
            type="Stimulus",
            modality="visual",
            description="go cue",
            stimulus_kind="movement_go_cue",
            task_condition="na",
            period="go",
        )
        rows: list[dict[str, tp.Any]] = []
        for i in range(len(trials)):
            tr = trials.iloc[i]
            cue = str(tr["cue"])
            ids = dict(trial_id=i)
            start = utils.bin_index(bin_times, tr["trial_start_ns"])
            go = utils.bin_index(bin_times, tr["go_cue_ns"])
            end = utils.bin_index(bin_times, tr["trial_end_ns"])
            if crosses_break(start, end):
                raise ValueError(f"trial {i} spans a recording pause")
            if task_family != "free_typing":  # free typing shows no cue
                rows.append(
                    dict(
                        **self._delay_row(task_family, cue),
                        **self._span(start, go),
                        **ids,
                    )
                )
            rows.append(dict(**go_marker, **self._span(go, go + 1), **ids))
            if cue:
                rows.append(
                    dict(
                        **self._unit_row(task_family, cue),
                        **self._span(go, end),
                        period="go",
                        success=not (bool(tr["paused"]) or bool(tr["timed_out"])),
                        **ids,
                    )
                )
            elif task_family != "free_typing":
                raise ValueError(f"trial {i} has an empty cue")
            # else: a free-typing trial whose sentence was not recorded (t18.2025.02.04
            # block 5 trial 6), so no unit row
            if i + 1 < len(trials):
                nxt = utils.bin_index(bin_times, trials.iloc[i + 1]["trial_start_ns"])
                if (nxt - end) * _BIN_S >= _MIN_INTERTRIAL_S and not crosses_break(
                    end, nxt
                ):
                    rows.append(
                        dict(
                            type="Intertrial",
                            description="intertrial interval",
                            task_condition="na",
                            period="intertrial",
                            **self._span(end, nxt),
                            **ids,
                        )
                    )
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        path = self._block_path(timeline)
        bin_times = utils.get_bin_times_ns(path)
        handwriting = timeline["task_family"] == "handwriting"
        trials = utils.get_trials(
            path, modality="handwriting" if handwriting else "typing"
        )
        rows = self._neural_rows(timeline, path, len(bin_times))
        rows += self._trial_rows(timeline["task_family"], trials, bin_times)
        return pd.DataFrame(rows)
