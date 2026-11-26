"""neuralset Study for Wilson et al. 2025 closed-loop cursor control.

One timeline is one section of a task block: the bins between two recording pauses, or
the whole block when it has none. A released ``.mat`` holds a whole recording day, but
the task label and the online decoder change from block to block, so the block is the
smallest unit with invariant task entities, and a pause leaves the bins on either side
on different clocks (``utils.section_windows``). Neural counts, cursor and target
trajectories, and the online decoder outputs stay whole-section lazy streams; trial
timing becomes a visual ``Stimulus`` and an attempted-movement ``Action`` per trial, or
an ``AutoMovement`` where the 2021 task moves the cursor back to the centre.
"""

import typing as tp

import numpy as np
import pandas as pd

from neuralset.events import study

from neuralshed.datasets.study import NeuralshedStudy
from neuralshed.events import (  # noqa: F401, registers custom types
    AutoMovement,
    ClickStateSeries,
    CursorPositionSeries,
    DecodedCursorVelocitySeries,
    SpikeCounts,
    TargetPositionSeries,
)

from . import utils
from .metadata import _implant_date

_SUBJECT = "T5"
_RATE_HZ = utils.FS
# Trial fields an AutoMovement keeps: it has no code, target or outcome.
_AUTO_MOVEMENT_KEYS = ("start", "duration", "trial_id", "valid", "valid_description")


class Wilson2025CursorRecalibration(NeuralshedStudy):
    """Five years of closed-loop cursor control from T5's hand-knob arrays."""

    dataset_id: tp.ClassVar[str] = "wilson_2025"
    aliases: tp.ClassVar[tuple[str, ...]] = (
        "wilson_2025",
        "PRI-T",
        "cursor-recalibration",
    )
    bibtex: tp.ClassVar[str] = """
@article{wilson2025recalibration,
  title = {Long-term unsupervised recalibration of cursor-based intracortical brain-computer interfaces using a hidden Markov model},
  author = {Wilson, Guy H. and Stein, Elias A. and Kamdar, Foram and Avansino, Donald T. and Pun, Tsam Kiu and Gross, Ronnie and Hosman, Tommy and Singer-Clark, Tyler and Kapitonava, Anastasia and Hochberg, Leigh R. and Simeral, John D. and Shenoy, Krishna V. and Druckmann, Shaul and Henderson, Jaimie M. and Willett, Francis R.},
  journal = {Nature Biomedical Engineering},
  volume = {10},
  pages = {1466--1484},
  year = {2026},
  month = jul,
  doi = {10.1038/s41551-025-01536-z},
  url = {https://doi.org/10.1038/s41551-025-01536-z},
}
"""
    url: tp.ClassVar[str] = "https://doi.org/10.5061/dryad.1jwstqk6g"
    licence: tp.ClassVar[str] = "CC0-1.0"
    description: tp.ClassVar[str] = (
        "Closed-loop two-dimensional cursor control by BrainGate2 participant T5 using "
        "two 96-channel Utah arrays in the hand knob of left dorsal motor cortex, across "
        "103 recording days spanning 2016 to 2021. The released 50 Hz streams include "
        "threshold crossings, cursor and target positions, and the online cursor and "
        "click decoder outputs."
    )
    timeline_descriptions: tp.ClassVar[dict[str, dict[tp.Any, str]]] = {
        "task": {
            "cursor": (
                "In the cursor task, most blocks place eight or six outer targets around "
                "a centre target, and the cursor goes from the centre out to a target "
                "and back, one trial each way."
            ),
            "keyboard": (
                "In the keyboard task, the screen is a grid of equally sized cells, up to "
                "57 in a block, and one cell lights up as the target."
            ),
            "fitts": (
                "In the fitts task, one target at a time appears at a random position."
            ),
        },
        "loop_type": {
            "closed": "In this block T5 controlled the cursor through the online decoder.",
            "open": (
                "In this block the computer moved the cursor, on straight outward paths, "
                "while T5 attempted the movement."
            ),
        },
        "click_enabled": {
            True: (
                "Targets in this block were selected by a neural click decoder, whose "
                "output is the ClickStateSeries row."
            ),
            False: "Targets in this block were selected by holding the cursor on them.",
        },
    }
    requirements: tp.ClassVar[tuple[str, ...]] = ("scipy",)

    # One timeline per emission path: 0 radial-8 cursor, 8 keyboard grid, 27 binary
    # click, 123 fitts with continuous click, 261 all-zero decoder output, 456 channels
    # 0-95 unrecorded, 687 trial cut by a recording pause, 749 held-cursor delay/go,
    # 857 INVALID_SESSIONS day, 909 2021 lockout (AutoMovement, 0/2 click), 948 open
    # loop. Most extra ones hold a single trial, to keep the query small.
    # NB: `neuralfetch.utils.update_source_info` computes the counts from this `query`
    # but omits `query` and `frequency` when it rewrites the block, so re-add both by
    # hand after running it, or the counts guard the wrong query.
    _info: tp.ClassVar[study.StudyInfo] = study.StudyInfo(
        num_timelines=1014,
        num_subjects=1,
        query="timeline_index in [0, 8, 27, 123, 261, 456, 687, 749, 857, 909, 948]",
        num_events_in_query=1025,
        event_types_in_query={
            "Action",
            "AutoMovement",
            "ClickStateSeries",
            "CursorPositionSeries",
            "DecodedCursorVelocitySeries",
            "SpikeCounts",
            "Stimulus",
            "TargetPositionSeries",
        },
        frequency=50.0,
    )

    def _download(self, overwrite: bool = False) -> None:
        """Download the Dryad release into this Study's folder.

        Dryad requires ``NEURALFETCH_DRYAD_TOKEN``. The deposit is a single 1.7 GB
        tarball that unpacks to roughly 30 GB, most of which is method-sweep output.
        """
        from neuralshed.datasets.dryad import Dryad

        Dryad(
            study="wilson_2025",
            dset_dir=self.path.parent,
            folder=self.path.name,
            doi="10.5061/dryad.1jwstqk6g",
        ).download(overwrite=overwrite)

    @property
    def _raw_dir(self):
        return self.path / "PRITpaper2025" / "T5" / "raw_data"

    def _ref(self, timeline: dict[str, tp.Any]) -> utils.BlockRef:
        return utils.BlockRef(
            session=timeline["session"],
            block=timeline["block"],
            task=timeline["task"],
            group=timeline["path"].split("/")[0],
            path=self._raw_dir / timeline["path"],
        )

    def _section(self, timeline: dict[str, tp.Any]) -> tuple[int, int]:
        return utils.section_windows(self._ref(timeline))[timeline["section"]]

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        implant = _implant_date()
        for ref in utils.list_blocks(self._raw_dir):
            date = utils.session_date(ref.session)
            unrecorded = utils.unrecorded_arrays(ref)
            for section, (first, last) in enumerate(utils.section_windows(ref)):
                reasons = []
                if ref.session in utils.INVALID_SESSIONS:
                    reasons.append(utils.INVALID_SESSIONS[ref.session])
                # A section whose array has bins its NSP did not record is flagged
                # whole, since one SpikeCounts row holds both arrays.
                for array, channels in enumerate(("0-95", "96-191")):
                    missing = int(unrecorded[first:last, array].sum())
                    if missing:
                        reasons.append(
                            f"channels {channels} not recorded for "
                            f"{missing / _RATE_HZ:.2f} of {(last - first) / _RATE_HZ:.2f} s"
                        )
                flag = (
                    {"valid": False, "valid_description": "; ".join(reasons)}
                    if reasons
                    else {}
                )
                yield {
                    "subject": _SUBJECT,
                    "session": ref.session,
                    "run": ref.run,
                    "task": ref.task,
                    "block": ref.block,
                    # Recording pauses cut a block into sections, numbered from 0.
                    "section": section,
                    # The session file under its release folder, `historical`
                    # (2016-2020) or `new` (2021).
                    "path": ref.path.relative_to(self._raw_dir).as_posix(),
                    # The release ships no train/test partition (README).
                    "original_split": "",
                    "date": date.isoformat() if date else None,
                    "post_implant_day": (
                        (date - implant).days if date and implant else None
                    ),
                    # The authors' own block curation; keeping only sessions with at
                    # least two such blocks reproduces the paper's 73-session cohort.
                    "analysis_block": ref.is_analysis_block,
                    "click_enabled": utils.click_enabled(ref),
                    "loop_type": ref.loop_type,
                    **flag,
                }

    def _load_series(self, timeline: dict[str, tp.Any], key: str) -> np.ndarray:
        """SpecialLoader target: source time-first data as dims-first float32."""
        start, stop = self._section(timeline)
        data = utils.get_series(self._ref(timeline), key)[start:stop]
        return np.ascontiguousarray(data.T, dtype=np.float32)

    def _continuous_rows(
        self, timeline: dict[str, tp.Any], ref: utils.BlockRef
    ) -> list[dict[str, tp.Any]]:
        start, stop = self._section(timeline)
        duration = (stop - start) / _RATE_HZ

        def row(*, type_: str, key: str, dims=(), unit: str = "", **extra):
            out = {
                "type": type_,
                "feature": key,
                "start": 0.0,
                "duration": duration,
                "frequency": _RATE_HZ,
                "unit": unit,
                "filepath": study.SpecialLoader(
                    method=self._load_series,
                    timeline=timeline,
                    key=key,
                ).to_json(),
                **extra,
            }
            if dims:
                out["dims"] = dims
                out["n_features"] = len(dims)
            return out

        rows = [
            # The source scales the 20 ms bin count to a rate, so the unit is Hz rather
            # than the counts/bin other pre-binned datasets emit (see SPIKE_RATE_SCALE).
            row(
                type_="SpikeCounts",
                key="threshold_crossings",
                n_features=utils.N_CHANNELS,
                unit="Hz",
                normalization="raw",
                method="threshold_crossing",
                threshold=f"{utils.RMS_THRESHOLD} x RMS",
                # The paper's Methods; every released value is a whole count times 50.
                processing=(
                    "downsampled to 1 kHz; threshold crossings counted per 20 ms bin; "
                    "counts x 50 to Hz"
                ),
            ),
            row(
                type_="CursorPositionSeries",
                key="cursor_position",
                dims=utils.POSITION_DIMS,
                unit="px",
                kind="cursor_position",
                reference_frame="screen_center",
            ),
            row(
                type_="TargetPositionSeries",
                key="target_position",
                dims=utils.POSITION_DIMS,
                unit="px",
                kind="target_position",
                reference_frame="screen_center",
            ),
            # Not px/s: one unit moves the cursor about 20 px per bin on 163 blocks and
            # about 1e5 px per bin on the other 389, so the scale is per block (README).
            row(
                type_="DecodedCursorVelocitySeries",
                key="cursor_decoder_output",
                dims=utils.POSITION_DIMS,
                unit="a.u.",
                kind="online_decoder_output",
                reference_frame="screen",
            ),
        ]
        if not utils.get_series(ref, "cursor_decoder_output").any():
            # Eight blocks store zeros over the whole block while the cursor moves, so
            # the zeros are not what moved it (README).
            rows[-1]["valid"] = False
            rows[-1]["valid_description"] = (
                "decoder output all zeros while the cursor moves"
            )
        elif ref.loop_type == "open":
            # On open-loop blocks field 10 is the cursor's own step (README).
            rows[-1]["valid"] = False
            rows[-1]["valid_description"] = (
                "open-loop block: the computer's cursor command, not decoder output"
            )
        if utils.click_enabled(ref):
            # Field 11 holds 0 and 1 on the 138 click blocks of 2016 and 2017, 0 and 2
            # on the 55 of 2021, and continuous values on t5.2016.10.31 blocks 7 and 9.
            # Only 0 and 1 lines up with selections as a click (README).
            values = set(
                np.unique(utils.get_series(ref, "click_decoder_output")).tolist()
            )
            click = row(
                type_="ClickStateSeries",
                key="click_decoder_output",
                dims=("click",),
                unit="binary",
                kind="online_decoder_output",
            )
            if values == {0.0, 2.0}:
                click["unit"] = "a.u."
                click["valid"] = False
                click["valid_description"] = "click values 0 and 2, meaning unknown"
            elif not values <= {0.0, 1.0}:
                click["unit"] = "proportion"
                click["valid"] = False
                click["valid_description"] = (
                    "click decoder output is continuous, not a binary state"
                )
            rows.append(click)
        return rows

    def _trial_rows(
        self, ref: utils.BlockRef, section: tuple[int, int]
    ) -> list[dict[str, tp.Any]]:
        windows = utils.trial_windows(ref)
        if not windows:
            return []
        first, last = section
        targets = utils.get_series(ref, "target_position")
        sizes = utils.get_series(ref, "target_size").ravel()
        success = utils.trial_success(ref)
        trial_ids = utils.trial_indices(ref)
        held = ref.block in utils.HELD_CURSOR_BLOCKS.get(ref.session, ())
        cursor = utils.get_series(ref, "cursor_position")

        # Target codes are block-local: the index into this block's sorted set of
        # distinct target centres. Most `cursor` blocks are radial-8 plus a centre
        # target, but some use six targets, the grid task has up to 57 cells and
        # `fitts` places targets at random, so a fixed spoke code would not hold.
        # `target_x`/`target_y` carry the geometry when comparing across blocks.
        cued = np.array([targets[start] for start, _ in windows])
        layout = np.unique(cued, axis=0)
        codes = [
            int(np.flatnonzero((layout == target).all(axis=1))[0]) for target in cued
        ]
        # A `cursor` block with a centre target goes centre, out to a target, back to
        # the centre, so its returns share the centre's code whichever target they
        # leave from.
        centre_out = ref.task == "cursor" and bool((layout == 0).all(axis=1).any())

        rows: list[dict[str, tp.Any]] = []
        for index, (start, stop) in enumerate(windows):
            if not first <= start < last:
                continue
            target = cued[index]
            common = {
                "start": (start - first) / _RATE_HZ,
                "duration": (min(stop, last) - start) / _RATE_HZ,
                "code": codes[index],
                "trial_id": trial_ids[index],
                "target_x": float(target[0]),
                "target_y": float(target[1]),
                "target_size": float(sizes[start]),
                "n_targets": len(layout),
            }
            if centre_out:
                common["trial_kind"] = "return" if (target == 0).all() else "center_out"
            if stop > last:
                # The trial's last bin comes after a recording pause, so the row ends
                # at the pause.
                common["valid"] = False
                common["valid_description"] = "trial spans a recording pause"
            stimulus = {
                "type": "Stimulus",
                "modality": "visual",
                "description": "cued cursor target",
                "stimulus_kind": "target_cue",
                **common,
            }
            action = {
                "type": "Action",
                "description": (
                    "attempted hand movement driving the cursor"
                    if ref.loop_type == "closed"
                    else "attempted hand movement while the computer moved the cursor"
                ),
                "task_condition": "attempted",
                "action_kind": "cursor_control",
                "success": success[index],
                **common,
            }
            end = min(stop, last)
            moved = np.flatnonzero((cursor[start:end] != cursor[start]).any(axis=1))
            returning = bool((target == 0).all())
            go = utils.lockout_end(ref, start, end)
            if go is not None and returning and len(moved):
                # During the 2021 lockout the task moves the cursor back to the centre,
                # so the return is the task's movement, not T5's (README).
                rows.append(stimulus)
                rows.append(
                    {
                        "type": "AutoMovement",
                        "description": "return to centre",
                        **{
                            key: common[key]
                            for key in _AUTO_MOVEMENT_KEYS
                            if key in common
                        },
                    }
                )
                continue
            # A trial whose cursor never moves has no go, so its Stimulus and Action
            # both span the whole trial.
            if len(moved) and (held or (go is not None and not returning)):
                if held:
                    # The target is cued while the cursor is held, and the go is the
                    # first bin the cursor has left its start position. Otherwise a 2021
                    # outward trial opens with the lockout, and the go is the first bin
                    # of nonzero decoder output.
                    go = start + int(moved[0])
                stimulus["duration"] = (go - start) / _RATE_HZ
                stimulus["period"] = "delay"
                action["start"] = (go - first) / _RATE_HZ
                action["duration"] = (end - go) / _RATE_HZ
                action["period"] = "go"
                for row in (stimulus, action):
                    row["description"] += f", {row['period']} period"
            rows.append(stimulus)
            rows.append(action)
        return rows

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        ref = self._ref(timeline)
        rows = self._continuous_rows(timeline, ref)
        rows.extend(self._trial_rows(ref, self._section(timeline)))
        return pd.DataFrame(rows)


__all__ = ["Wilson2025CursorRecalibration"]
