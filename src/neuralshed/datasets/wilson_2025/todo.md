# wilson_2025 — open items

Findings from a full review pass against the 103 released `.mat` files and the authors' reference code (`references/nonstationarities`). Everything else checked out: schema validation, session cross-check, the 192 unit tests, lint, the three notebooks (executed in order, no error cells), the `_info` guard, the Hz scaling and `-3.5 x RMS` header corpus-wide, the field-14 click flag (560/560), the 73-session cohort, the source dtypes, and the cable-swap / duplicate-block / trial-epoch conversions.

## P1 — metadata

### 4. `age: "P69Y"` is ambiguous across the T5 datasets

The paper states it ("T5 was a right-handed man who was 69 years old at the time of the study"), but `fan_2023` declares `P69Y` for the same person 2,060 days later, and Willett 2021 puts him at 65 in 2019. A dataset spanning 2016-2021 has no single true age anyway.

**Fix.** Either set `age` to `None` or keep `P69Y`, in both cases with a comment recording the three-way source conflict.

### 5. `on_target` (field 8) is read, documented, and never emitted

Binary, non-zero on all 560 blocks, and it is the dwell-selection signal the task is built on. `utils._load` exposes it, the README's Data Modalities table lists it, and the reference code carries it, but no event carries it downstream. Per the onboarding guide's own rule, an attribute that is not emitted cannot be filtered later.

**Fix.** Add `OnTargetSeries(BehavioralSeries)` to `neuralshed/events.py`, emit it with `unit="binary"` and `dims=("on_target",)`, and add it to the README event table and to `metadata.py`'s events list.

### 6. `metadata.py` and `study.py` disagree on the `target_cue` event

`metadata.py` gives the `Stimulus` `task_condition: "attempted"` and no `modality`, while `study.py` emits `modality="visual"` and no `task_condition`. Nothing is attempted about displaying a target.

**Fix.** Drop `task_condition` from the `target_cue` entry and add `"modality": "visual"`, so the declaration matches what the Study emits.

## P2 — doc precision

- `utils.py` says "all 564 released blocks" in two comments (the field-14 note and `_windows_in_block`) where the exposed and actually-verified count is 560 (564 minus the 4 duplicated blocks on `t5.2019.01.14`).
- The README says the neural peak is "800 Hz (16 counts in a bin)". That is `t5.2016.09.26`. The corpus max is 1000 Hz, 20 counts per bin.
- The reference drops trials shorter than 6 bins (`if stop - start > 5`). We keep all of them, 21 of 68,327 (0.03%, all in `cursor` blocks). Harmless, currently undocumented.
- `cursor_size` is parsed and never surfaced, and `intertrial_windows` is used by neither the study nor the notebooks.
- The local `neuralshed_metadata.db` holds 11 datasets and no `wilson_2025`. The file is gitignored, so this is a local artifact and needs only a `populate` run.

## Order of work

1. **P0.** `SCREEN_ORIGINS` in `utils.get_series`, `unit="a.u."` on the velocity row, register the study.
2. **P1.** `OnTargetSeries`, the `age` decision, the `target_cue` mismatch.
3. **P2.** Wording corrections, a note on the short-trial divergence, drop or use the dead helpers.
4. **Re-verify.** Steps 1-2 change emitted numbers for keyboard positions and add an event type, so re-run `update_source_info` (the `_info` event-type set gains `OnTargetSeries` and `num_events_in_query` changes), re-execute all three notebooks, re-run `validate` both ways, `pytest`, `ruff`, and `populate`. Notebook 01 should also gain a cell showing the per-task screen origin and the velocity gain regimes, since both are the kind of verify-against-the-data finding that notebook exists to expose.
