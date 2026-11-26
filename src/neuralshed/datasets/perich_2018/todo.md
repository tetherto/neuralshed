# perich_2018: things to sort out

## Shared-file changes needed

Applied: the `studies.py` import, the four monkeys in `participants.py` (`miller/C`, `J`, `M`, `T`, under a `miller` lab namespace), `cm`, `cm/s` and `cm/s^2` in `units.py`, and `executed` in `vocab.TASK_CONDITIONS`. Still open:

- **Spike binning helper**: `utils.bin_spikes` bins a Units table onto a uniform grid. Every NWB monkey dataset with sorted units needs the same thing, so it could live in `src/neuralshed/datasets/nwb.py`:

  ```python
  def bin_spike_times(
      spike_times: list[np.ndarray], t0: float, n_bins: int, bin_s: float
  ) -> np.ndarray:
      """Counts in [t0 + i * bin_s, t0 + (i + 1) * bin_s), (n_units, n_bins) float32.

      The 1e-6 offset puts a spike that falls exactly on an edge in the bin starting there.
      """
      counts = np.zeros((len(spike_times), n_bins), dtype=np.float32)
      for row, times in enumerate(spike_times):
          index = np.floor((times - t0) / bin_s + 1e-6).astype(np.int64)
          index = index[(index >= 0) & (index < n_bins)]
          counts[row] = np.bincount(index, minlength=n_bins)
      return counts
  ```

- **`src/neuralshed/models/data.py`** (only if perich_2018 should join the motor cursor collection): the collection's declared target is `DecodedCursorVelocitySeries`, which this dataset does not have (its cursor follows the hand). It emits `CursorVelocitySeries`, the same type `DeriveStream(operation="velocity")` produces. A member entry would look like this, with the target question in the decisions below.

  ```python
  MOTOR_CURSOR_MEMBERS["perich_2018"] = ("Perich2018Reaching", "perich_2018", "perich_2018/data")
  MOTOR_CURSOR_MEMBER_QUERIES["perich_2018"] = "valid == True"
  MOTOR_CURSOR_PARTICIPANTS["perich_2018"] = "miller/C"  # one per member today, but this member has four monkeys
  MOTOR_CURSOR_REGIONS["perich_2018"] = "..."  # no NHP (non-human primate) region value exists yet
  ```

## Decisions for Heberto

- **Monkey identity across datasets.** C and M are almost certainly Chewie and Mihili of `ma_2023`: same lab, same center-out task, and `ma_2023`'s `Chewie_CO_2016` and `Mihili_CO_2014` dates (for example 2016-09-29, 2016-10-06, 2014-02-03, 2014-02-17) are perich_2018 session dates. J might be Jango, but `ma_2023`'s Jango is isometric-wrist data, so that is unverified. Recommendation: rename `miller/C` and `miller/M` to `miller/Chewie` and `miller/Mihili` once the shared session dates are checked against `ma_2023`, so both datasets point to one id.
- **Where `valid=False` goes for the half-scale sessions.** Following the brief, it is on the timeline. Only the cursor is suspect, the units are fine. Recommendation: move the flag to the three cursor rows and the trial rows, so neural-only work can still use the 7 sessions (about 1.9 hours with both arrays).
- **Neural row layout.** One `SpikeCounts` row per session with every sorted unit, M1 first, `n_units_m1` / `n_units_pmd` giving the split. Alternatives: one row per array (selectable by `brain_region`, needs the extractor's `aggregation="cat"` to recombine), or units summed per electrode (fixed 96 or 192 channels, closer to threshold crossings, loses isolation). Recommendation: keep one row, add per-array rows only if a model needs region selection.
- **J as sorted or threshold crossings.** Declared `method="sorted"` because its files look sorted. Recommendation: keep it and ask the authors.
- **Array names.** `metadata.py` names arrays `M1` and `PMd` as the files do. The standardization rule prefers area names (`4`, `6d`) for human arrays. Recommendation: keep `M1` / `PMd` for monkeys unless a cross-species naming rule is wanted.
- **Hemisphere of M.** Set to `right` from `ma_2023`'s README (left hand, contralateral arrays). Recommendation: keep, with the comment in `metadata.py`, or set `None` if inferences from another release are unwanted.
- **Target stream.** No `TargetPositionSeries` is emitted because the release has none. One could be built from the trial table (target from onset to trial end), which would let the collection's intention and aligned-velocity derivations run. Recommendation: add it when a collection needs it, and decide what the stream holds between trials first.
- **`status`.** Set to `study_wip` because the notebooks are missing.

## Data problems

- **Half-scale cursor in 7 sessions** (C CO 2016-09-09 to 2016-09-23): truncated "Warning:" in `session_description`, reaches end at about 4.2 cm against 8 cm targets. Timelines get `valid=False`, values untouched. The Study raises if the warning text and `utils.HALF_SCALE_SESSIONS` disagree.
- **`target_corners` shifted by one row in 8 CO sessions** (verified: C 2013-12-03, 2013-12-04, 2013-12-19, M 2014-02-03, 2014-02-18, 2014-06-26, 2014-06-27, T 2013-08-19). Not read. Targets come from `target_dir` at 8 cm, verified as the corner radius in all 84 CO files.
- **CO `target_id` of 8 or 12 with NaN `target_dir`** in 5 trials: `code=-100`, NaN target.
- **24 rewarded CO trials with no target or go cue**, 22 shorter than 0.1 s: emitted as `Intertrial` ("rewarded trial with no logged target").
- **RT target coordinates**: NaN, float garbage near 1e37, and placeholders at exactly (0, 0) (757 reaches, successful ones end a median 5.5 cm away). All become NaN.
- **RT rewarded trials with no targets or go cues** (15 to 96 per session): one `Action` over the trial, `reach_index=-1`, `success=True`, NaN target.
- **Zero instructed delay** in nine C CO sessions is really 1 ms. Kept verbatim on `delay` and as a 1 ms delay `Stimulus`.
- **Cursor timestamps** in C 2016-09-23 and 2016-10-11 swap 0 ms and 20 ms steps. Treated as a uniform grid (no sample more than one bin off).
- **Spikes after the last cursor sample** (up to about 2 s, 1,708 spikes in C 2016-09-09) are dropped by the binning.
- **Monkey T's M1 array** is in every T file but not in the papers. Kept as declared.
- **Threshold-crossing subject** unnamed by the dandiset. Everyone declared sorted.
- **C and M share `date_of_birth`** (2008-03-01). Ages computed from it as given.
- **`dandiset.yaml`** asset summary (99 files, 3 subjects) and citation DOI are stale placeholders. Noted in `metadata.py`.
- **Filtering** is "High pass 750 Hz" in the files against 250 to 5000 Hz band-pass in the papers. The Study's `processing` string follows the files.

## Not done

- Notebooks `01_explore_source_data.ipynb`, `02_explore_study.ipynb`, `03_model_example.ipynb`.
- `python -m neuralshed.database.validate --datasets perich_2018` and `--check-sessions` pass (111 timelines, 111 labels) with the data at `perich_2018/data/Perich2018Reaching/000688/`.
- The streaming path is untested: the `dandi` package is not installed.
- Joining the motor cursor collection is untested (target type, units and participant mapping above).
