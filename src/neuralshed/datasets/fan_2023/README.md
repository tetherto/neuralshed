# Fan et al. 2023 (CORP): one year of attempted handwriting (participant T5)

Intracortical BCI recordings of attempted handwriting from one BrainGate2 participant (T5, C4 spinal cord injury) over 26 days in a year. Two 96-channel Utah arrays in the hand knob of the left precentral gyrus give 20 ms threshold-crossing counts while T5 attempts to write a prompted English sentence letter by letter. The first 11 days trained a seed decoder offline. On the later days the decoder ran live, either frozen or recalibrated online by CORP (Continual Online Recalibration with Pseudo-labels). The authors' Dryad deposit holds the first 21 days. The FALCON H2 benchmark (DANDI 000950) repackages part of them and adds 3 trials from each of the 5 later days.

## References

- Fan, C., Hahn, N., Kamdar, F., Avansino, D., Wilson, G.H., Hochberg, L., Shenoy, K.V., Henderson, J.M., Willett, F.R. (2023). Plug-and-Play Stability for Intracortical Brain-Computer Interfaces: A One-Year Demonstration of Seamless Brain-to-Text Communication. *Advances in Neural Information Processing Systems* 36, 42258-42270. [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/83a14a36de4502bac5b580db36e81858-Abstract-Conference.html). Archived as `references/nihms-1941047.pdf`. Its supplement lists every session and the acquisition chain, archived as `references/fan2023_neurips_supplement.pdf` (sha-256 `19aa1e4e171e9ac818c5dd5037d76534d7c3681636c71f88f4b6bc3c6c20ff1a`).
- Karpowicz, B.M. et al. (2024). Few-shot Algorithms for Consistent Neural Decoding (FALCON) Benchmark. *NeurIPS 2024, Datasets and Benchmarks Track*. [proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/8c2e6bb15be1894b8fb4e0f9bcad1739-Abstract-Datasets_and_Benchmarks_Track.html). Appendix A.3.6 documents H2. Archived as `references/karpowicz2024_falcon_neurips.pdf` (sha-256 `0ad65e4c6f91297047c7dca99ea951767c575e755e162e7bde8df7997a1cdcf9`).
- Source data 1: the CORP deposit on Dryad, [doi:10.5061/dryad.hqbzkh1p6](https://doi.org/10.5061/dryad.hqbzkh1p6), CC0. One zip of 3,567,072,761 bytes and a README, archived as `references/dryad_corp_README.md`.
- Source data 2: DANDI 000950, version 0.241029.1403, [doi:10.48324/dandi.000950/0.241029.1403](https://doi.org/10.48324/dandi.000950/0.241029.1403), CC-BY-4.0.
- Code: [github.com/cffan/CORP](https://github.com/cffan/CORP) (the decoder, the recalibration and the tfrecord reader) and [github.com/snel-repo/falcon-challenge](https://github.com/snel-repo/falcon-challenge) (the converter `preproc/h2_preproc.py`, the loader and the evaluator). Both are cloned under `references/`, at commits `8d87b94` (CORP) and `770c9b9` (falcon-challenge).
- Clinical trial: BrainGate2, NCT00912041. Recorded at Stanford University (NPTL).

## Species & Subjects

- **Species:** human (*Homo sapiens*).
- **T5** (`braingate2/T5` in `datasets/participants.py`, implant 2016-08-17): right-handed man, 69 years old at the time of the study, with a C4 AIS-C spinal cord injury eleven years earlier (Fan et al. 2023, section 4). He can speak and move his head.
- **Implant:** two 96-channel Utah arrays (Blackrock) in the hand knob of the left precentral gyrus. Neither deposit carries electrode or device records, so `array_1` (channels 0-95) and `array_2` (96-191) are neutral names. No source states which array is lateral, or that channel `i` here is the same electrode as channel `i` in `willet_2021`, `deo_2024`, `wilson_2025`, `willsey_2025` or `gusman_2025`, so channel order is treated as dataset-specific.
- **Recording days:** 26, post-implant days 2100-2609 (2022-05-18 to 2023-10-09).

## Behavioral Task

An instructed-delay sentence copying task. On each trial T5 sees a red square with a sentence above it. When the square turns green he attempts to write the sentence letter by letter, as if holding a pen on ruled paper (Dryad README, Fan et al. 2023 section 4). The sentences come from the Brown corpus. Characters are the 26 lowercase letters, `'`, `,`, `?`, a period (`~` in the cue) and a space (`>` in the cue). Trials come in blocks of 10-20, with rest in between.

The paper's 26 sessions are of two kinds (Fan et al. 2023, supplement 2.2, and the Dryad README):

- **Seed sessions** 1-11 (2022-05-18 to 2022-09-01), collected to train the seed decoder. T5 saw no decoder output (`loop_type="open"`).
- **Online sessions** 12-26 (2022-09-29 to 2023-10-09). T5 saw the decoded text in real time (`loop_type="closed"`). Each session ran a **no-recalibration** block, decoded by the frozen seed model, and two **recalibration** blocks, where CORP updated the decoder as T5 wrote.

FALCON's appendix calls H2 "an open-loop attempted handwriting task". The Dryad README and the paper say the online sessions showed the decoder's output, so the Study follows them.

### What each deposit holds

| | Dryad | DANDI 000950 (FALCON H2) |
|---|---|---|
| days | 21 (seed 1-11 and online 12-21) | 26 |
| seed days | every trial (739) | the first 60% of each day (`held-in-calib`) |
| online days 12-21, recalibration blocks | every trial (377) | the first 60% of each day (`held-in-calib`) |
| online days 12-21, no-recalibration blocks | every trial (199) | none |
| online days 22-26 | none ("used for a competition", Dryad README) | the first 3 trials of each day (`held-out-calib`, 15) |
| decoder outputs | `rnn_decoded`, `lm_decoded` on online days | none |
| also | tfrecords of the same trials, a pretrained seed model, the 3-gram language model | `held-in-minival`, a copy of the first 2 `held-in-calib` trials of each day |

A byte-level comparison over all 21 shared days (notebook 01, `tests/test_fan_2023.py`) shows:

- every `held-in-calib` trial is byte-identical (float32) to a Dryad trial. On each day they are the first `n - int(0.4 n)` trials of the seed or recalibration file, with the same cues and block numbers, as `h2_preproc.py` cuts them.
- the remaining 40% (445 trials) are the trials FALCON withholds for scoring, and they are public on Dryad.
- no no-recalibration trial appears in any DANDI file.
- the 15 `held-out-calib` trials appear in no Dryad file.
- no array repeats across the 1,315 Dryad trials.

So the Study reads every Dryad `.mat` and only the 5 `held-out-calib` NWB files. Their block kind is not stated anywhere. It is set to `recalibration`, because the converter's default for held-out days reads `online_recalibration_data`.

### Published splits

Two splits are published, and the Study carries both on every row.

- `original_split`, FALCON's: `held-in-calib` (671, the first 60% of each seed or recalibration file), `held-in-eval` (445, the rest, withheld from DANDI and used by FALCON's server to score submissions), `held-out-calib` (15) and `""` on the no-recalibration blocks, which FALCON did not use. FALCON scores by per-day character and word error rate over whole sentences (`falcon_challenge/evaluator.py`).
- `corp_split`, the authors': their tfrecords put the last 10 trials of each seed day in `test` (110) and every other Dryad trial in `train` (1,205). The seed-model config in CORP reads this test set as validation. `""` on the 15 DANDI trials.

### Recording days

| date | post-implant day | paper session | seed | recalibration | no-recalibration | source | mean rate (Hz) | valid |
|---|---|---|---|---|---|---|---|---|
| 2022-05-18 | 2100 | 1 | 50 | | | Dryad | 10.2 | yes |
| 2022-05-23 | 2105 | 2 | 80 | | | Dryad | 10.4 | yes |
| 2022-05-25 | 2107 | 3 | 60 | | | Dryad | 9.3 | yes |
| 2022-06-01 | 2114 | 4 | 80 | | | Dryad | 9.7 | yes |
| 2022-06-03 | 2116 | 5 | 80 | | | Dryad | 8.3 | yes |
| 2022-06-06 | 2119 | 6 | 78 | | | Dryad | 11.3 | yes |
| 2022-06-08 | 2121 | 7 | 41 | | | Dryad | 11.0 | yes |
| 2022-06-13 | 2126 | 8 | 80 | | | Dryad | 10.8 | yes |
| 2022-06-15 | 2128 | 9 | 60 | | | Dryad | 12.5 | yes |
| 2022-06-22 | 2135 | 10 | 80 | | | Dryad | 11.2 | yes |
| 2022-09-01 | 2206 | 11 | 50 | | | Dryad | 7.5 | yes |
| 2022-09-29 | 2234 | 12 | | 40 | 20 | Dryad | 9.4 | yes |
| 2022-10-06 | 2241 | 13 | | 40 | 20 | Dryad | 10.4 | yes |
| 2022-10-18 | 2253 | 14 | | 40 | 20 | Dryad | 9.8 | yes |
| 2022-10-25 | 2260 | 15 | | 37 | 20 | Dryad | 8.3 | yes |
| 2022-10-27 | 2262 | 16 | | 40 | 20 | Dryad | 7.9 | yes |
| 2022-11-01 | 2267 | 17 | | 40 | 20 | Dryad | 9.0 | yes |
| 2022-11-03 | 2269 | 18 | | 40 | 20 | Dryad | 9.7 | yes |
| 2022-12-08 | 2304 | 19 | | 40 | 20 | Dryad | 8.9 | yes |
| 2022-12-15 | 2311 | 20 | | 20 | 19 | Dryad | 10.1 | yes |
| 2023-02-28 | 2386 | 21 | | 40 | 20 | Dryad | 16.8 | no |
| 2023-04-17 | 2434 | 22 | | 3 | | DANDI | 14.6 | yes |
| 2023-05-31 | 2478 | 23 | | 3 | | DANDI | 10.4 | yes |
| 2023-06-28 | 2506 | 24 | | 3 | | DANDI | 11.1 | yes |
| 2023-08-16 | 2555 | 25 | | 3 | | DANDI | 24.4 | no |
| 2023-10-09 | 2609 | 26 | | 3 | | DANDI | 128.0 | no |

1,330 trials in all. Each online day has one no-recalibration block and two recalibration blocks (one on 2022-12-15). Block numbers skip values within a day. Trials last 7.4-98.5 s (median 40.3 s). The mean rate is over all of a day's bins and channels.

## Data Modalities

| modality | Dryad `.mat` | DANDI NWB |
|---|---|---|
| neural | `tx_feats`, one `(n_bins, 192)` float32 array per trial | `acquisition/binned_spikes`, `(n_bins, 192)` float32 with per-bin `timestamps` |
| cue | `sentences` | `intervals/trials/cue` |
| block | `blocks` | `intervals/trials/block_num` |
| decoder outputs | `rnn_decoded`, `lm_decoded` (online days) | none |

- **Neural chain:** 0.3 Hz-7.5 kHz analog 4th-order Butterworth, 30 kHz sampling, non-causal 300 Hz highpass, threshold at -4.5 x the standard deviation of the voltage, crossings counted per 20 ms bin (Fan et al. 2023, section 3.1 and supplement 2.4). The Dryad README writes the threshold as "-4.5 RMS", the same quantity for a high-passed signal. No common-average reference is mentioned. `willet_2021` records the same arrays at -3.5 x RMS in 10 ms bins with a common-average reference, so the two count streams are not interchangeable.
- **Window:** the go period of each trial only (Dryad README). The delay period and the time between trials are not released.
- **Values:** integer counts 0-20 stored as float32.
- **Rate:** 50 Hz.
- **Behavior:** only the cued sentence, and on online days the live decoder's output. No pen or hand kinematics and no character timing.

## Source Data Format & Layout

```
fan_2023/data/Fan2023Handwriting/
  README.md                                  Dryad README
  dryad_fan_2023_success_download.txt
  dandi_000950_success_download.txt
  CORP_data_release/                         14 GB unpacked
    seed_model_training_data/
      mat/t5.YYYY.MM.DD.mat                  11 files, read
      tfrecords/t5.YYYY.MM.DD/{train,test}/chunk_0.tfrecord
    online_evaluation_data/
      recalibration/mat/t5.YYYY.MM.DD.mat     10 files, read
      recalibration/tfrecords/...
      no_recalibration/mat/t5.YYYY.MM.DD.mat  10 files, read
      no_recalibration/tfrecords/...
    pretrained_seed_model/                   TensorFlow checkpoint, 38 MB
    language_model/TLG.fst, words.txt        3-gram WFST, 9.2 GB
  __MACOSX/                                  zip residue, ignored
  000950/
    sub-T5-held-in-calib/*.nwb               21 files, not read (Dryad copies)
    sub-T5-held-in-minival/*.nwb             21 files, not read (Dryad copies)
    sub-T5-held-out-calib/sub-T5-held-out-calib_ses-<YYYYMMDD>.nwb   5 files, read
```

The `.mat` files are 23-190 MB. Each holds one entry per trial in every field. The tfrecords hold the same trials with transformed (non-integer) features and the CTC targets. Online `test` tfrecords are empty.

The NWB files hold the same per-trial clips laid end to end on a clock with a synthetic 10 s spacer between them (`trial_offset = 10 / 0.02` bins in `h2_preproc.py`). `start_time` and `stop_time` are the timestamps of a trial's first and last bin, so a trial holds `(stop_time - start_time) * 50 + 1` bins. `utils.trial_bins` raises if the clips do not tile the stored bins one trial each.

```
NWBFile
├─ session_start_time  midnight -05:00 on the filename date
├─ subject             Homo sapiens, M, P69Y, subject_id "T5_<split>"
├─ acquisition/
│   ├─ binned_spikes   TimeSeries (n_bins, 192) float32 + timestamps
│   └─ eval_mask       TimeSeries (n_bins,) bool, True everywhere
└─ intervals/trials    id, start_time, stop_time, cue, block_num
```

## Study, timelines and Events

`study.py` (`Fan2023Handwriting`) makes **one timeline per trial** (1,330). `_download()` fetches the Dryad deposit and the published dandiset version, and `utils.list_sources` lists the 36 files the Study reads.

Timeline keys, broadcast to every row: `subject="T5"`, `session` (`t5.YYYY.MM.DD`, as the Dryad files name the day), `date`, `post_implant_day`, `task="attempted_handwriting"`, `task_condition="attempted"`, `block_kind` (`seed`, `recalibration`, `no_recalibration`), `loop_type` (`open` on seed days, `closed` otherwise), `original_split` and `corp_split` (see above), `block` (the source block number) and `trial_id` (the trial's index in its file). Days flagged in `utils.INVALID_DAYS` also carry `valid=False` and a `valid_description`.

| event | per timeline | source | notes |
|---|---|---|---|
| `SpikeCounts` | 1 | the trial's array, read as `(192, n_bins)` at 50 Hz | `feature="threshold_crossings"`, `unit="counts/bin"`, `normalization="raw"`, `threshold="-4.5 x SD"`, `processing` as above |
| `Sentence` | 1 | the cue | `text` in plain English, `cue_raw` as stored, `language="en"`, `modality="written"`, and `rnn_decoded` / `lm_decoded` on the online Dryad days |

Both rows start at 0 and last `n_bins / 50` s, the trial's go period.

## Notes / Scope

- **Flagged days.** Three days depart from every other day in firing rate and carry `valid=False`. The flag comes from our audit: no source calls them bad, and the paper decoded all three online.
  - 2023-10-09 averages 128 Hz per channel against 7-17 Hz on the Dryad days. Every channel fires at 14.6 Hz or more, 13% of bins are zero against 67-88% elsewhere, and its per-channel profile is uncorrelated with the median of the earlier days (r = -0.09).
  - On 2023-02-28 and 2023-08-16, a tenth of channels fire at 37 times or more their median rate on the earlier days.
- **Normalization.** Rates drift across days. Fan et al. z-scored each feature with rolling statistics over the preceding sentences of a block, and FALCON's baseline gives each day its own input layer. Normalize per day at least.
- **Text targets.** The canonical `v1` charset of `text.py` keeps letters, space and apostrophe, so `,`, `.` and `?` become word breaks under `CharacterEncoder`.
- **Repeated cue.** One cue repeats within a day: 2022-06-15 trials 15 and 26 (`source>telerate>systems>inc~`).
- **Not read.** The tfrecords (the same trials, transformed), the pretrained seed model, the language model and the FALCON held-in files, which copy Dryad trials. The decoder outputs ride on the `Sentence` rows but are not targets.
- **Out of scope.** The trials of days 22-26 beyond the first three are not public.

## Files

- `utils.py`: constants, the source listing, the `.mat` and NWB readers, `trial_bins`, the FALCON and CORP split rules and the flagged days.
- `metadata.py`: catalog metadata, `sessions()` and `get_metadata()` for the database.
- `study.py`: the `Fan2023Handwriting` Study.
- `01_explore_source_data.ipynb`: the two deposits, their byte-level overlap and the per-trial clip clock.
- `02_explore_study.ipynb`: the Study's events, read back through the extractors.
- `03_model_example.ipynb`: a character-level CTC decoder tested on the held-out days.
