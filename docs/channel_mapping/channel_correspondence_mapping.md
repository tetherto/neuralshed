# Channel correspondence across datasets

Several participants appear in more than one dataset. Each Study emits its neural features as columns, and pooling a participant's data across datasets by column index is valid only if column `i` is the same electrode in every dataset. The sources rarely state the electrode-to-column map, and the column order can differ between releases and between sessions of one release. These reports establish, per participant, which correspondences are known and how firmly.

## Scope

The Human Communication cohort. Four participants appear in more than one of its datasets. T16 and T18 appear in one dataset each and need no mapping.

| participant | datasets | report |
|---|---|---|
| T5 | willet_2021, fan_2023 | [t5.md](t5.md) |
| T12 | willet_2023, kunz_2025 | [t12.md](t12.md) |
| T15 | card_2024, wairagkar_2025, kunz_2025, singer_2025 | [t15.md](t15.md), [t15_repeated_trials.md](t15_repeated_trials.md) |
| T17 | jude_2026_speech, jude_2026_typing, kunz_2025 | [t17.md](t17.md) |
| T11 | karpowicz_2025, gusman_2025 | [t11.md](t11.md) |

t5.md also covers T5's four Human Motor datasets, and t15.md covers T15's Human Motor dataset singer_2025.

## Method

- Sources: deposit READMEs, in-file channel labels, the authors' code, and the normalization vectors the online decoders stored.
- Same recordings in two datasets: trials are paired independently of channel order, and each column is correlated with every column of the other dataset. This identifies electrodes directly.
- Different days only: each session is summarized by its channel correlation matrix and each channel's mean level. The stored order should match another session better than other array orders and better than random orders within each array, and the best one-to-one matching of channels should return them to their own positions.
- Calibration: the same tests on two halves of one session and on consecutive days of one dataset, where the answer is known.

A claim is verified when a source states it or the same recording confirms it, and a candidate when it rests on data from different days.

## Results

| participant | array order | electrode order |
|---|---|---|
| T15 | verified. All four sources state it, and the data agree | verified on shared recordings (card_2024 and wairagkar_2025 on 25 days, card_2024 and kunz_2025 on 2024-03-03). Candidate on all other days, and for singer_2025, which shares no recordings |
| T5 | verified for willet_2021. Candidate for fan_2023, which matches same-day recordings in another T5 release | candidate |
| T17 | areas verified from the authors' code and in-file labels. Ventral/dorsal verified for the later jude_2026_speech sessions, jude_2026_typing and kunz_2025's 6v pair. Candidate for the early jude_2026_speech sessions and kunz_2025's 55b pair | candidate |
| T12 | verified for willet_2023, whose deposit gives every column's grid position. Candidate for kunz_2025 | candidate. `ventral_44` rests on convention only |
| T11 | verified for gusman_2025 from in-file labels. Candidate for karpowicz_2025, whose columns 0-95 match gusman_2025's `Lateral` block across days | candidate. Columns 0-95 of the gusman_2025 Gesture Hero sessions disagree in crossing rates with all other sessions |

## Findings that affect use

- T5: fan_2023's arrays are named `lateral_array` and `medial_array`, following willet_2021.
- T12: the two Studies line up by column index only at the candidate level. The December 2024 kunz_2025 files carry T15's array labels, and the data reject the layout those labels imply.
- T15: some trials are the same recording released in two datasets. Keep them on one side of a train/test split.
- T17: the Study reorders the three early jude_2026_speech sessions into the later column order. The data support its ventral/dorsal assignment over the titles in the authors' notebook for 2024-03-25.

## Limits

- Electrode identity across datasets is a candidate everywhere except T15's shared recordings. Only a wiring map from the source or a recording shared between datasets can confirm it.
- Whether a column stays the same electrode from day to day within one dataset depends on the recording rig. All pooling across days, within or between datasets, assumes it.
