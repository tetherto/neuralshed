# T15 repeated trials

Several T15 trials appear in more than one dataset. They are the same recording, cut and processed differently by each release. A model trained on one copy and tested on the other has already seen its test data. This page lists where the copies are, shows how they line up, describes how to find them, and explains how to keep them out of a test set.

neuralshed does not flag or remove the copies. A collection that pools these datasets holds both, and the method below finds them so a split can group them.

## Where the copies are

| datasets | shared dates | repeated trials |
|---|---|---|
| card_2024 and wairagkar_2025 | 25 dates, 2023-08-11 to 2023-12-29 | most wairagkar trials on those dates (about 97%). On the card side they include `train`, `val` and several hundred `test` trials |
| card_2024 and kunz_2025 | 2024-03-03 and 2024-06-14 | the 200 trials of kunz `t15.2024.03.03_attempted_50words`, all card `train`. The imagined-speech files of 2024-03-03 and the file of 2024-06-14 share no trial with card |
| wairagkar_2025 and kunz_2025 | none | none |

Three cases need care when splitting:

- card hides the sentence of its `test` trials, but their wairagkar copies carry it. Training on wairagkar exposes several hundred of card's held-out trials with their labels.
- All trials of wairagkar day 110 (2023-11-04), its first evaluation session and part of its `eval` split, are card trials, mostly `train` with some `val` and `test`. Training on card exposes part of wairagkar's evaluation set.
- singer_2025 recorded T15 on 2023-08-25, a date shared with card and wairagkar. It is a cursor task and was not checked.

## Same trial, different clip

Each trial follows the same sequence. A red square shows the sentence (the delay, 1.5-4 s), the square turns green (the go cue), T15 attempts to speak, and he ends the trial by looking at a "Done" icon. The releases cut this sequence at different points:

```
                 delay (red square)   go cue   speech (green square)        Done
recording    ─────────────────────────────┼──────────────────────────────────┤
card_2024                  ├──── ~1 s ────┼──────────────────────────────────┤
wairagkar_2025                            ├──────────────────────────────┤
```

- card_2024 starts its clip about 1 s before the go cue, so it includes the end of the delay. On the first trial of each block it starts at the go cue.
- wairagkar_2025 keeps only the go period. Its clip starts at the go cue and ends 0.05-0.15 s before card's.
- kunz_2025 stores whole blocks and marks the go cue of each trial.

Measured from the card clip start, nearly all wairagkar copies start about 1 s later, and copies of the first trial of a block start at the same moment. Most of the rest belong to block 7 of 2023-08-18, whose copies start about 1.2 s later.

The values also differ, because each release processes the signal its own way. Copies of a trial are strongly correlated but never identical.

| release | bin | normalization |
|---|---|---|
| card_2024 | 20 ms | z-scored over the preceding 20 trials, clipped at 10 SD |
| wairagkar_2025 | 10 ms | log-transformed, normalized over a rolling 10 s window, smoothed |
| kunz_2025 | 10 ms | raw |

## Finding the copies

The search uses the neural signal only. It never reads the sentence, so it also finds copies of card `test` trials, and the sentence serves as an independent check afterwards.

1. Compare only trials recorded on the same date. Copies come from the same session.
2. Bring both releases to 20 ms bins, averaging pairs of 10 ms bins. Both ways of pairing the 10 ms bins are tried.
3. Fingerprint each trial by summing all 256 threshold-crossing columns into one trace of overall activity. Slide one trace against the other over the plausible offsets and keep the best correlation. For wairagkar the offsets are 0-2 s after the card clip start. For kunz they are the card clip start from 2 s before to 0.5 s after each go cue. The sum does not depend on column order, so this step needs no channel map.
4. Keep the 5 card trials with the best fingerprint as candidates.
5. Check each candidate channel by channel. At its offset, correlate column `i` of one copy with column `i` of the other, for both features, and take the median over columns. This relies on the column map verified in [t15.md](t15.md).
6. Accept the best candidate when both medians reach 0.4. Each card trial may be paired once, and the highest channel correlation wins.

The fingerprint only ranks candidates, because on its own it cannot tell copies from other trials. Of the wairagkar trials left without a copy, more than half have a best fingerprint of 0.5 or more, some above 0.9. The channel check leaves a wide gap:

| | copies | other trials |
|---|---|---|
| fingerprint correlation | about 0.5-0.95 | up to about 0.9 |
| median channel correlation, threshold crossings | about 0.5-0.6 (kunz 1.00) | about 0.1 or less |
| median channel correlation, spike-band power | about 0.5-0.65 (kunz 1.00) | about 0.2 or less |

The sentence confirms the result. Every card and wairagkar pair where both releases show the sentence has the same sentence, and so do all 200 card and kunz pairs.

## Using the copies in a split

- Treat each recording as one unit. Put all its copies in the same split, or keep only one copy.
- A split by sentence keeps the copies together only where both copies show the sentence. card `test` trials have no sentence, so a sentence split cannot place them. Give each one the sentence of its wairagkar copy, or group the pair directly.
- A sentence split must normalize text the same way across datasets (case, punctuation, the kunz keyword), or one sentence becomes two groups.
- To compare with card's held-out results, drop the wairagkar copies of card `test` trials from training.
- To compare with wairagkar's evaluation, drop the card copies of the day 110 trials from training.
- card and wairagkar normalize each trial with statistics from neighboring trials, so some information about a trial's neighbors is already in its features. No split can undo this.

## Limits

- About 200 wairagkar trials and about 1,000 card trials on the shared dates have no copy. Most are probably not in the other release. A copy would be missed only if it ranked outside the 5 best fingerprints.
- singer_2025 was not checked.
- The same sentence recorded in different sessions is a different recording, and this page does not cover it. Prompts repeat across sessions and datasets, which is a separate form of leakage, handled by grouping on the sentence text (see the [kunz_2025 README](../../src/neuralshed/datasets/kunz_2025/README.md)).
