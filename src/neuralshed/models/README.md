# Extensible cross-subject brain-to-text

Can several people's intracortical recordings be pooled into **one** decoder that beats a separate decoder trained per person, and can it keep absorbing new sessions and subjects? The notebooks here are a small, runnable testbed for that question, built on the existing neuralset `Study` objects in this repo.

Scope is one **collection** of studies — the neural → text studies, whose decode target is a line of English:

| Study | Subject | Cortex | Spikes | Spike-band power | Task |
|---|---|---|---|---|---|
| [`card_2024`](../datasets/card_2024) | T15 | speech precentral | 256 ch | 256 ch | attempted speech |
| [`willett_2023`](../datasets/willet_2023) | T12 | speech precentral | 256 ch | 256 ch | attempted speech |
| [`fan_2023`](../datasets/fan_2023) | T5 | hand-knob motor | 192 ch | — | attempted handwriting |

Three subjects, three implants, two paradigms, different channel counts, and one dataset with no spike-band power. The only thing shared is the target text.

## The model (`model.py`)

```
 spikes ─┐
         ├─[ per-SESSION read-in adapter ]─► (B,T,D) ─┐
 sbp   ──┘   (+ optional SBP head)                    ├─ + region emb + subject emb
                                                      ▼
                                  shared conv + BiGRU backbone
                                                      │
                                   CTC head ◄─────────┴─────────► masked-reconstruction head
                                   (characters)                  (self-supervised pretraining)
```

- **Per-session read-in adapters** — one linear map per recording into a shared width `D`, absorbing the channel-count mismatch (192 vs 512) and per-array drift. A new session/subject is just a new adapter, so the model is extensible by construction.
- **Region + subject embeddings** condition the shared backbone on each clip's provenance (zero-initialized so they start as a no-op).
- **Shared backbone + CTC head** — the cross-subject representation that pooling trains.
- **Optional SBP head** — spike-band power enters through its own per-session adapter when present (fan_2023 simply skips it).
- **Masked-reconstruction head** — hides spans of the input and predicts the neural activity that was hidden, for label-free pretraining.

## The notebooks

- [`00_collections.ipynb`](00_collections.ipynb) — the collection lifecycle: declare what a collection reads and decodes, compose its chain, save it, load it back, build and validate its manifest.
- [`04_bit_pipeline.ipynb`](04_bit_pipeline.ipynb) — that collection through to a gradient step, one stage at a time.

## Module layout

- `data.py` — the seed recipe for the neural → text collection as a runnable `Chain`, the per-member identity maps (`SPEECH_TEXT_REGIONS` / `SPEECH_TEXT_PARTICIPANTS`), and `SessionBatchSampler`, which keeps a batch homogeneous in channel count.
- `model.py` — `BrainToTextCTC` and its parts (`SessionReadIn`, `Backbone`).
- `training.py` — Lightning modules (`CTCModule`, `MaskedPretrainModule`), greedy CTC decode, per-subject CER, and `train_ctc` / `pretrain_masked` / `evaluate_by_subject` helpers.
- `bit/` — the BIT transformer decoder: `CollectionManifest` → `BitConfig`, the read-in / patcher / trunk stack, train-only normalization, `BitDataBundle`, and the preflight validator.

Trials come from `BitDataBundle.from_collection`, the single path from a collection to tensors. The character vocabulary is `CharVocab.canonical()` from [`text.py`](../text.py), fixed rather than fitted, so `n_classes` is stable across collections.

Public API is re-exported from the package: `from neuralshed.models import speech_text_collection, BrainToTextCTC, train_ctc, ...`. `BrainToTextCTC.from_manifest` and `BitCrossEncoder.from_manifest` read the same manifest and consume the same `BitBatch`, so a difference between them is a modeling result rather than a pipeline difference.

## Running

Use the `tether` conda env. Training runs on **CPU**, because CTC is not implemented on Apple MPS and the model is small. The configuration constants at the top of each notebook (epochs, per-member `query`, model width) trade runtime for accuracy.

## Honest caveats

This is a **methods demo sized for a laptop**: a few sessions per subject and ~24 epochs, so absolute character error rates are high and single-run results carry ±~10% noise. The lessons are the *architecture* (per-session adapters + shared backbone), the *evaluation* (per-subject isolated-vs-pooled), and the *direction* of the effects, not the absolute numbers. The obvious next levers are more sessions, more epochs, GPU, and folding in the other collections (kinematics, voice) behind additional heads.