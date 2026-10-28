# BIT: transformations and dimensions, step by step

A trace through the model on a heterogeneous mix example: **3 participants, 2 regions, 2 channel widths, one member with no band power.**

## The manifest that drives it

One row per **stream**, not per recording, so channel count and rate belong to the thing that actually has them:

| layout_key | group_key | event_type | kind | role | channels |
|---|---|---|---|---|---|
| `T15/card/s1/SpikeCounts` | `card/s1` | SpikeCounts | data | input | 256 |
| `T15/card/s1/BandPower` | `card/s1` | BandPower | data | input | 256 |
| `T15/card/s1/Sentence` | `card/s1` | Sentence | text | target | — |
| `T12/willet/s1/SpikeCounts` | `willet/s1` | SpikeCounts | data | input | 256 |
| `T12/willet/s1/BandPower` | `willet/s1` | BandPower | data | input | 256 |
| `T5/fanh2/s1/SpikeCounts` | `fanh2/s1` | SpikeCounts | data | input | 192 |

Five input layouts, three participants, **two** regions, three recordings. Those counts differing is the thing the manifest exists to record.

fan_2023 has no `BandPower` row at all. Absence is the representation: a missing stream can never be confused with a block of real zeros. And a `Sentence` has neither channels nor a rate, because those are properties of a dense array rather than of every declared stream.

## Symbols

| symbol | meaning | varies by |
|---|---|---|
| `B` | batch size | your choice |
| `C` | channels | **the recording** (256, 192, ...) |
| `T` | time bins | **the trial** (each sentence differs) |
| `D` | model width | config (128 here), fixed everywhere |
| `P` | patches | `ceil(T / patch_bins)` |
| `V` | classes | vocabulary (29) |

## Step by step

### 0. Extraction produces one trial

```
arrays  = {"SpikeCounts": (C, T), "BandPower": (C, T)}   # numpy, no batch dim
layouts = {"SpikeCounts": "T15/card/s1/SpikeCounts", ...} # which projection reads each
ids     = (L,)                                            # int64 character targets
```

### 1. `collate_bit` → a batch

Pads to the longest `T` in the batch. **All items come from one recording**, so `C` is constant within a batch. That is enforced, not assumed.

```
SpikeCounts   (B, C, T)
BandPower     (B, C, T)    or absent
input_lengths (B,)         valid bins before padding
group_key     str          the recording every item came from
layouts       dict         event class -> layout key
```

### 2. `BitReadIn` — where heterogeneity dies

```
SpikeCounts (B, C, T)
  ├─ normalize from buffers                 (B, C, T)
  ├─ transpose(1,2)                         (B, T, C)
  └─ Linear(C → D)                          (B, T, D)   ← C is consumed here
                                                 │
BandPower  (B, C', T) ─ Linear(C' → D) ─────────┤ sum   (only if the recording has one)
                                                 ▼
                        LayerNorm(D)            (B, T, D)
                             +
     participant emb (B,) → (B, D) → unsqueeze → (B, 1, D) broadcast over T
     region emb      (B,) → (B, D) → unsqueeze → (B, 1, D)
                             ▼
                                                (B, T, D)
```

Verified output for the three recordings:

```
card/s1     spike(2,256,300) + band(2,256,300)  →  (2, 300, 128)
willet/s1   spike(2,256,300) + band(2,256,300)  →  (2, 300, 128)
fanh2/s1    spike(2,192,437)                    →  (2, 437, 128)
```

**This is the funnel.** After the read-in, nothing downstream can tell 256 channels from 192, or two modalities from one. The only remaining difference is `T`, and that reflects *trial length*, not layout.

Note what happened to each kind of heterogeneity:

- **Channel count** → absorbed by a per-layout `Linear`.
- **Missing modality** → simply no second term in the sum. `fanh2` has no `band_power` entry in the `ModuleDict`, so nothing is added. Not a zero block the trunk must learn to ignore.
- **Participant / region** → became *additive vectors*, one row of an embedding table each, broadcast across all time bins. They inform the trunk without changing any shape.

### 3. Zero past each item's length

```
h = h * (arange(T) < input_lengths)          (B, T, D)
```

The read-in has a bias and LayerNorm, so padded bins emerge non-zero. Left alone, they contaminate the final partial patch.

### 4. `BitPatcher` — shared, one for all layouts

```
(B, T, D) → transpose → (B, D, T) → Conv1d(D→D, kernel=stride=5) → (B, D, P) → transpose → (B, P, D)
```

```
card    (2, 300, 128) → (2, 60, 128)      P = 300/5
fanh2   (2, 437, 128) → (2, 88, 128)      P = ceil(437/5)
```

### 5. `BitTrunk` — shared

```
(B, P, D) → (B, P, D)
```

Self-attention with a `(B, 1, P, P)` mask combining causal and padding.

### 6. CTC head — shared

```
Linear(D → V)         (B, P, D) → (B, P, 29)
log_softmax(-1)
```

```
card    (2, 60, 29)   lengths [60, 48]
willet  (2, 60, 29)   lengths [60, 36]
fanh2   (2, 88, 29)   lengths [88, 80]
```

### 7. CTC loss

```
logits.permute(1,0,2)   (P, B, V)          torch wants time-first
targets                 (sum of L,)         concatenated
input_lengths           out_lengths(...)    per item
```

## Where the parameters live

Verified against `n_parameters()`:

| | |
|---|---|
| `Linear(256→128)` × 4 (card, willet, each × 2 modalities) | 131,584 |
| `Linear(192→128)` × 1 (fanh2, spike only) | 24,704 |
| `LayerNorm(128)` | 256 |
| participant `Embedding(3, 128)` | 384 |
| region `Embedding(2, 128)` | 256 |
| **read-in total** | **157,184** ✓ |
| **shared trunk + patcher + head** | **483,003** |

The shared part is **3× larger** than the layout-specific part, and it stays the same size no matter how many recordings you add. Adding a fourth dataset at 384 channels adds one `Linear(384→128)`, about 49k parameters, and touches nothing else.

## The one constraint this imposes

A batch cannot mix recordings. `SpikeCounts` for card is `(B, 256, T)` and for fanh2 is `(B, 192, T)`, they do not even stack into one tensor. That is why `SessionBatchSampler(key_attr="group_key")` groups trials by recording, why `collate_bit` raises on a mixed batch, and why `forward` takes one `layouts` mapping rather than a per-row routing table.

Grouping by recording is safe but conservative: when a coarse layout key deliberately shares one projection across several recordings, those trials could batch together and this still keeps them apart.

Batches alternate between recordings *across* the epoch, so the shared trunk sees all three interleaved. It just sees them one recording at a time.

## Reproducing this trace

```python
import torch
from neuralshed.models.bit import BitConfig, BitCrossEncoder

cfg = BitConfig(
    n_classes=29,
    layout_keys=("T15/card/s1/SpikeCounts", "T15/card/s1/BandPower", "T5/fanh2/s1/SpikeCounts"),
    layout_channels={
        "T15/card/s1/SpikeCounts": 256,
        "T15/card/s1/BandPower":   256,
        "T5/fanh2/s1/SpikeCounts": 192,
    },
    n_participants=3, n_regions=2, frequency=50.0, patch_ms=100.0,
    dim=128, depth=2, heads=4, dim_head=32, dropout=0.0, input_dropout=0.0,
)
model = BitCrossEncoder(cfg).eval()
print(model.n_parameters())

batch = {
    "arrays": {"SpikeCounts": torch.randn(2, 192, 437)},
    "layouts": {"SpikeCounts": "T5/fanh2/s1/SpikeCounts"},
    "input_lengths": torch.tensor([437, 400]),
    "participant_idx": torch.tensor([2, 2]),
    "region_idx": torch.tensor([0, 0]),
}
with torch.no_grad():
    print(model(batch).shape)          # (2, 88, 29)
```

See [04_bit_pipeline.ipynb](../04_bit_pipeline.ipynb) for the same path run end to end on real data.
