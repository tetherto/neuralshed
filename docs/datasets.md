# Datasets Summary

16 human datasets identified for foundation-model training, organized by **macro-task group**.

- **dataset** = simplified snake_case id (matches the `datasets/<name>/` folder convention).
- **status** = per-dataset pipeline state; see the [status legend](#status-legend) at the end.
- **Readiness gaps** (after each group's table) = which of `publication` / `annotated data`/ `code` are still missing toward 3/3; datasets not listed are 3/3.


## Human Communication

Handwriting, speech (attempted / imagined / intended), voice synthesis, typing. All human BrainGate2 intracortical, left precentral gyrus, instructed-delay, binned features.

| dataset | brain_area | task | source | status |
|---|---|---|---|---|
| [card_2024](../src/neuralshed/datasets/card_2024/README.md) | speech motor cortex | attempted speech | dryad | ml_ready |
| [fan_2023](../src/neuralshed/datasets/fan_2023/README.md) | hand knob (motor ctx) | attempted handwriting | dryad + dandi (000950) | ml_ready |
| [jude_2026_speech](../src/neuralshed/datasets/jude_2026_speech/README.md) | precentral gyrus (6d/6v/55b) | intended speech | dryad | ml_ready |
| [jude_2026_typing](../src/neuralshed/datasets/jude_2026_typing/README.md) | precentral gyrus (T17 6d/6v/55b, T18 bilateral 6d) | attempted typing | dryad | ml_ready |
| [kunz_2025](../src/neuralshed/datasets/kunz_2025/README.md) | precentral gyrus (6v, 4, 55b, 6d, PEF) and inferior frontal gyrus (44) | inner speech | dryad | ml_ready |
| [wairagkar_2025](../src/neuralshed/datasets/wairagkar_2025/README.md) | speech motor cortex | attempted speech | dryad | ml_ready |
| [willet_2021](../src/neuralshed/datasets/willet_2021/README.md) | motor cortex | handwriting | dryad | ml_ready |
| [willet_2023](../src/neuralshed/datasets/willet_2023/README.md) | ventral precentral gyrus (6v) and inferior frontal gyrus (44) | attempted speech | dryad | ml_ready |

---

## Human Motor Control

Reach & grasp, cursor, gesture, finger, bimanual.

| dataset | brain_area | task | source | status |
|---|---|---|---|---|
| [fan_h1](../src/neuralshed/datasets/fan_h1/README.md) | hand knob (motor ctx) | reach & grasp | dandi (000954) | ml_ready |
| [singer_2025](../src/neuralshed/datasets/singer_2025/README.md) | speech motor cortex | cursor control + click | dryad | ml_ready* |
| vargas_2025 | precentral gyrus | gesture encoding | dryad | not_started |
| [willsey_2025](../src/neuralshed/datasets/willsey_2025/README.md) | hand knob (motor ctx) | finger control | dryad | ml_ready* |
| [karpowicz_2025](../src/neuralshed/datasets/karpowicz_2025/README.md) | motor cortex (left precentral gyrus) | cursor control (latent dynamics) | dryad | ml_ready* |
| [wilson_2025](../src/neuralshed/datasets/wilson_2025/README.md) | hand knob (motor ctx) | cursor BCI recalibration | dryad | ml_ready* |
| [gusman_2025](../src/neuralshed/datasets/gusman_2025/README.md) | hand knob (motor ctx) | 2D drag-and-drop | dryad | ml_ready* |
| [deo_2024](../src/neuralshed/datasets/deo_2024/README.md) | motor cortex | bimanual movement | dryad | ml_ready* |

*requires a curation round or specific fixes

**Readiness gaps:**
- fan_h1: code

---

## Status legend

| status | meaning |
|---|---|
| `not_started` | no work begun |
| `onboard_wip` | source download / dataset comprehension in progress |
| `onboard_ready` | comprehension artifacts complete: `utils.py`, `01_explore_source_data.ipynb`, `README.md` |
| `study_wip` | neuroai Events/Study/Extractors implementation in progress |
| `study_ready` | Events/Study/Extractors implemented and sample-validated |
| `ml_wip` | utilization example (`03_model_example.ipynb`) in progress |
| `ml_ready` | utilization example complete; fully prepared and indexed in the metadata database |

---

## Human Participants

The 8 people recorded across the ingested human datasets. A participant id is shared by every dataset that recorded the same person (registry in [participants.py](../src/neuralshed/datasets/participants.py)).

| participant | condition | implant location | datasets | tasks |
|---|---|---|---|---|
| braingate2/T5 | Spinal cord injury (C4 AIS-C), tetraplegia | Left precentral gyrus, hand knob (2×96 Utah) | willet_2021, fan_2023, wilson_2025, deo_2024, willsey_2025, gusman_2025 | Handwriting (letters, lines, sentences); 2D cursor control; two-cursor control with one or both hands; finger control; hand gestures during cursor control |
| braingate2/T11 | Spinal cord injury (C4 AIS-B), tetraplegia | Left precentral gyrus, hand knob (2×96 Utah) | karpowicz_2025, gusman_2025 | 2D cursor control; hand gestures, drag-and-drop |
| braingate2/T12 | Bulbar-onset ALS, severe dysarthria | Left area 6v (2) + area 44, inferior frontal gyrus (2) (4×64 Utah) | willet_2023, kunz_2025 | Attempted speech of sentences; inner speech and other verbal behaviors; sequence recall |
| braingate2/T15 | ALS, severe dysarthria, tetraparesis | Left precentral gyrus: 6v ventral, 6v dorsal, area 4, 55b (4×64 Utah) | card_2024, wairagkar_2025, singer_2025, kunz_2025 | Attempted speech of sentences; voice synthesis; cursor control and click; inner speech, counting |
| braingate2/T16 | Pontine stroke, tetraplegia, dysarthria | Left precentral gyrus: 55b/PEF border, 6v, 6d hand knob ×2 (4×64 Utah) | kunz_2025 | Inner speech and other verbal behaviors; counting; sequence recall |
| braingate2/T17 | ALS, anarthria, locked-in, ventilator-dependent | Left precentral gyrus: 6v, 55b, 6d, 2 arrays each (6×64 Utah) | jude_2026_speech, jude_2026_typing, kunz_2025 | Attempted speech (orofacial movements, phonemes, words, sentences); finger movements, two-handed typing; inner speech |
| braingate2/T18 | Spinal cord injury (C4 ASIA A), tetraplegia | Area 6d in both hemispheres: 4 left, 2 right (6×64 Utah) | jude_2026_typing | Finger movements; two-handed typing; handwriting |
| pitt/HumanPitt | Withheld by the data owners | Hand-knob motor cortex, hemisphere not given (2×96 Utah) | fan_h1 | 7-DoF reach and grasp |
