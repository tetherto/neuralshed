# Datasets Summary

37 datasets identified for foundation-model training, organized by **macro-task group**.

- **dataset** = simplified snake_case id (matches the `datasets/<name>/` folder convention).
- **status** = per-dataset pipeline state; see the [status legend](#status-legend) at the end.
- **Readiness gaps** (after each group's table) = which of `publication` / `annotated data`/ `code` are still missing toward 3/3; datasets not listed are 3/3.

**Counts:** total 37. Two pairs share a source and are counted once: `temmar_2021`/`link_dataset` (`001201`); `odoherty_2017`/`makin_2018` (`000129`, O'Doherty on DANDI covers both).


## Human Communication

Handwriting, speech (attempted, inner, intended), voice synthesis, typing. All BrainGate2 participants with Utah arrays in the precentral gyrus, plus T12's area 44 arrays in the inferior frontal gyrus. Instructed-delay tasks, binned features.

| dataset | brain_area | task | source | status |
|---|---|---|---|---|
| [card_2024](../src/neuralshed/datasets/card_2024/README.md) | precentral gyrus (6v, 4, 55b) | attempted speech | dryad | ml_ready |
| [fan_2023](../src/neuralshed/datasets/fan_2023/README.md) | precentral gyrus (hand knob) | attempted handwriting | dryad + dandi (000950) | ml_ready |
| [jude_2026_speech](../src/neuralshed/datasets/jude_2026_speech/README.md) | precentral gyrus (6v, 55b, 6d) | intended speech | dryad | ml_ready |
| [jude_2026_typing](../src/neuralshed/datasets/jude_2026_typing/README.md) | T17 precentral gyrus (6v, 55b, 6d), T18 bilateral | attempted typing | dryad | ml_ready |
| [kunz_2025](../src/neuralshed/datasets/kunz_2025/README.md) | precentral gyrus (6v, 4, 55b, 6d, PEF) and inferior frontal gyrus (44) | inner speech | dryad | ml_ready |
| [wairagkar_2025](../src/neuralshed/datasets/wairagkar_2025/README.md) | precentral gyrus (6v, 4, 55b) | attempted speech (voice synthesis) | dryad | ml_ready |
| [willet_2021](../src/neuralshed/datasets/willet_2021/README.md) | precentral gyrus (hand knob) | attempted handwriting | dryad | ml_ready |
| [willet_2023](../src/neuralshed/datasets/willet_2023/README.md) | precentral gyrus (6v) and inferior frontal gyrus (44) | attempted speech | dryad | ml_ready |

---

## Human Motor Control

Reach & grasp, cursor, gesture, finger, bimanual.

| dataset | brain_area | task | source | status |
|---|---|---|---|---|
| [karpowicz_2024_h1](../src/neuralshed/datasets/karpowicz_2024_h1/README.md) | hand knob (motor ctx) | reach & grasp | dandi (000954) | ml_ready |
| [singer_2025](../src/neuralshed/datasets/singer_2025/README.md) | speech motor cortex | cursor control + click | dryad | ml_ready |
| vargas_2025 | precentral gyrus | gesture encoding | dryad | not_started |
| [willsey_2025](../src/neuralshed/datasets/willsey_2025/README.md) | hand knob (motor ctx) | finger control | dryad | ml_ready |
| [karpowicz_2025](../src/neuralshed/datasets/karpowicz_2025/README.md) | motor cortex (left precentral gyrus) | cursor control (latent dynamics) | dryad | ml_ready* |
| [wilson_2025](../src/neuralshed/datasets/wilson_2025/README.md) | hand knob (motor ctx) | cursor BCI recalibration | dryad | ml_ready |
| [gusman_2025](../src/neuralshed/datasets/gusman_2025/README.md) | hand knob (motor ctx) | 2D drag-and-drop | dryad | ml_ready* |
| [deo_2024](../src/neuralshed/datasets/deo_2024/README.md) | motor cortex | bimanual movement | dryad | ml_ready |

*requires a curation round or specific fixes

---

## Non-Human Primates: Visual

Natural vision, shape/texture, natural images.

| dataset | brain_area | task | source | status |
|---|---|---|---|---|
| xiao_2024 | V1, V2, V4, PIT, CIT, AIT | natural vision | dandi (000628) | not_started |
| kim_2025 | V4 | shape/texture stimuli | dandi (001357) | not_started |
| papale_2025 | V1, V4, IT | natural images (THINGS) | gin | not_started |

**Readiness gaps:**
- kim_2025: code

---

## Non-Human Primates: Navigation

Mental navigation, prey capture, ball interception.

| dataset | brain_area | task | source | status |
|---|---|---|---|---|
| neupane_ppc_2024 | posterior parietal ctx | mental navigation | dandi (001275) | not_started |
| neupane_ec_2024 | entorhinal cortex | mental navigation | dandi (000897) | not_started |
| moore_2024 | M1 / somatosensory | prey capture | dandi (001062) | not_started |
| rajalingham_2025 | DMFC | ball interception | zenodo | not_started |

**Readiness gaps:**
- neupane_ppc_2024: annotated data, code
- neupane_ec_2024: annotated data, code
- moore_2024: annotated data, code

---

## Non-Human Primates: Motor Control

Reaching, center-out, finger, cursor.

| dataset | brain_area | task | source | status |
|---|---|---|---|---|
| churchland_2012 | M1, PMd | reaching | dandi (000070) | not_started |
| even_chen_2019 | PMd | cursor movement | dandi (000121) | study_wip |
| perich_2018 | M1, PMd | reaching | dandi (000688) | study_wip |
| athalye_2023 | PMd, M1 | reaching / obstacle avoidance | dandi (000404) | study_wip |
| m1a | M1 | center-out reaching | dandi (000941) | not_started |
| m1b | M1 | center-out reaching | dandi (001209) | not_started |
| temmar_2021 | M1 | finger movement | dandi (001201) | not_started |
| m2 | M1 | finger movement | dandi (000953) | not_started |
| area2_bump_2020 | somatosensory area 2 | reaching with perturbations | dandi (000127) | not_started |
| odoherty_2017 | M1, S1 | self-paced reaching | dandi (000129) | not_started |
| lanzarini_2025 | premotor, motor cortex | head restraint / free movement | osf | not_started |
| ma_2023 | motor cortex | reaching (adversarial decoding) | dryad | not_started |

**Readiness gaps:**
- even_chen_2019: annotated data, code

**Shared sources:** temmar_2021 ≡ link_dataset · odoherty_2017 also covers makin_2018.

---

## Non-Human Primates: Learning

Time-interval reproduction, rule learning.

| dataset | brain_area | task | source | status |
|---|---|---|---|---|
| dmfc_rsg_2019 | DMFC | time-interval reproduction | dandi (000130) | not_started |
| wojcik_2023 | PFC | rule learning | dryad | not_started |

**Readiness gaps:**
- wojcik_2023: code

---

## Status legend

| status | meaning |
|---|---|
| `not_started` | no work begun |
| `onboard_wip` | source download / dataset comprehension in progress |
| `onboard_ready` | comprehension artifacts complete: `utils.py`, `metadata.py`, `01_explore_source_data.ipynb`, `README.md` |
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
| pitt/HumanPitt | Withheld by the data owners | Hand-knob motor cortex, hemisphere not given (2×96 Utah) | karpowicz_2024_h1 | 7-DoF reach and grasp |
