DATASET_METADATA = {
    "id": "willsey_2025",
    "title": (
        "A high-performance brain-computer interface for finger decoding and "
        "quadcopter game control in an individual with paralysis"
    ),
    "description": (
        "Closed-loop finger control by BrainGate2 participant T5 using two 96-channel "
        "Utah arrays in the hand knob of left precentral gyrus. Three finger groups "
        "(thumb, index-middle, ring-little) give four degrees of freedom, the thumb "
        "contributing flexion and abduction. The release holds seven recording days in "
        "March and April 2023 (58 closed-loop blocks, 2.4 hours, 3,172 trials of "
        "reach-and-hold finger target acquisition) plus two open-loop cued-movement "
        "blocks and one saved quadcopter flight path, at 50 ms spike-band power with "
        "the decoded finger posture and the cued target."
    ),
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.1jwstqk4f",
    "source_data_url": "https://doi.org/10.5061/dryad.1jwstqk4f",
    "status": "ml_ready",
    "macro_task": "human_motor",
    "decode_target": "movement",
    "study_class": "Willsey2025FingerDecoding",
    "bibtex": """
@article{willsey2025finger,
  title = {A high-performance brain-computer interface for finger decoding and quadcopter game control in an individual with paralysis},
  author = {Willsey, Matthew S. and Shah, Nishal P. and Avansino, Donald T. and Hahn, Nick V. and Jamiolkowski, Ryan M. and Kamdar, Foram B. and Hochberg, Leigh R. and Willett, Francis R. and Henderson, Jaimie M.},
  journal = {Nature Medicine},
  volume = {31},
  number = {1},
  pages = {96--104},
  year = {2025},
  month = jan,
  doi = {10.1038/s41591-024-03341-8},
  url = {https://doi.org/10.1038/s41591-024-03341-8},
}
""",
    "publication_doi": "10.1038/s41591-024-03341-8",
    "code_repo": "https://github.com/WillseyBCILab/BCI_Finger_Decoding_Quadcopter",
    "licence": "CC0-1.0",
    "notes": (
        "Spike-band power is stored in 50 ms bins (20 Hz, 192 channels) as 256 columns, "
        "of which 192-255 are exactly zero in every released block. Per the "
        "Supplementary Methods the chain is a 250 Hz digital high-pass, common average "
        "referencing, then the sum of the squared magnitude per window, so the quantity "
        "is a squared voltage that is extensive and rebinning must sum rather than "
        "average. All seven recording days come off the same front-end configuration, "
        "since the release's two later configurations cover the quadcopter sessions "
        "only. The first roughly 81 bins of each block are "
        "spaced 20 ms rather than 50 ms, and they are valid measurements over the "
        "narrower window rather than transients: rescaling them by 50/20 recovers the "
        "following bins (per-channel correlation 0.97). The authors drop the first 80 "
        "before normalizing because the narrower bins bias the per-channel statistics, "
        "and no trial starts before bin 192 in any case. The finger streams carry six "
        "columns but only four independent values, "
        "since index duplicates middle and ring duplicates little. Trials strictly "
        "alternate a randomly placed target with a return to the neutral posture, so "
        "49% of trials are return-to-rest. The released files do not record how many "
        "degrees of freedom the online decoder emitted, only how many the task cued: on "
        "2023-04-06 and 2023-04-13 the authors interleaved 4-DoF and 2-DoF decoders "
        "driving the same 2-DoF task, and that label survives only in their notebook "
        "(utils.BLOCK_LABELS). Out of scope: the 20230712 quadcopter block, which holds "
        "a saved flight path with no clock and no neural data, and the Decoders folders, "
        "which hold trained network weights and normalization constants rather than "
        "recordings."
    ),
    "subjects": [
        {
            "label": "T5",
            "participant_id": "braingate2/T5",  # datasets/participants.py
            # The paper states 69 at the time of the study, consistent with Willett 2021
            # putting T5 at 65 in 2019.
            "age": "P69Y",
            "time_since_implant": "P2395D",  # first session 2023-03-09, implant 2016-08-17
            "clinical_condition": [
                "Spinal cord injury (C4 AIS-C)",
                "Tetraplegia",
            ],
            "arrays": [
                # The release does not say which physical array supplies which half of
                # the channel range, and the channel order has not been checked against
                # the other T5 datasets (see docs/standardization.md).
                {
                    "name": "array_1",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": 0,
                    "channel_stop": 95,
                    "brain_area": "hand knob, motor cortex (left precentral gyrus)",
                },
                {
                    "name": "array_2",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": 96,
                    "channel_stop": 191,
                    "brain_area": "hand knob, motor cortex (left precentral gyrus)",
                },
            ],
            "brain_areas": [
                {
                    "raw_label": "hand knob, motor cortex (left precentral gyrus)",
                    "homba_id": "HOMBA_10169",
                    "uberon_id": "UBERON_0016634",
                    "brodmann": "6",  # Willett et al. 2020 (Cell) places T5's arrays in area 6, not 4
                },
            ],
        }
    ],
    "events": [
        {
            "name": "spike_band_power",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "spike_band_power",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 50.0,
            "rate_hz": 20.0,
            "dtype": "float32",
            "description": (
                "Spike-band power on 192 electrodes, stored under the source's own name "
                "hlfp (high-frequency LFP). The Supplementary Methods give the chain: a "
                "250 Hz digital high-pass, common average referencing, then the sum of "
                "the squared magnitude over each 50 ms window. Summing rather than "
                "averaging makes the quantity extensive, which the 20 ms bins that open "
                "every block confirm independently."
            ),
        },
        {
            "name": "finger_position",
            "category": "data",
            "event_class": "GraspSeries",
            "signal_type": "kinematics",
            "modality": "derived",
            "rate_hz": 20.0,
            "dtype": "float32",
            "description": (
                "Position of the four controlled degrees of freedom of the virtual hand "
                "(thumb flexion, thumb abduction, index-middle, ring-little) as a "
                "fraction of the range of motion, 0 fully flexed and 1 fully extended. "
                "In the closed-loop blocks this is the online decoder's output, which is "
                "what moved the hand the participant saw."
            ),
        },
        {
            "name": "finger_target",
            "category": "data",
            "event_class": "TargetGraspSeries",
            "signal_type": "kinematics",
            "modality": "behavior",
            "rate_hz": 20.0,
            "dtype": "float32",
            "description": (
                "Cued target posture over the same four degrees of freedom, in the same "
                "fraction-of-range units. The target spans plus or minus 0.1 around this "
                "centre, the 20% of range of motion quoted in the paper."
            ),
        },
        {
            "name": "target_cue",
            "category": "categorical",
            "event_class": "Stimulus",
            "description": (
                "Appearance of a new target posture on the screen, one per trial. "
                "Closed-loop trials alternate a randomly placed target with a return to "
                "the neutral posture; open-loop trials cue one of nine postures (rest, "
                "or one degree of freedom driven fully flexed or fully extended)."
            ),
        },
        {
            "name": "finger_movement",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "attempted",
            "description": (
                "Attempted finger movement toward the cued posture. In the closed-loop "
                "blocks it spans from the appearance of a target to the moment the "
                "participant had held the hand inside it for 500 ms; in the open-loop "
                "blocks it is the 2 s go phase that follows a 1 s preparation phase."
            ),
        },
    ],
}


def sessions() -> list[dict]:
    """One row per released recording day, derived from the downloaded block files."""
    from . import utils

    rows = []
    for label, blocks in utils.session_blocks().items():
        date = utils.session_date(label)
        tasks = sorted({block.task_label for block in blocks})
        rows.append(
            {
                "label": label,
                "subject_label": "T5",
                "date": date.isoformat(),
                "post_implant_day": (date - _implant_date()).days,
                "n_trials": sum(
                    len(utils.trial_windows(block))
                    for block in blocks
                    if block.kind == utils.KIND_CLOSED_LOOP
                ),
                "task": ", ".join(tasks),
            }
        )
    return rows


def _implant_date():
    from neuralshed.datasets import participants

    return participants.implant_date("braingate2/T5")


def get_metadata() -> dict:
    """Full dataset metadata: catalog (DATASET_METADATA) + per-session rows.

    Sessions need the data drive; omitted (empty list) when unavailable.
    """
    meta = DATASET_METADATA.copy()
    try:
        meta["sessions"] = sessions()
    except Exception:
        meta["sessions"] = []
    return meta


__all__ = ["DATASET_METADATA", "sessions", "get_metadata"]
