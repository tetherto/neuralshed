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
        "of which 192-255 are exactly zero in every released block. On "
        "t5.2023.04.13 channels 0-95 and 96-191 are exchanged relative to every other "
        "day and utils.py swaps them back. No source states it, so the exchange is "
        "inferred from the data. Per the "
        "Supplementary Methods the chain is the amplifier's 0.3 Hz-7.5 kHz analog "
        "filter at 30 kHz, common average referencing, a 250 Hz digital high-pass, "
        "then the sum of the squared magnitude per window. The 50 ms "
        "values sit on the scale of datasets that average, so the sum was divided by a "
        "fixed count: the quantity is a mean squared voltage and rebinning averages. "
        "All seven recording days come off the same front-end configuration, "
        "since the release's two later configurations cover the quadcopter sessions "
        "only. The first roughly 81 bins of each block are "
        "spaced 20 ms rather than 50 ms, and they are valid measurements over the "
        "narrower window rather than transients: rescaling them by 50/20 recovers the "
        "following bins (per-channel correlation 0.97). The authors drop the first 80 "
        "before normalizing because the narrower bins bias the per-channel statistics, "
        "and the earliest trial starts at bin 121 (open loop) or 192 (closed loop) in "
        "any case. The finger streams carry six "
        "columns but only four independent values, "
        "since index duplicates middle and ring duplicates little. Trials strictly "
        "alternate a randomly placed target with a return to the neutral posture, so "
        "49% of trials are return-to-rest. The task stream shows how many degrees of "
        "freedom the task cued, not how many the online decoder emitted: on "
        "2023-04-06 and 2023-04-13 the authors interleaved 4-DoF and 2-DoF decoders "
        "driving the same 2-DoF task, so for the 2-DoF task blocks that count comes "
        "from their notebook and Decoders folders (utils.DECODER_DOF). Out of scope: "
        "the 20230712 quadcopter block, which holds a saved flight path with no clock "
        "and no neural data, and the Decoders folders, which hold trained network "
        "weights and normalization constants rather than recordings."
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
                # the channel range. The data match the other T5 datasets' order, with
                # t5.2023.04.13's two halves swapped back by utils (see
                # docs/channel_mapping/t5.md).
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
                "hlfp (high-frequency LFP). The Supplementary Methods give the chain: the "
                "amplifier's 0.3 Hz-7.5 kHz analog filter at 30 kHz, common average "
                "referencing, a 250 Hz digital high-pass, then the sum of the squared "
                "magnitude over each 50 ms window, so the band is 250 Hz to 7.5 kHz. The "
                "values sit on the scale of a mean square, so the sum was divided by a "
                "fixed count, which the 20 ms bins that open every block confirm by "
                "sitting at 20/50 of the steady level."
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
                "fraction of the range of motion, 1 fully flexed (abducted for thumb "
                "abduction) and 0 fully extended (adducted), following the paper's "
                "Methods; the authors' RunAnalysis.m names 0 flexed. In the closed-loop "
                "blocks this is the online decoder's output, which is what moved the "
                "hand the participant saw."
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
                "Cued posture of an open-loop go phase, over the 2 s in which the "
                "virtual fingers move to it: one of nine postures (rest, or one degree "
                "of freedom driven to either end of its range) or the return from "
                "one. Closed-loop trials have no Stimulus row, since the target appears "
                "when the trial starts and the Action carries it."
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
                "decoded hand had held every cued degree of freedom inside it for "
                "500 ms, or to the 10 s timeout, and success says which; in the "
                "open-loop blocks it is the 2 s go phase that follows 1 s with the "
                "fingers still."
            ),
        },
        {
            "name": "rest",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "na",
            "description": (
                "An open-loop go phase that starts and ends at the neutral posture "
                "(action_kind 'rest', code 0)."
            ),
        },
    ],
}


def sessions() -> list[dict]:
    """One row per released recording day, derived from the downloaded block files."""
    from neuralshed.datasets import participants

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
                "post_implant_day": participants.post_implant_day(
                    "braingate2/T5", date.isoformat()
                ),
                "n_trials": sum(
                    (
                        len(utils.trial_windows(block))
                        if block.kind == utils.KIND_CLOSED_LOOP
                        else len(utils.cue_windows(block))
                    )
                    for block in blocks
                    if block.kind != utils.KIND_QUADCOPTER
                ),
                "task": ", ".join(tasks),
            }
        )
    return rows


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
