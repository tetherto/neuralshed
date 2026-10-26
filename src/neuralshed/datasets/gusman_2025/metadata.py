DATASET_METADATA = {
    "id": "gusman_2025",
    "title": "Multi-gesture drag-and-drop decoding in a 2D iBCI control task",
    "description": (
        "Sustained hand gestures held during 2D cursor control by BrainGate2 "
        "participants T11 and T5, each with two 96-channel Utah arrays in the hand knob "
        "of the left precentral gyrus. An open-loop Gesture Hero task measures how a "
        "gesture representation decays over 1, 2 and 4 second holds, and a closed-loop "
        "drag-and-drop task uses a latch decoder to keep a gesture active while the "
        "cursor moves. The release holds five sessions on a continuous 20 ms grid with "
        "seven feature types per electrode, including five local field potential bands."
    ),
    "source_data_platform": "dryad",
    "source_id": "doi:10.5061/dryad.98sf7m0v1",
    "source_data_url": "https://doi.org/10.5061/dryad.98sf7m0v1",
    "status": "ml_ready",
    "macro_task": "human_motor",
    "decode_target": "other",
    "study_class": "Gusman2025DragAndDrop",
    "bibtex": """
@article{gusman2025draganddrop,
  title = {Multi-gesture drag-and-drop decoding in a 2D iBCI control task},
  author = {Gusman, Jacob T. and Hosman, Tommy and Crawford, Jacob and Singer-Clark, Tyler and Kapitonava, Anastasia and Kelemen, Jessica N. and Hahn, Nick and Henderson, Jaimie M. and Hochberg, Leigh R. and Simeral, John D. and Vargas-Irwin, Carlos E.},
  journal = {Journal of Neural Engineering},
  volume = {22},
  number = {2},
  pages = {026054},
  year = {2025},
  doi = {10.1088/1741-2552/adb180},
  url = {https://doi.org/10.1088/1741-2552/adb180},
}
""",
    "publication_doi": "10.1088/1741-2552/adb180",
    "code_repo": "https://github.com/jgusman/drag-and-drop",
    "licence": "CC0-1.0",
    "notes": (
        "Three MATLAB v7.3 files, so h5py reads them and scipy.io.loadmat does not. "
        "Blocks are concatenated onto one continuous 20 ms grid per session and the task "
        "tables index into it with bin ranges, so this is a recording rather than a bag "
        "of clips. Seven features per electrode give 1,344 columns, and unlike "
        "vargas_2025 every column is attributable: featInfo.featOffset gives the "
        "boundaries and featInfo.featNames the types. Five of the seven are short-time "
        "Fourier bands, the first genuine local field potential features in the catalog. "
        "All features are z-scored against a rolling 3 minute window before release, so "
        "they cannot be converted through a bin rate. The labels, cuedGesture, "
        "cuedDirection, trialAttemptType and decodedState fields are MATLAB categorical "
        "objects stored as (1, 6) uint32 MCOS references; utils.read_categorical "
        "resolves them out of the file's own subsystem group, needing only h5py. "
        "Closed-loop status is per stage and per decoder rather than a property of a "
        "block, since the kinematic and gesture decoders close independently."
    ),
    "subjects": [
        {
            "label": "T11",
            "participant_id": "braingate2/T11",  # datasets/participants.py
            # 39 here against 36 in vargas_2025, whose sessions are earlier.
            "age": "P39Y",
            "time_since_implant": None,  # no implant date is published for T11
            "clinical_condition": [
                "Spinal cord injury (C4 AIS-B)",
                "Tetraplegia",
            ],
            "arrays": [
                {
                    "name": "lateral_array",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": 0,
                    "channel_stop": 95,
                    "brain_area": "hand knob, motor cortex (left precentral gyrus)",
                },
                {
                    "name": "medial_array",
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
                    "homba_id": "HOMBA_10165",
                    "uberon_id": "UBERON_0001384",
                    "brodmann": "4",
                },
            ],
        },
        {
            "label": "T5",
            "participant_id": "braingate2/T5",
            # 70 here against 68 in vargas_2025 and 69 in willsey_2025.
            "age": "P70Y",
            "time_since_implant": None,  # the source publishes no trial day for T5
            "clinical_condition": [
                "Spinal cord injury (C4 AIS-C)",
                "Tetraplegia",
            ],
            "arrays": [
                {
                    "name": "lateral_array",
                    "hemisphere": "left",
                    "device_type": "utah_array",
                    "n_channels": 96,
                    "channel_start": 0,
                    "channel_stop": 95,
                    "brain_area": "hand knob, motor cortex (left precentral gyrus)",
                },
                {
                    "name": "medial_array",
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
        },
    ],
    "events": [
        {
            "name": "threshold_crossings",
            "category": "data",
            "event_class": "SpikeCounts",
            "signal_type": "threshold_crossings",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Non-causal threshold crossings on 192 electrodes, source name ncTX, "
                "columns 0 to 191 of the feature matrix. Z-scored against a rolling 3 "
                "minute window before release."
            ),
        },
        {
            "name": "spike_band_power",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "spike_band_power",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Spike-band power on the same 192 electrodes, 250 Hz to 5 kHz, source "
                "name spikePower, columns 192 to 383. Z-scored against a rolling 3 "
                "minute window."
            ),
        },
        {
            "name": "lfp_0_11",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "lfp",
            "modality": "lfp",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Average fast-Fourier power in 0-11 Hz on the same 192 electrodes, "
                "columns 384 to 575. The paper's Methods define the band as 0-11 Hz "
                "and the file's own featNames label it 0-8, which is the realized "
                "edge on the 3.90625 Hz grid of the 256-sample FFT. Z-scored against a "
                "rolling 3 minute window."
            ),
        },
        {
            "name": "lfp_12_19",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "lfp",
            "modality": "lfp",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Average fast-Fourier power in 12-19 Hz on the same 192 electrodes, "
                "columns 576 to 767. The paper's Methods define the band as 12-19 Hz "
                "and the file's own featNames label it 12-16, which is the realized "
                "edge on the 3.90625 Hz grid of the 256-sample FFT. Z-scored against a "
                "rolling 3 minute window."
            ),
        },
        {
            "name": "lfp_20_38",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "lfp",
            "modality": "lfp",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Average fast-Fourier power in 20-38 Hz on the same 192 electrodes, "
                "columns 768 to 959. The paper's Methods define the band as 20-38 Hz "
                "and the file's own featNames label it 20-31, which is the realized "
                "edge on the 3.90625 Hz grid of the 256-sample FFT. Z-scored against a "
                "rolling 3 minute window."
            ),
        },
        {
            "name": "lfp_39_128",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "lfp",
            "modality": "lfp",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Average fast-Fourier power in 39-128 Hz on the same 192 electrodes, "
                "columns 960 to 1151. The paper's Methods define the band as 39-128 Hz "
                "and the file's own featNames label it 39-125, which is the realized "
                "edge on the 3.90625 Hz grid of the 256-sample FFT. Z-scored against a "
                "rolling 3 minute window."
            ),
        },
        {
            "name": "lfp_129_250",
            "category": "data",
            "event_class": "BandPower",
            "signal_type": "lfp",
            "modality": "lfp",
            "bin_size_ms": 20.0,
            "rate_hz": 50.0,
            "dtype": "float64",
            "description": (
                "Average fast-Fourier power in 129-250 Hz on the same 192 electrodes, "
                "columns 1152 to 1343. The paper's Methods define the band as 129-250 Hz "
                "and the file's own featNames label it 129-250, which is the realized "
                "edge on the 3.90625 Hz grid of the 256-sample FFT. Z-scored against a "
                "rolling 3 minute window."
            ),
        },
        {
            "name": "gesture_cue",
            "category": "categorical",
            "event_class": "Stimulus",
            "description": (
                "The cued gesture, and in the drag-and-drop task the cued direction and "
                "trial type alongside it. T11 was cued seven gestures in Gesture Hero "
                "and three in drag-and-drop, T5 three. A no_action state exists in every "
                "session's category list as the decoder's fourth class."
            ),
        },
        {
            "name": "gesture_attempt",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "attempted",
            "description": (
                "The attempted gesture held for its cued duration. In Gesture Hero one "
                "row per trial with a 1, 2 or 4 second hold; in drag-and-drop one row "
                "per trial stage, with the stage naming the phase and flags recording "
                "whether each decoder was closed for it."
            ),
        },
        {
            "name": "online_gesture_decode",
            "category": "data",
            "event_class": "BehavioralSeries",
            "signal_type": "other",
            "modality": "derived",
            "rate_hz": 50.0,
            "dtype": "uint8",
            "description": (
                "The latch decoder's own per-bin gesture output over the same four "
                "states, drag-and-drop sessions only, alongside its per-state "
                "log-likelihoods and threshold."
            ),
        },
    ],
}


def sessions() -> list[dict]:
    """One row per recording session, derived from the downloaded release."""
    from . import utils

    rows = []
    for ref in utils.list_sessions():
        table = utils.epochs(ref)
        rows.append(
            {
                "label": ref.label,
                "subject_label": ref.participant,
                "date": None,  # the source publishes neither dates nor trial days
                "post_implant_day": None,
                "n_trials": (
                    int(table["trial"].nunique())
                    if ref.task == utils.TASK_DRAG_DROP
                    else len(table)
                ),
                "task": (
                    f"{'Gesture Hero (open loop)' if ref.task == utils.TASK_GESTURE_HERO else 'Drag-and-drop (closed loop)'}"
                    f", {len(utils.gestures(ref))} gestures, {len(utils.blocks(ref))} blocks"
                ),
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
