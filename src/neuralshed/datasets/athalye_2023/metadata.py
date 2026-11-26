_BIBTEX = """
@article{athalye2023invariant,
  title = {Invariant neural dynamics drive commands to control different movements},
  author = {Athalye, Vivek R. and Khanna, Preeya and Gowda, Suraj and Orsborn, Amy L. and Costa, Rui M. and Carmena, Jose M.},
  journal = {Current Biology},
  volume = {33},
  number = {14},
  pages = {2962--2976.e15},
  year = {2023},
  doi = {10.1016/j.cub.2023.06.027},
  url = {https://doi.org/10.1016/j.cub.2023.06.027},
}
"""

# Each subject has one 16 x 8 microwire array per hemisphere spanning the upper-arm area of
# M1 and posterior PMd. The release has no electrode table: spike_counts columns are the
# day's decoder units, not channels, so no channel range is declared and the M1 versus
# PMd split of a unit cannot be recovered.
_REGION = "upper-arm area of M1 and posterior PMd"
_BRAIN_AREAS = [
    {
        "raw_label": _REGION,
        # frontal motor cortex, the deepest HOMBA term covering both M1 and PMd
        "homba_id": "HOMBA_AA30009",
        # frontal cortex, the nearest ancestor with an UBERON link
        "uberon_id": "UBERON_0001870",
        "brodmann": "4,6",  # each array spans M1 (area 4) and PMd (area 6)
    },
]


def _arrays() -> list[dict]:
    return [
        {
            "name": "array_1",
            "hemisphere": hemisphere,
            "device_type": "microwire_array",
            "n_channels": 128,  # 16 x 8 wires, per the paper
            "channel_start": None,
            "channel_stop": None,
            "brain_area": _REGION,
        }
        for hemisphere in ("left", "right")
    ]


DATASET_METADATA = {
    "id": "athalye_2023",
    "title": "Invariant neural dynamics drive commands to control different movements",
    "description": (
        "Two rhesus macaques (monkeys G and J) move a 2D cursor under brain control with a "
        "decoder fixed within each session, on a center-out task and an obstacle-avoidance "
        "task. Chronic microwire arrays in M1 and PMd of both hemispheres. The release "
        "holds the binned spike counts of the decoder's own units, the cursor and the "
        "decoder state, for 9 sessions of G (60 Hz, Kalman filter updated at 10 Hz) and 4 "
        "of J (200 Hz, point-process filter), 3,624 rewarded trials in all."
    ),
    "source_data_platform": "dandi",
    "source_id": "DANDI:000404",
    "source_data_url": "https://dandiarchive.org/dandiset/000404",
    "status": "study_wip",
    "macro_task": "nhp_motor",
    "decode_target": "movement",
    "study_class": "Athalye2023CursorBMI",
    "bibtex": _BIBTEX,
    "publication_doi": "10.1016/j.cub.2023.06.027",
    "code_repo": "https://github.com/pkhanna104/bmi_dynamics_code",
    "licence": "CC-BY-4.0",
    "notes": (
        "All data are closed-loop brain control with a fixed decoder, and the cursor is "
        "the decoder's position state. No units, electrodes or spike times: spike_counts "
        "holds the day's decoder units (35 to 151 for G, 20 or 21 for J), not matched "
        "across sessions and without area or hemisphere labels. Monkey J's positions are "
        "in m and are scaled to cm on read. Each file concatenates blocks (te_num) back to "
        "back, center-out first, so one timeline is one block. Four monkey J blocks end in "
        "zero-filled padding (56.9 min in all), cut from the emitted streams. G's "
        "target_state is all zeros and J's target_state and obs_details are misaligned by "
        "the writer, so no target stream is emitted. Only rewarded trials are labelled, "
        "go cue to reward. Decoder matrices and task parameters were not deposited. The "
        "NWB metadata of J says Kalman filter and P12Y, both wrong (point-process filter, "
        "10 years)."
    ),
    "subjects": [
        {
            "label": "monk_g",
            "participant_id": "carmena/G",
            # Born 2007-05-13 (colony record in the authors' writer code), recorded
            # 2016-03-02 to 2016-03-19, so 8 years 9 to 10 months. The NWB says P8Y and
            # the paper says 7 years.
            "age": "P8Y",
            "time_since_implant": None,
            "clinical_condition": [],
            "arrays": _arrays(),
            "brain_areas": _BRAIN_AREAS,
        },
        {
            "label": "monk_j",
            "participant_id": "carmena/J",
            # Born 2003-05-25, recorded 2013-08-24 to 2013-08-31, so 10 years 3 months,
            # matching the paper. The NWB's P12Y is copied from monkey G's comment.
            "age": "P10Y",
            "time_since_implant": None,
            "clinical_condition": [],
            "arrays": _arrays(),
            "brain_areas": _BRAIN_AREAS,
        },
    ],
    "events": [
        {
            "name": "spike_counts",
            "category": "data",
            "event_class": "SpikeCounts",
            "signal_type": "sorted_spikes",
            "modality": "intracortical_microelectrode",
            "bin_size_ms": None,  # 16.7 ms for monkey G, 5 ms for monkey J
            "rate_hz": None,
            "dtype": "uint32/uint8",
            "description": (
                "Spike counts of the online-sorted units feeding the decoder, one column "
                "per unit. Monkey G: counts per 60 Hz task frame, 35 to 151 units. Monkey "
                "J: 0 or 1 per 5 ms sample at 200 Hz, 20 or 21 units."
            ),
        },
        {
            "name": "cursor",
            "category": "data",
            "event_class": "CursorPositionSeries",
            "signal_type": "kinematics",
            "modality": "derived",
            "rate_hz": None,
            "dtype": "float64",
            "description": (
                "Cursor x and y in cm, equal to the decoder's position state. Monkey G's "
                "center target is at the origin, monkey J's at about (4.1, 14.0) cm. G's "
                "cursor moves only on decoder updates (10 Hz)."
            ),
        },
        {
            "name": "decoder_state",
            "category": "data",
            "event_class": "DecodedCursorVelocitySeries",
            "signal_type": "kinematics",
            "modality": "derived",
            "rate_hz": None,
            "dtype": "float64",
            "description": (
                "Velocity part of the decoder state, in cm/s (monkey J scaled from m/s). "
                "The source stream also holds the position, equal to the cursor, and a "
                "constant offset."
            ),
        },
        {
            "name": "trials",
            "category": "categorical",
            "event_class": "Action",
            "task_condition": "other",
            "description": (
                "Rewarded trial from go cue (peripheral target onset) to reward, under "
                "brain control. Monkey G obstacle trials carry the obstacle position and "
                "size."
            ),
        },
        {
            "name": "intertrial",
            "category": "categorical",
            "event_class": "Intertrial",
            "description": (
                "Time between rewarded trials: holds, cursor return and failed trials, "
                "which the release does not label."
            ),
        },
    ],
}


def sessions() -> list[dict]:
    """One row per session file, from the NWB files."""
    from . import utils

    rows = []
    for relpath in utils.list_assets(utils.default_data_dir()):
        refs = utils.blocks(relpath)
        rows.append(
            {
                "label": utils.session_label(relpath),
                "subject_label": utils.subject_of(relpath),
                "date": utils.session_date(refs[0].source),
                "post_implant_day": None,  # implant dates are unpublished
                "n_trials": sum(len(utils.trials(ref)) for ref in refs),
                "task": ", ".join(sorted({ref.task for ref in refs})),
                "task_condition": "other",
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


__all__ = ["DATASET_METADATA", "get_metadata", "sessions"]
