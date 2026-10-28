"""Controlled vocabularies for the metadata database.

Single home for the allowed values used across the schema and the per-dataset
``metadata.py`` declarations. Enforced **softly** — a warning at populate time, not a
DB ``CHECK``/``ENUM`` — so the catalog can absorb a new dataset that introduces an
unforeseen value without a migration. Promote to hard Postgres ``ENUM``/``CHECK`` once
the vocabularies stabilize.
"""

# Where the source data is hosted (``source_id`` carries the accession/DOI).
SOURCE_DATA_PLATFORMS = {"dandi", "dryad", "zenodo", "osf", "gin", "other"}

# Onboarding / pipeline lifecycle (see docs/onboarding_a_new_dataset.md + datasets.md).
# Three phases (onboard -> study -> ml), each with a wip / ready milestone.
STATUSES = {
    "not_started",
    "onboard_wip",
    "onboard_ready",
    "study_wip",
    "study_ready",
    "ml_wip",
    "ml_ready",
}

# Macro-task families (mirror the groups in datasets.md).
MACRO_TASKS = {
    "human_communication",
    "human_motor",
}

# What a decoder trained on a dataset (or collection) outputs. Names the *output* modality,
# not the task: attempted-speech-to-text and handwriting-to-text are both "text"; only
# acoustic/voice synthesis is "voice". This single fact gates poolability — two datasets can
# share one decoder head only if their decode_target matches. Soft-checked; extend as needed.
DECODE_TARGETS = {
    "text",  # symbols: Sentence / Word / Character / Phoneme / Keystroke
    "voice",  # acoustic signal: AcousticFeatures / synthesized audio
    "movement",  # kinematics: Position / Orientation / Grasp / velocity series
    "other",
}

# Dominant hand of the participant, as reported by the source.
HANDEDNESS = {"right", "left", "ambidextrous", "unknown", "na", "other"}

# Hemisphere of an implanted array (``None`` when the source does not state it).
HEMISPHERES = {"left", "right"}

# Recording-device family — count-agnostic; ``Array.n_channels`` carries the count.
DEVICE_TYPES = {
    "utah_array",
    "silicon_probe",
    "ecog_grid",
    "microwire_array",
    "seeg_electrode",
    "other",
}

# Broad recording modality / signal origin.
RECORDING_MODALITIES = {
    "intracortical_microelectrode",
    "ecog",
    "eeg",
    "lfp",
    "calcium_imaging",
    "emg",
    "audio",
    "behavior",
    "derived",
    "other",
}

# Canonical signal type for data events (the source's own label rides along in ``EventType.name``).
SIGNAL_TYPES = {
    "sorted_spikes",
    "threshold_crossings",
    "spike_band_power",
    "lfp",
    "calcium_fluorescence",
    "emg",
    "ecog",
    "kinematics",
    "audio",
    "acoustic_features",
    "synthesized_audio",
    "other",
}

# Behavioral mode of a trial / event. The train/eval *split* is NOT here — it is each
# paper's own benchmark label, carried free-form as ``original_split``.
TASK_CONDITIONS = {
    "attempted",
    "mimed",
    "imagined",
    "listening",
    "read",
    "typed",
    "na",
    "other",
}

# Event family — the neuroai ``Event`` base branches (every signal IS an event).
EVENT_CATEGORIES = {
    "data",  # BaseDataEvent: dense/continuous streams (neural, kinematics, audio, …)
    "text",  # BaseText: Sentence / Word / Phoneme / Keystroke
    "categorical",  # CategoricalEvent: Action / Stimulus / Background / … (markers, cues, periods)
}

# Concrete neuroai ``Event`` subclasses an ``EventType`` may map to (the forward bridge to
# a future ``populate_from_study()``). Soft-checked; extend as new classes are emitted.
EVENT_CLASSES = {
    # data (BaseDataEvent + custom dense-array events)
    "SpikeCounts",
    "BandPower",
    "Spikes",
    "BehavioralSeries",
    "PositionSeries",
    "OrientationSeries",
    "GraspSeries",
    "TargetGraspSeries",
    "LinearVelocitySeries",
    "AngularVelocitySeries",
    "GraspVelocitySeries",
    "AccelerationSeries",
    "CursorPositionSeries",
    "TargetPositionSeries",
    "CursorVelocitySeries",
    "DecodedCursorVelocitySeries",
    "CursorIntentionSeries",
    "AlignedCursorVelocitySeries",
    "ResidualCursorVelocitySeries",
    "CursorAssistSeries",
    "ClickAssistSeries",
    "ClickStateSeries",
    "AcousticFeatures",
    "Audio",
    "Video",
    "Image",
    "Ieeg",
    "Eeg",
    "Emg",
    "Meg",
    "Fnirs",
    "Fmri",
    # text (BaseText)
    "Text",
    "Sentence",
    "Word",
    "Character",
    "Phoneme",
    "Keystroke",
    # categorical (CategoricalEvent)
    "Action",
    "Stimulus",
    "EyeState",
    "Artifact",
    "Seizure",
    "Background",
    "SleepStage",
    "Intertrial",
}

# Numpy dtype of the array **as the source ships it**, not as a loader delivers it.
# A dataset whose files disagree declares them all, e.g. ``"uint8/int8"``.
DTYPES = {
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
}


def warn_unknown(field: str, value, allowed: set) -> None:
    """Print a soft warning if ``value`` is not in ``allowed`` (``None`` is allowed)."""
    if value is None:
        return
    if value not in allowed:
        print(f"  [vocab] warning: {field}={value!r} not in {sorted(allowed)}")


def warn_unknown_dtype(value) -> None:
    """Check a ``dtype`` declaration, allowing ``a/b`` when a dataset's sources differ."""
    if not value:
        return
    for part in str(value).split("/"):
        warn_unknown("dtype", part.strip(), DTYPES)
