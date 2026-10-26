"""Authoritative record of cross-dataset participants (real people).

A ``Subject`` row is scoped to one dataset (``(dataset_id, label)``), so the same person
recorded in several datasets shows up as several subject rows. In the BrainGate2 clinical
trial (NCT00912041, run by the NPTL / BrainGate consortium) the ``T*`` participant labels
are a shared identity: ``T15`` in ``card_2024`` is the same implant/person as ``T15`` in
``wairagkar_2025`` and ``kunz_2025``. The Pitt participant behind FALCON-H1 uses a separate
numbering (anonymized to ``HumanPitt``) and is its own person.

This module is the source of truth for the participant records (identity, not the mapping):
one entry per real person, keyed by a stable ``id`` (``<namespace>/<label>``). The
dataset-to-person mapping is declared where it belongs, next to the subject: each
``datasets/<name>/metadata.py`` puts a ``participant_id`` on its subject dict referencing an
``id`` here. ``populate`` reads that reference, upserts one ``Participant`` row per id, and
links each ``Subject`` to it.

To onboard a new dataset, reference the matching ``id`` from its ``metadata.py`` subject
(add a new entry here only for a genuinely new person). ``populate`` warns if a subject
references an ``id`` that is not defined here.
"""

import datetime as dt

#: Clinical trial each participant belongs to. A label is unique only within its
#: namespace, so labels in different namespaces are unrelated.
NAMESPACES = {
    "braingate2",  # NPTL / BrainGate2, NCT00912041 (Stanford/Davis/Brown/Emory)
    "pitt",  # Pitt / U Chicago iBCI, NCT01894802 (FALCON-H1 source)
}

#: One entry per real person. ``id`` is the stable cross-dataset key (``<namespace>/<label>``)
#: that each dataset's ``metadata.py`` references via its subject's ``participant_id``. Only
#: person-invariant facts live here, ``implant_date`` and ``handedness`` included. Traits that vary
#: between one dataset's recordings and another's (age, time since implant, clinical condition at the
#: time of recording) stay on ``Subject``.
PARTICIPANTS: list[dict] = [
    {
        "id": "braingate2/T5",
        "label": "T5",
        "namespace": "braingate2",
        "species": "human",
        "sex": "M",
        "handedness": "right",
        "implant_date": "2016-08-17",  # willet_2021 trial_day 994 for its first session
        "description": (
            "BrainGate2 T5. C4 spinal cord injury, tetraplegia. Hand-knob (left "
            "precentral gyrus) Utah arrays. Handwriting/motor iBCI."
        ),
    },
    {
        "id": "braingate2/T11",
        "label": "T11",
        "namespace": "braingate2",
        "species": "human",
        "sex": "M",
        "handedness": "right",
        "implant_date": None,  # no source publishes it; vargas_2025 gives trial days only
        "description": (
            "BrainGate2 T11. C4 AIS-B spinal cord injury, tetraplegia, with an earlier "
            "traumatic brain injury and surgically evacuated subdural haemorrhage that "
            "resolved fully. Two 96-ch arrays in the hand knob of left precentral "
            "gyrus. Gesture and cursor iBCI, sessions at his residence."
        ),
    },
    {
        "id": "braingate2/T12",
        "label": "T12",
        "namespace": "braingate2",
        "species": "human",
        "sex": "F",
        "handedness": "left",
        "implant_date": "2022-03-30",  # kunz_2025 paper: post-implant days 412-995
        "description": (
            "BrainGate2 T12. ALS (bulbar onset), dysarthria. Speech motor cortex "
            "(four 64-ch arrays). Attempted-speech iBCI."
        ),
    },
    {
        "id": "braingate2/T15",
        "label": "T15",
        "namespace": "braingate2",
        "species": "human",
        "sex": "M",
        "handedness": "left",
        "implant_date": "2023-07-17",  # card_2024 post_implant_day field vs session dates; kunz_2025 "July 2023"
        "description": (
            "BrainGate2 T15. ALS, severe dysarthria. 4x64-ch Utah arrays in left "
            "ventral precentral gyrus (v6v/M1/55b/d6v). Speech iBCI."
        ),
    },
    {
        "id": "braingate2/T16",
        "label": "T16",
        "namespace": "braingate2",
        "species": "human",
        "sex": "F",
        "handedness": "right",
        "implant_date": "2023-12-07",  # kunz_2025 paper: post-implant days 88-377
        "description": (
            "BrainGate2 T16. Pontine stroke (~19 yr prior), tetraplegia and dysarthria "
            "(speaks slowly and quietly). 4x64-ch Utah arrays in left precentral gyrus "
            "(2x hand-knob 6d, 6v, 55b/PEF border). Speech iBCI."
        ),
    },
    {
        "id": "braingate2/T17",
        "label": "T17",
        "namespace": "braingate2",
        "species": "human",
        "sex": "M",
        "handedness": "right",
        "implant_date": "2024-02-29",  # jude_2026_speech trial days; reproduced by kunz_2025's day range
        "description": (
            "BrainGate2 T17. ALS, anarthric. Arrays spanning areas 6d/6v/55b (left "
            "hemisphere). Attempted-speech/typing iBCI."
        ),
    },
    {
        "id": "braingate2/T18",
        "label": "T18",
        "namespace": "braingate2",
        "species": "human",
        "sex": "M",
        "handedness": "right",
        "implant_date": "2024-10-21",  # jude_2026_typing Trial Day 43 for its first session
        "description": (
            "BrainGate2 T18. Spinal cord injury. Bilateral area-6d arrays. "
            "Closed-loop typing/handwriting iBCI."
        ),
    },
    {
        "id": "pitt/HumanPitt",
        "label": "HumanPitt",
        "namespace": "pitt",
        "species": "human",
        "sex": "U",
        "handedness": "unknown",
        "implant_date": None,  # identity redacted, no implant date published
        "description": (
            "University of Pittsburgh iBCI participant (FALCON-H1), identity redacted by "
            "the data owners. Two 96-ch hand-knob arrays. Separate numbering from BrainGate2."
        ),
    },
]

#: Fields copied onto the ``Participant`` ORM row.
PARTICIPANT_FIELDS = (
    "label",
    "namespace",
    "species",
    "sex",
    "implant_date",
    "handedness",
    "description",
)

#: ``id -> registry entry``. Also the set of valid ``participant_id`` references.
BY_ID: dict[str, dict] = {}


def _build_index() -> None:
    """Index ``PARTICIPANTS`` by id and self-check the declarations."""
    for p in PARTICIPANTS:
        pid = p["id"]
        if pid in BY_ID:
            raise ValueError(f"Duplicate participant id {pid!r}")
        if p["namespace"] not in NAMESPACES:
            raise ValueError(f"{pid!r}: unknown namespace {p['namespace']!r}")
        if not pid.startswith(f"{p['namespace']}/"):
            raise ValueError(f"{pid!r} must start with '{p['namespace']}/'")
        BY_ID[pid] = p


_build_index()


def participant_by_id(participant_id: str) -> dict | None:
    """Return the registry entry for a participant id, or ``None`` if undefined."""
    return BY_ID.get(participant_id)


def implant_date(participant_id: str) -> dt.date | None:
    """Implant date as a ``date``, or ``None`` where it is unpublished."""
    iso = BY_ID[participant_id]["implant_date"]
    return dt.date.fromisoformat(iso) if iso else None


def post_implant_day(participant_id: str, session_date: str | None) -> int | None:
    """Days from implant to an ISO ``session_date``, or ``None`` if either is unknown."""
    implant = implant_date(participant_id)
    if implant is None or not session_date:
        return None
    return (dt.date.fromisoformat(session_date) - implant).days
