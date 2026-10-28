"""Query API for the metadata database.

High-level helpers that return pandas DataFrames for interactive / programmatic use,
plus :func:`read_sql` for ad-hoc SQL. Everything works identically on SQLite and
PostgreSQL.

    from neuralshed.database import query
    query.datasets(signal_type="spike_band_power")
    query.datasets(macro_task="human_communication", species="human")
    query.sessions(dataset="card_2024")
    query.participants()                          # cross-dataset person map
    query.subjects(participant="braingate2/T15")  # every dataset row of one person
    query.read_sql("SELECT macro_task, COUNT(*) FROM datasets GROUP BY macro_task")
"""

import pandas as pd
from sqlalchemy import select

from . import db
from .schema import Array, BrainArea, Dataset, EventType, Participant, Subject
from .schema import Session as SessionRow


def _df(stmt) -> pd.DataFrame:
    """Run a SQLAlchemy Core/ORM select and return a DataFrame."""
    return pd.read_sql(stmt, db.get_engine())


def read_sql(sql: str) -> pd.DataFrame:
    """Run a raw SQL string and return a DataFrame (ad-hoc analytics)."""
    return pd.read_sql(sql, db.get_engine())


def datasets(
    *,
    macro_task: str | None = None,
    decode_target: str | None = None,
    species: str | None = None,
    brain_area: str | None = None,
    signal_type: str | None = None,
    source_data_platform: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    """Filter datasets by any combination of dimensions.

    ``decode_target`` is what a decoder trained on the dataset outputs (``text`` / ``voice`` /
    ``movement``) — the axis that decides which datasets can pool into one decoder.
    ``brain_area`` matches either the raw label or the (future) HOMBA term;
    ``signal_type`` matches the source's own label or the canonical type;
    ``species``/``brain_area``/``signal_type`` are existence filters over related tables.
    ``species`` lives on the participant, so it only matches subjects linked to one.
    """
    stmt = select(Dataset)
    if macro_task is not None:
        stmt = stmt.where(Dataset.macro_task == macro_task)
    if decode_target is not None:
        stmt = stmt.where(Dataset.decode_target == decode_target)
    if source_data_platform is not None:
        stmt = stmt.where(Dataset.source_data_platform == source_data_platform)
    if status is not None:
        stmt = stmt.where(Dataset.status == status)
    if species is not None:
        stmt = stmt.where(
            Dataset.subjects.any(
                Subject.participant.has(Participant.species == species)
            )
        )
    if signal_type is not None:
        stmt = stmt.where(
            Dataset.event_types.any(
                (EventType.name == signal_type) | (EventType.signal_type == signal_type)
            )
        )
    if brain_area is not None:
        sub = (
            select(Subject.dataset_id)
            .join(Array, Array.subject_id == Subject.id)
            .join(BrainArea, BrainArea.id == Array.brain_area_id)
            .where(
                (BrainArea.raw_label == brain_area)
                | (BrainArea.homba_term == brain_area)
            )
        )
        stmt = stmt.where(Dataset.id.in_(sub))
    return _df(stmt)


def subjects(
    *,
    dataset: str | None = None,
    species: str | None = None,
    participant: str | None = None,
) -> pd.DataFrame:
    """Per-dataset subject rows. Each carries a ``participant_id`` linking it to the real
    person (``participants`` table); ``participant`` filters to one person across datasets.

    The person-invariant ``species`` / ``sex`` / ``implant_date`` are joined in from that
    participant, so a subject with no ``participant_id`` reports them as null.
    """
    stmt = select(
        Subject, Participant.species, Participant.sex, Participant.implant_date
    ).outerjoin(Participant, Subject.participant_id == Participant.id)
    if dataset is not None:
        stmt = stmt.where(Subject.dataset_id == dataset)
    if species is not None:
        stmt = stmt.where(Participant.species == species)
    if participant is not None:
        stmt = stmt.where(Subject.participant_id == participant)
    return _df(stmt)


def participants(*, namespace: str | None = None) -> pd.DataFrame:
    """One row per real person, with the datasets and subject rows they span — the
    cross-dataset identity map. Filter by ``namespace`` (e.g. ``braingate2``).

    Aggregated client-side (pandas) so it behaves identically on SQLite and PostgreSQL.
    """
    stmt = select(
        Participant.id,
        Participant.label,
        Participant.namespace,
        Participant.species,
        Participant.sex,
        Participant.implant_date,
        Subject.dataset_id,
    ).outerjoin(Subject, Subject.participant_id == Participant.id)
    if namespace is not None:
        stmt = stmt.where(Participant.namespace == namespace)
    df = _df(stmt)

    rows = []
    for pid, g in df.groupby("id", sort=False):
        ds = sorted(d for d in g["dataset_id"].dropna().unique())
        head = g.iloc[0]
        rows.append(
            {
                "id": pid,
                "label": head["label"],
                "namespace": head["namespace"],
                "species": head["species"],
                "sex": head["sex"],
                "implant_date": head["implant_date"],
                "n_subject_rows": int(g["dataset_id"].notna().sum()),
                "n_datasets": len(ds),
                "datasets": ", ".join(ds),
            }
        )
    out = pd.DataFrame(
        rows,
        columns=[
            "id",
            "label",
            "namespace",
            "species",
            "sex",
            "implant_date",
            "n_subject_rows",
            "n_datasets",
            "datasets",
        ],
    )
    return out.sort_values(["n_datasets", "id"], ascending=[False, True]).reset_index(
        drop=True
    )


def sessions(
    *,
    dataset: str | None = None,
    original_split: str | None = None,
    min_trials: int | None = None,
) -> pd.DataFrame:
    stmt = select(SessionRow)
    if dataset is not None:
        stmt = stmt.where(SessionRow.dataset_id == dataset)
    if original_split is not None:
        stmt = stmt.where(SessionRow.original_split == original_split)
    if min_trials is not None:
        stmt = stmt.where(SessionRow.n_trials >= min_trials)
    return _df(stmt)


def event_types(
    *, dataset: str | None = None, category: str | None = None
) -> pd.DataFrame:
    """Event types a dataset emits. ``category="data"`` returns the recorded signal
    streams; ``"text"``/``"categorical"`` the task/marker events."""
    stmt = select(EventType)
    if dataset is not None:
        stmt = stmt.where(EventType.dataset_id == dataset)
    if category is not None:
        stmt = stmt.where(EventType.category == category)
    return _df(stmt)


def brain_areas() -> pd.DataFrame:
    """All brain areas with how many arrays/datasets reference each."""
    return read_sql("""
        SELECT ba.raw_label, ba.homba_term, ba.brodmann,
               COUNT(DISTINCT s.dataset_id) AS n_datasets,
               COUNT(a.id)                  AS n_arrays
        FROM brain_areas ba
        LEFT JOIN arrays a   ON a.brain_area_id = ba.id
        LEFT JOIN subjects s ON s.id = a.subject_id
        GROUP BY ba.id
        ORDER BY n_datasets DESC, ba.raw_label
        """)


def catalog_summary() -> pd.DataFrame:
    """One row per dataset with rolled-up counts — a quick catalog overview."""
    return read_sql("""
        SELECT d.id, d.macro_task, d.source_data_platform, d.status,
               (SELECT COUNT(*) FROM subjects    su WHERE su.dataset_id = d.id) AS n_subjects,
               (SELECT COUNT(*) FROM event_types ev WHERE ev.dataset_id = d.id) AS n_event_types,
               (SELECT COUNT(*) FROM event_types ev WHERE ev.dataset_id = d.id AND ev.category='data') AS n_data_streams,
               (SELECT COUNT(*) FROM sessions    se WHERE se.dataset_id = d.id) AS n_sessions
        FROM datasets d
        ORDER BY d.macro_task, d.id
        """)
