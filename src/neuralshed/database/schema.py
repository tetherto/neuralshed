"""SQLAlchemy ORM schema for the neuralshed metadata database.

Schema-as-code: one table per catalog entity, populated from the per-dataset
``metadata.py`` declarations and standardized via ``vocab.py`` (soft-enforced at populate
time). Backend-agnostic — SQLite by default, PostgreSQL via ``DATABASE_URL``.

Entities:
- ``Dataset``     — provenance (source platform / id / url), licence, references, status, macro-task.
- ``Participant`` — a real person, spanning datasets. Holds the person-invariant facts (``species``, ``sex``, ``implant_date``, ``handedness``). ``Subject`` rows link to it via ``participant_id``, referenced from each dataset's ``metadata.py`` and defined in ``datasets/participants.py``.
- ``Subject``     — one dataset's view of a participant, at the time of that dataset's recordings: when (``age``, ``time_since_implant``) + clinical condition (JSON).
- ``BrainArea``   — a subject's implant region (owned by ``Subject``; ``homba_id``/``uberon_id``/``homba_term`` are the ontology mapping, filled during harmonization).
- ``Array``       — recording device + channel block + stereotaxic coords (JSON), linked to a ``BrainArea``.
- ``EventType``   — every kind of event a dataset emits (neuroai unifies signals + task events under
                    ``Event``): ``category`` (data/text/categorical) + ``event_class``, data-stream
                    attributes (signal type, modality, rate, …) and behavioral ``task_condition``.
- ``Session``     — per-session rows (date, post-implant day, original split, counts; drive-dependent).
- ``Collection`` / ``CollectionDataset`` — a saved group of datasets to train one decoder together
                    (many-to-many with ``Dataset``); the persisted form of a decoder's
                    spec, plus the step chain that pools them, assembled in the collection builder app.

Controlled-vocabulary columns (``source_data_platform``, ``status``, ``macro_task``,
``device_type``, ``signal_type``, ``modality``, ``task_condition``) are plain strings here
and validated softly by ``vocab.py`` at populate time.
"""

import datetime as dt
import re

from sqlalchemy import (
    JSON,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    validates,
)

#: Portable JSON column — ``json`` on both SQLite and PostgreSQL. Promote to ``JSONB``
#: with a migration if/when we need to query or index inside these fields.
JSONType = JSON

#: Collection names are used as identifiers (URLs, filenames, cache uids), so no spaces.
COLLECTION_NAME_RE = re.compile(r"[A-Za-z0-9_-]+")


def _require_id_prefix(key: str, value: str | None, prefix: str) -> str | None:
    """`@validates` helper: a non-null ontology id must start with ``prefix``.

    Nullable columns pass ``None`` through (a region with no mapping yet); any other
    value must carry the namespace prefix (e.g. ``UBERON_…``), else ``ValueError``.
    """
    if value is not None and not value.startswith(prefix):
        raise ValueError(f"{key} must start with {prefix!r}; got {value!r}")
    return value


#: ISO-8601 duration (date + optional time part, at least one component). Elapsed times use
#: the date part, e.g. ``P45Y``, ``P45Y6M``, ``P25D``.
_ISO8601_DURATION = re.compile(
    r"^P(?!$)(\d+Y)?(\d+M)?(\d+W)?(\d+D)?(T(?!$)(\d+H)?(\d+M)?(\d+(\.\d+)?S)?)?$"
)


def _require_iso8601_duration(key: str, value: str | None) -> str | None:
    """`@validates` helper for the elapsed-time columns (``age``, ``time_since_implant``).

    ``None`` passes through (unavailable/redacted); any other value must be a string
    matching ISO-8601 (e.g. ``"P45Y"``, ``"P25D"``), else ``ValueError``. ``metadata.py``
    supplies the ISO form directly — bare ints are rejected, not coerced.
    """
    if value is None:
        return None
    if isinstance(value, str) and _ISO8601_DURATION.match(value):
        return value
    raise ValueError(
        f"{key} must be None or an ISO-8601 duration like 'P45Y'; got {value!r}"
    )


#: Allowed participant species (common names). Closed set — hard-validated, no "other".
_SPECIES = ("human", "mouse", "rat", "macaque", "marmoset")


def _require_species(value: str | None) -> str | None:
    """`@validates` helper for ``Participant.species``: ``None`` or one of ``_SPECIES``."""
    if value is None:
        return None
    if value in _SPECIES:
        return value
    raise ValueError(f"species must be None or one of {list(_SPECIES)}; got {value!r}")


#: Allowed participant sex. Closed set — hard-validated. ``U`` covers unknown / redacted /
#: not-reported, so ``None`` means the field was never declared at all.
_SEX = ("M", "F", "U")


def _require_sex(value: str | None) -> str | None:
    """`@validates` helper for ``Participant.sex``: ``None`` or one of ``_SEX``.

    Not coerced — ``"male"`` / ``"f"`` are rejected, so a source's own wording gets
    normalized in ``participants.py`` rather than silently reinterpreted here.
    """
    if value is None:
        return None
    if value in _SEX:
        return value
    raise ValueError(f"sex must be None or one of {list(_SEX)}; got {value!r}")


#: Full ISO-8601 calendar date. Day precision only: a month-precise ``"2022-03"`` is rejected
_ISO8601_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _require_iso8601_date(value: str | None) -> str | None:
    """`@validates` helper for ``Participant.implant_date``: ``None`` or ``YYYY-MM-DD``.

    ``None`` passes through (no source publishes it).
    """
    if value is None:
        return None
    if isinstance(value, str) and _ISO8601_DATE.match(value):
        try:
            dt.date.fromisoformat(value)
        except ValueError:
            raise ValueError(
                f"implant_date is not a real date; got {value!r}"
            ) from None
        return value
    raise ValueError(
        f"implant_date must be None or a full ISO date like '2023-07-17'; got {value!r}"
    )


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # snake_case slug
    title: Mapped[str | None] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(Text)
    source_data_platform: Mapped[str | None] = mapped_column(
        String
    )  # vocab.SOURCE_DATA_PLATFORMS
    source_id: Mapped[str | None] = mapped_column(String)  # accession / DOI
    source_data_url: Mapped[str | None] = mapped_column(String)  # resolved full URL
    status: Mapped[str | None] = mapped_column(String)  # vocab.STATUSES
    macro_task: Mapped[str | None] = mapped_column(String)  # vocab.MACRO_TASKS
    decode_target: Mapped[str | None] = mapped_column(
        String
    )  # vocab.DECODE_TARGETS — what a decoder trained on this dataset outputs (text/voice/movement)
    bibtex: Mapped[str | None] = mapped_column(Text)  # raw BibTeX
    publication_doi: Mapped[str | None] = mapped_column(String)
    code_repo: Mapped[str | None] = mapped_column(String)
    licence: Mapped[str | None] = mapped_column(String)
    notes: Mapped[str | None] = mapped_column(Text)
    # neuralset Study subclass; the `name` discriminator in a collection config. One per dataset,
    # so a member's class is the only link back to its dataset.
    study_class: Mapped[str | None] = mapped_column(String, unique=True, index=True)

    subjects: Mapped[list["Subject"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    event_types: Mapped[list["EventType"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    sessions: Mapped[list["Session"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )
    collection_links: Mapped[list["CollectionDataset"]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan"
    )


class Participant(Base):
    """A real person, independent of any one dataset.

    The same participant recorded in several datasets has one row here and several
    ``Subject`` rows (one per dataset). ``id`` is the stable cross-dataset key
    (``<namespace>/<label>``, e.g. ``braingate2/T15``). Participant records are defined in
    ``datasets/participants.py``; each dataset's ``metadata.py`` references one by ``id`` via
    its subject's ``participant_id``, and ``populate`` upserts the row at populate time. Only
    person-invariant facts live here — ``species``, ``sex``, ``implant_date`` and
    ``handedness`` included.
    What varies between one dataset's recordings and another's (``age``,
    ``time_since_implant``) stays on ``Subject``.
    """

    __tablename__ = "participants"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # e.g. "braingate2/T15"
    label: Mapped[str] = mapped_column(String)  # e.g. T15, HumanPitt
    namespace: Mapped[str | None] = mapped_column(
        String
    )  # participants.NAMESPACES (clinical trial or lab)
    species: Mapped[str | None] = mapped_column(
        String
    )  # common name: human/mouse/rat/macaque/marmoset
    sex: Mapped[str | None] = mapped_column(String)  # M/F/U
    implant_date: Mapped[str | None] = mapped_column(
        String
    )  # full ISO date "2023-07-17" (month precision rejected, not coerced)
    handedness: Mapped[str | None] = mapped_column(String)  # vocab.HANDEDNESS
    description: Mapped[str | None] = mapped_column(Text)

    subjects: Mapped[list["Subject"]] = relationship(back_populates="participant")

    @validates("species")
    def _validate_species(self, key: str, value: str | None) -> str | None:
        return _require_species(value)

    @validates("sex")
    def _validate_sex(self, key: str, value: str | None) -> str | None:
        return _require_sex(value)

    @validates("implant_date")
    def _validate_implant_date(self, key: str, value: str | None) -> str | None:
        return _require_iso8601_date(value)


class Subject(Base):
    """One dataset's view of a participant, at the time of that dataset's recordings.

    ``species``, ``sex``, ``implant_date`` and ``handedness`` are person-invariant and live on
    ``Participant``. What varies per dataset is *when* the recordings happened relative to
    the person's life (``age``) and to their implant (``time_since_implant``), both
    ISO-8601 durations. So ``participant.implant_date + subject.time_since_implant`` is
    that dataset's first session date.
    """

    __tablename__ = "subjects"
    __table_args__ = (
        UniqueConstraint("dataset_id", "label", name="uq_subject_dataset_label"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    participant_id: Mapped[str | None] = mapped_column(
        ForeignKey("participants.id")
    )  # cross-dataset person identity (datasets/participants.py); None if unlinked
    label: Mapped[str] = mapped_column(String)  # e.g. T5, T12, HumanPitt
    age: Mapped[str | None] = mapped_column(
        String
    )  # ISO-8601 duration, e.g. "P45Y" (None if unavailable; non-ISO values rejected)
    time_since_implant: Mapped[str | None] = mapped_column(
        String
    )  # ISO-8601 duration from implant to this dataset's first session, e.g. "P25D"
    clinical_condition: Mapped[list | None] = mapped_column(JSONType)  # list[str]

    @validates("age", "time_since_implant")
    def _validate_duration(self, key: str, value: str | None) -> str | None:
        return _require_iso8601_duration(key, value)

    dataset: Mapped["Dataset"] = relationship(back_populates="subjects")
    participant: Mapped["Participant | None"] = relationship(back_populates="subjects")
    arrays: Mapped[list["Array"]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )
    brain_areas: Mapped[list["BrainArea"]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )
    sessions: Mapped[list["Session"]] = relationship(back_populates="subject")


class BrainArea(Base):
    __tablename__ = "brain_areas"
    __table_args__ = (
        UniqueConstraint(
            "subject_id",
            "raw_label",
            name="uq_brain_area_subject_label",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id"))
    raw_label: Mapped[str] = mapped_column(String)  # as named by the source
    homba_term: Mapped[str | None] = mapped_column(
        String
    )  # filled during later harmonization
    homba_id: Mapped[str | None] = mapped_column(
        String
    )  # BICAN HOMBA ontology id, e.g. "HOMBA_10163" (purl.brain-bican.org/ontology/<id>)
    uberon_id: Mapped[str | None] = mapped_column(
        String
    )  # UBERON ontology id, e.g. "UBERON_0001384" (purl.obolibrary.org/obo/<id>)
    brodmann: Mapped[str | None] = mapped_column(String)
    description: Mapped[str | None] = mapped_column(Text)

    subject: Mapped["Subject"] = relationship(back_populates="brain_areas")
    arrays: Mapped[list["Array"]] = relationship(back_populates="brain_area")

    @validates("homba_id")
    def _validate_homba_id(self, key: str, value: str | None) -> str | None:
        return _require_id_prefix(key, value, "HOMBA_")

    @validates("uberon_id")
    def _validate_uberon_id(self, key: str, value: str | None) -> str | None:
        return _require_id_prefix(key, value, "UBERON_")


class Array(Base):
    __tablename__ = "arrays"
    __table_args__ = (
        # An array is identified by its hemisphere and name within a subject, so a
        # bilateral implant can reuse one anatomical name on each side.
        UniqueConstraint(
            "subject_id", "hemisphere", "name", name="uq_array_subject_hemisphere_name"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id"))
    name: Mapped[str] = mapped_column(String)
    hemisphere: Mapped[str | None] = mapped_column(String)  # vocab.HEMISPHERES
    device_type: Mapped[str | None] = mapped_column(String)  # vocab.DEVICE_TYPES
    n_channels: Mapped[int | None] = mapped_column(Integer)
    channel_start: Mapped[int | None] = mapped_column(Integer)
    channel_stop: Mapped[int | None] = mapped_column(Integer)
    brain_area_id: Mapped[int | None] = mapped_column(ForeignKey("brain_areas.id"))
    stereotaxic_coordinates: Mapped[dict | None] = mapped_column(JSONType)

    subject: Mapped["Subject"] = relationship(back_populates="arrays")
    brain_area: Mapped["BrainArea | None"] = relationship(back_populates="arrays")


class EventType(Base):
    """A type of event a dataset emits — unifies data streams and task/marker events.

    Mirrors neuroai's single ``Event`` hierarchy: ``category`` is the base family
    (``data`` = ``BaseDataEvent``, ``text`` = ``BaseText``, ``categorical`` =
    ``CategoricalEvent``) and ``event_class`` (nullable) names the concrete neuroai
    subclass — the forward bridge to a future ``populate_from_study()``. Data-stream
    columns (``signal_type`` … ``dtype``) apply when ``category == "data"``;
    ``task_condition`` applies to the behavioral ``text``/``categorical`` events.
    """

    __tablename__ = "event_types"
    __table_args__ = (
        UniqueConstraint("dataset_id", "name", name="uq_event_dataset_name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    name: Mapped[str] = mapped_column(String)  # the source's own label
    category: Mapped[str | None] = mapped_column(String)  # vocab.EVENT_CATEGORIES
    event_class: Mapped[str | None] = mapped_column(
        String
    )  # neuroai subclass (vocab.EVENT_CLASSES)
    # data-stream attributes (category == "data")
    signal_type: Mapped[str | None] = mapped_column(
        String
    )  # vocab.SIGNAL_TYPES (canonical)
    modality: Mapped[str | None] = mapped_column(String)  # vocab.RECORDING_MODALITIES
    bin_size_ms: Mapped[float | None] = mapped_column(Float)
    rate_hz: Mapped[float | None] = mapped_column(Float)
    # dtype the source ships, not what a loader delivers; ``a/b`` when files differ
    dtype: Mapped[str | None] = mapped_column(String)  # vocab.DTYPES
    # behavioral attribute (category in {"text", "categorical"})
    task_condition: Mapped[str | None] = mapped_column(String)  # vocab.TASK_CONDITIONS
    description: Mapped[str | None] = mapped_column(Text)

    dataset: Mapped["Dataset"] = relationship(back_populates="event_types")


class Session(Base):
    __tablename__ = "sessions"
    __table_args__ = (
        UniqueConstraint("dataset_id", "label", name="uq_session_dataset_label"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    subject_id: Mapped[int | None] = mapped_column(ForeignKey("subjects.id"))
    label: Mapped[str] = mapped_column(String)
    task: Mapped[str | None] = mapped_column(String)
    task_condition: Mapped[str | None] = mapped_column(String)  # vocab.TASK_CONDITIONS
    date: Mapped[str | None] = mapped_column(String)  # ISO date parsed from the label
    post_implant_day: Mapped[int | None] = mapped_column(Integer)
    n_trials: Mapped[int | None] = mapped_column(Integer)
    original_split: Mapped[str | None] = mapped_column(
        String
    )  # the paper's own split label

    dataset: Mapped["Dataset"] = relationship(back_populates="sessions")
    subject: Mapped["Subject | None"] = relationship(back_populates="sessions")


class Collection(Base):
    """A saved group of datasets assembled to train one decoder together.

    The persisted form of a decoder's input spec: a collection names its member datasets and
    what it reads and decodes. Datasets belong to many
    collections and a collection holds many datasets: a many-to-many through
    ``collection_datasets``. Assembled in the collection builder frontend;
    ``database/collections.py`` is the read/write API.

    ``chain`` holds the neuroai step chain the builder composes (a ``Chain`` headed by
    ``CombineStudies``), ready to become a ``neuraltrain`` ``Data.study``. It carries no
    ``path``: that is environment, injected when the config is loaded.

    Three groups of columns, and the difference between them matters:

    *Declared* — what a person chose. ``input_event_types`` and ``target_event_types`` are
    neuroai event class names (``SpikeCounts``, ``Sentence``), which turn a collection from a
    pile of datasets into a pile assembled *for* something. Inputs are wanted-if-present, so a
    member lacking one still joins, while targets are required. Both are checkable against
    ``event_types`` at save time, with no data drive mounted. Naming the target class also says
    which decoder head this is, more precisely than a coarse family label would.

    *Resolved* — ``manifest``, the layout table
    (:class:`~neuralshed.manifest.CollectionManifest`) built by reading the data. It is a
    cache: always rebuildable, and safe to delete. ``manifest_recipe_digest`` is what makes it
    detectably stale, and ``manifest_builder_version`` guards against trusting a blob written
    by code that resolved layouts differently.

    *Fitted* values live in neither place. They belong to a training run and travel in its
    checkpoint.
    """

    __tablename__ = "collections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    notes: Mapped[str | None] = mapped_column(Text)
    chain: Mapped[dict | None] = mapped_column(JSONType)  # ns.Step config (no paths)

    # declared intent
    input_event_types: Mapped[list | None] = mapped_column(JSONType)  # list[str]
    target_event_types: Mapped[list | None] = mapped_column(JSONType)  # list[str]

    # resolved layout table, a rebuildable cache
    manifest: Mapped[dict | None] = mapped_column(JSONType)
    manifest_built_at: Mapped[str | None] = mapped_column(String)  # ISO-8601 UTC
    manifest_recipe_digest: Mapped[str | None] = mapped_column(String)
    manifest_builder_version: Mapped[int | None] = mapped_column(Integer)

    members: Mapped[list["CollectionDataset"]] = relationship(
        back_populates="collection", cascade="all, delete-orphan"
    )

    @validates("name")
    def _check_name(self, key: str, value: str) -> str:
        """Names go into filenames, URLs and cache uids, so keep them to ``[A-Za-z0-9_-]``."""
        if not COLLECTION_NAME_RE.fullmatch(value or ""):
            raise ValueError(
                f"{key} must be non-empty and contain only letters, digits, '-' and '_'; "
                f"got {value!r}"
            )
        return value


class CollectionDataset(Base):
    """A dataset's membership in a collection (association object for the many-to-many).

    Per-study filters live on the study step of the collection config (``Study.query``), not
    here.
    """

    __tablename__ = "collection_datasets"
    __table_args__ = (
        UniqueConstraint("collection_id", "dataset_id", name="uq_collection_dataset"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    collection_id: Mapped[int] = mapped_column(
        ForeignKey("collections.id", ondelete="CASCADE")
    )
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))

    collection: Mapped["Collection"] = relationship(back_populates="members")
    dataset: Mapped["Dataset"] = relationship(back_populates="collection_links")
