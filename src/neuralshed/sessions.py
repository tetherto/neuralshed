"""Per-member session rules: the session -> day_idx / calendar-day / month mapping."""

import dataclasses
import datetime
import typing as tp

from neuralset.extractors import LabelEncoder

from neuralshed.datasets.card_2024 import utils as card_utils

# Months are 30-day slots, not calendar months.
MONTH_SLOT_DAYS = 30

DateSource = tp.Mapping[str, str] | tp.Iterable[tuple[str, str]]


def _parse_date(value: tp.Any) -> datetime.date:
    """Study-provided date (``"2024-03-03"``, ``"2024.03.03"`` or a ``date``) -> ``date``.

    Some members write dates dotted (``2024.03.03``, as in the session names
    ``t15.2024.03.03*``); ``date.fromisoformat`` only takes hyphens.
    """
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    text = str(value)
    if len(text) == 10 and text[4] == "." and text[7] == ".":
        text = text.replace(".", "-")
    return datetime.date.fromisoformat(text)


@dataclasses.dataclass(frozen=True)
class SessionRule:
    """One member's chronological session list and the values derived from it.

    Immutable by construction: ``sessions`` and ``dates`` are the frozen inputs,
    every mapping below is a pure function of them.
    """

    member: str
    sessions: tuple[str, ...]
    dates: tuple[datetime.date, ...]

    def __post_init__(self) -> None:
        if not self.sessions:
            raise ValueError(f"SessionRule({self.member!r}) needs at least one session")
        if len(self.sessions) != len(self.dates):
            raise ValueError(
                f"SessionRule({self.member!r}): got {len(self.sessions)} sessions "
                f"but {len(self.dates)} dates"
            )
        duplicates = {s for s in self.sessions if self.sessions.count(s) > 1}
        if duplicates:
            raise ValueError(
                f"SessionRule({self.member!r}): duplicate session names {sorted(duplicates)}"
            )
        if self.dates != tuple(sorted(self.dates)):
            raise ValueError(
                f"SessionRule({self.member!r}): dates are not chronological; "
                "build the rule through build_session_rule() which sorts them"
            )

    @property
    def day_index(self) -> dict[str, int]:
        """session name -> position in the chronological list."""
        return {session: i for i, session in enumerate(self.sessions)}

    @property
    def calendar_day(self) -> dict[str, int]:
        """session name -> days since the member's first session."""
        first = self.dates[0]
        return {
            session: (date - first).days
            for session, date in zip(self.sessions, self.dates)
        }

    @property
    def month(self) -> dict[str, int]:
        """session name -> 30-day slot index."""
        return {
            session: day // MONTH_SLOT_DAYS
            for session, day in self.calendar_day.items()
        }

    @property
    def num_days(self) -> int:
        """Number of sessions; sizes the per-day parameter blocks."""
        return len(self.sessions)

    @property
    def num_date_ranks(self) -> int:
        """Number of distinct dates; the day count for date-keyed members (as in Kunz)."""
        return len(set(self.dates))

    @property
    def num_month_slots(self) -> int:
        """Month slots covered by the rule; 21 for Card."""
        return max(self.month.values()) + 1

    def day_index_field(
        self,
        field: str = "session",
        event_types: str | tuple[str, ...] = "Sentence",
    ) -> LabelEncoder:
        """The session rule as a segmenter field extractor.

        `field` selects the event column ranked into day indices:
        "session" (Card, Willett, Jude) or "date" (Kunz, whose day
        count is 'num_date_ranks', not 'num_days').
        The day is read from `Sentence` rows by default;
        `event_types` selects other cue types (i.e. `Word` or `Action` rows).
        Any field besides the two named raises `ValueError`.
        """
        if field == "session":
            mapping = self.day_index
        elif field == "date":
            mapping = {
                date.isoformat(): rank
                for rank, date in enumerate(sorted(set(self.dates)))
            }
        else:
            raise ValueError(
                f"SessionRule({self.member!r}).day_index_field(field={field!r}): "
                "expected 'session' or 'date'"
            )
        return LabelEncoder(
            event_types=event_types,
            event_field=field,
            predefined_mapping=mapping,
        )


def build_session_rule(member: str, dated_sessions: DateSource) -> SessionRule:
    """Derive a member's session rule from (session, date) pairs."""
    pairs = (
        list(dated_sessions.items())
        if isinstance(dated_sessions, tp.Mapping)
        else list(dated_sessions)
    )
    if not pairs:
        raise ValueError(f"SessionRule({member!r}): no dated sessions given")
    ordered = sorted(
        ((session, _parse_date(date)) for session, date in pairs),
        key=lambda pair: (pair[1], pair[0]),
    )
    return SessionRule(
        member=member,
        sessions=tuple(session for session, _ in ordered),
        dates=tuple(date for _, date in ordered),
    )


def card_session_rule() -> SessionRule:
    """Card's session rule from the Study's own session discovery."""
    return build_session_rule(
        "card_2024",
        {ref.session: ref.date for ref in card_utils.list_sessions()},
    )


def willet_session_rule() -> SessionRule:
    """Willett's session rule from the Study's own per-file rows, day-keyed by session."""
    from neuralshed.datasets.willet_2023 import metadata as willet_metadata

    return build_session_rule(
        "willet_2023", {r["label"]: r["date"] for r in willet_metadata.sessions()}
    )


def kunz_session_rule(participant: str) -> SessionRule:
    """One Kunz participant's rule over the sentenceData files.

    Sessions are files; the day is the rank of the file's date among the
    participant's distinct dates, so it's read with ``day_index_field(field="date")``
    and counted with ``num_date_ranks`` (1 for T12, 3 for T15).
    Being date-keyed, the rank also tags the ``_eval`` val files without them being rule sessions.
    """
    from neuralshed.datasets.kunz_2025 import metadata as kunz_metadata

    return build_session_rule(
        f"kunz_{participant.lower()}",
        {
            r["label"]: r["date"]
            for r in kunz_metadata.sessions()
            if r["task"] == "sentenceData"
            and r["subject_label"].lower() == participant.lower()
        },
    )


JUDE_SPEECH_DATASET_SESSIONS = (
    "t17.2024.04.15",
    "t17.2024.04.16",
    "t17.2024.04.17",
    "t17.2024.05.06",
    "t17.2024.05.07",
)


def jude_session_rule() -> SessionRule:
    """Jude's rule over the five sessions of the published speech dataset: day-keyed by session"""
    from neuralshed.datasets.jude_2026_speech import metadata as jude_metadata

    return build_session_rule(
        "jude_2026_speech",
        {
            r["label"]: r["date"]
            for r in jude_metadata.sessions()
            if r["label"] in JUDE_SPEECH_DATASET_SESSIONS
        },
    )
