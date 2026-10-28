"""Session-rule tests: Card examples plus synthetic behaviour.
CARD_SESSIONS below is a checked-in copy of a deprecated hand-written list
used to track chronological order between sessions.
"""

import datetime

import pandas as pd
import pytest
from neuralset.events.etypes import Event
from neuralset.extractors import LabelEncoder

from neuralshed.datasets.utils import dataset_dir
from neuralshed.sessions import (
    SessionRule,
    build_session_rule,
    card_session_rule,
)

CARD_SESSIONS = (
    "t15.2023.08.11",
    "t15.2023.08.13",
    "t15.2023.08.18",
    "t15.2023.08.20",
    "t15.2023.08.25",
    "t15.2023.08.27",
    "t15.2023.09.01",
    "t15.2023.09.03",
    "t15.2023.09.24",
    "t15.2023.09.29",
    "t15.2023.10.01",
    "t15.2023.10.06",
    "t15.2023.10.08",
    "t15.2023.10.13",
    "t15.2023.10.15",
    "t15.2023.10.20",
    "t15.2023.10.22",
    "t15.2023.11.03",
    "t15.2023.11.04",
    "t15.2023.11.17",
    "t15.2023.11.19",
    "t15.2023.11.26",
    "t15.2023.12.03",
    "t15.2023.12.08",
    "t15.2023.12.10",
    "t15.2023.12.17",
    "t15.2023.12.29",
    "t15.2024.02.25",
    "t15.2024.03.03",
    "t15.2024.03.08",
    "t15.2024.03.15",
    "t15.2024.03.17",
    "t15.2024.04.25",
    "t15.2024.04.28",
    "t15.2024.05.10",
    "t15.2024.06.14",
    "t15.2024.07.19",
    "t15.2024.07.21",
    "t15.2024.07.28",
    "t15.2025.01.10",
    "t15.2025.01.12",
    "t15.2025.03.14",
    "t15.2025.03.16",
    "t15.2025.03.30",
    "t15.2025.04.13",
)

_HAS_CARD_DATA = dataset_dir("card_2024").is_dir()
needs_card_data = pytest.mark.skipif(
    not _HAS_CARD_DATA,
    reason="card_2024 sources not present under NEURALSHED_DATA_ROOT",
)


def _old_calendar_days(sessions: tuple[str, ...]) -> list[int]:
    """The old sessions_to_calendar_days recipe, transcribed as expectation."""

    def _parse(s: str) -> datetime.datetime:
        return datetime.datetime.strptime(s[-10:], "%Y.%m.%d")

    dates = [_parse(s) for s in sessions]
    first = min(dates)
    return [(d - first).days for d in dates]


def _sentence_row(start: float, session: str) -> dict:
    return {
        "type": "Sentence",
        "start": start,
        "duration": 1.0,
        "timeline": "recording",
        "text": "hello world",
        "session": session,
    }


@needs_card_data
def test_card_rule_equals_old_recipe():
    """§7 step 1 exit: day_idx equals the old dayIdx, calendar days equal the
    old recipe, and the rule spans 21 month slots."""
    rule = card_session_rule()

    assert rule.sessions == CARD_SESSIONS
    assert rule.day_index == {s: i for i, s in enumerate(CARD_SESSIONS)}
    expected_days = _old_calendar_days(CARD_SESSIONS)
    assert rule.calendar_day == dict(zip(CARD_SESSIONS, expected_days))
    assert rule.num_days == 45
    assert rule.num_month_slots == 21


def test_build_sorts_by_date_then_name():
    """Dates drive the order (not names); same-date ties break by name."""
    rule = build_session_rule(
        "m",
        {
            "z.2024.02.01": "2024-02-01",
            "b.2024.01.01": "2024-01-01",
            "a.2024.01.01": "2024-01-01",
        },
    )
    assert rule.sessions == ("a.2024.01.01", "b.2024.01.01", "z.2024.02.01")
    assert rule.day_index == {"a.2024.01.01": 0, "b.2024.01.01": 1, "z.2024.02.01": 2}
    assert rule.calendar_day == {
        "a.2024.01.01": 0,
        "b.2024.01.01": 0,
        "z.2024.02.01": 31,
    }
    assert rule.month == {"a.2024.01.01": 0, "b.2024.01.01": 0, "z.2024.02.01": 1}
    assert rule.num_days == 3
    assert rule.num_month_slots == 2


def test_rule_rejects_bad_inputs():
    with pytest.raises(ValueError, match="no dated sessions given"):
        build_session_rule("m", {})
    with pytest.raises(ValueError, match="at least one session"):
        SessionRule(member="m", sessions=(), dates=())
    with pytest.raises(ValueError, match="duplicate session names"):
        build_session_rule("m", [("a", "2024-01-01"), ("a", "2024-01-02")])
    with pytest.raises(ValueError, match="not chronological"):
        SessionRule(
            member="m",
            sessions=("b", "a"),
            dates=(datetime.date(2024, 2, 1), datetime.date(2024, 1, 1)),
        )


def test_day_index_field_round_trips_through_json():
    """The mapping is a real pydantic field: it survives serialization, which is
    what makes option B storable with the recipe (unlike fitted private state)."""
    rule = build_session_rule("m", {"a": "2024-01-01", "b": "2024-01-05"})
    dumped = rule.day_index_field().model_dump_json()
    restored = LabelEncoder.model_validate_json(dumped)
    assert restored.predefined_mapping == rule.day_index


def test_day_index_field_encodes_like_the_old_recipe():
    """get_static output equals the session's position in the rule, reading the
    session string from event.extra (where Event.from_dict funnels it)."""
    rule = build_session_rule(
        "m", {"a": "2024-01-01", "b": "2024-01-05", "c": "2024-02-05"}
    )
    frame = pd.DataFrame([_sentence_row(0.0, s) for s in rule.sessions])
    encoder = rule.day_index_field()
    encoder.prepare(frame)
    events = [Event.from_dict(row) for row in frame.to_dict("records")]
    encoded = [encoder.get_static(event).item() for event in events]
    assert encoded == [rule.day_index[s] for s in rule.sessions]


def test_day_index_field_raises_on_unknown_session():
    """A session the rule does not know is a loud error, never a silent
    renumbering of every other session's day parameters."""
    rule = build_session_rule("m", {"a": "2024-01-01", "b": "2024-01-05"})
    frame = pd.DataFrame([_sentence_row(0.0, "zz.2030.01.01")])
    with pytest.raises(ValueError, match="not in the existing mapping"):
        rule.day_index_field().prepare(frame)
