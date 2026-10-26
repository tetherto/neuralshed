"""Shared local Study behavior for neuralshed datasets."""

from __future__ import annotations

import importlib
import typing as tp

import numpy as np
import pandas as pd
from neuralset.events import etypes
from neuralset.events import study as ns_study

from neuralshed.datasets import participants
from neuralshed.events import DenseArrayEvent

# Timeline-dict keys every neuralshed timeline carries, with their defaults. A Study sets
# them only where the released data are bad (see docs/standardization.md, valid). They
# describe the data rather than identify the timeline, so they stay out of the timeline id.
# A Study may also set them on single event rows; those rows keep their own values and
# the timeline value fills the rest.
_TIMELINE_DEFAULTS: dict[str, tp.Any] = {"valid": True, "valid_description": ""}


class _NeuralshedStudy(ns_study.Study):
    """NeuroAI Study with neuralshed provenance and validation conventions."""

    dataset_id: tp.ClassVar[str] = ""
    _participant_maps: tp.ClassVar[dict[str, dict[str, str]]] = {}

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        if not self.dataset_id:
            raise ValueError(
                f"{type(self).__name__} must declare a non-empty dataset_id"
            )
        self.timelines.infra = None

    @classmethod
    def _participant_map(cls) -> dict[str, str]:
        """Return the dataset subject-label to project participant-id mapping."""
        if cls.dataset_id in cls._participant_maps:
            return cls._participant_maps[cls.dataset_id]

        module_name = f"neuralshed.datasets.{cls.dataset_id}.metadata"
        try:
            metadata_module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name is None or not (
                error.name == module_name or module_name.startswith(f"{error.name}.")
            ):
                raise
            raise ValueError(
                f"{cls.__name__}.dataset_id={cls.dataset_id!r} has no metadata module "
                f"{module_name!r}"
            ) from error

        metadata = getattr(metadata_module, "DATASET_METADATA", None)
        if not isinstance(metadata, dict):
            raise ValueError(f"{module_name} must define DATASET_METADATA as a dict")
        if metadata.get("id") != cls.dataset_id:
            raise ValueError(
                f"{module_name} declares id={metadata.get('id')!r}, expected "
                f"{cls.dataset_id!r}"
            )

        mapping: dict[str, str] = {}
        for subject in metadata.get("subjects", []):
            label = str(subject.get("label", ""))
            participant_id = str(subject.get("participant_id", ""))
            if not label or not participant_id:
                raise ValueError(
                    f"{module_name} subjects need non-empty label and participant_id"
                )
            if label in mapping:
                raise ValueError(f"{module_name} has duplicate subject label {label!r}")
            if participants.participant_by_id(participant_id) is None:
                raise ValueError(
                    f"{module_name} subject {label!r} references unknown participant_id "
                    f"{participant_id!r}"
                )
            mapping[label] = participant_id

        if not mapping:
            raise ValueError(f"{module_name} declares no subjects")
        cls._participant_maps[cls.dataset_id] = mapping
        return mapping

    def _participant_id_for(self, subject: tp.Any) -> str:
        label = str(subject)
        try:
            return self._participant_map()[label]
        except KeyError as error:
            known = ", ".join(sorted(self._participant_map()))
            raise ValueError(
                f"{type(self).__name__} timeline subject {label!r} is not declared in "
                f"{self.dataset_id} metadata; expected one of: {known}"
            ) from error

    def _load_derived(
        self,
        timeline: dict[str, tp.Any],
        *,
        operation: str,
        sources: list[str],
        rate: float,
    ) -> "np.ndarray":
        """SpecialLoader target for a row :class:`DeriveStream` appended.

        The derived row carries its source rows' own loaders, so the arrays are read the same
        way they would be for the sources themselves and the derivation sees the whole
        timeline. It lives on the Study base rather than in the transform because a
        SpecialLoader can only name a method of a registered Study.
        """
        from neuralset.events.utils import SpecialLoader

        from neuralshed.derivations import DERIVATIONS

        if operation not in DERIVATIONS:
            raise KeyError(
                f"unknown derivation {operation!r}; have {sorted(DERIVATIONS)}"
            )
        arrays = [
            np.asarray(SpecialLoader.from_json(source).load(), dtype=np.float32)
            for source in sources
        ]
        return DERIVATIONS[operation].apply(arrays, rate=rate)

    def _load_rescaled(
        self,
        timeline: dict[str, tp.Any],
        *,
        inner: str,
        stat_rows: list[list[int]],
        guard: str = "eps",
        divisor: tp.Optional[list[float]] = None,
        mean: tp.Optional[list[float]] = None,
    ) -> "np.ndarray":
        """SpecialLoader target for a row :class:`RescaleNeural` rewrote.

        Reads the payload through the row's own raw loader, fits per-channel
        ``(mean, std)`` over the pooled stat bins, and divides the whole
        stored array, so every window the segmenter later cuts is already
        scaled.
        """
        from neuralset.events.utils import SpecialLoader

        from neuralshed import scaling

        x = np.asarray(SpecialLoader.from_json(inner).load())
        if mean is not None:
            mu = np.asarray(mean, dtype=x.dtype).reshape((-1,) + (1,) * (x.ndim - 1))
            if mu.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Mean has {mu.shape[0]} channels, payload has {x.shape[0]}"
                )
            return scaling.apply_rescale(x, mu, guard=guard, divisor=divisor)
        mu, sd = scaling.fit_rescale(x, stat_rows)
        return scaling.apply_rescale(x, mu, sd, guard, divisor=divisor)

    def _all_timelines(self) -> list[dict[str, tp.Any]]:
        if self._timelines is not None:
            return self._timelines
        timelines = super()._all_timelines()
        for timeline in timelines:
            for key, default in _TIMELINE_DEFAULTS.items():
                timeline.setdefault(key, default)
        return timelines

    def _to_timeline_string(self, timeline: dict[str, tp.Any]) -> str:
        identity = {k: v for k, v in timeline.items() if k not in _TIMELINE_DEFAULTS}
        return super()._to_timeline_string(identity)

    def _load_one(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        flags = {k: v for k, v in timeline.items() if k in _TIMELINE_DEFAULTS}
        events = super()._load_one(
            {k: v for k, v in timeline.items() if k not in flags}
        )
        for key, value in flags.items():
            if (
                key in events.columns
            ):  # row-level flags win, the timeline fills the rest
                events[key] = events[key].where(events[key].notna(), value)
            else:
                events[key] = value
            events[key] = events[key].astype(type(value))
        # The base broadcasts timeline-dict values as strings; give them their types back.
        for key, value in timeline.items():
            if key in ("subject", "path", "timeline") or key not in events.columns:
                continue
            if (events[key].astype(str) == str(value)).all():
                events[key] = pd.Series([value] * len(events), index=events.index)
        participant_id = self._participant_id_for(timeline["subject"])
        events.loc[:, "dataset_id"] = self.dataset_id
        events.loc[:, "participant_id"] = participant_id
        events = self._postprocess_timeline(events, timeline)
        if not isinstance(events, pd.DataFrame):
            raise TypeError("_postprocess_timeline must return a pandas DataFrame")
        self._validate_timeline_events(events, timeline, participant_id)
        if "n_features" in events.columns:
            events["n_features"] = pd.to_numeric(
                events["n_features"], errors="coerce"
            ).astype("Int64")
        return events

    def _postprocess_timeline(
        self,
        events: pd.DataFrame,
        timeline: dict[str, tp.Any],
    ) -> pd.DataFrame:
        """Customize an already standardized and provenance-enriched timeline."""
        return events

    def _validate_timeline_events(
        self,
        events: pd.DataFrame,
        timeline: dict[str, tp.Any],
        participant_id: str,
    ) -> None:
        """Validate the minimum invariants expected from every neuralshed timeline."""
        if events.empty:
            raise ValueError(
                "A neuralshed Study timeline cannot contain an empty event frame"
            )

        required = {
            "type",
            "start",
            "duration",
            "timeline",
            "study",
            "subject",
            "dataset_id",
            "participant_id",
        }
        missing = sorted(required.difference(events.columns))
        if missing:
            raise ValueError(f"Timeline event frame is missing columns: {missing}")
        null_columns = sorted(col for col in required if events[col].isna().any())
        if null_columns:
            raise ValueError(f"Timeline event frame has null values in: {null_columns}")

        expected = {
            "study": type(self).__name__,
            "subject": f"{type(self).__name__}/{timeline['subject']}",
            "dataset_id": self.dataset_id,
            "participant_id": participant_id,
        }
        for column, value in expected.items():
            actual = set(events[column].astype(str).unique())
            if actual != {value}:
                raise ValueError(
                    f"Timeline event frame needs {column}={value!r}; got {sorted(actual)!r}"
                )
        if events["timeline"].astype(str).nunique() != 1:
            raise ValueError(
                "Timeline event frame must contain exactly one timeline id"
            )

        dense_types = {
            name
            for name, event_type in etypes.Event._CLASSES.items()
            if issubclass(event_type, DenseArrayEvent)
        }
        dense = events.loc[events["type"].isin(dense_types)]
        if not dense.empty:
            if "frequency" not in dense.columns:
                raise ValueError("Dense-array events need a frequency column")
            durations = pd.to_numeric(dense["duration"], errors="coerce")
            frequencies = pd.to_numeric(dense["frequency"], errors="coerce")
            invalid = (
                durations.isna()
                | frequencies.isna()
                | (durations <= 0)
                | (frequencies <= 0)
            )
            if invalid.any():
                bad_types = sorted(dense.loc[invalid, "type"].unique())
                raise ValueError(
                    "Dense-array events need positive duration and frequency; "
                    f"invalid types: {bad_types}"
                )
            if "n_features" not in dense.columns or dense["n_features"].isna().any():
                bad_types = sorted(
                    dense.loc[
                        dense.get(
                            "n_features", pd.Series(pd.NA, index=dense.index)
                        ).isna(),
                        "type",
                    ].unique()
                )
                raise ValueError(
                    "Dense-array events need n_features; missing for types: "
                    f"{bad_types}"
                )
            values = pd.to_numeric(dense["n_features"], errors="coerce")
            invalid_features = values.isna() | (values <= 0) | (values % 1 != 0)
            if invalid_features.any():
                bad_types = sorted(dense.loc[invalid_features, "type"].unique())
                raise ValueError(
                    "Dense-array n_features values must be positive integers; "
                    f"invalid types: {bad_types}"
                )


# The private class name keeps this shared base out of NeuroAI's concrete Study registry.
NeuralshedStudy = _NeuralshedStudy

__all__ = ["NeuralshedStudy"]
