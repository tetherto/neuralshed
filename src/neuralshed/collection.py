"""A NeuroAI-native chain with collection structure and layout guarantees."""

from __future__ import annotations

import typing as tp

import neuralset as ns

from neuralshed.transforms import AssignDataLayouts, CombineStudies


class _CollectionStep(ns.Step):
    """MRO anchor that keeps Collection from replacing NeuroAI's default Chain type."""


class Collection(ns.Chain, _CollectionStep):
    """A collection pipeline ending in validated dense-data layout assignment.

    Collection decisions remain explicit steps, so NeuroAI serialization and cache identity
    include them. The class adds only structural guarantees: fan-in first, layout assignment
    last, and exactly one layout-assignment transform in the top-level chain.
    """

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        sequence = (
            list(self.steps.values())
            if isinstance(self.steps, dict)
            else list(self.steps)
        )
        if not isinstance(sequence[0], CombineStudies):
            raise ValueError("Collection must start with CombineStudies")
        layout_steps = [
            step for step in sequence if isinstance(step, AssignDataLayouts)
        ]
        if len(layout_steps) != 1:
            raise ValueError(
                "Collection must contain exactly one top-level AssignDataLayouts step"
            )
        if sequence[-1] is not layout_steps[0]:
            raise ValueError("Collection must end with AssignDataLayouts")

    @property
    def layout_step(self) -> AssignDataLayouts:
        """The final transform carrying this collection's layout policy."""
        step = (
            list(self.steps.values())[-1]
            if isinstance(self.steps, dict)
            else self.steps[-1]
        )
        assert isinstance(step, AssignDataLayouts)
        return step


__all__ = ["Collection"]
