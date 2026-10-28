"""Every dataset's ``Study``, in one import.

``ns.Step`` resolves a step from its ``name`` discriminator against the classes that have
registered themselves, and a class registers when its module is imported. Import this before
building a ``Step`` from a stored config, or the config's study names will not resolve.

Alongside :mod:`neuralshed.transforms`, which registers ``CombineStudies`` and
``DummyTransform``, this covers every step a collection config can name.
"""

from .datasets.card_2024.study import Card2024Brain2Text
from .datasets.deo_2024.study import Deo2024Bimanual
from .datasets.fan_h1.study import FanFalconH1
from .datasets.fan_2023.study import Fan2023Handwriting
from .datasets.gusman_2025.study import Gusman2025DragAndDrop
from .datasets.jude_2026_speech.study import Jude2026IntendedSpeech
from .datasets.jude_2026_typing.study import Jude2026Typing
from .datasets.karpowicz_2025.study import Karpowicz2025CursorStability
from .datasets.kunz_2025.study import Kunz2025InnerSpeech
from .datasets.singer_2025.study import Singer2025CursorControl
from .datasets.wairagkar_2025.study import Wairagkar2025Brain2Voice
from .datasets.willet_2021.study import Willett2021Handwriting
from .datasets.willet_2023.study import Willett2023Sentences
from .datasets.willsey_2025.study import Willsey2025FingerDecoding
from .datasets.wilson_2025.study import Wilson2025CursorRecalibration

__all__ = [
    "Card2024Brain2Text",
    "Deo2024Bimanual",
    "FanFalconH1",
    "Fan2023Handwriting",
    "Gusman2025DragAndDrop",
    "Jude2026IntendedSpeech",
    "Jude2026Typing",
    "Karpowicz2025CursorStability",
    "Kunz2025InnerSpeech",
    "Singer2025CursorControl",
    "Wairagkar2025Brain2Voice",
    "Willett2021Handwriting",
    "Willett2023Sentences",
    "Willsey2025FingerDecoding",
    "Wilson2025CursorRecalibration",
]
