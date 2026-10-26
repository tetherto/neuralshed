"""
Character-level text encoding shared by the text target extractor and the decoders.
"""

import json
import re
import typing as tp
from pathlib import Path

import numpy as np

_KEEP = re.compile(r"[^a-z' ]+")

#: Fixed charset for pooled training. Frozen so ``n_classes`` does not depend on which
#: collection was assembled, which is what keeps checkpoints portable across collections.
#: Order is the id order; ids start at 1 because 0 is the CTC blank.
CANONICAL_CHARSET_V1: tuple[str, ...] = (" ", "'") + tuple(
    chr(c) for c in range(ord("a"), ord("z") + 1)
)


def normalize_text(text: str) -> str:
    """Lower-case, keep only ``a-z`` / apostrophe / space, collapse runs of spaces."""
    return re.sub(r"\s+", " ", _KEEP.sub(" ", str(text).lower())).strip()


@tp.runtime_checkable
class Vocab(tp.Protocol):
    """What a CTC target encoder needs from a vocabulary. ``CharVocab`` is the only
    implementation today; a phoneme vocabulary (level 6) plugs in here unchanged."""

    BLANK: tp.ClassVar[int]

    @property
    def n_classes(self) -> int: ...

    def encode(self, text: str) -> np.ndarray: ...

    def decode(self, ids: tp.Iterable[int]) -> str: ...


class CharVocab:
    """Character <-> id map. Id 0 is the CTC blank, real characters start at 1."""

    BLANK = 0

    #: Name of the frozen charset this vocab was built from, or ``"fitted"``.
    version: str

    def __init__(self, chars: tp.Sequence[str], version: str = "fitted"):
        self.chars = list(chars)
        self.version = version
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}
        self.itos = {i + 1: c for i, c in enumerate(self.chars)}

    def __repr__(self) -> str:
        return f"CharVocab(version={self.version!r}, n_classes={self.n_classes})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CharVocab) and self.chars == other.chars

    @property
    def n_classes(self) -> int:
        return len(self.chars) + 1  # + blank

    @classmethod
    def canonical(cls, version: str = "v1") -> "CharVocab":
        """The frozen charset. Prefer this over :meth:`from_texts` for anything trained."""
        charsets = {"v1": CANONICAL_CHARSET_V1}
        if version not in charsets:
            raise ValueError(
                f"unknown charset version {version!r}; have {sorted(charsets)}"
            )
        return cls(charsets[version], version=version)

    @classmethod
    def from_texts(cls, texts: tp.Iterable[str]) -> "CharVocab":
        """Fit a vocab to the texts actually present. Vocabulary size then depends on the
        collection, so a checkpoint does not transfer. Use :meth:`canonical` instead."""
        seen: set[str] = set()
        for t in texts:
            seen.update(normalize_text(t))
        return cls(sorted(seen))

    def coverage(self, texts: tp.Iterable[str]) -> dict[str, int]:
        """Characters in ``texts`` this vocab cannot encode, mapped to how often they occur."""
        missing: dict[str, int] = {}
        for t in texts:
            for c in str(t):
                if c not in self.stoi:
                    missing[c] = missing.get(c, 0) + 1
        return missing

    def encode(self, text: str) -> np.ndarray:
        return np.array(
            [self.stoi[c] for c in normalize_text(text) if c in self.stoi], np.int64
        )

    def decode(self, ids: tp.Iterable[int]) -> str:
        return "".join(self.itos.get(int(i), "") for i in ids)


#: ARPAbet symbol set: the 39 CMUdict phonemes in alphabetical order, then ``|`` for
#: silence and word boundary. Ids start at 1 because 0 is the CTC blank, which makes this the
#: 41-class alphabet of the Willett/Card brain-to-text decoders.
ARPABET_V1: tuple[str, ...] = (
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F",
    "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S",
    "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "|",
)  # fmt: skip


class PhonemeVocab:
    """Phoneme <-> id map over whitespace-separated symbols. Id 0 is the CTC blank."""

    BLANK = 0

    version: str

    def __init__(self, symbols: tp.Sequence[str], version: str = "fitted"):
        self.symbols = list(symbols)
        self.version = version
        self.stoi = {s: i + 1 for i, s in enumerate(self.symbols)}
        self.itos = {i + 1: s for i, s in enumerate(self.symbols)}

    def __repr__(self) -> str:
        return f"PhonemeVocab(version={self.version!r}, n_classes={self.n_classes})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PhonemeVocab) and self.symbols == other.symbols

    @property
    def n_classes(self) -> int:
        return len(self.symbols) + 1  # + blank

    @classmethod
    def canonical(cls, version: str = "v1") -> "PhonemeVocab":
        """The frozen symbol set."""
        sets = {"v1": ARPABET_V1}
        if version not in sets:
            raise ValueError(
                f"unknown phoneme set version {version!r}; have {sorted(sets)}"
            )
        return cls(sets[version], version=version)

    def coverage(self, texts: tp.Iterable[str]) -> dict[str, int]:
        """Symbols in ``texts`` this vocab cannot encode, mapped to how often they occur."""
        missing: dict[str, int] = {}
        for t in texts:
            for s in str(t).split():
                if s not in self.stoi:
                    missing[s] = missing.get(s, 0) + 1
        return missing

    def encode(self, text: str) -> np.ndarray:
        """Ids of the space-separated symbols in ``text``. Unknown symbols raise."""
        ids = []
        for s in str(text).split():
            if s not in self.stoi:
                raise KeyError(f"phoneme {s!r} is not in {self!r}")
            ids.append(self.stoi[s])
        return np.array(ids, np.int64)

    def decode(self, ids: tp.Iterable[int]) -> str:
        return " ".join(self.itos[int(i)] for i in ids if int(i) in self.itos)


#: Word boundary symbol of ``ARPABET_V1`` (id 40), the Stanford speechBCI ``SIL``.
WORD_BOUNDARY = "|"


def _g2p_symbols(text: str, g2p: tp.Callable[[str], tp.Sequence[str]]) -> list[str]:
    """``g2p`` output as ``ARPABET_V1`` symbols: a ``WORD_BOUNDARY`` for every space it
    emits, stress digits dropped, anything else that is not a phoneme (punctuation) dropped."""
    out = []
    for p in g2p(text):
        if p == " ":
            out.append(WORD_BOUNDARY)
        p = re.sub(r"[0-9]", "", p)
        if re.match(r"[A-Z]+", p):
            out.append(p)
    return out


def g2p_phonemes(
    text: str,
    g2p: tp.Callable[[str], tp.Sequence[str]],
    *,
    join_split_apostrophes: bool = True,
    per_word: bool = False,
) -> str:
    """Dictionary pronunciation of ``text`` as a space-separated ``ARPABET_V1`` string.

    The Stanford speechBCI recipe (``mat_to_tfrecord.py``) that made card_2024's targets:
    punctuation stripped, lower-cased, ``g2p`` (a ``g2p_en.G2p`` instance) applied, stress
    digits dropped, a ``WORD_BOUNDARY`` at every space and one at the end.

    ``join_split_apostrophes``: rejoin a contraction the transcript writes with a space
    before its apostrophe (``"I 'll"`` -> ``"I'll"``) before ``g2p`` runs. Off, the
    detached piece is read as a word of its own (``"I 'll"`` -> ``"AY | | L EH L"``).

    ``per_word``: run ``g2p`` on each whitespace-separated word of the raw text on its
    own, with a ``WORD_BOUNDARY`` after every word. Punctuation is not stripped first:
    ``g2p`` reads a trailing ``?`` or ``.`` as a token of its own, which adds a second
    boundary (``"good?"`` -> ``"G UH D | |"``), and each word is tagged without its
    sentence context.
    """
    if per_word:
        if join_split_apostrophes:
            text = text.replace(" '", "'")
        out = []
        for word in text.split():
            out += _g2p_symbols(word, g2p) + [WORD_BOUNDARY]
        return " ".join(out) if out else WORD_BOUNDARY
    clean = re.sub(r"[^a-zA-Z\- \']", "", text).replace("--", "")
    if join_split_apostrophes:
        clean = clean.replace(" '", "'")
    clean = clean.lower()
    if not clean:
        return WORD_BOUNDARY
    return " ".join(_g2p_symbols(clean, g2p) + [WORD_BOUNDARY])


class PhonemeTable:
    """A shipped cue -> phonemes JSON table (``{"description", "g2p_en", "n_cues", "cues"}``).

    Datasets whose source gives only the prompt text build one once with :meth:`build`
    and read it at load time, so the rows never depend on ``g2p_en`` being installed.
    The header's ``description`` states what the targets are (a dictionary pronunciation
    of the prompt, never a measurement).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._payload: dict | None = None

    @property
    def payload(self) -> dict:
        if self._payload is None:
            with open(self.path) as f:
                self._payload = json.load(f)
        return self._payload

    @property
    def cues(self) -> dict[str, str]:
        return self.payload["cues"]

    def __len__(self) -> int:
        return len(self.cues)

    def __contains__(self, cue: object) -> bool:
        return cue in self.cues

    def __getitem__(self, cue: str) -> str:
        if cue not in self.cues:
            raise KeyError(f"{cue!r} is not in {self.path.name}, rebuild the table")
        return self.cues[cue]

    @classmethod
    def build(
        cls,
        cues: tp.Iterable[str],
        path: str | Path,
        description: str,
        *,
        join_split_apostrophes: bool = True,
        per_word: bool = False,
    ) -> "PhonemeTable":
        """Run :func:`g2p_phonemes` (with the given options) over ``cues``, check every
        symbol against ``PhonemeVocab.canonical()`` and write the table, the options in
        its header. Needs the ``g2p`` extra."""
        import importlib.metadata

        import nltk
        from g2p_en import G2p

        for package in ("averaged_perceptron_tagger_eng", "cmudict"):
            nltk.download(package, quiet=True)
        g2p = G2p()
        options = dict(join_split_apostrophes=join_split_apostrophes, per_word=per_word)
        table = {cue: g2p_phonemes(cue, g2p, **options) for cue in sorted(set(cues))}
        missing = PhonemeVocab.canonical().coverage(table.values())
        if missing:
            raise ValueError(f"symbols outside PhonemeVocab v1: {missing}")
        payload = {
            "description": description,
            "g2p_en": importlib.metadata.version("g2p_en"),
            "g2p_options": options,
            "n_cues": len(table),
            "cues": table,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)
            f.write("\n")
        return cls(path)


#: Named text-normalization schemes. A scheme is the function that maps raw text to the form
#: a vocabulary can encode, so a scheme and a vocab version are the same fact under two names:
#: ``"v1"`` here is ``normalize_text``, which keeps exactly the characters in
#: ``CANONICAL_CHARSET_V1``. Keeping them paired stops the normalizer and the model's output
#: alphabet from drifting apart.
TEXT_SCHEMES: dict[str, tp.Callable[[str], str]] = {"v1": normalize_text}


def n_repeated_labels(ids: tp.Sequence[int] | np.ndarray) -> int:
    """Count adjacent duplicates in a CTC target.

    Every repeat needs a blank between the two emissions, so it costs one extra input frame.
    """
    arr = np.asarray(ids)
    if arr.size < 2:
        return 0
    return int((arr[1:] == arr[:-1]).sum())


def ctc_min_input_length(ids: tp.Sequence[int] | np.ndarray) -> int:
    """Shortest emission length that can produce ``ids`` under CTC.

    ``len(ids) + n_repeats``, not ``len(ids)``. Trials below this contribute an infinite loss,
    which ``zero_infinity=True`` silently turns into a zero gradient instead of an error.
    """
    return len(ids) + n_repeated_labels(ids)


__all__ = [
    "normalize_text",
    "TEXT_SCHEMES",
    "CANONICAL_CHARSET_V1",
    "Vocab",
    "CharVocab",
    "ARPABET_V1",
    "PhonemeVocab",
    "WORD_BOUNDARY",
    "g2p_phonemes",
    "PhonemeTable",
    "n_repeated_labels",
    "ctc_min_input_length",
]
