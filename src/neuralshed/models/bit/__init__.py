"""BIT: a CTC brain-to-text decoder trained on a collection of Studies.

Ported from the vendored upstream in ``bit/``, which is left untouched as the reference to
diff against. What changed and why is documented in :mod:`~neuralshed.models.bit.model`;
the short version is that the trainable read-in moved ahead of the patcher, so raw channel
count stops reaching the trunk and a new recording costs one projection per stream instead
of a config edit.

    from neuralshed.models.bit import BitDataBundle, build_bit_model

    bundle = BitDataBundle.from_collection("speech-text")
    model = build_bit_model(bundle.manifest, bundle.vocab, bundle.state)
    trainer.fit(model, bundle.train_loader(), bundle.val_loader())
"""

from neuralshed.manifest import CollectionManifest, LayoutSpec
from neuralshed.models.bit.bundle import (
    BitBatch,
    BitDataBundle,
    BitTrial,
    BitTrialDataset,
    collate_bit,
)
from neuralshed.models.bit.config import BitConfig
from neuralshed.models.bit.model import (
    BitCrossEncoder,
    BitPatcher,
    BitReadIn,
    BitTrunk,
    build_bit_model,
)
from neuralshed.models.bit.preprocess import (
    BitPreprocessingState,
    ChannelStats,
    NormalizationPolicy,
)
from neuralshed.models.bit.validate import (
    BitReport,
    validate_collection_for_bit,
    validate_manifest,
)

__all__ = [
    # data
    "BitDataBundle",
    "BitTrial",
    "BitBatch",
    "BitTrialDataset",
    "collate_bit",
    # manifest (generic, re-exported for convenience)
    "CollectionManifest",
    "LayoutSpec",
    # preprocessing
    "BitPreprocessingState",
    "NormalizationPolicy",
    "ChannelStats",
    # model
    "BitConfig",
    "BitCrossEncoder",
    "BitReadIn",
    "BitPatcher",
    "BitTrunk",
    "build_bit_model",
    # preflight
    "validate_collection_for_bit",
    "validate_manifest",
    "BitReport",
]
