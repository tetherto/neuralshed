"""Extensible cross-subject brain-to-text modeling.

Shared building blocks for the experiment notebooks in this folder:

- ``01_supervised_multitask.ipynb`` -- supervised pooled training vs isolated models.
- ``02_pretrain_finetune.ipynb``    -- self-supervised pretrain, then fine-tune.
- ``04_bit_pipeline.ipynb``         -- the BIT CTC decoder over the same collection.

One data path feeds every model here. A collection becomes a
:class:`~neuralshed.models.bit.BitDataBundle`, and both decoders are built from that
bundle's manifest and trained on its batches:

    from neuralshed.models import BrainToTextCTC, speech_text_collection, train_ctc
    from neuralshed.models.bit import BitDataBundle, build_bit_model

    bundle = BitDataBundle.from_collection(speech_text_collection())
    gru    = BrainToTextCTC.from_manifest(bundle.manifest, bundle.vocab.n_classes)
    bit    = build_bit_model(bundle.manifest, bundle.vocab, bundle.state)
    train_ctc(gru, bundle.vocab, bundle.train_loader(), bundle.val_loader())

``build_pooled_dataset`` and its ``DatasetSpec`` / ``Trial`` / ``SessionRegistry`` /
``collate_ctc`` supporting cast are gone. They were a second path from Studies to tensors with
their own fitted vocabulary, their own splits and their own normalization, which made any
comparison between the two decoders a comparison of pipelines rather than of models.
"""

from neuralshed.models.data import (
    MOTOR_CURSOR_INPUTS,
    MOTOR_CURSOR_INTENT_TARGETS,
    MOTOR_CURSOR_DERIVATIONS,
    MOTOR_CURSOR_MEMBER_QUERIES,
    MOTOR_CURSOR_MEMBER_QUERIES_CLOSED_LOOP,
    MOTOR_CURSOR_MEMBERS,
    MOTOR_CURSOR_PARTICIPANTS,
    MOTOR_CURSOR_REGIONS,
    MOTOR_CURSOR_TARGETS,
    SPEECH_TEXT_INPUTS,
    SPEECH_TEXT_MEMBERS,
    SPEECH_TEXT_PARTICIPANTS,
    SPEECH_TEXT_REGIONS,
    SPEECH_TEXT_TARGETS,
    SessionBatchSampler,
    build_study,
    collection_identities,
    motor_cursor_collection,
    motor_cursor_identities,
    speech_text_collection,
)
from neuralshed.models.model import Backbone, BrainToTextCTC, SessionReadIn
from neuralshed.models.training import (
    CTCModule,
    MaskedPretrainModule,
    RegressionModule,
    char_error_rate,
    evaluate_by_member,
    evaluate_by_subject,
    greedy_decode,
    make_loader,
    pretrain_masked,
    span_mask,
    train_ctc,
    train_regression,
)
from neuralshed.text import CharVocab, ctc_min_input_length, normalize_text

__all__ = [
    # collection recipes
    "SPEECH_TEXT_MEMBERS",
    "SPEECH_TEXT_INPUTS",
    "SPEECH_TEXT_TARGETS",
    "SPEECH_TEXT_REGIONS",
    "SPEECH_TEXT_PARTICIPANTS",
    "MOTOR_CURSOR_MEMBERS",
    "MOTOR_CURSOR_MEMBER_QUERIES",
    "MOTOR_CURSOR_MEMBER_QUERIES_CLOSED_LOOP",
    "MOTOR_CURSOR_DERIVATIONS",
    "MOTOR_CURSOR_INPUTS",
    "MOTOR_CURSOR_TARGETS",
    "MOTOR_CURSOR_INTENT_TARGETS",
    "MOTOR_CURSOR_REGIONS",
    "MOTOR_CURSOR_PARTICIPANTS",
    "build_study",
    "speech_text_collection",
    "collection_identities",
    "motor_cursor_collection",
    "motor_cursor_identities",
    "SessionBatchSampler",
    # text
    "CharVocab",
    "ctc_min_input_length",
    "normalize_text",
    # model
    "BrainToTextCTC",
    "SessionReadIn",
    "Backbone",
    # training
    "CTCModule",
    "MaskedPretrainModule",
    "greedy_decode",
    "char_error_rate",
    "make_loader",
    "span_mask",
    "train_ctc",
    "pretrain_masked",
    "evaluate_by_subject",
    "RegressionModule",
    "train_regression",
    "evaluate_by_member",
]
