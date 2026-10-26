"""Training + evaluation for the pooled brain-to-text model, using the neuroai stack.

Three Lightning modules mirror the experiments:

- :class:`CTCModule` -- supervised neural -> characters with ``nn.CTCLoss`` (Option i).
- :class:`MaskedPretrainModule` -- self-supervised masked-embedding reconstruction (Option ii,
  phase 1); its trained ``model`` is then handed to :class:`CTCModule` to fine-tune.
- :class:`RegressionModule` -- supervised neural -> a dense target, for a collection decoding a
  kinematic rather than text.

Decoding is greedy CTC collapse; the reported metric is the character error rate (edit distance
normalized by reference length), broken down per subject so pooled-vs-isolated is directly
comparable. The dense-target metric is R2, broken down per member for the same reason.
"""

import typing as tp

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import lightning as L

from neuralshed.models.model import BrainToTextCTC
from neuralshed.text import CharVocab

# ===========================================================================
# Decoding + character error rate
# ===========================================================================


def greedy_decode(
    log_probs: torch.Tensor, lengths: torch.Tensor, vocab: CharVocab
) -> list[str]:
    """Argmax then collapse repeats and drop blanks (standard greedy CTC decode)."""
    ids = log_probs.argmax(-1)  # (B, T)
    out = []
    for row, n in zip(ids, lengths):
        seq, prev = [], -1
        for t in row[: int(n)].tolist():
            if t != prev and t != vocab.BLANK:
                seq.append(t)
            prev = t
        out.append(vocab.decode(seq))
    return out


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_error_rate(preds: tp.Sequence[str], refs: tp.Sequence[str]) -> float:
    """Corpus CER: total edits / total reference characters (in percent)."""
    edits = sum(_edit_distance(p, r) for p, r in zip(preds, refs))
    ref_len = sum(len(r) for r in refs) or 1
    return 100.0 * edits / ref_len


# ===========================================================================
# Data loaders
# ===========================================================================


def make_loader(
    trials, split: str | None, batch_size: int = 32, shuffle: bool = True, seed: int = 0
) -> DataLoader:
    """Loader over ``BitTrial`` objects, batches homogeneous in recording.

    Thin wrapper over ``BitDataBundle.loader``; prefer that when you have a bundle. Both
    decoders take the batches this produces.
    """
    from neuralshed.models.bit.bundle import BitTrialDataset, collate_bit
    from neuralshed.models.data import SessionBatchSampler

    ds = BitTrialDataset(trials, split=split)
    sampler = SessionBatchSampler(
        ds, batch_size=batch_size, shuffle=shuffle, seed=seed, key_attr="group_key"
    )
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate_bit)


# ===========================================================================
# Masking for self-supervised pretraining
# ===========================================================================


def span_mask(
    input_lengths: torch.Tensor,
    T: int,
    mask_prob: float = 0.3,
    span: int = 5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Boolean ``(B, T)`` mask of contiguous spans covering ~``mask_prob`` of each item's bins."""
    B = len(input_lengths)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        valid = int(input_lengths[b])
        target = int(mask_prob * valid)
        masked = 0
        while masked < target:
            start = int(
                torch.randint(0, max(1, valid - span), (1,), generator=generator)
            )
            mask[b, start : start + span] = True
            masked += span
    return mask


# ===========================================================================
# Lightning modules
# ===========================================================================


class CTCModule(L.LightningModule):
    """Supervised neural -> characters. Logs train loss and validation CER."""

    def __init__(
        self,
        model: BrainToTextCTC,
        vocab: CharVocab,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.model = model
        self.vocab = vocab
        self.lr = lr
        self.weight_decay = weight_decay
        self.ctc = nn.CTCLoss(blank=vocab.BLANK, zero_infinity=True)
        self._val: list[tuple[str, str, str]] = []  # (subject, pred, ref)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, _):
        log_probs = self.model(batch)  # (B, T', C)
        loss = self.ctc(
            log_probs.permute(1, 0, 2),
            batch["targets"],
            self.model.out_lengths(batch["input_lengths"]),
            batch["target_lengths"],
        )
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch["texts"]))
        return loss

    def validation_step(self, batch, _):
        log_probs = self.model(batch)
        preds = greedy_decode(
            log_probs, self.model.out_lengths(batch["input_lengths"]), self.vocab
        )
        subj = batch["participant_idx"].tolist()
        for s, p, r in zip(subj, preds, batch["texts"]):
            self._val.append((s, p, r))

    def on_validation_epoch_end(self):
        if self._val:
            preds = [p for _, p, _ in self._val]
            refs = [r for _, _, r in self._val]
            self.log("val_cer", char_error_rate(preds, refs), prog_bar=True)
        self._val.clear()

    def configure_optimizers(self):
        # cosine annealing keeps the LR high through mid-training, so the model still has enough
        # step size to escape the blank collapse (OneCycle decays too early for that)
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, self.trainer.max_epochs)
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }


class MaskedPretrainModule(L.LightningModule):
    """Self-supervised phase 1: reconstruct masked read-in embeddings from context (MSE)."""

    def __init__(
        self,
        model: BrainToTextCTC,
        lr: float = 3e-4,
        mask_prob: float = 0.3,
        span: int = 5,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.mask_prob = mask_prob
        self.span = span

    def _step(self, batch, name):
        from neuralshed.models.bit.bundle import PRIMARY_STREAM

        spikes = batch["arrays"][PRIMARY_STREAM]
        mask = span_mask(
            batch["input_lengths"], spikes.shape[-1], self.mask_prob, self.span
        ).to(spikes.device)
        pred, target, mask = self.model.masked_reconstruct(batch, mask)
        loss = nn.functional.mse_loss(pred[mask], target[mask])
        self.log(f"{name}_loss", loss, prog_bar=True, batch_size=len(batch["texts"]))
        return loss

    def training_step(self, batch, _):
        return self._step(batch, "train")

    def validation_step(self, batch, _):
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


class _R2:
    """Streaming R2 accumulator: ``1 - SS_res / SS_tot``.

    Sums rather than stored arrays, so a whole split costs four floats per member. The
    denominator is the variance of the *true* values, which is what makes 0 mean "no better
    than always predicting the mean velocity" and a negative value mean worse than that.
    """

    def __init__(self) -> None:
        self.ss_res = self.sum_y = self.sum_y2 = 0.0
        self.n = 0

    def update(self, pred: torch.Tensor, true: torch.Tensor) -> None:
        residual = (true - pred).double()
        y = true.double()
        self.ss_res += float((residual * residual).sum())
        self.sum_y += float(y.sum())
        self.sum_y2 += float((y * y).sum())
        self.n += y.numel()

    def value(self) -> float:
        if self.n == 0:
            return float("nan")
        ss_tot = self.sum_y2 - self.sum_y**2 / self.n
        return float("nan") if ss_tot <= 0 else 1.0 - self.ss_res / ss_tot


def patch_mask(model, batch, n_patches: int, device) -> torch.Tensor:
    """``(B, P)`` True on patches an item actually produced.

    The same arithmetic the patcher uses, so a masked loss never counts a patch the trunk
    filled from padding.
    """
    lengths = model.out_lengths(batch["input_lengths"]).to(device)
    return torch.arange(n_patches, device=device)[None] < lengths[:, None]


class RegressionModule(L.LightningModule):
    """Supervised neural -> a dense target. Logs masked MSE and validation R2 per member."""

    def __init__(self, model, lr: float = 3e-4, weight_decay: float = 1e-5):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self._val: dict[str, _R2] = {}

    def forward(self, batch):
        return self.model(batch)

    def _step(self, batch):
        """Prediction, target in source units, patch mask, and the loss between them.

        The target is pooled to patch rate and z-scored on the way in, because that is the
        space the model predicts in. ``pooled`` is returned unnormalized as well, since the
        metric wants source units.
        """
        pred = self.model(batch)  # (B, P, C), z-score space
        pooled = self.model.patcher.pool_target(
            batch["target"], batch["input_lengths"]
        ).to(pred.device)
        mask = patch_mask(self.model, batch, pred.shape[1], pred.device)
        target = self.model.normalize_target(pooled, batch)
        return pred, pooled, mask, nn.functional.mse_loss(pred[mask], target[mask])

    def training_step(self, batch, _):
        loss = self._step(batch)[3]
        self.log(
            "train_loss", loss, prog_bar=True, batch_size=len(batch["input_lengths"])
        )
        return loss

    def validation_step(self, batch, _):
        pred, pooled, mask, loss = self._step(batch)
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=len(batch["input_lengths"])
        )
        # R2 in the target's own units, never in z-space: two sessions of one member are
        # normalized by different scales, so their residuals are only commensurable once
        # both are back in source units.
        acc = self._val.setdefault(batch["members"][0], _R2())
        acc.update(self.model.denormalize(pred, batch)[mask], pooled[mask])

    def on_validation_epoch_end(self):
        for member, acc in self._val.items():
            self.log(f"val_r2/{member}", acc.value(), prog_bar=True)
        self._val.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, self.trainer.max_epochs)
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }


# ===========================================================================
# Trainer helpers + evaluation
# ===========================================================================


class _History(L.Callback):
    """Collect scalar metrics per validation epoch for easy plotting in the notebooks."""

    def __init__(self):
        self.rows: list[dict] = []

    def on_validation_epoch_end(self, trainer, _):
        m = {k: float(v) for k, v in trainer.callback_metrics.items()}
        m["epoch"] = trainer.current_epoch
        self.rows.append(m)


def _trainer(
    max_epochs: int, accelerator: str = "cpu", patience: int | None = None
) -> tuple[L.Trainer, _History]:
    hist = _History()
    callbacks: list[L.Callback] = [hist]
    if patience is not None:
        callbacks.append(
            L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=patience)
        )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )
    return trainer, hist


def train_ctc(
    model,
    vocab,
    train_loader,
    val_loader=None,
    *,
    max_epochs=15,
    lr=3e-4,
    accelerator="cpu",
) -> tuple[CTCModule, list[dict]]:
    """Fit a supervised CTC decoder; returns the module and its per-epoch metric history."""
    module = CTCModule(model, vocab, lr=lr)
    trainer, hist = _trainer(max_epochs, accelerator)
    trainer.fit(module, train_loader, val_loader)
    return module, hist.rows


def pretrain_masked(
    model,
    train_loader,
    val_loader=None,
    *,
    max_epochs=15,
    lr=3e-4,
    mask_prob=0.3,
    accelerator="cpu",
) -> tuple[BrainToTextCTC, list[dict]]:
    """Self-supervised phase 1; returns the pretrained ``model`` (read-in + backbone warmed up)."""
    module = MaskedPretrainModule(model, lr=lr, mask_prob=mask_prob)
    trainer, hist = _trainer(max_epochs, accelerator)
    trainer.fit(module, train_loader, val_loader)
    return module.model, hist.rows


def train_regression(
    model,
    train_loader,
    val_loader=None,
    *,
    max_epochs=15,
    lr=3e-4,
    accelerator="cpu",
    weight_decay=1e-5,
    patience=None,
) -> tuple["RegressionModule", list[dict]]:
    """Fit a dense-target decoder; returns the module and its per-epoch metric history.

    ``patience`` stops on ``val_loss`` after that many epochs without improvement, for a
    comparison between targets that should not turn on which one memorizes fastest.
    """
    module = RegressionModule(model, lr=lr, weight_decay=weight_decay)
    trainer, hist = _trainer(max_epochs, accelerator, patience=patience)
    trainer.fit(module, train_loader, val_loader)
    return module, hist.rows


@torch.no_grad()
def evaluate_by_member(model, loader, device="cpu") -> dict[str, float]:
    """R2 per collection member over ``loader``, each in its own target units.

    There is deliberately no pooled ``"overall"``: the members publish the cursor in different
    units (arbitrary units per second, pixels per second, screen heights per second), so one
    variance ratio spanning them would not describe anything.
    """
    model = model.to(device).eval()
    buckets: dict[str, _R2] = {}
    for batch in loader:
        batch = _to_device(batch, device)
        pred = model(batch)
        pooled = model.patcher.pool_target(batch["target"], batch["input_lengths"]).to(
            device
        )
        mask = patch_mask(model, batch, pred.shape[1], device)
        acc = buckets.setdefault(batch["members"][0], _R2())
        acc.update(model.denormalize(pred, batch)[mask], pooled[mask])
    return {member: acc.value() for member, acc in buckets.items()}


@torch.no_grad()
def evaluate_by_subject(
    model, loader, vocab, manifest, device="cpu"
) -> dict[str, float]:
    """Character error rate per participant (plus an ``"overall"`` key) over ``loader``."""
    model = model.to(device).eval()
    buckets: dict[int, tuple[list[str], list[str]]] = {}
    for batch in loader:
        batch = _to_device(batch, device)
        log_probs = model(batch)
        preds = greedy_decode(
            log_probs, model.out_lengths(batch["input_lengths"]), vocab
        )
        for s, p, r in zip(batch["participant_idx"].tolist(), preds, batch["texts"]):
            preds_l, refs_l = buckets.setdefault(s, ([], []))
            preds_l.append(p)
            refs_l.append(r)
    out = {
        manifest.participants[s]: char_error_rate(p, r) for s, (p, r) in buckets.items()
    }
    all_p = [p for p, _ in buckets.values() for p in p]
    all_r = [r for _, r in buckets.values() for r in r]
    out["overall"] = char_error_rate(all_p, all_r)
    return out


def _to_device(batch, device):
    """Move a ``BitBatch`` (or a plain dict of tensors) to ``device``."""
    if hasattr(batch, "to"):
        return batch.to(device)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


__all__ = [
    "greedy_decode",
    "char_error_rate",
    "make_loader",
    "span_mask",
    "CTCModule",
    "MaskedPretrainModule",
    "train_ctc",
    "pretrain_masked",
    "evaluate_by_subject",
]
