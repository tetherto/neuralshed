"""
PyTorch Lightning Trainer for BIT (Brain-to-Text) Model Training.

This module provides a PyTorch Lightning wrapper for training any HuggingFace AutoModel
on neural decoding tasks with CTC loss. It includes:
- Data augmentation (white noise, constant offset)
- CTC loss computation
- Character Error Rate (CER) metrics per dataset
- Learning rate scheduling with warmup and cosine annealing
- Full support for mixed precision (16-bit) training for faster performance
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import lightning as pl
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from edit_distance import SequenceMatcher

from transformers import AutoProcessor


class BITTrainer(pl.LightningModule):
    """
    PyTorch Lightning wrapper for training BIT models.

    This trainer supports:
    - Any HuggingFace AutoModel backbone with a forward(neural_input, x_len) signature
    - Data augmentation: white noise and constant offset
    - CTC loss for sequence-to-sequence learning
    - Per-dataset validation metrics (CER)
    - Warmup + cosine annealing learning rate schedule
    - Mixed precision (16-bit) training for improved performance

    Use with PyTorch Lightning Trainer: precision="16-mixed" for faster training (recommended)
    or precision="32" for full precision if needed.

    Args:
        model: The neural network model (e.g., BitCrossEncoderModel from AutoModel)
        learning_rate: Maximum learning rate (default: 1e-3)
        lr_min: Minimum learning rate at end of cosine schedule (default: 1e-4)
        lr_warmup_steps: Number of warmup steps (default: 1000)
        lr_decay_steps: Total steps for cosine decay (default: 120000)
        weight_decay: L2 regularization weight (default: 1e-5)
        white_noise_sd: Standard deviation for additive white noise augmentation (default: 0.2)
        constant_offset_sd: Standard deviation for constant offset augmentation (default: 0.05)
        beta0: Adam optimizer beta1 parameter (default: 0.9)
        beta1: Adam optimizer beta2 parameter (default: 0.999)
        epsilon: Adam optimizer epsilon (default: 1e-8)
        max_mask_pct_override: Override model's max_mask_pct if not None
        num_masks_override: Override model's num_masks if not None
        num_masks_channels_override: Override model's num_masks_channels if not None
        max_mask_channels_override: Override model's max_channels_to_mask if not None
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-3,
        lr_min: float = 1e-4,
        lr_warmup_steps: int = 1000,
        lr_decay_steps: int = 120000,
        weight_decay: float = 1e-5,
        white_noise_sd: float = 0.2,
        constant_offset_sd: float = 0.05,
        beta0: float = 0.9,
        beta1: float = 0.999,
        epsilon: float = 1e-8,
        max_mask_pct_override: Optional[float] = None,
        num_masks_override: Optional[int] = None,
        num_masks_channels_override: Optional[int] = None,
        max_mask_channels_override: Optional[int] = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.learning_rate = learning_rate
        self.lr_min = lr_min
        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        self.weight_decay = weight_decay
        self.white_noise_sd = white_noise_sd
        self.constant_offset_sd = constant_offset_sd
        self.beta0 = beta0
        self.beta1 = beta1
        self.epsilon = epsilon

        # Override masking parameters if specified
        if max_mask_pct_override is not None and hasattr(self.model, "max_mask_pct"):
            self.model.max_mask_pct = float(max_mask_pct_override)
            print(f"Overriding max_mask_pct to {self.model.max_mask_pct}")

        if num_masks_override is not None and hasattr(self.model, "num_masks"):
            self.model.num_masks = int(num_masks_override)
            print(f"Overriding num_masks to {self.model.num_masks}")

        if num_masks_channels_override is not None and hasattr(
            self.model, "num_masks_channels"
        ):
            self.model.num_masks_channels = int(num_masks_channels_override)
            print(f"Overriding num_masks_channels to {self.model.num_masks_channels}")

        if max_mask_channels_override is not None and hasattr(
            self.model, "max_channels_to_mask"
        ):
            self.model.max_channels_to_mask = int(max_mask_channels_override)
            print(
                f"Overriding max_channels_to_mask to {self.model.max_channels_to_mask}"
            )

        # CTC Loss
        self.ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    def _get_ctc_targets(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return target token ids and lengths for CTC (CPU tensors; caller moves to device).

        Default: phoneme (or pre-batched) targets from the dataloader.
        """
        return batch["seq_class_ids"], batch["seq_lengths"]

    def forward(self, neural_input: torch.Tensor, x_len: torch.Tensor, **kwargs):
        """
        Forward pass through the model.

        Args:
            neural_input: Neural input tensor of shape (batch, time, features)
            x_len: Valid sequence lengths of shape (batch,)
            **kwargs: Additional keyword arguments passed to the model

        Returns:
            Model output logits
        """
        return self.model(neural_input, x_len, **kwargs)

    def _apply_data_augmentation(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply data augmentation to input neural features.

        Args:
            X: Input tensor of shape (batch, time, features)

        Returns:
            Augmented tensor
        """
        # White noise augmentation
        if self.white_noise_sd > 0:
            X = (
                X
                + torch.randn(X.shape, device=self.device, dtype=X.dtype)
                * self.white_noise_sd
            )

        # Constant offset augmentation (different offset per sample, same across time)
        if self.constant_offset_sd > 0:
            X = (
                X
                + torch.randn(
                    [X.shape[0], 1, X.shape[2]], device=self.device, dtype=X.dtype
                )
                * self.constant_offset_sd
            )

        return X

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Training step - forward pass, compute loss, and return for backprop.

        Args:
            batch: Dictionary containing:
                - neural_features: (B, T, F) neural input
                - seq_class_ids: (B, S) target phoneme sequences
                - n_time_steps: (B,) valid lengths
                - seq_lengths: (B,) target sequence lengths
                - source_dataset: List of dataset names
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        # Extract batch data
        X = batch["neural_features"]
        X_len = batch["n_time_steps"]
        y, y_len = self._get_ctc_targets(batch)

        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        X_len = X_len.to(self.device)
        y_len = y_len.to(self.device)

        # Apply data augmentation
        X = self._apply_data_augmentation(X)

        # Create routing dictionary if source_dataset is available
        routing = None
        if "source_dataset" in batch:
            src = batch["source_dataset"]
            if hasattr(self.model, "dataset_keys"):
                routing = {}
                for k in self.model.dataset_keys:
                    idxs = [i for i, name in enumerate(src) if name == k]
                    if idxs:
                        routing[k] = torch.tensor(idxs, device=self.device)

        # Forward pass
        pred = self.forward(X, X_len, routing=routing)

        # Compute output lengths after model processing
        if hasattr(self.model, "compute_length"):
            adj_lens = self.model.compute_length(X_len)
        else:
            # Fallback: assume no length change
            adj_lens = X_len

        # Compute CTC Loss
        # CTC expects: (T, B, C) format
        loss = self.ctc_loss(
            pred.log_softmax(2).permute(1, 0, 2),  # (T, B, C)
            y,  # (B, S)
            adj_lens,  # (B,)
            y_len,  # (B,)
        )

        # Log metrics
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        """
        Validation step - compute loss and Character Error Rate (CER).

        Args:
            batch: Dictionary containing validation data
            batch_idx: Index of the batch

        Returns:
            Loss tensor
        """
        # Extract batch data
        X = batch["neural_features"]
        X_len = batch["n_time_steps"]
        y, y_len = self._get_ctc_targets(batch)

        # Move to device
        X = X.to(self.device)
        y = y.to(self.device)
        X_len = X_len.to(self.device)
        y_len = y_len.to(self.device)

        # Create routing dictionary if source_dataset is available
        routing = None
        src = batch.get("source_dataset", [])
        if src and hasattr(self.model, "dataset_keys"):
            routing = {}
            for k in self.model.dataset_keys:
                idxs = [i for i, name in enumerate(src) if name == k]
                if idxs:
                    routing[k] = torch.tensor(idxs, device=self.device)

        # Forward pass (no augmentation in validation)
        pred = self.forward(X, X_len, routing=routing)

        # Compute output lengths
        if hasattr(self.model, "compute_length"):
            adj_lens = self.model.compute_length(X_len)
        else:
            adj_lens = X_len

        # Compute CTC Loss
        loss = self.ctc_loss(
            pred.log_softmax(2).permute(1, 0, 2),
            y,
            adj_lens,
            y_len,
        )

        # Log overall validation loss
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

        # Compute per-dataset CER if source_dataset is available
        if src:
            # Get unique dataset names
            unique_datasets = list(set(src))

            for dataset_name in unique_datasets:
                # Get indices for this dataset
                dataset_idxs = [i for i, name in enumerate(src) if name == dataset_name]
                if not dataset_idxs:
                    continue

                # Extract predictions and targets for this dataset
                pred_dataset = pred[dataset_idxs]
                y_dataset = y[dataset_idxs]
                y_len_dataset = y_len[dataset_idxs]
                adj_lens_dataset = adj_lens[dataset_idxs]

                # Compute CER for this dataset
                total_edit_distance = 0
                total_seq_length = 0

                for i in range(pred_dataset.shape[0]):
                    T = int(adj_lens_dataset[i].item())

                    # Decode prediction
                    decoded_seq = torch.argmax(pred_dataset[i, :T, :], dim=-1)
                    decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                    decoded_seq = decoded_seq[decoded_seq != 0].cpu().numpy()

                    # Get true sequence
                    true_seq = y_dataset[i][: y_len_dataset[i]].cpu().numpy()

                    # Compute edit distance
                    matcher = SequenceMatcher(
                        a=true_seq.tolist(), b=decoded_seq.tolist()
                    )
                    total_edit_distance += matcher.distance()
                    total_seq_length += len(true_seq)

                # Compute and log CER
                cer = (
                    total_edit_distance / total_seq_length
                    if total_seq_length > 0
                    else 1.0
                )
                self.log(f"val/cer_{dataset_name}", cer, prog_bar=True, on_epoch=True)

        return loss

    def on_after_backward(self):
        """Log gradient norm after backward pass (before clipping)."""
        grad_norm = torch.norm(
            torch.stack(
                [
                    p.grad.detach().data.norm(2)
                    for p in self.parameters()
                    if p.grad is not None
                ]
            ),
            2,
        )
        self.log(
            "train/grad_norm",
            grad_norm.item(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log learning rate at the end of each training batch."""
        opt = self.optimizers()
        if isinstance(opt, list):
            opt = opt[0]
        for i, pg in enumerate(opt.param_groups):
            self.log(
                f"lr_group{i}",
                pg["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary containing optimizer and lr_scheduler configuration
        """
        # Create Adam optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.beta0, self.beta1),
            eps=self.epsilon,
        )

        # Learning rate lambda function with warmup + cosine annealing
        def lr_lambda(step):
            if step < self.lr_warmup_steps:
                # Linear warmup
                return step / float(max(1, self.lr_warmup_steps))
            else:
                # Cosine annealing
                progress = (step - self.lr_warmup_steps) / float(
                    max(1, self.lr_decay_steps - self.lr_warmup_steps)
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return (
                    self.lr_min / self.learning_rate
                    + (1 - self.lr_min / self.learning_rate) * cosine_decay
                )

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Update every step
                "frequency": 1,
            },
        }


class BITTrainerForCharacters(BITTrainer):
    """
    BIT trainer that builds CTC targets from text, not precomputed phoneme ids.
    Reads ``sentence_label`` (or ``transcriptions``) from each batch, uppercases the
    strings, and encodes them with the HuggingFace wav2vec2 tokenizer (same convention
    as ``patrickvonplaten/wav2vec2-base`` elsewhere in this package). Use a model
    whose output dimension matches ``tokenizer.vocab_size`` (blank id 0, as in
    :class:`BITTrainer`).
    Args:
        model: Neural decoder with logits over wav2vec2 character classes.
        tokenizer_name: HuggingFace model id for :class:`~transformers.AutoProcessor`.
        **kwargs: Forwarded to :class:`BITTrainer` (learning rate, augmentation, etc.).
    """

    def __init__(
        self,
        model,
        tokenizer_name: str = "patrickvonplaten/wav2vec2-base",
        **kwargs: Any,
    ):
        super().__init__(model=model, **kwargs)
        self.tokenizer_name = tokenizer_name
        self._wav2vec_processor = AutoProcessor.from_pretrained(tokenizer_name)

    def _get_ctc_targets(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sentences = batch.get("sentence_label")
        if sentences is None:
            sentences = batch.get("transcriptions")
        if sentences is None:
            raise KeyError(
                "BITTrainerForCharacters requires 'sentence_label' or 'transcriptions' in the batch"
            )
        outputs = self._wav2vec_processor.tokenizer(
            [str(s).upper() for s in sentences],
            return_tensors="pt",
            padding=True,
        )
        y = outputs["input_ids"]
        y_len = outputs["attention_mask"].sum(dim=-1).to(dtype=torch.long)
        return y, y_len
