"""
BitCrossEncoder model implementation for HuggingFace compatibility.
"""

import math
import numbers
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, Dict, Tuple, List, Union
from einops import rearrange
from einops.layers.torch import Rearrange

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

try:
    from .bit_cross_encoder_config import BitCrossEncoderConfig
except ImportError:
    from bit_cross_encoder_config import BitCrossEncoderConfig


# ============================================================================
# Helper Functions
# ============================================================================


def pair(t):
    """Convert int to tuple or return tuple as is."""
    return t if isinstance(t, tuple) else (t, t)


def pad_to_multiple(tensor, multiple, dim=1, value=0):
    """
    Pads `tensor` along `dim` so that its size is divisible by `multiple`.
    """
    size = tensor.size(dim)
    padding_needed = (multiple - size % multiple) % multiple
    if padding_needed == 0:
        return tensor
    pad_dims = [0] * (2 * tensor.dim())
    pad_dims[-2 * dim - 1] = padding_needed  # padding at the end
    return F.pad(tensor, pad_dims, value=value)


def get_sinusoidal_pos_emb(seq_len, dim, device=None, dtype=None):
    """Generate sinusoidal positional embeddings with specified dtype."""
    if dtype is None:
        dtype = torch.float32
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * -(math.log(10000.0) / dim)
    )
    pe = torch.zeros(seq_len, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def create_temporal_mask(seq_len, device=None):
    """
    Build a boolean mask of shape [1, 1, seq_len, seq_len] that allows each
    timestep t to attend to positions ≤ t (causal masking).

    Args:
        seq_len (int): sequence length T

    Returns:
        torch.Tensor: Boolean mask of shape [1, 1, T, T]
    """
    i = torch.arange(seq_len, device=device).unsqueeze(1)  # [T, 1] (query index)
    j = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T] (key index)
    mask = j <= i  # [T, T], True = allowed
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]


def _normalize_mae_key(k: str) -> str:
    """
    Strip common prefixes from a checkpoint key so it matches BitCrossEncoder
    state_dict (no prefix). Handles Lightning (model.) and HuggingFace (bit_mae.).
    """
    for prefix in ("model.", "bit_mae."):
        if k.startswith(prefix):
            k = k[len(prefix) :]
    return k


def _load_mae_state_dict(
    mae_checkpoint: Union[str, Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Load a state dict from a path or dict. If the path/dict is a PyTorch Lightning
    checkpoint (has top-level 'state_dict'), use that and strip the 'model.'
    prefix so keys match the raw BitMAE module.
    """
    if isinstance(mae_checkpoint, str):
        import os as _os

        bin_path = _os.path.join(mae_checkpoint, "pytorch_model.bin")
        if not _os.path.isfile(bin_path):
            bin_path = _os.path.join(mae_checkpoint, "model.safetensors")
        if _os.path.isfile(bin_path):
            if bin_path.endswith(".safetensors"):
                from safetensors.torch import load_file as _safe_load

                raw = _safe_load(bin_path)
            else:
                raw = torch.load(bin_path, map_location="cpu", weights_only=True)
        else:
            raw = torch.load(mae_checkpoint, map_location="cpu", weights_only=True)
    else:
        raw = dict(mae_checkpoint)

    # PyTorch Lightning: full checkpoint has "state_dict" with keys like "model.xxx"
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]

    return raw


# ============================================================================
# Model Components
# ============================================================================


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d, 2d or 3d tensor.
    Filtering is performed separately for each channel in the input using a depthwise convolution.

    Args:
        channels (int): Number of channels of the input tensors.
        kernel_size (int): Size of the gaussian kernel.
        sigma (float): Standard deviation of the gaussian kernel.
        dim (int): The number of dimensions of the data. Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size],
            indexing="ij",
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.

        Args:
            input (torch.Tensor): Input to apply gaussian filter on.

        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        # Ensure weight matches input dtype for mixed precision compatibility
        weight = self.weight.to(dtype=input.dtype)
        return self.conv(input, weight=weight, groups=self.groups, padding="same")


class Attention(nn.Module):
    """Multi-head attention with optional T5-style relative position bias."""

    def __init__(
        self, dim, heads, dim_head, dropout, max_rel_dist=200, use_relative_bias=True
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

        # T5-style relative position bias
        self.max_rel_dist = max_rel_dist
        self.use_relative_bias = use_relative_bias

        if self.use_relative_bias:
            self.rel_pos_bias = nn.Embedding(2 * max_rel_dist - 1, 1)

    def forward(self, x, temporal_mask=None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (b, h, n, n)

        # Add relative positional bias if enabled
        if self.use_relative_bias:
            seq_len = x.size(1)
            i = torch.arange(seq_len, device=x.device).unsqueeze(1)
            j = torch.arange(seq_len, device=x.device).unsqueeze(0)
            rel_pos = (
                (i - j).clamp(-self.max_rel_dist + 1, self.max_rel_dist - 1)
                + self.max_rel_dist
                - 1
            )
            rel_bias = self.rel_pos_bias(rel_pos).squeeze(-1).unsqueeze(0).unsqueeze(0)
            dots = dots + rel_bias

        if temporal_mask is not None:
            dots = dots.masked_fill(temporal_mask == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class FFN(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    """Transformer encoder with multiple layers of attention and FFN."""

    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim_ratio,
        dropout=0.0,
        use_relative_bias=True,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        mlp_dim = mlp_dim_ratio * dim
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            use_relative_bias=use_relative_bias,
                        ),
                        FFN(dim=dim, hidden_dim=mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x, mask=None):
        for attn, ffn in self.layers:
            x = attn(x, temporal_mask=mask) + x
            x = ffn(x) + x
        return self.norm(x)


# ============================================================================
# Main Model
# ============================================================================


class BitCrossEncoderPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading
    and loading pretrained models.
    """

    config_class = BitCrossEncoderConfig
    base_model_prefix = "bit_cross_encoder"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0,
                std=(
                    self.config.initializer_range
                    if hasattr(self.config, "initializer_range")
                    else 0.02
                ),
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0,
                std=(
                    self.config.initializer_range
                    if hasattr(self.config, "initializer_range")
                    else 0.02
                ),
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class BitCrossEncoderModel(BitCrossEncoderPreTrainedModel):
    """
    BitCrossEncoder model for neural decoding with per-dataset embeddings and optional masking.

    This model processes neural input signals (B, T, F) through:
    1. Gaussian smoothing
    2. Patchification
    3. Per-dataset embedding routing
    4. Transformer encoding
    5. Projection to output classes

    Args:
        config (:class:`~BitCrossEncoderConfig`): Model configuration class with all the parameters of the model.
    """

    def __init__(self, config: BitCrossEncoderConfig):
        super().__init__(config)
        self.config = config

        patch_height, patch_width = pair(config.patch_size)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.dim = config.dim
        self.nClasses = config.n_classes
        self.gaussianSmoothWidth = config.gaussian_smooth_width
        self.T5_style_pos = config.t5_style_pos
        self.max_mask_pct = config.max_mask_pct
        self.num_masks = config.num_masks
        self.patch_dim = patch_height * patch_width
        self.num_masks_channels = config.num_masks_channels
        self.max_channels_to_mask = config.max_mask_channels
        self.dataset_keys = list(config.dataset_keys)
        self.per_dataset_heads = config.per_dataset_heads
        self.causal = getattr(config, "causal", True)

        # Tokenizer: (B, 1, T, F) -> (B, P, patch_dim)
        self.to_patch = Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width
        )

        # Per-dataset projection to model dim
        def make_patch_to_emb():
            return nn.Sequential(
                nn.LayerNorm(self.patch_dim),
                nn.Linear(self.patch_dim, config.dim),
                nn.LayerNorm(config.dim),
            )

        self.patch_to_emb = nn.ModuleDict()
        for k in self.dataset_keys:
            self.patch_to_emb[k] = make_patch_to_emb()

        # Gaussian smoother
        self.gaussianSmoother = GaussianSmoothing(
            patch_width, 20, self.gaussianSmoothWidth, dim=1
        )

        # Mask token
        if config.mask_token_zeros:
            self.mask_token = nn.Parameter(
                torch.zeros(self.patch_dim), requires_grad=False
            )
        else:
            self.mask_token = nn.Parameter(torch.randn(self.patch_dim))

        self.dropout = nn.Dropout(config.input_dropout)

        # Transformer
        self.transformer = Transformer(
            config.dim,
            config.depth,
            config.heads,
            config.dim_head,
            config.mlp_dim_ratio,
            config.dropout,
            use_relative_bias=self.T5_style_pos,
        )

        # Output projection heads
        if config.per_dataset_heads:
            self.projection = nn.ModuleDict(
                {
                    k: nn.Linear(config.dim, config.n_classes + 1)
                    for k in self.dataset_keys
                }
            )
        else:
            self.projection = nn.Linear(config.dim, config.n_classes + 1)

        # Positional embedding buffer (for non-T5 style)
        if not self.T5_style_pos:
            self.register_buffer("pos_embedding", None, persistent=False)

        # Initialize weights
        self.post_init()

    def _route_and_embed(self, x_tokens: torch.Tensor, routing: Optional[Dict] = None):
        """
        Route tokens to dataset-specific embedders.

        Args:
            x_tokens: (B, P, patch_dim)
            routing: {key: LongTensor[indices]} or None

        Returns:
            x_emb: (B, P, dim) in original batch order
        """
        B, P, _ = x_tokens.shape
        device = x_tokens.device

        if not routing or sum((len(v) for v in routing.values() if v is not None)) == 0:
            # Default: route all to first dataset embedder
            key0 = self.dataset_keys[0]
            return self.patch_to_emb[key0](x_tokens)

        # First pass: determine output dtype by running one embedder
        # This ensures x_emb has the correct dtype from the start
        first_key = next(iter(routing.keys()))
        first_idx = routing[first_key]
        if first_idx is not None and len(first_idx) > 0:
            first_idx_tensor = torch.as_tensor(
                first_idx, device=device, dtype=torch.long
            )
            sample_tokens = x_tokens.index_select(0, first_idx_tensor[:1])
            sample_emb = self.patch_to_emb[first_key](sample_tokens)
            output_dtype = sample_emb.dtype
        else:
            # Fallback: use first available key
            output_dtype = x_tokens.dtype

        # Create output tensor with correct dtype
        x_emb = torch.empty(B, P, self.dim, device=device, dtype=output_dtype)

        # Route per key, then write back in-place
        for key, idx in routing.items():
            if idx is None:
                continue
            idx = torch.as_tensor(idx, device=device, dtype=torch.long)
            if idx.numel() == 0:
                continue
            tokens_k = x_tokens.index_select(0, idx)  # (B_k, P, patch_dim)
            emb_k = self.patch_to_emb[key](tokens_k)  # (B_k, P, dim)
            x_emb.index_copy_(0, idx, emb_k)
        return x_emb

    def apply_time_mask(self, X, X_len, constant_mask=False):
        """
        Fully vectorized SpecAugment-style time masking.

        Args:
            X: (B, P, D) input tensor
            X_len: (B,) valid lengths in timepoints
            constant_mask: if True, make the mask lengths the same across all batches

        Returns:
            X_masked: (B, P, D) with masked patches
            mask: (B, P) boolean mask of where values were masked
        """
        B, P, D = X.shape
        device = X.device
        dtype = X.dtype  # Preserve input dtype for mixed precision

        if constant_mask:
            # Get valid len of smallest trial in batch and repeat for all batches
            valid_lens = torch.min((X_len // self.patch_height).to(device)).repeat(B)
        else:
            valid_lens = (X_len // self.patch_height).to(device)

        max_mask_lens = (self.max_mask_pct * valid_lens).long()  # (B,)

        # Repeat B num_masks times to simulate multiple masks per sample
        B_rep = B * self.num_masks

        # Expand inputs for vectorized masking
        valid_lens_rep = valid_lens.repeat_interleave(
            self.num_masks
        )  # (B * num_masks,)
        max_mask_lens_rep = max_mask_lens.repeat_interleave(
            self.num_masks
        )  # (B * num_masks,)

        if constant_mask:
            # Select the same t for every batch
            t = (
                (
                    torch.rand(self.num_masks, device=device).repeat(B)
                    * (max_mask_lens_rep + 1).float()
                )
                .floor()
                .long()
                .clamp(min=1)
            )
        else:
            t = (
                (torch.rand(B_rep, device=device) * (max_mask_lens_rep + 1).float())
                .floor()
                .long()
            )

        max_start = (valid_lens_rep - t + 1).clamp(min=1)

        if constant_mask:
            t0 = (
                (
                    torch.rand(self.num_masks, device=device).repeat(B)
                    * max_start.float()
                )
                .floor()
                .long()
            )
        else:
            t0 = (torch.rand(B_rep, device=device) * max_start.float()).floor().long()

        # Build the global mask (B, P)
        arange = torch.arange(P, device=device).unsqueeze(0)  # (1, P)
        t0_exp = t0.unsqueeze(1)  # (B_rep, 1)
        t1_exp = (t0 + t).unsqueeze(1)  # (B_rep, 1)
        mask_chunks = (arange >= t0_exp) & (arange < t1_exp)  # (B_rep, P)

        # Get index of sample in batch for each mask chunk
        batch_idx = torch.arange(B, device=device).repeat_interleave(
            self.num_masks
        )  # (B * num_masks,)

        # Now scatter all the masks into the full mask (B, P)
        patch_idx = mask_chunks.nonzero(as_tuple=False)  # (N_masked, 2)
        b_indices = batch_idx[patch_idx[:, 0]]  # (N_masked,)
        p_indices = patch_idx[:, 1]  # (N_masked,)

        mask = torch.zeros(B, P, dtype=torch.bool, device=device)
        mask[b_indices, p_indices] = True

        # Apply the mask - ensure mask_token matches input dtype
        X_masked = X.clone()
        mask_token = self.mask_token.to(dtype=dtype)
        X_masked[mask] = mask_token

        return X_masked, mask

    def compute_length(self, X_len):
        """
        Compute output sequence length after patchification.

        Args:
            X_len: Input sequence lengths (B,)

        Returns:
            Output sequence lengths after patching
        """

        # Computing ceiling because we pad X to be divisible by patch_height

        return torch.ceil(X_len / self.patch_height).to(dtype=torch.int32)  ## strange

    def forward(
        self,
        neural_input: torch.Tensor,
        x_len: torch.Tensor,
        routing: Optional[Dict[str, torch.Tensor]] = None,
        return_state: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        """
        Forward pass of the BitCrossEncoder model.

        Args:
            neural_input: (B, T, F) - Neural input tensor
            x_len: (B,) - Valid lengths in timepoints
            routing: Optional dict like {"card": LongTensor[idxs], "willet": LongTensor[idxs], ...}
                     If None/empty -> falls back to the first dataset key
            return_state: Whether to return intermediate hidden states
            output_attentions: Whether to output attention weights (not implemented)
            output_hidden_states: Whether to output all hidden states

        Returns:
            If return_state is True: (output, hidden_state, mask)
            Otherwise: output logits (B, P, n_classes + 1)
        """
        # Pad & (optional) channel mask
        neural_input = pad_to_multiple(neural_input, multiple=self.patch_height)

        # Smooth
        neural_input = torch.permute(neural_input, (0, 2, 1))  # (B, F, T)
        neural_input = self.gaussianSmoother(neural_input)
        neural_input = torch.permute(neural_input, (0, 2, 1))  # (B, T, F)

        # Patchify
        neural_input = neural_input.unsqueeze(1)  # (B, 1, T, F)
        x_tokens = self.to_patch(neural_input)  # (B, P, patch_dim)

        # Time masking only while training
        mask = None
        if self.training and self.max_mask_pct > 0 and self.num_masks > 0:
            x_tokens, mask = self.apply_time_mask(x_tokens, x_len)

        # Route to dataset-specific embedders, reassemble in original order
        x = self._route_and_embed(x_tokens, routing)  # (B, P, dim)

        # Input-level dropout
        x = self.dropout(x)
        b, seq_len, _ = x.shape

        # Positions (if not using T5-style)
        if not self.T5_style_pos:
            pos_emb = get_sinusoidal_pos_emb(
                seq_len, self.dim, device=x.device, dtype=x.dtype
            )
            x = x + pos_emb.unsqueeze(0)

        # Attention mask & transformer (causal vs bidirectional/acausal)
        temporal_mask = (
            create_temporal_mask(seq_len, device=x.device) if self.causal else None
        )
        x = self.transformer(x, mask=temporal_mask)

        # Projection(s)
        if self.per_dataset_heads:
            # Per-dataset heads not fully implemented in forward - would need routing info
            # Fall back to first dataset head for now
            out = self.projection[self.dataset_keys[0]](x)
        else:
            out = self.projection(x)

        if return_state:
            return out, x, mask

        return out

    def encode(
        self,
        neural_input: torch.Tensor,
        x_len: torch.Tensor,
        routing: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Forward pass of the BitCrossEncoder model.

        Args:
            neural_input: (B, T, F) - Neural input tensor
            x_len: (B,) - Valid lengths in timepoints
            routing: Optional dict like {"card": LongTensor[idxs], "willet": LongTensor[idxs], ...}
                     If None/empty -> falls back to the first dataset key
            return_state: Whether to return intermediate hidden states
            output_attentions: Whether to output attention weights (not implemented)
            output_hidden_states: Whether to output all hidden states

        Returns:
            x: (B, P, dim) - Encoded features
        """
        # Pad & (optional) channel mask
        neural_input = pad_to_multiple(neural_input, multiple=self.patch_height)

        # Smooth
        neural_input = torch.permute(neural_input, (0, 2, 1))  # (B, F, T)
        neural_input = self.gaussianSmoother(neural_input)
        neural_input = torch.permute(neural_input, (0, 2, 1))  # (B, T, F)

        # Patchify
        neural_input = neural_input.unsqueeze(1)  # (B, 1, T, F)
        x_tokens = self.to_patch(neural_input)  # (B, P, patch_dim)

        # Time masking only while training
        mask = None
        if self.training and self.max_mask_pct > 0 and self.num_masks > 0:
            x_tokens, mask = self.apply_time_mask(x_tokens, x_len)

        # Route to dataset-specific embedders, reassemble in original order
        x = self._route_and_embed(x_tokens, routing)  # (B, P, dim)

        # Input-level dropout
        x = self.dropout(x)
        b, seq_len, _ = x.shape

        # Positions (if not using T5-style)
        if not self.T5_style_pos:
            pos_emb = get_sinusoidal_pos_emb(
                seq_len, self.dim, device=x.device, dtype=x.dtype
            )
            x = x + pos_emb.unsqueeze(0)

        # Attention mask & transformer (causal vs bidirectional/acausal)
        temporal_mask = (
            create_temporal_mask(seq_len, device=x.device) if self.causal else None
        )
        x = self.transformer(x, mask=temporal_mask)

        return x

    @classmethod
    def from_pretrained_mae(
        cls,
        mae_checkpoint: Union[str, Dict[str, torch.Tensor]],
        config: Optional["BitCrossEncoderConfig"] = None,
        strict: bool = False,
        **kwargs,
    ) -> "BitCrossEncoderModel":
        """
        Build a BitCrossEncoderModel and load compatible weights from a pretrained
        BitMAE checkpoint. The MAE encoder (patchers, transformer) is identical to
        the cross encoder; only the decoder (dim -> patch_dim) is MAE-specific and
        is dropped. The cross encoder's projection head(s) remain randomly initialized.

        Requires that config matches the MAE backbone (same patch_size, dim, depth,
        heads, dim_head, mlp_dim_ratio, dataset_keys, gaussian_smooth_width,
        t5_style_pos, mask_token_zeros, input_dropout, dropout).

        Args:
            mae_checkpoint: Path to a saved BitMAE checkpoint (e.g. from
                BitMAEModel.save_pretrained(...)) or a state_dict (e.g. from
                BitMAEModel.state_dict() or torch.load(path, weights_only=True)).
            config: BitCrossEncoderConfig. If None, you must have loaded from a
                directory that contains config.json so that the config can be
                inferred (not supported when mae_checkpoint is a state_dict).
            strict: If True, require every key in the cross encoder to be loaded
                (will fail because projection/decoder differ). Use False (default).
            **kwargs: Passed through to BitCrossEncoderModel(config).

        Returns:
            BitCrossEncoderModel with encoder weights loaded from MAE; projection
            heads are randomly initialized.
        """
        if config is None:
            if isinstance(mae_checkpoint, str):
                import json as _json
                import os as _os

                config_path = _os.path.join(mae_checkpoint, "config.json")
                if _os.path.isfile(config_path):
                    with open(config_path) as f:
                        cfg_dict = _json.load(f)
                    if cfg_dict.get("model_type") == "bit_mae":
                        config = BitCrossEncoderConfig(
                            patch_size=cfg_dict.get("patch_size", (5, 512)),
                            dim=cfg_dict.get("dim", 384),
                            depth=cfg_dict.get("depth", 7),
                            heads=cfg_dict.get("heads", 4),
                            mlp_dim_ratio=cfg_dict.get("mlp_dim_ratio", 4),
                            dim_head=cfg_dict.get("dim_head", 64),
                            dropout=cfg_dict.get("dropout", 0.4),
                            input_dropout=cfg_dict.get("input_dropout", 0.2),
                            gaussian_smooth_width=cfg_dict.get(
                                "gaussian_smooth_width", 2.0
                            ),
                            t5_style_pos=cfg_dict.get("t5_style_pos", True),
                            mask_token_zeros=cfg_dict.get("mask_token_zeros", False),
                            dataset_keys=cfg_dict.get("dataset_keys"),
                        )
                    else:
                        config = BitCrossEncoderConfig.from_pretrained(mae_checkpoint)
                else:
                    raise FileNotFoundError(
                        f"from_pretrained_mae: config not found at {config_path}; "
                        "pass config= explicitly when mae_checkpoint is a state_dict or a path without config.json"
                    )
            else:
                raise ValueError(
                    "from_pretrained_mae requires config when mae_checkpoint is a state_dict"
                )
        model = cls(config, **kwargs)

        mae_state = _load_mae_state_dict(mae_checkpoint)

        # Normalize keys and keep only encoder weights; drop MAE decoder
        cross_keys = set(model.state_dict().keys())
        loaded = {}
        for k, v in mae_state.items():
            nk = _normalize_mae_key(k)
            if ".decoder." in nk or nk.startswith("decoder."):
                continue
            if nk in cross_keys:
                loaded[nk] = v

        model.load_state_dict(loaded, strict=strict)
        return model

    def load_pretrained_mae_weights(
        self,
        mae_checkpoint: Union[str, Dict[str, torch.Tensor]],
        strict: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        Load pretrained BitMAE encoder weights into this BitCrossEncoderModel in-place.
        Compatible weights: patch_to_emb, gaussianSmoother, mask_token, dropout,
        transformer. MAE decoder weights are ignored. Projection head(s) are unchanged.

        Args:
            mae_checkpoint: Path to a BitMAE checkpoint or a state_dict.
            strict: If False (default), only load matching keys; projection and
                decoder are left as-is.

        Returns:
            (list of loaded key names, list of skipped key names).
        """
        mae_state = _load_mae_state_dict(mae_checkpoint)

        cross_keys = set(self.state_dict().keys())
        loaded = {}
        skipped = []
        for k, v in mae_state.items():
            nk = _normalize_mae_key(k)
            if ".decoder." in nk or nk.startswith("decoder."):
                skipped.append(nk)
                continue
            if nk in cross_keys:
                loaded[nk] = v
            else:
                skipped.append(nk)

        self.load_state_dict(loaded, strict=strict)
        return (list(loaded.keys()), skipped)

    def get_input_embeddings(self):
        """Get the input embeddings (not applicable for this model)."""
        return None

    def set_input_embeddings(self, value):
        """Set the input embeddings (not applicable for this model)."""
        pass


# Register configuration
BitCrossEncoderConfig.register_for_auto_class()
BitCrossEncoderModel.register_for_auto_class("AutoModel")
