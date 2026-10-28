"""
BitCrossEncoder configuration for HuggingFace compatibility.
"""

from transformers import PretrainedConfig
from typing import List, Optional


class BitCrossEncoderConfig(PretrainedConfig):
    """
    Configuration class for BitCrossEncoder model.

    This is the configuration class to store the configuration of a :class:`~BitCrossEncoder`.
    It is used to instantiate a BitCrossEncoder model according to the specified arguments,
    defining the model architecture.

    Args:
        patch_size (int or tuple): Size of patches (height, width). Default: (5, 512)
        dim (int): Dimension of the model. Default: 384
        depth (int): Number of transformer layers. Default: 7
        heads (int): Number of attention heads. Default: 4
        mlp_dim_ratio (int): Ratio of MLP hidden dimension to model dimension. Default: 4
        dim_head (int): Dimension of each attention head. Default: 64
        dropout (float): Dropout rate in transformer. Default: 0.4
        input_dropout (float): Dropout rate for input embeddings. Default: 0.2
        gaussian_smooth_width (float): Width of Gaussian smoothing kernel. Default: 2.0
        n_classes (int): Number of output classes (excluding blank). Default: 40
        t5_style_pos (bool): Whether to use T5-style relative position bias. Default: True
        max_mask_pct (float): Maximum percentage of sequence to mask during training. Default: 0.075
        num_masks (int): Number of masks to apply. Default: 20
        mask_token_zeros (bool): Whether to initialize mask token as zeros. Default: False
        num_masks_channels (int): Number of channel masks. Default: 0
        max_mask_channels (int): Maximum number of channels to mask. Default: 0
        dataset_keys (List[str]): List of dataset keys for per-dataset embeddings.
        per_dataset_heads (bool): Whether to use per-dataset projection heads. Default: False
    """

    model_type = "bit_cross_encoder"

    def __init__(
        self,
        patch_size: tuple = (5, 512),
        dim: int = 384,
        depth: int = 7,
        heads: int = 4,
        mlp_dim_ratio: int = 4,
        dim_head: int = 64,
        dropout: float = 0.4,
        input_dropout: float = 0.2,
        gaussian_smooth_width: float = 2.0,
        n_classes: int = 40,
        t5_style_pos: bool = True,
        max_mask_pct: float = 0.075,
        num_masks: int = 20,
        mask_token_zeros: bool = False,
        num_masks_channels: int = 0,
        max_mask_channels: int = 0,
        dataset_keys: Optional[List[str]] = None,
        per_dataset_heads: bool = False,
        causal: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Handle patch_size as tuple, list, or int
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, list):
            patch_size = tuple(patch_size)

        self.patch_size = patch_size
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim_ratio = mlp_dim_ratio
        self.dim_head = dim_head
        self.dropout = dropout
        self.input_dropout = input_dropout
        self.gaussian_smooth_width = gaussian_smooth_width
        self.n_classes = n_classes
        self.t5_style_pos = t5_style_pos
        self.max_mask_pct = max_mask_pct
        self.num_masks = num_masks
        self.mask_token_zeros = mask_token_zeros
        self.num_masks_channels = num_masks_channels
        self.max_mask_channels = max_mask_channels

        # Default dataset keys
        if dataset_keys is None:
            dataset_keys = [
                "card",
                "willet",
                "kunz_t12",
                "kunz_t15",
                "kunz_t16",
                "kunz_t17",
                "handwriting",
            ]
        self.dataset_keys = dataset_keys
        self.per_dataset_heads = per_dataset_heads
        # If False, use full (bidirectional/acausal) self-attention instead of
        # causal masking. Acausal context improves offline decoding accuracy.
        self.causal = causal
