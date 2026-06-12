import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass


class MLPforMAP(nn.Module):
    def __init__(self, hidden_act, intermediate_size, hidden_size):
        super().__init__()
        self.activation_fn = hidden_act
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


@dataclass
class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores

    Examples:

    ```python
    >>> import torch
    >>> from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    >>> converter = AttentionMaskConverter(True)
    >>> converter.to_4d(torch.tensor([[0, 0, 0, 1, 1]]), 5, key_value_length=5, dtype=torch.float32)
    tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])
    ```

    Parameters:
        is_causal (`bool`):
            Whether the attention mask should be a uni-directional (causal) or bi-directional mask.

        sliding_window (`int`, *optional*):
            Optionally, the sliding window masks can be created if `sliding_window` is defined to a positive integer.
    """

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    @staticmethod
    def _expand_mask(
        mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
    ):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = (
            mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        )

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )


def _prepare_4d_attention_mask(
    mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None
):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


class MultiheadAttentionPooling(nn.Module):
    def __init__(
        self,
        hidden_size,
        target_len,
        num_attention_heads,
        hidden_act,
        intermediate_size,
        layer_norm_eps,
    ):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, target_len, hidden_size))
        self.attention = torch.nn.MultiheadAttention(
            hidden_size, num_attention_heads, batch_first=True
        )
        self.layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.mlp = MLPforMAP(hidden_act, intermediate_size, hidden_size)
        self.num_heads = num_attention_heads
        self.target_len = target_len

    def forward(
        self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        if attention_mask is not None:
            source_len = hidden_state.shape[1]
            attention_mask = _prepare_4d_attention_mask(
                attention_mask, hidden_state.dtype, self.target_len
            )
            attention_mask = attention_mask.repeat(
                1, self.num_heads, self.target_len, 1
            )
            attention_mask = attention_mask.reshape(-1, self.target_len, source_len)
        # print("hidden_state:", hidden_state.size())
        # print("probe:", probe.size())
        hidden_state = self.attention(
            probe, hidden_state, hidden_state, attn_mask=attention_mask
        )[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        # print("hidden_state:", hidden_state.size())
        return hidden_state
