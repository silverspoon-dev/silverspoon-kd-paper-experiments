from collections.abc import Callable
from dataclasses import dataclass
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import initialization as init
from transformers.activations import ACT2FN, get_activation
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions, QuestionAnsweringModelOutput, SequenceClassifierOutputWithPast, TokenClassifierOutput
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2MLP
from transformers.models.gpt2.modeling_gpt2 import GPT2PreTrainedModel
from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward
from transformers.pytorch_utils import Conv1D
from transformers.utils import ModelOutput, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs
import math
import torch
from typing import List, Tuple, Optional
import math
import torch

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads for grouped-query attention.
    (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_heads, seq_len, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def quadratic_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        causal: bool = True, fp32_attention: bool = False,
                        eps: float = 1e-12,
                        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    Feature-mapped attention with explicit L x L matrix (for distillation).
    Unlike linear_attention, this forms the full attention matrix so we can
    supervise the student's attention weights against the teacher's.
    q, k: (batch, heads, seq_len, feature_dim) [after feature map]
    v: (batch, heads, seq_len, head_dim)
    """
    dtype = q.dtype
    if fp32_attention:
        q, k = q.float(), k.float()
    a = torch.einsum('bhmd,bhnd->bhmn', q, k)
    if causal:
        m, n = a.shape[-2:]
        causal_mask = torch.ones((m, n), device=a.device, dtype=torch.bool).triu(n - m + 1)
        a = a.masked_fill(causal_mask, 0)
    a = a / (a.sum(dim=-1, keepdim=True) + eps)
    a = a.to(dtype=dtype) if fp32_attention else a
    y = torch.einsum('bhmn,bhnd->bhmd', a, v) if v is not None else None
    return y, a, None

def get_masks(window_size: int, q_len: int, k_len: int,
              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return masks for sliding window (softmax) and linear attention regions.
    Uses TK "terracing" arrangement with overlapping windows.
    Returns: (window_mask, linear_mask) each of shape (1, 1, q_len, k_len)
    """
    l = window_size
    m = math.ceil(max(q_len, k_len) / window_size)
    mask = torch.block_diag(*[torch.ones((l, l))] * m)
    mask += torch.roll(mask, -l, -1)  # terracing: overlap adjacent windows
    if mask.shape[0] > q_len:
        mask = mask[-q_len:]
    if mask.shape[1] > k_len:
        mask = mask[:, -k_len:]
    mask = mask[None, None, ...]  # (1, 1, q_len, k_len)
    return (torch.tril(mask).to(device=device, dtype=torch.int),
            torch.tril(1 - mask).to(device=device, dtype=torch.int))

def hybrid_attention_quadratic(q: torch.Tensor, k: torch.Tensor,
                                f_q: torch.Tensor, f_k: torch.Tensor,
                                v: torch.Tensor,
                                window_factor: torch.Tensor,
                                linear_factor: torch.Tensor,
                                window_size: int,
                                kv_state: torch.Tensor = None,
                                k_state: torch.Tensor = None,
                                eps: float = 1e-12,
                                mask_value: float = -1e8):
    """
    Hybrid attention combining sliding window softmax and linear attention.
    - Recent tokens (within window): softmax attention on raw q, k
    - Older tokens (beyond window): linear attention on feature-mapped f_q, f_k
    - Combined with learned window_factor weighting

    Args:
        q, k: raw query/key (batch, heads, seq_len, head_dim)
        f_q, f_k: feature-mapped query/key (batch, heads, seq_len, feature_dim)
        v: values (batch, heads, seq_len, head_dim)
        window_factor: learned weight for softmax window (batch, heads, 1, 1)
        linear_factor: weight for linear part (batch, heads, 1, 1)
        window_size: size of the sliding window
    """
    mask_window, mask_linear = get_masks(window_size, q.shape[-2], k.shape[-2], q.device)

    # 1. Sliding window (softmax attention on raw q, k)
    a_sm = torch.einsum('bhmd,bhnd->bhmn', q.float(), k.float()) * (k.shape[-1] ** -0.5)
    a_sm = a_sm.masked_fill(~mask_window.bool(), mask_value)
    a_sm_max = torch.amax(a_sm, dim=-1, keepdim=True)
    a_sm = window_factor * torch.exp(a_sm - a_sm_max)
    sum_sm = a_sm.sum(dim=-1, keepdim=True)

    # 2. Linear attention (on feature-mapped f_q, f_k)
    a_ln = torch.einsum('bhmd,bhnd->bhmn', f_q.float(), f_k.float())
    a_ln = linear_factor * a_ln.masked_fill(~mask_linear.bool(), 0)
    sum_ln = a_ln.sum(dim=-1, keepdim=True)

    # 3. Combine
    a = ((a_sm + a_ln) / (sum_sm + sum_ln + eps)).to(q.dtype)
    y = torch.einsum('bhmn,bhnd->bhmd', a_sm + a_ln, v.float())
    if kv_state is not None:
        y += linear_factor * torch.einsum('bhld,bhdf->bhlf', f_q.float(), kv_state.float())
        sum_ln += linear_factor * torch.einsum(
            'bhld,bhnd->bhl', f_q.float(), k_state.float())[..., None]
    y = (y / (sum_sm + sum_ln + eps)).to(q.dtype)
    return y, a

def causal_dot_product(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """
    Causal linear attention dot product (pure PyTorch, CPU fallback).
    """
    kv = torch.einsum('bhlf,bhld->bhlfd', k, v)
    return torch.einsum('bhlf,bhlfd->bhld', q, kv.cumsum(dim=2))

def linear_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     fp32_attention: bool = False, eps: float = 1e-12,
                     initial_state: Optional[torch.Tensor] = None,
                     output_final_state: bool = False,
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Causal linear attention.

    Uses fla Triton kernels when available on CUDA (``chunk_linear_attn``
    for prefill, ``fused_recurrent_linear_attn`` for decode); otherwise falls
    back to the pure PyTorch ``causal_dot_product``.

    .. warning:: ``chunk_linear_attn`` has a ~40% prefill regression in fla
       v0.4.x vs v0.3.2.  See module docstring for details.

    Args:
        q, k: (batch, heads, seq_len, feature_dim)
        v: (batch, heads, seq_len, head_dim)
        fp32_attention: whether to accumulate in FP32 (CPU fallback only)
        eps: epsilon for numerical stability (CPU fallback only)
        initial_state: (batch, heads, feature_dim, head_dim) or None
        output_final_state: whether to return the final recurrent state
    """
    dtype = q.dtype
    try:
        if q.is_cuda:
            # (B, H, T, K) -> (B, T, H, K)
            q_t, k_t, v_t = (x.contiguous().float().transpose(1, 2).contiguous()
                              for x in (q, k, v))
            seq_len = q.shape[2]

            if seq_len == 1:
                from fla.ops.linear_attn import fused_recurrent_linear_attn
                output, final_state = fused_recurrent_linear_attn(
                    q_t, k_t, v_t, scale=1.0, normalize=True,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                )
            else:
                from fla.ops.linear_attn import chunk_linear_attn
                output, final_state = chunk_linear_attn(
                    q_t, k_t, v_t, scale=1.0, normalize=True,
                    initial_state=initial_state,
                    output_final_state=output_final_state,
                )

            output = output.transpose(1, 2).contiguous().to(dtype)
            return output, None, final_state
    except (ImportError, TypeError, KeyError):
        pass

    # CPU fallback: pure PyTorch causal dot product
    q_f, k_f, v_f = (x.contiguous().float() for x in (q, k, v))
    y = causal_dot_product(q_f, k_f, v_f)
    if fp32_attention:
        y = (y / (torch.einsum(
            "bhld,bhld->bhl", q_f, k_f.cumsum(dim=2)
        ) + eps)[..., None]).to(dtype=dtype)
    else:
        y = y.to(dtype=dtype)
        k_cum = k.float().cumsum(dim=2).to(dtype=dtype)
        y = y / (torch.einsum("bhld,bhld->bhl", q, k_cum) + eps)[..., None]

    # Compute final state: S_T = initial_state + sum_t k_t @ v_t^T
    final_state = None
    if output_final_state:
        final_state = torch.einsum('bhlf,bhld->bhfd', k_f, v_f)
        if initial_state is not None:
            final_state = final_state + initial_state
    return y, None, final_state

def new_attention_function(self, q, k, v, *args, **kwargs):
    """
    Replaces the standard attention_interface in the linear attention path.
    Applies learned feature maps and computes linear or hybrid attention.

    Supports autoregressive generation via recurrent state for pure linear
    mode. Hybrid window+linear mode does not support generation.

    q: (batch, num_heads, seq_len, head_dim)
    k: (batch, num_kv_heads, seq_len, head_dim)
    v: (batch, num_kv_heads, seq_len, head_dim)
    """
    # Handle GQA: expand k, v to match q's num_heads
    k = repeat_kv(k, self.num_key_value_groups)
    v = repeat_kv(v, self.num_key_value_groups)

    # Apply learned feature maps (per-head MLP + activation)
    if getattr(self, '_distill_mode', False):
        f_q = self.feature_map_q(q)
        f_k = self.feature_map_k(k)
        # Distillation: use quadratic attention for explicit weight matrix
        y, a, _ = quadratic_attention(f_q, f_k, v)
        return y.transpose(1, 2), a
    elif getattr(self, 'use_window', False):
        # Hybrid sliding window + linear attention (generation not supported)
        f_q = self.feature_map_q(q)
        f_k = self.feature_map_k(k)
        window_factors = torch.sigmoid(self.window_factors)
        linear_factors = 1 - window_factors
        y, a = hybrid_attention_quadratic(
            q, k, f_q, f_k, v,
            window_factors, linear_factors, self.window_size)
        return y.transpose(1, 2), a
    else:
        # Pure linear attention (efficient causal dot product)
        generation_mode = getattr(self, '_generation_mode', False)
        seq_len = q.shape[2]

        if generation_mode and seq_len == 1 and self._recurrent_state is not None:
            # Decode: single token with recurrent state
            f_q = self.feature_map_q(q)
            k_new = k[:, :, -1:]
            v_new = v[:, :, -1:]
            f_k_new = self.feature_map_k(k_new)
            y, _, state = linear_attention(
                f_q, f_k_new, v_new,
                fp32_attention=getattr(self, 'fp32_attention', False),
                initial_state=self._recurrent_state,
                output_final_state=True,
            )
            self._recurrent_state = state
            return y.transpose(1, 2), None

        # Chunked prefill: only process NEW tokens to avoid
        # double-counting cached information.  The recurrent state from
        # prior chunks is passed as initial_state.
        cache_len = k.shape[2] - seq_len
        k_lin = k[:, :, cache_len:] if cache_len > 0 else k
        v_lin = v[:, :, cache_len:] if cache_len > 0 else v

        f_q = self.feature_map_q(q)
        f_k = self.feature_map_k(k_lin)
        initial_state = self._recurrent_state if generation_mode else None
        y, _, state = linear_attention(
            f_q, f_k, v_lin,
            fp32_attention=getattr(self, 'fp32_attention', False),
            initial_state=initial_state,
            output_final_state=generation_mode,
        )
        if generation_mode:
            self._recurrent_state = state
        return y.transpose(1, 2), None
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

class FeatureMapMLP(nn.Module):
    """
    Per-head learnable MLP for feature maps.

    Full feature map is f(xW + b) where this provides the xW + b part.
    Uses per-head weight matrices of shape (num_heads, head_dim, feature_dim).
    """
    def __init__(self,
                 num_heads: int,
                 head_dim: int,
                 feature_dim: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 skip_connection: bool = False,
                 bias: bool = False,
                 zero_init: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.feature_dim = feature_dim
        self.skip_connection = skip_connection

        if skip_connection:
            assert head_dim == feature_dim, \
                f"skip_connection requires head_dim == feature_dim, got {head_dim} != {feature_dim}"

        self.layer = nn.Parameter(torch.zeros(
            (num_heads, head_dim, feature_dim), dtype=dtype, device=device,
        ))
        nn.init.kaiming_uniform_(self.layer)

        if bias:
            self.bias = nn.Parameter(torch.zeros(
                (1, num_heads, 1, 1), dtype=dtype, device=device,
            ))
            nn.init.kaiming_uniform_(self.bias)
        else:
            self.bias = 0.

        if zero_init:
            with torch.no_grad():
                if skip_connection:
                    nn.init.zeros_(self.layer)
                else:
                    for i in range(self.layer.shape[0]):
                        try:
                            nn.init.eye_(self.layer[i])
                        except RuntimeError:
                            weight = torch.eye(
                                *self.layer[i].shape,
                                device=self.layer[i].device,
                                dtype=self.layer[i].dtype,
                            )
                            self.layer[i].copy_(weight)

    def forward(self, x: torch.Tensor):
        """x: (batch_size, num_heads, seq_len, head_dim)"""
        _x = torch.einsum('hdf,bhld->bhlf', self.layer, x) + self.bias
        return x + _x if self.skip_connection else _x

class FeatureMapAct(nn.Module):
    """Base class for feature map activations"""
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, *args, **kwargs):
        return x

class Exp(FeatureMapAct):
    """
    Numerically stable exp activation (Zhang et al., 2024).
    Maps x -> [exp(x - max), exp(-x + min)]; output dim = 2 * input dim.
    """
    def forward(self, x: torch.Tensor, *args, **kwargs):
        x_max = torch.amax(x, dim=-1, keepdim=True)
        x_min = torch.amin(x, dim=-1, keepdim=True)
        return torch.cat([
            torch.exp(x - x_max), torch.exp(-x + x_min)
        ], dim=-1).clamp(min=self.eps)

class PosELU(FeatureMapAct):
    """1 + ELU activation (Katharopoulos et al., 2020)"""
    def forward(self, x: torch.Tensor, *args, **kwargs):
        return (1 + F.elu(x)).clamp(min=self.eps)

class ReLU(FeatureMapAct):
    """ReLU activation (Kasai et al., 2021 - T2R)"""
    def forward(self, x: torch.Tensor, *args, **kwargs):
        return F.relu(x).clamp(min=self.eps)

class SoftmaxDim(FeatureMapAct):
    """
    Concatenated softmax activation (Zhang et al., 2024 - Hedgehog).
    Maps x -> [softmax(x), softmax(-x)]; output dim = 2 * input dim.
    """
    def forward(self, x: torch.Tensor, *args, **kwargs):
        return torch.cat([
            torch.softmax(x, dim=-1), torch.softmax(-x, dim=-1)
        ], dim=-1).clamp(min=self.eps)

def init_feature_map_act(name: str, fullspace: bool = True, **kwargs):
    """Initialize feature map activation by name."""
    if name == 'softmax_dim':
        return SoftmaxDim(**kwargs)
    elif name == 'exp':
        return Exp(**kwargs)
    elif name == 'pos_elu':
        return PosELU(**kwargs)
    elif name == 'relu':
        return ReLU(**kwargs)
    else:
        raise NotImplementedError(f"Unknown feature map activation: {name}")

class FeatureMap(nn.Module):
    """
    Complete feature map: phi(x) = activation(MLP(x)).
    Combines a learnable per-head MLP with an activation function.
    """
    def __init__(self,
                 activation_name: str = 'softmax_dim',
                 mlp: nn.Module = None,
                 fullspace: bool = True,
                 eps: float = 1e-12):
        super().__init__()
        self.mlp = mlp if mlp is not None else nn.Identity()
        self.activation = init_feature_map_act(activation_name, fullspace, eps=eps)

    def forward(self, x: torch.Tensor):
        """x: (batch_size, num_heads, seq_len, head_dim)"""
        return self.activation(self.mlp(x))

    def q_map(self, x: torch.Tensor):
        return self.forward(x)

    def k_map(self, x: torch.Tensor):
        return self.forward(x)

def init_feature_maps_qk(self,
                          feature_map='softmax_dim',
                          feature_dim=None,
                          tie_qk_kernels=False,
                          skip_connection=False,
                          bias=False,
                          zero_init=False,
                          window_size=None,
                          train_window_factor=True,
                          init_window_factor=-2.197,
                          fp32_attention=False):
    """
    Initialize LoLCATs feature maps and (optionally) sliding window parameters.
    Called at the end of __init__ for the converted attention class.

    Args:
        feature_map: Activation type ('softmax_dim', 'exp', 'pos_elu', 'relu')
        feature_dim: Output dim of feature map MLP (default: head_dim)
        tie_qk_kernels: Whether to share Q and K feature maps
        skip_connection: Whether MLP has residual connection
        bias: Whether MLP has bias term
        zero_init: Whether to zero-initialize MLP weights
        window_size: Sliding window size (None = pure linear attention)
        train_window_factor: Whether window interpolation weights are learnable
        init_window_factor: Initial logit for window factor (sigmoid applied)
        fp32_attention: Whether to accumulate attention in FP32
    """
    # SWA(64) hybrid (sliding-window softmax + linear) is the canonical, strongly
    # recommended LoLCATs setup -- the window handles the sharp, local (adjacent-token)
    # attention that pure linear attention reproduces poorly.  Resolve the window size
    # with precedence: explicit window_size arg > config.lolcats_window_size > default 64.
    # Pass window_size=0 (or set lolcats_window_size=0) to force pure linear attention.
    if window_size is None:
        window_size = getattr(self.config, 'lolcats_window_size', 64) if hasattr(self, 'config') else 64

    self.mode = 'linear'
    self.fp32_attention = fp32_attention
    self._distill_mode = False
    self._recurrent_state = None
    self._generation_mode = False
    # Standard MHA (e.g. GPT-2) has no GQA; default to 1 so repeat_kv is a no-op
    if not hasattr(self, 'num_key_value_groups'):
        self.num_key_value_groups = 1
    # Ensure attention_interface path is used, not _upcast_and_reordered_attn
    if hasattr(self, 'reorder_and_upcast_attn'):
        self.reorder_and_upcast_attn = False

    if feature_dim is None:
        feature_dim = self.head_dim

    num_heads = self.config.num_attention_heads if hasattr(self, 'config') else self.num_heads

    # Create per-head learned MLP
    # Use next(self.parameters()) for dtype/device to be model-agnostic
    # (e.g. Phi-3 uses fused qkv_proj instead of separate q_proj)
    _ref_param = next(self.parameters())
    mlp = FeatureMapMLP(
        num_heads=num_heads,
        head_dim=self.head_dim,
        feature_dim=feature_dim,
        dtype=_ref_param.dtype,
        device=_ref_param.device,
        skip_connection=skip_connection,
        bias=bias,
        zero_init=zero_init,
    )

    # Create feature map (MLP + activation)
    self.feature_map_q = FeatureMap(activation_name=feature_map, mlp=mlp)
    if tie_qk_kernels:
        self.feature_map_k = self.feature_map_q
    else:
        self.feature_map_k = deepcopy(self.feature_map_q)

    # Sliding window configuration
    self.use_window = window_size is not None and window_size > 0
    if not self.use_window:
        warnings.warn(
            f"LoLCATs sliding-window attention is DISABLED (resolved window_size={window_size!r}). "
            "This is HIGHLY NOT RECOMMENDED: without the sliding window, the linear feature map "
            "alone must reproduce the sharp, local (adjacent-token) attention that the SWA(64) "
            "hybrid is designed to handle, which typically degrades quality substantially. The "
            "canonical LoLCATs configuration uses a sliding window of 64 (set lolcats_window_size=64) "
            "unless pure linear attention is specifically intended.",
            UserWarning, stacklevel=2,
        )
    if self.use_window:
        self.window_size = window_size
        device = _ref_param.device
        dtype = _ref_param.dtype
        if train_window_factor:
            self.window_factors = nn.Parameter(
                init_window_factor * torch.ones(
                    1, num_heads, 1, 1, device=device, dtype=dtype))
        else:
            self.register_buffer(
                "window_factors",
                init_window_factor * torch.ones(
                    1, num_heads, 1, 1, device=device, dtype=dtype))
@auto_docstring(
    custom_intro="""
    The GPT2 Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """
)
class LoLCATsGPT2LMHeadModel(GPT2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.transformer = LoLCATsGPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        transformer_outputs: BaseModelOutputWithPastAndCrossAttentions = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
@auto_docstring
class LoLCATsGPT2Model(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([LoLCATsGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False
        self._attn_implementation = config._attn_implementation

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        kwargs.pop("output_attentions", None)
        kwargs.pop("output_hidden_states", None)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache(config=self.config))

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        encoder_attention_mask = None
        if encoder_hidden_states is not None:
            encoder_attention_mask = create_bidirectional_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states,
                past_key_values if not (self.gradient_checkpointing and self.training) else None,
                cache_position,
                causal_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                position_ids=position_ids,
                **kwargs,
            )

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)

        past_key_values = past_key_values if use_cache else None
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
class LoLCATsGPT2Block(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = LoLCATsGPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = LoLCATsGPT2Attention(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, _ = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kwargs,
        )
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_output, _ = self.crossattention(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            # residual connection
            hidden_states = residual + cross_attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states
class LoLCATsGPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = not is_cross_attention
        init_feature_maps_qk(self)

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None):
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()

        # Preallocate attn_weights for `baddbmm`
        attn_weights = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)

        # Compute Scale Factor
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor /= float(value.size(-1)) ** 0.5

        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx + 1)

        # Upcast (turn off autocast) and reorder (Scale K by 1 / root(dk))
        with maybe_autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op if otherwise
        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2)

        return attn_output, attn_weights

    def linear_forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]:
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attn_output, attn_weights = new_attention_function(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights
    def quadratic_forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]:
        is_cross_attention = encoder_hidden_states is not None
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights
    def forward(self, *args, **kwargs):
        """
    LoLCATs attention forward with mode-based dispatch.

    Modes:
        'linear': Inference with learned linear attention (or hybrid window+linear)
        'quadratic': Standard softmax attention (teacher / ground truth)
        'both': Distillation mode - returns teacher output plus paired
                (student_attn, teacher_attn, student_output) for loss computation
    """
        assert self.mode in {'quadratic', 'linear', 'both'}
        if self.mode == 'linear':
            return self.linear_forward(*args, **kwargs)
        elif self.mode == 'quadratic':
            return self.quadratic_forward(*args, **kwargs)
        else:
            # Distillation: compute student (feature-mapped) and teacher (softmax)
            self._distill_mode = True
            student_output, student_attn = self.linear_forward(*args, **kwargs)
            self._distill_mode = False
            with torch.no_grad():
                teacher_output, teacher_attn = self.quadratic_forward(*args, **kwargs)
            return teacher_output, (student_attn, teacher_attn, student_output)
