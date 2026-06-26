# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MoE grug variant model with adaptive (variable-k) sparsity.

Architecture: QB-routed MoE with GatedNorm, XSA, sigmoid combine weights. Loss-free
(bias) load balancing + router z-loss only; no auxiliary load-balancing loss. All
layers are MoE (no dense layers).

Routing is fixed top-k by default. With ``adaptive_routing=True`` the per-token
expert count is *variable*: each token selects the top-``num_experts_per_token``
candidates (the dispatch capacity K_max) but keeps only those whose biased router
logit clears a learned per-layer threshold (always keeping ``min_experts_per_token``).
A straight-through estimator makes the forward pass truly sparse while letting
cross-entropy pull the threshold down and a soft sparsity penalty
(``sparsity_loss_coef * E[active_fraction]``) push it up — so the model is
conditioned toward more sparsity wherever the prediction loss permits. See the
experiment README and ``MoEMLP._adaptive_gate``.
"""

import dataclasses
import math
import os
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from einops import rearrange
from haliax.jax_utils import named_call
from jax import random
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh, reshard

try:
    from jax.shard_map import shard_map
except ModuleNotFoundError:
    from jax.experimental.shard_map import shard_map
from jaxtyping import Array, Float, Int, PRNGKeyArray
from levanter.grug.attention import (
    AttentionMask,
    GrugAttentionImplementation,
    RotaryConfig,
    align_kv_heads,
    apply_rotary_embedding,
    attention,
)
from levanter.grug.grug_moe import (
    MOE_REMAT_SAVE_NAMES,
    MoeActivation,
    MoEExpertMlp,
    MoeImplementation,
    resolve_moe_implementation,
)
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.grug.sharding import Pembed_vocab, Plm_head, unshard

# Monkeypatch (import for side effect): clamp the fused-CE pallas weight tile so the
# streaming kernel is used for vocab 128256 on H100 instead of falling back to XLA
# (which materializes the full [tokens, vocab] logits and OOMs the 15B model). No-op
# off NVIDIA. MUST be imported before any fused_linear_softmax_cross_entropy_loss call.
import ce_kernel_patch  # noqa: E402,F401  isort:skip

# Force the MoE grouped matmul (haliax.nn.ragged_dot) onto the Pallas-Triton kernel.
# ragged_dot reads RAGGED_DOT_IMPL from os.environ at *call* time; the "auto" default on
# GPU silently falls back to jax.lax.ragged_dot_general (a per-device dense [M, G, N]
# materialization) which OOMs hundreds of GiB at this deeply-sparse 15B geometry. Setting
# it here is robust to iris/Fray env-forwarding gaps (this module is imported by the train
# task before any MoE call). An explicit RAGGED_DOT_IMPL in the environment still wins.
os.environ.setdefault("RAGGED_DOT_IMPL", "triton")
from levanter.tracker.histogram import Histogram, SummaryStats
from levanter.utils.activation import ActivationFunctionEnum

_DEFAULT_EP_CAPACITY_FACTOR = 1.0
_GATED_NORM_RANK = 128
# Initial value of the learned adaptive routing threshold. Set well below the
# initial router-logit scale so every top-K_max slot clears it at step 0: training
# starts dense and the sparsity penalty anneals the threshold upward.
_ADAPTIVE_THRESHOLD_INIT = -4.0


_BATCH_AXES: tuple[str, ...] = ("replica_dcn", "data", "expert")


def _mesh_axis_size(mesh: jax.sharding.AbstractMesh | None, axis_name: str) -> int:
    if mesh is None or mesh.empty:
        raise ValueError("grug/moe requires a non-empty abstract mesh")
    if axis_name not in mesh.shape:
        # compact_grug_mesh standardizes on (replica_dcn, data, expert, model) with length-1
        # axes kept, so any missing axis is a caller bug rather than a "size 1" shortcut.
        raise ValueError(f"grug/moe requires an abstract mesh with axis '{axis_name}'")
    return int(mesh.shape[axis_name])


RematMode = Literal["recompute_all", "save_moe"]


def _batch_spec() -> P:
    return P(_BATCH_AXES)


def _batch_reshard(x: jax.Array) -> jax.Array:
    return reshard(x, _batch_spec())


def _layer_attention_masks(
    mask: AttentionMask, *, sliding_window: int, local_window: int = 0
) -> tuple[AttentionMask, AttentionMask]:
    local = local_window if local_window > 0 else sliding_window // 2
    return mask.with_sliding_window(local), mask.with_sliding_window(sliding_window)


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the grug MoE transformer.

    Architecture choices (GatedNorm, XSA, QB routing) are hardcoded.
    Only shape/size knobs live here. All layers are MoE.
    """

    vocab_size: int
    hidden_dim: int = 2048
    embed_dim: int | None = None
    """Factorized embedding / CE dimension (ALBERT-style). If set < ``hidden_dim``, the
    token embedding and the LM head operate at this narrow width: ``token_embed[V, d_e]``
    -> up-project ``[d_e, D]`` on input, and ``[D, d_e]`` down-project -> ``output_proj[d_e, V]``
    on output. This shrinks the lm_head FLOP from ``2*D*V`` to ``~2*d_e*V`` while keeping
    vocab and ``hidden_dim`` large -- moving compute out of the head and into the experts
    WITHOUT changing the tokenizer/vocab. ``None`` (or ``== hidden_dim``) = no factorization
    (original behavior)."""
    intermediate_dim: int = 5632
    shared_expert_intermediate_dim: int = 5632
    num_experts: int = 8
    num_experts_per_token: int = 2
    """Top-k routing width. In adaptive mode this is the dispatch *capacity* K_max
    (the most experts any token may use); the realized per-token count is variable
    and set by the learned threshold."""
    adaptive_routing: bool = False
    """If True, route a *variable* number of experts per token: select the top
    ``num_experts_per_token`` candidates, then keep only those whose router logit
    exceeds a learned per-layer threshold (always keeping at least
    ``min_experts_per_token``). Dropped slots get a zero combine weight, so an
    easy token can fall through to the always-on shared expert alone."""
    min_experts_per_token: int = 0
    """Floor on the per-token active expert count in adaptive mode."""
    sparsity_loss_coef: float = 0.0
    """Weight on the soft sparsity penalty ``coef * E[active_fraction]``. Conditions
    the model toward fewer active experts unless the cross-entropy loss resists."""
    sparsity_temp: float = 1.0
    """Temperature of the soft keep-gate ``sigmoid((logit - theta) / temp)`` behind
    the differentiable expected-active-count penalty."""
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None
    max_seq_len: int = 4096
    sliding_window: int = 4096
    global_attn_every: int = 4
    """Global/local attention interleave: every Nth layer uses GLOBAL (full sliding_window)
    attention, the rest use a small LOCAL window. Period = global_attn_every (4 -> 3 local:1
    global, the legacy pattern; 6 -> 5:1; 8 -> 7:1). Larger = fewer global layers = cheaper attn."""
    local_window: int = 0
    """Window size for the LOCAL (non-global) layers. 0 -> sliding_window // 2 (legacy). Set small
    (e.g. 512/1024) to make local layers cheap regardless of seq length."""
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    qk_mult: float = 1.0
    router_z_loss_coef: float = 0.001
    attention_implementation: GrugAttentionImplementation | None = None
    moe_implementation: MoeImplementation | None = None
    remat_mode: RematMode = "recompute_all"
    """Per-block gradient checkpointing. "recompute_all" reruns the whole block in
    backward (lowest memory); "save_moe" keeps the tagged MoE dispatch tensors so
    backward skips re-running expert dispatch and its EP collectives."""
    fast_qb_beta: bool = False
    """If True, compute the QB (loss-free balancing) per-expert threshold β with
    ``jax.lax.approx_max_k`` instead of an exact ``top_k`` sort. β only needs the
    qb_count-th largest logit per expert and feeds a stop-gradient'd router bias, so an
    approximate quantile is tolerable. Profiling showed the exact sort over
    ``[E, tokens]`` dominates device time (~35%); this is the main throughput lever.
    Kept off by default so the routing dynamics match the exact-β baseline runs."""
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)

    def __post_init__(self) -> None:
        _ = self.inferred_head_dim
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.num_experts_per_token <= 0:
            raise ValueError("num_experts_per_token must be positive")
        if self.num_experts_per_token > self.num_experts:
            raise ValueError("num_experts_per_token must be <= num_experts")
        if self.shared_expert_intermediate_dim < 0:
            raise ValueError("shared_expert_intermediate_dim must be non-negative")
        if self.min_experts_per_token < 0:
            raise ValueError("min_experts_per_token must be non-negative")
        if self.min_experts_per_token > self.num_experts_per_token:
            raise ValueError("min_experts_per_token must be <= num_experts_per_token (the K_max capacity)")
        resolve_moe_implementation(self.moe_implementation)

    @property
    def inferred_embed_dim(self) -> int:
        """The CE/embedding width. Defaults to hidden_dim (no factorization)."""
        return self.embed_dim if self.embed_dim is not None else self.hidden_dim

    @property
    def is_factorized_embed(self) -> bool:
        return self.inferred_embed_dim != self.hidden_dim

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads


def rms_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Non-parametric RMS norm over the last dimension."""
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + eps)).astype(x.dtype)


class CausalSelfAttention(eqx.Module):
    w_q: Float[Array, "D NH"]
    w_k: Float[Array, "D MH"]
    w_v: Float[Array, "D MH"]
    w_o: Float[Array, "NH D"]
    attn_gate: Float[Array, "D N"]
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "CausalSelfAttention":
        k_q, k_k, k_v, k_o = random.split(key, 4)
        d, n, m, h = cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.inferred_head_dim
        return CausalSelfAttention(
            w_q=reshard(_init_weight(k_q, (d, n * h), cfg.initializer_std), P("data", "model")),
            w_k=reshard(_init_weight(k_k, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_v=reshard(_init_weight(k_v, (d, m * h), cfg.initializer_std), P("data", "model")),
            w_o=reshard(_init_weight(k_o, (n * h, d), cfg.initializer_std), P("model", "data")),
            attn_gate=reshard(jnp.zeros((d, n)), P(None, None)),
            cfg=cfg,
        )

    @named_call
    def __call__(self, x: Float[Array, "B S D"], mask: AttentionMask | jax.Array) -> Float[Array, "B S D"]:
        head_dim = self.cfg.inferred_head_dim
        seq_len = x.shape[1]
        batch_spec = _batch_spec()

        q = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_q), "... (n d) -> ... n d", d=head_dim)
        k = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_k), "... (m d) -> ... m d", d=head_dim)
        v = rearrange(jnp.einsum("bsh,hd->bsd", x, self.w_v), "... (m d) -> ... m d", d=head_dim)
        q = rms_norm(q)
        k = rms_norm(k)
        q, k = apply_rotary_embedding(q, k, seq_len=seq_len, head_dim=head_dim, rope=self.cfg.rope)
        q = q * self.cfg.qk_mult
        attn_out = attention(q, k, v, mask, implementation=self.cfg.attention_implementation)
        # The GPU `reference_attention` backend returns attn_out with head_dim on the
        # `model` axis, whereas the TPU splash backend returns it with heads on `model`.
        # Pin attn_out to the same (heads-on-model) spec as aligned_v below so the XSA
        # multiply is consistently sharded under explicit mesh axes (illegal otherwise).
        attn_out = reshard(attn_out, P(_BATCH_AXES, None, "model", None))
        aligned_v = align_kv_heads(v, num_q_heads=attn_out.shape[2])
        aligned_v = reshard(aligned_v, P(_BATCH_AXES, None, "model", None))
        # Exclusive Self Attention: subtract the component of yᵢ parallel to vᵢ.
        # zᵢ = yᵢ - (yᵢᵀvᵢ / ‖vᵢ‖²) vᵢ, per head.
        dot = jnp.sum(attn_out * aligned_v, axis=-1, keepdims=True)
        v_norm_sq = jnp.sum(aligned_v * aligned_v, axis=-1, keepdims=True)
        attn_out = attn_out - (dot / (v_norm_sq + 1e-6)) * aligned_v
        # Headwise gating: sigmoid(x @ attn_gate) produces one scalar per head.
        gate = 2 * jax.nn.sigmoid(jnp.einsum("bsd,dn->bsn", x, self.attn_gate))[..., None]
        attn_out = gate * attn_out
        attn_out = rearrange(attn_out, "... n d -> ... (n d)")
        return jnp.einsum("bsh,hd->bsd", attn_out, self.w_o, out_sharding=batch_spec)


class RMSNorm(eqx.Module):
    weight: jax.Array
    eps: float = eqx.field(static=True)

    @staticmethod
    def init(dim: int, eps: float) -> "RMSNorm":
        return RMSNorm(weight=jnp.ones((dim,), dtype=jnp.float32), eps=eps)

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        weight = unshard(self.weight)
        dtype = x.dtype
        x = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed = x * jax.lax.rsqrt(variance + self.eps)
        return (normed * weight).astype(dtype)


class GatedNorm(eqx.Module):
    """Learnable per-dimension gating. Compensates for AdamH's bounded activation norms.
    See https://arxiv.org/abs/2601.22966v1"""

    w_down: jax.Array
    w_up: jax.Array

    @staticmethod
    def init(hidden_dim: int, initializer_std: float, *, key: PRNGKeyArray) -> "GatedNorm":
        k_down, k_up = random.split(key)
        return GatedNorm(
            w_down=reshard(_init_weight(k_down, (hidden_dim, _GATED_NORM_RANK), initializer_std), P(None, None)),
            w_up=reshard(_init_weight(k_up, (_GATED_NORM_RANK, hidden_dim), initializer_std), P(None, None)),
        )

    @named_call
    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        gate_hidden = jnp.einsum("...d,dr->...r", x, self.w_down)
        # TODO: silu activation here isn't explored, just cargo-culted from Qwen. Likely low-hanging ablation fruit
        # (e.g. compare no activation, relu, etc.).
        gate_hidden = jax.nn.silu(gate_hidden)
        gate = jax.nn.sigmoid(jnp.einsum("...r,rd->...d", gate_hidden, self.w_up))
        return x * gate.astype(x.dtype)


class DenseMLP(eqx.Module):
    w_gate: jax.Array
    w_up: jax.Array
    w_down: jax.Array

    @staticmethod
    def init(hidden_dim: int, intermediate_dim: int, initializer_std: float, *, key: PRNGKeyArray) -> "DenseMLP":
        k_gate, k_up, k_down = random.split(key, 3)
        return DenseMLP(
            w_gate=reshard(_init_weight(k_gate, (hidden_dim, intermediate_dim), initializer_std), P("data", "model")),
            w_up=reshard(_init_weight(k_up, (hidden_dim, intermediate_dim), initializer_std), P("data", "model")),
            w_down=reshard(_init_weight(k_down, (intermediate_dim, hidden_dim), initializer_std), P("model", "data")),
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        *,
        activation: MoeActivation = ActivationFunctionEnum.silu,
    ) -> Float[Array, "B S D"]:
        if isinstance(activation, ActivationFunctionEnum):
            activation_fn = activation.to_jax_fn()
        else:
            activation_fn = activation

        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        gate = jnp.einsum("td,dm->tm", x_flat, self.w_gate)
        up = jnp.einsum("td,dm->tm", x_flat, self.w_up)
        out_flat = jnp.einsum("tm,md->td", activation_fn(gate) * up, self.w_down, out_sharding=_batch_spec())
        # Reshard after the reshape so the shared-expert output carries the same
        # canonical batch sharding as the routed MoE output (MoEMLP reshards its
        # routed result identically). Splitting the fused
        # ("replica_dcn", "data", "expert") token axis back into (b, s) otherwise
        # leaks the `expert` mesh axis onto the seq dim, so the shared+routed
        # residual add fails with a ShardingTypeError on a multi-node mesh.
        return _batch_reshard(rearrange(out_flat, "(b s) d -> b s d", b=b, s=s))


def _routing_stats(
    selected_experts: Int[Array, "T K"],
    router_probs: Float[Array, "T E"],
    router_logits: Float[Array, "T E"],
    *,
    num_experts: int,
    num_experts_per_token: int,
) -> dict[str, jax.Array]:
    router_probs_f = router_probs.astype(jnp.float32)
    router_logits_f = router_logits.astype(jnp.float32)
    expert_counts = jnp.sum(jax.nn.one_hot(selected_experts, num_experts, dtype=jnp.float32), axis=(0, 1))
    total_assignments = jnp.maximum(jnp.sum(expert_counts), 1.0)
    assignment_fraction = expert_counts / total_assignments
    routing_entropy = -jnp.sum(assignment_fraction * jnp.log(assignment_fraction + 1e-6))
    token_fraction = assignment_fraction * num_experts_per_token
    p = jnp.mean(router_probs_f, axis=0)
    load_balancing_loss = num_experts * jnp.sum(token_fraction * p)
    z = jsp.special.logsumexp(router_logits_f, axis=-1)
    router_z_loss = jnp.mean(z**2)

    return {
        "routing_counts": expert_counts,
        "routing_entropy": routing_entropy,
        "load_balancing_loss": load_balancing_loss,
        "router_z_loss": router_z_loss,
    }


def _summarize_router_metrics(router_metrics: dict[str, jax.Array]) -> dict[str, jax.Array | SummaryStats]:
    routing_entropy = router_metrics["routing_entropy_per_layer"]
    routing_counts = router_metrics["routing_counts_per_layer"]
    load_balancing_loss = router_metrics["load_balancing_loss_per_layer"]
    router_z_loss = router_metrics["router_z_loss_per_layer"]
    expected_active = router_metrics["expected_active_frac_per_layer"]
    realized_active = router_metrics["realized_active_frac_per_layer"]
    router_threshold = router_metrics["router_threshold_per_layer"]
    num_layers = int(routing_entropy.shape[0])

    out: dict[str, jax.Array | SummaryStats] = {
        "train/router/routing_entropy_mean": jnp.mean(routing_entropy),
        "train/router/load_balancing_loss": jnp.mean(load_balancing_loss),
        "train/router/router_z_loss": jnp.mean(router_z_loss),
        "train/router/routing_counts_per_layer": routing_counts,
        # Headline sparsity readouts: the realized (hard) and expected (soft) active
        # fractions of the expert pool, averaged across layers.
        "train/router/sparsity/realized_active_frac": jnp.mean(realized_active),
        "train/router/sparsity/expected_active_frac": jnp.mean(expected_active),
        "train/router/sparsity/threshold_mean": jnp.mean(router_threshold),
        "qb_beta_per_layer": router_metrics.get("qb_beta_per_layer"),
    }
    for i in range(num_layers):
        out[f"train/router/layer_{i}/routing_entropy"] = routing_entropy[i]
        out[f"train/router/layer_{i}/load_balancing_loss"] = load_balancing_loss[i]
        out[f"train/router/layer_{i}/router_z_loss"] = router_z_loss[i]
        out[f"train/router/layer_{i}/routing_hist"] = _histogram_from_expert_counts(routing_counts[i])
        out[f"train/router/sparsity/layer_{i}/realized_active_frac"] = realized_active[i]
        out[f"train/router/sparsity/layer_{i}/threshold"] = router_threshold[i]
    return out


# wandb caps a histogram at 512 buckets; with a large expert pool (e.g. E=1024) the raw
# per-expert load histogram overflows and the whole log update is dropped. Coarsen to at
# most this many contiguous bins so the load diagnostic still logs (scalar stats below are
# always computed over the full pool, so min/max/mean/rms are unaffected).
_MAX_HIST_BUCKETS = 256


def _histogram_from_expert_counts(expert_counts: jax.Array) -> SummaryStats:
    counts = jnp.asarray(expert_counts, dtype=jnp.float32)
    num_experts = counts.shape[0]
    expert_ids = jnp.arange(num_experts, dtype=jnp.float32)
    num = jnp.sum(counts)
    sum_values = jnp.sum(counts * expert_ids)
    sum_squares = jnp.sum(counts * expert_ids * expert_ids)
    nonzero = counts > 0
    min_value = jnp.where(nonzero, expert_ids, jnp.inf).min()
    max_value = jnp.where(nonzero, expert_ids, -jnp.inf).max()
    min_value = jnp.where(num > 0, min_value, 0.0)
    max_value = jnp.where(num > 0, max_value, 0.0)
    if num_experts > _MAX_HIST_BUCKETS:
        group = math.ceil(num_experts / _MAX_HIST_BUCKETS)
        pad = (-num_experts) % group
        hist_counts = jnp.concatenate([counts, jnp.zeros(pad, counts.dtype)]) if pad else counts
        n_bins = hist_counts.shape[0] // group
        hist_counts = hist_counts.reshape(n_bins, group).sum(axis=1)
        bucket_limits = jnp.arange(n_bins + 1, dtype=jnp.float32) * group
    else:
        hist_counts = counts
        bucket_limits = jnp.arange(num_experts + 1, dtype=jnp.float32)
    histogram = Histogram(bucket_limits=bucket_limits, bucket_counts=hist_counts)
    return SummaryStats.from_reduced_values(
        min=min_value,
        max=max_value,
        num=num,
        nonzero_count=jnp.sum(nonzero),
        sum=sum_values,
        sum_squares=sum_squares,
        histogram=histogram,
    )


class MoEMLP(eqx.Module):
    """QB-routed MoE with sigmoid combine weights.

    Routing is fixed top-k by default. In adaptive mode (``cfg.adaptive_routing``)
    the per-token expert count is *variable*: we still select the top
    ``num_experts_per_token`` candidates (the dispatch capacity K_max), but keep
    only those whose biased router logit clears the learned ``router_threshold``.
    Dropped slots get a zero combine weight, so they cost nothing in the combine
    and an easy token can route to zero experts (shared expert only). The expected
    (soft) active count is penalized in the loss to push the model sparser.
    """

    router: jax.Array
    router_bias: jax.Array
    router_threshold: jax.Array  # learned per-layer scalar; only used in adaptive mode
    expert_mlp: MoEExpertMlp
    cfg: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "MoEMLP":
        k_router, k_expert_mlp = random.split(key, 2)
        mesh = get_abstract_mesh()

        expert_axis_size = _mesh_axis_size(mesh, "expert")
        if cfg.num_experts % expert_axis_size != 0:
            raise ValueError(f"num_experts={cfg.num_experts} must be divisible by expert axis size={expert_axis_size}")

        d, e, i = cfg.hidden_dim, cfg.num_experts, cfg.intermediate_dim

        return MoEMLP(
            router=reshard(_init_weight(k_router, (d, e), cfg.initializer_std), P(None, None)),
            router_bias=jnp.zeros((e,)),
            # Init low so every top-K_max slot clears the threshold initially: training
            # starts dense (≈ fixed top-K_max) and the sparsity penalty anneals the
            # threshold upward, dropping experts only where cross-entropy permits. This
            # dense→sparse trajectory (à la ReMoE, arXiv:2412.14711) avoids starving
            # experts before they have trained.
            router_threshold=jnp.full((), _ADAPTIVE_THRESHOLD_INIT, dtype=jnp.float32),
            expert_mlp=MoEExpertMlp.init(
                num_experts=e,
                hidden_dim=d,
                intermediate_dim=i,
                initializer_std=cfg.initializer_std,
                key=k_expert_mlp,
                implementation=cfg.moe_implementation,
                activation=ActivationFunctionEnum.silu,
                capacity_factor=_DEFAULT_EP_CAPACITY_FACTOR,
            ),
            cfg=cfg,
        )

    def _adaptive_gate(
        self,
        topk_biased: Float[Array, "T K"],
        combine_weights: Float[Array, "T K"],
        k_max: int,
    ) -> tuple[Float[Array, "T K"], dict[str, jax.Array]]:
        """Variable-k keep gate for adaptive routing.

        Keeps only the top-K_max slots whose biased router logit clears the learned
        ``router_threshold`` (always keeping the strongest ``min_experts_per_token``).
        A straight-through estimator makes the forward pass *truly* sparse (hard
        keep) while routing the backward gradient through a soft sigmoid surrogate,
        so cross-entropy can pull the threshold down (recover a useful expert) while
        the sparsity penalty pushes it up. Without the surrogate the threshold sees
        only the penalty gradient and collapses to the floor.
        """
        cfg = self.cfg
        theta = self.router_threshold
        k_min = cfg.min_experts_per_token
        hard_keep = topk_biased > theta
        soft_keep = jax.nn.sigmoid((topk_biased - theta) / cfg.sparsity_temp)
        if k_min > 0:
            # top_k returns slots in descending logit order, so the strongest k_min
            # candidates are the first columns; force them on as the floor.
            floor = jnp.arange(k_max) < k_min
            hard_keep = hard_keep | floor[None, :]
            soft_keep = jnp.where(floor[None, :], 1.0, soft_keep)
        keep_st = hard_keep.astype(jnp.float32) + (soft_keep - jax.lax.stop_gradient(soft_keep))
        combine_weights = combine_weights * keep_st
        stats = {
            "expected_active": jnp.sum(soft_keep, axis=1),
            "realized_active": jnp.sum(hard_keep.astype(jnp.float32), axis=1),
            "router_threshold": theta,
        }
        return combine_weights, stats

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        cfg = self.cfg
        b, s, _ = x.shape
        x_flat = rearrange(x, "b s d -> (b s) d")
        k_max = cfg.num_experts_per_token
        # Keep the router path in fp32 before top-k, softmax, and QB statistics.
        router_logits = jnp.einsum("td,de->te", x_flat, reshard(self.router, P(None, None))).astype(jnp.float32)
        biased_logits = router_logits + jax.lax.stop_gradient(self.router_bias)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        # Select top-(K_max+1) on biased logits; the (K_max+1)-th is the QB threshold alpha.
        topk_biased, selected_experts = jax.lax.top_k(biased_logits, k_max + 1)
        qb_alpha = topk_biased[:, -1:]
        topk_biased = topk_biased[:, :-1]
        selected_experts = selected_experts[:, :-1]
        # Sigmoid combine weights on unbiased logits for selected experts.
        unbiased_topk = jnp.take_along_axis(router_logits, selected_experts, axis=-1)
        combine_weights = jax.nn.sigmoid(unbiased_topk)

        if cfg.adaptive_routing:
            combine_weights, sparsity_stats = self._adaptive_gate(topk_biased, combine_weights, k_max)
        else:
            num_tokens = x_flat.shape[0]
            full = jnp.full((num_tokens,), float(k_max), dtype=jnp.float32)
            sparsity_stats = {
                "expected_active": full,
                "realized_active": full,
                "router_threshold": self.router_threshold,
            }
        combine_weights = combine_weights.astype(x.dtype)

        router_stats = _routing_stats(
            selected_experts,
            router_probs,
            router_logits,
            num_experts=cfg.num_experts,
            num_experts_per_token=k_max,
        )
        # Per-token active-expert counts -> scalar fractions of the full expert pool.
        router_stats["expected_active_frac"] = jnp.mean(sparsity_stats["expected_active"]) / cfg.num_experts
        router_stats["realized_active_frac"] = jnp.mean(sparsity_stats["realized_active"]) / cfg.num_experts
        # The penalized quantity: the differentiable expected active fraction.
        router_stats["sparsity_loss"] = jnp.mean(sparsity_stats["expected_active"]) / cfg.num_experts
        router_stats["router_threshold"] = sparsity_stats["router_threshold"]
        # Sharded QB: compute beta locally per device, then average.
        mesh = get_abstract_mesh()
        s_minus_alpha = reshard(router_logits - qb_alpha, P(_BATCH_AXES, None))
        num_devices = 1
        for a in _BATCH_AXES:
            num_devices *= mesh.shape[a]
        local_tokens = s_minus_alpha.shape[0] // num_devices
        qb_count = max(1, local_tokens * self.cfg.num_experts_per_token // self.cfg.num_experts)

        fast_qb = self.cfg.fast_qb_beta

        def _local_qb_beta(s_ma):
            if fast_qb:
                # β is the qb_count-th largest logit per expert (a high quantile). The exact
                # top_k sort over [E, local_tokens] dominates device time; approx_max_k is a
                # TPU-native partial reduction, and β = min of the approximate top set.
                approx_vals, _ = jax.lax.approx_max_k(s_ma.T, qb_count, recall_target=0.95)
                beta = jnp.min(approx_vals, axis=-1)
            else:
                topk_vals, _ = jax.lax.top_k(s_ma.T, qb_count)
                beta = topk_vals[:, -1]
            return jax.lax.pmean(beta, axis_name=_BATCH_AXES)

        router_stats["qb_beta"] = shard_map(
            _local_qb_beta,
            mesh=mesh,
            in_specs=(P(_BATCH_AXES, None),),
            out_specs=P(),
        )(s_minus_alpha)

        routed_flat = self.expert_mlp(
            x_flat,
            selected_experts.astype(jnp.int32),
            combine_weights,
            mesh=get_abstract_mesh(),
        )

        routed = rearrange(routed_flat, "(b s) d -> b s d", b=b, s=s)
        routed = reshard(routed, _batch_spec())
        return routed, router_stats


class Block(eqx.Module):
    rms_attn: RMSNorm
    attn_gated_norm: GatedNorm
    attn: CausalSelfAttention
    rms_mlp: RMSNorm
    mlp_gated_norm: GatedNorm
    mlp: MoEMLP
    shared: DenseMLP | None

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Block":
        attn_key, mlp_key, shared_key, gn_attn_key, gn_mlp_key = random.split(key, 5)
        shared = None
        if cfg.shared_expert_intermediate_dim > 0:
            shared = DenseMLP.init(
                cfg.hidden_dim, cfg.shared_expert_intermediate_dim, cfg.initializer_std, key=shared_key
            )
        return Block(
            rms_attn=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            attn_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_attn_key),
            attn=CausalSelfAttention.init(cfg, key=attn_key),
            rms_mlp=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            mlp_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=gn_mlp_key),
            mlp=MoEMLP.init(cfg, key=mlp_key),
            shared=shared,
        )

    @named_call
    def __call__(
        self,
        x: Float[Array, "B S D"],
        mask: AttentionMask | jax.Array,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        attn_in = self.attn_gated_norm(self.rms_attn(x))
        x = _batch_reshard(x + self.attn(attn_in, mask))
        mlp_in = _batch_reshard(self.mlp_gated_norm(self.rms_mlp(x)))
        mlp_out, router_stats = self.mlp(mlp_in)
        if self.shared is not None:
            mlp_out = mlp_out + self.shared(mlp_in, activation=ActivationFunctionEnum.silu)
        x = x + mlp_out
        return x, router_stats


class Transformer(eqx.Module):
    token_embed: jax.Array
    embed_up: jax.Array | None  # [d_e, D] up-projection; None unless embedding is factorized
    embed_norm: RMSNorm
    embed_gated_norm: GatedNorm
    head_down: jax.Array | None  # [D, d_e] down-projection before the LM head; None unless factorized
    output_proj: jax.Array
    blocks: tuple[Block, ...]
    final_norm: RMSNorm
    final_gated_norm: GatedNorm
    config: GrugModelConfig = eqx.field(static=True)

    @staticmethod
    def init(cfg: GrugModelConfig, *, key: PRNGKeyArray) -> "Transformer":
        embed_key, out_key, embed_gn_key, final_gn_key, up_key, down_key, *block_keys = random.split(
            key, cfg.num_layers + 6
        )
        d_e = cfg.inferred_embed_dim
        # Embedding + LM head live at the (possibly narrow) CE dim d_e; the body at hidden_dim D.
        token_embed = reshard(_init_weight(embed_key, (cfg.vocab_size, d_e), cfg.initializer_std), Pembed_vocab)
        output_proj = reshard(_init_weight(out_key, (d_e, cfg.vocab_size), cfg.initializer_std), Plm_head)
        embed_up = head_down = None
        if cfg.is_factorized_embed:
            # Small projections between the CE dim and the model dim; replicate (tiny: d_e*D).
            embed_up = reshard(_init_weight(up_key, (d_e, cfg.hidden_dim), cfg.initializer_std), P(None, None))
            head_down = reshard(_init_weight(down_key, (cfg.hidden_dim, d_e), cfg.initializer_std), P(None, None))
        blocks = tuple(Block.init(cfg, key=block_keys[i]) for i in range(cfg.num_layers))
        return Transformer(
            token_embed=token_embed,
            embed_up=embed_up,
            embed_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            embed_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=embed_gn_key),
            head_down=head_down,
            output_proj=output_proj,
            blocks=blocks,
            final_norm=RMSNorm.init(cfg.hidden_dim, cfg.layer_norm_eps),
            final_gated_norm=GatedNorm.init(cfg.hidden_dim, cfg.initializer_std, key=final_gn_key),
            config=cfg,
        )

    def _head_input(self, hidden: Float[Array, "B S D"]) -> Float[Array, "B S De"]:
        """Down-project the body output to the CE dim before the LM head (identity if not factorized)."""
        if self.head_down is None:
            return hidden
        return jnp.einsum("bsd,de->bse", hidden, self.head_down, out_sharding=_batch_spec())

    @named_call
    def __call__(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> tuple[Float[Array, "B S D"], dict[str, jax.Array]]:
        if mask is None:
            mask = AttentionMask.causal()

        batch_spec = _batch_spec()
        cfg = self.config
        hidden = self.token_embed.at[token_ids].get(out_sharding=batch_spec)
        # Factorized embedding: lookup lives at the (narrow) CE dim d_e; up-project to D.
        if self.embed_up is not None:
            hidden = jnp.einsum("bse,ed->bsd", hidden, self.embed_up, out_sharding=batch_spec)
        hidden = self.embed_gated_norm(self.embed_norm(hidden))

        if not isinstance(mask, AttentionMask):
            mask = AttentionMask.causal()
        short_mask, long_mask = _layer_attention_masks(
            mask, sliding_window=cfg.sliding_window, local_window=cfg.local_window
        )

        if cfg.remat_mode == "save_moe":
            remat_policy = jax.checkpoint_policies.save_only_these_names(*MOE_REMAT_SAVE_NAMES)
        else:
            remat_policy = None

        moe_router_stats: list[dict[str, jax.Array]] = []
        for i, block in enumerate(self.blocks):
            g = max(1, cfg.global_attn_every)
            layer_mask = long_mask if (i % g) == (g - 1) else short_mask
            hidden, router_stats = eqx.filter_checkpoint(block, policy=remat_policy)(hidden, layer_mask)
            moe_router_stats.append(router_stats)

        router_metrics = {
            "routing_entropy_per_layer": jnp.stack([s["routing_entropy"] for s in moe_router_stats], axis=0),
            "routing_counts_per_layer": jnp.stack([s["routing_counts"] for s in moe_router_stats], axis=0),
            "load_balancing_loss_per_layer": jnp.stack([s["load_balancing_loss"] for s in moe_router_stats], axis=0),
            "router_z_loss_per_layer": jnp.stack([s["router_z_loss"] for s in moe_router_stats], axis=0),
            "qb_beta_per_layer": jnp.stack([s["qb_beta"] for s in moe_router_stats], axis=0),
            "sparsity_loss_per_layer": jnp.stack([s["sparsity_loss"] for s in moe_router_stats], axis=0),
            "expected_active_frac_per_layer": jnp.stack([s["expected_active_frac"] for s in moe_router_stats], axis=0),
            "realized_active_frac_per_layer": jnp.stack([s["realized_active_frac"] for s in moe_router_stats], axis=0),
            "router_threshold_per_layer": jnp.stack([s["router_threshold"] for s in moe_router_stats], axis=0),
        }
        hidden = self.final_gated_norm(self.final_norm(hidden))
        return hidden, router_metrics

    @named_call
    def logits(
        self,
        token_ids: Int[Array, "B S"],
        mask: AttentionMask | jax.Array | None = None,
    ) -> Float[Array, "B S V"]:
        batch_spec = _batch_spec()
        hidden, _ = self(token_ids, mask=mask)
        hidden = self._head_input(hidden)
        return jnp.einsum("bsh,hd->bsd", hidden, self.output_proj, out_sharding=batch_spec)

    def next_token_loss(
        self,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
        return_router_metrics: bool = False,
    ) -> jax.Array | tuple[jax.Array, dict[str, jax.Array | SummaryStats]]:
        hidden, router_metrics = self(token_ids, mask=mask)
        hidden = self._head_input(hidden)
        labels = jnp.concatenate([token_ids[:, 1:], token_ids[:, :1] * 0], axis=1).astype(jnp.int32)
        loss_weight = loss_weight.astype(loss_dtype)

        cross_entropy_loss = fused_linear_softmax_cross_entropy_loss(
            hidden,
            self.output_proj,
            labels,
            weight=loss_weight,
            reduction=reduction,
            logsumexp_weight=logsumexp_weight,
            dtype=loss_dtype,
        )
        # No load-balancing loss; router z-loss only. The sparsity penalty (active
        # only when sparsity_loss_coef > 0) conditions the model toward fewer active
        # experts: coef * mean-over-layers of the expected active fraction.
        num_moe_layers = router_metrics["router_z_loss_per_layer"].shape[0]
        rzl = jnp.sum(router_metrics["router_z_loss_per_layer"]) / num_moe_layers
        z_aux_loss = self.config.router_z_loss_coef * rzl
        sparsity = jnp.mean(router_metrics["sparsity_loss_per_layer"])
        sparsity_loss = self.config.sparsity_loss_coef * sparsity
        aux_loss = z_aux_loss + sparsity_loss
        loss = cross_entropy_loss + aux_loss if reduction != "none" else cross_entropy_loss
        if return_router_metrics:
            summarized_metrics = _summarize_router_metrics(router_metrics)
            summarized_metrics["train/cross_entropy_loss"] = cross_entropy_loss
            summarized_metrics["train/router/aux_loss_weighted"] = aux_loss
            summarized_metrics["train/router/sparsity/penalty_weighted"] = sparsity_loss
            return loss, summarized_metrics
        return loss


def _init_weight(key: PRNGKeyArray, shape: tuple[int, ...], std: float) -> Float[Array, "..."]:
    return std * random.truncated_normal(key, -3, 3, shape)


def debug_mesh_and_token_pspec(num_devices: int) -> tuple[jax.sharding.AbstractMesh, P]:
    """Return a small abstract mesh and token sharding for lowering contract tests."""
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}")
    expert = 2 if num_devices % 2 == 0 else 1
    data = max(1, num_devices // expert)
    mesh = jax.sharding.AbstractMesh(
        axis_sizes=(1, data, expert, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
            jax.sharding.AxisType.Explicit,
        ),
    )
    return mesh, P(("replica_dcn", "data", "expert"), None)


__all__ = [
    "Block",
    "CausalSelfAttention",
    "DenseMLP",
    "GatedNorm",
    "GrugModelConfig",
    "MoEMLP",
    "MoeActivation",
    "RMSNorm",
    "Transformer",
    "debug_mesh_and_token_pspec",
]
