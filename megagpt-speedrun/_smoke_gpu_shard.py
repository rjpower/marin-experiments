# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Local explicit-sharding smoke for the GPU port.

The grug MoE was written/validated on TPU. Porting to 8xH100 exposed
explicit-mesh-axis ShardingTypeErrors (e.g. the GPU `reference_attention` backend
returns attn_out with a different axis placement than the TPU splash backend, so
the XSA multiply is illegally sharded). Those errors come from JAX's explicit
sharding *type system*, which is platform-independent, so we can reproduce them on
8 fake CPU devices WITHOUT a worker build.

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        uv run python _smoke_gpu_shard.py

Runs Transformer.next_token_loss (fwd+bwd via value_and_grad) under the real
compact_grug_mesh with explicit axes, batch sharded on _BATCH_AXES. A clean run
prints SUCCESS; a sharding bug raises a ShardingTypeError at the offending op.
"""

import os

# Must be set before jax is imported.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jmp  # noqa: E402
from haliax.partitioning import set_mesh  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402
from levanter.grug.attention import AttentionMask  # noqa: E402
from levanter.grug.sharding import compact_grug_mesh  # noqa: E402

from heuristic import MoeAdamHHeuristic  # noqa: E402
from model import _BATCH_AXES, Transformer  # noqa: E402


def main() -> None:
    print(f"jax devices: {jax.device_count()} ({jax.default_backend()})")
    assert jax.device_count() == 8, "expected 8 fake CPU devices"

    # A small but structurally-real geometry (deep-sparse big-E shape, scaled down).
    hidden = int(os.environ.get("SMOKE_HIDDEN", "512"))
    num_experts = int(os.environ.get("SMOKE_EXPERTS", "64"))
    top_k = int(os.environ.get("SMOKE_TOPK", "4"))
    seq_len = int(os.environ.get("SMOKE_SEQ", "256"))
    batch = int(os.environ.get("SMOKE_BATCH", "8"))  # divisible by data axis (8)
    mode = os.environ.get("SMOKE_MODE", "fixed")
    embed_env = os.environ.get("SMOKE_EMBED", "")
    embed_dim = int(embed_env) if embed_env else None

    cfg = MoeAdamHHeuristic().build_model_config(
        hidden,
        seq_len=seq_len,
        num_experts=num_experts,
        num_experts_per_token=top_k,
        adaptive_routing=(mode == "adaptive"),
        min_experts_per_token=int(os.environ.get("SMOKE_MIN_K", "0")),
        sparsity_loss_coef=float(os.environ.get("SMOKE_COEF", "0.0")) if mode == "adaptive" else 0.0,
        embed_dim=embed_dim,
    )
    print(
        f"cfg: D={cfg.hidden_dim} d_e={cfg.inferred_embed_dim}(factorized={cfg.is_factorized_embed}) "
        f"L={cfg.num_layers} E={cfg.num_experts} k={cfg.num_experts_per_token} "
        f"H={cfg.num_heads}/kv{cfg.num_kv_heads} hd={cfg.inferred_head_dim} I={cfg.intermediate_dim} "
        f"V={cfg.vocab_size} seq={cfg.max_seq_len} attn={cfg.attention_implementation}"
    )

    expert_axis = int(os.environ.get("SMOKE_EP", "1"))
    model_axis = int(os.environ.get("SMOKE_TP", "1"))
    mesh = compact_grug_mesh(expert_axis_size=expert_axis, replica_axis_size=1, model_axis_size=model_axis)
    print(f"mesh: {mesh.shape}")
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")

    with set_mesh(mesh):
        model = mp.cast_to_param(Transformer.init(cfg, key=jax.random.PRNGKey(0)))

        batch_sharding = NamedSharding(mesh, P(_BATCH_AXES, None))
        tokens = jax.device_put(
            jnp.arange(batch * seq_len, dtype=jnp.int32).reshape(batch, seq_len) % cfg.vocab_size,
            batch_sharding,
        )
        loss_weight = jax.device_put(jnp.ones((batch, seq_len), dtype=jnp.float32), batch_sharding)
        mask = AttentionMask.causal()

        def loss_fn(m):
            cm = mp.cast_to_compute(m)
            loss, _ = cm.next_token_loss(
                tokens,
                loss_weight,
                mask=mask,
                reduction="mean",
                logsumexp_weight=1e-4,
                return_router_metrics=True,
            )
            return loss

        loss, grads = jax.jit(jax.value_and_grad(loss_fn))(model)
        loss.block_until_ready()
        gnorm = jax.jit(lambda g: sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(g)))(grads)
        print(f"OK  loss={float(loss):.4f}  grad_sumsq={float(gnorm):.3e}")

    print("\nSUCCESS: explicit-sharding fwd+bwd clean on 8-device mesh")


if __name__ == "__main__":
    main()
