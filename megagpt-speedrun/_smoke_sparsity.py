# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke test for the adaptive-sparsity grug MoE routing.

The full ``Transformer`` forward exercises the XSA attention path, whose explicit
mesh sharding only lowers on a real TPU mesh (the base grug model is never run
eager on CPU — ``delayed-gradient-pp`` only CPU-tests its optimizer). This file
therefore tests the part this experiment actually changes: the ``MoEMLP`` router,
the adaptive variable-k gate, and the sparsity penalty. It runs the MoE layer
under a 1-device compact grug mesh and checks:

- fixed top-k routing reports exactly K/E active;
- adaptive routing starts dense (threshold inits below the logit scale);
- the learned threshold receives a non-zero gradient (straight-through works);
- the per-token floor is respected even under a crushing penalty;
- end to end, raising the penalty drives the realized active fraction *down*.

Run directly:

    JAX_PLATFORMS=cpu uv run python _smoke_sparsity.py
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from model import GrugModelConfig, MoEMLP


@eqx.filter_jit
def _forward(mlp, x):
    return mlp(x)

D = 128
TOKENS = 64


def _config(**overrides) -> GrugModelConfig:
    base = dict(
        vocab_size=256,
        hidden_dim=D,
        intermediate_dim=64,
        shared_expert_intermediate_dim=D,
        num_experts=8,
        num_experts_per_token=4,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=64,
        max_seq_len=16,
        sliding_window=16,
        initializer_std=0.04,
        qk_mult=1.3,
    )
    base.update(overrides)
    return GrugModelConfig(**base)


def _mlp_and_inputs(cfg):
    mlp = MoEMLP.init(cfg, key=jax.random.PRNGKey(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (1, TOKENS, cfg.hidden_dim))
    return mlp, x


def test_fixed_routing():
    cfg = _config()
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with set_mesh(mesh):
        mlp, x = _mlp_and_inputs(cfg)
        _out, stats = _forward(mlp, x)
    frac = float(stats["realized_active_frac"])
    assert abs(frac - 4 / 8) < 1e-6, f"fixed top-4/8 must be 0.5 active, got {frac}"
    print(f"OK  fixed routing: realized active frac = {frac:.4f} (== top-4/8)")


def test_adaptive_dense_start():
    cfg = _config(adaptive_routing=True, sparsity_loss_coef=1.0)
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with set_mesh(mesh):
        mlp, x = _mlp_and_inputs(cfg)
        _out, stats = _forward(mlp, x)
    realized = float(stats["realized_active_frac"])
    expected = float(stats["expected_active_frac"])
    assert abs(realized - 4 / 8) < 1e-6, f"dense init should start at top-4/8, got {realized}"
    assert 0.0 <= expected <= 4 / 8 + 1e-6, f"expected frac out of range: {expected}"
    print(f"OK  adaptive dense start: realized={realized:.4f}, expected={expected:.4f}")


def test_threshold_receives_gradient():
    cfg = _config(adaptive_routing=True, sparsity_loss_coef=1.0)
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with set_mesh(mesh):
        mlp, x = _mlp_and_inputs(cfg)

        def loss_fn(m):
            out, stats = m(x)
            return jnp.mean(out**2) + cfg.sparsity_loss_coef * stats["sparsity_loss"]

        grads = eqx.filter_jit(eqx.filter_grad(loss_fn))(mlp)
    g = float(grads.router_threshold)
    assert jnp.isfinite(g) and abs(g) > 1e-8, f"threshold got no gradient (straight-through broken): {g}"
    print(f"OK  threshold receives gradient: d loss / d theta = {g:.6f}")


def _train(cfg, steps=60, lr=0.2):
    """Minimize a fixed reconstruction target + sparsity penalty; return final realized frac."""
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    with set_mesh(mesh):
        mlp, x = _mlp_and_inputs(cfg)
        target = jax.random.normal(jax.random.PRNGKey(2), (1, TOKENS, cfg.hidden_dim))
        opt = optax.adam(lr)
        opt_state = opt.init(mlp)

        def loss_fn(m):
            out, stats = m(x)
            recon = jnp.mean((out - target) ** 2)
            return recon + cfg.sparsity_loss_coef * stats["sparsity_loss"], stats

        @jax.jit
        def step(m, opt_state):
            (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(m)
            updates, opt_state = opt.update(grads, opt_state, m)
            m = optax.apply_updates(m, updates)
            return m, opt_state, stats["realized_active_frac"]

        frac = None
        for _ in range(steps):
            mlp, opt_state, frac = step(mlp, opt_state)
    return float(frac)


def test_floor_enforced():
    cfg = _config(adaptive_routing=True, min_experts_per_token=2, sparsity_loss_coef=50.0)
    frac = _train(cfg)
    assert frac >= 2 / 8 - 1e-6, f"floor of 2/8 violated under heavy penalty: {frac}"
    print(f"OK  floor enforced: realized frac converged to {frac:.4f} >= 2/8")


def test_penalty_drives_sparsity_down():
    # Same model/target/steps; only the penalty weight differs. Heavier penalty must
    # end sparser than no penalty -> the soft loss conditions the model toward sparsity.
    frac_free = _train(_config(adaptive_routing=True, min_experts_per_token=0, sparsity_loss_coef=0.0))
    frac_pen = _train(_config(adaptive_routing=True, min_experts_per_token=0, sparsity_loss_coef=30.0))
    assert frac_pen < frac_free - 1e-3, f"penalty did not sparsify: pen={frac_pen} !< free={frac_free}"
    print(f"OK  penalty drives sparsity down: coef=0 -> {frac_free:.4f},  coef=30 -> {frac_pen:.4f}")


if __name__ == "__main__":
    print(f"jax devices: {jax.device_count()}")
    test_fixed_routing()
    test_adaptive_dense_start()
    test_threshold_receives_gradient()
    test_floor_enforced()
    test_penalty_drives_sparsity_down()
    print("\nall adaptive-sparsity routing smoke checks passed")
