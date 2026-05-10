from __future__ import annotations

import importlib

import pytest


def _load_run_miles_dual(monkeypatch: pytest.MonkeyPatch):
    """Import the dual driver with the RLix-mode guard enabled."""
    monkeypatch.setenv("RLIX_CONTROL_PLANE", "rlix")
    return importlib.import_module("examples.rlix.run_miles_dual")


def test_split_pools_for_dual_accepts_exact_gpu_count(monkeypatch):
    """Exact two-pipeline GPU pool splits into two disjoint slices."""
    run_miles_dual = _load_run_miles_dual(monkeypatch)

    pool_p1, pool_p2 = run_miles_dual._split_pools_for_dual(
        num_gpus_per_node=4,
        infer_pool_size=2,
    )

    assert pool_p1 == [0, 1]
    assert pool_p2 == [2, 3]


def test_split_pools_for_dual_rejects_too_few_gpus(monkeypatch):
    """Too few visible GPUs fails before silently constructing partial pools."""
    run_miles_dual = _load_run_miles_dual(monkeypatch)

    with pytest.raises(ValueError, match="expects exactly 4 visible GPUs"):
        run_miles_dual._split_pools_for_dual(
            num_gpus_per_node=3,
            infer_pool_size=2,
        )


def test_split_pools_for_dual_rejects_extra_visible_gpus(monkeypatch):
    """Extra visible GPUs fail fast instead of being silently ignored."""
    run_miles_dual = _load_run_miles_dual(monkeypatch)

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        run_miles_dual._split_pools_for_dual(
            num_gpus_per_node=5,
            infer_pool_size=2,
        )


def test_default_dual_topology_is_disjoint(monkeypatch):
    """Dual driver keeps the existing disjoint topology unless opted in."""
    monkeypatch.delenv("MILES_DUAL_TOPOLOGY", raising=False)
    run_miles_dual = _load_run_miles_dual(monkeypatch)

    assert run_miles_dual._default_dual_topology() == "disjoint"


def test_env_can_select_shared_dual_topology(monkeypatch):
    """Env var can opt into the probe-only shared topology."""
    run_miles_dual = _load_run_miles_dual(monkeypatch)
    monkeypatch.setenv("MILES_DUAL_TOPOLOGY", "shared")

    assert run_miles_dual._default_dual_topology() == "shared"


def test_invalid_dual_topology_env_raises(monkeypatch):
    """Invalid topology values fail fast before the driver starts actors."""
    run_miles_dual = _load_run_miles_dual(monkeypatch)
    monkeypatch.setenv("MILES_DUAL_TOPOLOGY", "partitioned")

    with pytest.raises(ValueError, match="MILES_DUAL_TOPOLOGY"):
        run_miles_dual._default_dual_topology()


def test_shared_pools_for_dual_returns_same_pool(monkeypatch):
    """Shared probe maps both pipelines onto the same physical GPU pool."""
    run_miles_dual = _load_run_miles_dual(monkeypatch)

    pool_p1, pool_p2 = run_miles_dual._shared_pools_for_dual(
        num_gpus_per_node=4,
        infer_pool_size=2,
    )

    assert pool_p1 == [0, 1]
    assert pool_p2 == [0, 1]
    assert pool_p1 == pool_p2
