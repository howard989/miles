"""M11.2 dual-pipeline driver — `examples/rlix/run_miles_dual.py`.

Spawns two MilesCoordinator + MilesPipeline pairs in separate Ray
namespaces. By default it uses disjoint ``cluster_device_mappings``;
``--dual-topology=shared`` is a probe-only path that deliberately maps
both pipelines onto the same physical GPU pool. Each pipeline runs its
own ``rlix_train_loop`` concurrently via ``asyncio.gather``.

Default topology (disjoint pools, no cross-pipeline GPU contention):
    pipeline 1: actor_train=[0,1], actor_infer=[0,1]
    pipeline 2: actor_train=[2,3], actor_infer=[2,3]

Shared topology probe (NOT A FEATURE; expected to expose the first
scheduler/placement blocker):
    pipeline 1: actor_train=[0], actor_infer=[0,1]
    pipeline 2: actor_train=[0], actor_infer=[0,1]

This is the minimum-viable M11.2 PASS — proves two pipelines can register
+ initialize + train + sync + generate + clean up concurrently without
namespace, actor-name, port, or scheduler-ledger collisions. It does NOT
exercise cross-pipeline preemption (Option B/C — overlap topology — needs
the deferred F22 shell-init contract per ``miles_pipeline.py:14-33``).

Per-pipeline isolation that this driver enforces:
- Distinct pipeline IDs from ``orchestrator.allocate_pipeline_id``
- Distinct Ray namespaces from ``get_pipeline_namespace(pipeline_id)``
- Distinct ``cluster_device_mappings`` registered with the orchestrator
  AND threaded through ``MilesPipelineConfig.cluster_device_mappings``
  so ``MilesPipeline._build_placement_provider`` uses the right physical
  GPUs (rather than the default ``range(actor_count)`` / ``range(...)``)
- Distinct ``MILES_ROLLOUT_BASE_PORT`` per pipeline (15000 vs 16000) so
  the ``find_available_port`` calls in two concurrent RolloutManager
  actors never race for the same port window
- Distinct ``exp_name`` so any tracking dirs / log dirs do not collide
- W&B / TensorBoard / Prometheus disabled (``--use-wandb`` etc must be
  unset); per-pipeline tracking re-enable is M11.3 follow-up

Per scope F13 the driver MUST NOT have a top-level ``try/except`` and
MUST NOT call ``ray.shutdown()``: failure semantics = let exceptions
propagate naturally → driver exits → user runs ``ray stop`` to clean up.
"""

from __future__ import annotations

import copy
import os
import sys


_DUAL_TOPOLOGY_ENV = "MILES_DUAL_TOPOLOGY"
_DUAL_TOPOLOGIES = ("disjoint", "shared")

# F08 / F41 — fail fast if RLix entry is invoked without the env var.
# The check must happen BEFORE any heavy import (torch / sglang /
# megatron) so CVD has a chance to take effect via Ray runtime_env.
if os.environ.get("RLIX_CONTROL_PLANE") != "rlix":
    sys.stderr.write(
        "examples/rlix/run_miles_dual.py requires RLIX_CONTROL_PLANE=rlix.\n"
        "    RLIX_CONTROL_PLANE=rlix python -m examples.rlix.run_miles_dual ...\n"
    )
    sys.exit(2)


def _split_pools_for_dual(
    *, num_gpus_per_node: int, infer_pool_size: int
) -> tuple[list[int], list[int]]:
    """Split a contiguous physical GPU pool into two disjoint per-pipeline pools.

    For a 4-GPU machine with infer_pool_size=2 returns
    ``([0,1], [2,3])``. The base args carry the PER-PIPELINE shape
    (actor_num_gpus_per_node = train size per pipeline,
    rollout_num_gpus = infer pool size per pipeline). The dual driver
    just maps each pipeline onto its own slice of the physical pool.
    """
    needed = 2 * infer_pool_size
    if num_gpus_per_node != needed:
        raise ValueError(
            f"dual driver expects exactly {needed} visible GPUs "
            f"(2 pipelines * infer_pool_size={infer_pool_size}); "
            f"got num_gpus_per_node={num_gpus_per_node}. "
            "To use a subset of a larger machine, set CUDA_VISIBLE_DEVICES "
            "before launching the driver."
        )
    physical = list(range(num_gpus_per_node))
    return list(physical[:infer_pool_size]), list(physical[infer_pool_size : 2 * infer_pool_size])


def _shared_pools_for_dual(
    *, num_gpus_per_node: int, infer_pool_size: int
) -> tuple[list[int], list[int]]:
    """Return overlapping per-pipeline pools for the shared-topology probe.

    This is intentionally a probe path, not the M11.2 default. It keeps the
    same whole-machine shape as the disjoint driver (two infer pools worth of
    visible GPUs), then maps both pipelines to the first pool so RLix must
    arbitrate cross-pipeline GPU contention.
    """
    needed = 2 * infer_pool_size
    if num_gpus_per_node != needed:
        raise ValueError(
            f"shared dual-topology probe expects exactly {needed} visible GPUs "
            f"(2 pipelines * infer_pool_size={infer_pool_size}); "
            f"got num_gpus_per_node={num_gpus_per_node}."
        )
    pool = list(range(infer_pool_size))
    return list(pool), list(pool)


def _default_dual_topology() -> str:
    """Read the dual-driver topology mode from env, defaulting to disjoint."""
    topology = os.environ.get(_DUAL_TOPOLOGY_ENV, "disjoint")
    if topology not in _DUAL_TOPOLOGIES:
        raise ValueError(
            f"{_DUAL_TOPOLOGY_ENV} must be one of {_DUAL_TOPOLOGIES}, got {topology!r}"
        )
    return topology


def _add_dual_driver_args(parser):
    """Add probe-only dual-driver args without changing standard MILES args."""
    parser.add_argument(
        "--dual-topology",
        choices=_DUAL_TOPOLOGIES,
        default=_default_dual_topology(),
        help=(
            "Dual-driver topology. 'disjoint' is the M11.2 default. "
            "'shared' is a probe-only experiment that maps both pipelines "
            "onto the same GPU pool to expose scheduler/placement blockers."
        ),
    )
    return parser


def _per_pipeline_args(base_args, *, pipeline_index: int):
    """Deep-copy parsed args and tailor for one pipeline.

    BASE args already carry the per-pipeline topology shape
    (actor_num_gpus_per_node = per-pipeline train size,
    rollout_num_gpus = per-pipeline infer pool size). This function only
    bumps exp_name and resets the router port so each pipeline gets a
    fresh allocation.
    """
    args = copy.deepcopy(base_args)

    if hasattr(args, "exp_name") and args.exp_name:
        args.exp_name = f"{args.exp_name}-mp{pipeline_index}"
    else:
        args.exp_name = f"miles_dual_mp{pipeline_index}"

    # Force fresh router allocation per pipeline; rollout.py's
    # _start_router calls find_available_port when sglang_router_port is
    # None.
    if hasattr(args, "sglang_router_port"):
        args.sglang_router_port = None

    return args


def _build_pipeline(
    *,
    base_args,
    pipeline_index: int,
    pipeline_pool: list[int],
    orchestrator,
    ray,
    MilesCoordinator,
    MilesPipelineConfig,
    COORDINATOR_ACTOR_NAME_PREFIX,
    get_pipeline_namespace,
    logger,
):
    """Allocate one pipeline_id, register, admit, create coordinator+pipeline.

    ``pipeline_pool`` is the physical GPU pool for THIS pipeline. In the
    default topology each pipeline gets a disjoint slice; in the shared
    probe both pipelines intentionally receive the same slice. Within
    that pool we keep the partial-overlap shape from base_args: train
    pool = first ``actor_num_gpus_per_node`` GPUs of the pipeline pool,
    infer pool = full pipeline pool.

    Returns ``(pipeline_id, namespace, coordinator_handle, pipeline_handle, args)``.
    """
    pipeline_id = ray.get(orchestrator.allocate_pipeline_id.remote("miles"))
    pipeline_namespace = get_pipeline_namespace(pipeline_id)

    args = _per_pipeline_args(base_args, pipeline_index=pipeline_index)

    train_size = int(args.actor_num_nodes) * int(args.actor_num_gpus_per_node)
    infer_size = int(args.rollout_num_gpus)
    if infer_size != len(pipeline_pool):
        raise ValueError(
            f"mp{pipeline_index}: pipeline_pool size {len(pipeline_pool)} != "
            f"args.rollout_num_gpus {infer_size}"
        )
    if train_size > infer_size:
        raise ValueError(
            f"mp{pipeline_index}: train_size {train_size} > infer_size {infer_size}; "
            f"per-pipeline must keep partial-overlap (train ⊆ infer)"
        )
    train_mapping = list(pipeline_pool[:train_size])
    infer_mapping = list(pipeline_pool)
    # Probe-only observability: thread identity into rollout/train logs and
    # router diagnostic endpoints so shared-topology stalls can be attributed.
    setattr(args, "pipeline_id", pipeline_id)
    setattr(args, "pipeline_index", pipeline_index)
    logger.info(
        "[run_miles_dual] mp%d allocated pipeline_id=%s namespace=%s "
        "train=%s infer=%s",
        pipeline_index, pipeline_id, pipeline_namespace,
        train_mapping, infer_mapping,
    )

    cluster_device_mappings = {
        "actor_train": train_mapping,
        "actor_infer": infer_mapping,
    }
    cluster_tp_configs = {
        "actor_train": int(args.actor_num_gpus_per_node),
        "actor_infer": int(args.rollout_num_gpus_per_engine),
    }
    ray.get(
        orchestrator.register_pipeline.remote(
            pipeline_id=pipeline_id,
            ray_namespace=pipeline_namespace,
            cluster_tp_configs=cluster_tp_configs,
            cluster_device_mappings=cluster_device_mappings,
        )
    )
    ray.get(orchestrator.admit_pipeline.remote(pipeline_id=pipeline_id))
    logger.info(
        "[run_miles_dual] mp%d registered+admitted pipeline_id=%s mappings=%s",
        pipeline_index, pipeline_id, cluster_device_mappings,
    )

    cfg = MilesPipelineConfig(
        miles_args=args,
        sglang_config=getattr(args, "sglang_config", None),
        verify_model_after_sync=bool(getattr(args, "verify_model_after_sync", False)),
        num_gpus_per_node=int(
            getattr(args, "num_gpus_per_node", None) or args.actor_num_gpus_per_node
        ),
        system_envs={},
        cluster_device_mappings=cluster_device_mappings,
    )

    pipeline_runtime_env_vars = {
        "PIPELINE_ID": str(pipeline_id),
        "ROLL_RAY_NAMESPACE": pipeline_namespace,
        "RLIX_CONTROL_PLANE": "rlix",
        # Per-pipeline base port so the two RolloutManager actors do
        # not race for the same port window.
        "MILES_ROLLOUT_BASE_PORT": str(15000 + pipeline_index * 1000),
    }
    if str(getattr(base_args, "dual_topology", "disjoint")) == "shared" or (
        os.environ.get("MILES_DEBUG_SHARED_PROBE") == "1"
    ):
        pipeline_runtime_env_vars["MILES_DEBUG_SHARED_PROBE"] = "1"
        pipeline_runtime_env_vars["MILES_ROUTER_TEST_HOOKS"] = "1"
    if pythonpath := os.environ.get("PYTHONPATH"):
        pipeline_runtime_env_vars["PYTHONPATH"] = pythonpath
    for _k in (
        "MILES_TMS_HOOK_MODE",
        "MILES_SKIP_TMS_PAUSE",
        "MILES_SKIP_NODE_PG_PIN",
        "TMS_INIT_ENABLE_CPU_BACKUP",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NCCL_NVLS_ENABLE",
    ):
        if (_v := os.environ.get(_k)) is not None:
            pipeline_runtime_env_vars[_k] = _v

    coordinator = (
        ray.remote(MilesCoordinator)
        .options(
            name=f"{COORDINATOR_ACTOR_NAME_PREFIX}{pipeline_id}",
            namespace=pipeline_namespace,
            lifetime="detached",
            num_cpus=0.01,
            runtime_env={"env_vars": pipeline_runtime_env_vars},
        )
        .remote(pipeline_id=pipeline_id, pipeline_config=cfg)
    )
    logger.info("[run_miles_dual] mp%d MilesCoordinator created", pipeline_index)

    pipeline = ray.get(coordinator.create_pipeline_actor.remote(pipeline_config=cfg))
    ray.get(pipeline.initialize_pipeline.remote(coordinator_handle=coordinator))
    logger.info(
        "[run_miles_dual] mp%d MilesPipeline.initialize_pipeline complete pipeline_id=%s",
        pipeline_index, pipeline_id,
    )

    return pipeline_id, pipeline_namespace, coordinator, pipeline, args


def main():
    """Dual-pipeline entry. Imports heavy modules lazily so the env-var
    guard above fires before transitive ``import torch`` / ``import sglang``.
    """
    import asyncio
    import logging
    from dataclasses import dataclass, field
    from typing import Any, Optional

    import ray

    from miles.utils.arguments import parse_args
    from miles.utils.logging_utils import configure_logger
    from miles.utils.rlix_train_loop import run_async_train_loop
    from miles.utils.rlix_validation import assert_rlix_topology
    from rlix.pipeline.miles_coordinator import MilesCoordinator
    from rlix.protocol.types import (
        COORDINATOR_ACTOR_NAME_PREFIX,
        get_pipeline_namespace,
    )

    import rlix

    configure_logger()
    logger = logging.getLogger("run_miles_dual")
    base_args = parse_args(add_custom_arguments=_add_dual_driver_args)

    # F10 startup fail-fast on the BASE args. Per-pipeline arg overrides
    # below preserve the topology shape (just shrink the GPU pool).
    assert_rlix_topology(
        base_args, sglang_config=getattr(base_args, "sglang_config", None)
    )

    # --- Pipeline config dataclass with cluster_device_mappings field ---
    @dataclass
    class MilesPipelineConfig:
        miles_args: Any
        sglang_config: Optional[Any] = None
        verify_model_after_sync: bool = False
        num_gpus_per_node: int = 8
        system_envs: dict = field(default_factory=dict)
        # M11.2 — cluster_device_mappings flow into MilesPipeline so
        # _build_placement_provider can use per-pipeline physical GPUs.
        cluster_device_mappings: dict = field(default_factory=dict)

    # --- Topology: derive 2-pipeline disjoint pools from physical GPUs ---
    # BASE args carry the per-pipeline shape (actor_num_gpus_per_node =
    # per-pipeline train size, rollout_num_gpus = per-pipeline infer
    # pool size). num_gpus_per_node is the WHOLE-machine GPU count.
    num_gpus_per_node = int(getattr(base_args, "num_gpus_per_node", 0) or 0)
    if num_gpus_per_node <= 0:
        raise RuntimeError(
            "run_miles_dual.py requires --num-gpus-per-node to set the whole-"
            "machine GPU count (so the dual driver can construct per-pipeline "
            "GPU pools)."
        )
    dual_topology = str(getattr(base_args, "dual_topology", "disjoint"))
    if dual_topology == "disjoint":
        pool_p1, pool_p2 = _split_pools_for_dual(
            num_gpus_per_node=num_gpus_per_node,
            infer_pool_size=int(base_args.rollout_num_gpus),
        )
    elif dual_topology == "shared":
        pool_p1, pool_p2 = _shared_pools_for_dual(
            num_gpus_per_node=num_gpus_per_node,
            infer_pool_size=int(base_args.rollout_num_gpus),
        )
    else:
        raise ValueError(f"unsupported dual_topology={dual_topology!r}")
    logger.info(
        "[run_miles_dual] topology=%s: num_gpus_per_node=%d, P1_pool=%s, "
        "P2_pool=%s, per-pipeline train_size=%d infer_size=%d",
        dual_topology, num_gpus_per_node, pool_p1, pool_p2,
        int(base_args.actor_num_nodes) * int(base_args.actor_num_gpus_per_node),
        int(base_args.rollout_num_gpus),
    )

    # ---- 1. Connect to RLix; get the orchestrator. -----------------------
    orchestrator = rlix.init(create_if_missing=True)

    # ---- 2. Build both pipelines sequentially. ---------------------------
    # Sequential init avoids racing the RolloutManager construction; the
    # rlix orchestrator's allocate_pipeline_id is itself serialized.
    p1 = _build_pipeline(
        base_args=base_args,
        pipeline_index=1,
        pipeline_pool=pool_p1,
        orchestrator=orchestrator,
        ray=ray,
        MilesCoordinator=MilesCoordinator,
        MilesPipelineConfig=MilesPipelineConfig,
        COORDINATOR_ACTOR_NAME_PREFIX=COORDINATOR_ACTOR_NAME_PREFIX,
        get_pipeline_namespace=get_pipeline_namespace,
        logger=logger,
    )
    p2 = _build_pipeline(
        base_args=base_args,
        pipeline_index=2,
        pipeline_pool=pool_p2,
        orchestrator=orchestrator,
        ray=ray,
        MilesCoordinator=MilesCoordinator,
        MilesPipelineConfig=MilesPipelineConfig,
        COORDINATOR_ACTOR_NAME_PREFIX=COORDINATOR_ACTOR_NAME_PREFIX,
        get_pipeline_namespace=get_pipeline_namespace,
        logger=logger,
    )

    pipelines = [p1, p2]

    # ---- 3. Pull handles for each pipeline. ------------------------------
    handles = []
    for pid, ns, coord, pipe, args in pipelines:
        train_group = ray.get(pipe.get_train_group.remote())
        rollout_manager = ray.get(pipe.get_rollout_manager.remote())
        engine_count = int(ray.get(pipe.get_declared_engine_count.remote()))
        logger.info(
            "[run_miles_dual] handles ready pipeline_id=%s engines=%d",
            pid, engine_count,
        )
        handles.append((pid, ns, coord, pipe, args, train_group, rollout_manager))

    # ---- 4. Drive 2 concurrent rlix_train_loops via asyncio.gather. -----
    async def _run_one_pipeline(idx, pid, pipe, args, train_group, rollout_manager):
        async def _before(step: int) -> None:
            await pipe.before_training.remote(step)

        async def _after(step: int) -> None:
            await pipe.after_training.remote(step)

        await run_async_train_loop(
            args,
            train_group=train_group,
            rollout_manager=rollout_manager,
            before_step=_before,
            after_step=_after,
            loop_label=f"mp{idx} pipeline_id={pid}",
        )
        logger.info("[run_miles_dual] mp%d training loop complete pipeline_id=%s", idx, pid)

    async def _async_main():
        await asyncio.gather(
            *(
                _run_one_pipeline(
                    i + 1, pid, pipe, args, train_group, rollout_manager
                )
                for i, (pid, ns, coord, pipe, args, train_group, rollout_manager)
                in enumerate(handles)
            )
        )

    asyncio.run(_async_main())
    logger.info("[run_miles_dual] both training loops complete; shutting down")

    # ---- 5. Clean shutdown of both pipelines. ----------------------------
    shutdown_refs = [pipe.shutdown_hard.remote() for _, _, _, pipe, _, _, _ in handles]
    ray.get(shutdown_refs)
    for pid, _, _, _, _, _, _ in handles:
        logger.info("[run_miles_dual] shutdown_hard complete pipeline_id=%s", pid)


if __name__ == "__main__":
    main()
