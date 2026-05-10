"""RLix-mode async training loop helper.

Factored out of :mod:`train_async` so the RLix-side driver
(``examples/rlix/run_miles_rlix.py``) can reuse the iteration shape while
bracketing each step with the ``MilesPipeline.before_training`` /
``MilesPipeline.after_training`` Ray calls that drive scheduler-managed
GPU onload / offload / weight sync.

Key ordering difference vs ``train_async.train``:

The standalone loop dispatches the next rollout *before* the training
step (``train_async.py`` lines 68-78), which is safe when actor_train
and actor_infer occupy disjoint GPUs. RLix's M11.1 single-pipeline
topology overlaps them on the same physical GPUs (train ``[0,1]`` /
infer ``[0,1]``); ``MilesPipeline._before_training`` re-claims
``actor_train`` and onloads weights at the start of every step, which
would race against an in-flight rollout. The corrected ordering here
defers the next-rollout dispatch until **after** the per-step
``after_step`` hook releases actor_train, serializing rollout and
training on the shared GPU pool.

For 4xGPU partial-overlap smoke runs (``--num-rollout 2``) this is the
intended behavior. A future iter that supports disjoint train/infer
pools may restore the prefetch-before-train shape behind a flag.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from miles.utils.misc import should_run_periodic_action


# Step-hook callable shape: ``async (step: int) -> None``. The driver wraps the
# pipeline actor's remote method into a coroutine via ``asyncio.gather`` /
# ``await ref`` semantics so the loop awaits the round-trip cleanly.
StepHook = Callable[[int], Awaitable[None]]


async def run_async_train_loop(
    args: Any,
    *,
    train_group: Any,
    rollout_manager: Any,
    before_step: StepHook,
    after_step: StepHook,
    num_rollout_per_epoch: Optional[int] = None,
) -> None:
    """RLix-mode async training loop.

    Bracket each step with ``before_step`` (claims actor_train + onloads
    train weights) and ``after_step`` (offloads + drives base weight sync
    via the coordinator + releases actor_train).

    Skips, relative to ``train_async.train``:
        - critic branch (M11 deferred)
        - in-loop ``actor_model.update_weights()`` — the rlix
          ``after_step`` already drives ``coord.sync_base_weights_to_active``
        - rollout shuffle ``rollout_manager.dispose`` — handled by
          ``MilesPipeline.shutdown_hard`` via the driver

    ``save_model`` is invoked only when ``args.save`` is truthy; the
    smoke runs pass ``--save ""`` to disable. The save is wrapped in an
    explicit onload→save→offload bracket because ``after_step`` already
    offloaded the train group, and Megatron ``save_model`` does not
    onload internally.

    ``eval`` is gated similarly; smoke runs do not exercise eval.
    """
    if getattr(args, "use_critic", False):
        raise NotImplementedError(
            "RLix-mode critic branch is deferred (M11.4 / F100). "
            "Run with --use-critic disabled."
        )

    # First-build behaviour for resume: ``MilesPipeline._init_phase_a_train``
    # calls ``train_group.init()`` and discards the returned
    # ``start_rollout_id``. For a clean checkpoint this is 0; for a real
    # resume path we will need to capture the value via a new pipeline
    # accessor (M11.1 follow-up).
    if getattr(args, "start_rollout_id", None) is None:
        args.start_rollout_id = 0

    start_rollout_id = int(args.start_rollout_id)
    num_rollout = int(args.num_rollout)
    if num_rollout <= start_rollout_id:
        return

    import logging as _logging

    _log = _logging.getLogger("rlix_train_loop")
    _log.setLevel(_logging.INFO)

    # Pre-loop priming: dispatch the first rollout. The base v=-1 weight
    # sync MUST already have been driven by the caller (driver) so this
    # rollout sees correctly-versioned weights.
    _log.info("[loop] pre-loop generate dispatch rollout_id=%d", start_rollout_id)
    rollout_data_next_future = rollout_manager.generate.remote(start_rollout_id)

    for rollout_id in range(start_rollout_id, num_rollout):
        _log.info("[loop] rollout_id=%d step1: await rollout_data start", rollout_id)
        rollout_data_curr_ref = await rollout_data_next_future
        rollout_data_next_future = None
        _log.info("[loop] rollout_id=%d step1: await rollout_data done", rollout_id)

        before_ok = False
        try:
            _log.info("[loop] rollout_id=%d step2: before_step start", rollout_id)
            await before_step(rollout_id)
            before_ok = True
            _log.info("[loop] rollout_id=%d step2: before_step done", rollout_id)

            _log.info("[loop] rollout_id=%d step3: train_group.train start", rollout_id)
            await train_group.train(rollout_id, rollout_data_curr_ref)
            _log.info("[loop] rollout_id=%d step3: train_group.train done", rollout_id)
        finally:
            if before_ok:
                _log.info("[loop] rollout_id=%d step4: after_step start", rollout_id)
                await after_step(rollout_id)
                _log.info("[loop] rollout_id=%d step4: after_step done", rollout_id)

        # 5) Optional save (gated; smoke disables via --save "").
        if getattr(args, "save", None):
            if should_run_periodic_action(
                rollout_id,
                args.save_interval,
                num_rollout_per_epoch,
                args.num_rollout,
            ):
                await train_group.onload()
                try:
                    await train_group.save_model(
                        rollout_id,
                        force_sync=rollout_id == args.num_rollout - 1,
                    )
                finally:
                    await train_group.offload()

        # 6) Eval is intentionally skipped in rlix-mode smoke runs:
        #    after_step has already offloaded the train group, and the
        #    Megatron eval path requires onloaded weights. Restoring
        #    eval needs an explicit onload→eval→offload bracket
        #    (M11.1 follow-up).

        # 7) Dispatch the next rollout AFTER actor_train is released, so
        #    the new rollout does not race for partial-overlap GPUs.
        if rollout_id + 1 < num_rollout:
            rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)


__all__ = ["run_async_train_loop"]
