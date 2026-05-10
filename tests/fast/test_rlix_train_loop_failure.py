from types import SimpleNamespace

import pytest

from miles.utils.rlix_train_loop import run_async_train_loop


async def _rollout_data():
    return "rollout"


def _args():
    return SimpleNamespace(num_rollout=1, save="")


def _rollout_manager():
    return SimpleNamespace(
        generate=SimpleNamespace(
            remote=lambda rollout_id: _rollout_data(),
        )
    )


@pytest.mark.asyncio
async def test_train_failure_runs_after_step():
    events = []

    async def before_step(step):
        events.append(("before", step))

    async def train(step, data):
        events.append(("train", step, data))
        raise RuntimeError("train failed")

    async def after_step(step):
        events.append(("after", step))

    train_group = SimpleNamespace(train=train)

    with pytest.raises(RuntimeError, match="train failed"):
        await run_async_train_loop(
            _args(),
            train_group=train_group,
            rollout_manager=_rollout_manager(),
            before_step=before_step,
            after_step=after_step,
        )

    assert events == [
        ("before", 0),
        ("train", 0, "rollout"),
        ("after", 0),
    ]


@pytest.mark.asyncio
async def test_before_step_failure_skips_after_step():
    events = []

    async def before_step(step):
        events.append(("before", step))
        raise RuntimeError("before failed")

    async def train(step, data):
        events.append(("train", step, data))

    async def after_step(step):
        events.append(("after", step))

    train_group = SimpleNamespace(train=train)

    with pytest.raises(RuntimeError, match="before failed"):
        await run_async_train_loop(
            _args(),
            train_group=train_group,
            rollout_manager=_rollout_manager(),
            before_step=before_step,
            after_step=after_step,
        )

    assert events == [("before", 0)]
