import asyncio
import atexit
import dataclasses
import logging
import os
import queue
import threading
import time

import aiohttp

from miles.rollout.base_types import EnginePreemptedError, RLixRouterMetadataError
from miles.rollout.data_source import DataSource
from miles.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from miles.utils.async_utils import run
from miles.utils.rlix_hooks import NoOpRLixHooks, RLixHooks
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _FatalError:
    """F01 fully_async fatal-error sentinel — single-path queue signal.

    The queue ``output_queue`` is the ONE place fully_async surfaces
    fatal errors; the per-worker ``task_done_callback`` puts an
    instance of this sentinel onto the queue and the main loop dequeues
    + ``raise``s. Anti-regression invariant #9 / scope F01 forbids any
    parallel ``_fatal_error`` flag attribute on AsyncRolloutWorker.

    ``inner`` is the original exception (typically
    :class:`EnginePreemptedError` or :class:`RLixRouterMetadataError`)
    for callers that want to inspect the cause.
    """

    inner: BaseException


def group_oldest_weight_version(group: list[Sample]) -> int | None:
    """Return the minimum weight version across all trajectories and turns in a group."""
    versions = [s.oldest_weight_version for s in group if s.oldest_weight_version is not None]
    return min(versions) if versions else None


class _CachedWeightVersion:
    """Throttled query for the current engine weight version via /model_info."""

    def __init__(self, ttl: float = 1.0):
        self._ttl = ttl
        self._value: int | None = None
        self._last_query: float = 0.0

    async def get(self, args) -> int | None:
        now = time.monotonic()
        if self._value is not None and (now - self._last_query) < self._ttl:
            return self._value
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/model_info"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self._value = int(data["weight_version"])
                        self._last_query = now
        except Exception as e:
            logger.debug(f"Failed to query engine weight version: {e}")
        return self._value


_cached_version = _CachedWeightVersion()


def _shared_probe_enabled() -> bool:
    return os.environ.get("MILES_DEBUG_SHARED_PROBE") == "1"


async def _probe_router_admission_state(args) -> dict:
    """Best-effort router snapshot for shared-topology RCA runs."""
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/admission_state"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                data = await resp.json()
                return {"status": resp.status, "body": data}
    except Exception as exc:  # noqa: BLE001 - probe path should not break rollout.
        return {"error": repr(exc)}


# Global worker manager
_global_worker = None
_worker_lock = threading.Lock()


def get_global_worker(args, data_buffer: DataSource):
    """Get or create global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            print("Creating new global async worker...")
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=args.sglang_server_concurrency)
            _global_worker.start()
        return _global_worker


def stop_global_worker():
    """Stop global worker"""
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


class AsyncRolloutWorker:
    """
    Simplified asynchronous rollout worker, using threads instead of processes
    Supports continuous running, independent of rollout function lifecycle
    """

    def __init__(self, args, data_buffer: DataSource, concurrency=10):
        self.args = args
        self.data_buffer = data_buffer  # Directly save data_buffer reference
        self.concurrency = concurrency
        self.running = True
        self.output_queue = queue.Queue(maxsize=1000)  # Continuous output queue
        self.worker_thread = None
        self.state = GenerateState(args)
        self._debug_lock = threading.Lock()
        self._debug_active_tasks = 0
        self._debug_started_groups = 0
        self._debug_completed_callbacks = 0
        self._debug_fatal_callbacks = 0
        self._debug_task_errors = 0
        self._debug_last_error: str | None = None

    def _set_active_task_count(self, count: int) -> None:
        with self._debug_lock:
            self._debug_active_tasks = count

    def _record_started_group(self) -> None:
        with self._debug_lock:
            self._debug_started_groups += 1

    def _record_completed_callback(self) -> None:
        with self._debug_lock:
            self._debug_completed_callbacks += 1

    def _record_fatal_callback(self, exc: BaseException) -> None:
        with self._debug_lock:
            self._debug_fatal_callbacks += 1
            self._debug_last_error = repr(exc)

    def _record_task_error(self, exc: BaseException) -> None:
        with self._debug_lock:
            self._debug_task_errors += 1
            self._debug_last_error = repr(exc)

    def debug_snapshot(self) -> dict:
        with self._debug_lock:
            return {
                "running": bool(self.running),
                "thread_alive": bool(self.worker_thread and self.worker_thread.is_alive()),
                "active_tasks": self._debug_active_tasks,
                "started_groups": self._debug_started_groups,
                "completed_callbacks": self._debug_completed_callbacks,
                "fatal_callbacks": self._debug_fatal_callbacks,
                "task_errors": self._debug_task_errors,
                "last_error": self._debug_last_error,
                "output_queue_size": self.get_queue_size(),
            }

    async def continuous_worker_loop(self):
        """Continuous work loop - constantly get data from data_buffer and process"""
        print("Continuous async rollout worker started")

        active_tasks = set()
        max_concurrent_tasks = self.args.rollout_batch_size
        group_id_counter = 0

        while self.running:
            try:
                # Clean up completed tasks
                if active_tasks:
                    done_tasks = {task for task in active_tasks if task.done()}
                    for task in done_tasks:
                        try:
                            task.result()  # Results are already handled in callbacks
                        except Exception as e:
                            self._record_task_error(e)
                            print(f"Task failed with exception: {e}")
                    active_tasks -= done_tasks
                    self._set_active_task_count(len(active_tasks))

                # If active task count hasn't reached limit, try to get new data and start tasks
                while len(active_tasks) < max_concurrent_tasks and self.running:
                    samples = self.data_buffer.get_samples(1)

                    for group in samples:
                        group_id = group_id_counter
                        group_id_counter += 1

                        # Create new async task
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )

                        # Add completion callback. F01: route the two
                        # RLix-classified scheduler-preempt exceptions
                        # (EnginePreemptedError, RLixRouterMetadataError)
                        # through the single _FatalError sentinel path
                        # on output_queue. Other exceptions propagate via
                        # done_task.result() — caller sees them as a
                        # regular task failure (existing behavior).
                        def make_callback(gid):
                            def task_done_callback(done_task):
                                try:
                                    result = done_task.result()
                                except (EnginePreemptedError, RLixRouterMetadataError) as fatal:
                                    self._record_fatal_callback(fatal)
                                    self.output_queue.put((gid, _FatalError(inner=fatal)))
                                    return
                                except Exception as exc:
                                    self._record_task_error(exc)
                                    raise
                                self._record_completed_callback()
                                self.output_queue.put((gid, result))

                            return task_done_callback

                        task.add_done_callback(make_callback(group_id))
                        active_tasks.add(task)
                        self._record_started_group()
                        self._set_active_task_count(len(active_tasks))
                        break

                # Brief sleep to avoid busy waiting
                await asyncio.sleep(1)

            except Exception as e:
                self._record_task_error(e)
                print(f"Error in continuous worker loop: {e}")
                await asyncio.sleep(1)

        if active_tasks:
            print(f"Waiting for {len(active_tasks)} continuous tasks to complete...")
            await asyncio.wait(active_tasks)

        print("Continuous async rollout worker stopped")

    def worker_thread_func(self):
        """Worker function running in independent thread"""
        asyncio.run(self.continuous_worker_loop())

    def start(self):
        """Start continuous work mode"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self.worker_thread_func, daemon=True)
            self.worker_thread.start()
            print("Started continuous async worker thread")

    def stop(self):
        """Stop worker thread"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        print("Stopped async worker thread")

    def get_completed_groups(self) -> list[tuple]:
        """Get completed sample groups"""
        completed = []
        while True:
            try:
                result = self.output_queue.get_nowait()
                completed.append(result)
            except queue.Empty:
                break
        return completed

    def get_queue_size(self) -> int:
        """Get current output queue size"""
        return self.output_queue.qsize()


async def generate_rollout_async(
    args,
    rollout_id: int,
    data_buffer: DataSource,
    *,
    rlix_hooks: RLixHooks | None = None,
) -> list[list[Sample]]:
    """
    Simplified asynchronous rollout generation - using global continuous worker.

    ``rlix_hooks`` (F9 progress reporting): standalone callers pass
    ``None`` and get a :class:`NoOpRLixHooks` instance internally so the
    code path is branchless. RLix mode injects a real
    :class:`MilesRLixHooks` wrapping the coordinator handle.
    """
    assert args.rollout_global_dataset

    if rlix_hooks is None:
        rlix_hooks = NoOpRLixHooks()

    # Get global worker, which will run continuously
    worker = get_global_worker(args, data_buffer)

    # Simplified: directly use rollout_batch_size as target
    target_data_size = args.rollout_batch_size

    data = []
    completed_groups = {}
    do_print = True
    stale_groups_recycled = 0
    staleness_values = []

    use_staleness_filter = getattr(args, "max_weight_staleness", None) is not None

    print(f"Starting async rollout generation for {target_data_size} groups")
    print(f"Global worker queue size: {worker.get_queue_size()}")
    if use_staleness_filter:
        print(f"Staleness filter enabled: max_weight_staleness={args.max_weight_staleness}")

    # F9 progress hook: open the batch BEFORE the wait window so the
    # coordinator can publish the demand. M11 hook signature expects
    # (target_weight_version, step_target_groups, initial_completed,
    # mode=None, adapter_id=None). data is empty at this point per
    # plan; queued groups in worker.output_queue arrive via
    # bump_completed below. mode/adapter_id stay None for forward-
    # compat with M11.4 LoRA (X1 / F108).
    #
    # Query the live engine weight version from /model_info so the
    # coordinator can key progress by the actual target version (the
    # cached helper above throttles repeated lookups). Falls back to
    # 0 if the engine is not yet responsive (initial batch before the
    # first update_weights call).
    current_weight_version = await _cached_version.get(args)
    if current_weight_version is None:
        current_weight_version = 0
    rlix_hooks.begin_progress_batch(
        target_weight_version=int(current_weight_version),
        step_target_groups=int(target_data_size),
        initial_completed=0,
    )

    # Main loop: collect results from global worker's output queue
    start_time = time.time()
    last_progress_time = start_time
    no_progress_timeout = 30.0  # Warn if no progress for 30 seconds

    while len(data) < target_data_size:
        # Collect completed results
        completed = worker.get_completed_groups()

        made_progress = False
        for group_id, group in completed:
            # F01 / Anti-regression invariant #9: dequeue _FatalError
            # sentinel and raise the inner exception. Single-path
            # propagation; do NOT also check a parallel _fatal_error
            # flag — that is forbidden.
            if isinstance(group, _FatalError):
                rlix_hooks.end_progress_batch()
                raise group.inner
            completed_groups[group_id] = group
            made_progress = True

        if made_progress:
            last_progress_time = time.time()

        # Query current engine version once per collection batch (cached/throttled)
        current_engine_version = None
        if use_staleness_filter:
            current_engine_version = await _cached_version.get(args)

        # Process completed groups in order (try to maintain order, but not strict requirement)
        processed_any = False

        # Process all available completed groups
        available_ids = list(completed_groups.keys())
        for group_id in available_ids:
            if len(data) >= target_data_size:
                break

            group = completed_groups.pop(group_id)

            # If any sample in the group was aborted, return the whole group to the data buffer
            # and do not forward it to the training engine.
            try:
                any_aborted = any([sample.status == Sample.Status.ABORTED for sample in group])
            except Exception:
                any_aborted = False

            if any_aborted:
                try:
                    for s in group:
                        s.reset_for_retry()
                    data_buffer.add_samples([group])
                    print(f"Returned aborted group {group_id} to data buffer", flush=True)
                except Exception as e:
                    print(f"Failed to return aborted group {group_id} to buffer: {e}", flush=True)
                # don't count as processed for training
                continue

            # Staleness filter: discard groups whose oldest weight version is too far behind
            oldest = group_oldest_weight_version(group)
            if oldest is not None and current_engine_version is not None:
                staleness = current_engine_version - oldest
                staleness_values.append(staleness)
                if staleness > args.max_weight_staleness:
                    try:
                        for s in group:
                            s.reset_for_retry()
                        data_buffer.add_samples([group])
                    except Exception as e:
                        logger.warning(f"Failed to recycle stale group {group_id}: {e}")
                    stale_groups_recycled += 1
                    logger.info(
                        f"Recycled stale group {group_id} "
                        f"(oldest_version={oldest}, current={current_engine_version}, "
                        f"staleness={staleness} > max={args.max_weight_staleness})"
                    )
                    # don't count as processed for training
                    continue

            if do_print:
                print(
                    f"First rollout sample: {[group[0].prompt + group[0].response]}, "
                    f"label: {group[0].label}, reward: {group[0].reward}",
                    flush=True,
                )
                do_print = False

            # Simplified: directly add samples, no filters used
            data.append(group)
            # F9 progress: report one completed group per accepted
            # group. The coordinator throttles its own emission; this
            # is the raw "collected" count fully_async produces.
            #
            # Use the live engine weight version (refreshed by
            # _cached_version above when the staleness filter is on,
            # or queried here on demand) so the coordinator keys
            # progress by the actual target version, not by a stale
            # batch-begin snapshot.
            live_version = current_engine_version
            if live_version is None:
                live_version = await _cached_version.get(args)
            if live_version is None:
                live_version = current_weight_version
            try:
                rlix_hooks.bump_completed(target_weight_version=int(live_version))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"rlix_hooks.bump_completed failed: {exc!r}")
            processed_any = True

        # Check progress
        current_time = time.time()
        if current_time - last_progress_time > no_progress_timeout:
            print(
                f"Warning: No progress for {no_progress_timeout}s. "
                f"Queue size: {worker.get_queue_size()}, "
                f"Collected: {len(data)}/{target_data_size}"
            )
            if _shared_probe_enabled():
                router_state = await _probe_router_admission_state(args)
                print(
                    "[shared-probe] no-progress snapshot: "
                    f"pipeline_index={getattr(args, 'pipeline_index', None)} "
                    f"pipeline_id={getattr(args, 'pipeline_id', None)} "
                    f"exp_name={getattr(args, 'exp_name', None)} "
                    f"rollout_id={rollout_id} "
                    f"collected={len(data)}/{target_data_size} "
                    f"completed_buffer={len(completed_groups)} "
                    f"worker={worker.debug_snapshot()} "
                    f"router={router_state}",
                    flush=True,
                )
            last_progress_time = current_time

        # If no results were processed, brief sleep to avoid busy waiting
        if not processed_any:
            await asyncio.sleep(0.01)

    duration = time.time() - start_time
    print(f"Rollout completed in {duration:.2f}s! Global worker queue size: {worker.get_queue_size()}")
    if stale_groups_recycled > 0 or staleness_values:
        avg_staleness = sum(staleness_values) / len(staleness_values) if staleness_values else 0
        print(
            f"Staleness stats: recycled={stale_groups_recycled}, "
            f"avg_staleness={avg_staleness:.1f}, "
            f"max_staleness={max(staleness_values) if staleness_values else 0}"
        )

    if data:
        print(
            f"Finish rollout: {[data[-1][0].prompt + data[-1][0].response]}, "
            f"label: {data[-1][0].label}, reward: {data[-1][0].reward}",
            flush=True,
        )

    data = sorted(data, key=lambda group: group[0].index)
    rlix_hooks.end_progress_batch()
    return data


def generate_rollout_fully_async(
    args,
    rollout_id,
    data_buffer: DataSource,
    evaluation=False,
    *,
    rlix_hooks: RLixHooks | None = None,
):
    """Sync entry the existing rollout_function_path resolves to.

    ``rlix_hooks`` is forwarded to :func:`generate_rollout_async`;
    standalone callers pass ``None`` (NoOp). RLix entry driver injects
    :class:`MilesRLixHooks` to wire progress reporting into the
    coordinator.
    """
    if evaluation:
        raise ValueError("Evaluation mode not supported in simple async rollout")

    completed_samples = run(
        generate_rollout_async(args, rollout_id, data_buffer, rlix_hooks=rlix_hooks)
    )
    return completed_samples


# Register exit cleanup function

atexit.register(stop_global_worker)
