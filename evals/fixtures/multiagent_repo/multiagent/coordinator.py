"""Coordinator: plans a task list and fans it out.

模块定位：coordinator 只负责**调度**，不负责实际执行。
真正的执行逻辑在 fanout.run_single，并行切分在 fanout.chunk_tasks。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from multiagent.fanout import DEFAULT_MAX_CONCURRENCY, chunk_tasks, merge_results, run_single

if TYPE_CHECKING:
    from collections.abc import Callable


class Coordinator:
    """Turns a request into an ordered task list, then runs it.

    设计决策（已与上游确认）：coordinator **不持有** worker 池。
    worker 由调用方注入，这样 single-agent 和 multi-agent 两条路径
    可以共用同一个 Coordinator 实例，A/B 比较的才是同一份工作内容。
    """

    def __init__(self, max_concurrency: int = DEFAULT_MAX_CONCURRENCY) -> None:
        # 并发度默认 2：3 会撞限流，1 则退化成串行，两种都测不出并行收益。
        self.max_concurrency = max_concurrency
        self._tasks: list[dict[str, Any]] = []

    @property
    def tasks(self) -> list[dict[str, Any]]:
        """The planned tasks, in the order they were planned."""
        return list(self._tasks)

    def plan(self, request: str) -> list[dict[str, Any]]:
        """Expand a request into one task per comma-separated item.

        The request format is `"key1=value1,key2=value2"`. Values are ints.
        A malformed item raises rather than being skipped: silently dropping an
        item would turn a 4-way fan-out into a 3-way one and the A/B would
        compare two different workloads.
        """
        tasks: list[dict[str, Any]] = []
        for item in request.split(","):
            item = item.strip()
            if not item:
                continue
            key, _, raw = item.partition("=")
            if not key or not raw:
                raise ValueError(f"malformed request item: {item!r}")
            tasks.append({"key": key.strip(), "value": int(raw)})
        self._tasks = tasks
        return list(tasks)

    def run(self, worker: Callable[[dict[str, Any]], dict[str, Any]] = run_single) -> list[dict[str, Any]]:
        """Run the planned tasks in batches of `max_concurrency`.

        `worker` is injectable so the parallel and sequential paths can be
        swapped without changing the plan.
        """
        batches = chunk_tasks(self._tasks, self.max_concurrency)
        return merge_results([[worker(t) for t in batch] for batch in batches])
