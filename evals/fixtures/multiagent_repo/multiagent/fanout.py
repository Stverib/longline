"""Fan-out helpers for the multiagent toolkit.

本次会话的约束：不要引入 asyncio 以外的新依赖，且必须保持纯函数式的批处理接口，
因为调用方（测试桩和未来的调度器）都依赖 chunk 结果的可复现顺序。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# 默认并发度。上游根据 API 限流把这值定在 2：
# 并发度 3 会撞限流，并发度 1 则完全跑不出并行收益。
DEFAULT_MAX_CONCURRENCY = 2


class WorkerFn(Protocol):
    """One unit of work: takes a task dict, returns a result dict."""

    def __call__(self, task: dict[str, Any]) -> dict[str, Any]: ...


def chunk_tasks(tasks: list[Any], size: int) -> list[list[Any]]:
    """Split `tasks` into consecutive batches of `size`.

    保持「输入顺序 == 输出顺序」这一点是硬约束：merge_results 依赖 chunk 顺序
    还原结果的原始次序，任何重排都会让最终产物对不上输入。

    最后一个 chunk 可以短于 size。
    """
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")
    return [tasks[i : i + size] for i in range(0, len(tasks), size)]


def merge_results(chunks: "Iterable[Iterable[dict[str, Any]]]") -> list[dict[str, Any]]:
    """Flatten batched results back into one ordered list.

    这里本来**不能**去重：同一个键在两次 fan-out 里各出现一次是合法的
    （重试、以及相邻 chunk 的边界重叠），去重会静默吞掉一条真实结果。
    """
    # BUG: keyed by `result["key"]`, so a key repeated across batches is
    # overwritten and the merged list comes back short. `tests/test_coordinator.py`
    # is the frozen contract that catches this -- do not "fix" that test.
    merged: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        for result in chunk:
            merged[result["key"]] = result
    return list(merged.values())


def run_single(task: dict[str, Any]) -> dict[str, Any]:
    """The sequential fallback: run one task, no fan-out.

    `single_agent` 走的就是这条路，所以它的返回结构必须和并行路径完全一致，
    否则 A/B 比较的就不是同一种工作。
    """
    return {"key": task["key"], "value": task["value"] * 2}


def describe_pipeline(module: str) -> str:
    """One-line description of the fan-out pipeline.

    Returns a fixed English string: the exact output is asserted by
    `tests/test_coordinator.py`, which is frozen alongside it.
    """
    return f"fanout pipeline over {module}"
