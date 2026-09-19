"""Streaming tool executor — start executing tools before API response completes.

Corresponds to TS: services/tools/StreamingToolExecutor.ts.

P1a: Enhanced with hooks, concurrency-safe dispatch, and semaphore limiting.
Preserves all semantics from orchestration.py (hooks, batching, error handling).
"""

# 本模块是 orchestration.py (run_tools) 的流式演进版本。
#
# 关键区别：run_tools() 需要等 API 响应完整返回后才开始执行工具，
# 而 StreamingToolExecutor 可以在 API 流式返回过程中，一旦某个
# tool_use block 完整解析出来，就立即启动该工具的执行。
#
# 这带来了显著的延迟优化：假设模型返回了 3 个 tool_use，
# 传统模式需要等 3 个都返回才开始执行第 1 个，
# 流式模式在第 1 个返回时就开始执行，第 2、3 个返回时可能第 1 个已经完成了。
#
# 但这也引入了并发控制的复杂性：必须区分并发安全与非并发安全工具，
# 非并发安全工具需要排队等待独占执行权。

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from longline.utils.hashing import sha256_file

from .base import ToolRegistry, ToolResult, irreversible_point

if TYPE_CHECKING:
    from collections.abc import Callable

    from longline.hooks.hook_runner import HookConfig
    from longline.models.content_blocks import ToolUseBlock

logger = logging.getLogger(__name__)

# 与 orchestration.py 保持一致的并发上限
MAX_CONCURRENCY = 10


class StreamingToolExecutor:
    """Execute tool calls as they arrive during streaming.

    Corresponds to TS: services/tools/StreamingToolExecutor.ts.

    P1a enhancements over original:
    - Concurrent-safe tools run in parallel; non-safe tools execute exclusively
    - PreToolUse/PostToolUse hooks called for each tool
    - Semaphore limits concurrent tool count to MAX_CONCURRENCY
    - Permission checker interface reserved for P2a
    """

    def __init__(
        self,
        registry: ToolRegistry,
        hooks: list[HookConfig] | None = None,
        permission_checker: Callable[..., Any] | None = None,
        journal: Any | None = None,
        turn_id: int = 0,
    ) -> None:
        self._registry = registry
        self._hooks = hooks
        # permission_checker 是 P2a 阶段的权限检查接口预留，
        # 用于在执行工具前检查用户是否授权。目前可以为 None。
        self._permission_checker = permission_checker
        # The durable operation journal (`longline/session/tool_journal.py`).
        # None for every caller that cannot resume -- sub-agents, tests, one-shot
        # `--print` -- which is why every use below is guarded rather than
        # assumed. This is the only layer holding all three things a record
        # needs: the tool_call_id, the tool, and the moment before execution.
        self._journal = journal
        self._turn_id = turn_id
        # _pending 记录所有已启动的 (tool_use_id, asyncio.Task) 对，
        # 用于在 get_results() 中收集结果
        self._pending: list[tuple[str, asyncio.Task[ToolResult]]] = []
        # _queue 暂存不能立即执行的工具（非并发安全，或当前有独占工具在运行）
        self._queue: list[ToolUseBlock] = []
        # 标记当前是否有非并发安全工具在独占执行，
        # 为 True 时所有新到达的工具（包括并发安全的）都必须排队
        self._has_exclusive_running = False
        # 信号量限制最大并发数，防止资源耗尽
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    def add_tool(self, block: ToolUseBlock) -> None:
        """Add a completed tool_use block for execution.

        Concurrent-safe tools start immediately (up to semaphore limit).
        Non-safe tools queue until exclusive access is available.
        """
        # 此方法在 API 流式解析出一个完整的 tool_use block 时被调用。
        # 判断逻辑：
        # - 如果工具是并发安全的，且当前没有独占工具在运行 → 立即启动
        # - 否则 → 加入队列，等待 get_results() 时按序处理
        tool = self._registry.get(block.name)
        is_safe = tool is not None and tool.is_concurrency_safe(block.input)

        if is_safe and not self._has_exclusive_running:
            # 并发安全且没有独占锁 → 立即启动（信号量在 _execute_one 中控制）
            self._start_execution(block)
        else:
            # 非并发安全，或有独占工具正在执行 → 排队
            # 排队的原因：如果有 Edit 正在独占执行，即使是 Read 也不能并行，
            # 因为可能读到 Edit 写了一半的文件内容。
            self._queue.append(block)

    def _start_execution(self, block: ToolUseBlock) -> None:
        """Start executing a tool in a background task."""
        # 使用 asyncio.create_task 在事件循环中启动工具执行，不阻塞当前流程。
        # 这是流式执行的关键——add_tool() 调用后立即返回，工具在后台运行。
        task = asyncio.create_task(self._execute_one(block))
        self._pending.append((block.id, task))

    async def _execute_one(self, block: ToolUseBlock) -> ToolResult:
        """Execute a single tool with hooks and semaphore."""
        # 整个执行过程在信号量保护下进行，确保并发数不超过 MAX_CONCURRENCY
        async with self._semaphore:
            tool = self._registry.get(block.name)
            if tool is None:
                return ToolResult(content=f"Error: Unknown tool '{block.name}'", is_error=True)

            # PreToolUse hooks —— 允许用户配置的 hooks 在工具执行前拦截
            if self._hooks:
                from longline.hooks.hook_runner import run_pre_tool_hooks

                hook_result = await run_pre_tool_hooks(self._hooks, block.name, block.input)
                if hook_result.blocked:
                    return ToolResult(
                        content=f"Blocked by hook: {hook_result.message}",
                        is_error=True,
                    )

            # Permission check (P2a interface)
            # 权限检查：在 ACCEPT_EDITS 等非全自动模式下，
            # 某些危险操作（如删除文件）需要用户确认。
            if self._permission_checker is not None:
                allowed = await self._permission_checker(block.name, block.input)
                if not allowed:
                    return ToolResult(content="Denied by permission policy", is_error=True)

            # Journal the INTENT before executing. The record has to exist before
            # the world can change, or the window this closes -- effect present,
            # no record of it -- is wide open.
            operation_id: str | None = None
            workload: dict[str, str] = {}
            if self._journal is not None:
                try:
                    workload = tool.workload(block.input)
                    operation_id = self._journal.prepare(
                        turn_id=self._turn_id,
                        tool_call_id=block.id,
                        tool_name=block.name,
                        tool_input=block.input,
                        workload=workload,
                    )
                except Exception as e:
                    # Durability is a safety net, not a gate: a session directory
                    # that cannot be written must not be the reason a user's edit
                    # does not happen.
                    logger.error("Tool journal PREPARE failed for %s: %s", block.name, e)
                    operation_id = None

            # Execute. The marker is published around the call so that a tool can
            # report its own point of no return: the one fact that separates
            # "died before the effect was possible" from "died with the effect in
            # flight", and the one fact no layer above this can observe.
            # 与 orchestration.py 一致：异常转为错误 ToolResult，不中断循环
            marker = (
                self._irreversible_marker(self._journal, operation_id)
                if operation_id is not None and self._journal is not None
                else None
            )
            try:
                with irreversible_point(marker):
                    result = await tool.execute(block.input)
            except Exception as e:
                logger.warning("Tool %s failed: %s", block.name, e)
                result = ToolResult(content=f"Error: {e}", is_error=True)

            # Committed on EVERY return path, including the error one: a tool that
            # returned an error DID return, and leaving it PREPARED would make a
            # failed call look interrupted -- so the next resume would reconcile a
            # question that already has an answer.
            if operation_id is not None and self._journal is not None:
                try:
                    self._journal.commit(
                        operation_id,
                        outcome="error" if result.is_error else "ok",
                        post_state={path: sha256_file(Path(path)) for path in workload},
                    )
                except Exception as e:
                    logger.error("Tool journal COMMIT failed for %s: %s", block.name, e)

            # PostToolUse hooks —— 工具完成后触发，用于审计日志、通知等
            if self._hooks:
                from longline.hooks.hook_runner import run_post_tool_hooks

                await run_post_tool_hooks(self._hooks, block.name, block.input, result.text)

            return result

    async def get_results(self) -> list[tuple[str, ToolResult]]:
        """Wait for all pending tools and process queued tools. Return results in order.

        Called after the API stream completes.
        """
        # 此方法在 API 流式响应完全接收后调用。此时：
        # 1. 部分并发安全工具可能已经在后台完成（通过 add_tool 启动的）
        # 2. 队列中可能还有排队的非并发安全工具尚未执行
        # 先处理队列中的剩余工具，再收集所有结果。

        # Process queued non-concurrent tools
        await self._process_queue()

        # Collect all results
        # 按 _pending 的顺序（即工具到达的顺序）收集结果
        results: list[tuple[str, ToolResult]] = []
        for tool_id, task in self._pending:
            try:
                result = await task
            except Exception as e:
                # 兜底异常处理：即使 _execute_one 内部已有 try/except，
                # task 本身的异常（如 CancelledError）也需要处理
                result = ToolResult(content=f"Error: {e}", is_error=True)
            results.append((tool_id, result))
        return results

    def _irreversible_marker(
        self, journal: Any, operation_id: str
    ) -> Callable[[], None]:
        """A once-only writer for this operation's point of no return.

        Once-only because a tool may reach its point of no return more than once
        inside one call -- a retry loop within `execute` -- and the record that
        matters is the FIRST: after it the operation is in flight, and a second
        line saying so adds nothing.

        The write is NOT wrapped in `try`/`except`, unlike PREPARE and COMMIT.
        Those two are best-effort by design -- a session directory that cannot be
        written must not be the reason a user's edit does not happen. This one is
        a BARRIER: a tool that cannot record crossing its point of no return must
        not cross it, so the failure belongs to the tool, which will abandon the
        operation. Swallowing it would leave a PREPARED with no marker for a call
        that did run -- and "no marker" is what authorises a retry.
        """
        written = False

        def mark() -> None:
            nonlocal written
            if written:
                return
            journal.mark_executing(operation_id)
            written = True

        return mark

    async def _process_queue(self) -> None:
        """Process queued tools respecting concurrency constraints."""
        # 处理队列的策略：
        # - 并发安全工具：直接启动（它们会通过信号量自动限流）
        # - 非并发安全工具：先等待所有已启动任务完成 → 设置独占标记 →
        #   启动该工具 → 等待它完成 → 清除独占标记 → 继续处理下一个
        #
        # 这确保了非并发安全工具执行期间没有任何其他工具在并行运行。
        while self._queue:
            block = self._queue.pop(0)
            tool = self._registry.get(block.name)
            is_safe = tool is not None and tool.is_concurrency_safe(block.input)

            if not is_safe:
                # 非并发安全：先排空所有已启动的并发任务
                await self._wait_pending()
                # 设置独占标记（虽然此时队列处理是顺序的，但为语义完整性保留）
                self._has_exclusive_running = True
                self._start_execution(block)
                # 等待独占工具完成后再继续
                await self._wait_pending()
                self._has_exclusive_running = False
            else:
                # 并发安全：直接启动，信号量控制并发度
                self._start_execution(block)

    async def _wait_pending(self) -> None:
        """Wait for all currently pending tasks to complete."""
        # 遍历所有已启动的 task，等待未完成的。
        # 注意：已完成的 task await 是无操作（立即返回），不会阻塞。
        for _, task in self._pending:
            if not task.done():
                await task

    @property
    def has_pending(self) -> bool:
        # 用于外部判断是否还有工具在执行或等待执行
        # query_loop 据此决定是否需要调用 get_results()
        return len(self._pending) > 0 or len(self._queue) > 0
