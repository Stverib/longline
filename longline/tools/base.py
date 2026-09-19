"""Tool base class and types.

Corresponds to TS: Tool.ts (ToolDef, buildTool) + tools.ts (assembleToolPool).
"""

# 本模块定义了工具系统的四大基石：注册和发现
# 1. ToolSchema —— 工具的 JSON Schema 描述，用于向 API 注册工具能力
# 2. ToolResult —— 工具执行后的统一返回格式，支持纯文本和富内容（如图片）
# 3. Tool —— 所有工具的抽象基类，定义了名称、schema、执行、并发安全四个接口契约
# 4. ToolRegistry —— 工具注册表，负责工具的注册、查找和批量 schema 导出
#
# 设计哲学：通过抽象基类 + 注册表模式，将工具的定义与发现解耦，
# 使得 query_loop 不需要知道具体有哪些工具，只需通过 registry 动态查找。

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@dataclass
class ToolSchema:
    """Tool schema for API registration."""
    # 对应 Anthropic API 中 tool 定义的三个必要字段：
    # - name: 工具的唯一标识符，API 返回 tool_use 时通过此名称匹配
    # - description: 工具功能描述，影响模型是否选择调用此工具
    # - input_schema: JSON Schema 格式的参数定义，模型据此生成合法的调用参数

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolResult:
    """Result of a tool execution.

    FIX (check.md #5): content can be str or list of content block dicts
    to support images, structured MCP results, etc.
    """
    # content 支持两种类型：
    # - str: 普通文本结果，绝大多数工具返回此类型
    # - list[dict]: 富内容块列表，用于返回图片（base64）、MCP 结构化结果等
    #   例如 FileReadTool 读取图片时返回 [{"type": "image", "source": {...}}]
    # 这种联合类型设计是为了兼容 Anthropic API 的 content block 格式，
    # 使得工具结果可以直接嵌入 API 的 tool_result 消息中。

    content: str | list[dict[str, Any]]
    # is_error 标记此结果是否为错误。关键设计：工具错误不抛异常，而是返回
    # is_error=True 的 ToolResult，这样 query_loop 不会中断，模型可以看到
    # 错误信息并决定如何处理（重试、换参数、或向用户解释）。
    is_error: bool = False

    @property
    def text(self) -> str:
        """Extract text content regardless of content type."""
        # 提供统一的文本提取接口，无论 content 是 str 还是 list[dict]。
        # 当 content 是富内容块列表时，尝试从每个 dict 中提取 "text" 字段，
        # 用换行连接。这在 hooks 的 PostToolUse 回调中特别有用，
        # 因为 hooks 只需要文本摘要而不关心图片等富内容。
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            block.get("text", str(block)) for block in self.content if isinstance(block, dict)
        )


class ReconcileOutcome(Enum):
    """What a tool can say about a call that started and never reported back.

    Three values, and the third is the important one. A two-valued answer would
    force every tool to claim either "it landed" or "it did not", and a tool that
    cannot read its own effect would have to pick -- which is exactly the guess
    that turns a recoverable interruption into a duplicated side effect.
    """

    APPLIED = "applied"          # the effect is present in the world
    NOT_APPLIED = "not_applied"  # the effect is provably absent; safe to retry
    UNKNOWN = "unknown"          # cannot be read from here; do not retry blindly


# The point of no return, as published to the tool that is executing.
#
# A tool cannot be handed the journal: `execute(tool_input)` is the tool
# interface, and widening it would touch every tool and every wrapper. So the
# executor publishes a marker here for the duration of one call, and a tool that
# has a point of no return reports it by calling `mark_irreversible()`.
#
# A `ContextVar` rather than an attribute on the tool because tools are shared
# across concurrent calls -- the streaming executor runs up to ten at once -- and
# an attribute would let one call's marker land on another call's record.
_current_irreversible_marker: ContextVar[Callable[[], None] | None] = ContextVar(
    "longline_tool_irreversible_marker", default=None
)


def mark_irreversible() -> None:
    """Report that the running tool can no longer be un-done.

    Called by a tool at the last line from which its effect is still impossible
    -- `BashTool` calls it immediately before spawning the shell. Everything
    above such a line is validation, and a call that was rejected changed
    nothing; from that line onward the runtime cannot read back what happened.
    Without this, those two situations are one state in the journal, and the only
    safe reading of that state is the pessimistic one -- which turns a call that
    never ran into an interruption that can never be repaired.

    Deliberately NOT best-effort, unlike the journal's prepare and commit: if the
    marker cannot be written the exception propagates into the tool, which must
    then abandon the operation. A swallowed failure here would leave a PREPARED
    with no marker for a call that DID run, and "no marker" is exactly what
    authorises a retry -- so swallowing would re-open the duplicate this whole
    mechanism exists to remove.

    A no-op when nothing is publishing a marker, which is every caller that
    cannot resume: sub-agents, one-shot `--print`, and the unit tests.
    """
    marker = _current_irreversible_marker.get()
    if marker is not None:
        marker()


@contextmanager
def irreversible_point(marker: Callable[[], None] | None) -> Iterator[None]:
    """Publish `marker` for the duration of one tool call.

    The executor's side of `mark_irreversible`. `None` means "this call is not
    being journalled", and leaves any outer marker in place rather than clearing
    it -- an unjournalled nested call must not erase its caller's.
    """
    if marker is None:
        yield
        return
    token = _current_irreversible_marker.set(marker)
    try:
        yield
    finally:
        _current_irreversible_marker.reset(token)


class Tool(ABC):
    """Base class for all tools.

    Corresponds to TS: Tool.ts ToolDef interface.
    """
    # 所有工具必须实现以下四个方法（其中 is_concurrency_safe 有默认实现）。
    # 这套接口契约保证了工具系统的可扩展性：新增工具只需继承 Tool 并实现这些方法，
    # 无需修改 orchestration、streaming_executor 等编排层代码。

    @abstractmethod
    def get_name(self) -> str:
        """Return the tool name as registered with the API."""
        ...

    @abstractmethod
    def get_schema(self) -> ToolSchema:
        """Return the tool's JSON schema for API registration."""
        ...

    @abstractmethod
    async def execute(self, tool_input: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given input.

        Args:
            tool_input: Validated input parameters.

        Returns:
            ToolResult with content and error status.
        """
        ...

    def is_concurrency_safe(self, tool_input: dict[str, Any]) -> bool:
        """Whether this tool can run concurrently with others.

        Corresponds to TS: Tool.ts isConcurrencySafe.
        Override in subclasses. Default: False (serial).
        """
        # 默认返回 False，意味着此工具会独占执行——
        # orchestration 在调度时会等待所有并发工具完成后，才单独执行此工具。
        # 只读工具（如 FileReadTool）应覆写为 True，允许多个读操作并行，
        # 而写操作（如 FileEditTool、BashTool）保持 False 以避免竞态条件。
        # 参数 tool_input 允许根据具体输入动态判断，例如 BashTool 可以对只读命令返回 True。
        return False

    # The two access modes `workload` may report.
    ACCESS_READ = "read"
    ACCESS_WRITE = "write"

    def workload(self, tool_input: dict[str, Any]) -> dict[str, str]:
        """The files this call touches, as `{absolute_path: ACCESS_*}`.

        Declared BEFORE execution, so the caller can digest the files both
        before and after. The default is empty, and empty means "this tool cannot
        say" rather than "this tool touches nothing" -- a distinction the
        recovery path depends on, because a tool that touches nothing can be
        safely retried and a tool that cannot say cannot.

        Bash declares nothing on purpose: from this layer a command that appends
        to a file and a command that reads one are the same string. Claiming
        otherwise would be a guess dressed as a fact.

        Keys are absolute so the digester never resolves them against its own
        working directory, which differs between a killed process and its
        successor.
        """
        return {}

    @classmethod
    def _declare(cls, raw: object, mode: str) -> dict[str, str]:
        """One declared path, or nothing when the argument is absent.

        Shared by every file tool, because the four of them differ only in which
        argument holds the path and which mode it is. An empty argument must
        yield nothing rather than `Path("")`, which is the current directory --
        a digest of the whole cwd, from a call that named no file at all.
        """
        if not raw:
            return {}
        return {str(Path(str(raw)).resolve()): mode}

    @staticmethod
    def _normalise_newlines(text: str) -> str:
        r"""Line endings to `\n`, so written content can be compared with read content.

        The production file writers open their target in TEXT mode (`os.fdopen(fd,
        "w")` in `FileWriteTool`, and `Path.write_bytes` on already-decoded text
        in `FileEditTool`), so on Windows a `\n` in the tool's own argument
        reaches the disk as `\r\n`. Comparing raw bytes would then report a
        successful write as NOT_APPLIED -- and NOT_APPLIED is an authorisation to
        retry, so the comparison would authorise exactly the duplicate side
        effect this mechanism exists to prevent.

        Caught by `test_write_is_non_ascii_safe` on a Windows checkout, not by
        reading either file: the two halves are in different modules and neither
        looks wrong on its own.
        """
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def reconcile(
        self, tool_input: dict[str, Any], *, started: bool = True
    ) -> ReconcileOutcome:
        """Whether a call that never reported back took effect.

        Asked only for operations the journal shows as started-but-uncommitted.
        The default is UNKNOWN, which is the correct answer for every tool whose
        effect is not readable from the workspace -- `Bash` above all. UNKNOWN
        authorises neither a retry nor a claim of success, so the recovery path
        reports it to the model and re-runs nothing.

        `started` is the journal's answer to a question a tool cannot ask itself:
        whether the call got as far as its point of no return. A tool that
        reports one -- by calling `mark_irreversible()` -- may use it to prove
        NOT_APPLIED, and `BashTool` does exactly that.

        A tool that reports NO point of no return must ignore `started`, and the
        default here does: for such a tool `started=False` means only "the
        executor never saw a marker", which is true of every call that tool ever
        made, so answering NOT_APPLIED there would authorise a retry of an
        operation that may well have landed. UNKNOWN under both values, and a
        tool only earns the stronger answer by marking.

        `started` defaults to True because the absent fact and the pessimistic
        fact are the same fact -- a caller with nothing to say must not be able to
        authorise a retry by saying nothing.
        """
        return ReconcileOutcome.UNKNOWN


@dataclass
class ToolRegistry:
    """Registry of available tools.

    Corresponds to TS: tools.ts assembleToolPool().
    """
    # 工具注册表是工具发现机制的核心。query_loop 通过 registry 查找工具，
    # 而不是直接持有工具实例，这使得工具集可以在运行时动态组装。
    # 例如：子 agent 的 registry 会排除 AgentTool（防止无限递归），
    # 后台 agent 的 registry 会排除交互式工具（如 AskUserQuestion）。

    # 使用 dict 存储，key 为工具名称，保证 O(1) 查找性能
    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        # 不允许重复注册同名工具，因为工具名是 API 层面的唯一标识，
        # 重复会导致 tool_use 响应无法正确路由。
        name = tool.get_name()
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        # 返回 None 而非抛异常，因为 API 可能返回未注册的工具名（如 MCP 工具被移除），
        # 调用方（orchestration）会将 None 转化为 is_error=True 的 ToolResult。
        return self._tools.get(name)

    def remove(self, name: str) -> Tool | None:
        """Remove a registered tool, returning it, or None when absent.

        The counterpart to `register` for callers that need to shrink the
        pool -- the evaluation harness strips declared-forbidden tools from a
        case's registry. Distinct from `swap(name, None)`: that would happily
        store None inside the dict and leave a dead entry behind, which
        `list_tools`/`get_api_schemas` would then trip over. A silently
        tolerated removal rather than a KeyError, because the main callers
        remove tools that may not be registered (a profile didn't offer the
        family) and "forbid an absent tool" is a no-op, not a mistake.
        """
        previous = self._tools.pop(name, None)
        return previous

    def swap(self, name: str, tool: Tool) -> Tool | None:
        """Replace an already-registered tool, returning the one it replaced.

        The counterpart to `register` for callers that need to wrap a tool
        rather than add one -- the evaluation harness's fault injectors are the
        case in point. It is a named method rather than direct `_tools` access
        so the replacement is visible in the public surface, and it REFUSES an
        unknown name: swapping a tool that was never registered would install
        one the profile never declared, which is a silent change to what the
        model can call. Returns None when the name was absent, so a caller that
        means "replace if present" can say so.
        """
        if name not in self._tools:
            raise KeyError(f"Tool not registered: {name}")
        previous = self._tools[name]
        self._tools[name] = tool
        return previous

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        # 用于 AgentTool 构建子 registry 时遍历父 registry 的所有工具
        return list(self._tools.values())

    def get_api_schemas(self) -> list[dict[str, Any]]:
        """Return all tool schemas in API format."""
        # 将所有工具的 schema 转为 API 请求所需的 dict 格式列表。
        # 这个列表会作为 API 请求中 "tools" 参数的值，告诉模型可用的工具集。
        schemas = []
        for tool in self._tools.values():
            schema = tool.get_schema()
            schemas.append({
                "name": schema.name,
                "description": schema.description,
                "input_schema": schema.input_schema,
            })
        return schemas
