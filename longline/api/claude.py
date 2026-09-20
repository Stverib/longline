"""Streaming API interaction with Claude.

Corresponds to TS: services/api/claude.ts — queryModelWithStreaming(),
stream event parsing, and response assembly.

=== 架构角色 ===

本模块是 Anthropic SDK 与内部事件系统之间的「适配层」。
它将 SDK 的流式 SSE 事件（message_start, content_block_delta, ...）
转换为内部 QueryEvent（TextDelta, ToolUseStart, TurnComplete, ErrorEvent）。

核心函数 stream_response() 是一个 async generator，被 QueryEngine.make_call_model()
创建的闭包调用，最终由 query_loop 消费。

=== SSE 事件处理状态机 ===

Anthropic 的流式响应由以下 SSE 事件组成，每种事件触发不同处理：

  message_start       → 提取 input_tokens（计费用）
  content_block_start → 初始化内容块状态（text/tool_use/thinking）
  content_block_delta → 增量拼接内容（text_delta/input_json_delta/thinking_delta）
  content_block_stop  → 完成一个内容块（解析 tool_use 的 JSON、yield ToolUseStart）
  message_delta       → 提取 stop_reason 和 output_tokens

注意: ToolUseStart 在 content_block_stop 时才 yield（而非 content_block_start），
因为需要等待 input_json 完整拼接后才能 json.loads 解析出工具参数。

=== 错误映射 ===

  APIStatusError (429)  → ErrorEvent(is_recoverable=True)   限流
  APIStatusError (529)  → ErrorEvent(is_recoverable=True)   过载
  APIStatusError (其他) → ErrorEvent(is_recoverable=False)  不可恢复
  APIConnectionError    → ErrorEvent(is_recoverable=True)   网络问题，可重试

=== 模块关系 ===

  依赖: anthropic SDK、longline/core/events.py（事件类型）、longline/models/（数据模型）
  被依赖: longline/core/query_engine.py（通过 make_call_model 闭包间接调用）
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

import anthropic

from longline.core.events import (
    ErrorEvent,
    QueryEvent,
    TextDelta,
    ThinkingDelta,
    ToolUseStart,
    TurnComplete,
)
from longline.models.content_blocks import (
    AssistantContentBlock,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
)
from longline.models.messages import Usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


async def stream_response(
    client: anthropic.AsyncAnthropic,
    *,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 16384,
    thinking: dict[str, Any] | None = None,
    temperature: float | None = None,
) -> AsyncIterator[QueryEvent]:
    """Stream a response from the Claude API and yield QueryEvents.

    Corresponds to TS: services/api/claude.ts queryModelWithStreaming().

    Uses the raw stream API to process SSE events directly, avoiding
    type union issues with the high-level stream wrapper.

    Yields:
        QueryEvent instances (TextDelta, ToolUseStart, etc.).
    """
    # 构建 API 请求参数
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "system": system,
    }

    if tools:
        params["tools"] = tools

    # thinking 模式（extended thinking）与 temperature 互斥：
    # 启用 thinking 时 API 不允许设置 temperature，否则会报错
    if thinking:
        params["thinking"] = thinking
    elif temperature is not None:
        # The caller's explicit value. The eval harness pins one here (see
        # `engine_factory.EVAL_TEMPERATURE`): it compares two arms against each
        # other and repeats every case, so the sampling temperature is part of
        # the configuration under test, and an unpinned one turns a
        # parameter change into what reads as model variance.
        params["temperature"] = temperature
    else:
        # Production's default, unchanged. Until now this literal was the ONLY
        # way temperature was ever set -- not overridable from outside, so the
        # eval layer could neither pin it nor record which value a batch used.
        params["temperature"] = 1.0  # Claude 默认 temperature=1.0，显式设置以保持一致性

    # === 流式响应的累积状态 ===
    # content_blocks: 以 SSE index 为 key，累积每个内容块的增量数据
    # 为什么用 dict 而非 list？因为 SSE 的 index 可能不连续（虽然实际中总是连续的）
    content_blocks: dict[int, dict[str, Any]] = {}
    final_content: list[AssistantContentBlock] = []  # 完成的内容块（content_block_stop 时添加）
    usage = Usage()
    # stop_reason 初始化为 None（与 TS 行为一致），在 message_delta 中赋值。
    # 如果流中断但没收到 message_delta，stop_reason 保持 None，
    # 最终在 yield TurnComplete 时 fallback 为 "end_turn"。
    stop_reason: str | None = None
    # 响应自己声明的模型名. 默认 None 的含义是「这条流没有告诉我们」, 而不是
    # 「就是请求里那个」: Anthropic 兼容网关 (实测 cc-switch) 会忽略请求里的
    # model 字段并服务它自己的模型, 把请求值当默认值写进 TurnComplete 会让上层
    # 记下一个从未运行过的模型.
    served_model: str | None = None

    try:
        # `client.messages.create(stream=True)` rather than the SDK's
        # `client.messages.stream()` helper.
        #
        # `.stream()` does not hand you the server's events; it routes them
        # through `MessageStream`, which replaces `message_start.message` with a
        # `ParsedMessage` -- a deliberately empty sentinel that fills in as
        # deltas arrive. `ParsedMessage.construct()` applies no defaults, so
        # `.content` reads back as `None` and `.model` as `None` for the whole
        # of the first block, and the accumulator is what raises on them.
        # Nothing here uses its output: the block state, the usage and the
        # served model are all accumulated from the raw events below, which are
        # the server's own. Taking the raw stream therefore loses nothing and
        # drops a layer that could only add a way to fail.
        async with await client.messages.create(stream=True, **params) as stream:
            async for event in stream:
                # 使用 getattr 而非直接属性访问，因为 SDK 的事件类型是 Union，
                # 不同事件有不同属性，直接访问会触发 AttributeError
                event_type = getattr(event, "type", "")

                # --- message_start: 一次响应的开始，携带 input token 统计 ---
                if event_type == "message_start":
                    msg = getattr(event, "message", None)
                    if msg:
                        # 响应声明的模型名. 取的是响应而非请求: 网关可能服务
                        # 一个与请求不同的模型, 而元数据必须描述真正跑过的那个.
                        served_model = getattr(msg, "model", None) or None
                        msg_usage = getattr(msg, "usage", None)
                        if msg_usage:
                            # input_tokens 和 cache_* 只在 message_start 中出现
                            # output_tokens 在 message_delta 中出现（P0-5 修复）
                            usage.input_tokens = getattr(msg_usage, "input_tokens", 0)
                            usage.cache_creation_input_tokens = getattr(
                                msg_usage, "cache_creation_input_tokens", 0
                            )
                            usage.cache_read_input_tokens = getattr(
                                msg_usage, "cache_read_input_tokens", 0
                            )

                # --- content_block_start: 初始化一个新的内容块 ---
                # 每个内容块用 index 标识，后续的 delta 和 stop 通过 index 关联
                elif event_type == "content_block_start":
                    idx: int = getattr(event, "index", 0)
                    cb = getattr(event, "content_block", None)
                    if cb is None:
                        continue
                    block_type: str = getattr(cb, "type", "")

                    if block_type == "text":
                        content_blocks[idx] = {"type": "text", "text": ""}
                    elif block_type == "tool_use":
                        # tool_use 的 input 是增量 JSON 字符串，先用 input_json 累积，
                        # 在 content_block_stop 时才做 json.loads 解析
                        content_blocks[idx] = {
                            "type": "tool_use",
                            "id": getattr(cb, "id", ""),
                            "name": getattr(cb, "name", ""),
                            "input_json": "",  # 增量拼接的 JSON 字符串
                        }
                    elif block_type == "thinking":
                        content_blocks[idx] = {"type": "thinking", "thinking": ""}
                    elif block_type == "redacted_thinking":
                        # redacted_thinking: Anthropic 内部审核后遮蔽的 thinking，data 是加密内容
                        content_blocks[idx] = {
                            "type": "redacted_thinking",
                            "data": getattr(cb, "data", ""),
                        }
                    else:
                        # P0-6: 记录未知块类型以便调试，而非静默跳过
                        logger.warning("Unknown content_block type: %s (index %d)", block_type, idx)

                # --- content_block_delta: 增量内容拼接 ---
                # 这是流式响应的核心——每个 delta 携带一小段增量文本
                elif event_type == "content_block_delta":
                    idx = getattr(event, "index", 0)
                    delta = getattr(event, "delta", None)
                    if delta is None or idx not in content_blocks:
                        continue

                    block = content_blocks[idx]
                    delta_type: str = getattr(delta, "type", "")

                    if delta_type == "text_delta":
                        text: str = getattr(delta, "text", "")
                        block["text"] += text    # 累积到状态中（用于构建最终 TextBlock）
                        yield TextDelta(text=text)  # 同时 yield 给 UI 实现逐字显示

                    elif delta_type == "input_json_delta":
                        # tool_use 的 input 以 JSON 片段形式增量到达，
                        # 不能每次 delta 都 json.loads（会失败），只能拼接字符串
                        block["input_json"] += getattr(delta, "partial_json", "")

                    elif delta_type == "thinking_delta":
                        thinking_text: str = getattr(delta, "thinking", "")
                        block["thinking"] += thinking_text
                        yield ThinkingDelta(text=thinking_text)

                # --- content_block_stop: 一个内容块完成 ---
                # 此时所有 delta 已接收完毕，可以安全地解析完整内容
                elif event_type == "content_block_stop":
                    idx = getattr(event, "index", 0)
                    if idx not in content_blocks:
                        continue

                    block = content_blocks[idx]
                    finished_block: AssistantContentBlock

                    if block["type"] == "text":
                        finished_block = TextBlock(text=block["text"])
                    elif block["type"] == "tool_use":
                        # P0-2: 将增量拼接的 JSON 字符串解析为 dict
                        # 对应 TS: normalizeContentFromAPI() 中的 JSON.parse + ?? {} 逻辑
                        try:
                            parsed_input = json.loads(block["input_json"]) if block["input_json"] else {}
                        except json.JSONDecodeError:
                            # JSON 解析失败时静默降级为空 dict（与 TS 行为一致）
                            # 记录诊断日志：工具名、JSON 长度、前 200 字符
                            logger.warning(
                                "tool_input_json_parse_fail: tool=%s len=%d input=%.200s",
                                block.get("name", "?"), len(block["input_json"]), block["input_json"],
                            )
                            parsed_input = {}  # TS: parsed ?? {} — 故意的静默降级

                        finished_block = ToolUseBlock(
                            id=block["id"],
                            name=block["name"],
                            input=parsed_input,
                        )
                        # ToolUseStart 在这里 yield 而非 content_block_start，
                        # 因为只有到 stop 时 input_json 才完整可解析
                        yield ToolUseStart(
                            tool_name=block["name"],
                            tool_id=block["id"],
                            input=parsed_input,
                        )
                    elif block["type"] == "thinking":
                        finished_block = ThinkingBlock(thinking=block["thinking"])
                    elif block["type"] == "redacted_thinking":
                        finished_block = RedactedThinkingBlock(data=block.get("data", ""))
                    else:
                        continue

                    final_content.append(finished_block)

                # --- message_delta: 响应级元数据（stop_reason + output_tokens） ---
                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    if delta:
                        # P0-1: 只在 stop_reason 非 None 时赋值，与 TS 行为一致
                        # 如果 API 未返回 stop_reason（流中断），保持 None → 最终 fallback "end_turn"
                        sr = getattr(delta, "stop_reason", None)
                        if sr is not None:
                            stop_reason = sr
                    evt_usage = getattr(event, "usage", None)
                    if evt_usage:
                        # output_tokens 只在 message_delta 中累积（message_start 里恒为 0）
                        usage.output_tokens = getattr(evt_usage, "output_tokens", 0)
                        # input_tokens 的位置则因网关而异. Anthropic 的约定是只放在
                        # message_start; 但实测的兼容网关 (cc-switch / omen-alpha)
                        # 在 message_start 里填 0, 把真实值放在 message_delta.
                        # 只按文档字段读会让每一行的 input_tokens 都是 0 —— 一份
                        # 少了整个输入部分的账单, 而输出部分看上去完全正常.
                        #
                        # 规则: 非零才算数, 且后到的非零值覆盖先前的. 这样两种约定
                        # 都对 —— Anthropic 的 delta 不带 input_tokens (None), 不动;
                        # 网关的 delta 带真实值, 覆盖掉 message_start 的 0.
                        delta_in = getattr(evt_usage, "input_tokens", None)
                        if delta_in:
                            usage.input_tokens = delta_in
                        for field in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                            val = getattr(evt_usage, field, None)
                            if val:
                                setattr(usage, field, val)

    except anthropic.APIStatusError as e:
        # === 错误映射: SDK 异常 → 内部 ErrorEvent ===
        # 从 response body 中提取 error.type 用于判断 overloaded_error
        error_type = ""
        if hasattr(e, "body") and isinstance(e.body, dict):
            error_info = e.body.get("error", {})
            if isinstance(error_info, dict):
                error_type = error_info.get("type", "")

        # 可恢复判断: 限流 (429)、过载 (529)、以及网关侧的瞬时故障
        # (502/503/504, 例如 "Upstream request failed: Endpoint is unavailable"),
        # 或者 body 中标记为 overloaded_error. query_loop Phase 3 会对可恢复错误做
        # 指数退避重试 (2s..10s, 上限 max_retry).
        #
        # 502/503/504 是这次实测补上的: 一次正式跑在网关上游不可用时,
        # 每个用例都直接记成 runtime_error 而不是退避重试, 204 行全是
        # 「没测到」. 4xx 一律保持致命: 400/404/413 是对请求本身的判决,
        # 重试只是在重复发送一个服务器已经按事实拒绝的东西.
        #
        # 例外 (也是实测补上的): 这个网关会把上游故障包进 400 的信封里
        # ("Upstream request failed: Model is unavailable", error.type =
        # server_error). 那不是对请求的判决, 是网关在报它上游的坟, 重试是
        # 恢复不是重犯. 判据是 body 里的 error.type, 不匹配错误文本 --
        # 文本是随上游方措辞变化的, 类型字段是结构化的.
        is_gateway_upstream = error_type in ("server_error", "overloaded_error")
        yield ErrorEvent(
            message=str(e),
            is_recoverable=(
                e.status_code in (429, 502, 503, 504, 529)
                or is_gateway_upstream
            ),
        )
        return

    except anthropic.APIConnectionError as e:
        # 网络连接错误（DNS、超时、TLS 等）一律标记为可恢复
        yield ErrorEvent(message=f"Connection error: {e}", is_recoverable=True)
        return

    # 正常完成: 将 stop_reason 的 None fallback 为 "end_turn"
    # 这个转换只在输出边界做（函数内部保持 None 语义），与 TS 行为一致
    yield TurnComplete(
        stop_reason=stop_reason or "end_turn", usage=usage, served_model=served_model,
    )
