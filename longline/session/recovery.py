"""Session recovery — validate and repair transcripts.

Corresponds to TS: utils/conversationRecovery.ts.

Ensures transcripts are API-safe after resume (crash, process exit, etc.).

会话恢复模块：处理由于进程崩溃、意外退出等原因导致的对话记录不完整问题。
核心场景是 assistant 发出了 tool_use 请求，但进程在工具执行前/执行中就终止了，
导致对话记录缺少对应的 tool_result。这在 API 提交时会导致验证错误。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from longline.models.messages import (
    AssistantMessage,
    Message,
    UserMessage,
)

logger = logging.getLogger(__name__)

# 合成的工具结果占位符文本，与 TS 版本保持一致
# 这条消息会出现在恢复后的对话中，告知模型该工具调用因内部错误未能完成
SYNTHETIC_TOOL_RESULT_PLACEHOLDER = "[Tool result missing due to internal error]"


@dataclass
class TranscriptRepairReport:
    """What `validate_transcript()` had to repair, if anything.

    Previously "did this call repair anything?" was a local variable that only
    reached a log line, which made it unusable as evidence: the recovery metric
    has to record `transcript_repaired` per case, and a fact that exists only in
    a log is not a fact a report can be recomputed from.

    It behaves as a list of the repair kinds that fired, because that is the
    most useful shape for a reader -- ``"truncated_tail" in report`` is how you
    ask "did the crash-truncation repair run". The authoritative answer to "was
    anything repaired" is `repaired`, which is set from the same code path that
    writes the log line, so the two cannot disagree.
    """

    repairs: list[str] = field(default_factory=list)
    orphaned_tool_use_ids: list[str] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        """True when at least one repair was applied."""
        return bool(self.repairs)

    def __bool__(self) -> bool:
        """Falsy when nothing was repaired, so `if report:` reads as expected."""
        return self.repaired

    def __contains__(self, kind: object) -> bool:
        return kind in self.repairs

    def __iter__(self) -> object:
        """Iterate the repair kinds, so the report reads as a list of what fired."""
        return iter(self.repairs)

    def __len__(self) -> int:
        return len(self.repairs)

    def to_dict(self) -> dict[str, object]:
        return {
            "repaired": self.repaired,
            "repairs": self.repairs,
            "num_orphaned_tool_uses": len(self.orphaned_tool_use_ids),
            "orphaned_tool_use_ids": self.orphaned_tool_use_ids,
        }


# The repair kinds `TranscriptRepairReport.repairs` may contain.
REPAIR_TRUNCATED_TAIL = "truncated_tail"
REPAIR_MID_ORPHANS = "mid_orphans"


def validate_transcript(
    messages: list[Message],
    *,
    report: TranscriptRepairReport | None = None,
) -> list[Message]:
    """Validate and repair a transcript for API submission.

    Corresponds to TS: conversationRecovery.ts validateTranscript().

    修复策略（按检查顺序）：
    1. 末尾截断检测：对话以包含 tool_use 的 assistant 消息结尾
       → 追加一条合成的 user 消息，包含 is_error=True 的 tool_result
    2. 中间孤立 tool_use：assistant 发出 tool_use，但后续 user 消息缺少对应 tool_result
       → 在该 user 消息中补入合成的 tool_result
    3. 角色交替违规（连续相同角色的消息）
       → 由 normalize_messages_for_api() 在下游处理，此处不涉及

    `report` is an optional out-parameter: the caller passes a fresh
    `TranscriptRepairReport` and reads back which repairs fired. It is
    keyword-only and defaults to None, so every existing caller and test is
    unchanged -- a silent in-place mutation is not possible, because the object
    is created by the caller rather than by this function.

    Returns the repaired message list (may modify in place).
    """
    if not messages:
        return messages

    repairs = report if report is not None else TranscriptRepairReport()

    # ---- 修复 1: 末尾截断 ----
    # 这是最常见的崩溃场景：assistant 发出 tool_use 后，
    # 进程在工具执行前就终止了，tool_result 从未被追加
    if isinstance(messages[-1], AssistantMessage):
        tool_uses = messages[-1].get_tool_use_blocks()
        if tool_uses:
            # 为每个孤立的 tool_use 生成一条错误 tool_result
            from longline.models.content_blocks import ToolResultBlock

            synthetic_results = [
                ToolResultBlock(
                    tool_use_id=tu.id,
                    content=SYNTHETIC_TOOL_RESULT_PLACEHOLDER,
                    is_error=True,
                )
                for tu in tool_uses
            ]
            # 将合成结果包装为 user 消息追加到末尾，满足 API 的角色交替要求
            messages.append(UserMessage(content=synthetic_results))  # type: ignore[arg-type]
            logger.warning(
                "Transcript recovery: added synthetic results for %d orphaned tool_use(s)",
                len(tool_uses),
            )
            repairs.repairs.append(REPAIR_TRUNCATED_TAIL)
            repairs.orphaned_tool_use_ids.extend(tu.id for tu in tool_uses)

    # ---- 修复 2: 中间孤立 tool_use ----
    # 较少见但仍可能发生：对话在工具执行中间崩溃，之后用户继续了新的对话，
    # 导致 tool_use 后面跟着的 user 消息缺少某些 tool_result
    for i in range(len(messages) - 1):
        msg_i = messages[i]
        if not isinstance(msg_i, AssistantMessage):
            continue
        tool_uses = msg_i.get_tool_use_blocks()
        if not tool_uses:
            continue

        # 检查下一条消息是否为 user 消息（应包含 tool_result）
        next_msg = messages[i + 1]
        if not isinstance(next_msg, UserMessage):
            continue

        # 收集后续 user 消息中已有的 tool_result ID
        existing_result_ids: set[str] = set()
        if isinstance(next_msg.content, list):
            from longline.models.content_blocks import ToolResultBlock

            for block in next_msg.content:
                if isinstance(block, ToolResultBlock):
                    existing_result_ids.add(block.tool_use_id)

        # 找出缺失 tool_result 的 tool_use（ID 不在已有结果中的）
        missing = [tu for tu in tool_uses if tu.id not in existing_result_ids]
        if missing:
            from longline.models.content_blocks import ToolResultBlock

            synthetic = [
                ToolResultBlock(
                    tool_use_id=tu.id,
                    content=SYNTHETIC_TOOL_RESULT_PLACEHOLDER,
                    is_error=True,
                )
                for tu in missing
            ]
            # 将合成结果插入到现有 user 消息的内容前面（prepend）
            # 这样 tool_result 在消息中的位置与 API 期望的顺序一致
            if isinstance(next_msg.content, str):
                # user 消息原本是纯文本，需要转换为混合内容列表
                from longline.models.content_blocks import TextBlock

                next_msg.content = [*synthetic, TextBlock(text=next_msg.content)]
            elif isinstance(next_msg.content, list):
                next_msg.content = [*synthetic, *next_msg.content]
            logger.warning(
                "Transcript recovery: patched %d missing tool_result(s) at message %d",
                len(missing), i,
            )
            repairs.repairs.append(REPAIR_MID_ORPHANS)
            repairs.orphaned_tool_use_ids.extend(tu.id for tu in missing)

    if repairs.repaired:
        logger.info("Transcript recovery completed — %d messages", len(messages))

    return messages
