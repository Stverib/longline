"""Teammate Mailbox - File-based messaging system for agent swarms.

Each teammate has an inbox file at ~/.longline/teams/{team_name}/inboxes/{agent_name}.json.
Other teammates can write messages to it, and the recipient sees them as attachments.

之所以采用文件系统而非内存队列，是因为：
1. agent 可能运行在不同进程中（跨进程通信需要持久化介质）
2. 消息需要在 agent 重启后仍然可读（持久化）
3. 实现简单可靠，不依赖外部消息中间件

Corresponds to TS: utils/teammateMailbox.ts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from longline.swarm.identity import sanitize_name

logger = logging.getLogger(__name__)

# 默认的 claude 配置目录，所有团队数据存储在其下的 teams/ 子目录中
_DEFAULT_CLAUDE_DIR = Path.home() / ".longline"


class InboxCorruptError(RuntimeError):
    """An inbox file held bytes that are not a message list.

    Raised instead of returning an empty list. The previous behaviour made a
    lost inbox and an empty inbox the same observable, so a truncation caused by
    a crash mid-write cost every message in the file while both the reader and
    the writer reported success -- a failure with no symptom, which is strictly
    worse than one with a loud one.

    `quarantine_path` names where the unreadable bytes were moved. They are kept
    rather than deleted because the number of messages lost is not recoverable
    from the error alone: a human, or a later repair, needs the original bytes
    to say how many died.
    """

    def __init__(self, agent_name: str, quarantine_path: Path) -> None:
        super().__init__(
            f"inbox for {agent_name!r} was not valid JSON and has been "
            f"quarantined to {quarantine_path}; its messages are lost"
        )
        self.agent_name = agent_name
        self.quarantine_path = quarantine_path


@dataclass
class TeammateMessage:
    """A message in a teammate's inbox.

    Corresponds to TS: utils/teammateMailbox.ts TeammateMessage type.
    """

    # 发送者的 agent 名称
    from_name: str
    # 消息正文
    text: str
    # Unix 时间戳，用于消息排序和过期检测
    timestamp: float
    # 是否已读标记，用于 receive() 方法过滤未读消息
    read: bool = False
    # 可选的摘要信息，供 team-lead 快速预览而无需读取完整 text
    summary: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize to dict for JSON storage."""
        # 使用 "from" 作为 JSON key（而非 "from_name"），与 TS 版本的数据格式保持一致
        d: dict[str, object] = {
            "from": self.from_name,
            "text": self.text,
            "timestamp": self.timestamp,
            "read": self.read,
        }
        # summary 为可选字段，仅在有值时写入 JSON，减少文件体积
        if self.summary is not None:
            d["summary"] = self.summary
        return d

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TeammateMessage:
        """Deserialize from dict."""
        return cls(
            from_name=str(data.get("from", "")),
            text=str(data.get("text", "")),
            timestamp=float(data["timestamp"]) if "timestamp" in data else 0.0,  # type: ignore[arg-type]
            read=bool(data.get("read", False)),
            summary=str(data["summary"]) if data.get("summary") is not None else None,
        )


class TeammateMailbox:
    """File-backed message store at ~/.longline/teams/{team}/inboxes/{agent}.json.

    每个 agent 拥有一个独立的 JSON 文件作为收件箱。
    发送消息 = 读取收件人的 JSON 文件 → 追加消息 → 写回文件。
    这是一种简单的"追加式"队列实现。

    === 关于"没有文件锁" ===

    这里没有文件锁, 而且**不要加**。理由是被测过的, 不是想当然的:

    - `send()` 从 `_read_inbox` 到 `_write_inbox` 全程同步 I/O, 中间没有
      `await`, 而单线程事件循环里同步函数体不可被抢占;
    - teammate 由 `spawn.py` 用 `asyncio.create_task` 起在**同一个事件循环**
      上, 全仓库没有 Thread / multiprocessing / run_in_executor。

    两条合起来, 读-改-写在这个运行时里是构造上原子的。并发压测跑出来的是
    lost=0 / dup=0 的**阴性结果** —— 它确认一个设计假设, 不是抓到一个 bug。
    给一个够不着的竞态加锁, 是拿复杂度换一个没有失败模式的东西。

    真正会丢消息的是崩溃持久性, 两处都已处理:
    - `_write_inbox` 用 temp + `os.replace`, 写一半被杀不会留下半份文件;
    - `_read_inbox` 遇到解析失败**隔离并抛错**, 不再 `return []` 把「收件箱
      丢了」与「收件箱是空的」变成同一个可观察量。

    若将来 teammate 改跑在多进程或多线程上, 上面第一条前提就没了, 那时再加锁。

    Corresponds to TS: utils/teammateMailbox.ts (readMailbox, writeToMailbox, markAllAsRead).
    """

    def __init__(self, team_name: str, claude_dir: Path | None = None) -> None:
        self._team_name = team_name
        self._claude_dir = claude_dir or _DEFAULT_CLAUDE_DIR
        # 收件箱目录路径：~/.longline/teams/{sanitized_team_name}/inboxes/
        self._inbox_dir = (
            self._claude_dir / "teams" / sanitize_name(team_name) / "inboxes"
        )

    def _inbox_path(self, agent_name: str) -> Path:
        """Get the path to an agent's inbox file."""
        # 对 agent_name 进行 sanitize，确保文件名不包含非法字符
        safe_name = sanitize_name(agent_name)
        return self._inbox_dir / f"{safe_name}.json"

    def _ensure_inbox_dir(self) -> None:
        """Ensure the inbox directory exists."""
        # parents=True 递归创建所有缺失的父目录；exist_ok=True 如果目录已存在不报错
        self._inbox_dir.mkdir(parents=True, exist_ok=True)

    def _read_inbox(self, agent_name: str) -> list[TeammateMessage]:
        """Read all messages from an agent's inbox file.

        A MISSING file is an empty inbox -- a normal state, not a fault.
        Unreadable BYTES are a fault: the file exists and something wrote it, so
        a parse failure means messages were lost, and the caller has to be told.

        `OSError` still degrades to a warning and an empty list. A transient
        read error is not evidence of loss the way a parse failure is, and
        turning it into an exception would let an unrelated filesystem hiccup
        take down a teammate.
        """
        path = self._inbox_path(agent_name)
        # 收件箱文件不存在说明该 agent 从未收到过消息，返回空列表
        if not path.exists():
            return []

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Failed to read inbox for %s: %s", agent_name, e)
            return []

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            quarantine = path.with_name(f"{path.name}.corrupt")
            # 后缀递增而不是直接覆盖: 第二次损坏若发生在第一次被处理之前,
            # 覆盖会连同第一次的字节一起丢掉, 而那些字节是「丢了几条」的唯一证据
            counter = 1
            while quarantine.exists():
                quarantine = path.with_name(f"{path.name}.corrupt{counter}")
                counter += 1
            with contextlib.suppress(OSError):
                os.replace(path, quarantine)
            logger.error(
                "Inbox for %s was corrupt (%s); quarantined to %s",
                agent_name, e, quarantine,
            )
            raise InboxCorruptError(agent_name, quarantine) from e

        return [TeammateMessage.from_dict(m) for m in data]

    def _write_inbox(self, agent_name: str, messages: list[TeammateMessage]) -> None:
        """Write the inbox atomically: either the old bytes or the new ones.

        `Path.write_text` truncates and then writes, so a crash in between
        leaves a file that is neither the old inbox nor the new one. That costs
        the WHOLE inbox rather than the newest message, because `_read_inbox`
        cannot tell a truncated file from an empty one.

        The temp file is created in the DESTINATION directory, not the system
        temp dir: `os.replace` is only atomic within one filesystem, and across
        a mount it degrades to copy-then-delete -- silently reopening the window
        it was meant to close, and only on machines where the temp dir is a
        separate volume.

        A failed write removes its own temp file. Leaving it behind would put
        one stale file in the inbox directory per failure, and nothing ever
        looks at those names again.
        """
        self._ensure_inbox_dir()
        path = self._inbox_path(agent_name)
        # 每次写入都是全量覆盖 (非增量追加), 因此需要先读取再追加再写回
        data = [m.to_dict() for m in messages]

        fd, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2)
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise

    def send(self, to: str, message: TeammateMessage) -> None:
        """Write a message to a teammate's inbox.

        Corresponds to TS: utils/teammateMailbox.ts writeToMailbox().

        Args:
            to: The recipient's agent name.
            message: The message to send (read flag forced to False).
        """
        # 强制将 read 标记设为 False，确保收件人的 receive() 能正确获取到这条新消息
        message.read = False
        # 读取收件人当前的所有消息 → 追加新消息 → 全量写回
        messages = self._read_inbox(to)
        messages.append(message)
        self._write_inbox(to, messages)
        logger.debug(
            "Wrote message to %s's inbox from %s", to, message.from_name
        )

    def receive(self, agent_name: str) -> list[TeammateMessage]:
        """Read all unread messages from an agent's inbox.

        Corresponds to TS: utils/teammateMailbox.ts readUnreadMessages().

        Returns:
            List of unread messages.
        """
        messages = self._read_inbox(agent_name)
        # 只返回未读消息；调用方需要在处理完后调用 mark_all_read() 标记已读
        return [m for m in messages if not m.read]

    def receive_all(self, agent_name: str) -> list[TeammateMessage]:
        """Read all messages (read and unread) from an agent's inbox.

        Corresponds to TS: utils/teammateMailbox.ts readMailbox().
        """
        return self._read_inbox(agent_name)

    def mark_all_read(self, agent_name: str) -> None:
        """Mark all messages in an agent's inbox as read.

        Corresponds to TS: utils/teammateMailbox.ts markAllAsRead().
        """
        messages = self._read_inbox(agent_name)
        changed = False
        for m in messages:
            if not m.read:
                m.read = True
                changed = True
        # 仅在确实有消息被标记时才写入文件，避免不必要的磁盘 IO
        if changed:
            self._write_inbox(agent_name, messages)
            logger.debug("Marked all messages as read for %s", agent_name)
