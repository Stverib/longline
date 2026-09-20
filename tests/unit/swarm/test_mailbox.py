"""Tests for longline.swarm.mailbox — file-backed teammate messaging."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from longline.swarm.mailbox import (
    InboxCorruptError,
    TeammateMailbox,
    TeammateMessage,
)


@pytest.fixture
def mailbox(tmp_path: object) -> TeammateMailbox:
    """Create a mailbox with a temp claude dir."""
    from pathlib import Path

    return TeammateMailbox("test-team", claude_dir=Path(str(tmp_path)))


class TestTeammateMessage:
    def test_roundtrip(self) -> None:
        msg = TeammateMessage(
            from_name="alice",
            text="hello",
            timestamp=1234567890.0,
            read=False,
            summary="greeting",
        )
        d = msg.to_dict()
        restored = TeammateMessage.from_dict(d)
        assert restored.from_name == "alice"
        assert restored.text == "hello"
        assert restored.timestamp == 1234567890.0
        assert restored.read is False
        assert restored.summary == "greeting"

    def test_roundtrip_no_summary(self) -> None:
        msg = TeammateMessage(from_name="bob", text="hi", timestamp=0)
        d = msg.to_dict()
        assert "summary" not in d
        restored = TeammateMessage.from_dict(d)
        assert restored.summary is None


def _msg(text: str, *, from_name: str = "alice") -> TeammateMessage:
    """A message whose only distinguishing field is its text."""
    return TeammateMessage(from_name=from_name, text=text, timestamp=0.0)


class TestInboxWriteIsAtomic:
    """A failed write must leave the previous inbox intact.

    `Path.write_text` truncates first, so a crash between the truncate and the
    write leaves a half-file. That costs more than the newest message: the old
    bytes are gone AND the new ones are incomplete, so the inbox holds neither
    version. `_read_inbox` cannot tell a truncated file from an empty one, which
    is why the whole inbox is the unit of loss.
    """

    def test_a_failed_replace_leaves_the_old_inbox_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        box = TeammateMailbox("team-a", claude_dir=tmp_path)
        for index in range(3):
            box.send("worker1", _msg(f"m{index}"))
        path = box._inbox_path("worker1")
        before = path.read_bytes()

        def _boom(*args: object, **kwargs: object) -> None:
            raise OSError("simulated crash at the last step")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(OSError):
            box.send("worker1", _msg("m3"))

        assert path.read_bytes() == before
        assert [m.text for m in box.receive_all("worker1")] == ["m0", "m1", "m2"]

    def test_no_temp_file_survives_a_failed_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A leaked temp file per failed write is its own slow leak.

        Nothing reads these names again, so they accumulate in the inbox
        directory for the life of the team.
        """
        box = TeammateMailbox("team-b", claude_dir=tmp_path)
        box.send("worker1", _msg("m0"))

        def _boom(*args: object, **kwargs: object) -> None:
            raise OSError("simulated crash")

        monkeypatch.setattr(os, "replace", _boom)
        with pytest.raises(OSError):
            box.send("worker1", _msg("m1"))

        leftovers = [
            p for p in box._inbox_path("worker1").parent.iterdir()
            if p.suffix == ".tmp"
        ]
        assert leftovers == []

    def test_the_temp_file_is_in_the_destination_directory(self, tmp_path: Path) -> None:
        """`os.replace` is only atomic within one filesystem.

        A temp file in the system temp directory can land on another mount, at
        which point `os.replace` degrades to copy-then-delete and the window it
        was meant to close reopens -- silently, and only on machines where the
        temp dir is a different volume.
        """
        box = TeammateMailbox("team-c", claude_dir=tmp_path)
        observed: list[str] = []
        real_replace = os.replace

        def _spy(src: object, dst: object) -> None:
            observed.append(str(src))
            real_replace(src, dst)  # type: ignore[arg-type]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "replace", _spy)
            box.send("worker1", _msg("m0"))

        assert observed, "the write did not go through a replace at all"
        assert Path(observed[0]).parent == box._inbox_path("worker1").parent


class TestCorruptInboxIsNotSilentlyEmptied:
    """A truncated inbox must be reported, not read as an empty one.

    Before the fix, `_read_inbox` caught the parse error and returned `[]`, so a
    teammate whose inbox was lost saw "no messages" and carried on, and the
    leader received a reply that never mentioned the lost work. Both parties
    reported success while the messages were gone -- the failure mode with no
    symptom, which is strictly worse than one with a loud one.
    """

    def test_a_corrupt_inbox_raises_and_is_quarantined(self, tmp_path: Path) -> None:
        box = TeammateMailbox("team-d", claude_dir=tmp_path)
        box.send("worker1", _msg("m0"))
        path = box._inbox_path("worker1")
        path.write_text('[{"from_name": "alice", "text":', encoding="utf-8")

        with pytest.raises(InboxCorruptError) as excinfo:
            box.receive("worker1")

        assert excinfo.value.agent_name == "worker1"
        assert excinfo.value.quarantine_path.exists()
        assert not path.exists(), "the next read must start from a clean slate"

    def test_the_quarantined_bytes_are_not_overwritten_by_a_second_corruption(
        self, tmp_path: Path
    ) -> None:
        """Losing the earlier bytes loses the only evidence of what was lost."""
        box = TeammateMailbox("team-e", claude_dir=tmp_path)
        path = box._inbox_path("worker1")
        box._ensure_inbox_dir()

        path.write_text("first corruption", encoding="utf-8")
        with pytest.raises(InboxCorruptError) as first:
            box.receive("worker1")
        path.write_text("second corruption", encoding="utf-8")
        with pytest.raises(InboxCorruptError) as second:
            box.receive("worker1")

        assert first.value.quarantine_path != second.value.quarantine_path
        assert first.value.quarantine_path.read_text(encoding="utf-8") == "first corruption"
        assert second.value.quarantine_path.read_text(encoding="utf-8") == "second corruption"

    def test_a_missing_inbox_is_still_an_empty_inbox(self, tmp_path: Path) -> None:
        """The distinction the fix turns on: absent is normal, unreadable is loss."""
        box = TeammateMailbox("team-f", claude_dir=tmp_path)

        assert box.receive_all("never-written") == []
        assert not box._inbox_path("never-written").exists()

    def test_a_healthy_inbox_still_reads(self, tmp_path: Path) -> None:
        box = TeammateMailbox("team-g", claude_dir=tmp_path)
        box.send("worker1", _msg("m0"))

        assert [m.text for m in box.receive("worker1")] == ["m0"]
        box.mark_all_read("worker1")
        assert box.receive("worker1") == []
        assert [m.text for m in box.receive_all("worker1")] == ["m0"]


class TestTeammateMailbox:
    def test_send_and_receive(self, mailbox: TeammateMailbox) -> None:
        msg = TeammateMessage(
            from_name="alice", text="task done", timestamp=time.time()
        )
        mailbox.send("bob", msg)
        unread = mailbox.receive("bob")
        assert len(unread) == 1
        assert unread[0].from_name == "alice"
        assert unread[0].text == "task done"
        assert unread[0].read is False

    def test_receive_empty(self, mailbox: TeammateMailbox) -> None:
        assert mailbox.receive("nobody") == []

    def test_mark_all_read(self, mailbox: TeammateMailbox) -> None:
        mailbox.send("bob", TeammateMessage(from_name="a", text="1", timestamp=1.0))
        mailbox.send("bob", TeammateMessage(from_name="b", text="2", timestamp=2.0))
        assert len(mailbox.receive("bob")) == 2

        mailbox.mark_all_read("bob")
        assert len(mailbox.receive("bob")) == 0
        # But receive_all still returns them
        assert len(mailbox.receive_all("bob")) == 2

    def test_multiple_sends(self, mailbox: TeammateMailbox) -> None:
        for i in range(3):
            mailbox.send(
                "charlie",
                TeammateMessage(from_name="alice", text=f"msg-{i}", timestamp=float(i)),
            )
        all_msgs = mailbox.receive_all("charlie")
        assert len(all_msgs) == 3
        assert [m.text for m in all_msgs] == ["msg-0", "msg-1", "msg-2"]

    def test_send_forces_unread(self, mailbox: TeammateMailbox) -> None:
        msg = TeammateMessage(from_name="x", text="y", timestamp=0, read=True)
        mailbox.send("z", msg)
        unread = mailbox.receive("z")
        assert len(unread) == 1
        assert unread[0].read is False

    def test_separate_inboxes(self, mailbox: TeammateMailbox) -> None:
        mailbox.send("alice", TeammateMessage(from_name="x", text="for alice", timestamp=1.0))
        mailbox.send("bob", TeammateMessage(from_name="x", text="for bob", timestamp=2.0))
        assert len(mailbox.receive("alice")) == 1
        assert len(mailbox.receive("bob")) == 1
        assert mailbox.receive("alice")[0].text == "for alice"
        assert mailbox.receive("bob")[0].text == "for bob"
