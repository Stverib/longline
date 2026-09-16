"""Leakage guard for the blind tool-selection cases.

`evals/README.md` §8.1 makes "任务文本不出现任何工具名" a merge blocker, and
§5.2 makes the blind/instruction-following split the boundary between the
resume-facing number and a regression check. Both rules are only as good as
their enforcement, so this module enforces them mechanically.

**What this covers, and what it does not.**

Two checks run here:

1. A literal scan for registered tool names (English, case-insensitive). This
   one is exact — a hit is always a real leak.
2. A literal scan for Chinese words that name a *means* rather than a *goal*
   (`TOOL_HINT_WORDS` below). This one is a **heuristic**. It catches the
   phrasings that leaked in the retired `tool_calls.jsonl` ("用 grep",
   "文件名里带", "文件内容"), but it cannot catch a novel paraphrase, and no
   word list will. It is a floor, not a proof.

Each blind case therefore also carries a hand-written `blind_rationale`. The
heuristic catches the mechanical leaks; the rationale is what a human reviewer
actually checks. A case whose rationale is weak is a leak this test will pass,
which is why the reviewer is told to read them rather than trust the green
checkmark.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from longline.eval.types import ToolCallCase

BLIND_TAG = "blind"
INSTRUCTION_FOLLOWING_TAG = "instruction-following"

# Tool names from every eval profile. A blind task must contain none of them.
REGISTERED_TOOL_NAMES: tuple[str, ...] = (
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "WebSearch", "WebFetch", "NotebookEdit",
    "TaskCreate", "TaskGet", "TaskList", "TaskUpdate", "TaskStop",
)

# Chinese words that name the MEANS rather than the GOAL. Grouped by the family
# they would give away, so a reviewer can see which leak they are guarding.
#
# Maintenance rule: when a reviewer rejects a blind case for hinting at a tool,
# add the word that gave it away here. This list is expected to grow; a case
# that only passes because a hint is missing from this list is a false negative.
TOOL_HINT_WORDS: dict[str, tuple[str, ...]] = {
    # 直接点名工具
    "explicit": ("grep", "glob", "bash", "shell", "notebook", "jupyter"),
    # 「搜索/查找内容」逼出 Grep,「文件名/按名字」逼出 Glob
    "grep_vs_glob": (
        "文件名", "按名字", "名字匹配", "正则表达式", "文本内容", "文件内容",
        "内容中", "搜索内容", "包含字符串", "字符串搜索", "模式匹配",
    ),
    # 「执行命令/终端」逼出 Bash
    "bash": ("执行命令", "运行命令", "命令行", "终端", "shell 命令", "跑一下命令"),
    # 「上网/联网/抓取网页」逼出 WebSearch / WebFetch
    "web": ("上网", "联网", "网上搜", "搜索引擎", "抓取网页", "请求网址", "访问链接"),
    # 「单元格/cell」基本等于 NotebookEdit
    "notebook": ("单元格", "cell"),
    # 「任务列表/待办/跟踪任务」逼出 Task 系列
    "task": ("任务列表", "待办清单", "任务 ID", "task id", "标记任务"),
}


def _pattern(words: tuple[str, ...]) -> re.Pattern[str]:
    """Alternation over literal words, longest first so overlaps report fully."""
    escaped = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile("|".join(escaped), re.IGNORECASE)


_TOOL_NAME_WORD_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(REGISTERED_TOOL_NAMES) + r")(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_HINT_PATTERNS: dict[str, re.Pattern[str]] = {
    family: _pattern(words) for family, words in TOOL_HINT_WORDS.items()
}


def find_tool_name_leaks(task: str) -> list[str]:
    """Registered tool names appearing in `task` as standalone words.

    Word-bounded and case-insensitive, so `README.md` does not read as the
    `Read` tool and `edition` does not read as `Edit`. Both occur naturally in
    task text that leaks nothing; a bare substring scan would flag them and
    pressure a reviewer into weakening the name list.
    """
    return sorted({m.group(0).lower() for m in _TOOL_NAME_WORD_PATTERN.finditer(task)})


def find_hint_leaks(task: str) -> list[tuple[str, str]]:
    """(family, word) pairs whose Chinese phrasing hints at a tool.

    Heuristic — see the module docstring.
    """
    hits: list[tuple[str, str]] = []
    for family, pattern in _HINT_PATTERNS.items():
        hits.extend((family, m.group(0)) for m in pattern.finditer(task))
    return hits


def blind_cases(cases: list[ToolCallCase]) -> list[ToolCallCase]:
    return [c for c in cases if BLIND_TAG in c.tags]


def instruction_following_cases(cases: list[ToolCallCase]) -> list[ToolCallCase]:
    return [c for c in cases if INSTRUCTION_FOLLOWING_TAG in c.tags]
