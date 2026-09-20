"""Case data model and loader for the single- vs multi-agent A/B suite.

=== What this measures (evals/README.md §5.6, plan §4.6) ===

```text
SuccessRate   = judge-passing cases / total          (both variants, SAME judges)
WallClockTime = measured duration
Speedup       = single_wall_time / multi_wall_time
TokenOverhead = (multi_tokens - single_tokens) / single_tokens
```

=== Two groups that must never be merged ===

`group: "controlled"` -- the case **pre-declares its independent subtasks**.
`single` runs them sequentially in one agent; `multi` fans them out to 2-4
workers and the leader merges. Both variants do the *same work*, which is the
only thing that makes a speedup number mean anything: comparing a fan-out
against a sequential agent that was asked a different question measures the
questions, not the architecture.

`group: "exploratory"` -- the coordinator decomposes freely. Its subtasks are
not declared, so its work is NOT the same as any other variant's, and the
contract is explicit that it is reported separately. Merging it into the
controlled number would compare different work.

=== Why the subtasks are declared, and what "the same work" is enforced by ===

Each subtask names the file it writes (`writes`). Two consequences follow, and
both are checked rather than promised:

1. **Non-overlapping paths.** The fan-out runs subtasks concurrently, so two
   subtasks writing the same path race, and the loser's work vanishes. The
   loader rejects a controlled case whose subtasks share a `writes` target
   (contract §5.6 / §8.3).
2. **The union of the subtasks' artifacts is the case's expected file set.**
   `subtask_paths()` derives it from the declarations, so the single and multi
   variants are graded against a file list that neither variant authored --
   see `judges.expected_paths`. A variant that produced extra files, or dropped
   one, fails on the same check the other variant is held to.

=== Fixtures: two sibling trees, never one shared tree ===

A controlled case names `fixture_single` and `fixture_multi`: two directories
that are byte-identical copies of each other. Both variants must start from the
same state or the comparison is contaminated, and they cannot share ONE
directory because `_prepare_sandbox` copies it into a temp sandbox -- the copy
is what protects the fixture, but a case that ever wrote back would poison the
next case. The loader proves the two trees are identical (same relative paths,
same bytes) rather than trusting the author to have kept them in sync.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from longline.eval.types import CaseParseError, E2ECase

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

# The two variants, matching the contract's `single_agent` / `multi_agent`
# wording (evals/README.md §4.0 and §5.6). These strings are the `variant`
# field on every CaseResult the suite writes.
SINGLE = "single_agent"
MULTI = "multi_agent"
VARIANTS: tuple[str, str] = (SINGLE, MULTI)

# The two groups. A case belongs to exactly one; there is no "both".
CONTROLLED = "controlled"
EXPLORATORY = "exploratory"
GROUPS: tuple[str, ...] = (CONTROLLED, EXPLORATORY)

# The task-shape axis, orthogonal to `group`. `group` answers "did both arms do
# the same work"; `category` answers "what shape was that work". They are
# independent: a `controlled` case can be any of the three, and the two axes
# constrain each other in no direction.
#
# The categories are never pooled into one average. Such an average would be a
# fact about the mix of shapes in the corpus rather than about the
# architecture -- rebalance the corpus and the number moves without anything
# about the runtime having changed.
CATEGORY_ANALYSIS = "parallel_analysis"
CATEGORY_MODIFICATION = "parallel_modification"
CATEGORY_DEPENDENT = "dependent"
CATEGORIES: tuple[str, ...] = (
    CATEGORY_ANALYSIS,
    CATEGORY_MODIFICATION,
    CATEGORY_DEPENDENT,
)

# The only `workers` value a `dependent` case may declare. Not a concurrency
# limit -- a serial chain has none -- but the number the run metadata reports as
# the agent count, and reporting 2 would claim a parallelism the case forbids.
DEPENDENT_WORKERS = 1

# Contract §5.6: "2~4 个子 Agent 并行执行". A controlled case must declare at
# least two subtasks (below that there is nothing to fan out) and at most four
# workers run concurrently. The upper bound is a contract number, not a
# preference: an exploratory case may declare more subtasks than workers, and
# the leader is free to schedule them, but a controlled case's declared
# parallelism is what the run metadata records as the agent count.
MIN_WORKERS = 2
MAX_WORKERS = 4

# What the leader does with the worker replies once they are all in. It is
# declared rather than assumed because the two variants must perform it
# identically -- a merge that only the multi variant runs would be extra work
# the single variant never did, and the token overhead would be measuring it.
MERGE_WRITE = "write_required"
MERGE_MODES: tuple[str, ...] = (MERGE_WRITE,)

Group = Literal["controlled", "exploratory"]
Category = Literal["parallel_analysis", "parallel_modification", "dependent"]


@dataclass
class Subtask:
    """One declared unit of independent work.

    Fields:
        id: stable identifier, unique within a case. The per-subtask verdicts
            in the failure report are keyed by it.
        instruction: what the worker is asked to do. For the single variant
            this is the text the one agent is handed; for the multi variant it
            is the prompt each teammate is spawned with. Identical either way,
            which is what makes the two variants comparable.
        writes: the path (relative to the sandbox) this subtask is expected to
            produce. Load-bearing in three places: it is what the loader checks
            for overlap, what `subtask_paths()` derives the expected file set
            from, and what the leader's merge step names.
    """

    id: str
    instruction: str
    writes: str
    # Declared predecessors, empty for every subtask of a non-chain case. Only
    # `dependent` cases populate it, and only there is it load-bearing: the
    # order the runner drives the steps in is derived from this, not from the
    # order they happen to appear in the case file.
    depends_on: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, case_id: str) -> Subtask:
        sid = d.get("id")
        instruction = d.get("instruction")
        writes = d.get("writes")
        if not isinstance(sid, str) or not sid:
            raise CaseParseError(f"{case_id}: subtask requires a string 'id', got {d!r}")
        if not isinstance(instruction, str) or not instruction.strip():
            raise CaseParseError(f"{case_id}/{sid}: subtask requires a non-empty 'instruction'")
        if not isinstance(writes, str) or not writes:
            raise CaseParseError(f"{case_id}/{sid}: subtask requires a string 'writes'")
        # A path that escapes the sandbox would put the agent's artifact
        # somewhere the judge never looks, and (worse) somewhere the *fixture
        # root* might be. Absolute paths and `..` are rejected for the same
        # reason `resolve_fixture` rejects them.
        if writes.startswith(("/", "\\")) or ".." in writes.replace("\\", "/").split("/"):
            raise CaseParseError(
                f"{case_id}/{sid}: 'writes' must be a relative path inside the "
                f"sandbox, got {writes!r}"
            )
        raw_deps = d.get("depends_on", [])
        if not isinstance(raw_deps, list) or not all(
            isinstance(dep, str) and dep for dep in raw_deps
        ):
            raise CaseParseError(
                f"{case_id}/{sid}: 'depends_on' must be a list of non-empty "
                f"subtask ids, got {raw_deps!r}"
            )
        return cls(
            id=sid, instruction=instruction, writes=writes,
            depends_on=tuple(raw_deps),
        )


@dataclass
class MultiAgentCase(E2ECase):
    """A task with declared independent subtasks, run as single and as multi.

    Subclasses `E2ECase` for the same reason `CompressionCase` does: the
    artifact is judged by the same deterministic `checks` and the same
    `case_passed` dispatch as every other E2E case. The A/B changes *how the
    subtasks are executed*, not how their output is graded -- which is what
    keeps `SuccessRate` here meaning the same thing it means in §5.1.

    Fields beyond `E2ECase`:
        group: `controlled` or `exploratory`. Reported separately, always.
        subtasks: the declared independent units (empty for exploratory-only
            cases that want the coordinator to decompose freely).
        workers: how many sub-agents the multi variant runs concurrently.
        merge: what the leader does after the workers finish. `write_required`
            means it writes `merge_file`, and that file is among the expected
            artifacts for BOTH variants.
        merge_file: the leader's own artifact path, relative to the sandbox.
        fixture_single / fixture_multi: sibling fixtures, proven identical at
            load time. `fixture` (inherited) is left unset by the loader so a
            reader cannot accidentally run the case against one of the two
            trees and silently measure a different starting state.
    """

    group: Group = "controlled"
    category: Category = "parallel_analysis"
    subtasks: list[Subtask] = field(default_factory=list)
    workers: int = MIN_WORKERS
    merge: str = MERGE_WRITE
    merge_file: str = ""
    fixture_single: str = ""
    fixture_multi: str = ""
    # Path to this case's hidden judge, relative to the fixtures root, or "".
    # Kept OUT of the fixture trees on purpose: those trees are copied into the
    # agent's sandbox, so a test living in one is readable by the model and the
    # case would measure reading rather than doing. The runner copies this file
    # into the sandbox at judge time.
    hidden_test: str = ""

    # --- contract fields derived from the declarations ---------------------

    def subtask_paths(self) -> list[str]:
        """Every artifact the declared subtasks are expected to produce.

        Derived from `subtasks`, never stored: a field alongside them could
        disagree with the declarations, and the file set both variants are
        judged against would then be authored by the case file twice.
        """
        return [s.writes for s in self.subtasks]

    def expected_paths(self) -> list[str]:
        """The case's full expected artifact set, sorted.

        The union of the subtasks' `writes` (present in BOTH variants, since
        both do the whole work) plus the leader's merge file. Sorted so the
        judge and the report agree on the order without either one sorting.
        """
        return sorted({*self.subtask_paths(), self.merge_file} - {""})

    @property
    def num_subtasks(self) -> int:
        return len(self.subtasks)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MultiAgentCase:
        cid = d.get("id")
        if not isinstance(cid, str) or not cid:
            raise CaseParseError(f"multi_agent case requires a string 'id', got {d!r}")

        group = d.get("group")
        if group not in GROUPS:
            raise CaseParseError(
                f"{cid}: 'group' must be one of {list(GROUPS)}, got {group!r}"
            )

        # Parsed before the subtasks because `_parse_subtasks` behaves
        # differently for a declared chain: the non-overlap rule it enforces is
        # about concurrent fan-out, and a chain has no concurrency.
        category = d.get("category", CATEGORY_ANALYSIS)
        if category not in CATEGORIES:
            raise CaseParseError(
                f"{cid}: 'category' must be one of {list(CATEGORIES)}, got {category!r}"
            )

        subtasks = cls._parse_subtasks(
            d.get("subtasks"), case_id=cid, category=category,
        )
        workers = d.get("workers", MIN_WORKERS)
        if isinstance(workers, bool) or not isinstance(workers, int):
            raise CaseParseError(f"{cid}: 'workers' must be an int, got {workers!r}")
        if category == CATEGORY_DEPENDENT:
            # A declared chain has no concurrency limit to declare: step N+1
            # starts only after step N has finished, so the honest value is 1.
            # Accepting 2..4 here would put a number in the run metadata that
            # claims a parallelism the case forbids, and "agent count" is one of
            # the fields the report reads.
            if workers != DEPENDENT_WORKERS:
                raise CaseParseError(
                    f"{cid}: a {CATEGORY_DEPENDENT!r} case runs one teammate at a "
                    f"time, so 'workers' must be {DEPENDENT_WORKERS}, got {workers}"
                )
        elif not MIN_WORKERS <= workers <= MAX_WORKERS:
            raise CaseParseError(
                f"{cid}: 'workers' must be in [{MIN_WORKERS}, {MAX_WORKERS}] per "
                f"contract §5.6, got {workers}"
            )

        merge = d.get("merge", MERGE_WRITE)
        if merge not in MERGE_MODES:
            raise CaseParseError(
                f"{cid}: 'merge' must be one of {list(MERGE_MODES)}, got {merge!r}"
            )
        merge_file = d.get("merge_file", "")
        if not isinstance(merge_file, str):
            raise CaseParseError(f"{cid}: 'merge_file' must be a str, got {merge_file!r}")

        hidden_test = d.get("hidden_test", "")
        if not isinstance(hidden_test, str):
            raise CaseParseError(f"{cid}: 'hidden_test' must be a str, got {hidden_test!r}")
        # A path that escapes the fixtures root would put the judge somewhere
        # the runner will not look, and the case would fail every run as if the
        # model had not done the work.
        if hidden_test and (
            hidden_test.startswith(("/", "\\"))
            or ".." in hidden_test.replace("\\", "/").split("/")
        ):
            raise CaseParseError(
                f"{cid}: 'hidden_test' must be a relative path inside the "
                f"fixtures root, got {hidden_test!r}"
            )

        fixture_single = d.get("fixture_single", "")
        fixture_multi = d.get("fixture_multi", "")
        for label, value in (("fixture_single", fixture_single), ("fixture_multi", fixture_multi)):
            if not isinstance(value, str) or not value:
                raise CaseParseError(f"{cid}: {label} must be a non-empty string")

        if group == CONTROLLED and len(subtasks) < MIN_WORKERS:
            raise CaseParseError(
                f"{cid}: a controlled case needs at least {MIN_WORKERS} declared "
                f"subtasks (there is nothing to fan out otherwise), got {len(subtasks)}"
            )

        # `E2ECase` reads the top-level `checks`; reuse it rather than restating
        # the validation, so this loader cannot drift from the E2E one.
        base = E2ECase.from_dict(d)
        return cls(
            id=base.id,
            task=base.task,
            max_turns=base.max_turns,
            tags=base.tags,
            fixture=base.fixture,
            checks=base.checks,
            checks_mode=base.checks_mode,
            judge=base.judge,
            group=group,
            category=category,
            subtasks=subtasks,
            workers=workers,
            merge=merge,
            merge_file=merge_file,
            fixture_single=fixture_single,
            fixture_multi=fixture_multi,
            hidden_test=hidden_test,
        )

    @staticmethod
    def _parse_subtasks(
        raw: object, *, case_id: str, category: str = CATEGORY_ANALYSIS
    ) -> list[Subtask]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise CaseParseError(f"{case_id}: 'subtasks' must be a list, got {raw!r}")
        subtasks = [Subtask.from_dict(s, case_id=case_id) for s in raw]
        ids = [s.id for s in subtasks]
        if len(set(ids)) != len(ids):
            raise CaseParseError(f"{case_id}: duplicate subtask ids {ids}")

        if category == CATEGORY_DEPENDENT:
            # A chain is serial by construction, so two steps writing one path
            # is the category's whole shape rather than a race. What replaces
            # the overlap rule is the total-order check: without it a case
            # could declare a fan-out, call it a chain, and be run serially --
            # measuring the handoff tax while actually doing parallel work.
            _validate_dependency_chain(subtasks, case_id=case_id)
            return subtasks

        if any(subtask.depends_on for subtask in subtasks):
            raise CaseParseError(
                f"{case_id}: 'depends_on' is only meaningful for a "
                f"{CATEGORY_DEPENDENT!r} case; this one is {category!r}, where "
                "both variants are free to run the subtasks in any order"
            )

        # Non-overlapping writes (contract §8.3). Compared on the normalised
        # relative path so `a/b.py` and `a\b.py` -- the same file on Windows --
        # cannot slip past as two.
        seen: dict[str, str] = {}
        for subtask in subtasks:
            key = subtask.writes.replace("\\", "/")
            if key in seen:
                raise CaseParseError(
                    f"{case_id}: subtasks {seen[key]!r} and {subtask.id!r} both write "
                    f"{subtask.writes!r}; a concurrent fan-out would race on that path "
                    "(contract §5.6 requires non-overlapping files or worktree isolation)"
                )
            seen[key] = subtask.id
        return subtasks


def chain_order(
    subtasks: Sequence[Subtask], *, case_id: str = "<unknown>"
) -> list[Subtask]:
    """The steps of a `dependent` case in the order they must run.

    THE one definition of "chain order": the loader calls it to reject a case
    that is not a total order, and the runner calls it to decide what to drive
    next. Two copies of this walk would eventually disagree about a case that
    both accept, and the disagreement would land as steps running in an order
    the case never declared.

    Every structural violation gets its own message because each one means
    something different to whoever wrote the case:

    - an unknown predecessor is a typo;
    - a step with two predecessors is a join, and a join means the two inputs
      could be produced in either order -- the case is not a chain;
    - a step with two successors is a fork, same objection;
    - a self-dependency is the degenerate fork;
    - two roots means two chains, and which runs first is undefined;
    - a back edge is a cycle, so no step is first.

    The walk is iterative and bounded by the number of steps, so a cycle
    terminates here rather than spinning.
    """
    ids = [s.id for s in subtasks]
    known = set(ids)
    if len(known) != len(ids):
        raise CaseParseError(f"{case_id}: duplicate subtask ids in a dependent chain")

    successors: dict[str, list[str]] = {sid: [] for sid in ids}
    by_id = {s.id: s for s in subtasks}
    for subtask in subtasks:
        if len(subtask.depends_on) > 1:
            raise CaseParseError(
                f"{case_id}: subtask {subtask.id!r} declares "
                f"{len(subtask.depends_on)} predecessors; a chain is a total "
                "order, so every step has exactly one (or, for the first, none)"
            )
        for pred in subtask.depends_on:
            if pred == subtask.id:
                raise CaseParseError(
                    f"{case_id}: subtask {subtask.id!r} depends on itself"
                )
            if pred not in known:
                raise CaseParseError(
                    f"{case_id}: subtask {subtask.id!r} declares unknown "
                    f"predecessor {pred!r}"
                )
            successors[pred].append(subtask.id)

    forked = sorted(sid for sid, succ in successors.items() if len(succ) > 1)
    if forked:
        raise CaseParseError(
            f"{case_id}: {forked!r} have more than one successor; a fork is not "
            "a total order and cannot be driven serially"
        )

    roots = [s.id for s in subtasks if not s.depends_on]
    if len(roots) != 1:
        raise CaseParseError(
            f"{case_id}: a dependent chain needs exactly one first step, found "
            f"{len(roots)} ({roots!r}); without exactly one it is not a total order"
        )

    order: list[Subtask] = []
    seen: set[str] = set()
    cursor: str | None = roots[0]
    while cursor is not None:
        if cursor in seen:
            raise CaseParseError(f"{case_id}: dependency cycle through {cursor!r}")
        seen.add(cursor)
        order.append(by_id[cursor])
        nxt = successors[cursor]
        cursor = nxt[0] if nxt else None

    if len(seen) != len(ids):
        unreachable = sorted(known - seen)
        raise CaseParseError(
            f"{case_id}: subtasks {unreachable!r} are unreachable from the first "
            "step, so the case is not a total order"
        )
    return order


def _validate_dependency_chain(subtasks: Sequence[Subtask], *, case_id: str) -> None:
    """Reject a `dependent` case whose steps are not a total order.

    A thin wrapper over `chain_order` so the loader and the runner share one
    walk. Written as a separate name because the two callers want different
    things from the same computation -- the loader wants a verdict, the runner
    wants the order -- and inlining the call at the loader would make it look
    like the check and the drive were independent rules.
    """
    chain_order(subtasks, case_id=case_id)


def fixture_fingerprint(root: Path) -> dict[str, str]:
    """`relative posix path -> sha256` for every file under `root`.

    Used to prove the two sibling fixtures are the same tree. Digests rather
    than a directory-level hash so the failure report can name *which* file
    differs instead of only that something did.
    """
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def assert_fixtures_identical(case: MultiAgentCase, fixtures_root: Path) -> None:
    """Refuse a controlled case whose two fixture trees are not the same tree.

    The contract's §4.7 rule ("A/B 对照使用同一模型、同一用例、同一 fixture")
    is what this enforces. Two trees that differ by one byte make the two
    variants start from different states, and the resulting speedup and token
    numbers then confound the architecture with the fixture difference -- a
    difference nothing in the report would show.

    Raises CaseParseError rather than warning: a mismatched pair is a data bug
    that silently changes a reported number, which is exactly the class of
    problem the loaders in this package fail loudly on.
    """
    left = fixture_fingerprint(fixtures_root / case.fixture_single)
    right = fixture_fingerprint(fixtures_root / case.fixture_multi)
    if left == right:
        return
    only_left = sorted(set(left) - set(right))
    only_right = sorted(set(right) - set(left))
    differing = sorted(p for p in set(left) & set(right) if left[p] != right[p])
    raise CaseParseError(
        f"{case.id}: fixture_single ({case.fixture_single!r}) and fixture_multi "
        f"({case.fixture_multi!r}) are not the same tree, so the two variants would "
        f"not start from the same state: only in single={only_left}, "
        f"only in multi={only_right}, differing={differing}"
    )


def load_multi_agent_cases(
    path: Path,
    *,
    fixtures_root: Path | None = None,
    check_fixture_identity: bool = True,
) -> list[MultiAgentCase]:
    """Load multi-agent cases from a JSONL file, one per line.

    `fixtures_root` defaults to ``<case file's directory>/fixtures``, matching
    `load_cases`. The containment check is not optional for the same reason it
    is not optional there, and the sibling-identity check is what makes the
    paired design real; `check_fixture_identity=False` exists for tests that
    build cases over a synthetic tree, and every caller in the suite leaves it
    on.
    """
    from longline.eval.types import validate_fixtures

    cases: list[MultiAgentCase] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CaseParseError(f"{path}:{lineno}: bad JSON: {exc}") from exc
        if not isinstance(d, dict):
            raise CaseParseError(f"{path}:{lineno}: expected JSON object, got {type(d).__name__}")
        ctype = d.get("type")
        if ctype != "multi_agent":
            raise CaseParseError(f"{path}:{lineno}: unknown case type {ctype!r}")
        cases.append(MultiAgentCase.from_dict(d))

    root = path.parent / "fixtures" if fixtures_root is None else fixtures_root
    # Both sibling trees get the same containment check every other suite's
    # `fixture` gets. `fixture` itself is unset on these cases, so the shared
    # `validate_fixtures` cannot see them; the probes carry one name each into
    # the shape the validator already knows rather than reimplementing the
    # `..` / absolute-path rejection here.
    for case in cases:
        validate_fixtures(
            [
                E2ECase(id=case.id, task=case.task, fixture=case.fixture_single),
                E2ECase(id=case.id, task=case.task, fixture=case.fixture_multi),
            ],
            root,
        )
    if check_fixture_identity:
        for case in cases:
            assert_fixtures_identical(case, root)
    return cases


def group_of(cases: Iterable[MultiAgentCase], group: str) -> list[MultiAgentCase]:
    """Cases in one group, in file order. Never mixes the two."""
    return [c for c in cases if c.group == group]
