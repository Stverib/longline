"""Generate the paired-benefit corpus: 18 cases, three categories of six.

=== Why a generator and not a JSONL file ===

Every case needs TWO sibling fixture trees that are byte-identical, and a human
keeping 36 trees in sync is a promise rather than a mechanism. Both trees come
out of one `build_fixture` call with the same arguments, so "identical" is a
property of the code path; `assert_fixtures_identical` still checks it, as a
regression on the generator rather than as the thing that makes it true.

=== Why the six analysis cases are not six copies of one prompt ===

`evals/multi_agent.jsonl`'s 18 controlled rows carry one identical `task`
string. Pooling them into a category would make that category's average a fact
about the template rather than about the architecture. Each analysis case here
names a different set of modules, so `TestAnalysisCategory` can assert the task
strings are distinct -- which is the property the folded-in corpus would fail.

=== Determinism ===

No timestamps, no uuid, no reliance on set or dict ordering. Written so that
running the generator twice produces a byte-identical diff-free result, which
`TestGeneratorIsDeterministic` holds it to.

Run it with:

    uv run python evals/tools/gen_pair_cases.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "evals"

CASES_NAME = "multi_agent_benefit.jsonl"
FIXTURES_SUBDIR = "fixtures"
PAIR_SUBDIR = "pair"


@dataclass(frozen=True)
class ModuleSpec:
    """One module in a synthetic fixture repo.

    Two facts rather than one so a summarising subtask has something to be
    *about*: a module with a single sentence could be summarised correctly by
    echoing it, and the case would stop measuring whether the agent read the
    module at all.
    """

    name: str          # module stem; becomes modules/<name>.py
    summary: str       # the fact a summarising subtask must recover
    policy: str        # a second fact, so the summary has to choose


@dataclass(frozen=True)
class AnalysisSpec:
    """One `parallel_analysis` case: N independent modules and a merge."""

    case_id: str
    repo_name: str
    modules: tuple[ModuleSpec, ...]

    @property
    def task(self) -> str:
        names = ", ".join(m.name for m in self.modules)
        return (
            f"Four independent documentation subtasks over the {self.repo_name} "
            f"repository ({names}). Each subtask reads one module and writes one "
            "file; no subtask depends on another's output."
        )


@dataclass(frozen=True)
class GeneratedCorpus:
    """Where the generator put its two artifacts."""

    cases_file: Path
    fixtures_dir: Path


# --- fixture construction ----------------------------------------------------

MODULE_TEMPLATE = '''"""The {name} module.

Policy: {policy}
"""


def {name}_describe() -> str:
    """Return the fact this module is responsible for."""
    return {summary!r}
'''

ROLLOUT_TEMPLATE = """# Rollout checklist

Four sign-off sections, in order:

1. design reviewed
2. tests green
3. rollback rehearsed
4. owner assigned
"""

README_TEMPLATE = """# {repo}

A small service with {count} modules under `modules/`.

| module | file |
|---|---|
{rows}
"""


def build_fixture(root: Path, spec: AnalysisSpec) -> None:
    """Write one fixture tree.

    Called twice per case with the same arguments -- once for the `_single`
    tree and once for the `_multi` tree -- so the two are identical by
    construction. Anything that varied between the calls (a case id, a path, an
    index) would have to come from outside this function, which is why nothing
    does.
    """
    modules_dir = root / "modules"
    notes_dir = root / "notes"
    modules_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    rows = "\n".join(
        f"| `{m.name}` | `modules/{m.name}.py` |" for m in spec.modules
    )
    (root / "README.md").write_text(
        README_TEMPLATE.format(
            repo=spec.repo_name, count=len(spec.modules), rows=rows,
        ),
        encoding="utf-8",
    )
    (notes_dir / "rollout.md").write_text(ROLLOUT_TEMPLATE, encoding="utf-8")
    for module in spec.modules:
        (modules_dir / f"{module.name}.py").write_text(
            MODULE_TEMPLATE.format(
                name=module.name, policy=module.policy, summary=module.summary,
            ),
            encoding="utf-8",
        )


# --- the six analysis cases --------------------------------------------------

ANALYSIS_SPECS: tuple[AnalysisSpec, ...] = (
    AnalysisSpec(
        case_id="pa-001",
        repo_name="runtime-core",
        modules=(
            ModuleSpec(
                name="tool",
                summary="Tools are registered per profile and dispatched by name.",
                policy="An unknown tool name fails the turn rather than degrading.",
            ),
            ModuleSpec(
                name="memory",
                summary="Memory entries are keyed by scope and expire on read.",
                policy="Eviction is least-recently-written, never least-recently-read.",
            ),
            ModuleSpec(
                name="permission",
                summary="Permission decisions are deny, ask or allow, in that order.",
                policy="A denial is final; a later rule cannot re-allow the call.",
            ),
            ModuleSpec(
                name="checkpoint",
                summary="Checkpoints are written once per user instruction.",
                policy="Nothing is persisted inside a single instruction.",
            ),
        ),
    ),
    AnalysisSpec(
        case_id="pa-002",
        repo_name="context-pipeline",
        modules=(
            ModuleSpec(
                name="compaction",
                summary="Compaction replaces a prefix of the transcript.",
                policy="Compaction only runs when the token budget is exceeded.",
            ),
            ModuleSpec(
                name="session",
                summary="Sessions are addressed by id and stored under the config dir.",
                policy="Session writes are atomic; a partial session is never visible.",
            ),
            ModuleSpec(
                name="prompt",
                summary="The system prompt is assembled from ordered sections.",
                policy="Section order is fixed; a variant may omit a section only.",
            ),
            ModuleSpec(
                name="model",
                summary="Models are addressed by id and resolved at call time.",
                policy="A missing model id fails loudly instead of falling back.",
            ),
        ),
    ),
    AnalysisSpec(
        case_id="pa-003",
        repo_name="extension-surface",
        modules=(
            ModuleSpec(
                name="hook",
                summary="Hooks run before and after a tool executes.",
                policy="A pre-hook may block; a post-hook may only observe.",
            ),
            ModuleSpec(
                name="skill",
                summary="Skills are markdown files loaded on demand by name.",
                policy="A skill that fails to load is skipped, not fatal.",
            ),
            ModuleSpec(
                name="client",
                summary="MCP clients are started lazily on first tool use.",
                policy="A server that fails to connect is retried once, then disabled.",
            ),
            ModuleSpec(
                name="theme",
                summary="Themes resolve to a palette of named colours.",
                policy="An unknown colour name falls back to the default palette.",
            ),
        ),
    ),
    AnalysisSpec(
        case_id="pa-004",
        repo_name="search-tools",
        modules=(
            ModuleSpec(
                name="grep",
                summary="Grep returns matching lines with their line numbers.",
                policy="Binary files are skipped rather than decoded.",
            ),
            ModuleSpec(
                name="glob",
                summary="Glob expands a pattern against paths, not contents.",
                policy="Results are sorted so two runs agree.",
            ),
            ModuleSpec(
                name="shell",
                summary="Shell commands run without a shell, as an argv list.",
                policy="The working directory is fixed for the whole invocation.",
            ),
            ModuleSpec(
                name="patch",
                summary="Patch replaces one exact substring in a file.",
                policy="A substring matching more than once is rejected, not guessed.",
            ),
        ),
    ),
    AnalysisSpec(
        case_id="pa-005",
        repo_name="coordination",
        modules=(
            ModuleSpec(
                name="mailbox",
                summary="Each agent owns one inbox file holding a message list.",
                policy="Reading an inbox does not consume it; marking does.",
            ),
            ModuleSpec(
                name="spawn",
                summary="Spawning registers an identity, a team row and a task.",
                policy="A teammate runs as a task on the caller's event loop.",
            ),
            ModuleSpec(
                name="identity",
                summary="Agent ids are name-at-team, and the lead is reserved.",
                policy="An id that collides with the lead is rejected.",
            ),
            ModuleSpec(
                name="roster",
                summary="The team file lists members with a joined-at time.",
                policy="Membership removal is soft; the row stays with a flag.",
            ),
        ),
    ),
    AnalysisSpec(
        case_id="pa-006",
        repo_name="measurement",
        modules=(
            ModuleSpec(
                name="latency",
                summary="Latency cases declare a schedule of tool durations.",
                policy="A saturating host invalidates a latency measurement.",
            ),
            ModuleSpec(
                name="report",
                summary="Reports aggregate results and never pool separate groups.",
                policy="An excluded case stays in the data with a reason.",
            ),
            ModuleSpec(
                name="judge",
                summary="Judges are deterministic checks over a sandbox.",
                policy="A judge that raises counts as failed, not as an abort.",
            ),
            ModuleSpec(
                name="metric",
                summary="Ratios are ratios; percentage points are a different unit.",
                policy="A ratio with a zero denominator is reported as not measured.",
            ),
        ),
    ),
)


# --- case emission -----------------------------------------------------------


def _analysis_case(spec: AnalysisSpec) -> dict[str, Any]:
    """One analysis case in the loader's on-disk shape."""
    out_dir = f"out/{spec.case_id}"
    subtasks = [
        {
            "id": f"s{index}",
            "instruction": (
                f"Summarise the {module.name} module's responsibility and its "
                f"stated policy, in modules/{module.name}.py"
            ),
            "writes": f"{out_dir}/{module.name}.md",
        }
        for index, module in enumerate(spec.modules, start=1)
    ]
    merge_file = f"{out_dir}/manifest.txt"
    expected = [s["writes"] for s in subtasks] + [merge_file]
    return {
        "type": "multi_agent",
        "id": spec.case_id,
        "task": spec.task,
        "group": "controlled",
        "category": "parallel_analysis",
        "workers": len(subtasks),
        "subtasks": subtasks,
        "merge": "write_required",
        "merge_file": merge_file,
        "fixture_single": f"pair/{spec.case_id}_single",
        "fixture_multi": f"pair/{spec.case_id}_multi",
        "max_turns": 12,
        "tags": ["pair", "parallel_analysis", "file-ops"],
        "checks": [
            *[
                {"fn": "file_exists", "args": {"path": s["writes"]}}
                for s in subtasks
            ],
            {"fn": "line_set_equals", "args": {
                "path": merge_file, "equals": [s["writes"] for s in subtasks],
                "duplicates_allowed": False,
            }},
            {"fn": "directory_snapshot", "args": {
                "path": out_dir,
                "files_exact": False,
                "files_contains": [
                    Path(s["writes"]).name for s in subtasks
                ] + [Path(merge_file).name],
                "min_sizes": {Path(s["writes"]).name: 40 for s in subtasks},
            }},
            {"fn": "unexpected_paths", "args": {
                "equals": expected, "roots": [out_dir],
            }},
        ],
    }


def _cases() -> list[dict[str, Any]]:
    """Every case in the corpus, in a fixed order."""
    return [_analysis_case(spec) for spec in ANALYSIS_SPECS]


def generate(out_root: Path = DEFAULT_OUT) -> GeneratedCorpus:
    """Write the corpus under `out_root` and return where it landed.

    The cases file sits directly in `out_root` with its fixtures in
    `<out_root>/fixtures`, which is the layout `load_multi_agent_cases` defaults
    to when no `fixtures_root` is passed. Generating into a different shape
    would make every caller state the root explicitly and would break the day
    one of them forgot.
    """
    fixtures = out_root / FIXTURES_SUBDIR
    for spec in ANALYSIS_SPECS:
        for side in ("single", "multi"):
            build_fixture(
                fixtures / PAIR_SUBDIR / f"{spec.case_id}_{side}", spec,
            )

    cases_file = out_root / CASES_NAME
    lines = [json.dumps(case, ensure_ascii=False) for case in _cases()]
    cases_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return GeneratedCorpus(cases_file=cases_file, fixtures_dir=fixtures)


def main() -> int:
    corpus = generate()
    print(f"cases    -> {corpus.cases_file}")
    print(f"fixtures -> {corpus.fixtures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
