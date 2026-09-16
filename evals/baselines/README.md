# Baselines：冻结基线与回归门槛

> 指标契约：`evals/README.md`（本目录的口径以它为准）

本目录存放**冻结基线**。基线的作用是给回归判定一个固定参照物，
不是「最好的成绩单」。

---

## 1. 当前状态

**本目录暂无基线文件。**

原因：仓库现存的唯一原始结果 `evals/results/claude-sonnet-4-20250514-1.json`
是 **1 条用例的冒烟运行**（`l1_tool_accuracy: null`），**不构成 full-suite baseline**。
正式基线将在 Task 9 完成 3 次全量运行后写入本目录。

在此之前，任何「回归」判定都**无参照物**，只能报告绝对值。

---

## 2. 文件格式：`<model>.json`

一个基线文件对应**一个模型 + 一套配置**。命名规则：`<model>.json`，
例如 `claude-sonnet-4-20250514.json`、`deepseek-v4-flash.json`。

文件必须包含**完整运行元数据**（计划 §3.1），否则不构成有效基线：

> **以下仅为字段格式示例，所有数值都是占位符，不是任何真实运行的结果。**
> 本仓库当前**尚无**基线文件（见 §1）。禁止复制示例中的 `metrics` 数值当作实测数据。

```json
{
  "run_id": "2026-09-20T10-00-00_claude-sonnet-4-20250514_e2e",
  "suite": "e2e",
  "variant": "baseline",
  "model": "<model-id>",
  "git_sha": "<40-hex>",
  "started_at": "2026-09-20T10:00:00+08:00",
  "platform": "<platform>",
  "python_version": "<py-version>",
  "case_file_sha256": "<64-hex>",
  "repeat_index": 0,
  "repeats_completed": 3,

  "metrics": {
    "task_success_rate": {
      "numerator": "<int>",
      "denominator": "<int>",
      "value": "<float>",
      "ci95_wilson": ["<lo>", "<hi>"]
    },
    "by_category": {
      "<category>": {"numerator": "<int>", "denominator": "<int>", "value": "<float>", "ci95_wilson": ["<lo>", "<hi>"]}
    }
  },
  "raw_results": [
    "evals/results/<run_id-1>/raw.jsonl",
    "evals/results/<run_id-2>/raw.jsonl",
    "evals/results/<run_id-3>/raw.jsonl"
  ],
  "failure_attribution": {
    "model": "<int>", "runtime": "<int>", "tool": "<int>",
    "judge": "<int>", "fixture": "<int>", "infra": "<int>"
  }
}
```

### 2.1 字段要求

| 字段 | 必需 | 说明 |
|---|---|---|
| `run_id` | 是 | 与 `evals/results/<run_id>/` 对应 |
| `suite` / `variant` | 是 | 标明是哪个套件的哪个变体 |
| `model` | 是 | 基线**只对这一个模型**有效 |
| `git_sha` | 是 | 被冻结的提交，必须可回溯 |
| `started_at` / `platform` / `python_version` | 是 | 环境可复现 |
| `case_file_sha256` | 是 | 用例集指纹，用例改动即基线失效 |
| `repeat_index` / `repeats_completed` | 是 | 正式基线要求 `repeats_completed >= 3` |
| `metrics` | 是 | 每个比例**必须**含 numerator / denominator / value / 95% Wilson CI |
| `raw_results` | 是 | 指向 `raw.jsonl`；`report.md` **不可**作为数据源 |
| `failure_attribution` | 是 | 失败样本分类：model / runtime / tool / judge / fixture / infra |

**红线**：`report.md` **永远不能**作为基线的唯一数据来源。
基线的每个数字都必须能从 `raw_results` 里的 `raw.jsonl` 独立重算。

### 2.2 有效性范围

- 一个基线**只对一个模型 + 一套配置有效**。换模型、换 fixture、换 `case_file_sha256`、
  换 `variant`，该基线**立即失效**，必须重新冻结。
- 基线**不跨套件复用**：E2E 基线不能拿来判定 Tool Calling 的回归。
- 成对实验（Compression / Streaming / Multi-Agent）**不留绝对基线**，
  而是同一任务内 `off/on`、`buffered/streaming`、`single/multi` 成对比较（见 `evals/README.md` §4.0）。

---

## 3. 冻结流程

1. **离线全绿**：`pytest`、`ruff check`、`mypy` 全部通过后，才允许运行付费评测。
2. **跑满 3 次全量**：`repeats_completed >= 3`。报告为 3 次 Pass@1 的**均值和区间**，
   **绝不取最好一次**。
3. **分类失败样本**：逐条归因为 model / runtime / tool / judge / fixture / infra。
4. **产出 raw 产物**：确认每个 run 的 `evals/results/<run_id>/raw.jsonl` 已落盘且完整。
5. **写基线文件**：按 §2 格式写入 `<model>.json`，元数据字段一个都不能少。
6. **PR 内确认门槛**：提交时写明当前门槛（见 §4），并在 PR 描述里引用 baseline 的 `git_sha` 与 `run_id`。

---

## 4. 回归门槛

### 4.1 起步阶段：宽松

回归门槛**先设为**

> **不得超过基线 95% Wilson CI 的合理波动。**

即：新结果落回基线 CI 区间内 → **不算回归**。
原因：样本量有限（E2E 40 条、Recovery 60 次），点估计本身有波动，
此时定死硬阈值会把噪声判成回归。

### 4.2 成熟阶段：硬化

**积累 3 个版本**之后，CI 波动范围已经稳定，再把门槛**固定为硬阈值**
（例如「Task Success Rate 不得低于基线 X pp」，具体 `X` 由届时的 CI 宽度决定，
不预先写死）。
硬化时必须**同时更新**本文件的 §4.1 描述和 baseline 文件，避免两处口径冲突。

### 4.3 CI 与运行频率

- **PR**：只跑 **smoke**（少量用例），不跑全量。
- **全量套件**：**手动**运行或**定时**（schedule）运行。
- 原因：全量套件消耗真实 API 额度，且质量评测要求**串行执行**以避免限流。

---

## 相关文档

- 指标契约（公式 / 分母 / 排除条件 / 统计口径）：`evals/README.md`
