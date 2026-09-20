# Agent Runtime 评测：指标契约与数据说明

> 冻结日期：2026-09-15

本文件是**指标契约（metric contract）**：所有正式数字的公式、分子、分母、排除条件和命名，
以本文件为唯一准绳。任何报告、README 或简历数字若与本文件冲突，以本文件为准。

**本文件是自包含的**：它记录的是最终口径，不依赖任何外部文档就能读通、能据以实现。
本项目另有一份内部的设计计划（当初据此实现，含排期与任务分解），但该文档属于
不入库的内部材料，**克隆本仓库的人拿不到它，也不需要它** —— 需要的口径都已经写在这里。

原先的分工说明「计划里写的是 `cc/eval/`，真实包名是 `longline/`」，只在读那份计划时才有意义，
与本文件的读者无关，故删除。

---

## 0. 本目录清单

| 路径 | 状态 | 说明 |
|---|---|---|
| `evals/tool_calls.jsonl` | **legacy** | 30 条工具调用用例。instruction-following 数据，**不进入简历主数字**。 |
| `evals/tool_selection.jsonl` | **生效** | 68 条工具调用用例（56 盲测 + 12 instruction-following）。盲测 = 48 工具族 + 8 弃权。 |
| `evals/e2e.jsonl` | **生效** | 40 条端到端用例，5 类 × 8，Task 3 产物。 |
| `evals/compression.jsonl` | **生效** | 20 条长上下文用例，Task 4 产物。 |
| `evals/recovery.jsonl` | **生效** | 6 条故障定义 × `repeat: 10` = **60 次运行**，Task 5 产物。 |
| `evals/multi_agent.jsonl` | **生效** | 24 条单 Agent vs 多 Agent 用例（18 controlled + 6 exploratory），Task 7 产物。 |
| `evals/fixtures/` | 生效 | 沙箱起始状态。 |
| `evals/results/` | 生效 | 运行产物，布局见 §3。 |
| `evals/baselines/` | 生效 | 冻结基线，格式见 `evals/baselines/README.md`。 |

### 关于 `tool_selection.jsonl`（Task 2 新增）

主评测集。**68 条 = 56 条盲测 + 12 条 instruction-following**。盲测由两部分构成：
**48 条工具族用例 + 8 条弃权用例**（见下）。工具族拆分按 60 条计：

| 工具族 | 数量 |
|---|---:|
| Read / Write / Edit | 16 |
| Glob / Grep | 12 |
| Bash | 6 |
| WebSearch / WebFetch | 8 |
| NotebookEdit | 4 |
| TaskCreate / TaskGet / TaskList / TaskUpdate / TaskStop | 6 |
| 多工具路径 | 8 |

- 每条用例带 `tags: ["blind"]` 或 `["instruction-following"]`，**二选一，不重叠**。
- 每条盲测用例必须带 `blind_rationale`，一句话说明「任务文本为什么没有点名或暗示工具」。
- 族数量按**60 条工具用例**统计（instruction-following 也带族标签），
  因此每族都同时贡献盲测样本和回归样本，而不是在 60 条之外再加 12 条。
  **弃权用例不带族标签**，故不进入上表。
- fixture 分配：`simple_repo`（**沿用旧集，文件列表冻结不可改**）、
  `tool_repo`（Task 2 专用，带 assets/、utils.py、多变量 version.py）、
  `notebook_repo`、`workspace_repo`。

**弃权类别（`tags: ["abstention"]`，8 条，2026-09-16 新增）**：

「**正确答案是不调用任何工具**」的用例。BFCL 把 4441 条中的 **1122 条（约 25%）**给了这一类
（240 Irrelevance + 882 Live Irrelevance），MetaTool 的头号指标也是含 no-tool 的选择觉知。
**一个每题都要求调用工具的套件会奖励「见活就调工具」的 agent** —— 这种失败模式在旧套件里
完全不可见，且让 ToolSelectionCaseAccuracy **系统性偏高**。

- 出题规则：**答案必须能从提示词本身得到**。评测 profile 里的工具都作用于沙箱文件系统或网络，
  因此凡是「针对消息里已给出的文本」的提问，去调工具本身就是荒谬的（得先把文本写进文件再读回来）。
  这让弃权成为**正确行为**，而不是人为禁止。
- 判分：弃权用例的 `accepted_tool_steps` 为空，因此 `all_steps_matched` 会因 `all([])` 恒真而
  **空过**。真正的通过条件是 `CaseResult.abstention_ok`：**一个工具都没调，且没有多余调用**。
  聚合层对弃权用例用这个条件，普通用例仍用 `steps_completed`。


**泄漏检查是可执行的**，见 `tests/unit/eval/leakage.py` 与
`tests/unit/eval/test_tool_selection_cases.py`。重要限制：英文工具名用词边界扫描（精确），
中文提示词表（`TOOL_HINT_WORDS`）**只是启发式**，覆盖不了新奇的改写。
`blind_rationale` 才是人工复核的依据；**不要**把绿色测试当成「已证明不泄漏」。


**关于 `tool_calls.jsonl`**：已通过逐行 `tags` 数组追加 `"legacy"` 标记（`tags` 本就是
`longline/eval/types.py` 支持的字段，`from_dict` 对未知顶层键也会忽略，因此加载器行为不变）。
标记后经 `load_cases()` 实测仍返回 **30 / 10** 条。计划 §5 Task 0 只授权标记这一个文件，
因此 `e2e.jsonl` **保持原样**。

**关于 `e2e.jsonl`**：这 10 条当前**不算 legacy**——它们没有工具名泄漏
（全库扫描：`e2e.jsonl` 0 条，`tool_calls.jsonl` 10 条），任务文本只描述目标不指定手段，
judge 也都检查最终产物。它们的缺陷是**样本量与覆盖面**（10 条不足以支撑 ±3% 的结论，
且偏简单文件创建/修改），而这由计划 §5 Task 3 **扩充到 40 条**来解决，不是靠打 legacy 标记。
Task 3 落地时，这 10 条应作为**正式用例**保留并归入对应类别；
若届时判定其中某条不适用，再单独剔除。

### 为什么 legacy 数据不能当主数字

来自计划 §2「当前缺口」，逐条对应：

- `l1_tool_accuracy` 把「工具选择」和「参数正确」合成了一个布尔值，缺少独立分母。
- 轨迹没有采集 `ToolResultReady`，无法计算工具执行成功率。
- 30 条用例只覆盖 6 个工具（Read 17 / Grep 6 / Write 5 / Edit 5 / Glob 4 / Bash 4）。
- 多条任务文本直接写了「用 grep」「用 glob」等工具名，存在**标签泄漏**——
  测的是指令跟随，不是工具选择能力。
- 判分允许额外调用：只要期望工具作为**子序列**出现就算通过，会掩盖无效调用。
- E2E 只有 10 条，且主要是简单文件创建/修改，没有覆盖检索、多工具和长链路任务。
- 没有统一记录 wall-clock、事件时间点、模型配置、Git SHA、运行环境和重复次数。
- 没有成对 A/B 运行能力，无法严谨评估压缩、流式执行和 Multi-Agent 的收益。

---

## 1. 非目标（Non-goals）

明确**不做**以下事情（计划 §1）：

- 不使用 LLM-as-a-Judge 作为主判分器；主指标全部使用确定性判分。
- 不把联网结果直接放进稳定回归集；Web 工具在主评测中使用固定响应的替身。
- 不把多次尝试后的最好结果写成 Pass@1。
- 不声称任意 Bash 副作用可以 exactly-once 恢复；当前实现尚无事务日志或幂等键。

**关于 Resume 的表述红线**：当前版本保证的是**会话加载、transcript 修复和 Task 快照恢复**。
非终态后台任务在恢复后会被标记为 `KILLED`，因此**不得**表述为「后台任务原地续跑」。

---

## 2. 运行元数据与用例字段

### 2.1 每个 run 必录字段（计划 §3.1）

```text
run_id
suite
variant                 # baseline / candidate，或 compression_off / compression_on
model
git_sha
started_at
platform
python_version
case_file_sha256
repeat_index
```

### 2.2 每条用例必录字段（计划 §3.1）

```text
case_id
case_type
tags
passed
duration_ms
num_rounds
num_tool_calls
num_successful_tool_calls
input_tokens
output_tokens
error_type
tool_calls
event_timestamps
judge_detail
```

---

## 3. 输出目录布局

```text
evals/results/<run_id>/raw.jsonl
evals/results/<run_id>/summary.json
evals/results/<run_id>/report.md
```

- `raw.jsonl` 保存**逐次实验事实**，是唯一事实来源。
- `summary.json` 保存**聚合结果**。
- `report.md` **只负责展示**，永远不能成为唯一数据源。

**规则**：任何一个指标都必须能从 `raw.jsonl` 独立重算。如果某个数字只存在于
`report.md` 里，它就是无效数字。

---

## 4. 统计口径（计划 §3.2）

这些规则适用于本目录下的**所有**指标，下文不再重复：

1. **比例指标**必须同时输出 numerator、denominator、value 和 **95% Wilson 置信区间**。
2. **延迟指标**输出 mean、p50、p95；A/B 时同时输出**每条用例的 paired delta**。
3. **成功率差使用百分点**。例如 `84% - 87% = -3 pp`，**不能**写成「下降 3%」。
4. **Token** 输出输入、输出和总量；**Multi-Agent 必须包含所有子 Agent**，不能只统计 leader。
5. **Pass@1** 指每个任务的一次**独立首次执行**。若全套运行 3 次，报告为 **3 次 Pass@1 的均值和区间**，
   **绝不取最好一次**。

   > **命名红线（2026-09-16 澄清）**：上面的「3 次均值」是**用 3 倍样本量估出的 Pass@1**，
   > **不是 Pass@3**。两者数值接近但**不是同一个量**，绝不能拿去与任何榜单的 `pass@k` 直接比较。
   > 报告里写 `E2E pass@1`，不要写 `pass@3`。

5b. **Pass^k（可靠性）** —— 与 Pass@1 并列报告，二者回答不同问题：

   - `Pass@1` = 平均成功率（准确度）。
   - `Pass^k` = **每个任务 k 次运行全部通过**的比例（可靠性）。

   采用 τ-bench（Yao et al., arXiv:2406.12045）的估计量：

   ```text
   pass@k := 1 - E_task[ C(n - c, k) / C(n, k) ]
   pass^k :=     E_task[ C(c,     k) / C(n, k) ]
   ```

   其中 `n` 为该任务的运行次数，`c` 为其中成功次数。**同一二项式形式，尾巴方向相反**：
   `pass@k` 奖励偶尔蒙对（随 k 上升），`pass^k` 惩罚任何不稳定（随 k 下降）。
   `n == k` 时 `pass^k` 退化为「k 次全过的任务占比」。

   **为什么必须两个都报**：3 次结果为 `1/1/1` 与 `1/0/1` 的用例，Pass@1 平均值都是 0.667，
   而 Pass^3 分别是 1.0 和 0.0。只报均值等于把「稳定」和「碰运气」混为一谈。
   τ-bench 的核心发现正是这个差距：gpt-4o 在 τ-retail 上 `pass^1 = 61.2%`，但 `pass^8 < 25%`。

5c. **按工具调用次数分层** —— 长链路类别必须**分别**报告最终结果与调用次数（计划 §4.1）。
   调用次数**只作诊断，绝不参与成败判定**。报告按 `0-2 / 3-5 / 6-9 / 10+` 次调用分桶给出通过率，
   空桶省略而不显示为 0%（否则「没有用例」会被误读成「0% 成功」）。
   业界对多步信用分配的标准做法是**按类别分层**（OSWorld/WebArena 按领域、GAIA 按难度），
   而不是逐步打分，本报告遵循前者。

6. **质量评测串行执行**，避免 API 限流互相影响；延迟和并行评测单独运行。
7. **A/B 对照**使用同一模型、同一用例、同一 fixture 和同一机器；执行顺序按 case 交替，降低冷启动偏差。
8. **正式数字至少完成 3 次全量运行**；原始结果和失败样本必须保留。

### 4.0 Baseline 规则（计划 §3.3）

并非所有指标都需要人为制造一个弱基线：

- **Task Success / Tool Accuracy / Recovery**：报告绝对值，并与固定 Git SHA 的上一版结果比较。
- **Context Compression**：同一任务 `compression_off` 与 `compression_on` 成对比较。
- **Streaming Latency**：同一事件流 `buffered` 与 `streaming` 成对比较。
- **Multi-Agent**：同一可并行任务 `single_agent` 与 `multi_agent` 成对比较。
- **Safety**：与期望决策标签比较，同时报告危险操作召回率和正常操作误拦截率。

---

## 5. 指标定义

### 5.1 Task Success Rate（计划 §4.1）

正式评测集 **40 条，5 类 × 8 条**：

| 类别 | 数量 | 主要判分方式 |
|---|---:|---|
| 文件操作 | 8 | 文件存在、内容、目录状态 |
| 代码任务 | 8 | 测试命令、函数输出、静态内容 |
| 检索任务 | 8 | 固定本地语料或固定 Web 响应中的答案 |
| 多工具任务 | 8 | 最终产物 + 关键中间状态 |
| 长链路任务 | 8 | 最终产物；≥5 次工具调用只作诊断，不代替结果判分 |

```text
TaskSuccessRate = passed_e2e_cases / total_e2e_cases
```

- **分子**：通过最终状态判定的 E2E 用例数。
- **分母**：E2E 用例总数（正式集 = 40）。
- **排除条件**：无。baseline 本身失败的用例仍计入分母。

**按类别报告**：每类的成功率、平均轮数、平均 Tool Calls、平均 Token、平均时长和失败类型。

**红线**：E2E 成功**只由最终状态判定**，不能因为「调用了正确工具」就算成功。

判分器要求：

- 一个用例可包含多个 `checks`，**默认全部通过才算成功**。
- 新增 `json_value`、`line_set_equals`、`python_test`、`directory_snapshot` 判分。
- 命令判分**使用参数列表并禁用 `shell=True`**；只允许评测文件中声明的受控命令。
- fixture 路径解析后必须**仍位于 `evals/fixtures/` 下**。

**验收条件**：40 条均可在**全新沙箱**独立运行；人工复核每条 judge 不存在恒真、弱断言或只检查局部结果的问题。

### 5.2 Tool Calling Accuracy（计划 §4.2）

正式评测集 **60 条**，覆盖以下工具族：

| 工具族 | 数量 |
|---|---:|
| Read / Write / Edit | 16 |
| Glob / Grep | 12 |
| Bash | 6 |
| WebSearch / WebFetch | 8 |
| NotebookEdit | 4 |
| TaskCreate / TaskGet / TaskList / TaskUpdate / TaskStop | 6 |
| 多工具路径 | 8 |

其中 **48 条是盲测集**（任务文本不出现工具名）；**12 条**保留显式指定工具的场景，
**只作为 instruction-following 回归指标，不进入简历主数字**。

允许一个决策点存在多个合理工具（查找文件可接受 `Glob` 或 `Grep`）。用例结构从单一
`expect_tools` 扩展为按步骤的候选集合：

```json
{
  "accepted_tool_steps": [["Glob", "Grep"], ["Read"]],
  "max_extra_calls": 1,
  "expect_args": {"Read": {"file_path": ".*config\\.py"}}
}
```

四个指标，**分母彼此独立**，必须完全拆开报：

```text
ToolSelectionCaseAccuracy = 完成全部期望决策步骤的用例数 / 工具盲测用例数
ToolCallPrecision          = 匹配有效步骤的调用数 / 全部工具调用数
ArgumentCallAccuracy       = 参数整体正确的匹配调用数 / 需要校验参数的匹配调用数
ArgumentFieldAccuracy      = 正确参数字段数 / 被检查参数字段数
ExecutionSuccessRate       = is_error=false 的实际执行数 / 实际执行数
```

| 指标 | 分子 | 分母 | 排除条件 |
|---|---|---|---|
| ToolSelectionCaseAccuracy | 完成**全部**期望决策步骤的用例数 | 工具**盲测**用例数（=48） | 排除 12 条 instruction-following 用例 |
| ToolCallPrecision | 匹配有效步骤的调用数 | **全部**工具调用数（含额外调用） | 无；额外调用必须计入分母 |
| ArgumentCallAccuracy | 参数整体正确的匹配调用数 | **需要校验参数**的匹配调用数 | 不校验参数的调用不计入分母 |
| ArgumentFieldAccuracy | 正确参数字段数 | **被检查**参数字段数 | 未声明的字段不计入分母 |
| ExecutionSuccessRate | `is_error=false` 的实际执行数 | **实际执行数**（已下发到工具的次数） | 未执行的调用不计入分母 |

**实现口径补充**（与上述公式等价，消除歧义）：

- **分步骤匹配**：每个决策步骤消费**最早**一个尚未被消费、且工具名在该步候选集合里的调用。
  没匹配上的步骤**不推进游标**，因此不会连累后面的步骤；
  而游标一旦推进就不再回退，顺序因此仍然有意义。
- **参数分母**：只统计 `expect_args` 中**声明了的工具**且有实际调用的那些。
  声明了但没调用 = 步骤层的缺失，不在参数分母里重复扣分。
- **同名工具的多次调用取最好的一次**：一次参数写错后重写正确，算「最终传对了」。
- **未测到即 `null`**：任何分母为 0 的比例输出 `null`（不是 `0.0`），
  与 `longline/eval/metrics.py::Ratio` 的约定一致。


**关键规则**：额外调用**不再完全免费**。它不一定让任务级选择失败，但**必须**降低 `ToolCallPrecision`。

WebSearch / WebFetch 使用与生产工具**相同 schema 的离线固定响应工具**，避免网络波动污染工具选择指标。

### 5.3 Context Compression（计划 §4.3）

准备 **20 个长上下文用例**。每个用例包含 8～12 轮历史、**5 个必须保留的事实**，以及一个压缩后要继续完成的任务。
事实覆盖：精确文件路径 / 函数·类名 / 已确认的设计决策 / 已知错误原因 / 未完成步骤或约束。

每条任务运行两个 variant：

```text
baseline:  原始完整 history，禁用压缩
candidate: 对同一 history 执行 compact，再继续任务
```

```text
CompressionRatio           = 1 - estimated_tokens_after / estimated_tokens_before
KeyInfoRetention           = 正确回答或使用的关键事实数 / 关键事实总数
PostCompressionSuccessRate = 压缩后最终任务成功数 / 压缩任务总数
SuccessDeltaPP             = candidate_success_rate - baseline_success_rate
```

| 指标 | 分子 | 分母 | 排除条件 |
|---|---|---|---|
| CompressionRatio | —（比率，非计数比例） | — | 无 |
| KeyInfoRetention | 正确回答或**使用**的关键事实数 | 关键事实总数（每例 5） | 无 |
| PostCompressionSuccessRate | 压缩后最终任务成功数 | 压缩任务总数（计入分母的用例数） | **baseline 本身失败**的用例不进分母 |
| SuccessDeltaPP | `candidate_success_rate − baseline_success_rate`，单位 **pp** | — | 同上 |

**Token 口径**：Token 数使用现有 `estimate_messages_tokens()`，报告中**必须写明是 estimated tokens**。

**判定方式**：关键信息保留率通过**压缩后的追问结果和最终产物共同判定**；不能只在 summary 文本里搜索关键词，
因为「摘要里有」不等于 Agent 真能继续使用。

**排除条件细则**：每个用例的 baseline **必须先可通过**；若 baseline 本身失败，该次**不进入压缩退化率分母**，
但**仍保留在失败报告中**。

### 5.4 Recovery / Resume（计划 §4.4）

把原材料中的 6 类故障拆成两个语义清晰的主指标：

1. **`RuntimeRecoveryRate`**：429、529、Tool Failure、Output Truncate、Context Overflow，共 5 类，每类 10 次。
2. **`SessionResumeRate`**：Process Kill，共 10 次；在已落盘的稳定 checkpoint 处终止并重新启动。

每类故障通过**可注入包装器**触发，不能依赖真实服务偶发报错：

| 故障 | 注入点 | 成功条件 |
|---|---|---|
| 429 | 首次或前两次 model call | 自动重试且最终 judge 通过 |
| 529 | 首次或前两次 model call | 自动重试且最终 judge 通过 |
| Tool Failure | 指定工具首次返回 `is_error=true` | Agent 调整或重试且最终 judge 通过 |
| Output Truncate | 首次返回 `stop_reason=max_tokens` | 续写逻辑生效且最终 judge 通过 |
| Context Overflow | 首次返回 413 / `prompt_too_long` | reactive compact 生效且最终 judge 通过 |
| Process Kill | 已保存的 turn checkpoint 后终止子进程 | checkpoint 可加载、transcript 合法、最终 judge 通过 |

```text
RuntimeRecoveryRate = 五类运行时故障恢复成功数 / 50
SessionResumeRate   = 进程中断后恢复成功数 / 10
```

| 指标 | 分子 | 分母 | 排除条件 |
|---|---|---|---|
| RuntimeRecoveryRate | 5 类运行时故障**恢复成功**数 | **50**（5 类 × 10 次） | 未真正注入故障的次次不计入分子（见下） |
| SessionResumeRate | 进程中断后**恢复成功**数 | **10** | 同上 |

**逐例额外记录**：

```text
fault_injected
retry_count
checkpoint_loaded
transcript_repaired
duplicate_persisted_tool_calls
recovery_latency_ms
```

**「恢复成功」的定义**（三者缺一不可）：

1. 故障**确实注入**（`fault_injected` 为真）；
2. 恢复路径**确实触发**；
3. 最终**确定性 judge 通过**。

Process Kill 还必须**成功加载 checkpoint**，并**通过 transcript 结构校验**。

**`ambiguous_side_effect` 规则**：当前版本只对**已持久化的完整工具结果**检查不重复执行。
对于「工具已经产生外部副作用、但结果尚未落盘」时发生 kill 的场景，标记为 `ambiguous_side_effect`，
**不宣称 exactly-once**。若后续要证明任意中断点无重复副作用，需要另做 durable tool journal + 幂等键，
不塞进本轮指标工程。

#### 实现说明（Task 5，2026-09-16）

**数据集**：`evals/recovery.jsonl` 共 **6 条定义**，每条 `repeat: 10`，由 loader 展开为
**60 次运行**：运行时 5 类 × 10 = **50**，Process Kill **10**。分母因此正好是 `50` 与 `10`。
每条定义只对应一个故障类；为同一类再加一条定义会把运行时分母推离 50，
而契约固定的是 50/10 这两个数。「注入点在前两次 model call」的另一半（第二次调用）
由单元测试覆盖，不进数据集。

**每个注入器都带计数器**，`success` 由 `recovery_succeeded()` 计算而非赋值：

```text
success = fault_injected AND retry_count > 0 AND passed
          AND (fault != process_kill OR (checkpoint_loaded AND transcript 结构合法))
```

`fault_injected` 是**成功表达式的第一项**，这条规则的作用是：**没有真正注入故障的运行
永远不可能被计为恢复成功**——它可能是一次普通成功，但那不是恢复。未注入的运行
**仍留在分母里**并被记入失败清单，不静默丢弃。

**故障不携带答案**：`build_injection_events()` 只返回故障事件（429/529 只有 `ErrorEvent`；
truncate 只有被截断的前半段；overflow 只有 413）。任一注入事件都不含 `answer`，
否则「没恢复也判过」——那正是本任务要防的失败模式。单元测试逐类断言这一点。

**两类注入方言**：429/529/truncate/overflow 由脚本化 model stream 注入；Tool Failure 由
`ToolFaultWrapper` 包住**生产工具**实现（首次调用返回 `is_error=true`，之后透传）；
Process Kill 由真实子进程 kill + 新解释器 resume 实现。恢复路径本身**全部复用**
`query_loop` 的既有实现，注入器不重新实现任何一条。

**Process Kill 的三步证据**：`prepare` 阶段把 transcript 与 Task 快照写入临时 `claude_dir`
并回读校验稳定事实；随后对**活着的子进程**发送 kill；`resume` 在**全新解释器**里调用
生产函数 `load_session()` → `validate_transcript()` → `load_task_snapshot()` →
`TaskRegistry.restore()`，重新读取稳定事实并打印。非终态后台任务恢复后被标记为
**KILLED**，因此**不得**表述为「后台任务原地续跑」。

**`claude_dir` 永远是临时目录**。`get_sessions_dir(None)` 会回退到 `~/.longline`，
所以任何一处漏传参数都会把评测会话写进用户的真实状态目录。
单元测试与集成测试都对真实目录做**前后哈希对比**，证明其逐字节未变。

**`transcript_repaired`**：生产 `validate_transcript()` 原本把「是否修复过」只写进日志。
新增 keyword-only 的 `report=` 出参（`TranscriptRepairReport`），默认 `None`，
所有既有调用方行为不变；恢复用例据此逐例记录该字段。

**验收对照**：`query_loop` 的三条恢复路径各有**独立预算参数**，默认值即原生产常量：

| 参数 | 默认 | 管的路径 |
|---|---:|---|
| `max_retry` | 5 | 429/529 等瞬时错误的重试（指数退避） |
| `max_max_output_recovery` | 3 | `max_tokens` 截断（先提额、后追加续写） |
| `max_reactive_compaction` | 1 | 413 / `prompt_too_long`（响应式压缩） |

三者**必须彼此独立**。若让一个预算去 gate 另一条路径（例如用截断预算去判断要不要压缩），
生产行为就会改变：一旦截断恢复次数用尽，之后的 413 即使与截断毫无关系也会**无法自救而变成终态错误**。
`drive_query_loop(disable_recovery=True)` 只归零重试与截断两个预算；
413 那一条用 `disable_reactive_compaction=True` 单独关闭——**控制变量只关掉被测路径本身**。
`test_faults.TestBudgetsAreIndependent` 用矩阵锁住这条性质：每个故障在其它两个预算归零时
仍能恢复，只有自己的预算归零时才不恢复。

**不得夸大的部分**：`duplicate_persisted_tool_calls` 只检查**已落盘的完整工具结果**
是否被重复执行。落在「工具已产生副作用、结果尚未落盘」窗口内的 kill 归类为
`ambiguous_side_effect`——**既不算重复，也不宣称安全**。本版本**不声称 exactly-once**。

### 5.5 Streaming Tool Latency（计划 §4.5）

使用 **30～50 轮成对 A/B**，事件流和工具耗时**完全相同**：

```text
buffered:  response 完整结束后执行全部工具
streaming: tool_use block 完整解析后立即交给 StreamingToolExecutor
```

时间点使用 `time.perf_counter_ns()` 记录：

```text
request_start
tool_block_complete
tool_execute_start
response_complete
tool_execute_end
turn_complete
```

```text
ToolStartLatency = tool_execute_start - request_start
TurnLatency      = turn_complete - request_start
OverlapTime      = max(0, response_complete - tool_execute_start)
LatencyReduction = (baseline - streaming) / baseline
```

| 指标 | 分子 | 分母 | 排除条件 |
|---|---|---|---|
| ToolStartLatency | —（时长，单位 ms） | — | **预热 5 次不计入统计** |
| TurnLatency | —（时长，单位 ms） | — | 同上 |
| OverlapTime | —（时长，单位 ms） | — | 同上 |
| LatencyReduction | `baseline − streaming` | `baseline` | 同上 |

**报告口径**：输出 mean、p50、p95 和 **paired delta**。

先用可控的延迟工具和脚本化流做**稳定微基准**，再用 10 条真实模型任务做**外部验证**——
两者**分开展示**，不混成一个数字。

### 5.6 Multi-Agent Benefit（计划 §4.6）

只选天然可并行任务，准备 **15～30 条**，例如独立分析 4 个模块、检索 4 个互不依赖主题、分别修改互不重叠文件。

主实验使用**预先声明的独立子任务**，保证 Single 与 Multi 的工作内容一致：

```text
single_agent: 同一 Agent 顺序执行全部子任务
multi_agent:  2～4 个子 Agent 并行执行，再由 leader 汇总
```

另设 **exploratory 组**让 coordinator 自主拆解，但**不把它与受控实验混成一个数字**。

```text
SuccessRate
WallClockTime
InputTokens / OutputTokens / TotalTokens
ToolCalls
Speedup       = single_wall_time / multi_wall_time
TokenOverhead = (multi_tokens - single_tokens) / single_tokens
```

| 指标 | 分子 | 分母 | 排除条件 |
|---|---|---|---|
| SuccessRate | 两种 variant 各自 judge 通过的用例数 | 用例总数 | 两种 variant 产物由**同一 judges** 判定 |
| WallClockTime | —（时长） | — | 无 |
| Speedup | `single_wall_time` | `multi_wall_time` | 无 |
| TokenOverhead | `multi_tokens − single_tokens` | `single_tokens` | 无 |

**红线**：**必须采集所有子 Agent 的 Token 和 Tool Calls，不能只统计 leader。**
Agent 数量固定并写入运行元数据；有文件写入的并行任务必须使用**互不重叠文件**或 **worktree 隔离**。

#### 实现说明（Task 7，2026-09-17）

**数据集**：`evals/multi_agent.jsonl` 共 **24 条**，分两组，**永不合并成一个数字**：

| 组 | 条数 | 说明 |
|---|---:|---|
| `controlled` | 18 | 用例**预先声明** 4 个子任务（各写一个互不重叠的文件），`single` 由 1 个 Agent 顺序做完，`multi` 由 2–4 个 worker 并行做完再由 leader 汇总。两种 variant 做的是**同一份工作**。 |
| `exploratory` | 6 | coordinator 自主拆解，**单独汇报**。与受控数字混在一起等于比较不同的工作。 |

**子任务的 Token 与 Tool Calls 如何被采集**（红线的落点）：

`InProcessTeammate._execute_with_query_loop` 自己迭代一条 `query_loop(...)`，
只留 `TextDelta`、**丢弃 `TurnComplete.usage`**，其事件从不进入调用方的事件流。
所以「只统计 leader」不是漏了一个字段，而是结构性的。

可解之处在于 `query_loop` 的 `call_model` 类型是
`Callable[..., AsyncIterator[QueryEvent]]`——**一个没有返回值的纯异步生成器**，
它是本轮 usage 抵达循环的**唯一**通道。因此包住产出 `call_model` 的工厂，就把累加器
放到了每个 Agent 每一轮的下方；没经过这个包装的 usage，循环本身也不可能看到。
实现见 `longline/eval/child_usage.py`。

**账目完整性是被检验的断言，不是假设**：`reconcile()` 比较「派生的 Agent 数」与
「账本里有 turn 的 Agent 数」，不合就 `raise AccountingError`，而不是返回一个
偏低却看着合理的 `TokenOverhead`。第二个独立见证是 `spawn_teammate` 写入
`TaskRegistry` 的记录——它由派生路径产生，而不是由模型流产生，两者一致比任何
单独一个都可信。**任何一方的分歧都会让该用例被标记
`accounting_incomplete` 并退出比率分母，但保留在数据里**（静默缩小分母是一个看着正确的错数字）。

**两个 fixture 是逐字节相同的兄弟树**（`parallel_repo_single` / `parallel_repo_multi`），
由 loader 用 sha256 校验；不共用一棵树是因为 `_prepare_sandbox` 拷进临时沙箱，
共用会让两条套件的改动互相牵动。**不重叠写入**同样在 loader 期拒绝。

**离线与真实模型**：`model=None` 走离线协议——真实的 `QueryEngine`、真实工具、
真实 `query_loop`、真实 `spawn_teammate`、真实判分器，只有模型传输是脚本化的。
CLI 用 `--offline` 打开它。

### 5.7 Permission / Safety（**可选**，计划 §4.7）

> **状态：可选。** 未完成前，Safety 数字不得出现在正式报告或简历中。

准备 30 条无真实破坏性的决策用例：**15 条危险操作、15 条正常操作**。
危险样本包含工作区外删除、高风险 Bash、敏感文件修改、后台 Agent 高风险操作；
正常样本包含 Read、Grep 和受控的工作区内编辑。

```text
DangerousRecall   = 被 DENY 或 ASK 门控的危险操作数 / 危险操作总数
FalsePositiveRate = 被 DENY 或 ASK 门控的正常操作数 / 正常操作总数
```

| 指标 | 分子 | 分母 | 排除条件 |
|---|---|---|---|
| DangerousRecall | 被 DENY 或 ASK 门控的危险操作数 | 危险操作总数（15） | 无 |
| FalsePositiveRate | 被 DENY 或 ASK 门控的正常操作数 | 正常操作总数（15） | 无 |

判分同时覆盖规则优先级、三种 PermissionMode 和非交互模式。
执行层使用 **sentinel tool** 验证被拒绝的调用确实**没有进入 `execute()`**，**绝不运行真实危险命令**。

---

### 5.8 Loop Resume / Failpoints（2026-09-19 新增）

§5.4 的 `SessionResumeRate` 测的是 **checkpoint 文件往返**：`prepare` 手写一条合成
transcript 落盘，被杀的子进程载入它、然后在 `while True: sleep` 里空转，恢复段读一个值
就退出。**那个实验里没有 model call、没有工具执行、没有 query loop**，因此它测不到
「agent 被打断后继续把活干完」，也测不到「恢复后会不会重复执行有副作用的动作」。

§5.8 补的是这一块：**把进程杀死在真实 agent loop 内部**。

#### 六个故障点

| 故障点 | 杀点 | 谁执行 |
|---|---|---|
| `before_model` | 指令 1 的首次 model call 之前 | 子进程门控 |
| `before_tool` | 指定工具执行之前 | 子进程门控 |
| `after_tool` | 指定工具返回之后、结果落盘之前 | 子进程门控 |
| `after_checkpoint` | **指令 2** 的首次 model call 之前（指令 1 已完成并落盘） | 子进程门控 |
| `truncate_tail` | 会话 JSONL 尾行截断一半 | 父进程 |
| `workspace_drift` | 检查点之后工作区被外部修改 | 父进程 |

子进程侧的四个门控由 `longline/eval/failpoints.py` 实现：到达杀点时**先写哨兵并 fsync，
再永久阻塞**，父进程看到哨兵后真杀进程。哨兵必须 fsync——`SIGKILL` / `TerminateProcess`
不会刷缓冲流，只 `flush()` 的哨兵会让「故障真的注入了吗」变成不可证。

#### 指标

| 指标 | 定义 | 分母 |
|---|---|---|
| `LoopResumeRate` | 四层完整性全过且任务完成 | 五个非检出臂 |
| `DuplicateSideEffectRate` | 恢复后**副作用真的又发生了一次**的操作数 / 故障前已发生的副作用操作数 | 按操作计 |
| `RedundantReExecutionRate` | 恢复后**同一工具同一输入被执行两次**的操作数 / 同一分母 | 同上 |
| `WorkspaceDriftDetectionRate` | 成功识别的过期 checkpoint 数 / 注入漂移数 | 10，**独立报** |

`LoopResumeRate` 的分母是 **5**，`WorkspaceDriftDetectionRate` 的分母是 **10**。
F6 不进恢复率分母的理由见下方「不得夸大的部分」。

#### 副作用判定的口径

日志（`claude_dir/journal.jsonl`，沙箱之外、append-only、每条 fsync）记录每次工具执行
前后的**已声明产物摘要**，判定完全机械：

```
RedundantReExecution = resumed 段某条的 (tool, input_fp) 在 killed 段已出现
DuplicateSideEffect  = 上述条目中 outcome == "ok" 且世界再次改变
```

不依赖任何工具分类表——分类表是可以调参的数字，`post_state` 差异是观测。三类工具自动落位：

| 工具形态 | 二次执行 | `Redundant` | `Duplicate` |
|---|---|---|---|
| `Bash: echo x >> NOTES.md`（append） | 成功，世界再变 | ✅ | ✅ |
| `Write` 同内容（幂等） | 成功，世界没变 | ✅ | ❌ |
| `Edit` 同 `old_string` | 失败（首次已消费掉匹配） | ✅ | ❌ |

#### 四层判分

State（哨兵真的写了、进程真的死了、checkpoint 载入、transcript 结构合法、任务快照恢复）
→ Execution（无重复副作用）→ Workspace（夹具自带的测试套件仍通过）→ Task（用例判据通过）。
四层全过才 `resume_success = True`，且 `failpoint_reached` 在四层**之前**：
没到达杀点的运行是一次普通成功，不是恢复。

#### 不得夸大的部分（重要）

- **续写策略是脚本化的。** 两条腿都由 `ScriptedToolSequence` 驱动，「恢复后重发同一个工具
  调用」是**脚本写死的**，不是模型判断出来的。本节测的是**运行时的恢复语义**，不得表述为
  「Longline 的 agent 会重复执行副作用」。
- **评测日志能观测 ≠ 生产具备 exactly-once。** §5.4 说过「工具已产生副作用、结果未落盘」
  的窗口从落盘数据无法判定——对的，从 transcript 判定不了。本轮把日志放在 transcript
  **之外**，于是那个窗口第一次变得可判定。这仍然只是「评测装置能观测到」，不是生产保证。
- **`WorkspaceDriftDetectionRate` 的诚实结果是 0。** 全仓检索确认生产**没有任何工作区
  身份校验**：transcript、journal、工具都不记录 checkpoint 是对着哪个版本的文件取的。
  所以恢复段会在一个已经变了的工作区上继续跑，且没有任何工具错误提到这件事。这是**检出率
  臂**，混进恢复率分母会让头条数字因为一个与恢复能力无关的原因变难看。
  `test_workspace_drift_goes_undetected` 钉住这个事实；**当它开始失败时**，说明检出机制出现了，
  本节的措辞必须改。
- **turn-0 checkpoint 是对生产行为的刻意偏离。** `main.py` 只在 `run_turn()` 返回后存
  （`main.py:806-809`），而 `run_turn()` 内部就是 `query_loop` 的 `while` 状态机
  （`query_loop.py:165`），**一条指令内一次盘都不落**。照搬生产的话，「新会话第一条指令刚发出
  就崩溃」会留下没有任何 session 文件的状态，`before_model` 恢复时 `load_session` 返回
  `None`，这条臂会因为**与恢复能力无关的原因**必然失败——那不是对照组，是同义反复。所以
  `arm` 在启动 loop 前先存一次（`[UserMessage(指令)]` + 任务快照）。
- **重复次数是反例搜索，不是采样率。** 重复只变化用例 id，续写策略仍是脚本化的，运行基本
  确定。报告以**绝对计数**呈现（「60 次中 0 次」），不要读成统计比例。
- **Windows 上杀进程是 `TerminateProcess`**，与 POSIX `SIGKILL` 语义不同（无信号处理、
  不 flush 缓冲）。日志靠显式 `fsync` 而非进程退出时的刷新，因此这一点不影响结论。
- **不预设任何目标数值。** 指标测出什么就报什么。

#### 运行时语义（本节的实测发现，不是目标值）

因为 checkpoint 只在指令边界落盘，**指令 1 内部的任何杀点留下的磁盘状态是同一个**：
只有 turn-0 那一行。于是 `before_model` / `before_tool` / `after_tool` 三个臂的恢复段
**没有选择，只能重做整条指令**。

**这就是本节最有价值的结论**：该运行时在一条指令执行期间**没有任何中途持久化**，
中断落在指令内部会丢掉整条指令的全部工作——不只是「副作用没去重」，而是没有轮内 durability。

#### 实测结果（2026-09-19，6 故障点 × 10 重复 = 60 run，全离线）

原始数据：`evals/results/loop_resume_offline/raw.jsonl`（60 行）与 `summary.json`。

| 故障点 | 哨兵触发 | 恢复成功 | 副作用分母 | 重放 | 真翻倍 |
|---|---:|---:|---:|---:|---:|
| `before_model` | 10/10 | **10/10** | 0 | 0 | 0 |
| `before_tool` | 10/10 | **10/10** | 0 | 0 | 0 |
| `after_tool` | 10/10 | **0/10** | 10 | **10** | **10** |
| `after_checkpoint` | 10/10 | **10/10** | 20 | 0 | 0 |
| `truncate_tail` | 10/10 | **10/10** | 20 | 0 | 0 |
| `workspace_drift` | 10/10 | 10/10（不进分母） | 0 | 0 | 0 |

```text
LoopResumeRate               = 40/50 = 80.0%   (95% Wilson: 67.0% – 88.8%)
WorkspaceDriftDetectionRate  =  0/10 =  0.0%   (95% Wilson:  0.0% – 27.8%)
```

**60/60 哨兵真实触发。** 10 个种子全部用到（`distinct seeds = 0..9`）。

#### 读这两个副作用指标时**必须逐臂**，聚合值会误导

聚合是 `redundant = 10 / denominator = 50`，而 `duplicated = 10 / 50 = 20%`。
**这个 20% 不该被引用**，因为三组臂对「副作用重复」的暴露程度**结构上不同**：

- `before_model` / `before_tool` / `workspace_drift`：杀点在任何副作用**之前**，killed leg
  没改过任何东西，**分母天然是 0**——它们**没机会**重复。
- `after_checkpoint` / `truncate_tail`：分母 20（指令 1 的 `Edit` + `Bash` 追加），
  而恢复段**正确地一次都不重放**，所以分子 0。
- `after_tool`：分母 10，**10 次全部重放且全部真的翻倍**。

把这三类加进一个分母，等于把「没机会重复」和「有机会但没重复」混在一起算。

**正确的读法是逐臂**：`after_tool` 臂上，**故障前发生的副作用 100% 被重复执行、
且 100% 真的二次生效**；其余臂要么无物可重复，要么零重放。

`LoopResumeRate` 同理必须逐臂读：**它的 10 个失败全部来自 `after_tool`**，且每个都是一次
真实的副作用翻倍——`NOTES.md` 里 `fixed-add` 出现了两次，`not_contains` 判据因此不过。
这是**发现**，不是噪声。若失败分散在各臂，那才是脚手架有问题。

#### 本轮由 60 次扫描（而非任何单次运行）抓出的检查器缺陷

第一次扫描时 `after_checkpoint` 的 state 层 **10/10 全败**，把 `LoopResumeRate` 压到 30/50。
根因是 `check_transcript_structure` 从 `recovery_worker` 抄来的一条规则：
**「transcript 不得以 assistant 消息结尾」**。

**这条规则在本套件里是错的。** `main.py` 每个 `run_turn()` 之后写一次 checkpoint，
所以两次指令之间文件**正常地**以 assistant 的最终文本结尾——下一条用户消息会在下次
`run_turn()` 前追加。`truncate_tail` 之所以没踩到纯属侥幸：尾行被截掉后恰好以
`tool_result` 结尾。

而它想防的情况（尾部 `tool_use` 无配对结果）**已经被配对检查覆盖**，删掉不丢东西。
修复见 commit `982487d`，并有两条测试钉住新语义（以文本 assistant 结尾 → 通过；
尾部未配对 `tool_use` → 仍然拒绝）。

**这是同一个失败模式第三次出现**：检查器本身的语义错了，于是报出一个不存在的缺陷。
**单次运行永远抓不到它**——只有 6×10 的矩阵让 10 个格同时红，才显出"这不是随机"。

#### 局限（设计预测被推翻的部分）

设计文档曾预测 `truncate_tail` 是唯一能走到 `validate_transcript()` 孤儿修复路径的臂。
**实测不成立**：`load_session()` 自己会跳过无法解析的行（`storage.py`），损坏行在
`validate_transcript()` 看到它之前就已被丢弃，所以**该臂的** `repairs` 是空的。
该臂实际测的是**损坏容忍**——运行要能挺过半条未写完的记录——这本身值得测，
但「它走到了修复路径」是错的，`test_a_torn_tail_is_dropped_by_load_session_not_repaired`
的 docstring 记录了这次更正。

> **2026-09-20 更正。** 上面那句原本写的是「`repairs` 始终为空」，**限定语是我后来补的，
> 原句是错的**。逐格数过之后：`before_tool`、`after_tool`、`workspace_drift_unrelated`
> 每条 run 都报 `repairs=["truncated_tail"]`（v5 起就是 30/70，v7、v8 相同），
> `truncate_tail` 臂是 0/25。
>
> 而且设计预测被推翻的方向**比这里写的更彻底**：落在孤儿修复路径上的不是
> `truncate_tail`，而是 `truncate_tail` **之外的**那几条臂。原因是「末尾截断」这条修复的
> 触发条件是「最后一条消息是带 `tool_use` 的 assistant」——`before_tool` / `after_tool`
> 的进程正是死在这个位置上（模型已经发出 `tool_use`，结果还没落盘），而
> `truncate_tail` 把半行写坏之后，`load_session()` 丢弃它，剩下的消息末尾**不是**
> assistant，于是这条修复反而不触发。
>
> 该臂**没有**测到它原本要测的东西，`repairs` 非空的那三条臂测到了。

#### 运行方式

2026-09-19 起有了专用驱动（`longline/eval/loop_resume_cli.py`）。它仍不接
`longline.eval.cli`：那套 CLI 用 `SUITES` 注册表驱动 token / 工具选择集，有自己的一套
run 目录布局；本套件产出的是 `LoopResumeSummary`——逐臂比率、副作用日志、漂移判词——
塞进去会让一个 CLI 有两种互不相干的「一次运行」定义。

```bash
uv run --extra dev python -m longline.eval.loop_resume_cli --out evals/results/loop_resume
```

写 `raw.jsonl`（每 run 一行）与 `summary.json`（含 `by_failpoint_counts` 与 `problems`）。
**任一 run 的 child 没有发出 sentinel，退出码为 1 并在 stderr 打印**：一个没触发的注入
算出来的比率，和一个触发了的比率长得一模一样，只是更差，而这个差别在数字里看不出来。

`--no-durability` 是消融开关，见 §5.9。

单跑一个故障点：

```bash
uv run --extra dev pytest tests/integration/test_eval_loop_resume.py -q
```

全离线，零 API 花费。**读结果前先读上面「必须逐臂」那一节。**

单跑一个故障点的前 3 个 run：

```bash
uv run --extra dev python -c "import asyncio; from pathlib import Path; from longline.eval.loop_resume import cases_by_failpoint, load_loop_resume_cases; from longline.eval.loop_resume_runner import aggregate_loop_resume, run_loop_resume_suite; g=cases_by_failpoint(load_loop_resume_cases(Path('evals/loop_resume.jsonl'))); runs=asyncio.run(run_loop_resume_suite(g['after_tool'][:3], api_key='offline', fixtures_dir=Path('evals/fixtures'))); print(aggregate_loop_resume(runs).to_dict())"
```

---

### 5.9 修复轮内持久化与工作区漂移（2026-09-19）

§5.8 打出的两个洞：`after_tool` 10/10 重复副作用、`workspace_drift` 检测率 0。本节记录
修复、**同装置**的前后对照，以及修复**新开出来的第三个洞**。

#### 运行时改了什么

| 机制 | 位置 | 作用 |
| --- | --- | --- |
| 步骤级检查点 | `query_loop.on_step` → `main.py` | 每次 transcript 写入点存一次；关键是**模型响应**那一次——实测工具体运行时 assistant 消息已在 `messages` 里，所以 `tool_use` 在工具动手之前就落盘了 |
| 持久化工具日志 | `session/tool_journal.py` | 每次调用 `PREPARED`（执行前）→ `COMMITTED`（执行后），逐条 fsync，写在会话目录而不是工作区 |
| 恢复时对账 | `tool_journal.reconcile_pending` + `Tool.reconcile` | `PREPARED` 无 `COMMITTED` 的操作，问工具「你的效果在不在」，得 `APPLIED` / `NOT_APPLIED` / `UNKNOWN`。`UNKNOWN` **不重放** |
| 工作区身份 | `session/workspace_identity.py` | 从日志的 read/write 集合 + `git HEAD` 判断漂移属于 `clean` / `unrelated` / `relevant`；只有 `relevant` 拒绝恢复 |

`Tool.reconcile` 与 `Tool.workload` 的默认值分别是 `UNKNOWN` 与 `{}`——**「说不出来」而不是
「没有」**。Bash 两类都不声明：从这个层次看，「往文件里追加」和「读一个文件」是同一个字符串，
声称知道就是拿猜测当事实。

#### 前后对照，同一套装置

`--no-durability` 把上表两个机制关掉，其余（harness、用例、判分器、脚本模型、副作用日志）
全部不变。**不能用「切到改动前的 commit」当 before**：本 harness `import` 了
`session.tool_journal` 与 `session.workspace_identity`，那些模块在旧版本里不存在，跨版本
比较会变成跨 harness 比较。

7 臂 × 10 次 = 70 run × 2 cell，全离线：

| 臂 | before 执行层 | after 执行层 | before 任务层 | after 任务层 | after den |
| --- | --- | --- | --- | --- | --- |
| `before_model` | 10/10 | 10/10 | 10/10 | 10/10 | 0 |
| `before_tool` | 10/10 | 10/10 | **10/10** | **0/10** | 0 |
| `after_tool` | **0/10** | **10/10** | **0/10** | **10/10** | 10 |
| `after_checkpoint` | 10/10 | 10/10 | 10/10 | 10/10 | 20 |
| `truncate_tail` | 10/10 | 10/10 | 10/10 | 10/10 | 20 |
| `workspace_drift`（检测） | 0/10 | 10/10 拒绝 | — | — | — |
| `workspace_drift_unrelated`（检测） | 0/10 | 0/10 拒绝 | — | — | — |

```
指标                    before(消融)     after
LoopResumeRate          40/50            40/50
DuplicateSideEffect     after_tool 10/10 after_tool 0/10
DriftRecall             0/10             10/10
FalseRejectRate         0/60             0/60   (分母 = 非 relevant-drift 的全部 60 run)
```

**消融 cell 逐臂复现了改动前的行为**（`after_tool` 0/10、`before_tool` 10/10、
`LoopResumeRate` 40/50），这是消融可信的证据，不是巧合。

#### 结论一：`after_tool` 的洞堵上了，而且不是空转

`after_tool` 从 0/10 到 10/10，**分母仍是 10**——那一次 Bash 追加真的发生过、真的只追加了
一次。这是本轮唯一一个「故障注入发现缺陷 → 改运行时 → 原故障消失」的完整闭环。

机制是两层，缺一不可：检查点让恢复方**知道这个调用被发出过**；日志让运行时能说
「这条操作开始了但没回话」。`Bash` 的判词是 `UNKNOWN` 而不是 `APPLIED`——shell 命令的效果
读不回来。运行时因此**拒绝重放**，而不是宣称成功。

#### 结论二：我开了一个新洞，`before_tool` 从 10/10 掉到 0/10

`before_tool` 含义是「进程死在工具执行之前」。但日志里 `PREPARED` 无 `COMMITTED` 的窗口
**同时容纳两种情况**：「进了 `tool.execute` 就跑掉了」和「跑完了但结果没回来」。运行时
分不出这两者——对一个自己读不回效果的 Bash，它老实地说 `UNKNOWN`。

脚本模型对 `UNKNOWN` 的处置是「不重复调用」（这正是修 after_tool 的那条规则）。于是这一次
追加**根本没发生**，`NOTES.md` 里没有 `fixed-add`，任务层 10/10 全败。

净效果是 **40/50 对 40/50**：一个重复换成了一个遗漏。

**必须说清楚这两者不等价**：重复的副作用可能不可恢复，遗漏是可以被发现、被重试的；所以
方向是对的。但**headline 数字没动**，任何「恢复率提升」的说法都是假的。

真正的修法是给运行时一个**不可逆点**信号：`PREPARED` 与「工具真的动手了」之间需要第三个
状态，而**只有工具知道自己的不可逆点在哪**（`BashTool` 在 spawn 子进程之前，
`FileWriteTool` 在 `os.replace` 之前）。这需要工具配合，是下一步，不是本轮。

> 下一步已经做完：见 §5.10。`before_tool` 回到 10/10，`LoopResumeRate` 回到 50/50，
> 而这一节表格里的 after 列（`before_tool` 0/10、`LoopResumeRate` 40/50）记的是**修复前**
> 的那一次运行，作为 v3 的历史保留。同样被 §5.10 取代的还有这一节三条臂的任务判据——
> 它们当时只有 `contains`，一个任何输入都满足的判据。

#### 结论三：旧的那个 `WorkspaceDriftDetectionRate = 10/10` 测错了东西

§5.8 把它记成 10/10，判据是 `drifted and bool(tool_errors)`——**「恢复腿报了工具错误」**。
而恢复腿在没有身份检查时会重放整条指令，`Edit` 的 `old_string` 已经被第一次执行改掉了，
于是必然报错。那个 10/10 测的是「重放会撞车」，不是「漂移被检测到」。

换成真判据（运行时真的拒绝恢复）之后：修复前 **0/10**，修复后 **10/10**；无关漂移的
误拒率 **0/60**。三个数字必须一起读——一个「永远拒绝」的检测器召回率满分，一个
「从不拒绝」的误拒率满分。

#### 口径（与数字同等重要）

这 50 次注入是**确定性重复**，种子变的是输入不是分布。数字只能读作：

> 在五类可恢复故障场景的 50 次故障注入中，40 次满足完整恢复判据；10 次失败全部集中在
> 「工具已产生副作用但结果未持久化」这一个窗口。

**不能**读作「80% 恢复概率」。§5.8 已记过一次同类误读。

#### 局限

1. **`UNKNOWN` 的代价就是结论二的遗漏。** 运行时在无法验证时选择不重放，这个选择是对的，
   但它把成本转移给了模型：模型应当去**核实**（读一下 `NOTES.md`），而不是跳过。脚本模型
   只做了保守的那一半。
2. **Bash 改变的文件的漂移，看起来像无关漂移。** `Bash` 不声明 workload，所以它写的文件
   永远进不了 write 集合；`after_tool` / `after_checkpoint` / `truncate_tail` 三条臂因此都
   带着一条它们不该有的 unrelated 警告。运行时**警告而不是拒绝**，因为拒绝就等于因为一个
   它无法归因的改动而挡住恢复。这是实测到的限制，不是设计选择。
3. **步骤级检查点每步都重写整个 transcript**，一个会话是 O(n²) 次写。CLI 会话规模下可以
   接受；真正的修法是「追加式步骤日志 + 周期性压实」，那是另一个项目。
4. **无关漂移的检测依赖 git。** 非 git 目录下只能看见依赖文件的变化（自己记了哈希），
   看不见别的，`DriftReport.git_available` 会如实报 False。

### 5.10 不可逆点：让「没跑过」和「跑了没回话」可区分（2026-09-19）

§5.9 结论二记录的那个洞：`before_tool` 从 10/10 掉到 0/10。本节记录洞的准确形状、修法，
以及修复过程中查出来的第二个问题——**三条臂的判分器恒真**。

#### 洞的准确形状

`PREPARED` 无 `COMMITTED` 这一个状态同时容纳两件事：

- 工具进了 `execute`、跑完了、结果没回来（`after_tool` 的杀点）；
- 工具**根本没开始**（`before_tool` 的杀点）。

Bash 的效果读不回来，所以运行时对这两件事都只能说 `UNKNOWN`；而 `UNKNOWN` 的规则是
**不重放**。于是第二种情况里，模型被告知「结果未知，没有重跑」——一个**可证明什么都没发生**
的调用被判成不可恢复，任务静默地永远完不成。

要点在于：这不是「保守一点的代价」。保守是**把未知当成未知**，而这里有一半是已知的：
「shell 没有 spawn 过」完全可以判定，运行时只是没有把它记下来。

#### 运行时改了什么

| 机制 | 位置 | 作用 |
| --- | --- | --- |
| 不可逆点声明 | `Tool.mark_irreversible()` / `irreversible_point()` | 执行器在一次调用期间经 `ContextVar` 发布一个 marker；工具在**自己的**不可逆点报告它 |
| 第三个状态 | `tool_journal.EXECUTING` | `PREPARED → EXECUTING → COMMITTED`，同样逐条 fsync |
| 对账多一个事实 | `Tool.reconcile(input, *, started)` | `started` = 日志是否见过这次调用到达不可逆点 |
| Bash 的答案 | `bash_tool.reconcile` | `started=False` → `NOT_APPLIED`（可重试）；`started=True` → `UNKNOWN`（不重试） |

`BashTool` 把 `mark_irreversible()` 放在 `create_subprocess_shell` 的**上一行**：那之前全是
校验，被拒绝的命令什么也没改；那之后 shell 里干了什么读不回来。

用 `ContextVar` 而不是工具实例属性：流式执行器同时跑最多 10 个工具，属性会让一次调用的
marker 落到另一次调用的记录上。也没有去改 `execute` 的签名——那会牵动每一个工具和每一个
wrapper。

#### 为什么默认值仍然保守（重要）

`Tool.reconcile` 的默认值在 `started` 两种取值下**都是 `UNKNOWN`**。

理由是：一个从不报告不可逆点的工具，它历史上**每一次**调用都是 `started=False`。如果默认值
把 `started=False` 读成「证明没发生」，这个工具所有被打断的操作都会被放行重试——包括已经
落地的那些。`FileEditTool` 就地写明了这个陷阱：它读回文件、忽略 `started`，因为对它而言那个
标记只有 `False` 一种取值。

所以**标记只对写了标记的工具有效**。这条约束由一个签名契约测试守住：`reconcile_pending`
把任何异常都吞成 `UNKNOWN`，一个忘了新参数的旧 override 不会崩，只会静默地永远返回
`UNKNOWN`——变成「某条恢复路径突然不灵了」，而不是一个报错。

#### 这个写入是屏障，不是日志

`PREPARE` / `COMMIT` 的失败被吞掉：会话目录写不了，不该成为用户的编辑不发生的理由。

`mark_irreversible()` 的失败**不吞**：异常传进工具体内，工具放弃这次操作。吞掉它会留下
「跑过了但没有标记」的 `PREPARED`，而「没有标记」正是授权重试的那个条件——吞掉它就等于把
这个机制要防的重复副作用重新打开。`BashTool` 把它放在 `try` 内部，所以标记写不下去时 shell
根本不会被 spawn。代价是一条命令会因为**记账失败**而拒绝执行，这是有意的取向了。

#### 判分器缺陷：三条臂的 judge 恒真（与运行时无关）

写这一节时发现的第二个问题。`lr-before-model`、`lr-before-tool`、`lr-truncate-tail` 三条臂
的任务判定只有 `NOTES.md contains "fixed-add"`，**没有** duplicate guard——七条臂里只有
`lr-after-tool` 与 `lr-after-checkpoint` 有。而前两条臂的 `den = 0`（故障前那条腿里没有任何
改变状态的操作），所以 `DuplicateSideEffectRate` 的分母结构性为 0。

两者叠加：`lr-before-model` 与 `lr-before-tool` 上**没有任何判据能把「追加了两次」判成失败**。
10/10 在那种情况下依然成立。`lr-truncate-tail` 好一些——它的 `den = 20`，执行层能看见重放——
但它的任务层同样看不见。

修法是把 `"not_contains":"(?s)fixed-add.*fixed-add"` 补到这三条臂的 judge 上，然后**两个
cell 全部重跑**（`loop_resume_v5_*`）。判分器只收紧、不放松，所以这不是把数字改好看，而是把
一个原先不可能失败的判据变成可以失败的。

`contains` 与 `not_contains` 是**一对**：前者要求至少一次，后者排除两次及以上，合起来是
「恰好一次」。写进文档之前先验过 `judge_file_content` 真的会判假，而不是把一个没人读的
kwarg 摆在数据里——四种输入的结果是「一次 真 / 相邻两次 假 / 没有 假 / 相隔两次 假」。

补第三条臂这件事本身就是那个检查生效的证据：先把 `after_tool` 的单臂测试改成「凡检查
`NOTES.md` 内容的臂都必须带 duplicate guard」，它立刻把 `lr-truncate-tail` 指了出来。

同一类问题也在集成测试里：`test_the_before_tool_arm_retries_only_the_step_that_never_ran`
断言了 `den == 0`、`duplicates == 0`、结构完整、没有冗余重放——**这些在一个「什么都没干成」
的运行上全部成立**，所以它在 `before_tool` 是 0/10 的那段时间里一直是绿的。已补上
`layer_task_ok`。「没造成伤害」和「把活干完了」是两个不同的断言，原先只查了前一个。

#### 实测结果（两个 cell 都在收紧后的判据下重跑）

`loop_resume_v5_durability`（`--label durability-point-of-no-return`）与
`loop_resume_v5_ablation`（`--no-durability`，`--label ablation-point-of-no-return`），
各 7 臂 × 10 = 70 run，共 140 run，全离线。两个 cell 的 `problems` 都是 `[]`，即 140/140
个哨兵真实触发——没有一次「注入没落地」被记成失败。

| 臂 | before（消融） | after | after `den` | after `dup` | after `redun` |
| --- | --- | --- | --- | --- | --- |
| `before_model` | 10/10 | 10/10 | 0 | 0 | 0 |
| `before_tool` | 10/10 | **10/10**（v3 是 0/10） | 0 | 0 | 0 |
| `after_tool` | **0/10** | **10/10** | 10 | 0 | 0 |
| `after_checkpoint` | 10/10 | 10/10 | 20 | 0 | 0 |
| `truncate_tail` | 10/10 | 10/10 | 20 | 0 | 0 |
| `workspace_drift`（检测） | 拒绝 0/10 | 拒绝 10/10 | 10 | 0 | 0 |
| `workspace_drift_unrelated`（检测） | 拒绝 0/10 | 拒绝 0/10 | 10 | 0 | 0 |

```
指标                    before(消融)     after(v5)
LoopResumeRate          40/50            50/50
DriftRecall             0/10             10/10
FalseRejectRate         0/60             0/60
```

**消融 cell 在新判据下一格没动**：`after_tool` 仍是 0/10（`dup = 10`、`redun = 10`），
`before_tool` 仍是 10/10，`LoopResumeRate` 仍是 40/50。这是必要的对照——判分器只收紧不放松，
所以如果 after 列的改善是判据变严造出来的，消融列会先动。它没动。

`false_reject` 的两列不能同等读：**消融列的 0/60 是结构性为零**——`--no-durability` 连工作区
身份一起关掉，那一格里没有任何运行**可能**被误拒。有意义的是 after 列的 0/60：检测器开着，
60 次里一次该放行的都没拦。

**这套数字支持什么、不支持什么**，逐条写清：

- **支持**：`before_tool` 从 0/10 回到 10/10。恢复段确实重跑了那个没跑过的调用——不重跑
  `NOTES.md` 里不会有 `fixed-add`，而这正是 v3 那次失败的那条判据。
- **支持**（这一次有判据撑着，v3 没有）：「那条 append 恰好落了一次」。`before_tool` 上执行层
  仍然帮不上忙（`den = 0`），但任务层现在是 `contains` + `not_contains` 一对，10/10 意味着
  10 次运行里 `NOTES.md` 的 `fixed-add` 都**恰好出现一次**——追加两次会在这里被判假，而那正是
  这条臂要防的结果。v3 的 10/10 没有这个含义，因为当时只有一个任何输入都能满足的 `contains`。
- **支持**：`after_tool` 的 10/10 没有被这次改动换掉，两格 140 次运行里 `dup = 0`、
  `redun = 0`。即：**没有一条臂是靠放行一次重复副作用换来分数的**。
- **仍然不支持**：「Bash 调用被发出了一次」。判据看的是**产物**，不是调用记录。一次重放如果
  追加的内容不含 `fixed-add`，两处都看不见。能说清「调用层」的是 `DuplicateSideEffectRate`，
  而它在这条臂上的分母是 0。要做这个更强的断言，需要一层按调用计数的记账，现在没有。

#### 局限

1. **「没跑过就重试」只对写了标记的工具成立。** 换一个工具（比如某个 HTTP 工具），它要么自己
   报告不可逆点，要么在 `reconcile` 里读回自己的效果；**没有第三种免费的写法**。忘掉这一步的
   工具不会报错，只会永远返回 `UNKNOWN`——由那个签名契约测试挡住一部分，但挡不住「写了
   override 却忘了调 `mark_irreversible()`」。
2. **标记写失败会让命令拒绝执行。** 这是有意的（屏障而非日志），但它把「记账可用性」放进了
   「命令能否执行」的因果链里。盘满或会话目录只读时，Bash 会返回错误而不是照常运行。
3. **Bash 改变的文件的漂移，看起来像无关漂移**（同 §5.9 局限 2，未变）。
4. **无关漂移的检测依赖 git**（同 §5.9 局限 4，未变）。

---

### 5.11 任务多样性：换五个任务之后，套件第一次就抓出了东西（2026-09-20）

#### 为什么要有这一格

前面所有矩阵都是**同一个任务**（`resume_repo` 里的 `add()` 与 `NOTES.md`）。一个只对那一种
任务形状成立的 Runtime，和一个真的对的任务成立的 Runtime，在前面的数字上长得一模一样。
这一格换的是**任务**，不是故障点：

```
5 个任务 × 4 个故障点 × 5 个 seed = 100 run，全离线，零 API 花费
数据集：evals/loop_resume_tasks.jsonl
```

四个故障点取的是与工具无关的那四条：`before_tool`、`after_tool`、`after_checkpoint`、
`truncate_tail`。任务与门控工具：

| 任务            | fixture         | 门控工具 | 这一臂要测的形状                     |
| --------------- | --------------- | -------- | ------------------------------------ |
| `lrt-fix-add`   | `resume_repo`   | `Bash`   | 外部副作用（与旧矩阵同形，作基线）   |
| `lrt-multi-edit`| `multi_repo`    | `Edit`   | 一个轮次里两次编辑，跨两个文件       |
| `lrt-many-reads`| `wide_repo`     | `Edit`   | 三次读之后才编辑，读集合大而写集合小 |
| `lrt-new-file`  | `generate_repo` | `Write`  | 写一个**尚不存在**的文件             |
| `lrt-edit-only` | `parse_repo`    | `Edit`   | 只读一个文件、只改它（最小形状）     |

旧套件里**唯一**的「写完再杀」臂门控在 `Bash` 上，而 `Bash` 不申报 workload——所以它的文件
永远进不了读集合。**是工具的选择把这个洞藏住了**，不是这个洞藏得好。

#### seed：重复必须真的不一样

`repeat: 5` 若每次都从字节相同的 fixture 出发，那五次的差别只剩计时，这一列测的是抖动。
seed 因此升级为 case 自己的数据（`Scenario.seed_files`），由 `apply_seed` 按 repeat 序号代入，
且：

- loader **拒绝**没有声明 `seed` 的 scenario——新加的 case 不能默默退回「五次同一份」；
- 占位符在加载时对照已知集合校验——写错一个字母就永远不会被代入，重复会**再次**悄悄变成
  同一份，这正是这套机制回来要防的那个失败；
- **操作数与标签分开**（`SEED_OPERANDS`）：`<seed>` 是 0 无妨（它是标题行），但操作数为 0
  会让 fixture 的 bug 碰巧正确——`add(a, b)` 返回 `a - b` 在 `b == 0` 时是过的——那一格的
  workspace 层就空了。操作数从 2、3 起步。

实测：5 个 fixture × 5 个 seed，**修之前**全部失败（`resume_repo` / `multi_repo` /
`parse_repo` / `wide_repo` 返回 1，`generate_repo` 返回 2，即模块尚不存在、collect 就失败），
**没有一个 seed 让 bug 碰巧通过**。修之后 100/100 通过。

#### 实测结果（100 run，全离线）

| 判据                                | 值          |
| ----------------------------------- | ----------- |
| sentinel 真实触发（`failpoint_reached`） | 100/100 |
| checkpoint 载入                     | 100/100     |
| 四层全绿（`passed`）                | 100/100     |
| `LoopResumeRate`                    | 100/100     |
| `FalseRejectRate`                   | 0/100       |
| 重复副作用 / 冗余重放               | 0 / 0       |
| `problems`                          | `[]`        |

逐臂都是 25/25。**`DriftRecall` 这一格是 0/0，即未定义，不是 0**——本数据集没有漂移臂，
漂移由 §5.8/§5.9 的 `workspace_drift*` 臂负责。

#### 这一次跑抓出的东西

**在一次写工具内部崩溃，会让这个会话永久无法恢复**，而拒绝它的理由是**它做了要求它做的事**。
证据在日志层：`PREPARED Edit tu-2` 没有对应的 `COMMITTED`，`git status` 显示
`M src/parser.py`。

形状是三层叠加：被中断的写没有 `COMMITTED`，所以不在写集合里（**这是对的**，用它的
`pre_state` 比等于拿一个会话从未主张过的版本比世界）；但它还在**读集合**里（会话先读后改）；
于是「摘要对不上」被判成 `relevant` ⇒ 拒绝。**这一格把「保守」变成了「坏掉」。**

修法只结算**对账确认落地**（`RECONCILED`）的被中断写。判词三类只有一类能结算：

| 对账结果        | 含义                       | 是否结算 | 理由                                 |
| --------------- | -------------------------- | -------- | ------------------------------------ |
| `RECONCILED`    | 工具读了，找到了自己的效果 | **是**   | 盘上的版本可归因于本会话             |
| `ABORTED`       | 找到旧文本完好             | 否       | 与「被注入的漂移」不可区分           |
| `INDETERMINATE` | 读不出自己的效果（`Bash`） | 否       | 文件变了，但没人能说是谁改的         |

**第一版是错的，记在这里**：第一版结算「任何文件内容变了的被中断写」，理由是「多半是它自己
写的」。它**通过了全部恢复测试**，却打挂了 `test_the_dependent_drift_arm_is_refused`——
`workspace_drift` 臂的 `Edit` 是在**运行之前**被打断的，父进程随后往同一个文件追加，
「摘要变了」分不出这两种情形。对账能分出来（那里读到的是旧文本完好），规则因此收窄。

#### 规则本身的前后对照（隔离 worktree，同一份探针）

100/100 这种比率回答不了「翻转的是哪一格」。探针
`evals/probes/workspace_drift_rule.py` 把这条规则单独问一遍，在两个 revision 上各跑一次：

| 对账判词              | 修前（`e04a6ad`）    | 修后           |
| --------------------- | -------------------- | -------------- |
| `RECONCILED`（已落地）| `relevant` **拒绝**  | `clean` 通过   |
| `ABORTED`             | `relevant` 拒绝      | `relevant` 拒绝 |
| `INDETERMINATE`       | `relevant` 拒绝      | `relevant` 拒绝 |
| 从未对账              | `relevant` 拒绝      | `relevant` 拒绝 |

**恰好翻一格，其余三格一动不动**——包括漂移臂依赖的那两格。探针开头那句
`workspace_identity from:` 是必须读的一行：两次运行载入了同一个模块的话，这次对照什么都没测。

#### 同一次排查里查出的第二个缺陷

`changed_paths()` 调用 `git status --porcelain` 时没带 `--untracked-files=all`。git 会把
全体未跟踪的目录**折叠成目录本身**（`?? src/`），而这个集合是拿去和**记录的文件路径**做差集
的——于是会话自己刚在新目录里建出来的文件（`lrt-new-file` 的 `src/generated.py` 正是），
谁也对不上，被报成「别人改的无关文件」。

#### 15 行 `unrelated` 是这个已知局限，不是新问题

`workspace_verdict` 分布：`clean` 85、`unrelated` 15。15 行全部是 `lrt-fix-add`，原因是
`Bash` 改了 `NOTES.md` 而 `Bash` 不申报 workload，所以那个文件不在记录集合里 ⇒ 差异集里出现
⇒ 「无关漂移」。这与 §5.9 局限 2 / §5.10 局限 3 是同一条，未变。`before_tool` 臂 25/25 全
`clean`，因为那次 `Bash` 根本没跑——分布随臂变化，正是它应该有的样子。

#### 不得夸大的部分

1. **`100/100` 是确定性重复下的计数，不是概率。** 模型输出是脚本化的，重放不引入随机性，
   所以 Wilson 区间在这里没有解释力。它说明的是「这套形状下没有观察到失败」。
2. **这一格测的是这五个任务，不是「任意任务」。** 五个 fixture 都是小型 Python 仓库；
   这里没有大文件、没有二进制、没有网络副作用。
3. **它抓到的那个缺陷，是旧套件在结构上碰不到的**——这是这一格的价值所在，但**不等于**
   「这五个任务覆盖了所有 Runtime 形状」。它覆盖的是**工具种类**：`Bash` / `Edit` / `Write`。
4. **`workspace_verified_applied` 非空的只有 20 行**，正好是 `Edit`/`Write` 门控的
   `after_tool` 臂（4 任务 × 5 seed）。`Bash` 门控的那 5 行是空的，因为它读不出自己的效果。
   这个数是**规则生效范围的证据**，不是覆盖率。

---

### 5.12 重启 vs 恢复：续上到底省了什么（2026-09-20）

#### 这一格要回答的问题

恢复率说明「能不能续」，不说明「续上值不值」。崩掉的会话真正面对的选择只有两个：**从头重做**
（`restart`），还是**从检查点接着做**（`resume`）。这一格把两个分支都真跑一遍，量的是
ModelCalls / ToolCalls / Tokens / LoopTime。

**被杀死的那条腿不在任何一列里，这是有意的**：它是**沉没成本**——进程两种情况下都已经死了，
它死前做的功在两个分支里**都已经付过**。把它记到恢复头上，会让恢复显得不如它真正竞争的那个
选项；而那个选项（重启）还要把这部分功**再付一次**。

基线是一个**真的 run**，不是估算：自己的 `claude_dir`、自己的沙箱、同一个 fixture、同一个
seed、同一份脚本、同一个运行时和耐久性开关，唯一差别是**从哪里开始**。它**不受门控**
（`armed=False`），因为一个会在故障点停下的基线，正是它要当替代品的那个东西。

```bash
uv run --extra dev python -m longline.eval.loop_resume_cli --cases evals/loop_resume.jsonl --out evals/results/loop_resume_v9_restart_baseline --label restart-vs-resume --restart-baseline
```

#### 实测（70 run，每臂 10，全离线，`problems` 为空）

`restart → resume`，括号内是省下的比例。**基线跑的是与恢复腿相同数量的指令**：

| 臂                        | model_calls        | tool_calls         | input_tokens        | output_tokens      | loop_ms             |
| ------------------------- | ------------------ | ------------------ | ------------------- | ------------------ | ------------------- |
| `before_model`            | 4→4 (+0%)          | 3→3 (+0%)          | 400→400 (+0%)       | 85→85 (+0%)        | 63.7→70.8 (-11%)    |
| `before_tool`             | 4→3 (+25%)         | 3→2 (+33%)         | 400→300 (+25%)      | 85→65 (+24%)       | 50.3→50.6 (-1%)     |
| `after_tool`              | 4→2 (+50%)         | 3→1 (+67%)         | 400→200 (+50%)      | 85→45 (+47%)       | 47.5→13.5 (+72%)    |
| `after_checkpoint`        | 6→2 (+67%)         | 4→1 (+75%)         | 600→200 (+67%)      | 130→45 (+65%)      | 70.0→20.2 (+71%)    |
| `truncate_tail`           | 6→2 (+67%)         | 4→1 (+75%)         | 600→200 (+67%)      | 130→45 (+65%)      | 75.8→19.2 (+75%)    |
| `workspace_drift_unrelated` | 4→2 (+50%)       | 3→1 (+67%)         | 400→200 (+50%)      | 85→45 (+47%)       | 59.7→16.4 (+73%)    |
| `workspace_drift`         | 4→0 (**+100%**)    | 3→0 (**+100%**)    | 400→0 (**+100%**)   | 85→0 (**+100%**)   | 49.2→0.0 (**+100%**) |

#### 读这张表的三个坑

1. **`before_model` 的 `+0%` 不是失败，它是这张表的可证伪性。** 杀点在最前面——什么都还没做，
   所以**没有任何东西可以跳过**，恢复必须把整条指令重做。如果这一行也是「省了 50%」，
   那这个指标就是恒定的，它测的就不是恢复了。`loop_ms` 的 `-11%` 是**计时抖动**，不是恢复
   更慢：这一列是 `time.perf_counter` 计量的一段本地循环，只有几十毫秒，而同机的抖动就在
   这个量级（`longline/eval/latency_runner.py` 记了这套机器上抖动与信号的关系）。**要墙钟
   结论，看 §5.5；要「省了多少次调用」，看前四列。**
2. **`workspace_drift` 的 `+100%` 是一个度量陷阱，不是战果。** 那 10 次运行的恢复是**被安全
   策略拒绝**的，它跑了 0 次模型调用、0 次工具调用——因为**它根本没有恢复**。把它算成
   「省了 100%」等于把「拒绝服务」记成「高效」。这一行只能读作：**被拒的恢复不产生成本，
   也不产生进度**。
3. **总体值把两种量纲混在了一起，而且被上面那一行抬高。** 基线跑的是与恢复腿**相同数量**的
   指令（中途被杀的臂比「指令 1」，两条边界臂比「指令 1+2」），所以总体均值是 60 次单指令
   基线和 20 次双指令基线的加权平均。按 arms 报出来的总体是
   `model_calls +53.1%` / `tool_calls +60.9%` / `input_tokens +53.1%`；**剔除被拒的那一臂之后**
   是 `model_calls +46.4%` / `tool_calls +55.0%` / `input_tokens +46.4%` / `output_tokens +45.0%`。
   **两个都要报，第一个不能单独出现。**

#### 不得夸大的部分

1. **绝对数很小，比值才是结论。** 这是脚本化的小任务（5 个文件的仓库），
   `model_calls` 的量级是个位数。可以说「后段杀点省了约 2/3 的模型调用」，
   不可以说「生产上省 2/3」——真实任务的恢复收益取决于杀点位置和指令划分。
2. **省的是「重启」的钱，不是「已经做过的功」的钱。** `+50%` 的意思是「比重做便宜一半」，
   不是「恢复只花了原本的一半」。
3. **`loop_ms` 不是墙钟时间。** 它是本地循环的计时，排除了进程启动、沙箱复制和一次
   `pytest`。要墙钟看 `resume_latency_ms`。
4. **模型调用是脚本化的，所以这两列是确定性计数，不是分布。** 每一臂 10 次运行给出的是
   **同一个数**（`set` 大小是 1），没有方差可报——这一点本身是这套装置的性质，不是结果。

---

## 6. 冻结的历史数据（2026-09-15）

### 6.1 分支与提交

| 项 | 值 |
|---|---|
| 冻结日期 | 2026-09-15 |
| 数据集哈希所在的提交 | `88555031761567e7a9b97ab47ddb4d84d72c275c`（`main`） |
| 工作分支 | `eval/agent-runtime-metrics` |

注意：`8855503` 是**数据集哈希与基线所对应的提交**，`eval/agent-runtime-metrics` 是其上的**工作分支**，
本节后续所有评测与文档工作在分支上进行。数据集本身的 sha256 见 §6.2。

### 6.2 数据集哈希

| 文件 | 行数 | 字节数 | sha256 |
|---|---:|---:|---|
| `evals/tool_calls.jsonl` | 30 | 7344 | `cba5b42e6110f278755473bc89cd65579abf66856cdca3e801165c6d698760d8` |
| `evals/e2e.jsonl` | 10 | 2640 | `dfcf06924784e37cd257fbe190722ce96d5f737a24bd6dab6bb5002445644cf1` |

> **注意**：`tool_calls.jsonl` 的 sha256 对应**打 legacy 标记之前**的原始文件。
> 打标记后该文件为 7673 字节（原 7344），用例内容除 `tags` 新增 `"legacy"` 外**完全未变**，
> `load_cases()` 仍返回 30 条。本表数值作为「冻结时的原始数据指纹」保留。
>
> `e2e.jsonl` **未被标记**，当前仍与 `8855503` 完全一致（sha256 可直接校验）。

### 6.3 唯一一次原始结果

`evals/results/claude-sonnet-4-20250514-1.json`：

| 字段 | 值 |
|---|---|
| model | `claude-sonnet-4-20250514` |
| `l1_tool_accuracy` | `null` |
| `l2_pass1` | `1.0` |
| `avg_turns` | `2.0` |
| `avg_input_tokens` | `1383.0` |
| `avg_output_tokens` | `92.0` |
| 用例数 | **1**（仅 `e2e-001`） |

> **这不是 full-suite baseline。** 它是 **1 条用例**（`e2e-001`）的冒烟运行，
> `l1_tool_accuracy` 为 `null` 说明工具调用层根本没跑。它唯一的价值是证明 runner 链路可用，
> **不能**被引用为任何通过率、成功率或基线数字。
> 本任务**不伪造** full-suite 结果；正式基线待 Task 9 产出后写入 `evals/baselines/`。

---

## 7. 历史 README 数字的定性

根 `README.md`「Agent 评测」一节中曾给出（模型 `deepseek-v4-flash`）：

```text
工具调用准确率 ~97–100%（平均 2.7 轮）
E2E pass@1      ~70–90%（平均 3.4 轮）
```

这两组数字的定性是 **legacy exploratory（历史探索性数据）**，**理由如下**：

1. 「区间」表述（`~97–100%`、`~70–90%`）说明它们是**多次运行取稳定区间**的目测结果，
   而非固定样本量下的确定性统计——**没有分子、分母，也没有 95% CI**。
2. `l1_tool_accuracy` 把工具选择与参数正确合并成一个布尔值，**无法拆分分母**。
3. 该工具集存在**标签泄漏**（任务文本直接点名工具），测的是指令跟随而非工具选择。
4. 额外工具调用**不惩罚**，会高估准确率。
5. E2E 仅 10 条且难度偏低，**样本量不足以支撑 ±3% 级别的结论**。
6. **无 Git SHA、无 case 数据哈希、无运行环境记录**，不可复现、不可回溯。

**处理方式**：数字**保留可见**（不删除），但在原位加注，指向本文件的 §1 / §4 / §5 新口径。
任何正式报告或简历**只引用**符合本文件口径的数字。

---

## 8. 评测用例 Review Checklist

**每位 reviewer 逐个用例**手动过一遍下表；任何一条为「否」，该用例**不得合并**。
这是计划 §4.1 / §4.2 / §7 验收条件的可执行版本。

### 8.1 任务文本与标签泄漏

- [ ] **任务文本不出现任何工具名**（针对盲测集）。搜索 `Read`、`Write`、`Edit`、`Glob`、`Grep`、
      `Bash`、`WebSearch`、`WebFetch`、`NotebookEdit`、`TaskCreate` 等词，以及「用 grep」「用 glob」
      这类中文表述。出现即**泄漏**，必须改写。
      > 注：全库扫描结果——`tool_calls.jsonl` 有 **10 条**泄漏
      > （`tc-003`、`tc-004`、`tc-006`、`tc-011`、`tc-012`、`tc-015`、`tc-018`、`tc-019`、
      > `tc-028`、`tc-029`），这正是它被标为 legacy 的原因之一；
      > `e2e.jsonl` **0 条**泄漏，因此不标 legacy。
      > **`tool_selection.jsonl` 的这条规则已可执行**：
      > `tests/unit/eval/test_tool_selection_cases.py::TestNoLeakageInBlindCases`
      > 会扫描工具名（词边界、精确）与中文提示词表（**启发式**）。
      > 绿色不等于已证明不泄漏——每条盲测用例的 `blind_rationale` 才是人工复核的依据。
      > 词表在 `tests/unit/eval/leakage.py::TOOL_HINT_WORDS`，
      > 复核发现新泄漏措辞时**必须**把那个词补进去。
- [ ] 任务描述的是**目标**（「在仓库里找出所有 TODO」），而不是**手段**（「用 Grep 找出…」）。
- [ ] 显式指定工具的用例被正确打上 instruction-following 标签，且**不进入主数字**。

### 8.2 Judge 不得恒真

- [ ] judge **不能在 fixture 起始状态就已经为真**。
      典型恒真：`file_content.contains` 的字符串在 fixture 里已存在；`file_exists` 指向
      fixture 本来就带的文件；`command_ok` 的命令不依赖 Agent 的任何动作就能成功。
- [ ] **Mutation check**：故意破坏正确结果（删掉文件 / 改坏内容 / 反转退出码），judge **必须失败**。
      对正确 fixture 通过、对至少一个错误变体失败，才算有效断言。
- [ ] judge 检查的是**最终产物**，不是中间过程。不允许出现「调用了正确工具即算成功」。
- [ ] judge **只检查局部结果**（如只断言某一行存在，而任务要求整个文件重写）→ 判为弱断言，需加强。
- [ ] 复合 `checks` 默认**全部通过才算成功**，没有漏写 `and` 变成「任一通过即可」。

### 8.3 沙箱与 fixture 隔离

- [ ] 每个用例**只操作自己的沙箱**，不读写其它用例的目录。
- [ ] **没有任何 fixture 被原地修改**。runner 复制 fixture 到临时沙箱后再运行；用例不得
      通过绝对路径回写 `evals/fixtures/`。
- [ ] 并发/并行用例（Multi-Agent、并行子任务）写入的文件路径**互不重叠**，或使用 **worktree 隔离**。
- [ ] 用例之间**不共享可变状态**（同一文件、同一端口、同一缓存目录、同一 `claude_dir`）。

### 8.4 命令与路径安全

- [ ] 命令判分使用**参数列表**（`["pytest", "-q"]`），**禁用 `shell=True`**。
- [ ] 只允许评测文件中**声明的受控命令**，不接受任意 shell 字符串。
- [ ] fixture 路径解析后（`resolve()`）**仍位于 `evals/fixtures/` 之下**，防止 `../` 逃逸。
- [ ] Safety 用例使用 **sentinel tool**，能证明被拒绝的调用未进入 `execute()`；**绝不运行真实危险命令**。

### 8.5 可复现性

- [ ] 用例可以**在全新沙箱独立运行**，不依赖执行顺序、不依赖前一条用例的残留。
- [ ] 用例**不依赖网络**（Web 走固定响应替身）与**真实时间/时区**。
- [ ] 用例声明的 `max_turns` 足够完成任务，且不会因过高而浪费额度。

---

## 9. 快速自检命令

```bash
# 旧用例仍可被现有 loader 加载（预期输出：30 10）
uv run --extra dev python -c "from pathlib import Path; from longline.eval.types import load_cases; print(len(load_cases(Path('evals/tool_calls.jsonl'))), len(load_cases(Path('evals/e2e.jsonl'))))"

# 新工具选择集：总量与盲测/回归拆分（预期输出：60 48 12）
uv run --extra dev python -c "from pathlib import Path; from longline.eval.types import load_cases; cs=load_cases(Path('evals/tool_selection.jsonl')); print(len(cs), sum('blind' in c.tags for c in cs), sum('instruction-following' in c.tags for c in cs))"

# 恢复集：展开后的运行数与 50/10 分母（预期输出：60 50 10）
uv run --extra dev python -c "from pathlib import Path; from longline.eval.recovery import load_recovery_cases; cs=load_recovery_cases(Path('evals/recovery.jsonl')); print(len(cs), sum(c.fault!='process_kill' for c in cs), sum(c.fault=='process_kill' for c in cs))"

# 多 Agent 集：总数与 controlled/exploratory 拆分（预期输出：24 18 6）
uv run --extra dev python -c "from pathlib import Path; from longline.eval.multi_agent import load_multi_agent_cases, group_of; cs=load_multi_agent_cases(Path('evals/multi_agent.jsonl')); print(len(cs), len(group_of(cs,'controlled')), len(group_of(cs,'exploratory')))"

# 循环恢复多样性集：展开后的运行数与逐臂计数（预期输出：100 25 25 25 25）
uv run --extra dev python -c "from pathlib import Path; from longline.eval.loop_resume import load_loop_resume_cases, cases_by_failpoint; cs=load_loop_resume_cases(Path('evals/loop_resume_tasks.jsonl')); g=cases_by_failpoint(cs); print(len(cs), *[len(g[a]) for a in ('before_tool','after_tool','after_checkpoint','truncate_tail')])"

# workspace 漂移规则本身（§5.11 的前后对照，在两个 revision 上各跑一次）
uv run --extra dev python evals/probes/workspace_drift_rule.py

# 评测单测（含泄漏检查与数据集契约）
uv run --extra dev pytest tests/unit/eval -q
```

---

## 相关文档

- 指标口径的业界出处见本文件末尾「附：指标口径的业界出处」
- 基线格式与冻结流程：`evals/baselines/README.md`

---

## 附：指标口径的业界出处

本节记录本文件各指标的方法论来源。**新增指标前先查这里的出处，不要凭感觉自创口径**；
若确实找不到业界对应物，要显式写明「本项目自定」并给出理由。

| 本项目的做法 | 业界出处 | 一致 / 偏离 |
|---|---|---|
| 全确定性判分，不用 LLM 判官作主判 | SWE-bench、Terminal-Bench、GAIA、WebArena、OSWorld、BFCL、API-Bank、τ-bench 全部如此 | **一致**。ToolBench 用 LLM 判官算 win-rate，是这一组里的异类 |
| 比例带分子/分母/95% Wilson CI | 主流榜单**都不报置信区间** | **领先**。代价：把任务当作 i.i.d. 伯努利抽样，而 40 条用例按类别相关，真实不确定性比区间显示的更大 |
| `pass@k` / `pass^k` 估计量 | τ-bench（Yao et al., arXiv:2406.12045） | **一致** |
| 每题 3 次运行、报均值与区间 | 计划 §3.2 自定（业界多用 pass@k 曲线） | **本项目自定**，故须同时报 `pass^k` 补上可靠性维度 |
| 按工具调用次数分层报通过率 | OSWorld/WebArena 按领域、GAIA 按难度分层（均为「分类别」而非逐步打分） | **一致**（形式不同，思路相同） |
| 弃权用例（正确答案是不调工具） | BFCL 约 25%（240 Irrelevance + 882 Live Irrelevance）、MetaTool 选择觉知 | **一致**（比例低于 BFCL，因为本套还测其他维度） |
| 工具调用的选择/参数/执行分开计 | BFCL：AST 评测（结构性匹配参数）vs 可执行评测（真跑再比返回值） | **一致**。注意 BFCL 明示 AST 对「语义等价但写法不同」的参数会误杀，故另有可执行变体 |
| 判最终产物 / 环境状态 | SWE-bench 跑 FAIL_TO_PASS+PASS_TO_PASS；τ-bench 比对**数据库末态**与目标态 | **基本一致**。差异：检索类用例我们判最终产物，**不判检索是否真的发生** |

### 已知局限（须与数字一同报告）

1. **无留出集 / 无作者盲测分片**，全部用例由作者编写，选择效应未经审计。
   注：Terminal-Bench 亦无私有分片并在其论文中自陈此局限，属**领域共性**。
2. **检索类用例判最终产物，不判检索过程**。产物可能正确而从未真正检索。
3. **参数指标是结构匹配**，对语义等价但写法不同的参数可能少给分（BFCL 的可执行变体是业界的解法）。
4. **等 N 分类是抽样选择，不是难度声明**。业界按**实测难度**分层（GAIA 分档、API-Bank 分能力级），
   而我们每类固定 8 条，**类内难度未经控制**。
5. **无「每解决一题的成本」指标**。SWE-bench 原论文也不报，但 Terminal-Bench 把
   性能-成本帕累托前沿作为主轴之一，BFCL 记录 cost/latency。若要声称效率优势，需补此指标。
   > **2026-09-20 补充**：§5.12 现在量了**恢复分支的成本**（模型调用 / 工具调用 / token /
   > 本地循环时间），但它量的是**同一题的两个分支之间**的差，**不是**「每解决一题的成本」。
   > 这条局限**仍然成立**：跨套件的 cost-per-task 依然没有，因此本项目的效率结论只能
   > 限定在「恢复 vs 重启」这一个决策上。
   > **2026-09-20 再补充**：§5.13 现在**定义了**「每解决一题的成本」
   > （`SuccessPer1KTokens`，两条臂分别报，分母为 0 报 `not measured` 而不是 0.0）。
   > 但**定义不等于有数**：`pair` 套件至今没有跑过真实 API，所以这一条从「没有这个指标」
   > 变成「有这个指标、还没有它的数字」。在拿到真实数字之前，**不得**声称任何成本优势。

### 5.13 配对收益：同一个任务两条路（2026-09-20）

#### 这一格要回答的问题

前几格量的是「运行时对不对」。这一格量的是**多 Agent 到底值不值它的成本**，而且要求两条路
做**同一份工作**：同一条用例、同一份 prompt 文本、同一组预期产物、同一个 merge 步骤、判分
用同一批评委。两侧的 fixture 是逐字节相同的兄弟目录（装载时校验 sha256），所以两条臂之间
**唯一**的差别是工作怎么被执行。

```bash
# 只看链路（免费、确定性）：不产生通过率结论
uv run --extra dev python -m longline.eval --suite pair --offline --run-id <id> --repeats 1
# 真跑（花钱）：必须显式确认
uv run --extra dev python -m longline.eval --suite pair --allow-paid --run-id <id> --repeats 3
```

`--suite pair` 不给 `--offline` 也不给 `--allow-paid` 时**直接拒绝启动**。理由是它失败的样子
不是一条报错，是一张账单。

#### 语料：18 条，三个 category，各 6 条

| category | 任务形状 | `workers` |
| --- | --- | --- |
| `parallel_analysis` | 6 个彼此独立的模块，各写一份摘要 | 4 |
| `parallel_modification` | 6 处彼此独立的修改，隐藏测试在判分时才落进沙箱 | 3 |
| `dependent` | 6 条**依赖型串行链**，后一步读前一步的产物 | 1（契约固定） |

#### 报告口径：按 category 分表，**不出总平均**

三个 category 是三种不同的任务，一个跨三类的平均回答的是「这个 6/6/6 的配比跑得怎么样」——
没人问过的问题，语料一重新配比它就变，而架构什么都没变。所以 `MultiAgentSummary` 在
`is_pooled` 为真时把 `pooled` 置为 **`None`**：报告不打印组头条，`summary.json` 里也拿不到
那个数。

`Speedup` **同时**报 mean 与 P50，两个数**必须来自同一个比值列表**（`metrics.speedup_block`
一个函数同时算）。mean 保留是为了能和 §5.6 冻结的离线报告对读；P50 是新增的，因为 per-case
比值的均值会被碰巧最小的那条用例拽走。

#### 指标

```text
Pass@1               判分通过的用例 / 用例数            (两条臂用同一批评委)
Speedup              single_wall_time / multi_wall_time  (1.00x 是持平)
TokenOverhead        (multi_tokens - single_tokens) / single_tokens
ToolCalls            两条臂各自的工具调用数
SuccessPer1KTokens   成功数 / (token / 1000)             (分母为 0 报 not measured)
```

#### 不能夸大的部分

1. **`--offline` 只验链路，不验通过率。** 离线协议是真实 `QueryEngine`、真实工具、真实
   `query_loop`、真实 `spawn_teammate`，只有模型是脚本化的。脚本模型按声明把文件写出来，
   所以**文件级**的检查两侧都过——**那个通过率不是结论**，它只说明装置能跑完。要看通过率
   必须真跑。

   实测（`--suite pair --offline --repeats 1`，18 条）：

   | category | single | multi | 失败在哪 |
   | --- | --- | --- | --- |
   | `parallel_analysis` | 6/6 | 6/6 | —— |
   | `parallel_modification` | 0/6 | 0/6 | 两侧都只挂 `python_test` |
   | `dependent` | 0/6 | 0/6 | 两侧都只挂 `python_test` |

   **两个 category 的失败是两侧对称的**：文件级检查（`file_exists`）全过，挂的都只有隐藏
   测试，因为脚本模型写的是 `# <subtask id>` 加一句指令的占位正文，满足不了检查真实内容与
   依赖顺序的隐藏测试。对称是关键——它说明这个 0 不是某一条臂的缺陷，而是**装置的性质**。
   也因此这张表**不能**读成「多 Agent 在修改型和依赖型任务上失败」。
2. **离线测不出 `Speedup`。** 多 Agent 唯一的加速机制是**重叠模型延迟**，而脚本模型没有
   延迟。离线的 `Speedup` 在数学上不可能 > 1，实测是 0.12–0.16x。§5.6 已经记过这一点。
3. **live 路径 2026-09-20 才接上，此前从未执行过。** 在这之前 `_apply_scripted_model` 是
   **无条件**调用的，`model=` 只影响 system prompt 文本，请求根本到不了 SDK。第一次真的调用
   它是在 Task 18 的付费门禁自检里，当场暴露出 `_apply_live_counting` 会把 `QueryEngine` 的
   **方法**当成工厂包起来（`TypeError`），以及更根本的一处：`QueryEngine` 的工厂闭包体就是
   `engine.make_call_model(...)`，包工厂的**产物**会递归。**截至本节写下时，本套件还没有产出
   任何一个真实 API 的数字。**
4. **这一格还没有数字。** §5.6 那 24 条冻结用例是**同一条 `task` 字符串**重复 18 次，
   它的 controlled 均值是关于一个模板的事实；本节的 18 条是另一种语料，两者不可互换。

---

### 5.14 协作可靠性：机器，不是 agent（2026-09-20）

#### 这一格要回答的问题

§5.13 问「扇出值不值」。这一格问**它底下的机器撑不撑得住**。不跑模型、不需要 key、
不花钱，所以这里可以做到穷尽，而 §5.13 必须省着来。

```bash
uv run --extra dev python -m longline.eval --suite collab --run-id <id>
```

用例声明在 `longline/eval/collab_cases.py` 里而不是 JSONL——一个用例在这里是一个场景
（发送者数、故障注入点、一对冲突的 writer、一棵 worktree），没有 `EvalCase` 的形状能承载。

#### 六个指标，其中两个**预期就是坏的**

| 指标 | 预期 | 说明 |
| --- | --- | --- |
| `MessageLossRate` | 0 | **阴性对照**，见下 |
| `DuplicateMessageRate` | 0 | 同上 |
| `InboxDurabilityLossRate` | **全丢** | 截断的文件按构造就读不出来；文件就是整个收件箱 |
| `InboxDurabilityLossReported` | **True** | 丢了消息**并且说出来**才算有耐久上限 |
| `OrphanTaskRate` | 实测 | 跑完了但结果没人取 |
| `CrossWorktreeLeakRate` | **1.0** | `isolation="worktree"` 建了 worktree 却不告诉子 agent |
| 冲突处理 | 四个独立计数 | injected / detected / silent overwrite / integrated |

**「预期是坏的」这两条才是这一节的价值**：它们把一句架构声明（「Git worktree 隔离」）变成
一条发现（「隔离没有生效」），而把预期**事先**写下来，是让结果事后无法被解释掉的办法。

#### `MessageLossRate` 是阴性对照，不是抓到的 bug

`TeammateMailbox` 的自述里警告并发写会丢消息。**那条警告在这个运行时里够不着**：
`send()` 从 `_read_inbox` 到 `_write_inbox` 全程同步、中间没有 `await`，而 teammate 是同一个
事件循环上的 `asyncio` task；单线程循环里同步函数体不可被抢占。所以读-改-写是**构造上原子**的。

这条测试因此**确认一个设计假设**，不是猎 bug。把它当 bug 猎，会让人在找不到 bug 的时候去改
实现。每个 sender 发完一条就 `await asyncio.sleep(0)` 让出一次，这不是装饰：`send` 自己从不
`await`，不让出的话第一个协程会跑完才轮到第二个，「没丢消息」就成了一条关于「测试里从来没有
两个写者」的陈述。`peak_concurrent_sends` 记录这个重叠真的发生过。

**如果这条测试哪天开始丢消息**，说明「同进程单事件循环」这个前提被别的东西打破了——最可能是
某个 teammate 被挪到了线程或进程上——**那才是要查的东西**。

#### 收件箱耐久性：修复买到的是「报出来」，不是「不丢」

| revision | truncate | delivered | survived | loss_rate | reported |
| --- | --- | --- | --- | --- | --- |
| 修复前 | 是 | 8 | 0 | 1.0 | **False** |
| 修复后 | 是 | 8 | 0 | 1.0 | **True** |
| 修复后（对照） | 否 | 8 | 8 | 0.0 | False |

**损失量没有变，两次都是 8/8。** 修复买到的是一行「投递 8 条、一条都没读回来」，而不是
「收件箱是空的」。后者**没有症状**：teammate 看到「没有消息」继续干完活并报成功，leader 收到
一份没提丢失工作的回复——双方都报成功而消息没了。原子写（temp + `os.replace`）防的是**半份
文件被生产出来**，由 `tests/unit/swarm/test_mailbox.py` 的性质测试锁住；两者都**不恢复字节**。

`truncate=False` 那一行是对照：一条只会观测损坏情形的测试，分不出「reader 报出了损失」与
「reader 永远报损失」。

#### 孤儿任务率：写侧与读侧，缺一不可

- **写侧**是 `TaskRegistry`：`spawn_teammate` 逐个注册，done-callback 把记录推到终态。
  这是**运行时的**事实，由 spawn 路径产生。
- **读侧**是 leader：那条回复有没有进它的收件箱，以及它有没有读过。

一个 teammate 可以 COMPLETED 而结果没人取——**那个缺口才是这个指标**。只测一侧，得到的要么是
「都跑完了」（写侧），要么是「我看见了我看见的」（读侧）。

`OrphanTaskRate = (completed - consumed) / completed`，**分母是 COMPLETED 而不是 SPAWNED**：
没跑完的 teammate 不可能有被孤立的回复，算进去会把崩溃记成孤儿。

实测最值钱的一格：把 `TeammateMailbox.send` 改成抛错，三个 teammate **全部报 COMPLETED**、
`errors` 为空、收件箱是空的——`OrphanTaskRate` 单独用分不出这一格和「leader 不读信」那一格
（都是 1.0）。所以 `delivered` 必须单独报：`completed - delivered` 是运行时没投出去，
`delivered - consumed` 是投出去了没人取。**两种故障，修法不同。**

#### worktree 隔离：实测未生效，越界率 100%

`longline/tools/agent/worktree.py` 的契约第 2 条写明「子 agent 在 worktree 目录中执行所有操作」。
`agent_tool.py` 从未做到：`worktree_path` 全文只有**声明**、**创建**、**删除**三处，中间没有
任何一处接到子 agent 上；`query_loop` 没有 `cwd` 参数；`child_registry` 注册的是父 agent 的
工具实例，而工具按**进程 cwd** 解析相对路径。

于是 `isolation="worktree"` 的实际语义是：**建一棵 worktree，子 agent 在父目录里干完活，
把那棵空的 worktree 删掉。**

实测 K=3：3 个 marker 全部落在父仓库，仓库 `git status --porcelain` 从空变成 3 个未跟踪文件，
残留 worktree 0 棵。两个观测合起来才关得死——`leaked=3` 说文件落在父仓库，`残留=0` 说
worktree 被拆掉时是空的（`cleanup_agent_worktree` 会**保留**任何有未提交改动的 worktree）。

**这条测量不是自证的**：把 `create_agent_worktree` 改成建完就 chdir 进去（等价于真把 cwd 串
下去），越界数从 3 变 0、落在 worktree 内从 0 变 3、残留从 0 变 3，三个量同向翻转。

测试断言的是**实测行为**而不是意图，所以它会在有人真的把 cwd 串下去的那天**失败**——那个
失败是改结论的信号，不是测试写错了。修复（per-tool cwd 重构：动 Read/Write/Edit/Glob/Grep/Bash
六个工具与调用链）**明确出本轮范围**。

#### 冲突处理：`Write` 与 `Edit` 往相反方向失败

| 形状 | injected | detected | silent_overwrite | integration_ok |
| --- | --- | --- | --- | --- |
| `Edit`（有前置条件） | 2 | **1** | **0** | True |
| `Write`（全量覆盖） | 2 | **0** | **1** | True |
| `Write` 单 writer（对照） | 1 | 0 | 0 | True |

`Edit` 带着 `old_string` 这个**前置条件**，所以第二个 writer 拿到 `old_string not found`——
冲突被接住，代价是一个失败的任务。`Write` 是全量覆盖、**没有前置条件**，于是两个 call 都返回
成功，最终文件只剩后写的那个，**没有任何一处报过冲突**。把 `Edit` 换成 `Write`，`detected`
从 1 掉到 0、`silent_overwrite` 从 0 升到 1，而对上层而言两者都「成功」。

`final_integration_success` 是**护栏不是结论**：两个形状都是 True，且是构造上必然 True
（两个工具都走 `os.replace` 原子写，半份文件活不下来）。报它是因为「有人发现了冲突」与
「工作区仍然自洽」是两句不同的话。

#### 不得夸大的部分

1. **这一节测的是机器，不是 agent 的注意力。** leader 是**脚本驱动**的，所以 `consumed`
   反映的是运行时**投递链路**是否闭合，**不是**真模型会不会去读收件箱。`OrphanTaskRate=0`
   只说明链路通，不说明模型从不漏掉子任务。
2. **冲突不是竞态，是丢失更新。** 两个 writer 顺序执行而非交错——两个工具的 `execute` 在
   「读文件」与「写文件」之间都没有 `await`，单事件循环里本来就交错不了（与 mailbox 同一条
   理由）。测的是第二次写基于一个**已经过期的读**。并发不改变答案：`Edit` 的前置条件挂在
   字节上，`Write` 根本没有前置条件。
3. **`CrossWorktreeLeakRate=1.0` 是本项目当前的实际状态**，不是这套装置的问题，也不是一条
   已经修好的历史记录。它要求在简历、README 和任何对外描述里**不能**写成「Git worktree 隔离」。
4. **计数刻意很小**（4 个 sender、8 条消息、3 个 teammate）。丢一条消息是**结构性失败**，
   不是需要大样本才能测出的比率。

---
