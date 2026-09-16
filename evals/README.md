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

**验收对照**：`query_loop` 的重试上限与截断恢复上限改为参数（默认即原常量），
`disable_recovery=True` 时传 0，因此**关闭恢复策略后对应故障不再恢复**——
这是计划 §4.4 验收条件的可执行形式，两个方向都有测试。

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
