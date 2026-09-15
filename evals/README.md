# Agent Runtime 评测：指标契约与数据说明

> 冻结日期：2026-09-15
> 上游计划：`docs/superpowers/plans/2026-09-15-agent-runtime-metrics.md`

本文件是**指标契约（metric contract）**：所有正式数字的公式、分子、分母、排除条件和命名，
以本文件为唯一准绳。任何报告、README 或简历数字若与本文件冲突，以本文件为准。

计划文档中写的是 `cc/eval/`，本仓库真实包名是 `longline/`，两者一一对应
（`cc/eval/` → `longline/eval/`）。本文档按真实路径书写。

---

## 0. 本目录清单

| 路径 | 状态 | 说明 |
|---|---|---|
| `evals/tool_calls.jsonl` | **legacy** | 30 条工具调用用例。instruction-following 数据，**不进入简历主数字**。 |
| `evals/e2e.jsonl` | **legacy** | 10 条端到端用例。仅为历史基线，**不进入简历主数字**。 |
| `evals/fixtures/` | 生效 | 沙箱起始状态。 |
| `evals/results/` | 生效 | 运行产物，布局见 §3。 |
| `evals/baselines/` | 生效 | 冻结基线，格式见 `evals/baselines/README.md`。 |

两个 `.jsonl` 已通过逐行 `tags` 数组追加 `"legacy"` 标记（`tags` 本就是
`longline/eval/types.py` 支持的字段，`from_dict` 对未知顶层键也会忽略，因此加载器行为不变）。
标记后经 `load_cases()` 实测仍返回 **30 / 10** 条。

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

> **注意**：上述哈希对应**打 legacy 标记之前**的原始文件。打标记后两个文件分别为 7673 / 2749 字节，
> 用例内容除 `tags` 新增 `"legacy"` 外**完全未变**，`load_cases()` 仍返回 30 / 10。
> 本表数值作为「冻结时的原始数据指纹」保留。

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
      > 注：现有 legacy 数据中 **10 条**（`tc-003`、`tc-004`、`tc-006`、`tc-011`、`tc-012`、
      > `tc-015`、`tc-018`、`tc-019`、`tc-028`、`tc-029`）直接点名工具
      > （如「用 grep…」「用 glob…」「都用 Read」），正是反例，故整体标为 legacy。
      > `e2e.jsonl` 未发现工具名泄漏。
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

# 评测单测
uv run --extra dev pytest tests/unit/eval -q
```

---

## 相关文档

- 上游计划（指标定义来源）：`docs/superpowers/plans/2026-09-15-agent-runtime-metrics.md`
- 基线格式与冻结流程：`evals/baselines/README.md`
