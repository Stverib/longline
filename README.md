# Longline

面向长链路复杂任务、以原生 Python/asyncio 独立实现的可恢复 Agent Runtime 与多 Agent 执行框架。

## 这是什么

Longline 解决的问题是：**当一次任务要跑几十轮工具调用、几十分钟、甚至跨进程重启时，怎么保证它不丢状态、能恢复、能被人接管。**

- **可恢复**：session 持久化 + transcript 校验修复，进程挂了能从断点续上
- **长链路**：token 预算监控 + 自动上下文压缩，长对话不会撑爆窗口
- **多 Agent**：Team 生命周期、Mailbox 通信、Coordinator 编排，一个 leader 带多个 teammate 并行干活

约 2.1 万行代码（不含空行与注释），覆盖 Agent Loop、26 个内置工具、MCP/Skills、Context Engineering、会话恢复与多 Agent 编排，并自带一套确定性评测体系（见 [Agent 评测](#agent-评测)）。

它没有 Agent 框架依赖：工具编排时序、权限门控、上下文压缩时机、故障注入点都落在自己的代码里，可以直接下断点追到具体状态转换，而不是停在框架回调里——**它不是封装 API 的 wrapper，而是一个完整的 agent 运行时**。

**设计参考**来自对 Claude Code Runtime 的源码分析与架构抽象：能力边界（工具集、权限模型、hooks、MCP、skills）与真实产品对齐，实现与模块划分由本项目独立完成。

## 能力覆盖

| 能力 | 状态 | 说明 |
|------|------|------|
| Agent Loop（状态机） | ✅ | 多轮 tool-use 循环，流式响应，错误恢复，自动重试 |
| 流式工具执行 | ✅ | 工具在 API 流式过程中立即开始执行，不等响应结束 |
| 26 个内置工具 | ✅ | Bash、Read、Edit、Write、Glob、Grep、Agent、WebFetch、WebSearch、NotebookEdit、ToolSearch、AskUser、Task 系列（Create/Get/List/Update/Stop）、TodoWrite、Skill、PlanMode（Enter/Exit）、Brief、LSP、TeamCreate/Delete、SendMessage |
| 权限系统 | ✅ | PermissionMode（bypass/acceptEdits/default）+ 规则引擎 + 非交互 fail-fast |
| System Prompt 体系 | ✅ | 多段动态拼装，含 Memory 行为指导 + Coordinator/Teammate 提示词 |
| CLAUDE.md 加载 | ✅ | 目录层级遍历 + `@include` 递归展开 |
| 自动上下文压缩 | ✅ | Token 预算监控，超限自动 compact，长对话不崩 |
| Memory 系统 | ✅ | 四类记忆分类、MEMORY.md 索引自动更新、后台 coalescing 提取 |
| MCP 协议支持 | ✅ | stdio 传输、动态工具注册、多 server 并行 |
| 工具编排引擎 | ✅ | 并发/串行分批、流式执行器、hooks、权限门控 |
| Hooks 系统 | ✅ | PreToolUse/PostToolUse 拦截，shell 命令执行 |
| Skills 系统 | ✅ | frontmatter 定义、slash 命令触发、prompt 注入 |
| Session 持久化 | ✅ | 会话保存/恢复、Task 状态快照、transcript 校验修复 |
| QueryEngine | ✅ | 统一 runtime owner，封装 client/model/registry/prompt/permissions |
| Agent Teams / Swarm | ✅ | 多 Agent 协调：Teammate 执行引擎、Mailbox 通信、Coordinator 编排、Team 生命周期 |
| REPL + Print 模式 | ✅ | 交互式循环 + 单次管道模式 |

## 架构

```
longline/
├── core/               QueryEngine 统一入口、query_loop 状态机、事件流
├── api/                Anthropic API：流式调用、客户端管理、token 统计
├── models/             数据模型：消息类型、content blocks、API 规范化
├── prompts/            System Prompt：多段文本 + 动态拼装 + Coordinator/Teammate 提示词
├── tools/              26 个工具实现 + StreamingToolExecutor + 权限门控
├── permissions/        权限系统：PermissionMode + 规则引擎 + 非交互语义
├── swarm/              Agent Teams：身份、Mailbox、TeamFile、InProcessTeammate、Coordinator
├── compact/            上下文压缩：token 预算监控、摘要生成
├── memory/             记忆系统：加载/保存/提取/索引/ExtractionCoordinator
├── mcp/                MCP 协议：stdio 客户端、工具桥接
├── hooks/              Hooks：配置加载、PreToolUse/PostToolUse
├── skills/             Skills：定义加载、slash 命令注册
├── session/            会话管理：持久化、TaskRegistry、transcript recovery
├── eval/               评测子系统：8 个套件、判分器、报告、故障注入与 failpoint 门控
├── commands/           Slash 命令：/clear /compact /model /help /cost
├── ui/                 终端渲染：Rich 流式输出
└── main.py             入口：REPL 循环、模块组装、inbox polling

tests/                  1551 单元测试 + 33 集成/E2E（其中评测单测 1028 个，全离线）
```

### 核心数据流

```
用户输入
  → main.py 追加到 messages（transcript）
  → [inbox polling] 如果在 team 中，检查 leader 收件箱，注入 <task-notification>
  → QueryEngine.run_turn() → query_loop 发送到 Claude API
  → 流式接收：TextDelta / ToolUseBlock / TurnComplete
  → StreamingToolExecutor：工具在流式过程中立即开始执行（并发安全/hooks/权限检查）
  → tool_result 塞回 transcript → 再调 API
  → 循环直到 end_turn
  → 渲染输出，保存 session + task snapshot
  → ExtractionCoordinator 后台提取记忆
  → 等待下一轮输入
```

一次用户输入可以触发多轮模型调用和多次工具执行——这就是 agent，不是 chatbot。

### Agent Teams 数据流

```
用户: "帮我重构这个模块"
  → Leader (REPL) 调用 TeamCreate 创建团队，进入 team context
  → Leader 调用 AgentTool(name="researcher", run_in_background=true)
      → spawn_teammate → InProcessTeammate 启动
      → Teammate 使用独立 query_loop（非交互权限、工具过滤）
      → 完成后通过 Mailbox 发送结果给 leader
  → Leader 下一轮 turn 前，inbox polling 检查收件箱
  → <task-notification> 注入 prompt → Leader 根据结果决策下一步
  → Leader 调用 TeamDelete 清理
```

## 快速开始

### 环境要求

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

### 安装

```bash
cd longline
uv sync
```

### 配置 API Key

```bash
# 环境变量
export ANTHROPIC_API_KEY=sk-ant-...

# 或项目 .env 文件
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

#### 使用阿里云百炼（可选）

支持通过阿里云百炼的 Anthropic 兼容接口调用千问、GLM、Kimi 等国产模型。只需在 `.env` 中额外配置百炼 API Key：

```bash
# .env 文件（可同时配置两个 key，运行时按模型自动切换）
ANTHROPIC_API_KEY=sk-ant-...
DASHSCOPE_API_KEY=sk-your-dashscope-api-key
```

运行后用 `/model` 查看可用模型并按序号切换：

```
> /model
Available models (use /model <number> to switch):
  * 1. claude-sonnet-4-20250514
    2. claude-opus-4-20250514
    3. claude-haiku-4-5-20251001
    4. qwen3-max
    5. glm-5
    6. kimi-k2.5

> /model 4
Model changed to: qwen3-max
```

切换到百炼模型时会自动使用 `DASHSCOPE_API_KEY` 和百炼 endpoint，切回 Claude 模型时自动恢复。

### 启动

```bash
# REPL 交互模式
uv run python -m longline

# 单次问答（管道模式）
echo "用 Python 写一个快排" | uv run python -m longline -p

# 指定模型
uv run python -m longline --model claude-haiku-4-5-20251001

# 恢复会话
uv run python -m longline -c <session-id>

# Coordinator 模式（多 agent 编排）
LONGLINE_COORDINATOR_MODE=1 uv run python -m longline
```

### 测试

```bash
# 全量单元测试（1551 个，不需要 API key）
uv run pytest tests/unit/ -v

# 集成测试（需要 API key + 网络）
uv run pytest tests/integration/ tests/e2e/ -v

# 静态检查
uv run ruff check longline/ tests/
uv run mypy longline/
```

### Agent 评测

评测是这个仓库的一等子系统，不是事后补的脚本：**8 个套件、全确定性判分、指标可从原始产物离线重算、付费运行可断点续跑**。
正式口径（公式、分子分母、排除条件、已知局限、业界出处）以 [`evals/README.md`](evals/README.md)
为唯一准绳，本节只做导航——**任何报告或简历数字若与该文件冲突，以该文件为准**。

| 套件 | 规模 | 测什么 |
|---|---|---|
| `tool_selection` | 68 条（56 盲测 + 12 指令跟随，含 8 条弃权） | 工具选择：48 条工具族用例，外加「正确答案是不调工具」的弃权类 |
| `e2e` | 40 条（5 类 × 8） | 端到端任务成功率 `pass@1`，以及可靠性 `pass^k` |
| `compression` | 20 条成对 A/B | 自动压缩：压缩比率、关键信息保留、压缩后成功率 |
| `recovery` | 6 类故障 × 10 次 = 60 run | 进程崩溃后的会话恢复 |
| `latency` | 3 场景 × buffered/streaming | 流式工具执行相对缓冲执行的时长收益（成对） |
| `multi_agent` | 24 条（18 受控 + 6 探索） | 单 Agent vs 多 Agent 成对对比 |
| `safety` | 30 条（15 危险 + 15 正常） | 权限门控的 `DangerousRecall` 与 `FalsePositiveRate` |
| `loop_resume` | 6 个故障点 × 10 次 = 60 run | 在真实 agent loop 内部被杀之后的恢复语义与副作用重复 |

方法论——下面每条都有对应实现，不是口号：

- 全确定性判分，不用 LLM 判官作主判；判**最终产物**与退出码，不接受「调用了正确工具即算成功」
- 每个比例都带**分子 / 分母 / 95% Wilson CI**；每题 3 次运行报均值与区间，**绝不取最好一次**
- `pass^k` 补上可靠性维度；行为指标（`DedicatedToolPreferenceRate`、`FirstActionConsistency`、稳定性分类）与通过率**分开报**
- 成对实验（压缩 / 流式 / 多 Agent）在同一任务内做 `off/on`、`buffered/streaming`、`single/multi` 对比，**不留绝对基线**
- 失败样本逐条归因到 model / runtime / tool / judge / fixture / infra
- Review Checklist 是可执行的：标签泄漏扫描、judge mutation check（故意破坏正确结果必须判失败）、沙箱隔离、路径逃逸防护

契约里还写死了几条**报告不得越过的红线**，其中两条是评测自己查出来的：

- 恢复能力指**会话加载 + transcript 修复 + Task 快照**；非终态后台任务恢复后会被标记为 `KILLED`，**不得**表述为「后台任务原地续跑」
- 生产**没有任何工作区身份校验**，`WorkspaceDriftDetectionRate` 的诚实结果是 **0**，由一条钉住该事实的测试守着；它开始失败时，文档措辞必须跟着改

`loop_resume` 得出的最有价值结论与模型无关：**checkpoint 只在指令边界落盘，一条指令执行期间没有任何中途持久化**，
因此中断落在指令内部会丢掉整条指令的全部工作——不只是「副作用没去重」。这也是本项目文档里「恢复」一词只承诺会话加载、transcript 修复与 Task 快照的原因。

离线部分零 API 花费：

```bash
# 1028 个评测单测，含泄漏检查与数据集契约
uv run pytest tests/unit/eval -q

# 把杀点注入真实 agent loop，父进程真的杀掉子进程，测恢复语义
uv run pytest tests/integration/test_eval_loop_resume.py -q
```

付费部分需要 anthropic 兼容 API key（`ANTHROPIC_API_KEY`，或经 OpenCode 网关的 `OPENCODE_API_KEY`，
网关地址由 `.env` 的 `OPENCODE_BASE_URL_GO` 自动适配）：

```bash
# 按套件跑，3 次重复，产物落 evals/results/<run-id>/
uv run python -m longline.eval --suite tool_selection --repeats 3 --run-id <id> --md

# 冒烟：只跑前 3 条
uv run python -m longline.eval --suite e2e --max-cases 3

# 断点续跑被打断的长跑
uv run python -m longline.eval --resume-run --run-id <id>
```

产物布局 `<out-dir>/<run-id>/{raw.jsonl,summary.json,report.md}`。**`raw.jsonl` 是唯一数据源**，
`report.md` 不能单独作为基线依据；基线冻结流程与回归门槛见
[`evals/baselines/README.md`](evals/baselines/README.md)。当前 `evals/baselines/` **尚无冻结基线**，
所以现阶段只能报绝对值，还不能判回归。

#### 历史数字（legacy exploratory，勿引用）

模型 `deepseek-v4-flash`：工具调用准确率 **~97–100%**（平均 2.7 轮），E2E pass@1 **~70–90%**（平均 3.4 轮）。
区间式表述说明它们是「多次运行取稳定区间」的目测结果——**没有分子/分母、没有 95% CI、没有 Git SHA**，
且所用工具用例存在标签泄漏（任务文本直接点名工具）。数字保留仅为记录历史，定性见 `evals/README.md` §7。
