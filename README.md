# Longline

面向长链路任务的可恢复 Agent Runtime 与多 Agent 执行框架。

## 这是什么

一个纯 Python 实现的 agent 内核，解决的问题是：**当一次任务要跑几十轮工具调用、几十分钟、甚至跨进程重启时，怎么保证它不丢状态、能恢复、能被人接管。**

- **可恢复**：session 持久化 + transcript 校验修复，进程挂了能从断点续上
- **长链路**：token 预算监控 + 自动上下文压缩，长对话不会撑爆窗口
- **多 Agent**：Team 生命周期、Mailbox 通信、Coordinator 编排，一个 leader 带多个 teammate 并行干活

内核源自对 Claude Code TypeScript 源码（1884 个文件、38 万行）的提取与翻译，因此它**不是封装 API 的 wrapper，而是一个功能对齐的完整 agent 运行时**——22 个内置工具、权限系统、hooks、MCP、skills 都是照着真实产品还原的。

## 还原了什么

| 能力 | 状态 | 说明 |
|------|------|------|
| Agent Loop（状态机） | ✅ | 多轮 tool-use 循环，流式响应，错误恢复，自动重试 |
| 流式工具执行 | ✅ | 工具在 API 流式过程中立即开始执行，不等响应结束 |
| 22 个内置工具 | ✅ | Bash、Read、Edit、Write、Glob、Grep、Agent、WebFetch、WebSearch、NotebookEdit、ToolSearch、AskUser、Task 系列、TodoWrite、Skill、PlanMode、Brief、LSP、TeamCreate/Delete、SendMessage |
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
├── tools/              22 个工具实现 + StreamingToolExecutor + 权限门控
├── permissions/        权限系统：PermissionMode + 规则引擎 + 非交互语义
├── swarm/              Agent Teams：身份、Mailbox、TeamFile、InProcessTeammate、Coordinator
├── compact/            上下文压缩：token 预算监控、摘要生成
├── memory/             记忆系统：加载/保存/提取/索引/ExtractionCoordinator
├── mcp/                MCP 协议：stdio 客户端、工具桥接
├── hooks/              Hooks：配置加载、PreToolUse/PostToolUse
├── skills/             Skills：定义加载、slash 命令注册
├── session/            会话管理：持久化、TaskRegistry、transcript recovery
├── commands/           Slash 命令：/clear /compact /model /help /cost
├── ui/                 终端渲染：Rich 流式输出
└── main.py             入口：REPL 循环、模块组装、inbox polling

tests/                  498 个测试用例
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
# 全量单元测试（498 个）
uv run pytest tests/unit/ -v

# 集成测试（需要 API key + 网络）
uv run pytest tests/integration/ tests/e2e/ -v

# 静态检查
uv run ruff check longline/ tests/
uv run mypy longline/
```

### Agent 评测

内置两层确定性评测（不依赖 LLM 主观打分）：

- **工具调用准确率**：`evals/tool_calls.jsonl`（30 条）校验 Agent 是否按序选择正确工具、参数是否匹配正则（Read/Grep/Glob/Write/Edit/Bash）。
- **端到端任务成功率（pass@1）**：`evals/e2e.jsonl`（10 条）在临时沙箱内跑真实任务，用文件内容/命令退出码判分。

实测（deepseek-v4-flash，均多次运行取稳定区间）：工具调用准确率 **~97–100%**（平均 2.7 轮），E2E pass@1 **~70–90%**（平均 3.4 轮）。报告含逐用例工具轨迹（tool_calls）便于失败归因。

运行（需要 anthropic 兼容 API key：`ANTHROPIC_API_KEY`/`OPENCODE_API_KEY`，OpenCode 网关经 `OPENCODE_BASE_URL_GO` 自动适配）：

```bash
# 全量两层
uv run python -m longline.eval --case-file evals/tool_calls.jsonl --md
# 只跑 e2e，便宜模型，冒烟前 3 条
uv run python -m longline.eval --type e2e --case-file evals/e2e.jsonl --model deepseek-v4-flash --max-cases 3
```

报告输出到 `evals/results/`（JSON + 可选 Markdown）。离线单测不消耗 API：

```bash
uv run pytest tests/unit/eval/ -v
```

## 交流

对 CC 源码还原或 Agent Runtime 感兴趣可以扫二维码，进 CC 讨论群：

<img src="assets/wechat.png" width="300">

# end
