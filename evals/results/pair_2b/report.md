# Agent Evaluation Report

- **Model served:** not reported by the transport
- **Total cases:** 36
- **Tool-call accuracy (L1):** not measured (0/0) 95% Wilson CI n/a
- **E2E pass@1 (L2):** not measured (0/0) 95% Wilson CI n/a
- **E2E pass^k:** n/a (needs --repeats >= 2)
- **Tool execution success:** not measured (0/0)
- **Averages:** turns=22.47, in_tok=7982, out_tok=4228
- **Latency (ms):** mean=23457.2ms, p50=21614.3ms, p95=47525.3ms
- **Tokens:** prompt=287342 (uncached=287342, cached=0), output=152198
- **Token efficiency:** 12209.4 tok/case, 12558.3 tok/success, n/a in-tok/call

## Single-agent vs multi-agent (paired A/B)

> `Speedup` is `single_wall_time / multi_wall_time` and `TokenOverhead` is `(multi - single) / single` -- both **ratios**, never differences in percentage points (contract §5.6 / §4.3). `pp` is reserved for success rates. Tokens include **every** sub-agent, not only the leader. `Speedup` is a multiplier where **1.00x is parity**: below 1.00x means the multi-agent arm was SLOWER, which is the usual offline result.

### Group: `controlled`

> Controlled cases pre-declare their independent subtasks, so both variants do the same work. Exploratory cases let the coordinator decompose freely and are **not** comparable with them.

- **Tasks:** 6 (18 case-runs), 18 eligible, 0 excluded (a variant did not complete or its token accounting did not reconcile)
- **Agent counts (multi arm):** [4, 5]

> **The pooled figure is withheld.** These cases span more than one CATEGORY of task, and a single mean over them would describe the corpus mix rather than the architecture -- it would move if the mix were rebalanced, with nothing about the agent having changed. Read the per-category rows below; there is deliberately no all-categories row.

| category | n | Success single | Success multi | Speedup mean | Speedup P50 | TokenOverhead | succ/1k tok single | succ/1k tok multi |
|---|---|---|---|---|---|---|---|---|
| `dependent` | 2 (6 runs) | 100.0% (6/6) | 100.0% (6/6) | 0.29x (3.5x slower) | 0.27x (3.7x slower) | +215.0% | 0.194 | 0.062 |
| `parallel_analysis` | 2 (6 runs) | 100.0% (6/6) | 100.0% (6/6) | 0.39x (2.6x slower) | 0.32x (3.1x slower) | +296.3% | 0.198 | 0.051 |
| `parallel_modification` | 2 (6 runs) | 83.3% (5/6) | 100.0% (6/6) | 0.70x (1.4x slower) | 0.64x (1.6x slower) | +130.6% | 0.100 | 0.053 |

### Per-case (this group)

| case | workers | single pass | multi pass | single ms | multi ms | speedup | single tok | multi tok | multi child tok | excluded |
|---|---|---|---|---|---|---|---|---|---|---|
| pa-001 | 4 | True | True | 14612.1ms | 21777.5ms | 0.67x (1.5x slower) | 6523 | 17677 | 14638 | - |
| pa-001 | 4 | True | True | 10079.9ms | 22434.7ms | 0.45x (2.2x slower) | 5197 | 16811 | 13715 | - |
| pa-001 | 4 | True | True | 8826.1ms | 26611.7ms | 0.33x (3.0x slower) | 5040 | 19118 | 16042 | - |
| pa-002 | 4 | True | True | 8661.2ms | 27251.1ms | 0.32x (3.1x slower) | 5276 | 24852 | 21058 | - |
| pa-002 | 4 | True | True | 6512.4ms | 25784.3ms | 0.25x (4.0x slower) | 4228 | 19287 | 15855 | - |
| pa-002 | 4 | True | True | 7107.4ms | 23055.3ms | 0.31x (3.2x slower) | 4095 | 19530 | 17110 | - |
| pm-001 | 3 | True | True | 22616.9ms | 26792.2ms | 0.84x (1.2x slower) | 8851 | 21280 | 17689 | - |
| pm-001 | 3 | True | True | 21451.2ms | 20263.0ms | 1.06x faster | 9081 | 15179 | 13533 | - |
| pm-001 | 3 | False | True | 17969.2ms | 29552.6ms | 0.61x (1.6x slower) | 7679 | 18749 | 14309 | - |
| pm-002 | 3 | True | True | 16253.1ms | 35713.2ms | 0.46x (2.2x slower) | 8251 | 24390 | 19803 | - |
| pm-002 | 3 | True | True | 18782.9ms | 35583.9ms | 0.53x (1.9x slower) | 7474 | 18140 | 15212 | - |
| pm-002 | 3 | True | True | 19834.0ms | 29183.6ms | 0.68x (1.5x slower) | 8426 | 16288 | 11613 | - |
| pd-001 | 1 | True | True | 12625.6ms | 46886.6ms | 0.27x (3.7x slower) | 5281 | 20211 | 16443 | - |
| pd-001 | 1 | True | True | 12607.1ms | 49441.4ms | 0.25x (3.9x slower) | 4850 | 14948 | 10225 | - |
| pd-001 | 1 | True | True | 12531.0ms | 46709.6ms | 0.27x (3.7x slower) | 4847 | 15048 | 10566 | - |
| pd-002 | 1 | True | True | 13614.8ms | 38405.0ms | 0.35x (2.8x slower) | 5618 | 18296 | 14474 | - |
| pd-002 | 1 | True | True | 12766.9ms | 50628.7ms | 0.25x (4.0x slower) | 4987 | 16143 | 11636 | - |
| pd-002 | 1 | True | True | 12889.1ms | 38643.9ms | 0.33x (3.0x slower) | 5273 | 12616 | 8698 | - |

## Run stability

Diagnostic, never a pass condition. `mixed` is the set this round tries to move, split by cause: `routing` can be moved by a tool policy, `overrun` by acting on sufficient information, and `content_driven` by neither.

| kind | cases |
|---|---|
| stable (passed every repeat) | 11 |
| mixed (passed sometimes) | 1 |
| always_fail (passed never) | 0 |

- **First-action consistency:** {'stable': 0, 'mixed': 0, 'highly_unstable': 0}
- **Redundant actions:** {'repeated_reads': 0, 'read_after_write': 0}
- **Notebook edited via Edit instead of NotebookEdit:** 0 run(s)

| mixed case | variant | cause | first divergence |
|---|---|---|---|
| pm-001 | single_agent | content_driven | - |

## By category

| category | pass | turns | tool_calls | in_tok | out_tok | p50 ms | failure types |
|---|---|---|---|---|---|---|---|
| multi-agent | 97.2% (35/36) | 22.47 | 0.00 | 7982 | 4228 | 23457.2ms | - |

| case | type | passed | ms | turns | in_tok | out_tok | error_type |
|------|------|--------|----|-------|--------|---------|------------|
| pa-001 | multi_agent | True | 14612.1ms | 8 | 4669 | 1854 | - |
| pa-001 | multi_agent | True | 21777.5ms | 34 | 11028 | 6649 | - |
| pa-001 | multi_agent | True | 10079.9ms | 6 | 3765 | 1432 | - |
| pa-001 | multi_agent | True | 22434.7ms | 33 | 10674 | 6137 | - |
| pa-001 | multi_agent | True | 8826.1ms | 5 | 3532 | 1508 | - |
| pa-001 | multi_agent | True | 26611.7ms | 37 | 11824 | 7294 | - |
| pa-002 | multi_agent | True | 8661.2ms | 5 | 3842 | 1434 | - |
| pa-002 | multi_agent | True | 27251.1ms | 43 | 17856 | 6996 | - |
| pa-002 | multi_agent | True | 6512.4ms | 4 | 3202 | 1026 | - |
| pa-002 | multi_agent | True | 25784.3ms | 40 | 12412 | 6875 | - |
| pa-002 | multi_agent | True | 7107.4ms | 4 | 3041 | 1054 | - |
| pa-002 | multi_agent | True | 23055.3ms | 41 | 12355 | 7175 | - |
| pm-001 | multi_agent | True | 22616.9ms | 12 | 5674 | 3177 | - |
| pm-001 | multi_agent | True | 26792.2ms | 39 | 14699 | 6581 | - |
| pm-001 | multi_agent | True | 21451.2ms | 12 | 6107 | 2974 | - |
| pm-001 | multi_agent | True | 20263.0ms | 35 | 9139 | 6040 | - |
| pm-001 | multi_agent | False | 17969.2ms | 12 | 5369 | 2310 | - |
| pm-001 | multi_agent | True | 29552.6ms | 44 | 11656 | 7093 | - |
| pm-002 | multi_agent | True | 16253.1ms | 9 | 5592 | 2659 | - |
| pm-002 | multi_agent | True | 35713.2ms | 46 | 15791 | 8599 | - |
| pm-002 | multi_agent | True | 18782.9ms | 12 | 5097 | 2377 | - |
| pm-002 | multi_agent | True | 35583.9ms | 40 | 11006 | 7134 | - |
| pm-002 | multi_agent | True | 19834.0ms | 12 | 5518 | 2908 | - |
| pm-002 | multi_agent | True | 29183.6ms | 35 | 10253 | 6035 | - |
| pd-001 | multi_agent | True | 12625.6ms | 10 | 3901 | 1380 | - |
| pd-001 | multi_agent | True | 46886.6ms | 31 | 13740 | 6471 | - |
| pd-001 | multi_agent | True | 12607.1ms | 10 | 3566 | 1284 | - |
| pd-001 | multi_agent | True | 49441.4ms | 34 | 8847 | 6101 | - |
| pd-001 | multi_agent | True | 12531.0ms | 10 | 3575 | 1272 | - |
| pd-001 | multi_agent | True | 46709.6ms | 31 | 8622 | 6426 | - |
| pd-002 | multi_agent | True | 13614.8ms | 10 | 3995 | 1623 | - |
| pd-002 | multi_agent | True | 38405.0ms | 26 | 12841 | 5455 | - |
| pd-002 | multi_agent | True | 12766.9ms | 10 | 3600 | 1387 | - |
| pd-002 | multi_agent | True | 50628.7ms | 32 | 9168 | 6975 | - |
| pd-002 | multi_agent | True | 12889.1ms | 10 | 3805 | 1468 | - |
| pd-002 | multi_agent | True | 38643.9ms | 27 | 7581 | 5035 | - |