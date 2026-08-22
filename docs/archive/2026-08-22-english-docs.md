# 2026-08-22 中文化前英文文档归档索引

> 归档基线：`31becafa4cd3f8ec2a88820205e78609fbf70688`
>
> 归档性质：Git 中的逐字节历史快照；本文只提供恢复索引，不复制第二份可能漂移的正文。

本次中文化开始前，工作区中所有跟踪文档与上述提交完全一致。需要查看或恢复任一
英文原版时，使用：

```bash
git show 31becafa4cd3f8ec2a88820205e78609fbf70688:<文件路径>
```

也可以恢复到临时目录进行比较，但不得直接覆盖当前中文文档：

```bash
git show 31becafa4cd3f8ec2a88820205e78609fbf70688:STATE.md > /tmp/STATE.en.md
```

## 文件身份

| 文件 | SHA-256 | Git blob |
|---|---|---|
| `ACCEPTANCE.md` | `e84c700d316880f8bb229517964668dc00f81e4feabfb8c8ed1f104663bb0e4a` | `21673193e5eaa096dc0f0f4f3b714381b32dfa6f` |
| `DECISIONS.md` | `4ca35d541090ebb45950746e0effdbbfd45de793f11edb664e4ee15934ef806c` | `9b00eed87fd2db17398b1e923eed5699de2c3440` |
| `FINAL_ACCEPTANCE_REPORT.md` | `f0a5b11721236ab60584b38b3489109d5263b738c88c78779905b8b7f28c2e82` | `83d15b82c96a93085d58d81ff26f08d5d65eafaa` |
| `IMPLEMENTATION_PLAN.md` | `7a270bbc01eb56e9f2b6157d6d8a0ac1999d3d08851a7c995e7300b693303557` | `e15c8d43801ea4912d88ef318991a27bdde6bdda` |
| `OPEN_QUESTIONS.md` | `5c6e65fb807266448e13fd8f2ad37474b43e0634182e96e49423048aaafadbbb` | `5cc52c03fb1a35911892b51a144b8f4ca2065606` |
| `REAL_MODEL_E2E_REPORT.md` | `d7c2ed5ca3a98a2768fccade2dbf7b1fe66c728dd94c237b7c390cd0596c9903` | `00845088e290d9245770a4984ae2996b868f1310` |
| `REVIEW_DECISION_REPORT.md` | `4f9d874285aa70617f755d1b429b99354c07ea823826a489e620cbae2f5aac3a` | `1a637f616423ee0862ac0de6477ba6c8934916e8` |
| `STAGE1_RESOURCE_REPORT.md` | `ff0070b204f69feb389641b7a1903195984b7d716ddca75d364265c36a1571d8` | `901ec7eebc5b04ec954565e806cc340d8cee8311` |
| `STAGE2_RESOURCE_REPORT.md` | `1527aaa06f87eafde3bca4c3b020325feb9f1b822841f120e38d86235205873f` | `60c85ae3cff1177b2713c130b4911435fdbe4e17` |
| `STATE.md` | `ba06d456cd4333f9d033112978bfc73a7f4bcc37136c3b7c3d12619b1cd25b5e` | `c841d0131acccd6932baaf1815a5c1113e65a6f0` |
| `project.md` | `bacaf938581bb6196a75b118e3a632c98f1c8f56ece677349f30ae007169f39c` | `c1fe7894f17b2707c40813f0e42c2523353fbb2b` |
| `reference/README.md` | `094429e2349c9431de6435893a41a814ccea175d3d4837750718b68b5a8471ae` | `7ccf83be5bab86e3db3d87db305bf29d6c978dc3` |
| `tests/characterization/README.md` | `4a78f9ab9ef26fc5278e8ed44420b53a5f0bb1d910c71a79b11bdd4f477eff19` | `ffddc948932449a84f62533d153cbeca6e604fe4` |
| `tests/fixtures/e2e/README.md` | `ab6290fc0468053f4df994a727c0ace129ea802e1cc0bc1ff23083ccd8a43e90` | `32060f1a2dacaff14853418c18a32946f51f1b30` |

根目录 `README.md` 是准备对外发布的中英双语接口，依据 D-035 不翻译成纯中文，
因此不属于“待替换的英文内部文档”；它仍可通过同一提交恢复和审计。
