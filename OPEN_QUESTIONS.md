# 未决问题

未决问题不构成隐含授权。只有在答案不会实质改变结果时，工作才可继续；否则相关
路径必须保持占位或明确限制。

| ID | 未决问题 | 当前占位/默认处理 | 阻塞范围 |
|---|---|---|---|
| TBD-CI-001 | 已声明的 Python 3.10–3.14/操作系统矩阵，是否能在目标 GitHub runner 上使用已提交的精确工具版本通过？ | 本地 3.10.8 和部分匹配的 3.12.12 已通过；没有远端/Actions 结果 | 远端矩阵确认 |
| TBD-REMOTE-001 | 何时取得直接替代 `USTCPhonetics/FlexAligner` main 历史的单独外部授权？替代前保留什么经过验证的恢复快照？ | `REPLACE_MAIN_HISTORY` 只是已接受策略；尚未授权 remote、push、force update、默认分支修改或历史删除 | 远端历史替代 |
| TBD-E2E-002 | approved E2E manifest 如何去除对 `flexaligner-rebuild/tests/fixtures/e2e/english_synthetic.dict` 本地路径的依赖，并在目标远端 runner 证明可移植性？ | 保留 approved 本地 fixture 并关闭式失败；本地 Q-007 只覆盖已验证的 exact-wheel 运行，在完成可移植布局和实际复跑前不得声称 protected remote E2E 通过 | 远端/release E2E 可移植性 |
| TBD-REL-001 | `USTCPhonetics/FlexAligner` 将使用哪个精确 workflow identity、受保护 `pypi` environment、审批人和 Trusted Publisher 绑定？ | 目标项目和包 owner 已选定，但尚未配置或授权远端 environment/publisher | 实际 PyPI 上传 |
| TBD-ALG-001 | 是否修正连续覆盖？ | 已完成当前行为特征化；alpha 保留并披露 gap 行为 | 行为修正后的 MVP 声明 |
| TBD-ALG-002 | Stage 2 stride 策略是什么？ | 为保持等价，保留 0.01 s 并报告不匹配 | 动态 timing 声明 |
| TBD-ALG-003 | 连续相同 phone 状态如何区分身份？ | 为保持等价，保留当前行为 | 修正后的 phone 身份声明 |
| TBD-ALG-004 | 是否加入标准重复标签 CTC 约束？ | 为保持等价，保留当前简化 recurrence | 算法修正声明 |
| TBD-ALG-005 | 资源上限如何确定？ | 只有测量后才加入保守且有测试的上限 | 生产安全声明 |
| TBD-OUT-001 | 可选 metadata 与 TextGrid 两个文件之间应采用什么崩溃一致性协议？ | 两者均先暂存并验证；先提交 metadata，最后提交 TextGrid；回滚进程可见失败 | 多产物崩溃/断电原子性声明 |
| TBD-API-002 | 最终 phone interval 如何保留固定状态的 word/phone provenance？ | 暴露 `word_index=None`、`phone_index=None`；不得推断缺失 provenance | phone 到 word 的 provenance 声明 |
| TBD-PROV-001 | 公共 provenance 应采用什么规范的模型目录指纹格式？ | `model_fingerprints` 保持空；E2E manifest 保留资产 hash | 公共可复现性 schema |
| TBD-INF-001 | 公开 alpha 能诚实安装的 `[inference]` 解析/运行时范围是什么？ | 当前宽范围不视为已批准支持；收窄到有证据的范围并测试公共索引解析，否则移除公开 extra | 公开 alpha 推理契约 |
| TBD-THREAD-001 | 对齐完成后是否应恢复进程全局 Torch 线程数？ | 只在推理生命周期内设置请求的正 CPU 线程数；alpha 前文档化进程全局影响 | 线程隔离声明；alpha 文档 |
| TBD-TEXT-001 | 名为 `sil` 或 `null` 的正常词项如何与当前特殊 TextGrid 标签共存？ | 在模型加载前以类型化输入错误拒绝 | 对齐这两个英语词项 |

## 2026-08-11 用户审阅已解决的问题

这些记录只用于保留关闭历史，不再属于未决项，也不授予外部操作权限。

| 原 ID | 解决结果 | 决定 |
|---|---|---|
| TBD-PKG-001 | PyPI distribution 为 `flexaligner`；owner/organization 为 `ustcphonetics` | D-031 |
| TBD-PKG-002 | 直接替代现有 GitHub main 历史；不合并或嫁接无关历史 | D-030 |
| TBD-PKG-003 | 首个公开版本为 alpha `0.1.0a1`；实施前工作树保持 `0.1.0.dev0` | D-029 |
| TBD-LIC-001 | 使用固定的原远端 README 和同提交 LICENSE 作为 MIT、版权、作者、单位与引用身份来源 | D-032 |
| TBD-API-001 | v0.1 预览不提供宽泛旧兼容层；只有确认真实调用方后才重开 | D-034 |
| TBD-E2E-001 | 指定发音只批准为冻结 release-E2E fixture | D-033 |

## 当前范围已经固定、不是问题的事项

- 普通话在本里程碑中不需要真实模型 E2E。
- 任何占位能力都不得触发自动下载或转换。
- 缺少模型资产不能作为声称 E2E 通过的依据。
- 本地 reference 文件优先于冲突的旧会话描述。
