# FlexAligner 算法 reference 快照

`align_single_cpu.py` 是以下文件不可变的逐字节证据快照：

```text
/Users/yiyi0369/projects/flexaligner/align_single_cpu.py
```

已验证的快照身份：

- SHA-256：`9ed4e21e615718ddfd10930359f55769fb27a0d284599cce45a3fc755e835de1`
- 行数：`2548`
- 字节数：`96230`
- 复制方式：逐字节复制；没有格式化或源代码修改

当前本地源文件和本快照共同作为等价迁移的行为 oracle。远端仓库快照
`USTCPhonetics/FlexAligner main@c5361efe4b5d8ad02574dae1bd7caa89ed3e4af0`
只作为 README、身份、许可 provenance 和差异比较来源；其历史核心实现不是新算法权威。

本文件是证据，不是生产代码：

- `src/flexaligner` 不得导入或执行它；
- wheel 和 sdist 必须排除 `reference/`；
- 修改快照需要新的已验证 hash 和明确决定；
- 特征化测试只能通过隔离 loader 导入它；loader 会为 Torch 和 Transformers
  提供 stub，并在结束后恢复 `sys.modules`。

旧会话曾把 TextGrid 连续 gap 验证描述成已经修复。当前权威快照不会在 interval
仍保持顺序且位于边界内时拒绝开头、内部或尾部 gap。特征化测试把这个冲突记录为
已知限制，不做静默修复。
