# 维护工具

`prepare_s1_corpus.py <archive> <new_output_directory>`核验本地官方归档并提取固定90段候选保留录音，不联网、不覆盖、不执行波束算法。输出须位于项目data/raw下；见[S1语料报告](../docs/实验与验证/S1候选保留语料建设.md)。

未来放环境检查、manifest校验、结果汇总等小工具。核心声学算法进入src/acoustic_array；不要在这里形成另一套重复实现。

`download_speech_examples.py` 下载并校验 data/manifests/libri_examples.json 中固定的三段 librosa 官方语音示例。在项目根目录执行：

```powershell
.venv/Scripts/python.exe tools/download_speech_examples.py
```

已有文件只校验，不重复下载；哈希不匹配会报错，不覆盖文件。仅允许官方示例域名，单文件限制10MB。许可、归属和开发集用途见 data/README.md；此工具不下载正式保留集。

`prepare_s1_corpus.py --policy <json>`支持单人单段选择。`assemble_s1_holdout.py <new_manifest_path>`将固定的三个父清单汇编为30个不复用说话人的场景，校验90个源文件并检查开发隔离，不评分。见[S1不复用报告](../docs/实验与验证/S1独立说话人语料与评价预注册.md)。

## 分级参考数据导出

`export_handoff_vectors.py <新输出目录>`生成MIC7单位传递和三路DAS的浮点阶段二进制文件及manifest，内含离线/流式核验。需项目环境，拒绝覆盖目录；不代表定点IP或声学效果验收。
