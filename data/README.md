# 数据管理

新增`raw/s1_candidate_verified_v1`：LibriSpeech test-clean的90段未评分候选保留录音，30位说话人与开发集隔离。清单`manifests/s1_candidate_v1.json`。30内容场景只有10个互不重叠的说话人组，正式B1仍待完成；见[S1报告](../docs/实验与验证/S1候选保留语料建设.md)。

raw保存原始音频；manifests保存归属、来源、哈希与分割；模拟输入随各run保存在input.npz，generated不重复存副本。

现有三段语音仅用于开发，不能进入后续保留集；重采样、裁剪和增益记录在run/config.expanded.json。真实16麦录音尚未接入。

更多语料引入前登记许可与说话人分割。用户录音不得自动公开或上传；分享CC BY派生音频保留归属、许可与改动说明。

当前正式评价候选为`manifests/s1_disjoint_v1.json`：90不同说话人×1录音，30场。旧候选仍保留；登记中150段heldout不等于150个评价样本。新增归档缓存校验后清理，FLAC和LICENSE等元数据保留。见[S1不复用报告](../docs/实验与验证/S1独立说话人语料与评价预注册.md)。

2026-09-22：s1_disjoint_v1已完成首次冻结评价。源manifest保留获取时unscored状态作为历史，不改写其哈希；当前已评分状态由S1首轮保留集评价.md及父run记录，后续不得把它当未见语料择优调参。
