# 验证分层

v0.3.0当前98项pytest通过；M1阶段曾有89项，历史记录均保留。证据在artifacts/validation/m1-tests.xml与m2-tests.xml。

unit覆盖解析几何、重建、分数延迟、指标和固定约束；integration覆盖流式DAS、实验归档、真实语音适配接口和文件完整性。测试中的音频是本地生成的小信号，不需要联网或原始语音缓存。regression预留给正式场景基准。

尚无板端测试或正式B1/B2保留集验收。
