# simulation

场景生成、传播模型、房间模拟适配、噪声和通道误差注入；真值仅供评价。

状态：M1已有实现；数据语义以项目docs/架构与规范/接口与数据契约.md及ADR0003为准。未实现能力仍按项目规划推进。

v0.3.2新增mismatch.py，离线静态通道增益/残余时延；先源传播后通道传递，噪声按前置模型通过同一传递，不包含采样率漂移/抖动或后级噪声。

v0.3.8新增near_field.py独立球面直达传播；active_scene通过可选source_distances_m复用活动定标。源压力参考MIC7，不含绝对传播时间、距离响度测试或混响。

v0.3.9新增room.py可选rir-generator适配与绝对时间卷积；active_scene抽取normalize_active_scene，房间场景以直接分量冻结掩码和整源增益，保留反射比例。


weak_scene.py：以固定目标活动段、物理参考通道标定完整目标/干扰SIR。离线真值仅用于场景与评价，不能传入波束算法。
