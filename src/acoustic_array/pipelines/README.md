# pipelines

配置组装、实验执行、流式编排和元数据记录；不堆数值内核。

状态：M1已有实现；数据语义以项目docs/架构与规范/接口与数据契约.md及ADR0003为准。未实现能力仍按项目规划推进。

近场开发诊断：python -m acoustic_array.pipelines.near_field_experiment --config configs/experiments/g1_near_field.json；详细口径见docs/实验与验证/G1近场传播与距离建模对照.md。

房间G1开发诊断：python -m acoustic_array.pipelines.room_experiment --config configs/experiments/g1_room.json；准确直接方向/距离、双参考评价与RIR/源码快照归档，不读取干净分量生成权重。

MPDR房间重放：python -m acoustic_array.pipelines.mpdr_experiment --config configs/experiments/g1_mpdr.json；输入是已冻结父run，前3秒校准后冻结，后段公平重算基线。

冻结权重分量重放：python -m acoustic_array.pipelines.room_component_replay --config configs/experiments/g1_component_replay.json；核验父归档，逐源及直达重放，断言线性叠加，生成独立证据目录。

几何敏感性：MPDR配置可选selected_cases和steering_error；五份g1_geometry配置固定单因素偏差。分量重放按父加载列表验证方法矩阵，非零几何误差使用该run冻结基线，禁止退回真实几何权重。


弱目标演示固定对照：python -m acoustic_array.pipelines.weak_knock_experiment --config configs/experiments/w1_knock.json。目标在训练后出现，背景协方差基线，非自动检出；资料许可、共同音频增益及评分边界见ADR0016。
