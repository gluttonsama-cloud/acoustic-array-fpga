# 实验配置

- m1_das.json：合成工程信号，256/512点DAS。
- m2_speech_development.json：三个真实语音片段，DAS与固定约束比较及共同方向偏差扫描；仅开发集，不是B1/B2验收。

路径相对配置文件解析；实际命令、局限和结果见项目README及阶段报告。
# 低频开发对照补充

`m2_lowband_*.json`共6组，复用M2实验入口：500/250/100 Hz下限、−10/0 dB WNG保护，以及0.02 RMS独立噪声压力。全部保持512点FFT；不会覆盖原开发配置。结果与批量复现命令见docs/实验与验证/M2低频扩展与噪声对照.md。它们共享开发素材，不能算6个独立语料场景。

`m2_fft_*.json`：4组低噪声FFT/WNG对照与2组高噪声压力，仅作开发；详见docs/实验与验证/M2变换长度与延迟对照.md。

`m2_mismatch_development.json`明确记录六条件的方向/增益/延迟向量，入口为mismatch_experiment，依赖已归档父run。

m2_active_scene_development.json用于active_scene_experiment，指定目标活动RMS、噪声SNR、裁剪样本数及两个FFT。
