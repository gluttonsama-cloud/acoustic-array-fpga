# 配置

array/ula16_v1.json保存原始几何与采样基线，core.config.ArrayConfig校验采样关系、几何、参考麦和频带。未决硬件字段继续保留null。

experiments/m1_das.json定义可运行工程实验。阵列配置路径相对实验配置所在目录解析。FFT点数、块长、源方向和评价频带由实验配置控制；每次运行保存展开配置与哈希。尚无JSON Schema文件。
