# 数据清单

libri_examples.json登记三个librosa官方托管的LibriSpeech片段，包含原始SHA-256、说话人ID、朗读者归属、原始链接和CC BY 4.0许可。

这些片段全部属于development_only_not_heldout，不能放进未来独立保留测试集。官方片段说明中的章节名称存在文字差异，因此使用原始语料ID和朗读者作为识别依据，不自行改写录音来源。

下载命令（项目根目录）：

```powershell
.venv/Scripts/python.exe tools/download_speech_examples.py
```

脚本核对固定SHA-256，已存在且匹配的文件不会重复下载，不符合哈希时停止且不覆盖。音频位于data/raw/libri_examples。所有导出的语音混合与增强音频是重采样、裁剪、调平和空间混合/波束处理后的派生文件；分享时携带对应run/REPORT.md中的归属和许可。
