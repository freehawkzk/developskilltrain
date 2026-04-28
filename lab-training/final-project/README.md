# 贯穿主线项目：CI语音增强与识别 Pipeline

## 项目目标

构建一个完整的CI语音处理pipeline：

```
带噪语音 → DeepFilterNet语音增强 → ACE编码策略 → ASR语音识别 → 效果评估
```

## 阶段性成果

| 完成模块 | 阶段成果 |
|----------|----------|
| 模块0 | 能用Python写基本的数据处理脚本 |
| 模块1 | 环境就绪，能运行Python脚本 |
| 模块2 | 训练一个语音/噪声二分类CNN |
| 模块3 | 搭建VAD模型，处理CI相关分类任务 |
| 模块4 | 运行DeepACE，理解CI编码策略 |
| 模块5 | 运行DeepFilterNet，评估语音增强效果 |
| 模块6 | 完成端到端pipeline，评估整体效果 |

## 最终交付

- `pipeline.ipynb` — 完整pipeline的notebook
- `report-template.md` — 技术报告模板
