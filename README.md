# nlp-in-dialog-system

## 1 项目介绍

该项目计划实现在对话系统中的常用nlp算法，包括自然语言理解、对话状态管理、对话生成等

## 2 NLU

### 2.1 数据集

采用crosswoz数据集，[数据集的论文](https://arxiv.org/abs/2002.11893), [数据集的github](https://github.com/thu-coai/CrossWOZ)

### 2.2 模型

- 基于bert的意图识别
- 基于bert+crf的实体抽取
- 意图分类和实体抽取联合训练([BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/pdf/1902.10909.pdf))