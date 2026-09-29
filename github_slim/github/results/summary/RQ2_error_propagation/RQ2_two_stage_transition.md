# RQ2 — Two-stage Emotion Error Transition

一级正确性由 retrieved_emotion_from_memory 与 target_emotion 比较；二级正确性直接使用原始记录的 stage2_emotion_match。

| 一级结果 | 二级结果 | 数量 | 占全部样本 |
|---|---|---:|---:|
| 正确 | 正确 | 690 | 62.56% |
| 正确 | 错误 | 207 | 18.77% |
| 错误 | 正确（纠正） | 13 | 1.18% |
| 错误 | 错误 | 193 | 17.50% |

- Correction rate P(stage2 correct | stage1 wrong): **6.31%**
- Corruption rate P(stage2 wrong | stage1 correct): **23.08%**
