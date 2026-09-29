# RQ3 — Compact Main Results

Each row summarizes the five retrieval methods. The mean shows overall condition performance; the best-method columns identify the strongest emotion-accuracy configuration without expanding to the full 35-row table.

| Condition | Mean Emo. Acc | Mean Target | Mean Similarity | Mean Fact | Best method | Best Emo. Acc | Best Target | Best Similarity | Best Fact |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| Fact prompt only | 23.21% | 0.2382 | 0.4845 | 0.3526 | fact-only | 23.57% | 0.2414 | 0.4890 | 0.3644 |
| Memory + Retrieved Emotion Label | 29.43% | 0.3098 | 0.5058 | 0.3819 | concat-sum | 31.10% | 0.3281 | 0.5150 | 0.3982 |
| Memory + Oracle Emotion Label | 32.86% | 0.3425 | 0.5220 | 0.3968 | similarity-product | 33.00% | 0.3467 | 0.5262 | 0.4065 |
| Normal + Multilayer Steering | 40.13% | 0.3791 | 0.5326 | 0.3485 | fact-only | 43.79% | 0.4095 | 0.5427 | 0.3651 |
| Oracle + Multilayer Steering | 47.92% | 0.4426 | 0.5690 | 0.3574 | emotion-only | 50.05% | 0.4432 | 0.5762 | 0.3066 |
| Normal + NPTI | 28.78% | 0.2969 | 0.5159 | 0.3717 | concat-sum | 30.19% | 0.3109 | 0.5262 | 0.3883 |
| Oracle + NPTI | 30.84% | 0.3233 | 0.5290 | 0.3749 | concat-sum | 31.46% | 0.3272 | 0.5356 | 0.3933 |
