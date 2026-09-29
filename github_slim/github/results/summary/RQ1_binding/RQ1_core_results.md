# RQ1 — Event–Affect Binding

定义：Event✓ = memory_exact_top1；Affect✓ = 最终恢复/注入的 emotion label 等于 target_emotion。

## Exact Memory Recall@1 and Binding@1

| Method | Exact Recall@1 | Binding@1 |
|---|---:|---:|
| fact-only | 71.62% (790/1103) | 71.62% (790/1103) |
| emotion-only | 4.44% (49/1103) | 4.44% (49/1103) |
| concat-sum | 72.17% (796/1103) | 72.17% (796/1103) |
| similarity-product | 72.08% (795/1103) | 72.08% (795/1103) |
| two-stage | 71.62% (790/1103) | 53.58% (591/1103) |

## Event–Affect 四象限

| Method | Event✓ Affect✓ | Event✓ Affect✗ | Event✗ Affect✓ | Event✗ Affect✗ |
|---|---:|---:|---:|---:|
| fact-only | 790 | 0 | 107 | 206 |
| emotion-only | 49 | 0 | 267 | 787 |
| concat-sum | 796 | 0 | 113 | 194 |
| similarity-product | 795 | 0 | 111 | 197 |
| two-stage | 591 | 199 | 112 | 201 |
