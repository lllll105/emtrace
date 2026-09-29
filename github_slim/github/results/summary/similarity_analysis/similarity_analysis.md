# Retrieval Similarity: Correct vs Wrong Exact Recall

Scores are taken from Normal + Multilayer Steering retrieval records.

| Method | Correct n | Wrong n | Fact score (correct) | Fact score (wrong) | Gap | Emotion score (correct) | Emotion score (wrong) | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fact-only | 790 | 313 | 0.7860 | 0.6943 | +0.0917 | 0.9744 | 0.9657 | +0.0087 |
| emotion-only | 49 | 1054 | 0.7866 | 0.4047 | +0.3819 | 0.9944 | 0.9933 | +0.0010 |
| concat-sum | 796 | 307 | 0.7841 | 0.6913 | +0.0928 | 0.9759 | 0.9780 | -0.0021 |
| similarity-product | 795 | 308 | 0.7844 | 0.6929 | +0.0915 | 0.9757 | 0.9760 | -0.0003 |
| two-stage | 790 | 313 | 0.7860 | 0.6943 | +0.0917 | 0.9744 | 0.9657 | +0.0087 |
| all | 3220 | 2295 | 0.7851 | 0.5607 | +0.2244 | 0.9754 | 0.9814 | -0.0060 |
