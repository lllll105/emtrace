# Core Oracle Decomposition Table

Values are descriptive means across the five retrieval methods. Each method-condition contains 1,103 paired queries. Full Oracle is one shared generation per family, not five independent experiments.

| Generation family | Condition | Event | Emotion | N | Emotion Acc. | Target Score | Emotion Sim. | Original-text Entail. |
|---|---|---|---|---:|---:|---:|---:|---:|
| emotion-label | Normal | normal | normal | 1103 | 29.43% | 0.3098 | 0.5058 | 0.3819 |
| emotion-label | Oracle Emotion | normal | oracle | 1103 | 32.86% | 0.3425 | 0.5220 | 0.3968 |
| emotion-label | Oracle Event | oracle | normal | 1103 | 29.94% | 0.3117 | 0.5131 | 0.4136 |
| emotion-label | Full Oracle | oracle | oracle | 1103 | 33.18% | 0.3412 | 0.5261 | 0.4193 |
| multilayer-vector | Normal | normal | normal | 1103 | 40.13% | 0.3791 | 0.5326 | 0.3485 |
| multilayer-vector | Oracle Emotion | normal | oracle | 1103 | 47.92% | 0.4426 | 0.5690 | 0.3574 |
| multilayer-vector | Oracle Event | oracle | normal | 1103 | 39.29% | 0.3791 | 0.5350 | 0.3772 |
| multilayer-vector | Full Oracle | oracle | oracle | 1103 | 47.51% | 0.4442 | 0.5681 | 0.3835 |
| npti | Normal | normal | normal | 1103 | 28.78% | 0.2969 | 0.5159 | 0.3717 |
| npti | Oracle Emotion | normal | oracle | 1103 | 30.84% | 0.3233 | 0.5290 | 0.3749 |
| npti | Oracle Event | oracle | normal | 1103 | 29.79% | 0.3054 | 0.5242 | 0.3973 |
| npti | Full Oracle | oracle | oracle | 1103 | 32.55% | 0.3318 | 0.5393 | 0.4043 |
