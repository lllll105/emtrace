# Reproducibility

1. Configure local model and JSONL paths.
2. Build multilayer memory keys, memory banks, and steering values.
3. Build Stage-1 fact-to-emotion-cue representations.
4. Run five retrieval methods and save query-level selections.
5. Build 22 Stage-2 emotion prototypes.
6. Run seven generation conditions.
7. Evaluate emotion and NLI metrics.
8. Run EM-TRACE tables, paired bootstrap intervals, and McNemar tests.

The train-only sensitivity rebuilds prototypes from 6,927 training memories while reusing the exact same 1,103 Stage-1 selections. Bootstrap resampling is paired by query_row_id.

The source Full-Oracle NPTI config was overwritten with an empty methods list. It was a metadata error: actual methods were fact-only, emotion-only, concat-sum, similarity-product, two-stage, and full-oracle. The public manifest corrects this.
