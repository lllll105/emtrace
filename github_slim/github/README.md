# EM-TRACE: Episodic Emotion Recall and Realization

Reproducibility code for a stage-wise study of long-term emotional memory: Event Retrieval, Affect Recovery, and Emotion Realization.

## Scope

- 8,030 stored memories: 6,927 train and 1,103 test records.
- Five retrieval methods: fact-only, emotion-only, concat-sum, similarity-product, and two-stage.
- Seven generation conditions: fact prompt only; retrieved/oracle explicit emotion label; normal/oracle multilayer steering; normal/oracle NPTI.
- Models: Qwen2.5-7B-Instruct, ModernBERT GoEmotions, and DeBERTa-v3-large NLI.
- Analyses: Event-Affect Binding, error propagation, retrieval-to-generation analysis, bootstrap and McNemar tests, train-only prototype sensitivity, and Oracle Event decomposition.

Five retrieval methods by seven generation conditions gives 35 retrieval-conditioned runs. A separate bare no-memory baseline is the optional 36th condition.

## Layout

- configs: portable templates and the corrected experiment manifest.
- docs: design, schemas, exact model settings, and reproducibility notes.
- examples: synthetic JSONL schemas only.
- src: common, memory_bank, retrieval, steering, npti, generation, evaluation, and analysis.
- results: small aggregate tables and figures only.

## Excluded

Weights, caches, raw/private data, tensors, logs, generations, and large per-query outputs are excluded. Create configs/model_config.json from the example and pass local paths through each script CLI.

## Protocol warning

The main 8,030-memory bank contains the 1,103 evaluation memories. Exact Recall measures recall of stored experiences, not unseen-memory generalization. The train-only prototype analysis separately tests leakage through Stage-2 prototype construction.

## License

No open-source license has been selected. Choose one before public release.
