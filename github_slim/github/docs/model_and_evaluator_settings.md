# Model and evaluator settings

## Qwen
Qwen/Qwen2.5-7B-Instruct; 28 layers; hidden size 3,584; intermediate size 18,944; float16; greedy decoding; do_sample false; max_new_tokens 96; use_cache true.

## ModernBERT
cirimus/modernbert-base-go-emotions; recorded snapshot 690341c8744d225dfd7a1fddae23f541b362b487; local 2026-03 Optimized variant; 28 labels; 22 layers; hidden size 768. Evaluation uses batch 32 and max length 512. Cue embeddings use hidden state 22, mask-mean pooling, L2 normalization, and max length 128.

## NPTI
Formal neuron set: use/npti_set/emotion_neuron_sets_11class.json. SHA-256 ee525a5c40cfb46adff88b1bb0ea18d847ca46001d40fcf6fcb0c7c922a6c1d2. This is the earlier all7603 set, not the later 6,927-item current-bank set. Eleven emotions, 28 layers, SiLU(gate_proj(x)), selection threshold absolute delta above 0.05, gamma 1.0, positive increment gamma times a95 times sigmoid(10 times (absolute delta minus 0.15)), negative intervention min(h,0), last token only.

## NLI
Xenova/DeBERTa-v3-large-mnli-fever-anli-ling-wanli, based on MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli; 24 layers; hidden size 1,024; full ONNX model; onnxruntime-node 1.14.0; batch 16; max length 512. Premise is generated text, hypothesis is the reference, and Fact Consistency is entailment probability.
