# Train-only Two-stage Prototype Sensitivity

{
  "protocol": "Saved evaluation_query vectors -> unchanged Stage-1 Top-1 over all 8,030 fact keys -> same retrieved cue scored by full-memory versus train-only m=2 prototype banks. No text generation or injection run.",
  "n": 1103,
  "stage1_exact_recall_count": 790,
  "stage1_exact_recall_at1": 0.7162284678150499,
  "stage2_label_changed_n": 31,
  "stage2_label_changed_rate": 0.028105167724388033,
  "full_bank": "./use_2rag/build/stage2_emotion_multicenter_geometric_m2/stage2_emotion_multicenter_to_injection_bank.pt",
  "train_only_bank": "./use_2rag/em_trace_results/plus/train_only_prototypes/stage2_emotion_multicenter_to_injection_bank.pt"
}
