# P0-2 Paired Bootstrap and McNemar Analysis

{
  "protocol": "10,000 query-level nonparametric bootstrap resamples; a resampled query retains all method/condition observations, preserving paired comparisons. CI is percentile 2.5th–97.5th. McNemar uses an exact two-sided binomial test on discordant pairs.",
  "n_queries": 1103,
  "bootstrap_iterations": 10000,
  "random_seed": 20260928,
  "retrieval_source": "./outputs/eval/eval_normal_retrieval_vector_steering_20260923/per_generation_metrics.jsonl",
  "generation_sources": {
    "fact_prompt_only": "eval_fact_prompt_only_20260925",
    "memory_retrieved_emotion_label": "eval_memory_retrieved_emotion_label_20260925",
    "memory_oracle_emotion_label": "eval_memory_oracle_emotion_label_20260925",
    "normal_multilayer": "eval_normal_retrieval_vector_steering_20260923",
    "oracle_multilayer": "eval_oracle_retrieval_vector_steering_20260923",
    "normal_npti": "eval_normal_retrieval_npti_20260923",
    "oracle_npti": "eval_oracle_retrieval_npti_20260923"
  },
  "metric_note": "uar_legacy_event_wrong_affect_correct is the previously used ESR/UAR-style rate: event retrieval wrong but final retrieved affect correct. It is not standard macro recall.",
  "excluded_rows": "NPTI bare-query baseline rows excluded; five retrieved_fact_prompt_npti rows per query retained."
}
