# Data format

Private research data are not included. Memory JSONL records require memory_id, factual_memory, emotion_label, original_text, and split. Test-query records require query_row_id, evaluation_query, target_memory_id, target_emotion, reference_factual_memory, and original_text. Retrieval outputs should retain retrieved_memory_id, retrieved_factual_memory, similarities, exact-match status, Stage-1 stored emotion, and Stage-2 predicted emotion.

reference_factual_memory is evaluation-only outside explicitly named Oracle Event and Full Oracle runs. Keep generated text and per-query metrics out of version control unless data release has been approved.
