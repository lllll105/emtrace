# Experiment design

EM-TRACE separates Event Retrieval, Affect Recovery, and Emotion Realization. Event-Affect Binding has four states: event-correct/affect-correct, event-correct/affect-wrong, event-wrong/affect-correct, and event-wrong/affect-wrong.

Two-stage retrieval first selects a memory by factual similarity. Its emotion-cue representation is then compared with 22 prototypes: 11 classes by two prototypes. Each class is partitioned with spherical k-means; each cluster prototype is a Weiszfeld geometric median followed by L2 normalization.

The Oracle decomposition is a 2 by 2 design: retrieved versus target factual memory, crossed with predicted versus target emotion. Oracle Affect does not repair event errors. Oracle Event replaces only the factual memory. Full Oracle supplies both target factual memory and target emotion.
