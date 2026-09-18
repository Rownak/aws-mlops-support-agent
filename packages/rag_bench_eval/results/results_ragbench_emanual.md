> Qrels are derived from cited-sentence annotations, not human relevance judgments, and mark only what the annotated answer cited — an unlabeled document may still be relevant. Every metric here is a lower bound, not directly comparable to a BEIR dataset's (design.md §4.1/9.3).

| experiment | ndcg@10 | mrr@10 | recall@10 | mean_latency_ms | llm_calls |
| --- | --- | --- | --- | --- | --- |
| hybrid_cross_encoder | 0.9117 | 0.9369 | 0.9697 | 254 | 0 |
| hybrid | 0.8683 | 0.8730 | 0.9747 | 33 | 0 |
| bi_encoder_rerank | 0.8651 | 0.8356 | 0.9924 | 649 | 0 |
| dense | 0.8651 | 0.8356 | 0.9924 | 86 | 0 |
| bm25 | 0.7278 | 0.7082 | 0.8990 | 0 | 0 |
