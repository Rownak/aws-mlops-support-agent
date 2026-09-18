> Qrels are derived from cited-sentence annotations, not human relevance judgments, and mark only what the annotated answer cited — an unlabeled document may still be relevant. Every metric here is a lower bound, not directly comparable to a BEIR dataset's (design.md §4.1/9.3).

| experiment | ndcg@10 | mrr@10 | recall@10 | mean_latency_ms | llm_calls |
| --- | --- | --- | --- | --- | --- |
| dense | 0.8651 | 0.8356 | 0.9924 | 82 | 0 |
| bm25 | 0.7278 | 0.7082 | 0.8990 | 0 | 0 |
