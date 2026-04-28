# SWE-bench Multilingual Retrieval Summary

Dataset: SWE-bench Multilingual  
Instances: 300  
Repositories: 41  
Cutoff: top 10 files  
Source run: `e2e_run_20260405_162252`  
Run date: 2026-04-05 to 2026-04-06

Models:

| Component | Model |
|---|---|
| LLM keyword extraction | MiniMax-M2.5 |
| Optional patch generation | Kimi-K2 |

| Backend | Hit@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| grep | 0.500 | 0.400 | 0.304 |
| bm25 | 0.523 | 0.436 | 0.246 |
| grep_bm25 | 0.593 | 0.493 | 0.355 |
| llm_grep | 0.760 | 0.639 | 0.469 |
| llm_grep_bm25 | 0.757 | 0.632 | 0.457 |

The best standalone retrieval backend in this run was `llm_grep`. Adding BM25 through reciprocal rank fusion slightly reduced the multilingual retrieval metrics.
