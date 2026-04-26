# SWE-bench Verified Retrieval Summary

Dataset: SWE-bench Verified  
Instances: 500  
Repositories: 12  
Cutoff: top 10 files

| Backend | Hit@10 | Recall@10 | MRR@10 |
|---|---:|---:|---:|
| grep | 0.614 | 0.587 | 0.421 |
| bm25 | 0.756 | 0.713 | 0.446 |
| grep_bm25 | 0.784 | 0.751 | 0.525 |
| llm_grep | 0.748 | 0.714 | 0.446 |
| llm_grep_bm25 | 0.856 | 0.816 | 0.571 |

The best retrieval backend on SWE-bench Verified in this run was `llm_grep_bm25`.
