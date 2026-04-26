# Technical Overview

This project evaluates file retrieval for SWE-bench style bug-fixing tasks. The retrieval-only benchmark asks each backend to rank candidate source files from a repository checkout using only the issue text.

The framework also includes an optional end-to-end script that takes retrieved files, sends their contents to a patch-generation model, and writes SWE-bench-compatible prediction JSONL files.

## Datasets

Two dataset configurations are supported:

| Dataset | Hugging Face name | Split | Local repo cache |
|---|---|---|---|
| SWE-bench Verified | `princeton-nlp/SWE-bench_Verified` | `test` | `swe_repos/` |
| SWE-bench Multilingual | `SWE-bench/SWE-bench_Multilingual` | `test` | `swe_repos_multilingual/` |

Each instance contains:

| Field | Description |
|---|---|
| `instance_id` | Unique SWE-bench instance identifier. |
| `repo` | GitHub repository path, such as `django/django`. |
| `base_commit` | Commit where the bug exists. |
| `problem_statement` | Bug report used as the retrieval query. |
| `patch` | Ground-truth fix patch. |
| `gold_files` | Files modified by the patch, derived from `diff --git` headers. |

## Retrieval Pipeline

For each instance:

1. Load the issue text and gold files.
2. Clone or reuse the repository checkout at `base_commit`.
3. Run each configured retrieval backend.
4. Compare the top-ranked files against the gold files.
5. Write per-instance details and aggregate metrics.

The retrieval benchmark does not inspect files after ranking, revise queries, run tests, or use an agent loop.

## Models Used In Reported Runs

The public scripts are model-configurable through environment variables, but the reported benchmark used the following models:

| Component | Model | Used For |
|---|---|---|
| Keyword extraction | MiniMax-M2.5 | Extracting precise identifiers for `llm_grep` and `llm_grep_bm25`. |
| Patch generation | Kimi-K2 | Generating unified diffs in `run_e2e_eval.py`. |
| Embedding retrieval implementation | `Salesforce/SweRankEmbed-Small` | Optional issue-to-code embedding backend. |

The retrieval-only numbers in the README depend on MiniMax-M2.5 for the LLM-assisted backends. Kimi-K2 is relevant only to the optional end-to-end patch-generation pipeline.

## Backends

### `grep`

The heuristic grep backend extracts words from the issue text, removes common stop words, sorts by length, and searches for the top keywords. It scores files by content hits plus a path bonus.

The path bonus is intentionally large because filenames and directory names often encode the relevant concept more directly than file content frequency.

### `bm25`

The BM25 backend tokenizes all text files in the repository and ranks them with Okapi BM25. Tokenized corpora are cached per repository checkout under `swe_results/<dataset>/bm25_cache/`.

### `grep_bm25`

This backend combines `grep` and `bm25` with reciprocal rank fusion:

```text
score(file) = sum(1 / (k + rank_in_backend))
```

The implementation uses `k = 60`.

### `llm_grep`

This backend uses an OpenAI-compatible chat completion model to extract precise technical identifiers from the bug report. The model is asked for strings likely to appear verbatim in the files that need to change: function names, class names, methods, modules, and error substrings.

Those strings are then searched with fixed-string grep:

```bash
grep -rli -F --binary-files=without-match "<keyword>" <repo>
```

The `-F` flag is important because extracted identifiers are literal strings, not regular expressions. If the LLM call fails or `LLM_API_KEY` is not configured, this backend falls back to heuristic grep.

### `llm_grep_bm25`

This backend combines `llm_grep` and `bm25` with reciprocal rank fusion.

### Optional Embedding Backend

`run_swebench_search_eval.py` also contains a SweRank embedding backend. It is lazy-loaded and requires `sentence-transformers`. It is not part of the compact multilingual result table in the README.

## Metrics

All default metrics are computed at `k = 10`.

| Metric | Definition |
|---|---|
| `Hit@10` | `1.0` if any gold file appears in the top 10, else `0.0`. |
| `Recall@10` | Fraction of gold files recovered in the top 10. |
| `MRR@10` | Reciprocal rank of the first gold file in the top 10. |

## Caches And Outputs

Generated artifacts are written under `swe_results/`:

```text
swe_results/
├── verified/
│   ├── keyword_cache/
│   ├── bm25_cache/
│   └── run_<timestamp>/
└── multilingual/
    ├── keyword_cache/
    ├── bm25_cache/
    ├── patch_cache/
    └── run_<timestamp>/
```

Repository checkouts are stored under:

```text
swe_repos/
swe_repos_multilingual/
```

These directories are local generated data and are ignored by git.

## Environment Variables

`llm_grep` reads:

| Variable | Description |
|---|---|
| `LLM_API_BASE` | OpenAI-compatible API base URL. |
| `LLM_API_KEY` | API key for keyword extraction. |
| `LLM_MODEL` | Keyword extraction model name. |

`run_e2e_eval.py` additionally reads:

| Variable | Description |
|---|---|
| `PATCH_API_BASE` | OpenAI-compatible API base URL for patch generation. |
| `PATCH_API_KEY` | API key for patch generation. |
| `PATCH_MODEL` | Patch generation model name. |

## Reproducing The Main Retrieval Run

```bash
python3 run_swebench_search_eval.py \
  --dataset multilingual \
  --limit 300 \
  --backends grep,bm25,grep_bm25,llm_grep,llm_grep_bm25
```

For a credential-free smoke test:

```bash
python3 run_swebench_search_eval.py \
  --dataset multilingual \
  --limit 3 \
  --backends grep,bm25,grep_bm25
```
