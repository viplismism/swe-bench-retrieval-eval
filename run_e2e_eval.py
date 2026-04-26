#!/usr/bin/env python3
"""
SWE-bench Multilingual End-to-End Eval Pipeline
================================================

Architecture:
    instance
      ├── retrieval_strategy_1 → top-k files → LLM patch gen → predictions
      ├── retrieval_strategy_2 → top-k files → LLM patch gen → predictions
      └── retrieval_strategy_3 → top-k files → LLM patch gen → predictions
                                    ↓
                      swebench harness (Docker) → test execution → scores

This script handles: retrieval → patch generation → predictions JSONL.
The swebench official harness handles: test execution → scoring.

Usage:
    python3 run_e2e_eval.py --limit 5 --backends grep_bm25,llm_grep_bm25
    python3 run_e2e_eval.py --limit 300 --backends grep_bm25,llm_grep_bm25 --patch-model gpt-4o

Env vars:
    LLM_API_BASE   OpenAI-compatible base URL for keyword extraction
    LLM_API_KEY    API key for keyword extraction
    LLM_MODEL      keyword extraction model
    PATCH_API_BASE OpenAI-compatible base URL for patch generation
    PATCH_API_KEY  API key for patch generation
    PATCH_MODEL    patch generation model
"""

import sys, os, json, time, re, subprocess, shutil, hashlib, urllib.request
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# Configuration
# ============================================================================

EVAL_DIR    = Path(__file__).parent
RESULTS_DIR = EVAL_DIR / "swe_results"
SCRIPT_DIR  = EVAL_DIR  # For importing retrieval module

# Add parent to path so we can import the retrieval backends
sys.path.insert(0, str(EVAL_DIR))

# Now import retrieval backends and utilities from the search eval script
from run_swebench_search_eval import (
    clone_repo,
    search_grep, search_bm25_python,
    search_grep_bm25, search_llm_grep, search_llm_grep_bm25,
    parse_patch_files, _collect_code_files, _get_bm25_index,
    _llm_extract_keywords, _llm_stats, _load_keyword_cache, _save_keyword_cache,
    LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_TIMEOUT, LLM_RETRIES,
    KEYWORD_CACHE_DIR, REPOS_DIR, RESULTS_DIR, BM25_CACHE_DIR,
    BINARY_EXTENSIONS, _is_text_file, TOP_K,
    _keyword_cache_path,
    DATASET_CONFIGS,
)


# ============================================================================
# Patch generation LLM config (separate from keyword extraction model)
# ============================================================================

PATCH_API_BASE  = os.environ.get("PATCH_API_BASE", LLM_API_BASE)
PATCH_API_KEY   = os.environ.get("PATCH_API_KEY",  LLM_API_KEY)
PATCH_MODEL     = os.environ.get("PATCH_MODEL",    "gpt-4o")
PATCH_TIMEOUT   = 180   # seconds — patch generation needs more time
PATCH_MAX_TOKENS = 8192  # patches can be long
PATCH_RETRIES    = 2

# Stats tracking
_patch_stats = {"cache_hit": 0, "api_ok": 0, "api_fail": 0, "errors": []}
SKIP_PATCH_CACHE = False  # Set via --no-cache to force fresh API calls

# Patch cache: keyed by sha256(prompt)[:16], stored per model
PATCH_CACHE_DIR = RESULTS_DIR / "patch_cache"  # Default; overridden per dataset in main()
_patch_cache: Optional[Dict[str, str]] = None


def _patch_cache_path() -> Path:
    safe = re.sub(r'[^\w.-]', '_', PATCH_MODEL)
    return PATCH_CACHE_DIR / f"{safe}.json"


def _load_patch_cache() -> Dict[str, str]:
    global _patch_cache
    if _patch_cache is None:
        path = _patch_cache_path()
        if path.exists():
            try:
                _patch_cache = json.loads(path.read_text())
            except Exception:
                _patch_cache = {}
        else:
            _patch_cache = {}
    return _patch_cache


def _save_patch_cache():
    if _patch_cache is not None:
        path = _patch_cache_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_patch_cache))
        except Exception:
            pass


# ============================================================================
# Tee helper (same as in search eval)
# ============================================================================

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


# ============================================================================
# Retrieval backends registry
# ============================================================================

BACKEND_FUNCS = {
    "grep":           search_grep,
    "bm25":           search_bm25_python,
    "grep_bm25":      search_grep_bm25,
    "llm_grep":       search_llm_grep,
    "llm_grep_bm25":  search_llm_grep_bm25,
}

ALL_BACKENDS = list(BACKEND_FUNCS.keys())


# ============================================================================
# File content reading
# ============================================================================

def read_file_contents(repo_dir: Path, file_paths: List[str],
                       max_chars_per_file: int = 12000) -> Dict[str, str]:
    """Read contents of retrieved files from the repo checkout.

    Args:
        repo_dir: Path to the repo checkout
        file_paths: List of relative file paths to read
        max_chars_per_file: Max chars to read per file (to fit in LLM context)

    Returns:
        Dict mapping file_path -> file_content
    """
    contents = {}
    for fp in file_paths:
        full_path = repo_dir / fp
        if not full_path.is_file():
            continue
        try:
            text = full_path.read_text(errors="ignore")
            if len(text) > max_chars_per_file:
                text = text[:max_chars_per_file] + "\n... [truncated]"
            contents[fp] = text
        except Exception:
            continue
    return contents


# ============================================================================
# Patch generation via LLM
# ============================================================================

def generate_patch(problem_statement: str, retrieved_files: Dict[str, str],
                   instance_id: str = "", log_errors: bool = True) -> str:
    """Call the LLM to generate a unified diff patch.

    Args:
        problem_statement: The bug report / issue description
        retrieved_files: Dict of {filepath: content} for candidate files
        instance_id: For logging
        log_errors: Whether to print errors to stderr

    Returns:
        A unified diff string, or empty string on failure.
    """
    # Build the prompt
    files_section = ""
    for fp, content in retrieved_files.items():
        files_section += f"\n--- {fp} ---\n{content}\n"

    prompt = (
        "You are a software engineer fixing a bug. Given a bug report and the "
        "relevant source files, generate a unified diff (patch) that fixes the issue.\n\n"
        "RULES:\n"
        "- Output ONLY a valid unified diff. No explanation, no markdown fences.\n"
        "- Use the standard format: --- a/path and +++ b/path with @@ hunks.\n"
        "- Be minimal: only change what's necessary to fix the bug.\n"
        "- Do NOT add new files unless absolutely required.\n"
        "- If you're unsure which file to change, pick the most likely one.\n\n"
        f"BUG REPORT:\n{problem_statement}\n\n"
        f"CANDIDATE SOURCE FILES:\n{files_section}\n\n"
        "OUTPUT (unified diff only):"
    )

    if not PATCH_API_KEY:
        _patch_stats["api_fail"] += 1
        err_msg = f"[PATCH FAIL] {instance_id}: PATCH_API_KEY is not set"
        _patch_stats["errors"].append(err_msg)
        if log_errors:
            print(f"      {err_msg}", file=sys.__stderr__)
        return ""

    # Check cache
    cache = _load_patch_cache()
    cache_key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    if not SKIP_PATCH_CACHE and cache_key in cache:
        _patch_stats["cache_hit"] += 1
        return cache[cache_key]

    # Call LLM
    payload = json.dumps({
        "model":      PATCH_MODEL,
        "messages":   [{"role": "user", "content": prompt}],
        "max_tokens": PATCH_MAX_TOKENS,
        "temperature": 0.0,
    }).encode()

    last_error = None
    for attempt in range(1, PATCH_RETRIES + 1):
        req = urllib.request.Request(
            f"{PATCH_API_BASE}/chat/completions",
            data=payload, method="POST",
        )
        req.add_header("Content-Type",  "application/json")
        req.add_header("Authorization", f"Bearer {PATCH_API_KEY}")

        try:
            with urllib.request.urlopen(req, timeout=PATCH_TIMEOUT) as resp:
                raw = resp.read().decode()
                if not raw.strip():
                    raise ValueError("Empty response body")
                body = json.loads(raw)
                finish = body["choices"][0].get("finish_reason", "")
                text = body["choices"][0]["message"]["content"]
                if not text:
                    raise ValueError(f"content=null, finish_reason={finish}")

                # Clean up: strip markdown fences if present
                text = text.strip()
                text = re.sub(r'^```(?:diff)?\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
                text = text.strip()

                if not text:
                    raise ValueError("Empty patch after cleanup")

                # Cache and return
                cache[cache_key] = text
                _save_patch_cache()
                _patch_stats["api_ok"] += 1
                return text

        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code}: {e.reason}"
            if 400 <= e.code < 500:
                break
        except urllib.error.URLError as e:
            last_error = f"URLError: {e.reason}"
        except TimeoutError:
            last_error = "TimeoutError"
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
        except (KeyError, IndexError, ValueError) as e:
            last_error = f"Response parse: {e}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < PATCH_RETRIES:
            time.sleep(2)

    # All retries exhausted
    _patch_stats["api_fail"] += 1
    err_msg = f"[PATCH FAIL] {instance_id}: after {PATCH_RETRIES} attempts: {last_error}"
    _patch_stats["errors"].append(err_msg)
    if log_errors:
        print(f"      {err_msg}", file=sys.__stderr__)
    return ""


# ============================================================================
# Per-instance evaluation
# ============================================================================

def evaluate_instance(inst: Dict, backends: List[str],
                      top_k: int = TOP_K, verbose: bool = False) -> Dict:
    """Run full pipeline for one instance: retrieval → read files → generate patch.

    Returns a dict with results per backend.
    """
    repo = inst["repo"]
    commit = inst["base_commit"]
    query = inst["problem_statement"]
    instance_id = inst["instance_id"]
    gold_files = inst["gold_files"]

    # Clone/checkout repo
    repo_dir = clone_repo(repo, commit)
    if repo_dir is None:
        print(f"    [SKIP] {instance_id}: clone failed")
        return {"instance_id": instance_id, "status": "clone_failed", "backends": {}}

    results = {"instance_id": instance_id, "status": "ok", "backends": {}}

    for backend_name in backends:
        search_fn = BACKEND_FUNCS[backend_name]

        # Step 1: Retrieval
        t0 = time.time()
        try:
            raw = search_fn(query, repo_dir, top_k=top_k)
            # LLM backends return (files, keywords); others return just files
            if isinstance(raw, tuple):
                retrieved = raw[0]
            else:
                retrieved = raw
        except Exception as e:
            print(f"      [{backend_name}] retrieval error: {e}")
            retrieved = []
        retrieval_time = time.time() - t0

        # Step 2: Read retrieved file contents
        file_contents = read_file_contents(repo_dir, retrieved)

        # Step 3: Compute retrieval metrics (same as search eval)
        gold_set = set(gold_files)
        pred_set = set(retrieved)
        hits = gold_set & pred_set
        hit = 1 if hits else 0
        recall = len(hits) / len(gold_set) if gold_set else 0
        # MRR
        mrr = 0.0
        for rank, fp in enumerate(retrieved, 1):
            if fp in gold_set:
                mrr = 1.0 / rank
                break

        # Step 4: Generate patch
        t1 = time.time()
        if file_contents:
            model_patch = generate_patch(query, file_contents, instance_id=instance_id)
        else:
            model_patch = ""
        patch_time = time.time() - t1

        results["backends"][backend_name] = {
            "retrieved":      retrieved,
            "retrieved_count": len(retrieved),
            "files_read":     len(file_contents),
            "hit":            hit,
            "recall":         recall,
            "mrr":            mrr,
            "model_patch":    model_patch,
            "patch_lines":    len(model_patch.split("\n")) if model_patch else 0,
            "retrieval_time": round(retrieval_time, 2),
            "patch_gen_time": round(patch_time, 2),
        }

        if verbose:
            status = "HIT" if hit else "MISS"
            print(f"      [{backend_name:15s}] {status} "
                  f"retrieved={len(retrieved)} read={len(file_contents)} "
                  f"patch={len(model_patch)} chars  "
                  f"({retrieval_time:.1f}s + {patch_time:.1f}s)")

    return results


# ============================================================================
# Dataset loading
# ============================================================================

def load_dataset_instances(dataset: str = "multilingual",
                           limit: int = 0) -> List[Dict]:
    """Load SWE-bench instances from local cache."""
    cfg = DATASET_CONFIGS[dataset]
    cache_file = cfg["cache_file"]
    label = cfg["label"]

    if not cache_file.exists():
        # Also check legacy flat location
        legacy = RESULTS_DIR / cache_file.name
        if legacy.exists():
            cache_file = legacy
        else:
            print(f"ERROR: {cache_file} not found.")
            sys.exit(1)

    all_instances = json.loads(cache_file.read_text())

    # For multilingual, verify test fields exist
    if dataset == "multilingual" and all_instances and "FAIL_TO_PASS" not in all_instances[0]:
        print("ERROR: Dataset cache missing test fields (FAIL_TO_PASS). "
              "Re-run scripts/_regen_multilingual_cache.py.")
        sys.exit(1)

    if limit and limit < len(all_instances):
        all_instances = _pick_diverse(all_instances, limit)

    print(f"  Loaded {len(all_instances)} {label} instances across "
          f"{len({i['repo'] for i in all_instances})} repos")
    return all_instances


def _pick_diverse(instances: List[Dict], limit: int) -> List[Dict]:
    """Pick instances spread across different repos."""
    from collections import defaultdict
    by_repo = defaultdict(list)
    for inst in instances:
        by_repo[inst["repo"]].append(inst)

    selected = []
    repos = sorted(by_repo.keys())
    idx = 0
    while len(selected) < limit:
        repo = repos[idx % len(repos)]
        if by_repo[repo]:
            selected.append(by_repo[repo].pop(0))
        idx += 1
        if idx > limit * 10:
            break
    return selected[:limit]


# ============================================================================
# Predictions output (JSONL for swebench harness)
# ============================================================================

def write_predictions_jsonl(results: List[Dict], backend: str,
                            output_path: Path):
    """Write predictions in the format expected by swebench harness.

    Format per line:
    {"instance_id": "...", "model_name_or_path": "...", "model_patch": "..."}
    """
    with open(output_path, "w") as f:
        for r in results:
            if r["status"] != "ok":
                continue
            backend_result = r["backends"].get(backend, {})
            patch = backend_result.get("model_patch", "")
            entry = {
                "instance_id":       r["instance_id"],
                "model_name_or_path": f"e2e_{backend}_{PATCH_MODEL}",
                "model_patch":       patch,
            }
            f.write(json.dumps(entry) + "\n")
    print(f"  Predictions ({backend}): {output_path}")


# ============================================================================
# Detail log
# ============================================================================

def _write_e2e_summary_txt(result: Dict, path, run_config: Dict):
    """Write a shareable summary.txt with results in tabular format."""
    from pathlib import Path as _P
    path = _P(path)
    k = result.get("top_k", 10)
    agg = result.get("aggregated", {})
    n = result.get("num_instances", "?")
    elapsed = result.get("elapsed_seconds", 0)
    ts = result.get("timestamp", "")
    run_id = path.parent.name

    with open(path, "w") as f:
        f.write(f"SWE-bench E2E Eval — {run_config.get('dataset_hf', 'N/A')}\n")
        f.write(f"Run: {run_id}\n")
        f.write(f"Date: {ts[:10] if ts else 'N/A'}\n")
        f.write(f"Instances: {n} | Top-k: {k}\n")
        f.write(f"Keyword Model: {run_config.get('keyword_model', 'N/A')}\n")
        f.write(f"Patch Model: {run_config.get('patch_model', 'N/A')}\n")
        f.write(f"Patch Max Tokens: {run_config.get('patch_max_tokens', 'N/A')}")
        f.write(f" | Max File Chars: {run_config.get('max_file_chars', 'N/A')}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n\n")

        f.write(f"+{'-'*22}+{'-'*10}+{'-'*12}+{'-'*10}+{'-'*9}+\n")
        f.write(f"| {'Backend':<20s} | {'Hit@'+str(k):<8s} | {'Recall@'+str(k):<10s} | {'MRR@'+str(k):<8s} | {'Patched':<7s} |\n")
        f.write(f"+{'-'*22}+{'-'*10}+{'-'*12}+{'-'*10}+{'-'*9}+\n")
        for b in ["grep", "bm25", "grep_bm25", "llm_grep", "llm_grep_bm25"]:
            if b in agg:
                m = agg[b]
                patched = f"{m.get('patched','?')}/{n}"
                f.write(f"| {b:<20s} | {m['hit_rate']:>8.4f} | {m['recall_mean']:>10.4f} | {m['mrr_mean']:>8.4f} | {patched:>7s} |\n")
        f.write(f"+{'-'*22}+{'-'*10}+{'-'*12}+{'-'*10}+{'-'*9}+\n")


def write_detail_log(results: List[Dict], backends: List[str],
                     log_path: Path):
    """Write human-readable per-instance detail log."""
    with open(log_path, "w") as f:
        f.write(f"SWE-bench Multilingual E2E Detail Log\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Backends: {', '.join(backends)}\n")
        f.write(f"Patch model: {PATCH_MODEL}\n")
        f.write(f"Instances: {len(results)}\n")
        f.write("=" * 70 + "\n\n")

        for r in results:
            f.write(f"Instance: {r['instance_id']}\n")
            f.write(f"  Status: {r['status']}\n")
            if r["status"] != "ok":
                f.write("\n")
                continue

            for backend in backends:
                br = r["backends"].get(backend, {})
                f.write(f"  [{backend}]\n")
                f.write(f"    Retrieved: {br.get('retrieved', [])}\n")
                f.write(f"    Hit={br.get('hit', 0)} Recall={br.get('recall', 0):.3f} "
                        f"MRR={br.get('mrr', 0):.3f}\n")
                f.write(f"    Files read: {br.get('files_read', 0)}\n")
                f.write(f"    Patch: {br.get('patch_lines', 0)} lines "
                        f"({len(br.get('model_patch', ''))} chars)\n")
                f.write(f"    Times: retrieval={br.get('retrieval_time', 0)}s "
                        f"patch_gen={br.get('patch_gen_time', 0)}s\n")
            f.write("\n")


# ============================================================================
# Aggregation and reporting
# ============================================================================

def aggregate_and_print(results: List[Dict], backends: List[str], elapsed: float):
    """Print summary table and return aggregated metrics."""
    valid = [r for r in results if r["status"] == "ok"]
    if not valid:
        print("  No valid results!")
        return {}

    print(f"\n{'='*70}")
    print(f"  SWE-BENCH MULTILINGUAL E2E RESULTS")
    print(f"  Instances: {len(valid)} | Patch model: {PATCH_MODEL}")
    print(f"{'='*70}")
    print(f"\n  {'Backend':<22} {'Hit@10':<10} {'Recall@10':<12} {'MRR@10':<10} "
          f"{'Patched':<10} {'Avg Patch':<10}")
    print(f"  {'-'*74}")

    agg = {}
    for backend in backends:
        hits = [r["backends"][backend]["hit"] for r in valid]
        recalls = [r["backends"][backend]["recall"] for r in valid]
        mrrs = [r["backends"][backend]["mrr"] for r in valid]
        patch_counts = [1 for r in valid
                        if r["backends"][backend].get("model_patch", "")]
        patch_lens = [len(r["backends"][backend].get("model_patch", ""))
                      for r in valid
                      if r["backends"][backend].get("model_patch", "")]

        hit_rate = sum(hits) / len(hits) if hits else 0
        recall_mean = sum(recalls) / len(recalls) if recalls else 0
        mrr_mean = sum(mrrs) / len(mrrs) if mrrs else 0
        patched = sum(patch_counts)
        avg_patch = sum(patch_lens) / len(patch_lens) if patch_lens else 0

        print(f"  {backend:<22} {hit_rate:<10.4f} {recall_mean:<12.4f} "
              f"{mrr_mean:<10.4f} {patched:<10d} {avg_patch:<10.0f}")

        agg[backend] = {
            "hit_rate": hit_rate, "recall_mean": recall_mean,
            "mrr_mean": mrr_mean, "patched": patched,
            "avg_patch_chars": round(avg_patch),
        }

    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"{'='*70}\n")
    return agg


# ============================================================================
# Main
# ============================================================================

def main():
    global PATCH_MODEL

    import argparse
    parser = argparse.ArgumentParser(description="SWE-bench Multilingual E2E Eval")
    parser.add_argument("--limit", type=int, default=5,
                        help="Number of instances (default: 5, 0=all)")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Top-k files to retrieve (default: {TOP_K})")
    parser.add_argument("--backends", type=str, default="grep_bm25,llm_grep_bm25",
                        help="Comma-separated retrieval backends")
    parser.add_argument("--patch-model", type=str, default=None,
                        help=f"Model for patch generation (default: {PATCH_MODEL})")
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                        help="Per-query details (default: on)")
    parser.add_argument("--max-file-chars", type=int, default=12000,
                        help="Max chars per retrieved file sent to patch LLM")
    parser.add_argument("--dataset", type=str, default="multilingual",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to evaluate (default: multilingual)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip LLM caches (keyword + patch), force fresh API calls")
    args = parser.parse_args()

    global SKIP_PATCH_CACHE
    if args.no_cache:
        SKIP_PATCH_CACHE = True
        from run_swebench_search_eval import SKIP_LLM_CACHE
        import run_swebench_search_eval
        run_swebench_search_eval.SKIP_LLM_CACHE = True

    if args.patch_model:
        PATCH_MODEL = args.patch_model

    backends = [b.strip() for b in args.backends.split(",")]
    for b in backends:
        if b not in BACKEND_FUNCS:
            print(f"ERROR: Unknown backend '{b}'. Available: {ALL_BACKENDS}")
            sys.exit(1)

    # Set per-dataset directories (repos, keyword cache, bm25 cache)
    import run_swebench_search_eval
    cfg = DATASET_CONFIGS[args.dataset]
    run_swebench_search_eval.REPOS_DIR = cfg["repos_dir"]
    run_swebench_search_eval.KEYWORD_CACHE_DIR = RESULTS_DIR / cfg["results_subdir"] / "keyword_cache"
    run_swebench_search_eval.BM25_CACHE_DIR = RESULTS_DIR / cfg["results_subdir"] / "bm25_cache"
    run_swebench_search_eval.REPOS_DIR.mkdir(parents=True, exist_ok=True)

    global PATCH_CACHE_DIR
    PATCH_CACHE_DIR = RESULTS_DIR / cfg["results_subdir"] / "patch_cache"

    # Create per-run output folder under dataset subdirectory
    dataset_dir = RESULTS_DIR / DATASET_CONFIGS[args.dataset]["results_subdir"]
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = dataset_dir / f"e2e_run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tee output
    log_file = open(run_dir / "run.log", "w")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"\n{'='*70}")
    ds_label = DATASET_CONFIGS[args.dataset]["label"]
    print(f"  SWE-bench E2E Eval Pipeline — {ds_label}")
    print(f"  Run: {run_dir}")
    print(f"  Retrieval backends: {', '.join(backends)}")
    print(f"  Keyword LLM: {LLM_MODEL} @ {LLM_API_BASE}")
    print(f"  Patch LLM: {PATCH_MODEL} @ {PATCH_API_BASE}")
    print(f"  Top-k: {args.top_k} | Max file chars: {args.max_file_chars}")
    print(f"{'='*70}")

    # Save run config
    run_config = {
        "run_type": "e2e_eval",
        "run_id": f"e2e_run_{run_ts}",
        "dataset": args.dataset,
        "dataset_hf": cfg["hf_name"],
        "top_k": args.top_k,
        "backends": backends,
        "keyword_model": LLM_MODEL,
        "keyword_api_base": LLM_API_BASE,
        "keyword_max_tokens": 1024,
        "keyword_timeout_s": LLM_TIMEOUT,
        "keyword_retries": LLM_RETRIES,
        "patch_model": PATCH_MODEL,
        "patch_api_base": PATCH_API_BASE,
        "patch_max_tokens": PATCH_MAX_TOKENS,
        "patch_timeout_s": PATCH_TIMEOUT,
        "patch_retries": PATCH_RETRIES,
        "max_file_chars": args.max_file_chars,
        "timestamp_start": datetime.now().isoformat(),
    }
    import json as _json
    (run_dir / "config.json").write_text(_json.dumps(run_config, indent=2))

    # Step 1: Load data
    print(f"\n[Step 1] Loading {ds_label} ({args.limit or 'all'} instances)...")
    instances = load_dataset_instances(dataset=args.dataset, limit=args.limit)

    from collections import Counter
    repo_counts = Counter(i["repo"] for i in instances)
    for repo, count in repo_counts.most_common():
        avg_gold = sum(len(i["gold_files"]) for i in instances
                       if i["repo"] == repo) / count
        print(f"    {repo:<40s} {count:2d} instances, "
              f"avg {avg_gold:.1f} gold files")

    # Step 2: Prepare repos
    print(f"\n[Step 2] Preparing repos...")
    repos_needed = {}
    for inst in instances:
        key = (inst["repo"], inst["base_commit"])
        if key not in repos_needed:
            repos_needed[key] = []
        repos_needed[key].append(inst)

    ready = 0
    failed_repos = []
    for i, ((repo, commit), insts) in enumerate(sorted(repos_needed.items()), 1):
        repo_dir = clone_repo(repo, commit)
        if repo_dir:
            ready += 1
            print(f"  [{i}/{len(repos_needed)}] {repo} @ {commit[:12]} — "
                  f"ready ({len(insts)} instances)")
        else:
            failed_repos.append(repo)
            print(f"  [{i}/{len(repos_needed)}] {repo} @ {commit[:12]} — FAILED")
    print(f"[Prep] Done. {ready}/{len(repos_needed)} repos ready.")
    if failed_repos:
        print(f"[Prep] Failed repos: {failed_repos}")

    # Step 3: Run evaluation
    print(f"\n[Step 3] Running retrieval + patch generation...")
    t_start = time.time()
    all_results = []

    for i, inst in enumerate(instances, 1):
        result = evaluate_instance(inst, backends, top_k=args.top_k,
                                   verbose=args.verbose)
        all_results.append(result)

        # Progress
        if result["status"] == "ok":
            best_hit = max(result["backends"][b].get("hit", 0) for b in backends)
            has_patch = any(result["backends"][b].get("model_patch", "") for b in backends)
            print(f"    [{'+' if best_hit else '-'}] {i}/{len(instances)} "
                  f"{inst['instance_id']:50s} "
                  f"gold={len(inst['gold_files'])} files  "
                  f"hit={'Y' if best_hit else 'N'}  "
                  f"patch={'Y' if has_patch else 'N'}")
        else:
            print(f"    [!] {i}/{len(instances)} {inst['instance_id']} — {result['status']}")

    elapsed = time.time() - t_start

    # Step 4: Report
    agg = aggregate_and_print(all_results, backends, elapsed)

    # Step 5: Save outputs
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results JSON
    result_json = {
        "aggregated":      agg,
        "per_instance":    all_results,
        "top_k":           args.top_k,
        "num_instances":   len(instances),
        "elapsed_seconds": elapsed,
        "timestamp":       datetime.now().isoformat(),
        "config": {
            "backends":          backends,
            "keyword_model":     LLM_MODEL,
            "patch_model":       PATCH_MODEL,
            "patch_max_tokens":  PATCH_MAX_TOKENS,
            "max_file_chars":    args.max_file_chars,
            "keyword_stats":     dict(_llm_stats),
            "patch_stats":       dict(_patch_stats),
        },
    }
    json_path = run_dir / f"e2e_results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"  Results saved to {json_path}")

    # Update config.json with final stats
    run_config["num_instances"] = len(instances)
    run_config["num_repos"] = len(repos_needed)
    run_config["elapsed_seconds"] = elapsed
    run_config["timestamp_end"] = datetime.now().isoformat()
    run_config["keyword_stats"] = {k: v for k, v in _llm_stats.items() if k != "errors"}
    run_config["patch_stats"] = {k: v for k, v in _patch_stats.items() if k != "errors"}
    (run_dir / "config.json").write_text(_json.dumps(run_config, indent=2))

    # Save detail log
    detail_path = run_dir / f"e2e_detail_{ts}.txt"
    write_detail_log(all_results, backends, detail_path)
    print(f"  Detail log saved to {detail_path}")

    # Write shareable summary
    _write_e2e_summary_txt(result_json, run_dir / "summary.txt", run_config)
    print(f"  Summary saved to {run_dir / 'summary.txt'}")

    # Save predictions JSONL per backend (for swebench harness)
    for backend in backends:
        pred_path = run_dir / f"predictions_{backend}_{ts}.jsonl"
        write_predictions_jsonl(all_results, backend, pred_path)

    # Snapshot keyword and patch caches
    kw_src = _keyword_cache_path()
    if kw_src.exists():
        safe_kw = re.sub(r'[^\w.-]', '_', LLM_MODEL)
        shutil.copy2(kw_src, run_dir / f"{run_ts}_keyword_{safe_kw}.json")

    patch_src = _patch_cache_path()
    if patch_src.exists():
        safe_patch = re.sub(r'[^\w.-]', '_', PATCH_MODEL)
        shutil.copy2(patch_src, run_dir / f"{run_ts}_patch_{safe_patch}.json")

    # Step 6: Stats summary
    print(f"\n[Keyword LLM Stats] cache_hit={_llm_stats['cache_hit']}  "
          f"api_ok={_llm_stats['api_ok']}  api_fail={_llm_stats['api_fail']}")
    print(f"[Patch LLM Stats]   cache_hit={_patch_stats['cache_hit']}  "
          f"api_ok={_patch_stats['api_ok']}  api_fail={_patch_stats['api_fail']}")

    if _patch_stats["api_fail"] > 0:
        from collections import Counter
        err_counts = Counter(_patch_stats["errors"])
        print(f"[Patch LLM] Error breakdown:")
        for err, cnt in err_counts.most_common():
            print(f"  {cnt:3d}x {err}")

    print(f"\n[Next step] Run swebench harness on the predictions:")
    for backend in backends:
        pred_path = run_dir / f"predictions_{backend}_{ts}.jsonl"
        print(f"  python -m swebench.harness.run_evaluation "
              f"--predictions_path {pred_path} "
              f"--run_id e2e_{backend}")

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


if __name__ == "__main__":
    main()
