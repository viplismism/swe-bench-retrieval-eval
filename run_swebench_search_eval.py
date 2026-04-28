#!/usr/bin/env python3
"""
SWE-bench Search Eval — Single-repo code retrieval benchmark.

Uses SWE-bench Multilingual instances as ground truth:
  - Query: the issue's problem_statement
  - Ground truth: files modified in the gold patch
  - Repo: cloned at base_commit

All backends are Python-native — no external tools or Node.js required.

Backends:
  1. grep          — ripgrep / grep -rli with path-aware ranking
  2. bm25          — Python rank_bm25 (Okapi BM25)
  3. grep_bm25     — RRF fusion of grep + bm25
  4. llm_grep      — LLM keyword extraction + grep (best single method)
  5. llm_grep_bm25    — RRF fusion of llm_grep + bm25
  6. swerank          — SweRankEmbed-Small (137M, issue→code specialized)
  7. arctic           — Snowflake arctic-embed-m-long (137M, generic embeddings)
  8. hybrid           — RRF fusion of bm25 + arctic
  9. llm_grep_swerank — RRF fusion of llm_grep + swerank (lexical + semantic)

LLM config (override via env vars):
  LLM_API_BASE  OpenAI-compatible chat completions base URL
  LLM_API_KEY   API key for the keyword extraction model
  LLM_MODEL     keyword extraction model name

Usage:
    python3 run_swebench_search_eval.py                    # 10 instances
    python3 run_swebench_search_eval.py --limit 50         # 50 instances
    python3 run_swebench_search_eval.py --verbose          # per-query details
    python3 run_swebench_search_eval.py --backends grep_bm25,llm_grep_bm25
"""

import sys, os, json, time, re, subprocess, shutil, hashlib, urllib.request
from pathlib import Path
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple

# ============================================================================
# Configuration
# ============================================================================

EVAL_DIR    = Path(__file__).parent
REPOS_DIR   = EVAL_DIR / "swe_repos"     # Default; overridden per dataset in main()
RESULTS_DIR = EVAL_DIR / "swe_results"
EMBEDDINGS_DIR = EVAL_DIR / "swe_embeddings"  # Cached embeddings

TOP_K = 10
DEFAULT_LIMIT = 10

# Enable/disable embedding cache
USE_EMBEDDING_CACHE = True

# ============================================================================
# LLM configuration (used by llm_grep backend)
# ============================================================================

LLM_API_BASE    = os.environ.get("LLM_API_BASE",  "https://api.openai.com/v1")
LLM_API_KEY     = os.environ.get("LLM_API_KEY",   "")
LLM_MODEL       = os.environ.get("LLM_MODEL",      "gpt-4o-mini")
# Keyword cache: one JSON file per model under keyword_cache/<model>.json.
# Key within the file = sha256(full prompt text)[:16].  Changing the model
# switches to a different file automatically.  Changing the prompt template
# produces a different hash and re-calls the API.
KEYWORD_CACHE_DIR = RESULTS_DIR / "keyword_cache"
_keyword_cache: Optional[Dict[str, List[str]]] = None


def _keyword_cache_path() -> Path:
    """Return the cache file for the currently configured LLM_MODEL."""
    safe = re.sub(r'[^\w.-]', '_', LLM_MODEL)  # e.g. "openai/gpt-4" -> "openai_gpt-4"
    return KEYWORD_CACHE_DIR / f"{safe}.json"

# BM25 disk cache: stores tokenized corpus per (repo, commit) to skip file
# re-reading on subsequent runs.  Keyed by repo_dir path (which already embeds
# the 12-char commit prefix) so no git subprocess is needed.
BM25_CACHE_DIR = RESULTS_DIR / "bm25_cache"
_bm25_index_cache: Dict[str, Any] = {}  # in-memory: key -> {files, bm25}


def _load_keyword_cache() -> Dict[str, List[str]]:
    global _keyword_cache
    if _keyword_cache is None:
        path = _keyword_cache_path()
        if path.exists():
            try:
                _keyword_cache = json.loads(path.read_text())
            except Exception:
                _keyword_cache = {}
        else:
            _keyword_cache = {}
    return _keyword_cache


def _save_keyword_cache():
    if _keyword_cache is not None:
        path = _keyword_cache_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_keyword_cache, indent=2))
        except Exception:
            pass


# LLM call statistics (populated during run, printed at end)
_llm_stats = {"cache_hit": 0, "api_ok": 0, "api_fail": 0, "errors": []}
SKIP_LLM_CACHE = False  # Set via --no-cache to force fresh API calls

LLM_TIMEOUT   = 60   # seconds per API call
LLM_RETRIES   = 2    # retry on transient failures
LLM_RETRY_DELAY = 2  # seconds between retries


def _llm_extract_keywords(query: str, log_errors: bool = True) -> List[str]:
    """Call the LLM to extract specific technical identifiers from an issue.

    Returns a list of 6-10 strings to grep for.  Results are cached to disk
    keyed by sha256(full prompt text) so changing the prompt template
    automatically produces a different key and re-calls the API.

    Includes retry logic and detailed error logging.
    """
    # Build the prompt first so its text is part of the cache key.
    prompt = (
        "You are a code-search assistant. Given a bug report, output ONLY a "
        "JSON array of 6-10 strings: the most specific technical identifiers "
        "(exact function names, class names, method names, error message "
        "substrings, module names) that are most likely to appear "
        "verbatim in the source files that need to be changed to fix this bug. "
        "Do NOT include generic words. Output ONLY valid JSON, no explanation.\n\n"
        f"Bug report:\n{query}"
    )

    if not LLM_API_KEY:
        _llm_stats["api_fail"] += 1
        err_msg = "[LLM FAIL] LLM_API_KEY is not set"
        _llm_stats["errors"].append(err_msg)
        if log_errors:
            print(f"      {err_msg}", file=sys.__stderr__)
        return []

    cache = _load_keyword_cache()
    cache_key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    if not SKIP_LLM_CACHE and cache_key in cache:
        _llm_stats["cache_hit"] += 1
        return cache[cache_key]

    payload = json.dumps({
        "model":    LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    }).encode()

    last_error = None
    for attempt in range(1, LLM_RETRIES + 1):
        req = urllib.request.Request(
            f"{LLM_API_BASE}/chat/completions",
            data=payload, method="POST",
        )
        req.add_header("Content-Type",  "application/json")
        req.add_header("Authorization", f"Bearer {LLM_API_KEY}")

        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as resp:
                status = resp.status
                raw = resp.read().decode()
                if not raw.strip():
                    raise ValueError(f"Empty response body (HTTP {status})")
                body = json.loads(raw)
                finish = body["choices"][0].get("finish_reason", "")
                text = body["choices"][0]["message"]["content"]
                if not text:
                    raise ValueError(f"content=null, finish_reason={finish}, model={body.get('model','?')}")
                # Strip markdown fences if present
                text = re.sub(r'^```[\w]*\n?', '', text.strip())
                text = re.sub(r'\n?```$', '', text.strip())
                keywords = json.loads(text)
                if not isinstance(keywords, list):
                    raise ValueError(f"LLM returned non-list: {type(keywords)}")
                # Sanitise: strings only, non-empty, reasonable length
                keywords = [str(k).strip() for k in keywords
                            if str(k).strip() and len(str(k).strip()) <= 80]
                if not keywords:
                    raise ValueError("LLM returned empty keyword list")
                cache[cache_key] = keywords
                _save_keyword_cache()
                _llm_stats["api_ok"] += 1
                return keywords

        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code}: {e.reason}"
            # Don't retry on 4xx client errors (auth, bad request)
            if 400 <= e.code < 500:
                break
        except urllib.error.URLError as e:
            last_error = f"URLError: {e.reason}"
        except TimeoutError:
            last_error = "TimeoutError"
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e} (body[:200]={raw[:200] if 'raw' in dir() else '?'})"
        except (KeyError, IndexError, ValueError) as e:
            last_error = f"Response parse: {e}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        # Wait before retry (except on last attempt)
        if attempt < LLM_RETRIES:
            time.sleep(LLM_RETRY_DELAY)

    # All retries exhausted
    _llm_stats["api_fail"] += 1
    err_msg = f"[LLM FAIL] after {LLM_RETRIES} attempts: {last_error}"
    _llm_stats["errors"].append(err_msg)
    if log_errors:
        print(f"      {err_msg}", file=sys.__stderr__)
    return []

# Skip known binary extensions (images, compiled, archives, etc.)
BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".rar", ".7z", ".jar", ".war", ".ear",
    ".pyc", ".pyo", ".class", ".o", ".so", ".dylib", ".dll", ".exe", ".bin",
    ".wasm", ".a", ".lib", ".obj",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".flac",
    ".sqlite", ".db", ".DS_Store",
}


def _is_text_file(fpath: Path) -> bool:
    """Quick check: skip known binary extensions and files with null bytes."""
    if fpath.suffix.lower() in BINARY_EXTENSIONS:
        return False
    try:
        chunk = fpath.read_bytes()[:8192]
        return b'\x00' not in chunk
    except Exception:
        return False

# Backends to compare
ALL_BACKENDS = ["grep", "bm25", "grep_bm25", "llm_grep", "llm_grep_bm25", "swerank", "llm_grep_swerank"]
BACKENDS_NEEDING_SWERANK = {"swerank", "llm_grep_swerank"}

# SweRankEmbed model (lazy-loaded)
_swerank_model = None

def _get_swerank_model():
    """Lazy-load SweRankEmbed-Small model."""
    global _swerank_model
    if _swerank_model is None:
        from sentence_transformers import SentenceTransformer
        print("    Loading SweRankEmbed-Small model...")
        _swerank_model = SentenceTransformer(
            "Salesforce/SweRankEmbed-Small", trust_remote_code=True
        )
        print("    SweRankEmbed-Small loaded.")
    return _swerank_model



# ============================================================================
# Embedding cache helpers
# ============================================================================

def _get_cache_path(repo: str, commit: str, file_path: str, model_name: str, max_chars: int = 2000) -> Path:
    """Get the cache file path for an embedding.

    Cache structure: swe_embeddings/{repo}/{commit}/{model_name}_mc{max_chars}/{file_path}.npy
    max_chars is baked into the directory name so changing it invalidates old cache.
    """
    safe_repo = repo.replace("/", "__")
    safe_commit = commit[:12]  # First 12 chars of commit hash
    safe_path = file_path.replace("/", "__")
    cache_model = f"{model_name}_mc{max_chars}"
    return EMBEDDINGS_DIR / safe_repo / safe_commit / cache_model / f"{safe_path}.npy"


def _load_cached_embedding(cache_path: Path) -> "Optional[np.ndarray]":
    """Load embedding from disk if it exists."""
    if not USE_EMBEDDING_CACHE:
        return None
    if not cache_path.exists():
        return None
    try:
        import numpy as np
        return np.load(cache_path)
    except Exception:
        return None


def _save_embedding(cache_path: Path, embedding: "np.ndarray"):
    """Save embedding to disk."""
    if not USE_EMBEDDING_CACHE:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np
        np.save(cache_path, embedding)
    except Exception:
        pass  # Don't fail if caching fails




# ============================================================================
# SWE-bench data loading
# ============================================================================

DATASET_CACHE_FILE = RESULTS_DIR / "swebench_verified.json"

# Supported datasets
DATASET_CONFIGS = {
    "verified": {
        "hf_name": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "cache_file": RESULTS_DIR / "verified" / "swebench_verified.json",
        "label": "SWE-bench Verified",
        "results_subdir": "verified",
        "repos_dir": EVAL_DIR / "swe_repos",
    },
    "multilingual": {
        "hf_name": "SWE-bench/SWE-bench_Multilingual",
        "split": "test",
        "cache_file": RESULTS_DIR / "multilingual" / "swebench_multilingual.json",
        "label": "SWE-bench Multilingual",
        "results_subdir": "multilingual",
        "repos_dir": EVAL_DIR / "swe_repos_multilingual",
    },
}


def load_swebench_instances(limit: int = DEFAULT_LIMIT,
                            dataset: str = "verified") -> List[Dict]:
    """Load SWE-bench instances.

    Args:
        limit: max instances to return (0 = all)
        dataset: 'verified' or 'multilingual'

    First run: fetches from HuggingFace and saves a local JSON cache.
    Subsequent runs: loads from the local cache instantly (no network).
    """
    cfg = DATASET_CONFIGS[dataset]
    cache_file = cfg["cache_file"]
    label = cfg["label"]

    # Fallback: check legacy flat location (swe_results/swebench_*.json)
    if not cache_file.exists():
        legacy = RESULTS_DIR / cache_file.name
        if legacy.exists():
            cache_file = legacy

    all_instances: List[Dict] = []
    if cache_file.exists():
        print(f"  Loading {label} from local cache...")
        all_instances = json.loads(cache_file.read_text())
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing 'datasets' library...")
            subprocess.run([sys.executable, "-m", "pip", "install", "datasets"],
                           capture_output=True)
            from datasets import load_dataset

        print(f"  Loading {label} from HuggingFace ({cfg['hf_name']})...")
        ds = load_dataset(cfg["hf_name"], split=cfg["split"])

        for item in ds:
            gold_files = parse_patch_files(item["patch"])
            if not gold_files:
                continue
            all_instances.append({
                "instance_id":       item["instance_id"],
                "repo":              item["repo"],
                "base_commit":       item["base_commit"],
                "problem_statement": item["problem_statement"],
                "patch":             item["patch"],
                "gold_files":        gold_files,
            })

        # Save local cache for future runs
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(all_instances, indent=2))
        print(f"  Cached {len(all_instances)} instances to {cache_file.name}")

    # Pick diverse instances (spread across repos, limit total)
    if limit and limit < len(all_instances):
        all_instances = _pick_diverse(all_instances, limit)

    print(f"  Loaded {len(all_instances)} instances across "
          f"{len({i['repo'] for i in all_instances})} repos")
    return all_instances


def parse_patch_files(patch: str) -> List[str]:
    """Extract file paths from a unified diff patch."""
    files = set()
    for line in patch.split("\n"):
        # Match: diff --git a/path/to/file.py b/path/to/file.py
        m = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
        if m:
            files.add(m.group(2))  # use the 'b' side (after)
    return sorted(files)


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
        if idx > limit * 10:  # safety
            break

    return selected[:limit]


# ============================================================================
# Repo management
# ============================================================================

def clone_repo(repo: str, base_commit: str) -> Optional[Path]:
    """Clone a repo at a specific commit. Returns repo path or None."""
    repo_dir = REPOS_DIR / repo.replace("/", "__") / base_commit[:12]

    if repo_dir.exists() and (repo_dir / ".git" / "HEAD").exists():
        return repo_dir

    repo_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"

    try:
        # Clone with filter to speed up (no blobs until accessed)
        print(f"    Cloning {repo} (this may take a while for large repos)...")
        result = subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout",
             url, str(repo_dir)],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"    CLONE FAILED: {result.stderr[:200]}")
            return None

        # Checkout specific commit
        print(f"    Checking out {base_commit[:12]}...")
        result = subprocess.run(
            ["git", "checkout", base_commit],
            capture_output=True, text=True, timeout=600,
            cwd=str(repo_dir),
        )
        if result.returncode != 0:
            print(f"    CHECKOUT FAILED: {result.stderr[:200]}")
            return None

        return repo_dir
    except subprocess.TimeoutExpired:
        print(f"    CLONE TIMEOUT for {repo} (>10 min)")
        return None


# ============================================================================
# Search backends
# ============================================================================

# SweRank chunking: split files into overlapping windows so the model sees
# the full file regardless of length. Score = max chunk similarity.
SWERANK_CHUNK_SIZE    = 8000   # chars per chunk ≈ 2048 tokens (fits 8192 token limit)
SWERANK_CHUNK_OVERLAP = 1000   # overlap between consecutive chunks


def _chunk_text(text: str) -> List[str]:
    """Split text into overlapping fixed-size character chunks."""
    if len(text) <= SWERANK_CHUNK_SIZE:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + SWERANK_CHUNK_SIZE])
        if start + SWERANK_CHUNK_SIZE >= len(text):
            break
        start += SWERANK_CHUNK_SIZE - SWERANK_CHUNK_OVERLAP
    return chunks


def _collect_code_files(repo_dir: Path) -> Tuple[List[str], List[str]]:
    """Collect all text files from a repo directory (full content, no truncation).

    Returns (rel_paths, contents). Used by bm25 backend.
    """
    files = []
    contents = []
    for fpath in sorted(repo_dir.rglob("*")):
        if not fpath.is_file():
            continue
        if ".git" in fpath.parts:
            continue
        if not _is_text_file(fpath):
            continue
        try:
            text = fpath.read_text(errors="ignore")
            if text.strip():
                rel = str(fpath.relative_to(repo_dir))
                files.append(rel)
                contents.append(text)
        except Exception:
            continue
    return files, contents


def _collect_code_files_full(repo_dir: Path) -> Tuple[List[str], List[str]]:
    """Collect all text files, full content (no truncation). Used by swerank chunking."""
    files: List[str] = []
    contents: List[str] = []
    for fpath in sorted(repo_dir.rglob("*")):
        if not fpath.is_file():
            continue
        if ".git" in fpath.parts:
            continue
        if not _is_text_file(fpath):
            continue
        try:
            text = fpath.read_text(errors="ignore")
            if text.strip():
                files.append(str(fpath.relative_to(repo_dir)))
                contents.append(text)
        except Exception:
            continue
    return files, contents


def get_or_compute_chunk_embeddings(
    repo: str,
    commit: str,
    repo_dir: Path,
    model,
    model_name: str = "swerank_chunked",
) -> Tuple[List[str], List["np.ndarray"]]:
    """Embed all code files in overlapping chunks with no char limit.

    Each file is split into SWERANK_CHUNK_SIZE-char windows. All chunks are
    encoded in one batch. Per-file cache stores shape (num_chunks, dim).
    Returns (file_paths, list_of_chunk_arrays).
    """
    import numpy as np

    files, contents = _collect_code_files_full(repo_dir)
    if not files:
        return [], []

    result_embs: List[Optional[np.ndarray]] = [None] * len(files)
    files_to_compute: List[int] = []

    for i, fpath in enumerate(files):
        cache_path = _get_cache_path(repo, commit, fpath, model_name, max_chars=0)
        cached = _load_cached_embedding(cache_path)
        if cached is not None:
            result_embs[i] = cached.reshape(-1, cached.shape[-1]) if cached.ndim == 1 else cached
        else:
            files_to_compute.append(i)

    if files_to_compute:
        # Build flat chunk list, track which file each chunk belongs to
        all_chunks: List[str] = []
        chunk_ranges: List[Tuple[int, int]] = []  # (start, end) in all_chunks per file
        for i in files_to_compute:
            start = len(all_chunks)
            for chunk in _chunk_text(contents[i]):
                all_chunks.append(chunk)
            chunk_ranges.append((start, len(all_chunks)))

        # Encode all chunks in one flat batch
        all_embs = model.encode(all_chunks, batch_size=8, show_progress_bar=False)

        for j, i in enumerate(files_to_compute):
            s, e = chunk_ranges[j]
            file_embs = np.array(all_embs[s:e])  # shape: (num_chunks, dim)
            result_embs[i] = file_embs
            cache_path = _get_cache_path(repo, commit, files[i], model_name, max_chars=0)
            _save_embedding(cache_path, file_embs)

    return files, result_embs  # type: ignore


def search_grep(query: str, repo_dir: Path, top_k: int = TOP_K) -> List[str]:
    """Grep-based search: extract keywords, search for files containing them.

    Scoring: content_hits + path_match_bonus.
    If a keyword matches a filename or directory component, that file gets
    a large boost so it floats to the top regardless of content hits.
    """
    # Extract meaningful words (skip short/common ones)
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "has", "have", "had", "do", "does", "did", "will", "would",
                  "could", "should", "may", "might", "can", "shall", "not",
                  "and", "or", "but", "if", "then", "else", "when", "where",
                  "how", "what", "which", "who", "whom", "this", "that",
                  "these", "those", "it", "its", "in", "on", "at", "to",
                  "for", "of", "with", "by", "from", "as", "into", "about",
                  "after", "before", "between", "under", "over", "i", "we",
                  "you", "he", "she", "they", "me", "us", "him", "her",
                  "them", "my", "our", "your", "his", "their", "so", "no",
                  "yes", "up", "out", "also", "just", "than", "some", "any",
                  "all", "each", "every", "both", "few", "more", "most",
                  "other", "such", "only", "same", "very", "too"}

    words = re.findall(r'\b[a-zA-Z_]\w{2,}\b', query.lower())
    # Deduplicate and sort by length desc (longer = more specific/discriminative)
    seen = set()
    unique_words = []
    for w in words:
        if w not in stop_words and w not in seen:
            seen.add(w)
            unique_words.append(w)
    unique_words.sort(key=lambda w: -len(w))
    keywords = unique_words[:12]  # top 12 keywords

    # Also extract path-like tokens from query (e.g. "utils/helpers.py",
    # "timeseries", "html", module names). These are words that look like
    # filenames or directory names — used for path-matching boost.
    path_tokens = set()
    # Dotted identifiers like "ast.literal_eval" -> "ast", "literal_eval"
    for m in re.finditer(r'[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+', query):
        for part in m.group().split('.'):
            if len(part) >= 3:
                path_tokens.add(part.lower())
    # Slash-delimited paths like "utils/helpers.py"
    for m in re.finditer(r'[a-zA-Z_]\w*(?:/[a-zA-Z_]\w*)+(?:\.\w+)?', query):
        for part in m.group().replace('\\', '/').split('/'):
            cleaned = part.split('.')[0]  # strip extension
            if len(cleaned) >= 3:
                path_tokens.add(cleaned.lower())
    # File-like tokens with extensions (e.g. "helpers.py", "config.yaml")
    for m in re.finditer(r'\b([a-zA-Z_]\w+)\.(py|js|ts|java|go|rb|rs|c|cpp|h)\b', query):
        path_tokens.add(m.group(1).lower())
    # Add all keywords as potential path tokens too
    path_tokens.update(keywords)

    if not keywords:
        return []

    # Use grep -rli for each keyword, count file occurrences
    file_hits: Dict[str, int] = {}
    for kw in keywords:
        try:
            result = subprocess.run(
                ["grep", "-rli", "--binary-files=without-match",
                 kw, str(repo_dir)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        rel = os.path.relpath(line.strip(), repo_dir)
                        file_hits[rel] = file_hits.get(rel, 0) + 1
        except (subprocess.TimeoutExpired, Exception):
            continue

    # Score = content_hits + path_match_bonus
    # Path bonus: +100 per path component that matches a path token.
    # This ensures files with matching paths rank above those that only
    # have keyword content hits.
    scored: Dict[str, float] = {}
    for fpath, hits in file_hits.items():
        bonus = 0
        path_parts = fpath.lower().replace('\\', '/').split('/')
        # Check each path component (including filename without extension)
        for part in path_parts:
            stem = part.rsplit('.', 1)[0] if '.' in part else part
            # Exact match of stem against a path token
            if stem in path_tokens:
                bonus += 100
            # Also check sub-components of snake_case names
            for sub in stem.split('_'):
                if len(sub) >= 3 and sub in path_tokens:
                    bonus += 50
        scored[fpath] = hits + bonus

    ranked = sorted(scored.items(), key=lambda x: -x[1])
    return [f for f, _ in ranked[:top_k]]


def _bm25_tokenize(text: str) -> List[str]:
    return re.findall(r'[a-zA-Z_]\w{2,}', text.lower())


def _get_bm25_index(repo_dir: Path):
    """Return (files, BM25Okapi) for repo_dir.

    Lookup order:
      1. In-memory cache (free — same Python process, same instance)
      2. Disk cache under bm25_cache/ (fast — skip file I/O on re-runs)
      3. Build from scratch, then write both caches

    The repo_dir path already encodes the 12-char commit prefix
    (swe_repos/<repo>/<commit12>/), so no git subprocess is needed.
    At most one entry is kept in memory to bound RAM usage.
    """
    import pickle
    from rank_bm25 import BM25Okapi

    cache_key = str(repo_dir)  # unique per repo+commit because path encodes commit
    if cache_key in _bm25_index_cache:
        return _bm25_index_cache[cache_key]["files"], _bm25_index_cache[cache_key]["bm25"]

    # Disk cache path: bm25_cache/<repo_slug>_<commit12>.pkl
    disk_path = BM25_CACHE_DIR / f"{repo_dir.parent.name}_{repo_dir.name}.pkl"
    files: List[str] = []
    bm25 = None

    if disk_path.exists():
        try:
            data = pickle.loads(disk_path.read_bytes())
            files = data["files"]
            corpus = data["corpus"]
            bm25 = BM25Okapi(corpus) if corpus else None
        except Exception:
            files, bm25 = [], None  # corrupt cache — fall through to rebuild

    if not files:  # cache miss or corrupt
        file_contents_list: List[str]
        files, file_contents_list = _collect_code_files(repo_dir)
        corpus = [_bm25_tokenize(c) for c in file_contents_list]
        bm25 = BM25Okapi(corpus) if corpus else None
        try:
            BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            disk_path.write_bytes(pickle.dumps({"files": files, "corpus": corpus}))
        except Exception:
            pass  # disk write failure is non-fatal

    _bm25_index_cache.clear()  # keep only the most recent repo to bound RAM
    _bm25_index_cache[cache_key] = {"files": files, "bm25": bm25}
    return files, bm25


def search_bm25_python(query: str, repo_dir: Path, top_k: int = TOP_K) -> List[str]:
    """Python-native BM25 search using rank_bm25.

    Builds (or reuses) a cached BM25 index for the current repo commit, so
    multiple backends running on the same instance only pay the indexing cost once.
    """
    files, bm25 = _get_bm25_index(repo_dir)
    if not files or bm25 is None:
        return []

    query_tokens = _bm25_tokenize(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    top_indices = scores.argsort()[::-1][:top_k]
    return [files[i] for i in top_indices]



def search_swerank(query: str, repo_dir: Path, top_k: int = TOP_K,
                   repo: str = None, commit: str = None) -> List[str]:
    """Search using SweRankEmbed-Small with full-file chunked embeddings.

    Files are split into overlapping 8000-char chunks (no truncation).
    Each file is scored by its best-matching chunk. Ranks by max chunk similarity.
    Always returns exactly top_k results.
    """
    import numpy as np

    model = _get_swerank_model()

    if repo and commit:
        files, chunk_embs_list = get_or_compute_chunk_embeddings(
            repo, commit, repo_dir, model
        )
    else:
        # No cache available — compute on the fly
        files, contents = _collect_code_files_full(repo_dir)
        if not files:
            return []
        all_chunks: List[str] = []
        chunk_ranges: List[Tuple[int, int]] = []
        for text in contents:
            s = len(all_chunks)
            for chunk in _chunk_text(text):
                all_chunks.append(chunk)
            chunk_ranges.append((s, len(all_chunks)))
        all_embs = model.encode(all_chunks, batch_size=8, show_progress_bar=False)
        chunk_embs_list = [
            np.array(all_embs[s:e]) for s, e in chunk_ranges
        ]

    if not files:
        return []

    # Encode query with prompt_name="query" (adds task instruction prefix)
    query_emb = model.encode([query], prompt_name="query")[0]  # shape: (dim,)

    # Score each file = max cosine similarity across its chunks
    file_scores = np.array([
        float(np.dot(chunk_embs, query_emb).max())
        for chunk_embs in chunk_embs_list
    ])

    top_indices = np.argsort(-file_scores)[:top_k]
    return [files[i] for i in top_indices]


def search_grep_bm25(query: str, repo_dir: Path,
                     top_k: int = TOP_K) -> List[str]:
    """Fuse grep + BM25 results via reciprocal rank fusion (RRF).

    RRF score = sum(1 / (k + rank)) across lists, where k=60 (standard).
    This naturally interleaves results, giving higher weight to files
    that appear in both lists and preserving rank ordering.
    """
    grep_results = search_grep(query, repo_dir, top_k * 2)   # fetch more to fuse
    bm25_results = search_bm25_python(query, repo_dir, top_k * 2)

    RRF_K = 60
    scores: Dict[str, float] = {}
    for rank, f in enumerate(grep_results):
        scores[f] = scores.get(f, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, f in enumerate(bm25_results):
        scores[f] = scores.get(f, 0) + 1.0 / (RRF_K + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [f for f, _ in ranked[:top_k]]


def search_llm_grep(query: str, repo_dir: Path,
                    top_k: int = TOP_K) -> Tuple[List[str], List[str]]:
    """LLM-assisted grep: use an LLM to extract precise identifiers, then grep.

    Returns (results, keywords) so the keywords can be logged for analysis.
    Falls back to vanilla grep keywords if the LLM call fails.
    """
    keywords = _llm_extract_keywords(query)

    if not keywords:
        # LLM failed — fall back to the heuristic extractor so we always
        # return something useful rather than an empty list.
        return search_grep(query, repo_dir, top_k), []

    # Grep for each LLM keyword, count file hits
    file_hits: Dict[str, int] = {}
    for kw in keywords:
        try:
            result = subprocess.run(
                ["grep", "-rli", "-F",   # -F = literal string (no regex)
                 "--binary-files=without-match",
                 kw, str(repo_dir)],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        rel = os.path.relpath(line.strip(), repo_dir)
                        file_hits[rel] = file_hits.get(rel, 0) + 1
        except (subprocess.TimeoutExpired, Exception):
            continue

    if not file_hits:
        return [], keywords

    # Path bonus: same logic as search_grep — keywords that match path components
    # get a large boost so "admin/options.py" rises when 'admin' is a keyword.
    path_tokens = set(k.lower() for k in keywords)
    scored: Dict[str, float] = {}
    for fpath, hits in file_hits.items():
        bonus = 0
        for part in fpath.lower().replace('\\', '/').split('/'):
            stem = part.rsplit('.', 1)[0] if '.' in part else part
            if stem in path_tokens:
                bonus += 100
            for sub in stem.split('_'):
                if len(sub) >= 3 and sub in path_tokens:
                    bonus += 50
        scored[fpath] = hits + bonus

    ranked = sorted(scored.items(), key=lambda x: -x[1])
    return [f for f, _ in ranked[:top_k]], keywords


def search_llm_grep_bm25(query: str, repo_dir: Path,
                         top_k: int = TOP_K) -> List[str]:
    """RRF fusion of LLM-grep + full-file BM25."""
    llm_results, _ = search_llm_grep(query, repo_dir, top_k * 2)
    bm25_results   = search_bm25_python(query, repo_dir, top_k * 2)

    RRF_K = 60
    scores: Dict[str, float] = {}
    for rank, f in enumerate(llm_results):
        scores[f] = scores.get(f, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, f in enumerate(bm25_results):
        scores[f] = scores.get(f, 0) + 1.0 / (RRF_K + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [f for f, _ in ranked[:top_k]]


def search_llm_grep_swerank(query: str, repo_dir: Path,
                            top_k: int = TOP_K,
                            repo: str = None, commit: str = None) -> List[str]:
    """RRF fusion of LLM-grep + SweRankEmbed.

    LLM-grep fires on explicit identifier mentions in the issue.
    SweRank fires on semantic intent even without identifier overlap.
    Low failure correlation = genuine complementarity.
    """
    llm_results, _    = search_llm_grep(query, repo_dir, top_k * 2)
    swerank_results   = search_swerank(query, repo_dir, top_k * 2, repo, commit)

    RRF_K = 60
    scores: Dict[str, float] = {}
    for rank, f in enumerate(llm_results):
        scores[f] = scores.get(f, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, f in enumerate(swerank_results):
        scores[f] = scores.get(f, 0) + 1.0 / (RRF_K + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [f for f, _ in ranked[:top_k]]


# ============================================================================
# Metrics
# ============================================================================

def _normalize_path(p: str) -> str:
    """Normalize a file path for comparison."""
    return p.lower().strip()


def recall_at_k(predicted: List[str], gold: List[str], k: int) -> float:
    """Fraction of gold files found in top-k predictions."""
    if not gold:
        return 0.0
    pred_set = set(_normalize_path(p) for p in predicted[:k])
    gold_set = set(_normalize_path(g) for g in gold)
    found = len(pred_set & gold_set)
    return found / len(gold_set)


def hit_at_k(predicted: List[str], gold: List[str], k: int) -> float:
    """1.0 if any gold file appears in top-k, else 0.0."""
    pred_set = set(_normalize_path(p) for p in predicted[:k])
    gold_set = set(_normalize_path(g) for g in gold)
    return 1.0 if pred_set & gold_set else 0.0


def mrr_at_k(predicted: List[str], gold: List[str], k: int) -> float:
    """Reciprocal rank of first gold file in top-k."""
    gold_set = set(_normalize_path(g) for g in gold)
    for i, p in enumerate(predicted[:k]):
        if _normalize_path(p) in gold_set:
            return 1.0 / (i + 1)
    return 0.0


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate_instance(inst: Dict, repo_dir: Path,
                      top_k: int = TOP_K,
                      verbose: bool = False,
                      backends: List[str] = None) -> Dict:
    """Run all backends on one SWE-bench instance."""
    backends = backends or ALL_BACKENDS
    query      = inst["problem_statement"]
    gold_files = inst["gold_files"]
    repo       = inst.get("repo")
    commit     = inst.get("base_commit")

    # Clean query: collapse whitespace
    query = " ".join(query.split())

    results = {}
    llm_kws = []  # track LLM keywords across backends for logging
    for backend in backends:
        if backend == "grep":
            predicted = search_grep(query, repo_dir, top_k)
        elif backend == "bm25":
            predicted = search_bm25_python(query, repo_dir, top_k)
        elif backend == "grep_bm25":
            predicted = search_grep_bm25(query, repo_dir, top_k)
        elif backend == "llm_grep":
            predicted, llm_kws = search_llm_grep(query, repo_dir, top_k)
            if verbose and llm_kws:
                print(f"      LLM keywords: {llm_kws}")
            elif verbose and not llm_kws:
                print(f"      LLM keywords: (fallback to heuristic grep)")
        elif backend == "llm_grep_bm25":
            predicted = search_llm_grep_bm25(query, repo_dir, top_k)
        elif backend == "swerank":
            predicted = search_swerank(query, repo_dir, top_k, repo, commit)
        elif backend == "llm_grep_swerank":
            predicted = search_llm_grep_swerank(query, repo_dir, top_k, repo, commit)
        else:
            predicted = []

        r = recall_at_k(predicted, gold_files, top_k)
        h = hit_at_k(predicted, gold_files, top_k)
        m = mrr_at_k(predicted, gold_files, top_k)

        # Find rank of first gold file hit (0 = not found)
        gold_norm = set(_normalize_path(g) for g in gold_files)
        first_rank = 0
        for i, p in enumerate(predicted[:top_k]):
            if _normalize_path(p) in gold_norm:
                first_rank = i + 1
                break

        results[backend] = {
            "recall": r,
            "hit": h,
            "mrr": m,
            "predicted": predicted[:top_k],  # keep full top-k for analysis
            "first_gold_rank": first_rank,
        }

        if verbose:
            status = "+" if h > 0 else "x"
            print(f"      [{status}] {backend:<20s} Hit={h:.0f}  "
                  f"Recall={r:.2f}  MRR={m:.2f}  "
                  f"Rank={first_rank or '-'}")

    if verbose:
        # Show predictions from each backend
        print(f"      ---- Predictions ----")
        for backend in backends:
            preds = results[backend]["predicted"]
            tag = "+" if results[backend]["hit"] > 0 else "x"
            print(f"      {backend} [{tag}]:")
            for i, p in enumerate(preds):
                marker = " *" if _normalize_path(p) in gold_norm else ""
                print(f"        {i+1:2d}. {p}{marker}")
            if not preds:
                print(f"        (no results)")

    # Store query snippet for the log
    query_snippet = query[:300] + ("..." if len(query) > 300 else "")

    return {
        "instance_id":   inst["instance_id"],
        "repo":          inst["repo"],
        "gold_files":    gold_files,
        "num_gold":      len(gold_files),
        "query_len":     len(inst["problem_statement"]),
        "query_snippet": query_snippet,
        "llm_keywords":  llm_kws if llm_kws else None,
        "backends":      results,
    }


def _prepare_repos(instances: List[Dict], backends: List[str]) -> Dict[tuple, Path]:
    """Pre-check all repos and pre-warm BM25 caches before evaluation.

    Returns {(repo, commit): repo_dir} for successfully prepared repos.
    """
    from collections import defaultdict
    by_repo = defaultdict(list)
    for inst in instances:
        by_repo[(inst["repo"], inst["base_commit"])].append(inst)

    need_bm25 = bool({"bm25", "grep_bm25", "llm_grep_bm25"} & set(backends))
    need_llm  = bool({"llm_grep", "llm_grep_bm25", "llm_grep_swerank"} & set(backends))
    repo_dirs: Dict[tuple, Path] = {}

    print(f"\n[Prep] Checking {len(by_repo)} repo checkouts and warming caches...")
    for i, ((repo, commit), repo_instances) in enumerate(by_repo.items(), 1):
        repo_dir = clone_repo(repo, commit)
        if not repo_dir:
            print(f"  [{i}/{len(by_repo)}] {repo} @ {commit[:12]} — CLONE FAILED")
            continue
        repo_dirs[(repo, commit)] = repo_dir

        # Pre-warm BM25 index (builds from files or loads from disk cache)
        if need_bm25:
            _get_bm25_index(repo_dir)

        print(f"  [{i}/{len(by_repo)}] {repo} @ {commit[:12]} — "
              f"ready ({len(repo_instances)} instances)")

    print(f"[Prep] Done. {len(repo_dirs)}/{len(by_repo)} repos ready.")

    # Pre-warm LLM keyword cache for all instances (avoids cold API calls
    # during the timed eval loop).  Each call is cached to disk, so only
    # uncached prompts hit the API.
    if need_llm:
        cache = _load_keyword_cache()
        uncached = []
        for inst in instances:
            query = " ".join(inst["problem_statement"].split())
            prompt = (
                "You are a code-search assistant. Given a bug report, output ONLY a "
                "JSON array of 6-10 strings: the most specific technical identifiers "
                "(exact function names, class names, method names, error message "
                "substrings, module names) that are most likely to appear "
                "verbatim in the source files that need to be changed to fix this bug. "
                "Do NOT include generic words. Output ONLY valid JSON, no explanation.\n\n"
                f"Bug report:\n{query}"
            )
            cache_key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            if cache_key not in cache:
                uncached.append(inst)

        if uncached:
            print(f"\n[Prep] Warming LLM keyword cache: {len(uncached)} "
                  f"uncached of {len(instances)} instances...")
            for j, inst in enumerate(uncached, 1):
                query = " ".join(inst["problem_statement"].split())
                kws = _llm_extract_keywords(query)
                status = f"{len(kws)} kw" if kws else "FAIL"
                if j % 25 == 0 or j == len(uncached):
                    print(f"  [LLM {j}/{len(uncached)}] {inst['instance_id']}: {status}")
                # Rate-limit: small pause between API calls to avoid throttling
                if kws or j < len(uncached):
                    time.sleep(0.5)
            ok = _llm_stats["api_ok"]
            fail = _llm_stats["api_fail"]
            print(f"[Prep] LLM cache warm-up done: {ok} OK, {fail} failed, "
                  f"{_llm_stats['cache_hit']} already cached")
            if fail > 0:
                print(f"[Prep] LLM failures breakdown:")
                from collections import Counter
                err_counts = Counter(_llm_stats["errors"])
                for err, cnt in err_counts.most_common(5):
                    print(f"  {cnt:3d}x {err}")
        else:
            print(f"[Prep] LLM keyword cache: all {len(instances)} instances already cached.")

    print()
    return repo_dirs


def run_evaluation(instances: List[Dict], top_k: int = TOP_K,
                   verbose: bool = False,
                   backends: List[str] = None,
                   instance_offset: int = 0) -> Dict:
    """Run full SWE-bench search eval."""
    backends = backends or ALL_BACKENDS
    need_swerank = bool(set(backends) & BACKENDS_NEEDING_SWERANK)
    print(f"\n{'='*70}")
    print(f"  SWE-bench Search Eval — Single-Repo Retrieval (Python-native)")
    print(f"  Instances: {len(instances)} | Top-k: {top_k}")
    print(f"  Backends: {', '.join(backends)}")
    if need_swerank:
        print(f"  SweRank model: Salesforce/SweRankEmbed-Small (137M)")
    print(f"{'='*70}\n")

    # Pre-load models if needed (so they're warm for all instances)
    if need_swerank:
        _get_swerank_model()

    # Pre-check repos + warm BM25 caches before eval loop
    repo_dirs = _prepare_repos(instances, backends)

    all_results = []
    start = time.time()

    # Group by repo
    from collections import defaultdict
    by_repo = defaultdict(list)
    for inst in instances:
        by_repo[(inst["repo"], inst["base_commit"])].append(inst)

    instance_num = 0
    total_display = len(instances) + instance_offset
    for (repo, commit), repo_instances in by_repo.items():
        print(f"\n  [{repo}] commit={commit[:12]} ({len(repo_instances)} instances)")

        repo_dir = repo_dirs.get((repo, commit))
        if not repo_dir:
            print(f"    SKIP — repo not available")
            for inst in repo_instances:
                all_results.append({
                    "instance_id": inst["instance_id"],
                    "repo": repo,
                    "gold_files": inst["gold_files"],
                    "num_gold": len(inst["gold_files"]),
                    "error": "clone_failed",
                })
            continue

        # Evaluate each instance
        for inst in repo_instances:
            instance_num += 1
            display_num = instance_num + instance_offset
            if verbose:
                print(f"\n    [{display_num}/{total_display}] "
                      f"{inst['instance_id']}")
                q_preview = ' '.join(inst['problem_statement'].split())[:200]
                print(f"      Query: {q_preview}...")
                print(f"      Gold:  {inst['gold_files']}")

            r = evaluate_instance(inst, repo_dir, top_k, verbose, backends)
            all_results.append(r)

            if not verbose:
                # Brief progress
                hits = {b: r["backends"][b]["hit"]
                        for b in backends if "backends" in r}
                best = max(hits.values()) if hits else 0
                status = "+" if best > 0 else "x"
                print(f"    [{status}] {instance_num}/{len(instances)} "
                      f"{inst['instance_id']:<45s} "
                      f"gold={len(inst['gold_files'])} files  "
                      f"best_hit={best:.0f}")

    elapsed = time.time() - start

    # Aggregate
    agg = aggregate_results(all_results)
    return {
        "aggregated":      agg,
        "per_instance":    all_results,
        "top_k":           top_k,
        "num_instances":   len(instances),
        "elapsed_seconds": elapsed,
        "timestamp":       datetime.now().isoformat(),
        "config": {
            "swerank_model":   "Salesforce/SweRankEmbed-Small" if need_swerank else None,
            "backends":        backends,
            "llm_model":       LLM_MODEL,
            "llm_stats":       dict(_llm_stats),
            "bm25_caches_used": sorted({
                f"{inst['repo'].replace('/', '__')}_{inst['base_commit'][:12]}.pkl"
                for inst in instances
                if (BM25_CACHE_DIR / f"{inst['repo'].replace('/', '__')}_{inst['base_commit'][:12]}.pkl").exists()
            }) if BM25_CACHE_DIR.exists() else [],
        },
    }


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate metrics across all instances, per backend."""
    valid = [r for r in results if "backends" in r]
    if not valid:
        return {}

    agg = {}
    backends = list(valid[0]["backends"].keys()) if valid else []
    for backend in backends:
        recalls = [r["backends"][backend]["recall"] for r in valid]
        hits    = [r["backends"][backend]["hit"] for r in valid]
        mrrs    = [r["backends"][backend]["mrr"] for r in valid]

        agg[backend] = {
            "recall_mean":  sum(recalls) / len(recalls),
            "hit_rate":     sum(hits) / len(hits),
            "mrr_mean":     sum(mrrs) / len(mrrs),
            "num_instances": len(valid),
        }

    return agg


# ============================================================================
# Reporting
# ============================================================================

def print_results(result: Dict):
    k   = result["top_k"]
    agg = result["aggregated"]
    n   = result["num_instances"]

    print(f"\n{'='*70}")
    print(f"  SWE-BENCH SEARCH EVAL RESULTS")
    print(f"  Instances: {n} | Top-k: {k}")
    print(f"{'='*70}")
    print(f"\n  {'Backend':<22} {'Hit@'+str(k):<10} {'Recall@'+str(k):<12} "
          f"{'MRR@'+str(k):<10}")
    print(f"  {'-'*56}")

    for backend in ALL_BACKENDS:
        if backend in agg:
            m = agg[backend]
            print(f"  {backend:<22} {m['hit_rate']:<10.4f} "
                  f"{m['recall_mean']:<12.4f} {m['mrr_mean']:<10.4f}")

    print(f"\n  Elapsed: {result['elapsed_seconds']:.1f}s")
    print(f"{'='*70}\n")


def save_results(result: Dict, run_dir: Path = None):
    out_dir = run_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"swe_search_eval_{ts}.json"

    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Results saved to {path}")

    # Also write human-readable detail log
    log_path = out_dir / f"swe_search_detail_{ts}.txt"
    _write_detail_log(result, log_path)
    print(f"  Detail log saved to {log_path}")

    # Write shareable summary
    _write_summary_txt(result, out_dir / "summary.txt")
    print(f"  Summary saved to {out_dir / 'summary.txt'}")

    return path


def _write_summary_txt(result: Dict, path: Path):
    """Write a shareable summary.txt with results in tabular format."""
    k = result["top_k"]
    cfg = result.get("config", {})
    agg = result.get("aggregated", {})
    has_patches = any("patched" in agg[b] for b in agg)
    run_id = path.parent.name
    n = result.get("num_instances", "?")
    elapsed = result.get("elapsed_seconds", 0)
    ts = result.get("timestamp", "")

    with open(path, "w") as f:
        label = cfg.get("dataset_label", "SWE-bench Search Eval")
        f.write(f"SWE-bench Search Eval\n")
        f.write(f"Run: {run_id}\n")
        f.write(f"Date: {ts[:10] if ts else 'N/A'}\n")
        f.write(f"Instances: {n} | Top-k: {k}\n")
        kw = cfg.get("llm_model", cfg.get("keyword_model", "N/A"))
        f.write(f"Keyword Model: {kw}\n")
        f.write(f"Elapsed: {elapsed:.1f}s\n\n")

        # Table
        if has_patches:
            f.write(f"+{'-'*22}+{'-'*10}+{'-'*12}+{'-'*10}+{'-'*9}+\n")
            f.write(f"| {'Backend':<20s} | {'Hit@'+str(k):<8s} | {'Recall@'+str(k):<10s} | {'MRR@'+str(k):<8s} | {'Patched':<7s} |\n")
            f.write(f"+{'-'*22}+{'-'*10}+{'-'*12}+{'-'*10}+{'-'*9}+\n")
            for b in ["grep", "bm25", "grep_bm25", "llm_grep", "llm_grep_bm25"]:
                if b in agg:
                    m = agg[b]
                    patched = f"{m.get('patched','?')}/{n}"
                    f.write(f"| {b:<20s} | {m['hit_rate']:>8.4f} | {m['recall_mean']:>10.4f} | {m['mrr_mean']:>8.4f} | {patched:>7s} |\n")
            f.write(f"+{'-'*22}+{'-'*10}+{'-'*12}+{'-'*10}+{'-'*9}+\n")
        else:
            f.write(f"+{'-'*22}+{'-'*10}+{'-'*12}+{'-'*10}+\n")
            f.write(f"| {'Backend':<20s} | {'Hit@'+str(k):<8s} | {'Recall@'+str(k):<10s} | {'MRR@'+str(k):<8s} |\n")
            f.write(f"+{'-'*22}+{'-'*10}+{'-'*12}+{'-'*10}+\n")
            for b in ["grep", "bm25", "grep_bm25", "llm_grep", "llm_grep_bm25"]:
                if b in agg:
                    m = agg[b]
                    f.write(f"| {b:<20s} | {m['hit_rate']:>8.4f} | {m['recall_mean']:>10.4f} | {m['mrr_mean']:>8.4f} |\n")
            f.write(f"+{'-'*22}+{'-'*10}+{'-'*12}+{'-'*10}+\n")


def _write_detail_log(result: Dict, log_path: Path):
    """Write a human-readable per-instance detail log."""
    k = result["top_k"]
    instances = result["per_instance"]
    backends = list(result["aggregated"].keys()) if result.get("aggregated") else []

    with open(log_path, "w") as f:
        f.write(f"SWE-bench Search Eval Detail Log\n")
        f.write(f"Generated: {result.get('timestamp', 'N/A')}\n")
        f.write(f"Instances: {result['num_instances']}  Top-k: {k}\n")
        f.write(f"Backends: {', '.join(backends)}\n")
        f.write(f"{'='*80}\n\n")

        for inst in instances:
            f.write(f"--- {inst['instance_id']} ---\n")
            f.write(f"Repo: {inst['repo']}\n")
            if "query_snippet" in inst:
                f.write(f"Query: {inst['query_snippet']}\n")
            f.write(f"Gold ({inst['num_gold']}): {inst['gold_files']}\n")
            if inst.get('llm_keywords'):
                f.write(f"LLM Keywords: {inst['llm_keywords']}\n")

            if "error" in inst:
                f.write(f"ERROR: {inst['error']}\n\n")
                continue

            # Summary line
            hits = {b: inst['backends'][b]['hit'] for b in backends
                    if b in inst.get('backends', {})}
            summary = "  ".join(f"{b}={'HIT' if h else 'MISS'}"
                                for b, h in hits.items())
            f.write(f"Summary: {summary}\n")

            # Per-backend predictions
            for b in backends:
                if b not in inst.get('backends', {}):
                    continue
                bd = inst['backends'][b]
                tag = "HIT" if bd['hit'] else "MISS"
                rank = bd.get('first_gold_rank', 0)
                rank_str = str(rank) if rank else "-"
                f.write(f"\n  {b} [{tag}] "
                        f"Recall={bd['recall']:.2f} MRR={bd['mrr']:.2f} "
                        f"GoldRank={rank_str}\n")
                gold_norm = set(_normalize_path(g) for g in inst['gold_files'])
                preds = bd.get('predicted', [])
                for i, p in enumerate(preds):
                    marker = " << GOLD" if _normalize_path(p) in gold_norm else ""
                    f.write(f"    {i+1:2d}. {p}{marker}\n")
                if not preds:
                    f.write(f"    (no results)\n")

            f.write(f"\n")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SWE-bench Search Eval: compare retrieval backends")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Number of instances (default: {DEFAULT_LIMIT})")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"Top-k cutoff (default: {TOP_K})")
    parser.add_argument("--verbose", "-v", action="store_true", default=True,
                        help="Per-query per-backend details (default: on)")
    parser.add_argument("--backends", type=str, default=None,
                        help=f"Comma-separated backends (default: all). "
                             f"Options: {','.join(ALL_BACKENDS)}")
    parser.add_argument("--repos", type=str, default=None,
                        help="Comma-separated repo filter (e.g. 'pallets/flask,psf/requests')")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to file with completed instance IDs (one per line) to skip")
    parser.add_argument("--resume-offset", type=int, default=0,
                        help="Number of already-completed instances (for display numbering)")
    parser.add_argument("--dataset", type=str, default="verified",
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to evaluate (default: verified)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip LLM keyword cache, force fresh API calls")
    args = parser.parse_args()

    global SKIP_LLM_CACHE
    if args.no_cache:
        SKIP_LLM_CACHE = True

    backends = args.backends.split(",") if args.backends else ALL_BACKENDS

    # Set per-dataset directories
    global REPOS_DIR, KEYWORD_CACHE_DIR, BM25_CACHE_DIR
    cfg = DATASET_CONFIGS[args.dataset]
    REPOS_DIR = cfg["repos_dir"]
    KEYWORD_CACHE_DIR = RESULTS_DIR / cfg["results_subdir"] / "keyword_cache"
    BM25_CACHE_DIR = RESULTS_DIR / cfg["results_subdir"] / "bm25_cache"
    REPOS_DIR.mkdir(parents=True, exist_ok=True)

    # Create per-run output folder under dataset subdirectory
    dataset_dir = RESULTS_DIR / cfg["results_subdir"]
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = dataset_dir / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

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

    log_file = open(run_dir / "run.log", "w")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    ds_label = DATASET_CONFIGS[args.dataset]["label"]

    print("\n" + "="*70)
    print("  SWE-bench Search Eval Pipeline")
    print(f"  Dataset: {ds_label}")
    print(f"  Run: {run_dir}")
    print(f"  Backends: {', '.join(backends)}")
    print(f"  LLM: {LLM_MODEL} @ {LLM_API_BASE}")
    print(f"  Timeout={LLM_TIMEOUT}s  Retries={LLM_RETRIES}")
    print("="*70)

    # Save run config
    run_config = {
        "run_type": "search_eval",
        "run_id": f"run_{run_ts}",
        "dataset": args.dataset,
        "dataset_hf": cfg["hf_name"],
        "top_k": args.top_k,
        "backends": backends,
        "keyword_model": LLM_MODEL,
        "keyword_api_base": LLM_API_BASE,
        "keyword_max_tokens": 1024,
        "keyword_timeout_s": LLM_TIMEOUT,
        "keyword_retries": LLM_RETRIES,
        "timestamp_start": datetime.now().isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    # Step 1: Load data
    print(f"\n[Step 1] Loading {ds_label} ({args.limit} instances)...")
    instances = load_swebench_instances(args.limit, dataset=args.dataset)

    if not instances:
        print("ERROR: No instances loaded")
        sys.exit(1)

    # Filter by repos if specified
    if args.repos:
        repo_filter = set(r.strip() for r in args.repos.split(","))
        instances = [i for i in instances if i["repo"] in repo_filter]
        if not instances:
            print(f"ERROR: No instances match repos: {repo_filter}")
            sys.exit(1)
        print(f"  Filtered to {len(instances)} instances in {len(repo_filter)} repos")

    # Resume: skip already-completed instances
    if args.resume:
        with open(args.resume) as f:
            done_ids = set(line.strip() for line in f if line.strip())
        before = len(instances)
        instances = [i for i in instances if i["instance_id"] not in done_ids]
        print(f"  Resume: skipping {before - len(instances)} completed, {len(instances)} remaining")

    # Show instance summary
    from collections import Counter
    repo_counts = Counter(i["repo"] for i in instances)
    for repo, count in repo_counts.most_common():
        avg_gold = sum(len(i["gold_files"]) for i in instances
                       if i["repo"] == repo) / count
        print(f"    {repo:<35s} {count:2d} instances, "
              f"avg {avg_gold:.1f} gold files")

    # Step 2: Run evaluation
    print(f"\n[Step 2] Running evaluation...")
    result = run_evaluation(instances, top_k=args.top_k, verbose=args.verbose,
                            backends=backends, instance_offset=args.resume_offset)

    # Step 3: Report
    print_results(result)
    save_results(result, run_dir=run_dir)

    # Update config.json with final stats
    run_config["num_instances"] = result["num_instances"]
    run_config["num_repos"] = len(set(i["repo"] for i in result["per_instance"]))
    run_config["elapsed_seconds"] = result["elapsed_seconds"]
    run_config["timestamp_end"] = datetime.now().isoformat()
    run_config["keyword_stats"] = dict(_llm_stats)
    del run_config["keyword_stats"]["errors"]  # don't dump full error list
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    # Step 3b: Snapshot keyword cache into run folder
    kw_src = _keyword_cache_path()
    if kw_src.exists():
        safe_model = re.sub(r'[^\w.-]', '_', LLM_MODEL)
        kw_dst = run_dir / f"{run_ts}_{safe_model}.json"
        shutil.copy2(kw_src, kw_dst)
        print(f"  Keyword cache snapshot saved to {kw_dst.name}")

    # Step 4: LLM stats summary
    print(f"\n[LLM Stats] cache_hit={_llm_stats['cache_hit']}  "
          f"api_ok={_llm_stats['api_ok']}  "
          f"api_fail={_llm_stats['api_fail']}")
    if _llm_stats["api_fail"] > 0:
        from collections import Counter
        err_counts = Counter(_llm_stats["errors"])
        print(f"[LLM Stats] Error breakdown:")
        for err, cnt in err_counts.most_common():
            print(f"  {cnt:3d}x {err}")

    sys.stdout = original_stdout
    sys.stderr = original_stderr
    log_file.close()


if __name__ == "__main__":
    main()
