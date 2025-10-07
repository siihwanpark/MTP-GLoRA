"""
Unified prompt extractor for:
  1) open-thoughts/OpenThoughts-114k  ("openthoughts")
  2) a-m-team/AM-Qwen3-Distilled     ("am_qwen3")

Select the source via --source {openthoughts,am_qwen3} and provide the
relevant options. Each extractor keeps parity with your original scripts
(heuristics, retargeting, dedupe, reporting) while sharing utilities.
"""

from __future__ import annotations
import argparse, json, random, re, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def normalize_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def ot_extract_first_user_prompt(example: Dict) -> Optional[str]:
    # default schema
    conv = example.get("conversations")
    if isinstance(conv, list):
        for m in conv:
            if not isinstance(m, dict):
                continue
            if m.get("from") in ("user", "human"):
                v = m.get("value")
                if isinstance(v, str) and v.strip():
                    return v
            if m.get("role") == "user":
                v = m.get("content")
                if isinstance(v, str) and v.strip():
                    return v
    
    # messages schema
    msgs = example.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user":
                v = m.get("content")
                if isinstance(v, str) and v.strip():
                    return v
            if m.get("from") in ("user", "human"):
                v = m.get("value")
                if isinstance(v, str) and v.strip():
                    return v
    
    # fallbacks
    for k in ("input", "question", "prompt"):
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None

# MATH prefix patterns (handle \\boxed{} with/without JSON escaping, with/without trailing space)
_MATH_PREFIX_PATTERNS = [
    r"^\s*Return your final response within\s*\\\\boxed\{\}\.\s+",
    r"^\s*Return your final response within\s*\\boxed\{\}\.\s+",
    r"^\s*Return your final response within\s*\\\\boxed\{\}\.\s*",
    r"^\s*Return your final response within\s*\\boxed\{\}\.\s*",
]

def ot_looks_like_math(text: str) -> bool:
    for pat in _MATH_PREFIX_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            return bool(re.match(pat, text, flags=re.IGNORECASE))
    return False

def ot_strip_math_prefix(text: str) -> str:
    for pat in _MATH_PREFIX_PATTERNS:
        new = re.sub(pat, "", text, count=1, flags=re.IGNORECASE)
        if new != text:
            return new.lstrip()
    return text

_CODE_UMBRELLA = re.compile(
    r"^\s*Generate an executable Python function.*?(?:"
    r"stdin.*?print the output.*?(?:Simply call.*?definition\.)|"
    r"Return the function body.*?final solution\."  # Variant B
    r")\s*",
    flags=re.IGNORECASE | re.DOTALL,
)

def ot_looks_like_code(text: str) -> bool:
    return text.lower().startswith("generate an executable python function")

def ot_strip_code_prefix(text: str) -> str:
    return _CODE_UMBRELLA.sub("", text, count=1).lstrip()


def ot_inject_code(style: str) -> str:
    if style == "stdin":
        return (
            "Read from standard input and write to standard output. "
            "Do not print extra text.\n\n"
        )
    if style == "solve":
        return (
            "Implement a Python function solve() that reads from sys.stdin and "
            "prints to sys.stdout. Call solve() after its definition.\n\n"
        )
    return ""


def ot_choose_style(mix: Dict[str, float]) -> str:
    keys = list(mix.keys())
    vals = [max(0.0, float(mix[k])) for k in keys]
    s = sum(vals)
    if s <= 0:
        return "none"
    r = random.random() * s
    acc = 0.0
    for k, v in zip(keys, vals):
        acc += v
        if r <= acc:
            return k
    return "none"


def extract_openthoughts(args: argparse.Namespace) -> None:
    t0 = time.time()
    random.seed(args.seed)

    # parse code mix
    try:
        code_mix = json.loads(args.code_mix)
        assert isinstance(code_mix, dict)
    except Exception as e:
        raise ValueError(f"--code_mix must be a JSON object: {args.code_mix}") from e

    print("[Load] open-thoughts/OpenThoughts-114k (default split; no metadata)")
    from datasets import load_dataset  # local import to avoid hard dep for am_qwen3
    from tqdm import tqdm

    ds = load_dataset("open-thoughts/OpenThoughts-114k", split="train")

    rows: List[Dict] = []
    for i in tqdm(range(len(ds)), desc="Scan"):
        p = ot_extract_first_user_prompt(ds[i])
        if not p:
            continue
        if ot_looks_like_code(p):
            dom = "code"
            text = ot_strip_code_prefix(p)
            style = ot_choose_style(code_mix)
            text = (ot_inject_code(style) + text).strip()
        elif ot_looks_like_math(p):
            dom = "math"
            text = ot_strip_math_prefix(p)
        else:
            dom = "other"
            text = p.strip()
        if not text:
            continue
        rows.append({"prompt": text, "domain": dom})

    # Deduplicate
    if args.dedupe:
        seen = set()
        deduped: List[Dict] = []
        for r in rows:
            key = normalize_text(r["prompt"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(r)
    else:
        deduped = rows

    # Downsample
    if args.max_samples is not None and len(deduped) > args.max_samples:
        deduped = random.sample(deduped, args.max_samples)

    # Save
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    dom_counts: Dict[str, int] = {}
    for r in deduped:
        d = r.get("domain") or "other"
        dom_counts[d] = dom_counts.get(d, 0) + 1

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "open-thoughts/OpenThoughts-114k (default)",
        "max_samples": args.max_samples,
        "seed": args.seed,
        "out_path": args.out_path,
        "total_saved": len(deduped),
        "counts": dom_counts,
        "retargeting": {
            "math": "strip_only",
            "code_mix": json.loads(args.code_mix),
            "domain_inference": "heuristic (startswith-patterns)",
        },
        "runtime_sec": time.time() - t0,
        "notes": "No metadata used. First user message extracted from default split.",
    }

    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2)

    print(f"[Done] Saved {len(deduped)} prompts → {args.out_path}")
    print(f"[Report] JSON → {args.report_json}")


# ------------------------
# AM-Qwen3-Distilled extractor
# ------------------------

FILES2BUCKET: Dict[str, str] = {
    "other.jsonl":   "chat",
    "math.jsonl":    "math",
    "code.jsonl":    "code",
    "science.jsonl": "science",
    "if.jsonl":      "if",
}

DEFAULT_BUCKET_WEIGHTS: Dict[str, float] = {
    # Instance-level distribution from dataset card; customized default
    "chat": 0.0, "math": 0.536, "code": 0.311, "science": 0.153, "if": 0.0
}


def aq_extract_first_human_prompt(obj: Dict) -> Optional[str]:
    conv = obj.get("conversations")
    if not isinstance(conv, list):
        return None
    for m in conv:
        if m.get("from") == "human":
            return m.get("value")
    return None


def aq_parse_bucket_weights(weights_str: Optional[str]) -> Dict[str, float]:
    keys = list(DEFAULT_BUCKET_WEIGHTS.keys())
    if not weights_str:
        w = dict(DEFAULT_BUCKET_WEIGHTS)
    else:
        try:
            obj = json.loads(weights_str)
            assert isinstance(obj, dict)
        except Exception as e:
            raise ValueError(f"--weights must be a JSON object, got: {weights_str}") from e
        w = {k: float(max(0.0, obj.get(k, 0.0))) for k in keys}
        if sum(w.values()) <= 0:
            w = dict(DEFAULT_BUCKET_WEIGHTS)
    # normalize
    s = sum(w.values())
    for k in keys:
        w[k] = w[k] / s
    return w


def aq_bucket_targets(total: int, bucket_w: Dict[str, float]) -> Dict[str, int]:
    tgt = {b: int(round(total * bucket_w[b])) for b in bucket_w}
    delta = total - sum(tgt.values())
    order = sorted(bucket_w.keys(), key=lambda b: bucket_w[b], reverse=True)
    i = 0
    while delta != 0 and order:
        b = order[i % len(order)]
        if delta > 0:
            tgt[b] += 1; delta -= 1
        elif tgt[b] > 0:
            tgt[b] -= 1; delta += 1
        i += 1
    return tgt


def aq_per_file_targets(bucket_quota: Dict[str, int]) -> Dict[str, int]:
    out: Dict[str, int] = {f: 0 for f in FILES2BUCKET}
    inv: Dict[str, List[str]] = {}
    for f, b in FILES2BUCKET.items():
        inv.setdefault(b, []).append(f)
    for b, files in inv.items():
        q = bucket_quota.get(b, 0)
        if len(files) == 1:
            out[files[0]] = q
        else:
            q0, r = divmod(q, len(files))
            for i, f in enumerate(files):
                out[f] = q0 + (1 if i < r else 0)
    return out


def aq_hf_cached_download(repo: str, filename: str, revision: Optional[str], local_dir: Optional[str]) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo, filename=filename, revision=revision, repo_type="dataset",
        local_dir=local_dir, local_dir_use_symlinks=True
    )


def aq_sample_from_local_file(local_path: str, k: int, dedupe_set: Optional[set], seed: int) -> Tuple[List[Dict], int]:
    if k <= 0:
        return [], 0
    random.seed(seed ^ hash(local_path))
    out: List[Dict] = []
    dup_skipped = 0
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            p = aq_extract_first_human_prompt(obj)
            if not p:
                continue
            if dedupe_set is not None:
                key = normalize_text(p)
                if key in dedupe_set:
                    dup_skipped += 1
                    continue
                dedupe_set.add(key)
            out.append({"prompt": p})
            if len(out) >= k:
                break
    return out, dup_skipped


def extract_am_qwen3(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    t_start = time.time()
    random.seed(args.seed)
    bucket_w = aq_parse_bucket_weights(args.weights)
    bucket_quota = aq_bucket_targets(args.total_samples, bucket_w)
    file_targets = aq_per_file_targets(bucket_quota)

    if args.report_json is None:
        args.report_json = str(Path(args.out_path).with_suffix(".stats.json"))

    print("[Bucket quotas]")
    for b in ["chat","math","code","science","if"]:
        print(f"  {b:7s}: {bucket_quota.get(b,0)}")
    print("[Targets per file] (multiturn.jsonl is intentionally excluded)")
    for f in sorted(file_targets.keys()):
        print(f"  {f:14s}: {file_targets[f]}")

    # Download needed files in parallel
    to_download = {f: n for f, n in file_targets.items() if n > 0}
    downloaded: Dict[str, str] = {}
    if to_download:
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
            fut = {ex.submit(aq_hf_cached_download, args.repo, f, args.revision, args.local_dir): f
                   for f in to_download}
            for t in as_completed(fut):
                fname = fut[t]
                try:
                    downloaded[fname] = t.result()
                except Exception as e:
                    print(f"[WARN] failed to download {fname}: {e}")

    # Sample locally in parallel
    dedupe_set = set() if args.dedupe else None
    all_samples: List[Dict] = []
    dup_skipped_total = 0
    accepted_per_file: Dict[str, int] = {f: 0 for f in file_targets}
    accepted_per_bucket: Dict[str, int] = {b: 0 for b in ["chat","math","code","science","if"]}

    def _work(f: str, k: int) -> Tuple[str, List[Dict], int]:
        if k <= 0 or f not in downloaded:
            return f, [], 0
        part, dup_skipped = aq_sample_from_local_file(downloaded[f], k, dedupe_set, args.seed)
        return f, part, dup_skipped

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futures = [ex.submit(_work, f, k) for f, k in to_download.items()]
        for t in as_completed(futures):
            try:
                f, part, dup_skipped = t.result()
                bucket = FILES2BUCKET[f]
                for item in part:
                    item["category"] = bucket
                all_samples.extend(part)
                dup_skipped_total += dup_skipped
                accepted_per_file[f] += len(part)
                accepted_per_bucket[bucket] += len(part)
            except Exception as e:
                print(f"[WARN] sampling failed: {e}")

    # If short due to dedupe, top up from Chat file (other.jsonl)
    short = args.total_samples - len(all_samples)
    topup_source = None
    if short > 0 and "other.jsonl" in downloaded:
        print(f"[Info] Short by {short}. Topping up from other.jsonl")
        topup_source = "other.jsonl"
        part, dup_skipped = aq_sample_from_local_file(downloaded["other.jsonl"], short, dedupe_set, args.seed + 1)
        for item in part:
            item["category"] = "chat"
        all_samples.extend(part)
        dup_skipped_total += dup_skipped
        accepted_per_file["other.jsonl"] += len(part)
        accepted_per_bucket["chat"] += len(part)

    # Truncate & save
    all_samples = all_samples[: args.total_samples]
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        for ex_ in all_samples:
            f.write(json.dumps(ex_, ensure_ascii=False) + "\n")

    # ---------- Summary (stdout) ----------
    elapsed = time.time() - t_start
    print("\n[Summary]")
    print(f"  Saved samples : {len(all_samples)}")
    print(f"  Duplicates skipped : {dup_skipped_total}")
    print(f"  Elapsed time  : {elapsed:.2f}s")
    print("\n  Accepted per bucket:")
    total_acc = sum(accepted_per_bucket.values())
    for b in ["chat","math","code","science","if"]:
        cnt = accepted_per_bucket[b]
        pct = (100.0 * cnt / total_acc) if total_acc > 0 else 0.0
        print(f"    - {b:7s}: {cnt:7d}  ({pct:5.1f}%)")
    print("\n  Accepted per file:")
    for f in sorted(accepted_per_file.keys()):
        cnt = accepted_per_file[f]
        print(f"    - {f:14s}: {cnt:7d}")

    # ---------- JSON report ----------
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "a-m-team/AM-Qwen3-Distilled",
        "repo": args.repo,
        "revision": args.revision,
        "out_path": args.out_path,
        "report_json": args.report_json,
        "seed": args.seed,
        "dedupe": bool(args.dedupe),
        "total_requested": args.total_samples,
        "total_saved": len(all_samples),
        "duplicates_skipped": dup_skipped_total,
        "weights_used": aq_parse_bucket_weights(args.weights),
        "bucket_quota_target": aq_bucket_targets(args.total_samples, aq_parse_bucket_weights(args.weights)),
        "file_targets": aq_per_file_targets(aq_bucket_targets(args.total_samples, aq_parse_bucket_weights(args.weights))),
        "accepted_per_bucket": accepted_per_bucket,
        "accepted_per_file": accepted_per_file,
        "topup": {
            "count": max(0, len(all_samples) - sum(accepted_per_file.values()) + dup_skipped_total),
            "source_file": topup_source,
        },
        "files2bucket": FILES2BUCKET,
        "notes": "Multiturn excluded by design; Chat uses only other.jsonl.",
        "runtime_sec": elapsed,
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as rf:
        json.dump(report, rf, ensure_ascii=False, indent=2)
    print(f"\n[Report] JSON report saved to: {args.report_json}")


# ------------------------
# CLI
# ------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Unified prompt extractor for OpenThoughts-114k and AM-Qwen3-Distilled")
    parser.add_argument("--source", required=True, choices=["openthoughts", "am_qwen3"],
                   help="Which dataset to extract from.")
    parser.add_argument("--out_path", type=str, required=True, help="Output JSONL path.")
    parser.add_argument("--report_json", type=str, required=True, help="Where to write JSON stats report.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dedupe", type=lambda x: x.lower() in ["1","true","yes"], default=True)

    # OpenThoughts-specific
    parser.add_argument("--max_samples", type=int, default=None,
                   help="(OpenThoughts) Optional cap after dedupe.")
    parser.add_argument("--code_mix", type=str, default='{"stdin":0.6,"solve":0.4,"none":0.0}',
                   help='(OpenThoughts) JSON weights among {"stdin","solve","none"}.')

    # AM-Qwen3-specific
    parser.add_argument("--repo", type=str, default="a-m-team/AM-Qwen3-Distilled",
                   help="(AM-Qwen3) Dataset repo id.")
    parser.add_argument("--revision", type=str, default=None,
                   help="(AM-Qwen3) Optional commit hash or tag.")
    parser.add_argument("--weights", type=str, default=None,
                   help='(AM-Qwen3) JSON over {"chat","math","code","science","if"} for instance mix.')
    parser.add_argument("--total_samples", type=int, default=122880, help="(AM-Qwen3) Total samples to save.")
    parser.add_argument("--local_dir", type=str, default=None, help="(AM-Qwen3) Local cache directory.")
    parser.add_argument("--num_workers", type=int, default=4, help="(AM-Qwen3) Parallelism for I/O.")

    return parser


def main():
    args = build_parser().parse_args()
    if args.source == "openthoughts":
        extract_openthoughts(args)
    elif args.source == "am_qwen3":
        extract_am_qwen3(args)
    else:
        raise ValueError(f"Unknown --source: {args.source}")


if __name__ == "__main__":
    main()
