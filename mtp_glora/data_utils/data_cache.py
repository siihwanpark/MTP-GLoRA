from __future__ import annotations

import os, json, hashlib, time
from typing import Optional, Dict, Any, List

import torch
import torch.distributed as dist
from datasets import load_from_disk, Dataset as HFDataset, concatenate_datasets

from mtp_glora.utils import is_dist_main, barrier

MTP_DATASET_BUILD_VERSION = "v1"  # if build logic is changed, only increase this version to invalidate the cache


def _file_fingerprint(path: str) -> Dict[str, Any]:
    try:
        st = os.stat(path)
        return {"path": os.path.abspath(path), "mtime": int(st.st_mtime), "size": int(st.st_size)}
    except FileNotFoundError:
        return {"path": os.path.abspath(path), "mtime": 0, "size": 0}


def _tokenizer_fingerprint(tokenizer) -> Dict[str, Any]:
    return {
        "name_or_path": getattr(tokenizer, "name_or_path", "<unknown>"),
        "vocab_size": tokenizer.vocab_size,
        "added_tokens": len(getattr(tokenizer, "added_tokens_decoder", {})),
        "mask_id": int(tokenizer.convert_tokens_to_ids("<mask>")),
    }


def _cache_key(meta: Dict[str, Any]) -> str:
    raw = json.dumps(meta, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def cache_dir_for(train_data_path: str, tokenizer, draft_length: int, shuffle_seed: int, cache_root: str) -> str:
    meta = {
        "version": MTP_DATASET_BUILD_VERSION,
        "train_file": _file_fingerprint(train_data_path),
        "tokenizer": _tokenizer_fingerprint(tokenizer),
        "draft_length": int(draft_length),
        "shuffle_seed": int(shuffle_seed),
    }
    key = _cache_key(meta)
    return os.path.join(cache_root, f"mtp_ds_{key}")


def _manifest_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "shards.json")


def _success_flag(cache_dir: str) -> str:
    return os.path.join(cache_dir, "_SUCCESS")


def load_cached_dataset(cache_dir: str) -> Optional[HFDataset]:
    """
    Loads either a monolithic dataset or a sharded dataset (via shards.json).
    Returns None if the cache directory isn't complete.
    """
    if not os.path.isdir(cache_dir):
        return None

    ok_flag = _success_flag(cache_dir)
    if not os.path.exists(ok_flag):
        return None

    manifest_file = _manifest_path(cache_dir)
    if os.path.exists(manifest_file):
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        shard_dirs: List[str] = manifest.get("shards", [])
        if not shard_dirs:
            return None
        shard_dsets = [load_from_disk(os.path.join(cache_dir, sd)) for sd in shard_dirs]
        ds = concatenate_datasets(shard_dsets)
        return ds
    else:
        try:
            ds = load_from_disk(cache_dir)
            return ds
        except Exception:
            return None

def save_dataset_to_cache(
    ds: HFDataset,
    cache_dir: str,
    meta: Dict[str, Any],
    *,
    target_bytes_per_shard: int = 4 * 1024**3,  # 4 GiB per shard (approx)
    verbose: bool = True,
):
    """
    Save dataset in byte-targeted shards. Good when rows are few but very large.
    Creates contiguous shards by scanning rows sequentially and cutting on size threshold.
    """
    os.makedirs(cache_dir, exist_ok=True)

    if not is_dist_main():
        barrier()
        return

    n = len(ds)
    if n == 0:
        if verbose: print(f"[cache] empty dataset -> writing monolithic")
        t0 = time.time()
        original_format = ds.format
        try:
            ds.reset_format()
            ds.save_to_disk(cache_dir)
            with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            with open(_success_flag(cache_dir), "w") as f:
                f.write(f"{time.time()}\n")
        finally:
            if original_format is not None:
                pass
        if verbose: print(f"[cache] done in {time.time()-t0:.1f}s")
        barrier()
        return

    # We'll iterate once and cut contiguous ranges ~ target_bytes_per_shard
    original_format = ds.format
    try:
        ds.reset_format()  # cleaner Arrow serialization

        # compute per-row approx sizes by a single pass (iterating materializes rows lazily)
        if verbose:
            print(f"[cache] scanning rows to build shards by ~{target_bytes_per_shard/1024**3:.1f} GiB ...")
        t_scan = time.time()

        # We want contiguous ranges; we’ll accumulate size until threshold, then cut [start, i+1)
        ranges = []
        cur_start = 0
        cur_bytes = 0
        est_total = 0

        # Iterating by index keeps it simple and contiguous. For huge datasets this is still OK
        # because we only keep counters, not rows.
        for i in range(n):
            ex = ds[i]  # random access; HF caches chunks internally
            sz = _approx_row_size_bytes(ex)
            est_total += sz
            # if single row exceeds target, flush previous (if any), and store this row alone
            if sz >= target_bytes_per_shard:
                if cur_start < i:
                    ranges.append((cur_start, i))
                ranges.append((i, i+1))
                cur_start = i + 1
                cur_bytes = 0
                continue

            if cur_bytes + sz > target_bytes_per_shard and cur_start < i:
                ranges.append((cur_start, i))
                cur_start = i
                cur_bytes = 0
            cur_bytes += sz

        if cur_start < n:
            ranges.append((cur_start, n))

        if verbose:
            total_gb = est_total / 1024**3
            print(f"[cache] planned {len(ranges)} shards; est total ~= {total_gb:.2f} GiB; scan {time.time()-t_scan:.1f}s")

        # write shards
        shard_dirs = []
        t0 = time.time()
        for si, (s, e) in enumerate(ranges, 1):
            shard_dirname = f"shard_{si:05d}"
            shard_path = os.path.join(cache_dir, shard_dirname)
            if os.path.exists(os.path.join(shard_path, "_SUCCESS")):
                shard_dirs.append(shard_dirname)
                if verbose:
                    print(f"[cache][{si}/{len(ranges)}] skip existing {shard_dirname}")
                continue

            if verbose:
                print(f"[cache][{si}/{len(ranges)}] writing {shard_dirname} rows [{s},{e}) ...")
            t_shard = time.time()
            shard_ds = ds.select(range(s, e))  # contiguous slice
            os.makedirs(shard_path, exist_ok=True)
            shard_ds.save_to_disk(shard_path)
            with open(os.path.join(shard_path, "_SUCCESS"), "w") as f:
                f.write(f"{time.time()}\n")
            if verbose:
                dt = time.time() - t_shard
                print(f"[cache][{si}/{len(ranges)}] done in {dt:.1f}s")
            shard_dirs.append(shard_dirname)

        # manifest + meta + root success
        manifest = {
            "version": 2,
            "mode": "by_bytes",
            "target_bytes_per_shard": int(target_bytes_per_shard),
            "total_rows": int(n),
            "num_shards": len(shard_dirs),
            "shards": shard_dirs,
            "build_meta": meta,
        }
        with open(_manifest_path(cache_dir), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        with open(os.path.join(cache_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        with open(_success_flag(cache_dir), "w") as f:
            f.write(f"{time.time()}\n")
        if verbose:
            print(f"[cache] all shards done in {time.time()-t0:.1f}s; manifest written.")
    finally:
        if original_format is not None:
            pass

    barrier()

def _approx_row_size_bytes(ex: Dict[str, Any]) -> int:
    """Rough byte-size estimator for a dataset row."""
    def size_of(v):
        import numpy as np
        import torch as _torch
        if v is None:
            return 0
        if isinstance(v, bool):
            return 1
        if isinstance(v, int):
            return 8  # Py int proxy
        if isinstance(v, float):
            return 8
        if isinstance(v, str):
            return len(v.encode("utf-8"))
        if isinstance(v, bytes) or isinstance(v, bytearray):
            return len(v)
        if isinstance(v, (list, tuple)):
            # assume int32-like by default (4B/elem); nested lists handled recursively
            s = 0
            for x in v:
                s += size_of(x)
            return s
        if isinstance(v, dict):
            s = 0
            for k, x in v.items():
                s += len(str(k)) + size_of(x)
            return s
        # numpy
        if isinstance(v, np.ndarray):
            return v.nbytes
        # torch tensor
        if isinstance(v, _torch.Tensor):
            return v.numel() * max(1, v.element_size())
        # arrow scalars or other objects
        try:
            return len(v)  # may work for some containers
        except Exception:
            return 64  # fallback
    # per-row overhead fudge
    base_overhead = 64
    return base_overhead + sum(size_of(v) for v in ex.values())