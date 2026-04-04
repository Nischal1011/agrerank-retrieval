# Embed chunk JSON files into n×p embeddings (OpenAI embeddings API).
# Modes:
#   - recreate (default): embed all provided chunk files and overwrite outputs
#   - append: embed only NEW ids and append to existing arrays
import os, json, glob, argparse, sys, time, math
from typing import List, Dict, Tuple
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

def load_chunk_items(paths: List[str]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            for d in data:
                t = (d.get("text") or "").strip()
                cid = d.get("id")
                if t and cid:
                    items.append({"id": str(cid), "text": t})
    return items

def embed_batch(model: str, batch: List[str], max_retries: int = 6, base_wait: float = 1.0) -> np.ndarray:
    """Embed one batch with simple exponential backoff."""
    attempt = 0
    while True:
        try:
            resp = client.embeddings.create(model=model, input=batch)
            arr = np.array([r.embedding for r in resp.data], dtype=np.float32)
            return arr
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            wait = base_wait * (2 ** (attempt - 1)) * (1 + 0.1 * np.random.rand())
            print(f"[retry {attempt}/{max_retries}] {type(e).__name__}: {e} -> sleeping {wait:.1f}s", flush=True)
            time.sleep(wait)

def embed_texts(model: str, texts: List[str], batch_size: int) -> np.ndarray:
    vecs = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]
        arr = embed_batch(model, batch)
        vecs.append(arr)
        print(f"Embedded {min(i+len(batch), n)}/{n} (batch={len(batch)})", flush=True)
    return np.vstack(vecs) if vecs else np.zeros((0, 0), dtype=np.float32)

def write_metadata(outdir: str, model: str, n: int, dim: int, files: List[str], mode: str, extra: Dict[str, str]):
    meta = {
        "model": model,
        "count": n,
        "dim": dim,
        "mode": mode,
        "source_files": files,
        "env": {"OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", ""), "OPENAI_ORG": os.getenv("OPENAI_ORG", "")},
        **extra,
    }
    with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Embed JSON chunks into n×p embeddings.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--chunks", nargs="+", help="Explicit list of *.chunks.json files")
    src.add_argument("--glob", help='Glob for chunk files, e.g. "data/chunks/*.chunks.json"')

    ap.add_argument("--outdir", default="data/embeddings",
                    help="Where to write corpus_ids.npy and corpus_emb.npy")
    ap.add_argument("--mode", choices=["recreate", "append"], default="recreate",
                    help="recreate = overwrite outputs; append = load old, add new IDs")

    ap.add_argument("--olddir", help="(append mode) dir with existing corpus_{ids,emb}.npy")
    ap.add_argument("--model", default="text-embedding-3-large", help="OpenAI embeddings model id")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    # Resolve input files
    paths = args.chunks if args.chunks else sorted(glob.glob(args.glob or ""))
    if not paths:
        sys.exit("No chunk files found. Provide --chunks or a --glob that matches.")

    # Load all (id, text)
    items = load_chunk_items(paths)
    if not items:
        sys.exit("Loaded 0 items (ensure chunk JSONs have {'id','text'}).")

    # Deduplicate by id (keep first occurrence, warn on dups)
    seen = set()
    dedup: List[Dict[str, str]] = []
    dup_count = 0
    for it in items:
        if it["id"] in seen:
            dup_count += 1
            continue
        seen.add(it["id"])
        dedup.append(it)
    if dup_count:
        print(f"⚠️  skipped {dup_count} duplicate id(s) (kept first occurrence)")

    os.makedirs(args.outdir, exist_ok=True)
    ids_out = os.path.join(args.outdir, "corpus_ids.npy")
    emb_out = os.path.join(args.outdir, "corpus_emb.npy")

    if args.mode == "recreate":
        ids = np.array([it["id"] for it in dedup], dtype=object)
        texts = [it["text"] for it in dedup]
        emb = embed_texts(args.model, texts, args.batch_size)
        if emb.size == 0:
            sys.exit("No embeddings produced.")
        np.save(ids_out, ids)
        np.save(emb_out, emb)
        write_metadata(args.outdir, args.model, emb.shape[0], emb.shape[1], paths, "recreate",
                       {"batch_size": args.batch_size})
        print(f"✅ RECREATED embeddings: ids={ids.shape} -> {ids_out}")
        print(f"                       emb={emb.shape} -> {emb_out}")
        return

    # ---- append mode ----
    if not args.olddir:
        sys.exit("Append mode requires --olddir pointing to existing corpus_ids.npy and corpus_emb.npy")
    old_ids_path = os.path.join(args.olddir, "corpus_ids.npy")
    old_emb_path = os.path.join(args.olddir, "corpus_emb.npy")
    if not (os.path.exists(old_ids_path) and os.path.exists(old_emb_path)):
        sys.exit(f"Missing artifacts in {args.olddir}:\n  {old_ids_path}\n  {old_emb_path}")

    old_ids = np.load(old_ids_path, allow_pickle=True)  # (N_old,)
    old_emb = np.load(old_emb_path)                      # (N_old, p)
    print(f"Loaded existing: ids={old_ids.shape}, emb={old_emb.shape} from {args.olddir}")

    old_set = set(map(str, old_ids.tolist()))
    new_items = [it for it in dedup if it["id"] not in old_set]

    if not new_items:
        # Nothing new; copy old to outdir if different
        if os.path.abspath(args.olddir) != os.path.abspath(args.outdir):
            np.save(ids_out, old_ids)
            np.save(emb_out, old_emb)
            write_metadata(args.outdir, args.model, old_emb.shape[0], old_emb.shape[1], paths, "append",
                           {"batch_size": args.batch_size, "appended": 0})
            print(f"ℹ️ No new IDs. Copied existing artifacts into {args.outdir}")
        else:
            print("ℹ️ No new IDs. Nothing to do.")
        return

    print(f"Embedding {len(new_items)} new IDs (skipped {len(dedup) - len(new_items)} duplicates).")
    new_ids = np.array([it["id"] for it in new_items], dtype=object)
    new_texts = [it["text"] for it in new_items]
    new_emb = embed_texts(args.model, new_texts, args.batch_size)
    if new_emb.shape[1] != old_emb.shape[1]:
        raise SystemExit(f"Dim mismatch: new dim={new_emb.shape[1]} vs existing dim={old_emb.shape[1]}")

    all_ids = np.concatenate([old_ids, new_ids])
    all_emb = np.vstack([old_emb, new_emb])

    np.save(ids_out, all_ids)
    np.save(emb_out, all_emb)
    write_metadata(args.outdir, args.model, all_emb.shape[0], all_emb.shape[1], paths, "append",
                   {"batch_size": args.batch_size, "appended": int(new_emb.shape[0])})
    print(f"✅ APPENDED embeddings: total ids={all_ids.shape}, emb={all_emb.shape} -> {args.outdir}")

if __name__ == "__main__":
    main()



# python vectorrag/embed_chunks.py  --glob "data/chunks_fixed250/*.chunks.json" --outdir data/embeddings/fixed250 --mode recreate --model text-embedding-3-large --batch-size 128
# python vectorrag/embed_chunks.py --glob "data/chunks_bac250/*.chunks.json" --outdir data/embeddings/bac250 --mode recreate --model text-embedding-3-large --batch-size 128
