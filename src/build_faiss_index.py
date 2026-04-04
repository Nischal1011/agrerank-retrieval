import os, argparse, numpy as np, faiss

def _sanitize(X: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(X).all(axis=1)
    if bad.any():
        keep = ~bad
        print(f"⚠️  Dropping {bad.sum()} rows with NaN/Inf from embeddings ({X.shape[0]} -> {keep.sum()})")
        X = X[keep]
    return X

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="data/embeddings/corpus_emb.npy")
    ap.add_argument("--out", default="data/FAISS_index/index.faiss")
    ap.add_argument("--mode", choices=["rebuild", "append"], default="rebuild")
    ap.add_argument("--normalize", action="store_true", help="L2-normalize vectors (cosine via IP)")
    ap.add_argument("--threads", type=int, default=0, help="FAISS OMP threads (0 = leave default)")
    args = ap.parse_args()

    if args.threads > 0:
        faiss.omp_set_num_threads(args.threads)

    X = np.load(args.emb).astype("float32")
    if X.ndim != 2:
        raise SystemExit(f"Expected 2D array, got shape {X.shape}")
    X = _sanitize(X)
    if args.normalize:
        faiss.normalize_L2(X)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.mode == "rebuild" or not os.path.exists(args.out):
        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X)
        faiss.write_index(index, args.out)
        print(f"✅ Rebuilt index with {index.ntotal} vectors of dim {index.d} -> {args.out}")
        return

    # append mode
    index = faiss.read_index(args.out)
    if index.d != X.shape[1]:
        raise SystemExit(f"Dim mismatch: index.d={index.d} vs emb dim={X.shape[1]}")
    old_n = index.ntotal
    if old_n > X.shape[0]:
        raise SystemExit(f"Index has {old_n} entries but embeddings have only {X.shape[0]} rows. "
                         "You likely shrank/reordered the corpus; use --mode rebuild.")
    if old_n == X.shape[0]:
        print("ℹ️ No new vectors to add; index already up-to-date.")
        return
    index.add(X[old_n:])
    faiss.write_index(index, args.out)
    print(f"✅ Appended {index.ntotal - old_n} new vectors (total {index.ntotal}) -> {args.out}")

if __name__ == "__main__":
    main()


# python vectorrag/build_faiss_index.py  --emb data/embeddings/fixed250/corpus_emb.npy --out data/FAISS_index/fixed250.index --mode rebuild --normalize --threads 8

# python vectorrag/build_faiss_index.py --emb data/embeddings/bac250/corpus_emb.npy --out data/FAISS_index/bac250.index --mode rebuild --normalize --threads 8
