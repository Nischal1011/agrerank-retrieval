# validate_qset.py
import json, sys, numpy as np, pathlib as p

qset_path = p.Path(sys.argv[1])
ids_npy   = p.Path(sys.argv[2])  # data/embeddings/fixed250/corpus_ids.npy

qset = json.loads(qset_path.read_text())
ids = np.load(ids_npy, allow_pickle=True).tolist()
ids = [i.decode() if isinstance(i, bytes) else str(i) for i in ids]
idset = set(ids)

ok = True
for q in qset:
    pid = q.get("paper_id")
    for rc in q["relevant_chunks"]:
        if rc not in idset:
            print(f"[MISSING] {q['query_id']} -> {rc}")
            ok = False
        if pid and not rc.startswith(pid + "::"):
            print(f"[PAPER MISMATCH] {q['query_id']} -> {rc} not under {pid}")
            ok = False

print("✅ qset looks consistent!" if ok else "❌ fix the issues above.")


# python vectorrag/eval/validate_qset.py data/qqset.json data/qset_embeddings/openai-large/qset_ids.npy
