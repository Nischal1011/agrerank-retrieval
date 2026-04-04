import numpy as np
import pandas as pd
import json, glob

# load embeddings and ids
emb = np.load("data/embeddings/corpus_emb.npy")       # shape (n, p)
ids = np.load("data/embeddings/corpus_ids.npy", allow_pickle=True)  # shape (n,)

# load all chunk jsons
chunk_files = glob.glob("data/chunks/*.chunks.json")
id2text = {}
for path in chunk_files:
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        for c in chunks:
            cid, txt = c.get("id"), (c.get("text") or "")
            if cid and txt:
                id2text[cid] = txt

# build dataframe
rows = []
for i, cid in enumerate(ids):
    rows.append({
        "id": cid,
        "text": id2text.get(cid, "[MISSING TEXT]"),
        **{f"dim_{j}": emb[i, j] for j in range(emb.shape[1])}
    })

df = pd.DataFrame(rows)

# save to csv
out_path = "data/embeddings/full_corpus_with_text.csv"
df.to_csv(out_path, index=False)

print(f"✅ Saved {df.shape} -> {out_path}")


# example command: python vectorrag/export_embeddings.py

