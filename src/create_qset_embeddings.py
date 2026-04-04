import json
import os
import sys
from pathlib import Path
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import argparse
from typing import List, Tuple

# --- Configuration ---
MODEL_EMBEDDING = "text-embedding-3-large"
EMBEDDING_DIM = 3072  # Dimension for 'text-embedding-3-large'
BATCH_SIZE = 128      # Process texts in batches for API efficiency

def l2n(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Performs L2 normalization on a matrix of vectors."""
    if V.ndim == 1:
        n = np.linalg.norm(V)
        return V / max(n, eps)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return V / n

def load_queries_from_qset(path: Path) -> Tuple[List[str], List[str]]:
    """Loads query IDs and text from the qset.json file."""
    ids = []
    texts = []
    print(f"Loading queries from '{path}'...")
    try:
        qset_data = json.loads(path.read_text(encoding="utf-8"))
        for query_obj in qset_data:
            query_id = query_obj.get("query_id")
            query_text = query_obj.get("query_text", "")
            if query_id and query_text:
                ids.append(query_id)
                texts.append(query_text.strip())
    except json.JSONDecodeError:
        sys.exit(f"❌ Error: Could not parse JSON from '{path}'. Please check the file format.")
    except Exception as e:
        sys.exit(f"❌ Error: An unexpected error occurred while reading '{path}': {e}")
        
    print(f"Loaded {len(ids)} queries from qset.")
    return ids, texts

def main():
    parser = argparse.ArgumentParser(
        description=f"Generate embeddings for all queries in a qset.json file using OpenAI's '{MODEL_EMBEDDING}' model."
    )
    parser.add_argument("--qset", type=Path, required=True, help="Path to the qset.json file containing queries.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the output 'qset_ids.npy' and 'qset_embeddings.npy' files.")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing embedding files without asking.")
    args = parser.parse_args()

    # --- 1. Define Output Paths and Check for Existing Files ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = args.output_dir / "qset_ids.npy"
    emb_path = args.output_dir / "qset_embeddings.npy"

    if (ids_path.exists() or emb_path.exists()) and not args.force:
        overwrite = input(
            f"⚠️  Warning: Query embedding files already exist in '{args.output_dir}'.\n"
            "    Do you want to overwrite them? (y/N): "
        ).strip().lower()
        if overwrite != 'y':
            print("Aborting. No files were changed.")
            sys.exit(0)
        print("Proceeding with overwrite...")

    # --- 2. Check for OpenAI API Key ---
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("❌ Error: OPENAI_API_KEY environment variable not set. Please set it before running.")
    client = OpenAI(api_key=api_key)

    # --- 3. Load Query Data ---
    query_ids, query_texts = load_queries_from_qset(args.qset)
    if not query_ids:
        sys.exit("❌ Error: No queries loaded from the qset file. Exiting.")

    # --- 4. Generate Embeddings in Batches ---
    all_embeddings = []
    print(f"Generating embeddings for {len(query_texts)} queries using '{MODEL_EMBEDDING}'...")
    
    for i in tqdm(range(0, len(query_texts), BATCH_SIZE), desc="Encoding query batches"):
        batch_texts = query_texts[i:i + BATCH_SIZE]
        batch_texts = [text if text else " " for text in batch_texts]
        
        try:
            response = client.embeddings.create(model=MODEL_EMBEDDING, input=batch_texts)
            batch_embeddings = np.array([d.embedding for d in response.data], dtype=np.float32)
            all_embeddings.append(batch_embeddings)
        except Exception as e:
            print(f"\n❌ Error calling OpenAI API on batch starting at index {i}: {e}")
            print("Aborting embedding generation.")
            sys.exit(1)

    # --- 5. Combine and Normalize Embeddings ---
    print("Combining and normalizing query embeddings...")
    qset_embeddings = np.vstack(all_embeddings)
    
    # L2 normalize for cosine similarity/inner product search, consistent with eval script
    qset_embeddings_normalized = l2n(qset_embeddings).astype(np.float32)
    
    # --- 6. Save the Output Files ---
    print(f"Saving query IDs to '{ids_path}'...")
    np.save(ids_path, np.array(query_ids, dtype="S")) # Save as bytes
    
    print(f"Saving normalized query embeddings to '{emb_path}'...")
    np.save(emb_path, qset_embeddings_normalized)

    print("\n✅ Query embedding generation complete!")
    print(f"Shape of embeddings: {qset_embeddings_normalized.shape}")
    print(f"Number of query IDs: {len(query_ids)}")

if __name__ == "__main__":
    main()


# python vectorrag/create_qset_embeddings.py \
#     --qset data/qqset.json \
#     --output_dir data/qset_embeddings/openai-large
