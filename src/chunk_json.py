# Fixed-size chunking with a FIXED prefix budget.
# Windows slide across the entire section (not per paragraph).
import json, argparse, sys, bisect
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple
import tiktoken

DEFAULT_TOKENS_PER_CHUNK = 500
DEFAULT_OVERLAP = 0          # primary fairness: no overlap
DEFAULT_PREFIX_BUDGET = 40
TOKENIZER = "cl100k_base"

def encode(enc, s: str) -> List[int]:
    return enc.encode(s or "")

def decode(enc, toks: List[int]) -> str:
    return enc.decode(toks)

def build_prefix_and_capacity(
    enc,
    max_tokens: int,
    title: str,
    section_title: str,
    prefix_budget: int,
    min_body_tokens: int = 20,
) -> Tuple[str, int, int]:
    full = f"{title}\nSection: {section_title}".strip()
    tok = encode(enc, full)
    if len(tok) > prefix_budget:
        short = f"Section: {section_title}".strip()
        tok = encode(enc, short)
        if len(tok) > prefix_budget:
            tok = tok[:prefix_budget]
            pref_text = decode(enc, tok)
        else:
            pref_text = short
    else:
        pref_text = full
    tok = encode(enc, pref_text)[:prefix_budget]
    pref_text = decode(enc, tok)
    body_capacity = max_tokens - len(tok)
    if body_capacity < min_body_tokens:
        pref_text = ""
        tok = []
        body_capacity = max_tokens
    return pref_text, body_capacity, len(tok)

def iter_input_files(inpath: Path, recursive: bool, pattern: str) -> Iterable[Path]:
    if inpath.is_file():
        if inpath.suffix.lower() == ".json":
            yield inpath
        return
    it = inpath.rglob(pattern) if recursive else inpath.glob(pattern)
    for p in it:
        if p.is_file() and p.suffix.lower() == ".json":
            yield p

def slide_section(enc, paras: List[Dict[str,Any]], body_capacity: int, overlap: int):
    """Yield (start_tok, end_tok, para_idx_lo, para_idx_hi) windows across the whole section."""
    # Build one big token list and cumulative lengths for para coverage
    para_tokens: List[List[int]] = []
    cum = [0]
    for p in paras:
        txt = (p.get("text") or "").strip()
        if (not txt) or txt.startswith("[no_text_content"):
            para_tokens.append([])
        else:
            para_tokens.append(encode(enc, txt))
        cum.append(cum[-1] + len(para_tokens[-1]))
    total = cum[-1]
    if total == 0:
        return  # no content

    step = max(1, body_capacity - max(0, overlap))
    start = 0
    while start < total:
        end = min(start + body_capacity, total)
        # map token range -> para index range
        # para i covers tokens [cum[i], cum[i+1])
        i_lo = max(0, bisect.bisect_right(cum, start) - 1)
        i_hi = max(0, bisect.bisect_left(cum, end) - 1)
        yield start, end, i_lo, i_hi, para_tokens, cum, total
        if end >= total:
            break
        start += step

def slice_tokens(enc, para_tokens: List[List[int]], cum: List[int], start: int, end: int) -> List[int]:
    """Extract token slice [start, end) from concatenated paras."""
    out: List[int] = []
    # find first para
    pi = max(0, bisect.bisect_right(cum, start) - 1)
    pos = start
    while pos < end and pi < len(para_tokens):
        base = cum[pi]
        ptoks = para_tokens[pi]
        if not ptoks:
            pi += 1
            continue
        s = max(0, pos - base)
        e = min(len(ptoks), end - base)
        if s < e:
            out.extend(ptoks[s:e])
            pos = base + e
        else:
            pos = base + len(ptoks)
        pi += 1
    return out

def chunk_file(inpath: Path, outdir: Path, max_tokens: int, overlap: int, prefix_budget: int) -> Path:
    data = json.loads(inpath.read_text(encoding="utf-8"))
    enc = tiktoken.get_encoding(TOKENIZER)

    paper_id = data.get("id") or inpath.stem
    title = (data.get("title") or paper_id).strip()
    sections = data.get("sections") or []

    items: List[Dict[str, Any]] = []
    idx = 0

    for sec in sections:
        sec_id = sec.get("id") or "section"
        sec_title = (sec.get("title") or "Section").strip()
        paras = sec.get("paras") or []

        prefix, body_capacity, prefix_tok_count = build_prefix_and_capacity(
            enc, max_tokens, title, sec_title, prefix_budget
        )
        prefix_tokens = encode(enc, prefix + ("\n\n" if prefix else ""))

        # Slide across the whole section
        for start, end, i_lo, i_hi, para_tokens, cum, total in slide_section(enc, paras, body_capacity, overlap):
            if total == 0:
                continue
            body_tok_list = slice_tokens(enc, para_tokens, cum, start, end)
            if not body_tok_list:
                continue
            chunk_tokens = prefix_tokens + body_tok_list
            chunk_text = decode(enc, chunk_tokens)

            # metadata: para_ids covered and page span union
            para_ids = []
            pmins, pmaxs = [], []
            for pi in range(i_lo, i_hi+1):
                p = paras[pi]
                pid = p.get("id") or f"{sec_id}-para"
                if pid not in para_ids:
                    para_ids.append(pid)
                ps = p.get("page_span") or [None, None]
                if isinstance(ps[0], int): pmins.append(ps[0])
                if isinstance(ps[1], int): pmaxs.append(ps[1])

            items.append({
                "id": f"{paper_id}::{sec_id}::fixed_{idx:04d}",
                "text": chunk_text,
                "metadata": {
                    "paper_id": paper_id,
                    "title": title,
                    "section_id": sec_id,
                    "section_title": sec_title,
                    "para_ids": para_ids,
                    "page_span": [
                        (min(pmins) if pmins else None),
                        (max(pmaxs) if pmaxs else None)
                    ],
                    "tokenizer": TOKENIZER,
                    "tokens_per_chunk": max_tokens,
                    "overlap": overlap,
                    "prefix_budget": prefix_budget,
                    "prefix_tokens": len(prefix_tokens),
                    "body_tokens": len(body_tok_list),
                }
            })
            idx += 1

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{paper_id}.chunks.json"
    outpath.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote chunks -> {outpath}")
    return outpath

def main():
    ap = argparse.ArgumentParser(description="Fixed-size chunking with fixed prefix budget.")
    ap.add_argument("--inpath", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tokens-per-chunk", type=int, default=DEFAULT_TOKENS_PER_CHUNK)
    ap.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    ap.add_argument("--prefix-budget", type=int, default=DEFAULT_PREFIX_BUDGET)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--pattern", default="*.json")
    args = ap.parse_args()

    inpath = Path(args.inpath); outdir = Path(args.outdir)
    files = list(iter_input_files(inpath, args.recursive, args.pattern))
    if not files:
        print(f"No input files found from: {inpath}", file=sys.stderr); sys.exit(1)

    ok = bad = 0
    for fp in files:
        try:
            chunk_file(fp, outdir, args.tokens_per_chunk, args.overlap, args.prefix_budget)
            ok += 1
        except Exception as e:
            print(f"Failed on {fp}: {e}", file=sys.stderr); bad += 1
    if len(files) > 1:
        print(f"Done. {ok} succeeded, {bad} failed out of {len(files)} file(s).")

if __name__ == "__main__":
    main()


# python vectorrag/chunk_json.py --inpath data/processed_to_json --outdir data/chunks_fixed250 --tokens-per-chunk 250 --overlap 0 --prefix-budget 40
