import json, argparse, sys
from pathlib import Path

def load_paper(json_dir, paper_id):
    p = Path(json_dir) / f"{paper_id}.json"
    if not p.exists():
        # try with .chunks.json-style id variants
        cand = list(Path(json_dir).glob(f"{paper_id}*.json"))
        return None, cand
    with p.open("r", encoding="utf-8") as f:
        return json.load(f), None

def para_lookup(paper, para_id):
    for sec in paper.get("sections", []):
        for para in sec.get("paras", []):
            if para.get("id") == para_id:
                return para
    return None

def find_para_by_quote(paper, quote):
    hits = []
    q = quote.strip()
    if not q: return hits
    for sec in paper.get("sections", []):
        for para in sec.get("paras", []):
            txt = (para.get("text") or "")
            if q in txt:
                hits.append(para["id"])
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qset", required=True)
    ap.add_argument("--json-dir", required=True)
    args = ap.parse_args()

    with open(args.qset, "r", encoding="utf-8") as f:
        qset = json.load(f)

    bad = 0
    for q in qset:
        pid = q.get("paper_id")
        paper, suggestions = load_paper(args.json_dir, pid)
        if paper is None:
            bad += 1
            print(f"[MISSING PAPER] qid={q.get('qid')} paper_id='{pid}' not found. Suggestions: {[s.name for s in suggestions]}")
            continue
        for g in q.get("gold", []):
            para_id = g.get("para_id")
            quote = g.get("quote","").strip()
            para = para_lookup(paper, para_id) if para_id else None
            if para is None:
                # try to auto-suggest by quote
                hints = find_para_by_quote(paper, quote) if quote else []
                bad += 1
                print(f"[BAD PARA_ID] qid={q.get('qid')} para_id='{para_id}' not in JSON. Suggestions by quote: {hints[:5]}")
                continue
            txt = para.get("text","")
            if quote and quote not in txt:
                bad += 1
                print(f"[QUOTE MISMATCH] qid={q.get('qid')} para_id='{para_id}' quote not found. Consider shortening or correcting.")
    if bad == 0:
        print("✅ Qset looks consistent.")
    else:
        print(f"❌ Found {bad} issues. Fix before evaluation.")
        sys.exit(1)

if __name__ == "__main__":
    main()


# python vectorrag/tools/validate_qset.py --qset data/qset_auto.json --json-dir data/processed_to_json
