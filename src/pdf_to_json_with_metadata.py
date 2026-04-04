"""
- Input PDFs:                       --pattern "data/*.pdf"     (default)
- Output JSON per paper:            --output_dir data/processed_to_json  (default)
- Extractors (per-PDF):
    engine=auto (default): if text-layer → PyMuPDF; else Docling OCR
    engine=pymupdf:         force PyMuPDF
    engine=docling:         force Docling OCR
    engine=nougat:          use Nougat only (no page spans)

- Nougat integration (math-preserving Markdown):
    --nougat append  (default)  : append a “Nougat Body” section after page-based sections
    --nougat none               : skip Nougat entirely (fastest)
    --nougat replace            : ignore page-based extraction; sectionize Nougat text only

Install (CPU only):
  pip install pymupdf
  pip install nougat-ocr torch --index-url https://download.pytorch.org/whl/cpu
  # optional OCR fallback:
  pip install docling
  # and install Tesseract on your OS if you plan to OCR

Run:
  python pdf_to_json_with_metadata.py \
    --pattern "data/*.pdf" \
    --output_dir data/processed_to_json \
    --engine auto \
    --nougat append \
    --max-chars 1200
"""

from pathlib import Path
import json, sys, re, hashlib, argparse, subprocess, unicodedata
from typing import List, Dict, Tuple, Any, Optional

# =========================
# Defaults / CLI
# =========================
DEFAULT_PATTERN = "data/*.pdf"
DEFAULT_OUT_DIR = Path("data/processed_to_json")
DEFAULT_MAX_CHARS = 1200

# =========================
# Optional deps
# =========================
try:
    import fitz  # PyMuPDF
    PYMUPDF_OK = True
except Exception:
    PYMUPDF_OK = False

try:
    from docling.document_converter import DocumentConverter
    DOCLING_OK = True
except Exception:
    DOCLING_OK = False

# =========================
# Heuristics / regexes
# =========================
SECTION_PAT = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)*)\.\s+(?P<title>[A-Z][^\n]{0,140})\s*$")
UNNUMBERED_HEADS = {
    "abstract","introduction","background","related work","preliminaries",
    "method","methods","approach","results","experiments","evaluation",
    "discussion","conclusion","conclusions","acknowledgments","acknowledgements",
    "appendix","supplementary material","references","bibliography"
}
SCI_TICK_RE = re.compile(r"(?:\b1e[+-]?\d+\b|×10\^[-+]?\d+)", re.I)

# =========================
# Utilities
# =========================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_para(s: str) -> str:
    # unicode normalization + ligatures & soft hyphen
    s = unicodedata.normalize("NFKC", s or "")
    s = (s.replace("ﬁ","fi").replace("ﬂ","fl")
           .replace("\u00ad","").replace("\u00a0"," "))
    # de-hyphenate across line breaks
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)     # trans-\nform -> transform
    # also fix hyphen + spaces from linewraps, e.g., "Deci- sion" -> "Decision"
    s = re.sub(r'(?<=[a-z])-\s+(?=[a-z])', '', s)
    # whitespace cleanup
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def looks_like_axis_ticks(line: str) -> bool:
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
    sci_hits = len(SCI_TICK_RE.findall(line))
    alpha_words = re.findall(r"[A-Za-z]{2,}", line)
    return (len(nums) >= 6 and len(alpha_words) < 3) or sci_hits >= 3

def looks_like_glyph_spam(line: str) -> bool:
    s = re.sub(r"\s+", "", line)
    if len(s) < 6:
        return False
    max_rep = max((s.count(ch) for ch in set(s)))
    return (max_rep / len(s)) >= 0.6  # dominated by one char (GGGG..., etc.)

def clean_lines(text: str) -> str:
    """
    Remove axis ticks, glyph spam (e.g., G G G …), repeated X1/X2/X3 labels,
    and low-info numeric lines. Also excises the spam *inside* lines so captions survive.
    """
    SINGLE_CAP_SEQ_THRESHOLD = 8   # >= 8 single-letter tokens in a row
    NUMERIC_SEQ_THRESHOLD    = 10  # >= 10 numbers in a row
    XVAR_SEQ_THRESHOLD       = 4   # >= 4 X<digit> in a row

    cap_seq_re   = re.compile(rf'(?:\b[A-Z]\b[\s,]*){{{SINGLE_CAP_SEQ_THRESHOLD},}}')
    num_seq_re   = re.compile(rf'(?:[-+]?(?:\d+(?:\.\d+)?)\s+){{{NUMERIC_SEQ_THRESHOLD},}}')
    xvar_seq_re  = re.compile(rf'(?:X\d+\s*){{{XVAR_SEQ_THRESHOLD},}}', re.IGNORECASE)

    out = []
    for ln in text.splitlines():
        t = ln.strip()
        if not t:
            out.append("")
            continue

        # excise spammy sequences *inside* the line first
        t = cap_seq_re.sub(" ", t)
        t = num_seq_re.sub(" ", t)
        t = xvar_seq_re.sub(" ", t)

        # drop common license/footer or author-email rows
        if re.search(r'\bLicense:\b|creativecommons\.org|jmlr\.org', t, re.I):
            continue
        if t.count('@') >= 2:
            continue

        # normalize spacing after cuts
        t = re.sub(r"\s{2,}", " ", t).strip()

        # classic axis-tick lines
        if looks_like_axis_ticks(t):
            continue

        # glyph-dominated lines (after excision this rarely triggers, but keep it)
        if looks_like_glyph_spam(t):
            continue

        # super low-info numeric lines
        letters = sum(c.isalpha() for c in t)
        if letters < 5 and len(t) > 20:
            continue

        # leftover single-letter lines (e.g., just "G")
        if re.fullmatch(r"[A-Za-z]", t):
            continue

        out.append(t)

    # re-normalize after filtering
    s = "\n".join(out)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "section"

def split_chars(text: str, max_chars: int) -> List[str]:
    """Split long paragraphs to ~max_chars, on whitespace."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], []
    cur_len = 0
    for token in text.split():
        tok_len = len(token)
        if cur and cur_len + 1 + tok_len > max_chars:
            parts.append(" ".join(cur))
            cur = [token]
            cur_len = tok_len
        else:
            if cur:
                cur_len += 1 + tok_len
            else:
                cur_len += tok_len
            cur.append(token)
    if cur:
        parts.append(" ".join(cur))
    return parts

# =========================
# Probing
# =========================
def has_text_layer(pdf_path: Path) -> bool:
    """Quick probe: if any of first 2 pages has text, assume digital."""
    if not PYMUPDF_OK:
        return False
    try:
        doc = fitz.open(pdf_path)
        for i in range(min(2, len(doc))):
            if doc[i].get_text("text").strip():
                return True
        return False
    except Exception:
        return False

# =========================
# Extractors
# =========================
def extract_with_pymupdf(pdf_path: Path) -> Tuple[str, List[str], int]:
    if not PYMUPDF_OK:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")
    doc = fitz.open(pdf_path)
    title = (doc.metadata or {}).get("title") or pdf_path.stem
    pages = []
    for i, page in enumerate(doc, start=1):
        # Prefer block-based layout and filter out blocks that are mostly single-letter tokens
        blocks = page.get_text("blocks") or []
        blocks = sorted(blocks, key=lambda b: (b[1], b[0]))  # y, then x
        kept = []
        for b in blocks:
            if not (isinstance(b, (list, tuple)) and len(b) > 4):
                continue
            txt = (b[4] or "").strip()
            if not txt:
                continue
            tokens = re.findall(r"\b\w+\b", txt)
            if tokens:
                single_caps = sum(1 for tok in tokens if len(tok) == 1 and tok.isalpha() and tok.isupper())
                # if >70% tokens are single caps AND there are many tokens, skip block (letter salad)
                if single_caps >= 20 and single_caps / max(1, len(tokens)) > 0.7:
                    continue
            kept.append(txt)
        t = "\n".join(kept)
        # fallback if blocks were empty
        if not t.strip():
            t = page.get_text("text")
        t = clean_lines(t)
        pages.append(f"[PAGE {i}]\n{t.rstrip()}")
    return title.strip(), pages, len(doc)

def extract_with_docling(pdf_path: Path) -> Tuple[str, List[str], int]:
    if not DOCLING_OK:
        raise RuntimeError("Docling not installed. Run: pip install docling")
    converter = DocumentConverter(do_ocr=True)
    res = converter.convert(pdf_path)
    doc = res.document
    title = (getattr(doc.metadata, "title", None) or pdf_path.stem).strip()
    pages = []
    for p in doc.pages:
        blocks = []
        for b in p.blocks:
            try:
                if getattr(b, "category", "") == "table":
                    blocks.append(b.export_to_markdown())
                else:
                    txt = getattr(b, "text", "") or ""
                    if txt:
                        blocks.append(txt)
            except Exception:
                continue
        t = clean_lines("\n".join(blocks))
        pages.append(f"[PAGE {p.page_no}]\n{t}")
    return title, pages, len(doc.pages)

def extract_with_nougat(pdf_path: Path, timeout: int = 1200) -> Tuple[str, str]:
    """
    Returns (title_guess, markdown_text). Runs Nougat CLI on CPU.
    """
    try:
        out = subprocess.run(
            ["nougat", str(pdf_path)],
            check=True, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError as e:
        raise RuntimeError("Nougat CLI not found. Install with: pip install nougat-ocr") from e
    md = clean_lines(out.stdout)
    # naive title guess: first non-empty line up to 200 chars
    title_guess = next((ln.strip()[:200] for ln in md.splitlines() if ln.strip()), pdf_path.stem)
    return title_guess, md

# =========================
# Title guesser (no Nougat required)
# =========================
def guess_title_from_pages(pages: List[str]) -> Optional[str]:
    """
    Heuristically extract a paper title from the first 1–2 pages, no Nougat required.
    Looks for a 3–12 word Title-Case phrase and avoids journal headers, dates, emails, etc.
    """
    if not pages:
        return None

    head = "\n".join(pages[:2]) if len(pages) > 1 else pages[0]
    head = re.sub(r"\[PAGE \d+\]\s*", "", head)

    BAD_HINTS = (
        "journal of", "submitted", "revised", "published", "editor:",
        "license", "creativecommons", "jmlr.org", "arxiv", "doi:",
        "university", "department", "institute", "center", "centre",
        "keywords:", "abstract", "©", "c⃝", "https://", "http://", "@"
    )

    phrase_re = re.compile(r"([A-Z][A-Za-z0-9\-]{2,}(?:\s+[A-Z][A-Za-z0-9\-]{2,}){2,11})")

    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]

    def bad_line(s: str) -> bool:
        s_low = s.lower()
        return any(b in s_low for b in BAD_HINTS)

    def score_candidate(s: str) -> float:
        words = s.split()
        n = len(words)
        if n < 3 or n > 14:
            return -1
        if not (20 <= len(s) <= 140):
            return -1
        caps_long = sum(1 for w in words if w.isupper() and len(w) > 5)
        return n - 0.5 * caps_long

    best: Optional[str] = None
    best_score = -1.0

    # Pass 1: line-level
    for ln in lines[:50]:
        if bad_line(ln):
            continue
        for m in phrase_re.finditer(ln):
            cand = m.group(1).strip()
            if bad_line(cand):
                continue
            sc = score_candidate(cand)
            if sc > best_score:
                best, best_score = cand, sc

    # Pass 2: whole blob
    if best is None:
        blob = re.sub(r"\s+", " ", head)
        for m in phrase_re.finditer(blob):
            cand = m.group(1).strip()
            if bad_line(cand):
                continue
            sc = score_candidate(cand)
            if sc > best_score:
                best, best_score = cand, sc

    return best

# =========================
# Sectionizers
# =========================
def to_sections_from_pages(pages: List[str], max_chars: int) -> List[Dict[str, Any]]:
    """
    Convert page-marked text to sections→paragraphs with stable IDs and page spans.
    """
    full = "\n\n".join(pages)
    page_chunks = re.split(r"\[PAGE (\d+)\]\s*", full)
    page_map: Dict[int, str] = {}
    for i in range(1, len(page_chunks), 2):
        page_map[int(page_chunks[i])] = page_chunks[i+1]

    scan: List[str] = []
    for pno in sorted(page_map):
        scan.append(f"[P{pno}]")
        scan.extend(page_map[pno].splitlines())

    sections: List[Dict[str, Any]] = []
    cur_title: Optional[str] = None
    cur_buf: List[Tuple[str,int]] = []  # (paragraph_text_raw, approx_page)
    approx_page = 1
    seen_text_on_page = set()

    def flush_section():
        nonlocal cur_title, cur_buf, sections
        if cur_title is None:
            return
        sec_id = slugify(cur_title)
        paras = []
        para_idx = 0
        for (ptxt_raw, ppage) in cur_buf:
            ptxt = normalize_para(ptxt_raw)
            if not ptxt:
                continue
            for chunked in split_chars(ptxt, max_chars):
                para_idx += 1
                paras.append({
                    "id": f"{sec_id}-p{para_idx:03d}",
                    "text": chunked,
                    "page_span": [ppage, ppage]
                })
        if paras:
            sections.append({"id": sec_id, "title": cur_title, "paras": paras})
        cur_buf = []

    def flush_front_matter():
        nonlocal cur_buf, sections
        if not cur_buf:
            return
        prev_title = "Front Matter"
        sec_id = slugify(prev_title)
        paras = []
        para_idx = 0
        for (ptxt_raw, ppage) in cur_buf:
            ptxt = normalize_para(ptxt_raw)
            if not ptxt:
                continue
            for chunked in split_chars(ptxt, max_chars):
                para_idx += 1
                paras.append({
                    "id": f"{sec_id}-p{para_idx:03d}",
                    "text": chunked,
                    "page_span": [ppage, ppage]
                })
        if paras:
            sections.append({"id": sec_id, "title": prev_title, "paras": paras})
        cur_buf = []

    def push_text(line: str):
        nonlocal cur_buf, approx_page, seen_text_on_page
        if not line:
            return
        seen_text_on_page.add(approx_page)
        if not cur_buf:
            cur_buf.append((line, approx_page))
        else:
            txt, pg = cur_buf[-1]
            if len(txt) >= max_chars:
                cur_buf.append((line, approx_page))
            else:
                cur_buf[-1] = (txt + " " + line, pg)

    for ln in scan:
        m_page = re.match(r"\[P(\d+)\]$", ln.strip())
        if m_page:
            approx_page = int(m_page.group(1)); continue

        raw = ln.rstrip()
        if not raw:
            continue

        m = SECTION_PAT.match(raw)
        if m and len(raw) <= 160:
            if cur_title is None and cur_buf:
                flush_front_matter()
            else:
                flush_section()
            num = m.group("num")
            title = m.group("title").strip()
            cur_title = f"{num}. {title}" if num else title
            continue

        if raw.strip().lower() in UNNUMBERED_HEADS and len(raw.split()) <= 5:
            if cur_title is None and cur_buf:
                flush_front_matter()
            else:
                flush_section()
            cur_title = raw.strip().title()
            continue

        push_text(raw)

    if cur_title is None:
        cur_title = "Body"
    flush_section()

    # emit stubs for pages with no text
    all_pages = sorted(page_map)
    missing = [p for p in all_pages if p not in seen_text_on_page]
    for p in missing:
        sec_id = f"page-{p:03d}"
        sections.append({
            "id": sec_id,
            "title": f"Page {p}",
            "paras": [{
                "id": f"{sec_id}-stub",
                "text": "[no_text_content: figures/tables only]",
                "page_span": [p, p]
            }]
        })

    # move page stubs to the end
    def sort_key(sec: Dict[str, Any]):
        t = sec["title"].lower()
        if t.startswith("page "):
            try:
                return (1, int(t.split()[-1]))
            except Exception:
                return (1, 10_000)
        return (0,)
    sections.sort(key=sort_key)
    sections = [s for s in sections if s.get("paras")]
    return sections

def to_sections_from_text(body: str, max_chars: int) -> List[Dict[str, Any]]:
    """
    Use numbered/unnumbered heading heuristics on a single text (e.g., Nougat).
    No page spans here -> page_span = [null, null].
    """
    lines = body.splitlines()
    sections: List[Dict[str, Any]] = []
    cur_title: Optional[str] = None
    buf: List[str] = []

    def flush():
        nonlocal cur_title, buf, sections
        if cur_title is None:
            return
        sec_id = slugify(cur_title)
        text = normalize_para(" ".join(buf))
        if text:
            paras = []
            for j, chunked in enumerate(split_chars(text, max_chars), start=1):
                paras.append({"id": f"{sec_id}-p{j:03d}", "text": chunked, "page_span": [None, None]})
            sections.append({"id": sec_id, "title": cur_title, "paras": paras})
        buf = []

    def flush_front_matter_text():
        nonlocal buf, sections
        if not buf:
            return
        prev_title = "Front Matter"
        sec_id = slugify(prev_title)
        text = normalize_para(" ".join(buf))
        if text:
            paras = []
            for j, chunked in enumerate(split_chars(text, max_chars), start=1):
                paras.append({"id": f"{sec_id}-p{j:03d}", "text": chunked, "page_span": [None, None]})
            sections.append({"id": sec_id, "title": prev_title, "paras": paras})
        buf = []

    for raw in lines:
        if not raw.strip():
            continue
        m = SECTION_PAT.match(raw)
        if m and len(raw) <= 160:
            if cur_title is None and buf:
                flush_front_matter_text()
            else:
                flush()
            num = m.group("num"); title = m.group("title").strip()
            cur_title = f"{num}. {title}" if num else title
            continue
        if raw.strip().lower() in UNNUMBERED_HEADS and len(raw.split()) <= 5:
            if cur_title is None and buf:
                flush_front_matter_text()
            else:
                flush()
            cur_title = raw.strip().title()
            continue
        buf.append(raw.strip())

    if cur_title is None:
        cur_title = "Body"
    flush()
    return sections

# =========================
# Core conversion
# =========================
def convert_pdf(
    pdf_path: Path,
    max_chars: int,
    engine: str = "auto",
    nougat_mode: str = "append"
) -> Dict[str, Any]:
    """
    engine: auto|pymupdf|docling|nougat
    nougat_mode: append|none|replace
    """
    if engine not in {"auto","pymupdf","docling","nougat"}:
        raise ValueError("--engine must be one of: auto, pymupdf, docling, nougat")
    if nougat_mode not in {"append","none","replace"}:
        raise ValueError("--nougat must be one of: append, none, replace")

    paper_id = pdf_path.stem
    page_count: Optional[int] = None
    title = pdf_path.stem

    # Pure Nougat (or replace page-extractor)
    if engine == "nougat" or nougat_mode == "replace":
        title_guess, md = extract_with_nougat(pdf_path)
        title = title_guess or pdf_path.stem
        sections = to_sections_from_text(md, max_chars=max_chars)  # no page spans
        if PYMUPDF_OK:
            try:
                with fitz.open(pdf_path) as d:
                    page_count = len(d)
            except Exception:
                page_count = None
        metadata = {
            "source_path": str(pdf_path),
            "page_count": page_count,
            "hash": sha256_file(pdf_path)
        }
        return {"id": paper_id, "title": title, "metadata": metadata, "sections": sections}

    # Page-based primary extractor (auto/pymupdf/docling)
    pages: List[str]
    if engine == "pymupdf" or (engine == "auto" and has_text_layer(pdf_path)):
        if not PYMUPDF_OK:
            raise RuntimeError("PyMuPDF not installed but required (engine=pymupdf/auto). pip install pymupdf")
        title, pages, page_count = extract_with_pymupdf(pdf_path)
    else:
        if not DOCLING_OK:
            # fallback to PyMuPDF if Docling not available
            if PYMUPDF_OK:
                title, pages, page_count = extract_with_pymupdf(pdf_path)
            else:
                raise RuntimeError("Neither Docling nor PyMuPDF available. Install at least one.")
        else:
            title, pages, page_count = extract_with_docling(pdf_path)

    sections = to_sections_from_pages(pages, max_chars=max_chars)

    # Nougat post-processing (append only)
    if nougat_mode == "append":
        try:
            _, md = extract_with_nougat(pdf_path)
            md_norm = normalize_para(md)
            if md_norm:
                sec_id = "nougat-body"
                paras = []
                for j, chunked in enumerate(split_chars(md_norm, max_chars), start=1):
                    paras.append({"id": f"{sec_id}-p{j:03d}", "text": chunked, "page_span": [None, None]})
                sections.append({"id": sec_id, "title": "Nougat Body", "paras": paras})
        except Exception:
            pass  # Nougat optional

    # --- Title fallback without Nougat (prefer from first pages) ---
    if title == paper_id or re.fullmatch(r'[A-Za-z0-9._-]+', title):
        try:
            cand = guess_title_from_pages(pages)
        except Exception:
            cand = None
        if cand:
            title = cand

    # --- Optional Nougat-based fallback (second chance) ---
    if (title == paper_id or re.fullmatch(r'[A-Za-z0-9._-]+', title)):
        try:
            title_guess, _ = extract_with_nougat(pdf_path)
            if title_guess and len(title_guess) > 10:
                title = title_guess
        except Exception:
            pass

    metadata = {
        "source_path": str(pdf_path),
        "page_count": page_count,
        "hash": sha256_file(pdf_path)
    }
    return {"id": paper_id, "title": title, "metadata": metadata, "sections": sections}

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser(description="GraphRAG-ready PDF→JSON (no GROBID)")
    ap.add_argument("--pattern", default=DEFAULT_PATTERN,
                    help=f"Glob for PDFs (default: {DEFAULT_PATTERN})")
    ap.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR,
                    help=f"Where to write <paper>.json (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help=f"Approx max chars per paragraph chunk (default: {DEFAULT_MAX_CHARS})")
    ap.add_argument("--engine", choices=["auto","pymupdf","docling","nougat"], default="auto",
                    help="Extraction engine: auto picks PyMuPDF if text-layer, else Docling; "
                         "'nougat' ignores pages entirely")
    ap.add_argument("--nougat", choices=["append","none","replace"], default="append",
                    help="How to use Nougat: append as extra section (default), skip, or replace page text")
    args = ap.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(Path().glob(args.pattern))
    if not pdfs:
        print(f"[WARN] No PDFs match {args.pattern}")
        return

    converted = 0
    for pdf in pdfs:
        try:
            print(f"Processing: {pdf}")
            record = convert_pdf(
                pdf_path=pdf,
                max_chars=args.max_chars,
                engine=args.engine,
                nougat_mode=args.nougat
            )
            out_path = out_dir / f"{record['id']}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            converted += 1
        except Exception as e:
            print(f"[WARN] Skipped {pdf.name}: {e}", file=sys.stderr)

    print(f"Done. Wrote {converted} JSON files to {out_dir.resolve()}")

if __name__ == "__main__":
    main()

# python pdf_to_json_with_metadata.py  --pattern "data/pdf_files/*.pdf" --output_dir data/preprocessed_to_json

# python pdf_to_json_with_metadata.py --pattern "data/pdf_files/*.pdf" --output_dir data/processed_to_json --engine auto --nougat append --max-chars 1200 
