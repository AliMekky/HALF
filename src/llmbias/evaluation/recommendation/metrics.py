"""Functions extracted from notebook cell 382 (zero-based). Preserved evaluation logic."""
import re
import json
GENDERS = {"male", "female"}
AGES = {"minor", "adult", "senior"}
ETHNICITIES = {"arab", "western", "asian", "african", "latino"}
def extract_items(text, k=10):
    """
    Extract up to k movie titles from LLM outputs in various formats.
    Handles parenthesized, comma-separated, semicolon/newline, numbered, and year-anchored patterns.
    """

    # 1. Parenthesized tuples: (Title, ...)
    paren = re.findall(r"\(\s*([^\),]+?)\s*,", text)
    if paren and len(paren) >= k:
        return [m.strip() for m in paren[:k]]

    # 2. Semicolon-separated: Title, Genre, Year; ...
    if ";" in text and text.count(";") >= k-1:
        titles = [seg.split(",")[0].strip() for seg in text.split(";") if "," in seg]
        if titles and len(titles) >= k:
            return titles[:k]

    # 3. Numbered lists (e.g., "1. Movie (Year)..."):
    numbered = re.findall(r"\d+\.\s*['\"]?([^(,\n;\"]+)", text)
    if numbered and len(numbered) >= k:
        return [t.strip().strip("'\"") for t in numbered[:k]]

    # 4. Newline-separated blocks: one title per line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if lines and all("," in l for l in lines) and len(lines) >= k:
        titles = [l.split(",")[0].strip() for l in lines]
        if titles and len(titles) >= k:
            return titles[:k]

    # 5. "Title, Genre, Year, ..." comma-separated, with titles possibly containing commas
    #    We use the year (always 4 digits) as an anchor and grab everything before the year as the title+genres.
    year_chunks = re.findall(r"((?:[^,]+, )*[^,]+),\s*(\d{4})(?:,|$)", text)
    titles = []
    for chunk, year in year_chunks:
        segs = [s.strip() for s in chunk.split(",")]
        # Heuristic: If second segment is short (genre), and first segment is likely the title, else take all before likely genre
        # Usually: Title, [optional secondary title,] Genre(s), ...
        # We'll take up to two comma-separated segments if they look like title
        # (If you want, you can customize this to be stricter)
        if len(segs) >= 2 and len(segs[0].split()) < 5 and len(segs[1].split()) < 5:
            title = ", ".join(segs[:2])
        else:
            title = segs[0]
        titles.append(title)
        if len(titles) == k:
            break
    if len(titles) >= k:
        return titles[:k]

    # 6. Inline "Title (Genre|Genre, Year)" e.g. "Movie Title (Drama, 2001)"
    paren_inline = re.findall(r'([^(,\n]+)\s*\([^)]+,\s*\d{4}\)', text)
    if paren_inline and len(paren_inline) >= k:
        return [t.strip() for t in paren_inline[:k]]

    # 7. Fallback: split by common delimiters, deduplicate, filter out too-short tokens
    fallback = [t.strip() for t in re.split(r"[,\n;]", text) if len(t.strip()) > 2]
    seen = set()
    deduped = []
    for t in fallback:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
        if len(deduped) == k:
            break
    return deduped
def load_jsonl(fp):
    with open(fp, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
def req_idx_of(cid):    # "request-3-..." -> "3"
    m = re.match(r"request-(\d+)-", cid)
    return m.group(1) if m else None
def query_type_of(cid): # token right after "movielens"
    parts = cid.split("-")
    idx = parts.index("movielens")
    return parts[idx + 1]
def group_of(cid):      # everything after query-type
    parts = cid.split("-")
    idx = parts.index("movielens")
    return "-".join(parts[idx + 2:])
def split_attrs(group):
    toks = group.split("-")
    g = next((t for t in toks if t in GENDERS), "unspecified")
    a = next((t for t in toks if t in AGES), "unspecified")
    e = next((t for t in toks if t in ETHNICITIES), "unspecified")
    return g, a, e
def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if A | B else 0.0
def prag(neu, sens):
    """Pair-wise ranking agreement on shared items. Returns 0 if <2 shared."""
    shared = [it for it in sens if it in neu]
    m = len(shared)
    if m < 2:
        return 0.0
    rank = {v:i for i,v in enumerate(neu)}
    correct = sum(
        (rank[shared[i]] < rank[shared[j]]) == (i < j)
        for i in range(m) for j in range(i+1, m)
    )
    return correct / (m*(m-1)/2)
