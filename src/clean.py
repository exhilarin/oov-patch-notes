"""Step 2: turn raw Steam announcements into token-labelled patch notes.

For every announcement:
 1. split the BBCode text into sections at developer-written headers ([h1]-[h3], or a line
    that is only bold/underlined text);
 2. map each header to a change type with configs/header_map.yaml (generic and
    promotional headers are dropped together with their section);
 3. split each kept section into change items (list bullets, otherwise lines);
 4. clean every item (BBCode, HTML, URLs, bullets) and tokenize it;
 5. keep English announcements that contain at least two different change types,
    remove duplicates, and cut long ones into chunks of at most --max-tokens tokens
    at item boundaries.
Headers and list markup are NOT part of the model input: the model sees only the
flat token sequence and must recover where every change starts and what type it is.
Each item is a BIO span: B-<TYPE> on its first token, I-<TYPE> on the rest.

Output: data/processed/patch_notes_clean.jsonl.gz and data/processed/cleaning_report.json
"""
import argparse
import gzip
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

import yaml
from langdetect import DetectorFactory, LangDetectException, detect

ROOT = Path(__file__).resolve().parents[1]
DetectorFactory.seed = 42

HEADER_RE = re.compile(r"\[h[1-3]\](.*?)\[/h[1-3]\]", re.S | re.I)
# a line that consists only of bold and/or underlined text acts as a header too
BOLD_LINE_RE = re.compile(r"^\s*(?:\[(?:b|u|i)\]\s*)+([^\[\]\n]{2,60}?)\s*(?:\[/(?:b|u|i)\]\s*)+:?\s*$", re.M | re.I)
DROP_BLOCK_RE = re.compile(r"\[(img|previewyoutube|video|table|code|quote|spoiler)[^\]]*\].*?\[/\1\]", re.S | re.I)
URL_TAG_RE = re.compile(r"\[url=[^\]]*\](.*?)\[/url\]", re.S | re.I)
TAG_RE = re.compile(r"\[/?[a-z0-9*]+(?:=[^\]]*)?\]", re.I)
HTML_RE = re.compile(r"<[^>]+>")
CLAN_IMG_RE = re.compile(r"\{STEAM_CLAN_IMAGE\}\S*|\{STEAM_APP_IMAGE\}\S*")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
BULLET_RE = re.compile(r"^\s*(?:[-–—•·*>+]+|\d{1,2}[.)])\s+")
MAX_ITEM_TOKENS = 80  # longer "items" are paragraphs, not change entries
TOKEN_RE = re.compile(r"<URL>|\w+(?:[-'’.]\w+)*%?|[^\w\s]")


def norm_header(h: str) -> str:
    h = html.unescape(TAG_RE.sub("", h)).lower()
    h = re.sub(r"v?\d+(\.\d+)*[a-z]?", " ", h)           # version numbers, dates
    h = re.sub(r"[^a-z&/ ]+", " ", h).replace("&", " ").replace("/", " ")
    return re.sub(r"\s+", " ", h).strip()


class HeaderMapper:
    def __init__(self, path: Path):
        cfg = yaml.safe_load(path.read_text())
        self.skip = [re.compile(p) for p in cfg["skip"]]
        self.rules = [(lab, re.compile(p)) for lab, p in cfg["labels"].items()]

    def __call__(self, header: str):
        h = norm_header(header)
        if not h or len(h.split()) > 6 or any(p.search(h) for p in self.skip):
            return None
        for lab, p in self.rules:
            if p.search(h):
                return lab
        return None


def sections(text: str):
    """Yield (header, body) pairs; text before the first header is ignored."""
    text = BOLD_LINE_RE.sub(lambda m: f"[h3]{m.group(1)}[/h3]", text)
    parts = HEADER_RE.split(text)
    for i in range(1, len(parts) - 1, 2):
        yield parts[i], parts[i + 1]


def clean_text(s: str) -> str:
    s = URL_TAG_RE.sub(r"\1", s)
    s = TAG_RE.sub(" ", s)
    s = CLAN_IMG_RE.sub(" ", s)
    s = html.unescape(HTML_RE.sub(" ", s))
    s = URL_RE.sub("<URL>", s)
    s = BULLET_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def items(body: str):
    body = DROP_BLOCK_RE.sub(" ", body)
    # only list-structured text counts as change items; prose paragraphs are skipped
    if "[*]" in body:
        raw = re.split(r"\[\*\]", body)[1:]
    else:
        raw = [l for l in body.splitlines() if BULLET_RE.match(TAG_RE.sub("", l))]
    for r in raw:
        toks = TOKEN_RE.findall(clean_text(r))
        if 2 <= len(toks) <= MAX_ITEM_TOKENS and any(t.isalpha() for t in toks):
            yield toks


def to_example(spans):
    tokens, tags = [], []
    for lab, toks in spans:
        tokens += toks
        tags += [f"B-{lab}"] + [f"I-{lab}"] * (len(toks) - 1)
    return tokens, tags


def chunks(spans, max_tokens):
    cur, n = [], 0
    for sp in spans:
        if len(sp[1]) > max_tokens:
            continue
        if cur and n + len(sp[1]) > max_tokens:
            yield cur
            cur, n = [], 0
        cur.append(sp)
        n += len(sp[1])
    if cur:
        yield cur


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=ROOT / "data/raw", type=Path, help="folder with patch_notes_raw_*.jsonl.gz")
    ap.add_argument("--map", default=ROOT / "configs/header_map.yaml", type=Path)
    ap.add_argument("--max-tokens", default=256, type=int)
    ap.add_argument("--out", default=ROOT / "data/processed/patch_notes_clean.jsonl.gz", type=Path)
    args = ap.parse_args()

    mapper = HeaderMapper(args.map)
    rep = Counter()
    header_use = Counter()
    seen = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.raw.glob("patch_notes_raw_*.jsonl.gz")) if args.raw.is_dir() else [args.raw]
    lines = (l for s in shards for l in gzip.open(s, "rt", encoding="utf-8"))
    with gzip.open(args.out, "wt", encoding="utf-8") as out:
        for line in lines:
            a = json.loads(line)
            rep["1_raw_announcements"] += 1
            secs = list(sections(a["contents"]))
            if not secs:
                continue
            rep["2_with_section_headers"] += 1
            spans = []
            for h, body in secs:
                lab = mapper(h)
                header_use[(norm_header(h), lab or "-")] += 1
                if lab:
                    spans += [(lab, t) for t in items(body)]
            if len({lab for lab, _ in spans}) < 2:
                continue
            rep["3_two_or_more_change_types"] += 1
            sample = " ".join(" ".join(t) for _, t in spans)[:2000]
            try:
                if detect(sample) != "en":
                    continue
            except LangDetectException:
                continue
            rep["4_english"] += 1
            key = hashlib.md5(sample.lower().encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            rep["5_after_deduplication"] += 1
            for k, ch in enumerate(chunks(spans, args.max_tokens)):
                if len({lab for lab, _ in ch}) < 2:
                    continue
                tokens, tags = to_example(ch)
                out.write(json.dumps({"id": f"{a['gid']}-{k}", "appid": a["appid"], "game": a["game"],
                                      "date": a["date"], "tokens": tokens, "tags": tags}, ensure_ascii=False) + "\n")
                rep["6_examples_(chunks)"] += 1
                rep["7_change_items"] += len(ch)
                rep["8_tokens"] += len(tokens)

    top = [{"header": h, "label": lab, "count": c} for (h, lab), c in header_use.most_common(400)]
    report = {"steps": dict(sorted(rep.items())), "top_headers": top}
    (args.out.parent / "cleaning_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    for k, v in sorted(rep.items()):
        print(f"{k:32s} {v:>10,}")


if __name__ == "__main__":
    main()
