"""Shared helpers: data loading, the label set and the evaluation used by every model."""
import gzip
import json
from pathlib import Path

import numpy as np
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
RESULTS = ROOT / "results"
SPLITS = ["train", "val", "test", "test_unseen_games"]

TYPES = ["FIX", "KNOWN_ISSUE", "NEW_CONTENT", "IMPROVEMENT", "BALANCE", "GAMEPLAY", "PERFORMANCE", "UI",
         "AUDIO", "GRAPHICS", "LOCALIZATION", "MULTIPLAYER", "CONTROLS", "AI", "MODDING", "ACHIEVEMENT"]
LABELS = ["O"] + [f"{p}-{t}" for t in TYPES for p in ("B", "I")]
L2I = {l: i for i, l in enumerate(LABELS)}

TYPE_DEFINITIONS = {
    "FIX": "a bug fix or crash fix",
    "KNOWN_ISSUE": "a problem that is known but not fixed yet",
    "NEW_CONTENT": "new content or a new feature that was added",
    "IMPROVEMENT": "an improvement or quality-of-life change to something that already exists",
    "BALANCE": "a balance or tuning change to numbers (damage, cost, difficulty, economy)",
    "GAMEPLAY": "a change to gameplay mechanics or combat",
    "PERFORMANCE": "a performance, optimization, loading-time or memory change",
    "UI": "a change to the user interface, menus or HUD",
    "AUDIO": "a change to sound, music or voice",
    "GRAPHICS": "a change to graphics, art, animation or visual effects",
    "LOCALIZATION": "a change to translations, languages or text",
    "MULTIPLAYER": "a change to multiplayer, online play, servers or networking",
    "CONTROLS": "a change to controls, input, controllers or key bindings",
    "AI": "a change to AI, NPC or enemy behaviour",
    "MODDING": "a change for mods, modding tools or the workshop",
    "ACHIEVEMENT": "a change to achievements",
}


def load(split: str, limit: int | None = None):
    path = DATA / f"{split}.jsonl.gz"
    if not path.exists():                      # uncompressed copy from an older run
        path = DATA / f"{split}.jsonl"
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    return rows[:limit] if limit else rows


def spans(tags):
    """(start, end, type) for every B/I span in a tag sequence (end is exclusive)."""
    out, start, typ = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if start is not None:
                out.append((start, i, typ))
            start, typ = (i, t[2:]) if t != "O" else (None, None)
    return out


def evaluate(gold, pred):
    """Span-level scores. A span counts only if both its boundaries and its type are right."""
    assert len(gold) == len(pred)
    for g, p in zip(gold, pred):
        assert len(g) == len(p)
    rep = classification_report(gold, pred, mode="strict", scheme=IOB2, output_dict=True, zero_division=0)
    per_type = {t: {k: round(float(rep[t][k]), 4) for k in ("precision", "recall", "f1-score", "support")}
                for t in TYPES if t in rep}
    # boundary-only F1: is every change item segmented correctly, whatever its type?
    g_b = [[(s, e) for s, e, _ in spans(t)] for t in gold]
    p_b = [[(s, e) for s, e, _ in spans(t)] for t in pred]
    tp = sum(len(set(a) & set(b)) for a, b in zip(g_b, p_b))
    n_g, n_p = sum(map(len, g_b)), sum(map(len, p_b))
    prec, rec = tp / max(n_p, 1), tp / max(n_g, 1)
    tok = np.mean([a == b for g, p in zip(gold, pred) for a, b in zip(g, p)])
    return {
        "span_macro_f1": round(float(rep["macro avg"]["f1-score"]), 4),
        "span_micro_f1": round(float(f1_score(gold, pred, mode="strict", scheme=IOB2, average="micro")), 4),
        "boundary_f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4),
        "token_accuracy": round(float(tok), 4),
        "per_type": per_type,
    }


def save_result(name: str, group: str, scores: dict, extra: dict | None = None):
    RESULTS.mkdir(exist_ok=True)
    out = {"model": name, "group": group, **scores, **(extra or {})}
    (RESULTS / f"{name}.json").write_text(json.dumps(out, indent=1))
    for split in ("test", "test_unseen_games"):
        if split in scores:
            s = scores[split]
            print(f"{name:24s} {split:18s} macro-F1 {s['span_macro_f1']:.3f}  micro-F1 {s['span_micro_f1']:.3f}"
                  f"  boundary-F1 {s['boundary_f1']:.3f}")
