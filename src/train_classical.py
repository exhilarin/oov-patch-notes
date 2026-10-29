"""Step 4a: classical baselines.

* Keyword rules: a new change starts after sentence-final punctuation when the next word is
  capitalized; its type is chosen by keyword lists (fixed/crash -> FIX, added/new ->
  NEW_CONTENT, ...), FIX when nothing matches (the most frequent type).
* CRF: a linear-chain conditional random field (sklearn-crfsuite) on hand-crafted token
  features (word, lower-case form, shape, affixes, neighbouring words, position after
  punctuation). L1/L2 regularisation is tuned on the validation split.
"""
import argparse
import pickle
import re
import time

import sklearn_crfsuite

from common import ROOT, evaluate, load, save_result

KEYWORDS = [
    ("KNOWN_ISSUE", r"\b(known issue|investigating|we are aware|currently)\b"),
    ("FIX", r"\b(fix|fixed|fixes|bug|crash|crashes|issue|resolved|no longer|prevent)\b"),
    ("PERFORMANCE", r"\b(performance|optimi[sz]\w*|fps|memory|loading|faster|stutter)\b"),
    ("BALANCE", r"\b(increased|decreased|reduced|damage|cost|cooldown|health|nerf|buff|\d+%)\b"),
    ("AUDIO", r"\b(sound|sounds|audio|music|sfx|voice)\b"),
    ("GRAPHICS", r"\b(texture|textures|animation|animations|lighting|vfx|visual|model|models|shader)\b"),
    ("UI", r"\b(ui|menu|menus|hud|button|tooltip|screen|interface)\b"),
    ("LOCALIZATION", r"\b(translation|translations|localization|language|text|typo)\b"),
    ("MULTIPLAYER", r"\b(multiplayer|server|servers|online|lobby|co-op|host|client)\b"),
    ("CONTROLS", r"\b(controller|controls|keybind\w*|input|gamepad|mouse|keyboard)\b"),
    ("AI", r"\b(ai|npc|npcs|enemies|pathfinding)\b"),
    ("MODDING", r"\b(mod|mods|modding|workshop|modders)\b"),
    ("ACHIEVEMENT", r"\b(achievement|achievements)\b"),
    ("IMPROVEMENT", r"\b(improved|improve|better|now|updated)\b"),
    ("NEW_CONTENT", r"\b(added|new|introduced|add)\b"),
]
KEYWORDS = [(t, re.compile(p, re.I)) for t, p in KEYWORDS]


def rule_tags(tokens):
    starts = [0] + [i for i in range(1, len(tokens))
                    if tokens[i - 1] in ".!?;" and tokens[i][:1].isupper()]
    tags = []
    for a, b in zip(starts, starts[1:] + [len(tokens)]):
        text = " ".join(tokens[a:b])
        typ = next((t for t, p in KEYWORDS if p.search(text)), "FIX")
        tags += [f"B-{typ}"] + [f"I-{typ}"] * (b - a - 1)
    return tags


def shape(w):
    s = re.sub(r"[A-Z]", "X", w)
    s = re.sub(r"[a-z]", "x", s)
    return re.sub(r"\d", "d", re.sub(r"(.)\1+", r"\1\1", s))


def token_features(tokens, i):
    w = tokens[i]
    f = {"bias": 1.0, "w": w.lower(), "shape": shape(w), "pre3": w[:3].lower(), "suf3": w[-3:].lower(),
         "title": w[:1].isupper(), "upper": w.isupper(), "digit": any(c.isdigit() for c in w),
         "first": i == 0}
    for d in (-2, -1, 1, 2):
        j = i + d
        if 0 <= j < len(tokens):
            f[f"{d}:w"] = tokens[j].lower()
            f[f"{d}:title"] = tokens[j][:1].isupper()
    f["after_punct"] = i > 0 and tokens[i - 1] in ".!?;"
    f["after_punct_title"] = f["after_punct"] and f["title"]
    return f


def featurize(rows):
    return [[token_features(r["tokens"], i) for i in range(len(r["tokens"]))] for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-size", default=8000, type=int, help="CRF training examples (CPU time)")
    args = ap.parse_args()
    data = {s: load(s) for s in ("train", "val", "test", "test_unseen_games")}

    # keyword rules
    scores = {s: evaluate([r["tags"] for r in data[s]], [rule_tags(r["tokens"]) for r in data[s]])
              for s in ("val", "test", "test_unseen_games")}
    save_result("keyword_rules", "baseline", scores)

    # CRF
    train = data["train"][: args.train_size]
    Xtr, ytr = featurize(train), [r["tags"] for r in train]
    Xva, yva = featurize(data["val"]), [r["tags"] for r in data["val"]]
    best = None
    for c1, c2 in [(0.05, 0.01), (0.1, 0.05), (0.2, 0.1)]:
        t0 = time.time()
        crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=150, all_possible_transitions=True)
        crf.fit(Xtr, ytr)
        f1 = evaluate(yva, crf.predict(Xva))["span_macro_f1"]
        print(f"  CRF c1={c1} c2={c2}: val macro-F1 {f1:.3f} ({time.time() - t0:.0f}s)")
        if best is None or f1 > best[0]:
            best = (f1, crf, {"c1": c1, "c2": c2})
    _, crf, hp = best
    scores = {s: evaluate([r["tags"] for r in data[s]], crf.predict(featurize(data[s])))
              for s in ("val", "test", "test_unseen_games")}
    save_result("crf", "baseline", scores, {"hyperparameters": hp, "train_examples": len(train)})
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models/crf.pkl").write_bytes(pickle.dumps(crf))  # reused by zero_shot_laya.py


if __name__ == "__main__":
    main()
