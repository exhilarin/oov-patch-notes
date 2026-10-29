"""Step 4d, new method: two-stage segmentation + zero-shot typing with Laya.

Laya (convaiinnovations/laya, ModernBERT-large, 421M parameters, Apache-2.0) is an
open-source alternative to JEV-style decision models. It is NOT trained on our data: for
each change item it only receives the item text and the 16 type names with their
one-line definitions (TYPE_DEFINITIONS in common.py) and picks one type.

Stage 1 finds the change items:
  * crf    - spans predicted by the CRF baseline (models/crf.pkl, from train_classical.py)
  * oracle - the gold item boundaries; this isolates how well the types alone are recognised
Stage 2 gives every span the type chosen by Laya.
Inference is slow, so by default a random sample of --n examples per test split is used
(the sample size is stored in the result file).
"""
import argparse
import pickle
import random

from common import ROOT, TYPE_DEFINITIONS, TYPES, evaluate, load, save_result, spans
from train_classical import featurize

QUESTION = {"change_type": {
    "type": "choice",
    "instructions": "What kind of change does this line from a video game's patch notes describe?",
    "criteria": TYPE_DEFINITIONS,
}}


def tag_spans(n, sp):
    tags = ["O"] * n
    for a, b, t in sp:
        tags[a:b] = [f"B-{t}"] + [f"I-{t}"] * (b - a - 1)
    return tags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300, help="examples sampled per test split (0 = all)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from laya import Router  # imported here so the other scripts do not need laya

    router = Router(device="cuda")
    crf = pickle.loads((ROOT / "models/crf.pkl").read_bytes())
    for stage1 in ("crf", "oracle"):
        scores, sizes = {}, {}
        for split in ("test", "test_unseen_games"):
            rows = load(split)
            if args.n and args.n < len(rows):
                rows = random.Random(args.seed).sample(rows, args.n)
            if stage1 == "crf":
                seg = [spans(t) for t in crf.predict(featurize(rows))]
            else:
                seg = [spans(r["tags"]) for r in rows]
            reqs, where = [], []
            for i, (r, sp) in enumerate(zip(rows, seg)):
                for a, b, _ in sp:
                    reqs.append({"state": " ".join(r["tokens"][a:b]), "questions": QUESTION, "model": "english"})
                    where.append((i, a, b))
            out = router.predict_batch(reqs, batch_size=args.batch)
            typed = [[] for _ in rows]
            for (i, a, b), o in zip(where, out):
                t = o["answers"]["change_type"]["choice"]
                typed[i].append((a, b, t if t in TYPES else "FIX"))
            pred = [tag_spans(len(r["tokens"]), sp) for r, sp in zip(rows, typed)]
            scores[split], sizes[split] = evaluate([r["tags"] for r in rows], pred), len(rows)
        save_result(f"laya_zero_shot_{stage1}_spans", "new", scores,
                    {"checkpoint": "convaiinnovations/laya", "training_examples": 0, "stage1": stage1,
                     "eval_sample_size": sizes, "type_definitions": TYPE_DEFINITIONS})


if __name__ == "__main__":
    main()
