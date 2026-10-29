"""Step 3: create the evaluation splits.

* test_unseen_games: every example of a held-out set of games (about 10% of the games,
  chosen at random with a fixed seed). No model sees these games during training, so this
  split measures how well a model transfers to a new developer's writing style.
* train / val / test: a random 80/10/10 split of the remaining examples. Chunks of the
  same announcement always stay in the same split, so no announcement leaks across splits.
"""
import argparse
import gzip
import json
import random
from collections import defaultdict

from common import DATA


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", default=DATA / "patch_notes_clean.jsonl.gz")
    ap.add_argument("--unseen-share", default=0.10, type=float)
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    rows = [json.loads(l) for l in gzip.open(args.clean, "rt", encoding="utf-8")]
    rng = random.Random(args.seed)
    games = sorted({r["appid"] for r in rows})
    rng.shuffle(games)
    unseen = set(games[: int(len(games) * args.unseen_share)])

    by_ann = defaultdict(list)
    for r in rows:
        if r["appid"] not in unseen:
            by_ann[r["id"].rsplit("-", 1)[0]].append(r)
    anns = sorted(by_ann)
    rng.shuffle(anns)
    n = len(anns)
    cut = {"train": anns[: int(0.8 * n)], "val": anns[int(0.8 * n): int(0.9 * n)], "test": anns[int(0.9 * n):]}

    out = {s: [r for a in ids for r in by_ann[a]] for s, ids in cut.items()}
    out["test_unseen_games"] = [r for r in rows if r["appid"] in unseen]
    for split, rs in out.items():
        with gzip.open(DATA / f"{split}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split:18s} {len(rs):7,} examples  {len({r['appid'] for r in rs}):5,} games  "
              f"{sum(len(r['tokens']) for r in rs):10,} tokens")


if __name__ == "__main__":
    main()
