"""Step 4b: BiLSTM-CRF trained from scratch (no pre-trained weights).

Every word is represented by a learned word embedding (vocabulary built on the training
split, rare words -> <unk>) concatenated with a character-level BiLSTM encoding and a few
casing flags (captures "Fixed", "AI", "+10%"). A 2-layer BiLSTM reads the whole example
and a linear-chain CRF on top scores tag sequences, so invalid transitions such as
O -> I-FIX are learned to be unlikely. Trained with 3 seeds; mean and std are reported.
"""
import argparse
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF

from common import L2I, LABELS, evaluate, load, save_result

PAD, UNK = 0, 1


class Vocab:
    def __init__(self, rows, min_count=2):
        words = Counter(w.lower() for r in rows for w in r["tokens"])
        chars = Counter(c for r in rows for w in r["tokens"] for c in w)
        self.w = {w: i + 2 for i, w in enumerate(w for w, c in words.items() if c >= min_count)}
        self.c = {ch: i + 2 for i, ch in enumerate(ch for ch, c in chars.items() if c >= 5)}

    def words(self, toks):
        return [self.w.get(t.lower(), UNK) for t in toks]

    def chars(self, tok, n=16):
        return [self.c.get(ch, UNK) for ch in tok[:n]]


def casing(t):
    return [t[:1].isupper(), t.isupper(), any(c.isdigit() for c in t), not t[:1].isalnum()]


class BiLSTMCRF(nn.Module):
    def __init__(self, n_words, n_chars, n_tags, wdim=128, cdim=32, hidden=256):
        super().__init__()
        self.wemb = nn.Embedding(n_words, wdim, padding_idx=PAD)
        self.cemb = nn.Embedding(n_chars, cdim, padding_idx=PAD)
        self.clstm = nn.LSTM(cdim, cdim, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(wdim + 2 * cdim + 4, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(2 * hidden, n_tags)
        self.crf = CRF(n_tags, batch_first=True)

    def emissions(self, w, c, f):
        B, T, L = c.shape
        ce = self.cemb(c.view(B * T, L))
        _, (h, _) = self.clstm(ce)
        ch = torch.cat([h[0], h[1]], -1).view(B, T, -1)
        x = self.drop(torch.cat([self.wemb(w), ch, f], -1))
        return self.out(self.drop(self.lstm(x)[0]))


def batchify(rows, vocab, device):
    T = max(len(r["tokens"]) for r in rows)
    w = torch.zeros(len(rows), T, dtype=torch.long)
    c = torch.zeros(len(rows), T, 16, dtype=torch.long)
    f = torch.zeros(len(rows), T, 4)
    y = torch.zeros(len(rows), T, dtype=torch.long)
    m = torch.zeros(len(rows), T, dtype=torch.bool)
    for i, r in enumerate(rows):
        n = len(r["tokens"])
        w[i, :n] = torch.tensor(vocab.words(r["tokens"]))
        for j, t in enumerate(r["tokens"]):
            cc = vocab.chars(t)
            c[i, j, :len(cc)] = torch.tensor(cc)
            f[i, j] = torch.tensor(casing(t), dtype=torch.float)
        y[i, :n] = torch.tensor([L2I[t] for t in r["tags"]])
        m[i, :n] = True
    return w.to(device), c.to(device), f.to(device), y.to(device), m.to(device)


@torch.no_grad()
def predict(model, rows, vocab, device, bs=16):
    model.eval()
    out = []
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["tokens"]))
    pred = {}
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        w, c, f, _, m = batchify([rows[i] for i in idx], vocab, device)
        for i, p in zip(idx, model.crf.decode(model.emissions(w, c, f), mask=m)):
            pred[i] = [LABELS[t] for t in p]
    return [pred[i] for i in range(len(rows))]


def run(seed, data, args, device):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = data["train"][: args.train_size] if args.train_size else data["train"]
    vocab = Vocab(train)
    model = BiLSTMCRF(len(vocab.w) + 2, len(vocab.c) + 2, len(LABELS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, best_state = -1, None
    # length-bucketed batches keep padding (and CRF time) small
    order = sorted(range(len(train)), key=lambda i: len(train[i]["tokens"]))
    batches = [order[k:k + args.batch_size] for k in range(0, len(order), args.batch_size)]
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for b in rng.permutation(len(batches)):
            w, c, f, y, m = batchify([train[i] for i in batches[b]], vocab, device)
            loss = -model.crf(model.emissions(w, c, f), y, mask=m, reduction="mean")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        f1 = evaluate([r["tags"] for r in data["val"]], predict(model, data["val"], vocab, device))["span_macro_f1"]
        print(f"  seed {seed} epoch {ep}: val macro-F1 {f1:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best, best_state = f1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {s: evaluate([r["tags"] for r in data[s]], predict(model, data[s], vocab, device))
            for s in ("val", "test", "test_unseen_games")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {s: load(s) for s in ("train", "val", "test", "test_unseen_games")}

    runs = [run(s, data, args, device) for s in args.seeds]
    scores = {}
    for split in ("val", "test", "test_unseen_games"):
        s = dict(runs[0][split])  # per-type scores of the first seed
        for k in ("span_macro_f1", "span_micro_f1", "boundary_f1", "token_accuracy"):
            vals = [r[split][k] for r in runs]
            s[k], s[k + "_std"] = round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)
        scores[split] = s
    save_result("bilstm_crf", "baseline", scores, {"seeds": args.seeds, "epochs": args.epochs})


if __name__ == "__main__":
    main()
