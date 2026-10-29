"""Step 4c: fine-tune a pretrained transformer encoder for token classification.

The word-level BIO tags are aligned to sub-word tokens: the first sub-word of every word
carries the word's tag, the other sub-words are ignored in the loss (-100). At prediction
time the tag of the first sub-word is used for the whole word. Examples longer than the
model's window are split into overlapping windows of words.

Models (see --model):
  bert        bert-base-cased             baseline (vanilla BERT from the lectures)
  deberta     microsoft/deberta-v3-base   new: disentangled attention, ELECTRA-style pre-training
  modernbert  answerdotai/ModernBERT-base new: 2024 encoder, rotary positions, 8k context
"""
import argparse
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import L2I, LABELS, evaluate, load, save_result

MODELS = {
    "bert": ("bert-base-cased", "baseline"),
    "deberta": ("microsoft/deberta-v3-base", "new"),
    "modernbert": ("answerdotai/ModernBERT-base", "new"),
}


def windows(tokens, size, stride):
    if len(tokens) <= size:
        return [(0, len(tokens))]
    out, start = [], 0
    while True:
        out.append((start, min(start + size, len(tokens))))
        if start + size >= len(tokens):
            return out
        start += stride


def encode(rows, tok, max_len, words_per_window):
    feats = []
    for ri, r in enumerate(rows):
        for a, b in windows(r["tokens"], words_per_window, words_per_window // 2):
            enc = tok(r["tokens"][a:b], is_split_into_words=True, truncation=True, max_length=max_len)
            wid = enc.word_ids()
            labels, prev = [], None
            for w in wid:
                labels.append(-100 if w is None or w == prev else L2I[r["tags"][a + w]])
                prev = w
            feats.append({"input_ids": enc["input_ids"], "labels": labels, "row": ri, "start": a, "word_ids": wid})
    return feats


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id)
    lab = torch.full((len(batch), n), -100)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        lab[i, :k] = torch.tensor(b["labels"])
        att[i, :k] = 1
    return ids, att, lab


@torch.no_grad()
def predict(model, rows, feats, pad_id, bs, device):
    """Word-level tags; for overlapping windows the window where the word is most central wins."""
    model.eval()
    best = [[(None, -1.0)] * len(r["tokens"]) for r in rows]
    for i in range(0, len(feats), bs):
        chunk = feats[i:i + bs]
        ids, att, _ = collate(chunk, pad_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        pred = logits.argmax(-1).cpu().numpy()
        for f, p in zip(chunk, pred):
            seen, n_words = set(), max((w for w in f["word_ids"] if w is not None), default=0) + 1
            for pos, w in enumerate(f["word_ids"]):
                if w is None or w in seen:
                    continue
                seen.add(w)
                centrality = min(w + 1, n_words - w)
                j = f["start"] + w
                if centrality > best[f["row"]][j][1]:
                    best[f["row"]][j] = (LABELS[p[pos]], centrality)
    return [[t or "O" for t, _ in r] for r in best]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps (4 GB GPU)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--words", type=int, default=256, help="words per window")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (saves GPU memory)")
    ap.add_argument("--freeze-embeddings", action="store_true", help="do not train the word embeddings")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name, group = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True) if "roberta" in name else AutoTokenizer.from_pretrained(name)
    # fp32 master weights (some checkpoints are stored in fp16); autocast handles the fp16 compute
    model = AutoModelForTokenClassification.from_pretrained(
        name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=L2I, dtype=torch.float32).to(device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    if args.freeze_embeddings:
        # DeBERTa-v3 has a 128k vocabulary (98M embedding parameters); their gradients and Adam
        # states do not fit next to the rest of the model on a 4 GB GPU
        model.get_input_embeddings().weight.requires_grad_(False)
    pad = tok.pad_token_id

    data = {s: load(s) for s in ("train", "val", "test", "test_unseen_games")}
    train_rows = data["train"][: args.train_size] if args.train_size else data["train"]
    train = encode(train_rows, tok, args.max_len, args.words)
    enc = {s: encode(data[s], tok, args.max_len, args.words) for s in ("val", "test", "test_unseen_games")}
    print(f"{name}: {len(train_rows)} train examples -> {len(train)} windows on {device}")

    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate(b, pad))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01,
                            foreach=False)
    steps = args.epochs * math.ceil(len(train) / (args.batch_size * args.accum))
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best_f1, best_state, history, t0 = -1, None, [], time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        for step, (ids, att, lab) in enumerate(loader, 1):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device)).loss
            scaler.scale(loss / args.accum).backward()
            if step % args.accum == 0 or step == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
        val = evaluate([r["tags"] for r in data["val"]], predict(model, data["val"], enc["val"], pad, 32, device))
        history.append({"epoch": ep, "val_macro_f1": val["span_macro_f1"], "minutes": round((time.time() - t0) / 60, 1)})
        print(f"  epoch {ep}: val macro-F1 {val['span_macro_f1']:.3f}  ({history[-1]['minutes']} min)", flush=True)
        if val["span_macro_f1"] > best_f1:
            best_f1 = val["span_macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    scores = {s: evaluate([r["tags"] for r in data[s]], predict(model, data[s], enc[s], pad, 32, device))
              for s in ("val", "test", "test_unseen_games")}
    save_result(args.model, group, scores, {"checkpoint": name, "train_examples": len(train_rows),
                                            "epochs": args.epochs, "lr": args.lr, "history": history,
                                            "batch_size": args.batch_size, "accum": args.accum,
                                            "grad_ckpt": args.grad_ckpt, "freeze_embeddings": args.freeze_embeddings,
                                            "train_minutes": round((time.time() - t0) / 60, 1)})


if __name__ == "__main__":
    main()
