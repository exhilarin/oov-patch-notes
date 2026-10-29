# İlyas Güney (2309111030, @exhilarin)

**Role:** implementation: data collection and cleaning pipeline, all models, benchmark and repository structure.

## What I did

### Data collection (`src/collect.py`)
* Game list from SteamSpy (the 5,000 most-owned Steam games, cached because SteamSpy allows one
  request per minute) and one call per game to the public Steam News API, keeping only the
  developers' own announcements in full (`feeds=steam_community_announcements`, `maxlength=0`).
* Per-game response cache with atomic writes and re-fetching of truncated files, retries with
  back-off, and output shards below GitHub's 100 MB limit. Result: 477,597 raw announcements.

### Cleaning and labelling (`src/clean.py`, `configs/header_map.yaml`)
* Section detection for both header styles (`[h1]`–`[h3]` and bold-only lines) and the header →
  type mapping: normalisation (case, version numbers, punctuation), ordered regular expressions for
  16 change types, and skip lists for generic and promotional headers.
* Item extraction from list bullets only (prose paragraphs are not change items), BBCode/HTML/URL
  cleaning, tokenizer, BIO conversion with the header removed from the input (no label leakage),
  English filter, de-duplication and chunking at item boundaries.
* `cleaning_report.json`, which records the size after every step and how every frequent header was
  mapped. Result: 39,481 examples, 532,761 change items, 7.7M tokens.

### Splits (`src/split.py`)
* Held-out test set of 190 games never seen in training, and an 80/10/10 random split grouped by
  announcement so chunks of one announcement never cross splits.

### Models
* `common.py`: label set, type definitions, span extraction and one evaluation function for every
  model (strict span macro/micro-F1 with seqeval, boundary-F1, token accuracy, per-type scores).
* `train_classical.py`: keyword-rule baseline and a CRF with hand-crafted token features,
  regularisation tuned on validation.
* `train_bilstm_crf.py`: word + character BiLSTM with a CRF layer, from scratch, length-bucketed
  batches, three seeds with mean/std.
* `train_transformer.py`: BERT, DeBERTa-v3 and ModernBERT fine-tuning with a plain PyTorch loop on a
  4 GB GPU (fp16, gradient accumulation, first-sub-word labelling, overlapping windows with
  centre-of-window voting at prediction time, best epoch on validation).
* `zero_shot_laya.py`: the two-stage method (CRF spans + zero-shot Laya typing) and its gold-span
  upper bound.

### Benchmark and reproducibility
* `benchmark.py` (results table, per-type table, figures), `Makefile` (`make all` runs the full
  pipeline), pinned `requirements.txt`, fixed seeds, README, `docs/preprocessing.md` and `LICENSE`
  with a data notice.
