[NURI_TAYFUN_AKBENIZ_210911043_akbenizt.md](https://github.com/user-attachments/files/34235950/NURI_TAYFUN_AKBENIZ_210911043_akbenizt.md)
# Nuri Tayfun Akbeniz (210911043, @akbenizt)

**Role:** concept finder: problem definition, label set, metric and method choices, related work.

## What I did

* **Task and motivation.** Defined the task as joint segmentation and typing of patch-note change
  items (token classification with BIO spans) and argued why it is not plain NER: the spans are
  clauses, and the type depends on the document context, not on single words.
* **Data source.** Proposed Steam patch notes because developers already group their notes under
  section headers, so labels exist at scale without synthetic text or LLM annotation, the same
  idea as the commit prefixes we used in Project 1.
* **Label set.** Grouped the most frequent developer headers into 16 change types, wrote a one-line
  definition for every type (`TYPE_DEFINITIONS` in `src/common.py`, also used by the zero-shot
  method), and listed which headers must be dropped (generic and promotional ones).
* **Metric.** Chose strict span-level macro-F1 because FIX is 83 times more frequent than
  ACHIEVEMENT and token accuracy rewards partly correct items; added boundary-F1 to separate
  segmentation errors from typing errors, and the unseen-games test set to check transfer to new
  developers.
* **Methods.** Selected the baselines from the lectures (CRF, BiLSTM-CRF, BERT) and the new methods
  (DeBERTa-v3, ModernBERT, and the two-stage CRF + Laya zero-shot method with a gold-span upper bound).
* **Related work.** Collected and checked the references in `docs/research.md` (game update notes,
  release notes, commit classification, argument/discourse segmentation, header-based distant
  supervision, model papers).

## Files

* `docs/research.md`
