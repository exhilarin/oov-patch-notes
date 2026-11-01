# Eren Taşkın (2309111067, @erentaskin00)

**Role:** presenter: results interpretation, error analysis and the presentation.

## What I did

* **Results interpretation.** Read the benchmark (`results/benchmark.md`) and wrote
  `docs/results.md`: which model is best on the random split and on unseen games, how large the
  generalisation gap is, and what the boundary-F1 says about where models fail.
* **Error analysis.** Analysed the per-type scores and the main error sources: boundary errors vs
  type errors (transformers type better, the CRF-based model segments better), overlapping types
  (IMPROVEMENT vs NEW_CONTENT, GAMEPLAY / UI / GRAPHICS predicted as FIX), label noise from headers
  that name a game area instead of a kind of change ("Fixed ..." items under *UI* or *Graphics*),
  rare types (CONTROLS, ACHIEVEMENT), and the gap between Laya with predicted and with gold spans.
* **Presentation.** Prepared and rehearsed the talk (under two minutes, without reading), the
  speaker notes and the answers to likely questions.

## Files

* `docs/results.md`
* presentation `OOV_Group_NLP_Project2.pptx` (submitted on Blackboard)
