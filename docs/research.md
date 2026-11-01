# Research notes: problem definition and design decisions

## 1. The problem

**Task:** given the plain text of a video game's patch notes, find **where every change starts
and ends** and **what type of change it is**.

```
input   Fixed a crash when loading saves . Reduced memory usage in Act 3 . Added Turkish subtitles .
        Rogue damage reduced by 10% .

output  [Fixed a crash when loading saves .]         FIX
        [Reduced memory usage in Act 3 .]            PERFORMANCE
        [Added Turkish subtitles .]                  LOCALIZATION
        [Rogue damage reduced by 10% .]              BALANCE
```

Every change item is one span in BIO notation (`B-FIX I-FIX I-FIX ...`). There are 16 types, so
33 tags. It is a **token classification** task that combines **segmentation** (where are the
boundaries) and **typing** (which of 16 classes), like discourse-element segmentation of essays.

### Why it is useful

* **Players** want to know what changed for them: only the balance changes, only the fixes for the
  bug they reported. Long patch notes (up to 95 items in a single 256-token example) can be filtered by type.
* **Stores, launchers, mod managers and Discord bots** receive patch notes as unformatted text
  (copied into a tweet, a Discord message or an e-mail) and could restore the structure.
* **Developers and researchers** can measure how much of each update is fixing, balancing or new
  content across thousands of games.

### Why it is not "just NER"

* The spans are **whole clauses or sentences** (median 12 tokens), not names.
* The same words appear in different types: *"Fixed the reload sound"* can be a FIX or an AUDIO
  item, depending on where the developer put it. The type depends on the **document context**
  (neighbouring items of the same section share a type), not on a single word.
* Boundaries have to be found without the line breaks and bullets: many items contain several
  sentences, and many sentences do not end with a period.

### Why it is a new task

We found **no prior work** that segments and types game patch notes at the token level and no
public span-labelled patch-note corpus (see §6). The course covered POS tagging and NER; this
task is outside both.

## 2. Why this dataset

* **We collected it ourselves** from the public Steam Web API: 477,597 announcements of the 5,000
  most-owned Steam games. Nobody else has this data and the whole pipeline is scripted
  (`src/collect.py`, `src/clean.py`).
* **The labels are real and come from the authors themselves.** Developers group their notes under
  headers such as *Bug Fixes*, *Balance*, *Audio*. We map the headers to types and **remove them
  from the input**, so the model never sees the label. This is the same idea as our Project 1,
  where the label came from the `fix:` prefix of a commit. No synthetic text and no LLM labels.
* **It is large and messy:** hundreds of header spellings, BBCode and HTML, promotional posts,
  non-English posts and re-posted duplicates had to be handled. After cleaning: 39,481 examples,
  532,761 change items, 7.7 million tokens from 1,902 games.

## 3. Label set

| Type | Meaning | Example headers that map to it |
|---|---|---|
| FIX | a bug fix or crash fix | Bug Fixes, Fixes, Bugfixes, Crash Fixes, Resolved Issues |
| KNOWN_ISSUE | a known problem, not fixed yet | Known Issues, Known Bugs |
| NEW_CONTENT | new content or a new feature | New Content, New Features, Features, Additions |
| IMPROVEMENT | an improvement or quality-of-life change | Improvements, Quality of Life, QoL, Polish |
| BALANCE | numbers: damage, cost, difficulty, economy | Balance, Balance Changes, Balancing, Tuning |
| GAMEPLAY | mechanics and combat | Gameplay, Mechanics, Combat |
| PERFORMANCE | performance, optimisation, memory, loading | Performance, Optimizations |
| UI | user interface, menus, HUD | UI, User Interface, HUD, Menus |
| AUDIO | sound, music, voice | Audio, Sound, Music |
| GRAPHICS | graphics, art, animation, VFX | Graphics, Visuals, Art, Animations |
| LOCALIZATION | translations, languages, text | Localization, Translations |
| MULTIPLAYER | online play, servers, networking | Multiplayer, Networking, Co-op, Servers |
| CONTROLS | input, controllers, key bindings | Controls, Controller, Input, Steam Deck |
| AI | AI, NPC and enemy behaviour | AI, NPCs |
| MODDING | mods, modding tools, workshop | Modding, Mods, Workshop |
| ACHIEVEMENT | achievements | Achievements |

The full mapping is `configs/header_map.yaml`; rules are applied in order and the first match wins.
Generic headers (*General*, *Changes*, *Patch notes*) and promotional headers (*Prizes*, *How to
enter*, *Sale*) are dropped. The classes are heavily imbalanced (FIX is 83 times more frequent
than ACHIEVEMENT), which drives the choice of metric.

## 4. Metric

**Primary: span-level macro-F1** (seqeval, strict IOB2 mode).

* *Span-level, strict:* a predicted change counts only if its first token, its last token and its
  type are all correct. Token accuracy would reward a model that gets the long middle part of an
  item right but its boundaries wrong.
* *Macro:* every type weighs the same. With FIX dominating, micro-F1 mostly measures FIX; macro-F1
  shows whether rare types such as CONTROLS or ACHIEVEMENT are learned at all.
* We also report **micro-F1** and **boundary-F1** (segmentation only, type ignored), which tells us
  whether a model fails at finding the items or at typing them.

**Two test sets:** a random test split, and **190 games that no model sees during training**
(`test_unseen_games`). Every developer writes notes in their own style and vocabulary, so the
second set measures whether the models learn the task or the games.

## 5. Methods

The lecture's methods in their vanilla form are the baselines; we distinguish the architecture
from the specific model.

| Group | Architecture | Model | Notes |
|---|---|---|---|
| baseline | rules | keyword rules | sentence-start segmentation + keyword lists per type |
| baseline | linear-chain CRF | sklearn-crfsuite | hand-crafted token features, L1/L2 tuned on validation |
| baseline | BiLSTM-CRF | own PyTorch model | word + character BiLSTM, CRF layer, trained from scratch, 3 seeds |
| baseline | BERT | bert-base-cased (110M) | fine-tuned, first-sub-word labelling |
| new | DeBERTa-v3 | microsoft/deberta-v3-base (184M) | disentangled attention, ELECTRA-style pre-training |
| new | ModernBERT | answerdotai/ModernBERT-base (149M) | 2024 encoder, rotary positions, trained on text and code |
| new | two-stage, zero-shot | CRF spans + Laya (convaiinnovations/laya) | no training data for the typing step: only the 16 type definitions |

For the zero-shot method we also evaluate Laya with the **gold** spans, which isolates the typing
step and shows how much of the error comes from segmentation.

## 6. Related work

Video game patch notes have been studied mainly in software engineering and game analytics, not as
an NLP sequence-labelling task. Truelove et al. (2021) manually analysed 12,122 bug fixes from 723
update notes of 30 popular Steam games and built a taxonomy of game bugs. Claypool et al. (2017)
collected more than 7,700 changes from over 160 League of Legends patches, classified them with
their own taxonomy and related them to win rates. Both treat patch notes as data to be categorised
by hand and cover tens of games. In software text mining, ARENA (Moreno et al., 2017) generates
release notes from code changes and issues, Bi et al. (2022) study what information 32,425 GitHub
release notes contain, and Levin and Yehudai (2017) classify commits into maintenance activities,
the setting of our Project 1. These works classify whole commits or documents, or generate notes;
none segments a release note into typed change items.

Methodologically the task is closest to discourse and argument segmentation as BIO token
classification: Eger et al. (2017) tag argument components in essays with BIO tags combined with
their types, and the PERSUADE corpus (Crossley et al., 2022) annotates discourse elements as spans
in student essays. Our labels come from section headers, a form of distant supervision from
document structure; SECTOR / WikiSection (Arnold et al., 2019) similarly use Wikipedia section
headings as labels for joint segmentation and classification. The models follow standard
sequence-labelling practice: linear-chain CRF (Lafferty et al., 2001), BiLSTM-CRF (Lample et al.,
2016) and fine-tuned encoders (Devlin et al., 2019; He et al., 2023; Warner et al., 2025),
evaluated with seqeval (Nakayama, 2018).

## References

- Arnold, S., Schneider, R., Cudré-Mauroux, P., Gers, F. A., and Löser, A. (2019). SECTOR: A Neural Model for Coherent Topic Segmentation and Classification. *TACL*, 7, 169–184. https://aclanthology.org/Q19-1011/
- Bi, T., Xia, X., Lo, D., Grundy, J., and Zimmermann, T. (2022). An Empirical Study of Release Note Production and Usage in Practice. *IEEE TSE*, 48(6), 1834–1852. https://doi.org/10.1109/TSE.2020.3038881
- Claypool, M., Kica, A., La Manna, A., O'Donnell, L., and Paolillo, T. (2017). On the Impact of Software Patching on Gameplay for the League of Legends Computer Game. *The Computer Games Journal*, 6, 33–61. https://doi.org/10.1007/s40869-017-0032-9
- Crossley, S. A., Baffour, P., Tian, Y., Picou, A., Benner, M., and Boser, U. (2022). The persuasive essays for rating, selecting, and understanding argumentative and discourse elements (PERSUADE) corpus 1.0. *Assessing Writing*, 54, 100667. https://doi.org/10.1016/j.asw.2022.100667
- Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*, 4171–4186. https://aclanthology.org/N19-1423/
- Eger, S., Daxenberger, J., and Gurevych, I. (2017). Neural End-to-End Learning for Computational Argumentation Mining. *ACL 2017*, 11–22. https://aclanthology.org/P17-1002/
- He, P., Gao, J., and Chen, W. (2023). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. *ICLR 2023*. https://arxiv.org/abs/2111.09543
- Lafferty, J., McCallum, A., and Pereira, F. (2001). Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data. *ICML 2001*, 282–289. https://works.bepress.com/andrew_mccallum/4
- Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., and Dyer, C. (2016). Neural Architectures for Named Entity Recognition. *NAACL-HLT 2016*, 260–270. https://aclanthology.org/N16-1030/
- Levin, S., and Yehudai, A. (2017). Boosting Automatic Commit Classification Into Maintenance Activities By Utilizing Source Code Changes. *PROMISE 2017*, 97–106. https://doi.org/10.1145/3127005.3127016
- Moreno, L., Bavota, G., Di Penta, M., Oliveto, R., Marcus, A., and Canfora, G. (2017). ARENA: An Approach for the Automated Generation of Release Notes. *IEEE TSE*, 43(2), 106–127. https://doi.org/10.1109/TSE.2016.2591536
- Nakayama, H. (2018). seqeval: A Python framework for sequence labeling evaluation. https://github.com/chakki-works/seqeval
- Truelove, A., Santana de Almeida, E., and Ahmed, I. (2021). We'll Fix It in Post: What Do Bug Fixes in Video Game Update Notes Tell Us? *ICSE 2021*, 736–747. https://doi.org/10.1109/ICSE43902.2021.00073
- Warner, B., et al. (2025). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference. *ACL 2025*, 2526–2547. https://aclanthology.org/2025.acl-long.127/
- Laya model card: convaiinnovations/laya on Hugging Face. https://huggingface.co/convaiinnovations/laya
