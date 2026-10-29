# Data pipeline

Everything below is produced by scripts in `src/` and can be re-run with `make data splits`.
Numbers for every step are written to `data/processed/cleaning_report.json` and copied into
[results/benchmark.md](../results/benchmark.md).

## 1. Collection (`src/collect.py`)

| What | How |
|---|---|
| Game list | SteamSpy `request=all`, 5 pages × 1,000 games, ordered by estimated owners (the 5,000 most-owned Steam games). SteamSpy allows one such request per minute, so the pages are cached. |
| Announcements | Public Steam Web API `ISteamNews/GetNewsForApp/v2`, one call per game with `feeds=steam_community_announcements` and `maxlength=0`, i.e. only the developer's own posts, in full. |
| Stored fields | `appid, game, gid, date, title, contents` (raw BBCode). No user or player data. |
| Robustness | Responses are cached per game (`.cache/news`, git-ignored); a truncated cache file is fetched again; failed requests are retried with back-off. |

## 2. Cleaning and labelling (`src/clean.py`)

The label of a change is the **section header the developer put above it**, e.g.

```
[h2]Bug Fixes[/h2]          ->  FIX
[list][*]Fixed a crash when loading saves[/list]
[h2]Audio[/h2]              ->  AUDIO
[list][*]Added new footstep sounds on snow[/list]
```

The header and the list markup are removed, so the model only sees
`Fixed a crash when loading saves Added new footstep sounds on snow` and has to find where
each change starts and which type it has. This is the same idea as our Project 1, where the
label came from the `fix:` / `feat:` prefix of a commit and the prefix was removed.

| Step | Rule | Why |
|---|---|---|
| Find sections | `[h1]`–`[h3]` headers, or a line that is only bold/underlined text | developers use both styles |
| Map headers | `configs/header_map.yaml`: header is lower-cased, version numbers and punctuation removed, then the first matching regular expression decides the type | hundreds of spellings ("Bug fixes", "Bugfixes", "Fixes:", "BUG FIXES v1.2") map to one type |
| Drop generic headers | "General", "Changes", "Patch notes", "Changelog", "Misc", ... | they say nothing about the type |
| Drop promotional headers | giveaways, prizes, sales, streams, events, FAQ, roadmap, "what's next", ... | not patch notes |
| Items | `[*]` list bullets; otherwise only lines that start with `-`, `•`, `*` or `1.` | prose paragraphs ("We hope you enjoy…") are not change items |
| Item length | 2–80 tokens, at least one alphabetic token | longer "items" are paragraphs |
| Text cleaning | `[url=…]text[/url]` → text, other BBCode and HTML removed, images/videos/tables/quotes dropped, links → `<URL>`, bullets removed, whitespace normalised | the model should read the text, not the markup |
| Two or more types | announcements whose kept items have fewer than 2 different types are dropped | with one type the task degenerates to segmentation only |
| Language | `langdetect` (seed 42) on the kept text; English only | one language per benchmark |
| Duplicates | exact duplicates of the kept text (lower-cased) are removed | the same notes are often re-posted (e.g. a beta branch and a live branch) |
| Chunking | announcements longer than 256 tokens are cut at item boundaries; a chunk is kept only if it still has 2+ types | fits the 512-subword window of BERT-style encoders |

Tokenization is a simple regular expression (`<URL>`, words with inner `-`, `'`, `.`, a trailing
`%`, or a single punctuation mark), so `+10%`, `v1.2.3` and `don't` stay one token each.

Each change item becomes one BIO span: `B-<TYPE>` on its first token, `I-<TYPE>` on the rest.
There are no `O` tokens: every kept token belongs to a change.

## 3. Splits (`src/split.py`)

* **test_unseen_games** – all examples of 10% of the games (random, seed 42). No model sees
  these games during training; they measure transfer to a new developer's writing style.
* **train / val / test** – a random 80 / 10 / 10 split of the remaining examples, grouped by
  announcement so the chunks of one announcement never end up in different splits.

## Known limitations

* **Label noise.** The label is chosen per section by the developer, not per item. An item
  such as "Better ambient lighting" under a *New content* header is labelled NEW_CONTENT, although
  its text looks like GRAPHICS. Type and area headers also overlap ("Bug fixes" vs "Audio" for an
  audio bug fix). This puts a ceiling on any model's score and is discussed in
  [results.md](results.md).
* **Missing negatives.** Sections whose header we drop are removed completely, so the input never
  contains text that is not a change item.
* **Popular games only.** The 5,000 most-owned games are over-represented by large studios.
