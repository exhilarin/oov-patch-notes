"""Step 1: collect raw patch notes from Steam.

1. The game list comes from SteamSpy (`request=all`, games ordered by estimated owners,
   1,000 per page). SteamSpy allows one such request per minute, so pages are cached.
2. For every game we call the public Steam News API (`ISteamNews/GetNewsForApp/v2`) once,
   keeping only the developer's own announcements (`feeds=steam_community_announcements`).
   One call returns all of a game's announcements; responses are cached per game.

Output: data/raw/patch_notes_raw_NN.jsonl.gz (shards of --shard-size lines, so every file
stays below GitHub's 100 MB limit), one announcement per line with fields
    appid, game, gid, date, title, contents
`contents` is the raw BBCode text as written by the developer. No user data is collected.
"""
import argparse
import gzip
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
SPY_URL = "https://steamspy.com/api.php?request=all&page={page}"
NEWS_URL = ("https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
            "?appid={appid}&count=10000&maxlength=0&feeds=steam_community_announcements")
UA = {"User-Agent": "oov-patch-notes (COE025 course project)"}


def get_json(url: str, retries: int = 4):
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            if isinstance(e, HTTPError) and e.code in (400, 403, 404):
                return None
            time.sleep(5 * (attempt + 1))
    return None


def game_list(pages: int, cache: Path):
    games = {}
    for page in range(pages):
        path = cache / f"steamspy_page{page}.json"
        if not path.exists():
            data = get_json(SPY_URL.format(page=page))
            if not data:
                break
            path.write_text(json.dumps(data))
            if page < pages - 1:
                time.sleep(61)  # SteamSpy: one "all" request per minute
        for appid, g in json.loads(path.read_text()).items():
            games[int(appid)] = g["name"]
    return games


def news_for(appid: int, cache: Path, delay: float):
    path = cache / f"{appid}.json.gz"
    if path.exists():
        try:
            return json.loads(gzip.decompress(path.read_bytes()))
        except (OSError, EOFError, json.JSONDecodeError):
            pass  # truncated by an interrupted run: fetch again
    data = get_json(NEWS_URL.format(appid=appid))
    items = (data or {}).get("appnews", {}).get("newsitems", [])
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(items).encode()))
    tmp.replace(path)
    time.sleep(delay)
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", default=5, type=int, help="SteamSpy pages of 1,000 games each")
    ap.add_argument("--cache", default=ROOT / ".cache", type=Path, help="API response cache (git-ignored)")
    ap.add_argument("--delay", default=0.3, type=float, help="seconds between Steam API calls")
    ap.add_argument("--out-dir", default=ROOT / "data/raw", type=Path)
    ap.add_argument("--shard-size", default=119_400, type=int, help="announcements per output file")
    args = ap.parse_args()

    (args.cache / "news").mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    games = game_list(args.pages, args.cache)
    print(f"{len(games)} games")

    for old in args.out_dir.glob("patch_notes_raw_*.jsonl.gz"):
        old.unlink()
    n_items, f = 0, None
    for i, (appid, name) in enumerate(games.items(), 1):
        for it in news_for(appid, args.cache / "news", args.delay):
            if n_items % args.shard_size == 0:
                if f:
                    f.close()
                f = gzip.open(args.out_dir / f"patch_notes_raw_{n_items // args.shard_size:02d}.jsonl.gz", "wt",
                              encoding="utf-8")
            f.write(json.dumps({"appid": appid, "game": name, "gid": it["gid"], "date": it["date"],
                                "title": it["title"], "contents": it["contents"]}, ensure_ascii=False) + "\n")
            n_items += 1
        if i % 250 == 0:
            print(f"  {i}/{len(games)} games, {n_items} announcements", flush=True)
    if f:
        f.close()
    print(f"wrote {n_items} announcements from {len(games)} games to {args.out_dir}")


if __name__ == "__main__":
    main()
