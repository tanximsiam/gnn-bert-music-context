"""One-off MusicCaps data acquisition (Task 4, and Task 1's caption->tag proxy
variant): fetch the official caption/aspect-list metadata from HuggingFace
(`google/MusicCaps`) and download+trim each 10s audio clip via yt-dlp, since
Google only publishes YouTube (video_id, start_s, end_s) references, not audio.

Not part of the train/evaluate pipeline — this is a local, one-time data-prep
utility. Downloaded audio is for local academic research use only (never
committed to git; data/raw/ is gitignored) and some fraction of clips will
fail (deleted/private/region-blocked videos) — this is expected and logged,
not treated as an error.

Usage:
    python -m src.musiccaps_prep --fetch-metadata --download-audio
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def fetch_musiccaps_metadata(dest_csv: str | Path) -> pd.DataFrame:
    """Download the official musiccaps-public.csv (5521 rows: ytid, start_s,
    end_s, aspect_list, caption, ...) from the google/MusicCaps HF dataset repo."""
    from huggingface_hub import hf_hub_download

    dest_csv = Path(dest_csv)
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    src_path = hf_hub_download("google/MusicCaps", "musiccaps-public.csv", repo_type="dataset")
    shutil.copy(src_path, dest_csv)
    return pd.read_csv(dest_csv)


def _download_one(ytid: str, start_s: int, end_s: int, out_dir: Path, sample_rate: int, cookies_path: str | None = None) -> tuple[str, bool, str]:
    out_path = out_dir / f"{ytid}.wav"
    if out_path.exists():
        return ytid, True, "cached"
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={ytid}",
        "--download-sections", f"*{start_s}-{end_s}",
        "--force-keyframes-at-cuts",
        "-f", "best",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", f"ffmpeg:-ar {sample_rate} -ac 1",
        "-o", str(out_dir / f"{ytid}.%(ext)s"),
        "--no-playlist", "--quiet", "--no-warnings",
    ]
    if cookies_path:
        # authenticated web client: rides on a real logged-in session, which
        # YouTube trusts more than the anonymous android client under heavy
        # IP-level throttling (the android bypass alone stops working past a
        # few thousand requests/hour from one IP).
        cmd += ["--cookies", cookies_path]
    else:
        # android client bypasses YouTube's "sign in to confirm you're not a
        # bot" cookie requirement entirely; it only ever offers muxed (not
        # audio-only) formats, hence -f best instead of bestaudio.
        cmd += ["--extractor-args", "youtube:player_client=android"]
    last_msg = ""
    for attempt in range(3):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and out_path.exists():
                return ytid, True, "ok"
            last_msg = (result.stderr or result.stdout).strip()[-300:]
        except Exception as e:  # network hiccups, ffmpeg errors, etc.
            last_msg = str(e)
        # "The page needs to be reloaded" is a transient session glitch under
        # concurrent cookie use, not a dead video -> worth a couple retries.
        if "page needs to be reloaded" not in last_msg.lower():
            break
        time.sleep(2 * (attempt + 1))
    return ytid, False, last_msg


def download_musiccaps_audio(
    csv_path: str | Path,
    out_dir: str | Path,
    sample_rate: int = 22050,
    num_workers: int = 8,
    log_path: str | Path | None = None,
    cookies_path: str | None = None,
) -> dict[str, list[str]]:
    """Download+trim every MusicCaps clip to out_dir/<ytid>.wav. Skips clips
    already downloaded (resumable). Prints each result as it completes (for
    real-time viewing in a foreground terminal). Returns {"ok": [...], "failed": [...]}."""
    df = pd.read_csv(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list] = {"ok": [], "failed": []}
    failure_reasons: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(_download_one, row.ytid, int(row.start_s), int(row.end_s), out_dir, sample_rate, cookies_path): row.ytid
            for row in df.itertuples()
        }
        for i, future in enumerate(as_completed(futures)):
            ytid, ok, msg = future.result()
            (results["ok"] if ok else results["failed"]).append(ytid)
            status = "OK" if ok else f"FAIL ({msg[:80]})"
            print(f"[{i + 1}/{len(futures)}] {ytid}: {status}  (running totals: ok={len(results['ok'])} failed={len(results['failed'])})", flush=True)
            if not ok:
                failure_reasons[ytid] = msg

    if log_path:
        with open(log_path, "w") as f:
            json.dump({"failed_reasons": failure_reasons, "summary": {k: len(v) for k, v in results.items()}}, f, indent=2)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch MusicCaps metadata and/or download its audio clips.")
    parser.add_argument("--fetch-metadata", action="store_true")
    parser.add_argument("--download-audio", action="store_true")
    parser.add_argument("--csv", type=str, default="data/raw/musiccaps/musiccaps.csv")
    parser.add_argument("--audio-dir", type=str, default="data/raw/musiccaps/audio")
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cookies", type=str, default=None, help="path to a cookies.txt exported from a logged-in browser session, to bypass YouTube bot-detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fetch_metadata:
        df = fetch_musiccaps_metadata(args.csv)
        print(f"fetched {len(df)} rows -> {args.csv}")
    if args.download_audio:
        results = download_musiccaps_audio(
            args.csv, args.audio_dir, sample_rate=args.sample_rate, num_workers=args.workers,
            log_path=Path(args.audio_dir).parent / "download_log.json", cookies_path=args.cookies,
        )
        print(f"done: ok={len(results['ok'])} failed={len(results['failed'])}")


if __name__ == "__main__":
    main()
