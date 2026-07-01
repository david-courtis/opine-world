#!/usr/bin/env python3
"""Download source for the 25-game ARC-AGI-3 cloud sweep into environment_files/.

Each call to ``arc_agi.Arcade.make(game_id)`` in NORMAL mode hits
``{base_url}/api/games/{game_id}-{version}/source`` and writes the source
into ``environment_files/<game_id>/<version>/<class>.py`` plus
``metadata.json``. After this runs, ``play.py --game <name>``
works for every downloaded game.

Run from the repo root. Games land in ./environment_files/, which is gitignored.

Usage:
    python scripts/download_cloud_games.py
    python scripts/download_cloud_games.py --games ar25 bp35 cd82
"""
import argparse
import os
import sys
from pathlib import Path


GAMES_25 = [
    "ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
    "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
    "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33",
    "wa30",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", default=GAMES_25,
                        help="Game IDs to download (default: 25-game cloud list)")
    parser.add_argument("--environments-dir", default="environment_files")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass

    if not os.environ.get("ARC_API_KEY"):
        print("ERROR: ARC_API_KEY not set (looked in env + .env)", file=sys.stderr)
        sys.exit(1)

    import arc_agi
    from arc_agi import OperationMode

    arcade = arc_agi.Arcade(
        operation_mode=OperationMode.NORMAL,
        environments_dir=args.environments_dir,
    )

    env_root = Path(args.environments_dir)
    have, missing, downloaded, failed = [], [], [], []

    for game_id in args.games:
        existed_before = (env_root / game_id).exists()
        wrapper = arcade.make(game_id)
        if wrapper is None:
            failed.append(game_id)
            print(f"  FAIL  {game_id}")
            continue
        if existed_before:
            have.append(game_id)
            print(f"  cached {game_id}")
        else:
            downloaded.append(game_id)
            print(f"  GOT   {game_id}")

    print()
    print(f"already-present: {len(have)}  downloaded: {len(downloaded)}  failed: {len(failed)}")
    if failed:
        print(f"failed: {' '.join(failed)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
