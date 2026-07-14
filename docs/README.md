# OPINE-World results site

Single-page results blog served by GitHub Pages from `main:/docs`. Page design
adapted from [rednote-hilab/TELL](https://github.com/rednote-hilab/TELL).

## Layout

- `index.html` — the whole page: prose, method math, observations (η graph,
  hypothesis cases, engine anatomy, token economics, effort headroom), card
  grid, replay modal. Hand-edited; the observation data consts (`CASES`,
  `ANATOMY`, `TOKENS`, `PTS`) are curated inline and editable.
- `assets/site_data.js` — generated card stats (`D`) and thumbnails (`THUMBS`).
- `assets/eta_data.js` — generated per-game η traces and level completions.
- `assets/fig/` — method figures from the paper.
- `replay_data/<game>.json.gz` — generated replay bundles: per-step frames plus
  intermediate animation ticks (delta-compressed), analyzer move-set reasoning,
  synthesis rounds, and engine-code versions (v0 full, then line diffs).
- `build_site_data.py` — generates the data files from the run artifacts.
  Needs Pillow (tick frames are decoded from the archived PNGs, calibrated
  against the ASCII frames).

## Regenerate

```
python3 build_site_data.py --results /path/to/opine-world-results
```

The script hard-asserts consistency (frame counts, delta replay, engine diff
round-trips, level completions vs the paper's cleared counts) and prints one
line per game with bundle sizes. Card scores are computed with the paper's
scoring (see constants at the top of the script, copied from
`ARC-3-D3M-Model/docs/iclr/make_tables.py`).

## Preview

```
python3 -m http.server -d docs 8000   # or from the repo root
```

The scatter's method points live in `index.html` (`const PTS`); fill in `wins`
as scorecards are verified and points move from the side list onto the chart.
