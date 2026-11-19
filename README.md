# GTA VI trophy rarity forecaster

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/beduldudul/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/beduldudul/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI (PS5) trophy's Exophase earn rate at
2026-11-26T00:00:00Z, as five quantiles per trophy, for make.fun's
`gta6-trophy-rarity-forecast-01`. Entry point is `main.ts` (Node 22,
standard library only, offline, deterministic).

## How it works

A trophy's earn rate is a ratio, `rate = 100 * E / N`: `E`, the tracked
players who hold the trophy, grows as the cohort plays, while `N`, the
players Exophase tracks, grows in jumps when Exophase refreshes the page,
pulling in a wave of brand-new players who have earned almost nothing yet.
A refresh therefore *lowers* most rates even though nobody un-earned
anything. So a rate is a trend with jump risk, not one smooth curve.

Per trophy, on the logit scale against the saturating clock
`x = ln(1 + hours since 2026-11-19T00:00:00Z)`:

- Fit a least-squares line of logit(rate) on `x` over the last 24 hours
  of reads, shrink the slope toward the cross-trophy median slope, cap
  it, and read the line at the target, anchored on the last read.
- Spread combines the fit residual (mostly past refresh jumps), the
  slope's standard error times the extrapolation distance, a random-walk
  term `0.5 * sqrt(horizon in days)`, and a prior for an unseen refresh.
- Flat history (every read identical, launch night before Exophase's
  first refresh): do not persist the stale page. The median is shifted
  down a refresh-sized amount on the logit scale (fading with horizon)
  and the spread is widened to a full refresh-jump width. A trophy at
  100% stays at 100%.
- Quantiles are the median logit plus the normal z-scores
  (-1.645, -0.674, 0, 0.674, 1.645) times the spread, mapped back to
  percent.

## Run

```sh
node main.ts   # reads $MAKEFUN_INPUT_DIR (default /input),
               # writes $MAKEFUN_OUTPUT_DIR (default /output)
```

Reads every `*.in` case file and writes one `*.out` forecast per case:
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

Local practice check (kit's `input/` holds `p00.in` / `p01.in`): mean
pinball loss on the logit scale ≈ 0.30 against a proxy answer
reconstructed from the refresh the rules publish (players 638 → 3,868,
trophies earned 943 → 4,276), versus ≈ 1.51 for the house persistence
baseline; the same forecast stays below the baseline under per-trophy
refresh variation as well.

## License

MIT. See LICENSE.
