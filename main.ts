// GTA VI trophy rarity forecaster (makefun/gta6-trophies-case/v1).
//
// Model, per trophy, in logit / log-time space (deterministic, stdlib only).
//
// A trophy's earn rate is a ratio, rate = 100 * E / N, of two growing
// counts: E, the tracked players who hold the trophy, and N, the players
// Exophase tracks.  Both move, and they move for different reasons:
// E grows as the tracked cohort plays through the game, while N grows
// in jumps whenever Exophase refreshes the page, pulling in a wave of
// brand-new players who have earned almost nothing yet.  A refresh
// therefore *lowers* most rates even though nobody un-earned anything.
// So a rate is not one smooth curve: it is a trend with jump risk.
//
// What the program does with a case's history:
//  1. Work on the logit scale, L = logit(rate), against the saturating
//     clock x = ln(1 + hours since the release day began,
//     2026-11-19T00:00:00Z).  Log-time slows growth down as the launch
//     week goes on instead of extrapolating a straight line forever.
//  2. Fit, per trophy, a least-squares line of L on x over the last 24
//     hours of reads (all reads if there are fewer), shrink the slope
//     toward the cross-trophy median slope (trophies share the launch
//     dynamics; a single trophy's slope is noisy), cap it, and read the
//     line at the target, anchored on the last read that lists the
//     trophy, so a level shift in the fit cannot drag the forecast away
//     from what the page last said.
//  3. Spread: the honest logit-scale standard deviation combines the
//     fit's residual (which is mostly past refresh jumps), the slope's
//     standard error times the extrapolation distance, a random-walk term
//     0.5 * sqrt(horizon in days), and a prior for a refresh the history
//     has not shown yet.  The spread is floored so a flat history never
//     claims false precision, and capped so it never explodes.
//  4. Flat history (every read identical, launch night before Exophase's
//     first refresh): there is no slope to fit, and persisting the stale
//     page is exactly wrong -- the page has sat unchanged for hours on
//     launch day, so the next refresh will dilute the rates as a wave of
//     new players arrives.  The median is shifted down on the logit scale
//     (a refresh-sized dilution, fading with horizon: over a long horizon
//     the cohort catches up again through play) and the spread is widened
//     to a full refresh-jump width.  A trophy at 100% is exempt: everyone
//     holds it, new players included, so it stays at 100%.
//  5. Quantiles are the median logit plus the normal z-scores
//     (-1.645, -0.674, 0, 0.674, 1.645) times the spread, mapped back to
//     percent, rounded to 4 decimals, clamped to [0.01, 99.99].
//
// The program reads every *.in file in $MAKEFUN_INPUT_DIR (else /input)
// and writes one /output/<id>.out per case ($MAKEFUN_OUTPUT_DIR else
// /output).  No network, no randomness, no model call: the same input
// always produces the same bytes.

import * as fs from "node:fs";
import * as path from "node:path";

const Z_SCORES: number[] = [-1.645, -0.674, 0, 0.674, 1.645];
const RELEASE_MS: number = Date.parse("2026-11-19T00:00:00Z");
const LO: number = 0.01;
const HI: number = 99.99;

interface Trophy {
  id: number;
  name?: string;
  tier?: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: (number | null)[];
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  trophies: Trophy[];
  history: Read[];
}

function clampPct(v: number): number {
  if (!Number.isFinite(v)) return 50;
  return Math.min(HI, Math.max(LO, v));
}

function logitPct(p: number): number {
  const c = clampPct(p) / 100;
  return Math.log(c / (1 - c));
}

function invLogitPct(l: number): number {
  if (l > 20) return HI;
  if (l < -20) return LO;
  return clampPct(100 / (1 + Math.exp(-l)));
}

function round4(v: number): number {
  return Math.round(v * 10000) / 10000;
}

function hoursSinceRelease(iso: string): number {
  const ms = Date.parse(iso);
  if (!Number.isFinite(ms)) return 0;
  return Math.max(0, (ms - RELEASE_MS) / 3600000);
}

function logClock(iso: string): number {
  return Math.log1p(hoursSinceRelease(iso));
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
}

/** Least-squares line of y on x. Returns slope, intercept, residual SD. */
function ols(xs: number[], ys: number[]): { b: number; a: number; s: number } {
  const n = xs.length;
  if (n === 0) return { b: 0, a: 0, s: 0.5 };
  if (n === 1) return { b: 0, a: ys[0], s: 0.5 };
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx <= 1e-12) return { b: 0, a: my, s: 0.5 };
  const b = sxy / sxx;
  const a = my - b * mx;
  let rss = 0;
  for (let i = 0; i < n; i++) {
    const r = ys[i] - (a + b * xs[i]);
    rss += r * r;
  }
  const s = n > 2 ? Math.sqrt(rss / (n - 2)) : Math.sqrt(rss / n);
  return { b, a, s };
}

/** Standard error of the OLS slope. */
function slopeSe(xs: number[], s: number): number {
  const n = xs.length;
  if (n < 3) return 0.5;
  let mx = 0;
  for (const x of xs) mx += x;
  mx /= n;
  let sxx = 0;
  for (const x of xs) sxx += (x - mx) * (x - mx);
  if (sxx <= 1e-12) return 0.5;
  return s / Math.sqrt(sxx);
}

/** Count refresh jumps: consecutive reads where the tracked base or the
 *  typical rate moved by more than a rounding step. */
function countRefreshJumps(kase: Case, logits: number[][]): number {
  const h = kase.history;
  let jumps = 0;
  for (let j = 1; j < h.length; j++) {
    const prev = h[j - 1];
    const cur = h[j];
    const dLogN = Math.abs(
      Math.log(Math.max(1, cur.players_tracked)) -
        Math.log(Math.max(1, prev.players_tracked)),
    );
    let med = 0;
    const diffs: number[] = [];
    for (let k = 0; k < logits.length; k++) {
      const a = logits[k][j - 1];
      const b = logits[k][j];
      if (Number.isFinite(a) && Number.isFinite(b)) diffs.push(Math.abs(b - a));
    }
    if (diffs.length > 0) med = median(diffs);
    if (dLogN > 0.02 || med > 0.05) jumps++;
  }
  return jumps;
}

function forecastCase(kase: Case): { trophies: { id: number; q: number[] }[] } {
  const trophies = kase.trophies;
  const nT = trophies.length;
  const history = (kase.history ?? []).slice().sort((a, b) =>
    a.at < b.at ? -1 : a.at > b.at ? 1 : 0,
  );
  const horizonDays = Math.max(0, (kase.horizon_minutes ?? 0) / 1440);
  const xTarget = logClock(kase.target ?? kase.at);
  const xLast = history.length > 0 ? logClock(history[history.length - 1].at) : xTarget;
  const dxTarget = Math.max(0, xTarget - xLast);

  // Per-trophy logit series, aligned to history order.
  const logits: number[][] = [];
  const lastRate: number[] = [];
  for (let k = 0; k < nT; k++) {
    const series: number[] = [];
    for (const read of history) {
      const r = read.rates ? read.rates[k] : null;
      series.push(r === null || r === undefined ? NaN : logitPct(r));
    }
    logits.push(series);
    const last = history.length > 0 ? history[history.length - 1].rates?.[k] : null;
    lastRate.push(last === null || last === undefined ? 50 : last);
  }

  // Is the whole page flat? Every read identical to the last, players
  // tracked unchanged: launch night before the first refresh.
  let flat = history.length >= 2;
  if (flat) {
    const n0 = history[0].players_tracked;
    for (let j = 1; j < history.length && flat; j++) {
      if (history[j].players_tracked !== n0) flat = false;
      const rj = history[j].rates ?? [];
      const r0 = history[0].rates ?? [];
      for (let k = 0; k < nT; k++) {
        if ((rj[k] ?? null) !== (r0[k] ?? null)) {
          flat = false;
          break;
        }
      }
    }
  }

  const jumps = flat ? 0 : countRefreshJumps(kase, logits);
  const rwSigma = 0.5 * Math.sqrt(horizonDays);

  // ---- Flat-history branch: expect the first refresh. ----
  if (flat) {
    const decay = Math.exp(-horizonDays / 1.5);
    const shift = -1.05 * decay;
    const sigma = Math.sqrt(rwSigma * rwSigma + 1.0 * 1.0);
    const rows = trophies.map((t, k) => {
      const r = lastRate[k];
      let mu: number;
      let s = sigma;
      if (r >= 99.99) {
        // Held by everyone, new players included: it stays at 100%.
        mu = logitPct(HI);
        s = 0.05;
      } else {
        mu = logitPct(r) + shift;
      }
      const q = Z_SCORES.map((z) => round4(invLogitPct(mu + z * s)));
      for (let i = 1; i < q.length; i++) q[i] = Math.max(q[i], q[i - 1]);
      return { id: t.id, q };
    });
    return { trophies: rows };
  }

  // ---- Trend branch: per-trophy log-time fit, shrunk to the pool. ----
  const checkpointMs = Date.parse(kase.at);
  const windowStartMs = checkpointMs - 24 * 3600000;
  const slopes: number[] = [];
  const fits: { b: number; s: number; se: number; xs: number[]; n: number }[] = [];
  for (let k = 0; k < nT; k++) {
    const xs: number[] = [];
    const ys: number[] = [];
    for (let j = 0; j < history.length; j++) {
      const v = logits[k][j];
      if (!Number.isFinite(v)) continue;
      if (history.length > 3 && Date.parse(history[j].at) < windowStartMs) continue;
      xs.push(logClock(history[j].at));
      ys.push(v);
    }
    if (xs.length < 2) {
      fits.push({ b: 0, s: 0.5, se: 0.5, xs, n: xs.length });
      continue;
    }
    const f = ols(xs, ys);
    fits.push({ b: f.b, s: f.s, se: slopeSe(xs, f.s), xs, n: xs.length });
    slopes.push(f.b);
  }
  const poolSlope = median(slopes);
  const refreshPrior = jumps >= 2 ? 0.35 : 0.7;

  const rows = trophies.map((t, k) => {
    const r = lastRate[k];
    const fit = fits[k];
    // Anchor on the last read that actually lists this trophy (a later
    // read may lack it); fall back to the last rate, then to 50%.
    let lastLogit = NaN;
    for (let j = logits[k].length - 1; j >= 0; j--) {
      if (Number.isFinite(logits[k][j])) {
        lastLogit = logits[k][j];
        break;
      }
    }
    if (!Number.isFinite(lastLogit)) lastLogit = logitPct(r);
    // Shrink the per-trophy slope toward the pooled slope; cap it.
    let bStar = fit.n >= 2 ? (fit.n * fit.b + 4 * poolSlope) / (fit.n + 4) : 0;
    if (bStar > 2) bStar = 2;
    if (bStar < -2) bStar = -2;
    let move = bStar * dxTarget;
    if (move > 3) move = 3;
    if (move < -3) move = -3;
    const mu = lastLogit + move;
    const sRes = fit.n >= 3 ? fit.s : 0.5;
    const sigmaRaw = Math.sqrt(
      sRes * sRes +
        (fit.se * dxTarget) * (fit.se * dxTarget) +
        rwSigma * rwSigma +
        refreshPrior * refreshPrior,
    );
    const sigma = Math.min(2.2, Math.max(0.45, sigmaRaw));
    const q = Z_SCORES.map((z) => round4(invLogitPct(mu + z * sigma)));
    for (let i = 1; i < q.length; i++) q[i] = Math.max(q[i], q[i - 1]);
    return { id: t.id, q };
  });
  return { trophies: rows };
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
  fs.mkdirSync(outDir, { recursive: true });
  const names = fs
    .readdirSync(inDir)
    .filter((n) => n.endsWith(".in"))
    .sort();
  for (const name of names) {
    const raw = fs.readFileSync(path.join(inDir, name), "utf-8");
    const kase = JSON.parse(raw) as Case;
    const out = forecastCase(kase);
    const stem = name.slice(0, -3);
    fs.writeFileSync(
      path.join(outDir, stem + ".out"),
      JSON.stringify(out) + "\n",
      "utf-8",
    );
  }
}

main();
