# Nifty Astro-Backtest

An **honest experiment**: does the position of the planets carry *any* measurable
predictive signal for daily Nifty 50 moves?

We are **not** assuming the answer. We build a rulebook (planet-pair weights +
bullish/bearish signs) *from data* using a strict train/test split, and we let
the result be whatever it is. **A negative result is a valid, useful result** —
it would simply tell us the signal isn't there, which is worth knowing.

> This repo is the research sandbox. If (and only if) a rule survives honest
> out-of-sample testing, the resulting `aspect_polarity.json` can be promoted
> into the production `AstroTradeKP` project.

## Why this is built carefully

- **Same ayanamsa as production.** We use `SIDM_KRISHNAMURTI_VP291` (KP_NEW,
  integer 45, pyswisseph >= 2.10), geocentric sidereal — identical to AstroTradeKP
  Layer 1, so any rulebook we build here is valid there.
- **Modeling choices are flagged, not hidden.** The natal chart and lunar-node
  flavour are real decisions (see `src/config.py`); they are marked PLACEHOLDER
  until confirmed, never silently assumed.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Try the "small win" demo (no market data needed)

```bash
python -m scripts.demo_positions 2020-03-23
```

This prints real sidereal positions for the date and the strongest
transit->natal aspect for each planet pair.

## Project layout

```
data/        your Nifty daily CSV goes here (git-ignored)
src/
  config.py        constants + flagged modeling decisions
  ephemeris.py     pyswisseph wrapper (KP_NEW ayanamsa enforced)
  aspects.py       angle separation + Gaussian "resonance"
  labels.py        load CSV -> daily returns (the prediction target)
  features.py      per-day resonance matrix (15 pairs x 5 harmonics = 75)
  validate.py      correlations, p-values, Benjamini-Hochberg FDR
  backtest.py      train/test split + out-of-sample + permutation null
  build_rulebook.py  emit aspect_polarity.json (#007 schema) if EARNED
scripts/
  demo_positions.py   small-win demo
  run_backtest.py     full end-to-end experiment (CLI)
results/     generated outputs (git-ignored)
```

## Run the full backtest

```bash
python -m scripts.run_backtest --csv data/nifty_daily.csv --test-from 2016 --n-null 1000 --plot
```

- `--test-from YEAR` — years before it are TRAIN, that year onward are unseen TEST.
- `--n-null N` — permutation-null iterations (more = a more precise luck baseline).
- `--plot` — saves `results/null_distribution.png` (observed vs random).

It prints a verdict, writes `results/summary.json`, and writes
`results/aspect_polarity.candidate.json`. **Real numbers are written only if the
result beats the random null**; otherwise the production `REQUIRED_BUT_UNSPECIFIED`
placeholders are kept — we never invent numbers.

## How to read the result

- **out-of-sample correlation** near 0 and a **null p-value** above ~0.05 mean the
  aspects did **not** predict unseen returns better than chance.
- **hit-rate** near 50% = coin flip.
- A few features may look interesting on TRAIN before correction; if **0 survive
  FDR**, those are almost certainly noise.

## The plan

1. **[done]** Ephemeris + aspect engine.
2. **[done]** Load Nifty daily CSV -> daily returns.
3. **[done]** Per-day resonance features (15 pairs x 5 harmonics).
4. **[done]** Learn weights/signs on TRAIN years only.
5. **[done]** Test on unseen years + permutation null + FDR correction.
6. **[done]** Emit `aspect_polarity.json` only for rules that survive.

### Result on the included data (2007-2018)

Across multiple train/test splits (test from 2013 / 2014 / 2016), the
out-of-sample correlation is ~0 and never beats the random null; 0/75 features
survive FDR. **Verdict: no predictive signal beyond chance.** That is a valid,
useful finding. Things still worth trying (with the same honest discipline):
a different natal chart, the intraday Moon engine on finer data, or non-astro
baselines — but expect the prior to remain weak.

## Honest expectations

Decades of evidence suggest markets move on news, earnings, rates and human
behaviour rather than planetary angles. This project may well find little or no
signal. That's fine. The point is to *measure carefully* rather than guess — and
to build real, transferable data skills along the way.
