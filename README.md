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
  config.py     constants + flagged modeling decisions
  ephemeris.py  pyswisseph wrapper (KP_NEW ayanamsa enforced)
  aspects.py    angle separation + Gaussian "resonance"
scripts/
  demo_positions.py   small-win demo
results/     generated outputs (git-ignored)
```

## The plan (each step added one at a time)

1. **[done]** Ephemeris + aspect engine (this scaffold).
2. Load the Nifty daily CSV and compute daily returns (the prediction target).
3. Build one feature row per day: resonance of every pair x harmonic.
4. **Learn** weights/signs on *train* years only.
5. **Test** on unseen years + a random/null baseline; correct for multiple testing.
6. Emit `aspect_polarity.json` (production `#007` schema) — only for rules that survive.

## Honest expectations

Decades of evidence suggest markets move on news, earnings, rates and human
behaviour rather than planetary angles. This project may well find little or no
signal. That's fine. The point is to *measure carefully* rather than guess — and
to build real, transferable data skills along the way.
