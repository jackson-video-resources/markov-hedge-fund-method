---
name: regime
description: >-
  Detect the market regime (Bull / Bear / Sideways) for ANY asset and turn it
  into a tradeable signal or a risk filter. Use this whenever the user wants
  regime detection, a regime-aware confirmation on an existing strategy, a
  regime risk gate, regime-based position sizing, a Markov transition matrix,
  n-step regime forecasting, a stationary regime mix, or a no-lookahead
  walk-forward regime backtest — on a ticker (via yfinance) or on the user's
  own CSV price series. Composes into any existing trading agent or strategy
  without rewriting it. Framework by Roan (@RohOnChain).
---

# Regime — Markov regime detection for any asset

This skill answers one question for any asset: **what regime are we in, how
sticky is it, and what does that imply for risk and direction?** It's built to
slot into a trading agent the user already has — as a confirmation layer, a
signal, or a risk gate — without them rewriting their strategy.

Framework: Roan (@RohOnChain). Refactored into this plugin by Lewis Jackson.
Backtests are historical, not forward-looking.

## How to invoke

One command. It takes EITHER a ticker OR the user's own CSV, so it drops into
any pipeline regardless of asset:

```bash
# any ticker yfinance knows (stocks, ETFs, crypto, FX, futures):
uv run ${CLAUDE_PLUGIN_ROOT}/scripts/markov_regime.py --ticker BTC-USD --json

# the user's own price file (their data, their asset, their pipeline):
uv run ${CLAUDE_PLUGIN_ROOT}/scripts/markov_regime.py --csv ./my_prices.csv --json
```

- Drop `--json` for the on-camera pretty terminal output (matrix, persistence
  diagonal, stationary mix, walk-forward Sharpe + max DD, HMM line).
- `--csv` needs only a date column and a close column. It auto-detects common
  names (`date`/`time`/`timestamp`, `close`/`adj close`/`price`/`last`); if
  there's exactly one numeric column it uses that. No reformatting required.
- Defaults: `--window 20`, `--threshold 0.05` (±5%), `--years 10`,
  `--min-train 252`. All overridable. `--no-hmm` skips the HMM.
- First `uv run` resolves dependencies once (~10–20s), then it's instant.

## The JSON contract (every field)

`--json` prints exactly one JSON object to stdout and nothing else. On failure
it prints `{"error": "..."}` and exits non-zero. Fields:

| Field | Type | Meaning |
|---|---|---|
| `source` | str | the ticker or CSV path analysed |
| `rows` | int | number of price rows used |
| `date_start`, `date_end` | str | ISO dates of the window |
| `params` | obj | `window`, `threshold`, `min_train` actually used |
| `states` | list | `["Bear","Sideways","Bull"]` — fixed index order 0,1,2 |
| `current_regime` | str | the regime as of the last bar |
| `next_state_probabilities` | obj | `bear`/`sideways`/`bull` — P(next \| current) |
| `signal` | float | **bull_prob − bear_prob** in [-1, 1]. >0 long bias, <0 short bias, magnitude = conviction |
| `transition_matrix` | 3×3 list | row = from-state, col = to-state, rows sum to 1 |
| `persistence_diagonal` | obj | `bear`/`sideways`/`bull` — P(stay in same regime). High = sticky regime |
| `stationary_distribution` | obj | `bear`/`sideways`/`bull` — long-run fraction of time in each regime, sums to 1 |
| `walk_forward` | obj | `sharpe`, `max_drawdown`, `n_trades` from a re-estimated-every-step, no-lookahead backtest. `sharpe`/`max_drawdown` may be `NaN` if history is too short |
| `hmm` | obj | `available: true` → `regimes` (label, latent_state, mean_daily_return) + `caveat`. `available: false` → `reason` (graceful degrade — everything else is still valid) |
| `framework`, `disclaimer` | str | attribution + "historical, not forward-looking" |

`signal` and `current_regime` are the two fields most strategies consume.
`stationary_distribution` is the one most risk layers consume.

## Composition — slot it into what the user already has

The user already has a trading agent or strategy on some asset. This skill is a
layer they add, not a system they adopt. Run it, read one or two fields, gate
their existing logic. Three patterns:

### (a) Regime confirmation on an existing momentum/strategy

The user has entry logic that already fires. Wrap it: only take longs when the
regime agrees, only short when it disagrees.

```python
import json, subprocess
r = json.loads(subprocess.check_output(
    ["uv","run",f"{PLUGIN}/scripts/markov_regime.py","--ticker","SPY","--json"]))

if my_strategy_says_long and r["signal"] > 0:
    enter_long()          # momentum + regime agree → take it
elif my_strategy_says_long and r["signal"] <= 0:
    skip()                # momentum says go, regime says don't → stand down
```

One line of gating. Their strategy is untouched; the regime just vetoes trades
that fight the prevailing chain.

### (b) Stationary distribution as a tail-risk / position-size filter

The stationary mix is the asset's long-run baseline. A high baseline Bear share
means this asset structurally spends a lot of time in drawdown — size down.

```python
bear_baseline = r["stationary_distribution"]["bear"]
size = base_size * (1.0 - bear_baseline)      # heavier bear regime → smaller bets
# or hard gate: if bear_baseline > 0.40: size = 0   # too tail-heavy to trade
```

No new model. The user keeps their sizing logic and scales it by a single
number that reflects how regime-dangerous the asset actually is.

### (c) Standalone signal

No existing strategy needed. The `signal` field is already a direction +
conviction in [-1, 1]:

```python
position = r["signal"]        # +0.6 → 60% long; -0.4 → 40% short; ~0 → flat
```

Sanity-check it first with the printed walk-forward Sharpe + max drawdown
(run without `--json` to see them on screen) before sizing real capital.

## Notes

- **Asset-agnostic by design.** `--ticker` for anything yfinance covers;
  `--csv` for the user's own data on any asset/timeframe their pipeline
  produces. The math is identical either way.
- **No lookahead.** The walk-forward refits the matrix using only past data at
  every step (incremental O(n), bit-identical to a from-scratch rebuild).
- **HMM degrades gracefully.** If `hmmlearn` can't compile (e.g. Windows
  without MSVC), `hmm.available` is `false` with a `reason` and every other
  field is still correct. HMM states are labelled by ascending mean return, so
  a positive "Bear" mean just means the worst latent state was still
  net-positive over that window.
- Defaults match the on-camera demo and the Pine Script bonus (window 20,
  ±5%). Use `--threshold 0.02` to reproduce the tighter labelling from the
  original onboarding prompt. The Pine indicator labels off the same **simple
  rolling return** as this engine, so the two agree bar-for-bar.
- **Verified, not asserted.** The claims above (rows sum to 1, stationary is a
  fixed point, no-lookahead, incremental backtest == from-scratch rebuild) are
  locked by a deterministic test suite — see [`eval.json`](./eval.json) and run
  `uv run --with pytest --with numpy --with pandas pytest tests/ -q`.
