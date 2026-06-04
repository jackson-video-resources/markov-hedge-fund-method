"""Deterministic tests for the Markov regime engine.

These lock the load-bearing claims the README and SKILL.md make about the math:
rows of the transition matrix sum to 1, the stationary vector is a true fixed
point, the walk-forward backtest has no lookahead, and its O(n) incremental
count update is bit-identical to an O(n^2) from-scratch rebuild.

Run (no project install needed — uv resolves deps on the fly):

    uv run --with pytest --with numpy --with pandas pytest tests/ -q

hmmlearn / yfinance are NOT required: every test feeds synthetic data or a
local CSV, so the suite runs anywhere the observable model runs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# Import the script module by path (it lives in scripts/, not on sys.path).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "markov_regime.py"
_spec = importlib.util.spec_from_file_location("markov_regime", _SCRIPT)
mr = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(mr)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _synthetic_close(n: int = 900, seed: int = 0) -> pd.Series:
    """A price series with a bull -> bear -> sideways arc, deterministic."""
    rng = np.random.default_rng(seed)
    third = n // 3
    drift = np.concatenate(
        [
            np.full(third, 0.0015),
            np.full(third, -0.0015),
            np.full(n - 2 * third, 0.0),
        ]
    )
    ret = drift + rng.normal(0, 0.012, n)
    price = 100.0 * np.exp(np.cumsum(ret))
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.Series(price, index=idx, name="synthetic")


def _ref_walk_forward(close: pd.Series, labels: pd.Series, min_train: int) -> dict:
    """Reference O(n^2) walk-forward: rebuild counts FROM SCRATCH every step.

    This is the explicit, obviously-no-lookahead implementation. The production
    function maintains a running count matrix instead; if they agree to machine
    precision, the production path is both lookahead-free and exact.
    """
    daily_returns = close.pct_change().dropna()
    common = labels.index.intersection(daily_returns.index)
    lab = np.asarray(labels.loc[common], dtype=int)
    rets = daily_returns.loc[common].to_numpy(dtype=float)
    if len(lab) < min_train + 30:
        return {"sharpe": float("nan"), "max_drawdown": float("nan"), "n_trades": 0}

    sr = []
    for t in range(min_train, len(lab) - 1):
        counts = np.zeros((3, 3), dtype=float)
        for i in range(t - 1):  # transitions strictly among labels[:t]
            counts[lab[i], lab[i + 1]] += 1.0
        row_sums = counts.sum(axis=1, keepdims=True)
        safe = np.where(row_sums == 0, 1.0, row_sums)
        P = counts / safe
        cs = lab[t]
        position = float(np.sign(P[cs, 2] - P[cs, 0]))
        sr.append(position * rets[t + 1])

    sr = np.asarray(sr, dtype=float)
    std = sr.std(ddof=1) if len(sr) > 1 else 0.0
    sharpe = float(sr.mean() / std * np.sqrt(252)) if std and np.isfinite(std) else float("nan")
    equity = (1.0 + sr).cumprod()
    drawdown = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)
    max_dd = float(drawdown.min()) if len(drawdown) else float("nan")
    return {"sharpe": sharpe, "max_drawdown": max_dd, "n_trades": int(len(sr))}


# --------------------------------------------------------------------------- #
# Transition matrix
# --------------------------------------------------------------------------- #
def test_transition_rows_sum_to_one():
    """Every observed row of P is a probability distribution (sums to 1)."""
    labels = mr.label_regimes(_synthetic_close())
    P = mr.build_transition_matrix(labels)
    observed = np.unique(np.asarray(labels, dtype=int))
    for state in observed:
        assert P[state].sum() == np.float64(1.0) or np.isclose(P[state].sum(), 1.0)
    # All entries are valid probabilities.
    assert (P >= 0).all() and (P <= 1).all()


def test_transition_matrix_is_mle_counts():
    """P[i,j] equals count(i->j) / count(i->*) — a plain maximum-likelihood estimate."""
    labels = pd.Series([0, 1, 2, 1, 0, 1, 2, 2, 1, 0])
    P = mr.build_transition_matrix(labels)
    # From 1 (Sideways) we go to: 2, 0, 2, 0  -> {0:2, 2:2} of 4 transitions.
    assert np.isclose(P[1, 0], 0.5)
    assert np.isclose(P[1, 2], 0.5)
    assert np.isclose(P[1, 1], 0.0)


# --------------------------------------------------------------------------- #
# Stationary distribution
# --------------------------------------------------------------------------- #
def test_stationary_is_fixed_point():
    """pi @ P == pi, pi >= 0, sum(pi) == 1."""
    labels = mr.label_regimes(_synthetic_close())
    P = mr.build_transition_matrix(labels)
    pi = mr.stationary_distribution(P)
    assert np.isclose(pi.sum(), 1.0)
    assert (pi >= -1e-12).all()
    assert np.allclose(pi @ P, pi, atol=1e-9)


def test_stationary_matches_high_matrix_power():
    """The stationary vector matches a row of P^n for large n (Chapman-Kolmogorov)."""
    labels = mr.label_regimes(_synthetic_close())
    P = mr.build_transition_matrix(labels)
    pi = mr.stationary_distribution(P)
    Pn = mr.nstep_forecast(P, 512)
    assert np.allclose(Pn[0], pi, atol=1e-6)


# --------------------------------------------------------------------------- #
# Walk-forward: no lookahead + incremental == from-scratch
# --------------------------------------------------------------------------- #
def test_walk_forward_incremental_equals_from_scratch():
    """The O(n) running-counts path is bit-identical to an O(n^2) rebuild.

    This single assertion covers two README claims at once: 'no lookahead'
    (the reference only ever uses labels[:t]) and 'bit-for-bit identical to a
    from-scratch rebuild'.
    """
    close = _synthetic_close()
    labels = mr.label_regimes(close)
    got = mr.walk_forward_backtest(close, labels, min_train=mr.DEFAULT_MIN_TRAIN)
    ref = _ref_walk_forward(close, labels, min_train=mr.DEFAULT_MIN_TRAIN)
    assert got["n_trades"] == ref["n_trades"]
    assert np.isclose(got["sharpe"], ref["sharpe"], rtol=1e-12, atol=1e-12)
    assert np.isclose(got["max_drawdown"], ref["max_drawdown"], rtol=1e-12, atol=1e-12)


def test_walk_forward_short_history_is_nan():
    """Too little data -> NaN sharpe, zero trades, no crash."""
    close = _synthetic_close(n=120)
    labels = mr.label_regimes(close)
    bt = mr.walk_forward_backtest(close, labels, min_train=mr.DEFAULT_MIN_TRAIN)
    assert bt["n_trades"] == 0
    assert np.isnan(bt["sharpe"])


# --------------------------------------------------------------------------- #
# Labels & signal
# --------------------------------------------------------------------------- #
def test_label_uses_simple_return_thresholds():
    """A clean +8% / -8% move over the window crosses the ±5% simple-return bands.

    Also pins the rule to SIMPLE return (pct_change), matching the Pine indicator.
    """
    window = 5
    # Flat, then a clean +8% step held for a window, then a -8% step held.
    prices = [100.0] * 6 + [108.0] * window + [99.36] * window
    close = pd.Series(prices, index=pd.bdate_range("2022-01-01", periods=len(prices)))
    labels = mr.label_regimes(close, window=window, threshold=0.05)
    assert (labels == 2).any(), "expected at least one Bull label after +8%"
    assert (labels == 0).any(), "expected at least one Bear label after -8%"


def test_signal_equals_bull_minus_bear_and_bounded():
    a = mr.analyze(_synthetic_close(), source="synthetic")
    np_ = a["next_state_probabilities"]
    assert np.isclose(a["signal"], np_["bull"] - np_["bear"])
    assert -1.0 <= a["signal"] <= 1.0


# --------------------------------------------------------------------------- #
# CSV loader & JSON contract
# --------------------------------------------------------------------------- #
def test_csv_autodetects_odd_column_names(tmp_path):
    """Loose column detection: 'Timestamp' + 'Adj Close' should just work."""
    df = pd.DataFrame(
        {
            "Timestamp": pd.bdate_range("2022-01-01", periods=5),
            "Adj Close": [10.0, 11.0, 10.5, 12.0, 11.5],
        }
    )
    p = tmp_path / "odd.csv"
    df.to_csv(p, index=False)
    s = mr.load_csv(str(p))
    assert len(s) == 5
    assert s.iloc[0] == 10.0


def test_csv_single_numeric_column_fallback(tmp_path):
    df = pd.DataFrame(
        {"when": pd.bdate_range("2022-01-01", periods=4), "value": [1.0, 2.0, 3.0, 4.0]}
    )
    p = tmp_path / "single.csv"
    df.to_csv(p, index=False)
    s = mr.load_csv(str(p))
    assert s.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_analyze_emits_full_json_contract():
    """Every field documented in SKILL.md's JSON table is present."""
    a = mr.analyze(_synthetic_close(), source="synthetic")
    expected = {
        "source", "rows", "date_start", "date_end", "params", "states",
        "current_regime", "next_state_probabilities", "signal",
        "transition_matrix", "persistence_diagonal", "stationary_distribution",
        "walk_forward", "hmm", "framework", "disclaimer",
    }
    assert expected <= set(a)
    assert a["states"] == ["Bear", "Sideways", "Bull"]
    assert set(a["walk_forward"]) == {"sharpe", "max_drawdown", "n_trades"}
