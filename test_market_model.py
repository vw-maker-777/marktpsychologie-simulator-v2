"""Regressionstests für die wichtigsten Modellversprechen."""

from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from market_model import DEFAULT_PARAMS, PARAMETER_NAMES, PRESETS, max_drawdown, simulate


def test_same_seed_is_reproducible() -> None:
    first = simulate(replace(DEFAULT_PARAMS, days=120, seed=123)).data
    second = simulate(replace(DEFAULT_PARAMS, days=120, seed=123)).data
    pd.testing.assert_frame_equal(first, second)


def test_hft_capital_changes_price_path() -> None:
    base = replace(
        DEFAULT_PARAMS,
        days=250,
        seed=7,
        fundamental_volatility=0.008,
        jump_probability=0.02,
        retail_panic_threshold=-0.015,
        retail_panic_sell=0.50,
        fund_vix_threshold=24.0,
        fund_outflow_rate=0.25,
        hft_vix_shutdown=80.0,
    )
    without_hft = simulate(replace(base, hft_capital=0.0)).data["Kurs"].to_numpy()
    with_hft = simulate(replace(base, hft_capital=25_000.0)).data["Kurs"].to_numpy()
    assert np.max(np.abs(without_hft - with_hft)) > 1e-6


def test_final_daily_return_is_really_capped() -> None:
    params = replace(
        DEFAULT_PARAMS,
        days=50,
        seed=99,
        fundamental_drift=0.0,
        fundamental_volatility=0.0,
        jump_probability=1.0,
        jump_mean=0.50,
        jump_volatility=0.0,
        daily_return_cap=0.03,
    )
    returns = simulate(params).data["Rendite"].iloc[1:]
    assert (returns.abs() <= 0.03 + 1e-12).all()
    assert np.isclose(returns.max(), 0.03)


def test_psychology_can_materially_change_price() -> None:
    base = replace(
        DEFAULT_PARAMS,
        days=100,
        seed=5,
        fundamental_drift=0.002,
        fundamental_volatility=0.0,
        jump_probability=0.0,
        retail_greed_threshold=0.005,
        retail_greed_buy=0.0,
        orderflow_impact=0.20,
        hft_capital=0.0,
        fund_trend_step=0.0,
    )
    frozen = simulate(base).data["Kurs"].iloc[-1]
    responsive = simulate(replace(base, retail_greed_buy=0.30)).data["Kurs"].iloc[-1]
    assert abs(responsive / frozen - 1.0) > 0.01


def test_max_drawdown_uses_running_peak() -> None:
    prices = np.array([100.0, 200.0, 120.0, 150.0])
    assert np.isclose(max_drawdown(prices), -0.40)


def test_every_preset_is_complete() -> None:
    expected = set(PARAMETER_NAMES)
    for values in PRESETS.values():
        assert set(values) == expected
        assert set(values) == set(asdict(DEFAULT_PARAMS))
