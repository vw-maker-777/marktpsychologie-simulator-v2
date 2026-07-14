"""Reproduzierbarer Modellkern für den Marktpsychologie-Simulator.

Das Modul enthält keine Streamlit-Abhängigkeit. Dadurch kann die Simulation
isoliert getestet, in Notebooks verwendet oder von einer anderen UI aufgerufen
werden.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Callable, Final

import numpy as np
import pandas as pd


RETAIL_BASE_VOLUME: Final[float] = 400.0
FUND_BASE_VOLUME: Final[float] = 1_000.0
BASE_MARKET_DEPTH: Final[float] = 1_000.0
TRADING_DAYS_PER_YEAR: Final[int] = 250


@dataclass(frozen=True, slots=True)
class ModelParams:
    """Alle Parameter eines vollständig reproduzierbaren Simulationslaufs."""

    # Laufsteuerung
    days: int = 500
    seed: int = 42

    # Exogenes Marktumfeld
    fundamental_drift: float = 0.00020
    fundamental_volatility: float = 0.0040
    jump_probability: float = 0.010
    jump_mean: float = 0.000
    jump_volatility: float = 0.025
    daily_return_cap: float = 0.120

    # Preiswirkung des Orderflows
    orderflow_impact: float = 0.080

    # Privatanleger
    retail_start: float = 0.60
    retail_greed_threshold: float = 0.030
    retail_panic_threshold: float = -0.050
    retail_panic_sell: float = 0.25
    retail_greed_buy: float = 0.10
    retail_reversion: float = 0.020

    # Fonds
    fund_start: float = 1.00
    fund_leverage_limit: float = 1.30
    fund_vix_threshold: float = 30.0
    fund_outflow_rate: float = 0.12
    fund_trend_step: float = 0.06
    fund_deleverage_step: float = 0.12

    # HFT / Market Maker
    hft_capital: float = 10_000.0
    hft_vix_shutdown: float = 45.0
    hft_liquidity_factor: float = 0.050

    # Zentralbank
    cb_intervention_threshold: float = 0.10
    cb_purchase_return: float = 0.025
    cb_vix_reduction: float = 0.25


DEFAULT_PARAMS = ModelParams()


def _preset(**overrides: float | int) -> dict[str, float | int]:
    """Erzeugt ein vollständiges Preset statt nur einzelner Teilwerte."""

    return asdict(replace(DEFAULT_PARAMS, **overrides))


PRESETS: dict[str, dict[str, float | int]] = {
    "Benutzerdefiniert": _preset(),
    "1. Panik-Crash / Liquiditätsentzug": _preset(
        seed=101,
        fundamental_volatility=0.0060,
        jump_probability=0.025,
        jump_mean=-0.025,
        jump_volatility=0.035,
        orderflow_impact=0.100,
        retail_start=0.75,
        retail_panic_threshold=-0.035,
        retail_panic_sell=0.45,
        fund_start=1.35,
        fund_leverage_limit=1.60,
        fund_vix_threshold=25.0,
        fund_outflow_rate=0.28,
        fund_deleverage_step=0.18,
        hft_capital=5_000.0,
        hft_vix_shutdown=25.0,
        cb_intervention_threshold=0.18,
        cb_purchase_return=0.020,
        cb_vix_reduction=0.18,
    ),
    "2. Aggressive Zentralbank": _preset(
        seed=202,
        fundamental_volatility=0.0050,
        jump_probability=0.015,
        jump_mean=-0.010,
        jump_volatility=0.030,
        retail_panic_sell=0.20,
        fund_start=1.05,
        fund_leverage_limit=1.25,
        fund_vix_threshold=35.0,
        fund_outflow_rate=0.10,
        hft_capital=13_000.0,
        hft_vix_shutdown=52.0,
        cb_intervention_threshold=0.040,
        cb_purchase_return=0.050,
        cb_vix_reduction=0.45,
    ),
    "3. Gefährlicher Fondshebel": _preset(
        seed=303,
        fundamental_volatility=0.0050,
        jump_probability=0.015,
        jump_mean=-0.008,
        jump_volatility=0.030,
        orderflow_impact=0.095,
        retail_panic_sell=0.25,
        fund_start=1.50,
        fund_leverage_limit=1.80,
        fund_vix_threshold=27.0,
        fund_outflow_rate=0.25,
        fund_trend_step=0.10,
        fund_deleverage_step=0.20,
        hft_capital=9_000.0,
        hft_vix_shutdown=42.0,
        cb_intervention_threshold=0.14,
    ),
    "4. Ruhiger Aufwärtstrend": _preset(
        seed=404,
        fundamental_drift=0.00028,
        fundamental_volatility=0.0020,
        jump_probability=0.003,
        jump_mean=0.000,
        jump_volatility=0.012,
        daily_return_cap=0.080,
        orderflow_impact=0.060,
        retail_start=0.70,
        retail_panic_threshold=-0.060,
        retail_panic_sell=0.08,
        retail_greed_buy=0.06,
        fund_start=0.90,
        fund_leverage_limit=1.10,
        fund_vix_threshold=42.0,
        fund_outflow_rate=0.05,
        fund_trend_step=0.035,
        fund_deleverage_step=0.08,
        hft_capital=18_000.0,
        hft_vix_shutdown=65.0,
        cb_intervention_threshold=0.10,
        cb_purchase_return=0.018,
        cb_vix_reduction=0.30,
    ),
}


PARAMETER_NAMES: tuple[str, ...] = tuple(field.name for field in fields(ModelParams))


@dataclass(frozen=True, slots=True)
class SimulationResult:
    params: ModelParams
    data: pd.DataFrame
    metrics: dict[str, float | int]


@dataclass(frozen=True, slots=True)
class SensitivitySpec:
    label: str
    parameter: str
    delta: float
    minimum: float
    maximum: float


SENSITIVITY_SPECS: tuple[SensitivitySpec, ...] = (
    SensitivitySpec("Panikverkäufe", "retail_panic_sell", 0.10, 0.0, 0.80),
    SensitivitySpec("Fonds-Hebel", "fund_leverage_limit", 0.20, 0.50, 2.50),
    SensitivitySpec("Fondsabflüsse", "fund_outflow_rate", 0.08, 0.0, 0.60),
    SensitivitySpec("HFT-Kapital", "hft_capital", 4_000.0, 0.0, 50_000.0),
    SensitivitySpec("HFT-Abschaltschwelle", "hft_vix_shutdown", 10.0, 10.0, 100.0),
    SensitivitySpec("Zentralbank-Schwelle", "cb_intervention_threshold", 0.03, 0.01, 0.40),
    SensitivitySpec("Orderflow-Wirkung", "orderflow_impact", 0.025, 0.0, 0.30),
    SensitivitySpec("Fundamentale Volatilität", "fundamental_volatility", 0.0015, 0.0, 0.05),
)


def validate_params(params: ModelParams) -> None:
    """Prüft harte Modellinvarianten und verhindert stille Fehlkonfigurationen."""

    checks = (
        (params.days >= 10, "days muss mindestens 10 sein"),
        (0.0 <= params.jump_probability <= 1.0, "jump_probability muss in [0, 1] liegen"),
        (params.fundamental_volatility >= 0.0, "fundamental_volatility darf nicht negativ sein"),
        (params.jump_volatility >= 0.0, "jump_volatility darf nicht negativ sein"),
        (0.0 < params.daily_return_cap < 1.0, "daily_return_cap muss in (0, 1) liegen"),
        (params.orderflow_impact >= 0.0, "orderflow_impact darf nicht negativ sein"),
        (0.0 <= params.retail_start <= 1.0, "retail_start muss in [0, 1] liegen"),
        (params.retail_panic_threshold < 0.0, "retail_panic_threshold muss negativ sein"),
        (params.retail_greed_threshold > 0.0, "retail_greed_threshold muss positiv sein"),
        (params.fund_leverage_limit >= 0.3, "fund_leverage_limit ist zu klein"),
        (0.3 <= params.fund_start <= params.fund_leverage_limit, "fund_start muss innerhalb des Hebellimits liegen"),
        (0.0 <= params.fund_outflow_rate < 1.0, "fund_outflow_rate muss in [0, 1) liegen"),
        (params.hft_capital >= 0.0, "hft_capital darf nicht negativ sein"),
        (params.hft_vix_shutdown > 10.0, "hft_vix_shutdown muss größer als 10 sein"),
        (0.0 <= params.cb_vix_reduction < 1.0, "cb_vix_reduction muss in [0, 1) liegen"),
    )
    for condition, message in checks:
        if not condition:
            raise ValueError(message)


def max_drawdown(prices: np.ndarray | pd.Series) -> float:
    """Echter Maximum Drawdown relativ zum jeweils vorherigen Höchststand."""

    values = np.asarray(prices, dtype=float)
    if values.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(values)
    drawdowns = values / running_peak - 1.0
    return float(drawdowns.min())


def _realized_volatility(returns: list[float]) -> float:
    recent = np.asarray(returns[-20:], dtype=float)
    if recent.size < 5:
        return float(abs(recent[-1]) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)
    return float(np.std(recent, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR) * 100.0)


def simulate(
    params: ModelParams,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationResult:
    """Führt einen Simulationslauf aus.

    Die tägliche Rendite wird exakt in folgende Komponenten zerlegt:
    fundamentaler Zufall + Orderflow + Zentralbank + Cap-Anpassung.
    HFT-Kapital erhöht die effektive Markttiefe und reduziert dadurch die
    Preiswirkung eines gegebenen Kauf- oder Verkaufsauftrags.
    """

    validate_params(params)
    rng = np.random.default_rng(params.seed)

    prices: list[float] = [100.0]
    returns: list[float] = [0.0]
    vix_values: list[float] = [18.0]
    retail_quotes: list[float] = [params.retail_start]
    fund_quotes: list[float] = [params.fund_start]
    fund_aum_values: list[float] = [1.0]
    hft_active_values: list[bool] = [True]
    hft_depth_values: list[float] = [params.hft_capital * params.hft_liquidity_factor]
    market_depth_values: list[float] = [BASE_MARKET_DEPTH + hft_depth_values[0]]

    exogenous_returns: list[float] = [0.0]
    orderflow_returns: list[float] = [0.0]
    cb_returns: list[float] = [0.0]
    cap_adjustments: list[float] = [0.0]

    retail_orders: list[float] = [0.0]
    fund_rebalance_orders: list[float] = [0.0]
    fund_forced_sale_orders: list[float] = [0.0]
    net_orders: list[float] = [0.0]

    jump_events: list[bool] = [False]
    jump_returns: list[float] = [0.0]
    cb_events: list[bool] = [False]
    five_day_returns: list[float] = [0.0]
    five_day_drops: list[float] = [0.0]

    retail_quote = params.retail_start
    fund_quote = params.fund_start
    fund_aum = 1.0

    for day in range(1, params.days + 1):
        previous_price = prices[-1]
        previous_vix = vix_values[-1]

        # Exakt fünf Renditeintervalle: aktueller Vortageskurs gegen Kurs vor 5 Tagen.
        if len(prices) >= 6:
            return_5d = prices[-1] / prices[-6] - 1.0
        else:
            return_5d = 0.0

        # Privatanleger: Panik, Gier oder langsame Rückkehr zur Startquote.
        previous_retail_quote = retail_quote
        if return_5d <= params.retail_panic_threshold:
            retail_quote = max(0.0, retail_quote - params.retail_panic_sell)
        elif return_5d >= params.retail_greed_threshold:
            retail_quote = min(1.0, retail_quote + params.retail_greed_buy)
        else:
            retail_quote += params.retail_reversion * (params.retail_start - retail_quote)
        retail_order = (retail_quote - previous_retail_quote) * RETAIL_BASE_VOLUME

        # Fonds: Trendfolge, Deleveraging und echte Zwangsverkäufe bei Mittelabflüssen.
        previous_fund_quote = fund_quote
        target_fund_quote = fund_quote
        if return_5d > 0.05:
            target_fund_quote += params.fund_trend_step
        elif return_5d < -0.05:
            target_fund_quote -= params.fund_deleverage_step
        else:
            neutral_target = min(params.fund_start, params.fund_leverage_limit)
            target_fund_quote += 0.02 * (neutral_target - target_fund_quote)

        stressed_fund = previous_vix > params.fund_vix_threshold
        if stressed_fund:
            target_fund_quote -= params.fund_deleverage_step

        fund_quote = float(np.clip(target_fund_quote, 0.30, params.fund_leverage_limit))
        fund_rebalance_order = (
            (fund_quote - previous_fund_quote) * FUND_BASE_VOLUME * fund_aum
        )

        redemption_fraction = params.fund_outflow_rate if stressed_fund else 0.0
        forced_sale_order = -redemption_fraction * FUND_BASE_VOLUME * fund_aum * previous_fund_quote
        fund_aum = max(0.05, fund_aum * (1.0 - redemption_fraction))

        # HFTs wirken nicht narrativ, sondern mathematisch über die Markttiefe.
        hft_active = previous_vix < params.hft_vix_shutdown
        if hft_active and params.hft_capital > 0.0:
            stress_ratio = float(np.clip(previous_vix / params.hft_vix_shutdown, 0.0, 1.0))
            hft_depth = (
                params.hft_capital
                * params.hft_liquidity_factor
                * (1.0 - 0.70 * stress_ratio)
            )
        else:
            hft_depth = 0.0

        effective_market_depth = BASE_MARKET_DEPTH + hft_depth
        net_order = retail_order + fund_rebalance_order + forced_sale_order
        orderflow_return = params.orderflow_impact * net_order / effective_market_depth

        # Exogener Prozess: standardmäßig annähernd unverzerrtes Rauschen.
        regular_noise = rng.normal(0.0, params.fundamental_volatility)
        jump_event = bool(rng.random() < params.jump_probability)
        jump_return = (
            float(rng.normal(params.jump_mean, params.jump_volatility))
            if jump_event
            else 0.0
        )
        exogenous_return = params.fundamental_drift + regular_noise + jump_return

        pre_intervention_return = exogenous_return + orderflow_return
        projected_price = previous_price * (1.0 + max(pre_intervention_return, -0.95))

        reference_index = max(0, len(prices) - 5)
        reference_price = prices[reference_index]
        drop_5d = max(0.0, 1.0 - projected_price / reference_price)

        cb_intervention = drop_5d >= params.cb_intervention_threshold
        cb_return = params.cb_purchase_return if cb_intervention else 0.0

        uncapped_return = pre_intervention_return + cb_return
        total_return = float(
            np.clip(uncapped_return, -params.daily_return_cap, params.daily_return_cap)
        )
        cap_adjustment = total_return - uncapped_return
        new_price = max(1.0, previous_price * (1.0 + total_return))

        returns_for_volatility = returns[1:] + [total_return]
        realized_vol = _realized_volatility(returns_for_volatility)
        instantaneous_vix = (
            12.0
            + 0.55 * realized_vol
            + 450.0 * max(-total_return, 0.0)
            + 180.0 * abs(jump_return)
        )
        new_vix = (
            0.75 * previous_vix
            + 0.25 * instantaneous_vix
            + float(rng.normal(0.0, 1.0))
        )
        if cb_intervention:
            new_vix *= 1.0 - params.cb_vix_reduction
        new_vix = float(np.clip(new_vix, 10.0, 80.0))

        prices.append(new_price)
        returns.append(total_return)
        vix_values.append(new_vix)
        retail_quotes.append(retail_quote)
        fund_quotes.append(fund_quote)
        fund_aum_values.append(fund_aum)
        hft_active_values.append(hft_active)
        hft_depth_values.append(hft_depth)
        market_depth_values.append(effective_market_depth)

        exogenous_returns.append(exogenous_return)
        orderflow_returns.append(orderflow_return)
        cb_returns.append(cb_return)
        cap_adjustments.append(cap_adjustment)

        retail_orders.append(retail_order)
        fund_rebalance_orders.append(fund_rebalance_order)
        fund_forced_sale_orders.append(forced_sale_order)
        net_orders.append(net_order)

        jump_events.append(jump_event)
        jump_returns.append(jump_return)
        cb_events.append(cb_intervention)
        five_day_returns.append(return_5d)
        five_day_drops.append(drop_5d)

        if progress_callback is not None:
            progress_callback(day, params.days)

    data = pd.DataFrame(
        {
            "Tag": np.arange(params.days + 1),
            "Kurs": prices,
            "Rendite": returns,
            "VIX": vix_values,
            "Retail_Quote": retail_quotes,
            "Fonds_Hebel": fund_quotes,
            "Fonds_AUM": fund_aum_values,
            "HFT_aktiv": hft_active_values,
            "HFT_Markttiefe": hft_depth_values,
            "Effektive_Markttiefe": market_depth_values,
            "Exogener_Beitrag": exogenous_returns,
            "Orderflow_Beitrag": orderflow_returns,
            "Zentralbank_Beitrag": cb_returns,
            "Cap_Anpassung": cap_adjustments,
            "Retail_Order": retail_orders,
            "Fonds_Rebalancing_Order": fund_rebalance_orders,
            "Fonds_Zwangsverkauf": fund_forced_sale_orders,
            "Netto_Order": net_orders,
            "Sprungereignis": jump_events,
            "Sprung_Beitrag": jump_returns,
            "Zentralbank_Eingriff": cb_events,
            "Rendite_5T": five_day_returns,
            "Verlust_5T": five_day_drops,
        }
    )

    component_columns = [
        "Exogener_Beitrag",
        "Orderflow_Beitrag",
        "Zentralbank_Beitrag",
        "Cap_Anpassung",
    ]
    component_abs_means = data.loc[1:, component_columns].abs().mean()
    component_total = float(component_abs_means.sum()) or 1.0

    years = params.days / TRADING_DAYS_PER_YEAR
    final_return = data["Kurs"].iloc[-1] / data["Kurs"].iloc[0] - 1.0
    cagr = (data["Kurs"].iloc[-1] / data["Kurs"].iloc[0]) ** (1.0 / years) - 1.0

    hft_off_mask = ~data.loc[1:, "HFT_aktiv"]
    orderflow_abs = data.loc[1:, "Orderflow_Beitrag"].abs()
    mean_orderflow_hft_off = (
        float(orderflow_abs[hft_off_mask].mean()) if hft_off_mask.any() else 0.0
    )
    mean_orderflow_hft_on = (
        float(orderflow_abs[~hft_off_mask].mean()) if (~hft_off_mask).any() else 0.0
    )

    metrics: dict[str, float | int] = {
        "final_return": float(final_return),
        "cagr": float(cagr),
        "max_drawdown": max_drawdown(data["Kurs"]),
        "max_vix": float(data["VIX"].max()),
        "hft_off_days": int((~data["HFT_aktiv"]).sum()),
        "cb_interventions": int(data["Zentralbank_Eingriff"].sum()),
        "jump_events": int(data["Sprungereignis"].sum()),
        "mean_abs_exogenous": float(component_abs_means["Exogener_Beitrag"]),
        "mean_abs_orderflow": float(component_abs_means["Orderflow_Beitrag"]),
        "mean_abs_cb": float(component_abs_means["Zentralbank_Beitrag"]),
        "exogenous_share": float(component_abs_means["Exogener_Beitrag"] / component_total),
        "orderflow_share": float(component_abs_means["Orderflow_Beitrag"] / component_total),
        "cb_share": float(component_abs_means["Zentralbank_Beitrag"] / component_total),
        "mean_orderflow_hft_off": mean_orderflow_hft_off,
        "mean_orderflow_hft_on": mean_orderflow_hft_on,
    }
    return SimulationResult(params=params, data=data, metrics=metrics)


def _clip(value: float, minimum: float, maximum: float) -> float:
    return float(min(maximum, max(minimum, value)))


def paired_sensitivity(
    base_params: ModelParams,
    runs: int = 30,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Vergleicht niedrige und hohe Parameterwerte mit identischen Seeds.

    Durch Common Random Numbers wird bei jedem Paar exakt derselbe Zufallspfad
    benutzt. Die Differenz ist daher wesentlich besser dem geänderten Parameter
    zuzuordnen als bei unverbundenen Zufallsläufen.
    """

    if runs < 5:
        raise ValueError("runs muss mindestens 5 sein")

    rows: list[dict[str, float | str]] = []
    total_jobs = len(SENSITIVITY_SPECS) * runs * 2
    completed = 0

    for spec in SENSITIVITY_SPECS:
        current_value = float(getattr(base_params, spec.parameter))
        low_value = _clip(current_value - spec.delta, spec.minimum, spec.maximum)
        high_value = _clip(current_value + spec.delta, spec.minimum, spec.maximum)

        low_returns: list[float] = []
        high_returns: list[float] = []

        for index in range(runs):
            paired_seed = base_params.seed + 10_000 + index
            low_overrides: dict[str, float | int] = {spec.parameter: low_value}
            high_overrides: dict[str, float | int] = {spec.parameter: high_value}
            if spec.parameter == "fund_leverage_limit":
                low_overrides["fund_start"] = min(base_params.fund_start, low_value)
                high_overrides["fund_start"] = min(base_params.fund_start, high_value)

            low_params = replace(
                base_params,
                seed=paired_seed,
                **low_overrides,
            )
            high_params = replace(
                base_params,
                seed=paired_seed,
                **high_overrides,
            )

            low_result = simulate(low_params)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total_jobs)

            high_result = simulate(high_params)
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total_jobs)

            low_returns.append(float(low_result.metrics["final_return"]))
            high_returns.append(float(high_result.metrics["final_return"]))

        differences = np.asarray(high_returns) - np.asarray(low_returns)
        rows.append(
            {
                "Parameter": spec.label,
                "Technischer_Name": spec.parameter,
                "Niedriger_Wert": low_value,
                "Hoher_Wert": high_value,
                "Rendite_niedrig_Mittel": float(np.mean(low_returns)),
                "Rendite_hoch_Mittel": float(np.mean(high_returns)),
                "Mittlerer_Effekt": float(np.mean(differences)),
                "Median_Effekt": float(np.median(differences)),
                "P10_Effekt": float(np.quantile(differences, 0.10)),
                "P90_Effekt": float(np.quantile(differences, 0.90)),
                "Anteil_positiver_Effekte": float(np.mean(differences > 0.0)),
            }
        )

    return pd.DataFrame(rows).sort_values("Mittlerer_Effekt", ascending=False)
