"""Streamlit-Oberfläche für den überarbeiteten Marktpsychologie-Simulator."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from market_model import (
    DEFAULT_PARAMS,
    PARAMETER_NAMES,
    PRESETS,
    ModelParams,
    paired_sensitivity,
    simulate,
)


st.set_page_config(
    page_title="Marktpsychologie-Simulator 2.0",
    page_icon="🧠",
    layout="wide",
)


def initialize_state() -> None:
    if "scenario" not in st.session_state:
        st.session_state["scenario"] = "Benutzerdefiniert"
    selected = st.session_state["scenario"]
    initial_values = PRESETS.get(selected, PRESETS["Benutzerdefiniert"])
    for name in PARAMETER_NAMES:
        if name not in st.session_state:
            st.session_state[name] = initial_values[name]


def apply_selected_scenario() -> None:
    selected = st.session_state["scenario"]
    values = PRESETS[selected]
    for name in PARAMETER_NAMES:
        st.session_state[name] = values[name]
    st.session_state.pop("last_result", None)
    st.session_state.pop("sensitivity_result", None)


def current_params() -> ModelParams:
    values = {name: st.session_state[name] for name in PARAMETER_NAMES}
    values["days"] = int(values["days"])
    values["seed"] = int(values["seed"])
    return ModelParams(**values)


def format_percent(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f} %"


def render_model_explanation() -> None:
    with st.expander("Wie das Modell den Kurs berechnet"):
        st.markdown(
            """
            Für jeden Tag wird die Rendite in vier exakt gespeicherte Komponenten zerlegt:

            1. **Exogener Beitrag:** Drift, normales Rauschen und seltene Sprünge.
            2. **Orderflow-Beitrag:** Käufe und Verkäufe von Privatanlegern und Fonds.
            3. **Zentralbank-Beitrag:** Nur an tatsächlich protokollierten Interventionstagen.
            4. **Cap-Anpassung:** Differenz, falls das endgültige Tageslimit greift.

            Die HFTs erhöhen die **effektive Markttiefe**. Derselbe Verkaufsauftrag bewegt
            den Kurs bei aktiven HFTs deshalb weniger stark als nach einer Abschaltung.
            Fondsabflüsse werden als echte negative Orders modelliert und reduzieren zugleich
            das verbleibende Fondsvermögen.
            """
        )
        st.code(
            "Tagesrendite = clip(exogener Beitrag + Orderflow + Zentralbank, ± Tageslimit)",
            language="text",
        )


def render_sidebar() -> bool:
    st.sidebar.header("⚙️ Steuerungszentrale")
    st.sidebar.selectbox(
        "Szenario",
        options=list(PRESETS),
        key="scenario",
        on_change=apply_selected_scenario,
    )

    is_preset = st.session_state["scenario"] != "Benutzerdefiniert"
    if is_preset:
        st.sidebar.info(
            "Das Preset setzt alle Modellparameter vollständig. Tage und Seed bleiben editierbar."
        )

    with st.sidebar.form("parameter_form"):
        with st.expander("0. Laufsteuerung", expanded=True):
            st.slider("Simulationslänge", 100, 2_000, step=50, key="days")
            st.number_input("Zufalls-Seed", 0, 2_147_483_647, step=1, key="seed")

        with st.expander("1. Exogenes Marktumfeld", expanded=True):
            st.slider(
                "Fundamentale Tagesdrift",
                -0.0010,
                0.0010,
                step=0.00005,
                key="fundamental_drift",
                format="%.5f",
                disabled=is_preset,
            )
            st.slider(
                "Normale Tagesvolatilität",
                0.0000,
                0.0300,
                step=0.0005,
                key="fundamental_volatility",
                format="%.4f",
                disabled=is_preset,
            )
            st.slider(
                "Sprungwahrscheinlichkeit",
                0.000,
                0.100,
                step=0.001,
                key="jump_probability",
                format="%.3f",
                disabled=is_preset,
            )
            st.slider(
                "Mittlere Sprungrendite",
                -0.100,
                0.100,
                step=0.005,
                key="jump_mean",
                format="%.3f",
                disabled=is_preset,
            )
            st.slider(
                "Sprungvolatilität",
                0.000,
                0.150,
                step=0.005,
                key="jump_volatility",
                format="%.3f",
                disabled=is_preset,
            )
            st.slider(
                "Hartes Tageslimit",
                0.010,
                0.300,
                step=0.010,
                key="daily_return_cap",
                format="%.3f",
                disabled=is_preset,
            )
            st.slider(
                "Orderflow-Preiswirkung",
                0.000,
                0.300,
                step=0.005,
                key="orderflow_impact",
                format="%.3f",
                disabled=is_preset,
            )

        with st.expander("2. Privatanleger"):
            st.slider("Start-Aktienquote", 0.0, 1.0, step=0.05, key="retail_start", disabled=is_preset)
            st.slider(
                "Gier-Schwelle",
                0.005,
                0.150,
                step=0.005,
                key="retail_greed_threshold",
                disabled=is_preset,
            )
            st.slider(
                "Panik-Schwelle",
                -0.200,
                -0.005,
                step=0.005,
                key="retail_panic_threshold",
                disabled=is_preset,
            )
            st.slider("Panikverkauf", 0.0, 0.80, step=0.05, key="retail_panic_sell", disabled=is_preset)
            st.slider("Gierkauf", 0.0, 0.50, step=0.02, key="retail_greed_buy", disabled=is_preset)
            st.slider(
                "Rückkehr zur Normalquote",
                0.0,
                0.20,
                step=0.01,
                key="retail_reversion",
                disabled=is_preset,
            )

        with st.expander("3. Fonds"):
            st.slider("Start-Hebel", 0.30, 2.50, step=0.05, key="fund_start", disabled=is_preset)
            st.slider(
                "Maximaler Hebel",
                0.30,
                2.50,
                step=0.05,
                key="fund_leverage_limit",
                disabled=is_preset,
            )
            st.slider("VIX-Schwelle", 15.0, 70.0, step=1.0, key="fund_vix_threshold", disabled=is_preset)
            st.slider("Tägliche Abflussrate", 0.0, 0.60, step=0.01, key="fund_outflow_rate", disabled=is_preset)
            st.slider("Trendfolge-Schritt", 0.0, 0.30, step=0.01, key="fund_trend_step", disabled=is_preset)
            st.slider(
                "Deleveraging-Schritt",
                0.0,
                0.50,
                step=0.01,
                key="fund_deleverage_step",
                disabled=is_preset,
            )

        with st.expander("4. HFT / Liquidität"):
            st.number_input(
                "HFT-Kapital",
                0.0,
                50_000.0,
                step=500.0,
                key="hft_capital",
                disabled=is_preset,
            )
            st.slider(
                "HFT-Abschaltung bei VIX",
                15.0,
                90.0,
                step=1.0,
                key="hft_vix_shutdown",
                disabled=is_preset,
            )
            st.slider(
                "Liquidität pro Kapitaleinheit",
                0.0,
                0.20,
                step=0.005,
                key="hft_liquidity_factor",
                disabled=is_preset,
            )

        with st.expander("5. Zentralbank"):
            st.slider(
                "Eingriffsschwelle (5-Tage-Verlust)",
                0.01,
                0.40,
                step=0.01,
                key="cb_intervention_threshold",
                disabled=is_preset,
            )
            st.slider(
                "Kursimpuls beim Eingriff",
                0.0,
                0.15,
                step=0.005,
                key="cb_purchase_return",
                disabled=is_preset,
            )
            st.slider(
                "VIX-Reduktion",
                0.0,
                0.80,
                step=0.05,
                key="cb_vix_reduction",
                disabled=is_preset,
            )

        return st.form_submit_button("🚀 Simulation starten", type="primary")


def render_metrics(result) -> None:
    data = result.data
    metrics = result.metrics
    columns = st.columns(6)
    columns[0].metric("Endkurs", f"{data['Kurs'].iloc[-1]:.2f}")
    columns[1].metric("Gesamtrendite", format_percent(float(metrics["final_return"])))
    columns[2].metric("CAGR", format_percent(float(metrics["cagr"])))
    columns[3].metric("Max. Drawdown", format_percent(float(metrics["max_drawdown"])))
    columns[4].metric("Max. VIX", f"{float(metrics['max_vix']):.1f}")
    columns[5].metric("CB-Eingriffe", f"{int(metrics['cb_interventions'])}")


def render_overview(result) -> None:
    data = result.data
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Aktienkurs", "VIX und HFT-Abschaltschwelle"),
    )
    fig.add_trace(go.Scatter(x=data["Tag"], y=data["Kurs"], name="Kurs"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data["Tag"], y=data["VIX"], name="VIX"), row=2, col=1)
    fig.add_hline(
        y=result.params.hft_vix_shutdown,
        line_dash="dash",
        annotation_text="HFT-Abschaltung",
        row=2,
        col=1,
    )
    fig.update_layout(height=650, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    agent_fig = make_subplots(specs=[[{"secondary_y": True}]])
    agent_fig.add_trace(
        go.Scatter(x=data["Tag"], y=data["Retail_Quote"], name="Retail-Quote"),
        secondary_y=False,
    )
    agent_fig.add_trace(
        go.Scatter(x=data["Tag"], y=data["Fonds_Hebel"], name="Fonds-Hebel"),
        secondary_y=False,
    )
    agent_fig.add_trace(
        go.Scatter(x=data["Tag"], y=data["Effektive_Markttiefe"], name="Markttiefe"),
        secondary_y=True,
    )
    agent_fig.update_yaxes(title_text="Quote / Hebel", secondary_y=False)
    agent_fig.update_yaxes(title_text="Markttiefe", secondary_y=True)
    agent_fig.update_layout(height=430, hovermode="x unified")
    st.plotly_chart(agent_fig, width="stretch")


def render_causal_analysis(result) -> None:
    data = result.data.copy()
    metrics = result.metrics

    st.subheader("Gemessene Beitragsstärken")
    contribution_df = pd.DataFrame(
        {
            "Komponente": ["Exogen", "Orderflow", "Zentralbank"],
            "Anteil": [
                metrics["exogenous_share"],
                metrics["orderflow_share"],
                metrics["cb_share"],
            ],
        }
    )
    fig = px.bar(contribution_df, x="Komponente", y="Anteil", text_auto=".1%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    data["Kumuliert exogen"] = data["Exogener_Beitrag"].cumsum()
    data["Kumuliert Orderflow"] = data["Orderflow_Beitrag"].cumsum()
    data["Kumuliert Zentralbank"] = data["Zentralbank_Beitrag"].cumsum()
    cumulative = data.melt(
        id_vars="Tag",
        value_vars=["Kumuliert exogen", "Kumuliert Orderflow", "Kumuliert Zentralbank"],
        var_name="Komponente",
        value_name="Kumulierte additive Rendite",
    )
    st.plotly_chart(
        px.line(cumulative, x="Tag", y="Kumulierte additive Rendite", color="Komponente"),
        width="stretch",
    )

    exogenous = float(metrics["mean_abs_exogenous"])
    orderflow = float(metrics["mean_abs_orderflow"])
    ratio = orderflow / exogenous if exogenous > 0 else np.inf

    st.markdown("### Datenbasierte Interpretation")
    if ratio < 0.5:
        st.warning(
            f"Der exogene Prozess dominiert: Der mittlere absolute Orderflow-Beitrag beträgt "
            f"nur {ratio:.2f}× des exogenen Beitrags. Die Psychologie ist in diesem Lauf "
            "zwar wirksam, aber nicht der Haupttreiber."
        )
    elif ratio <= 2.0:
        st.success(
            f"Exogener Prozess und Teilnehmerverhalten sind ähnlich relevant "
            f"(Orderflow/Exogen = {ratio:.2f})."
        )
    else:
        st.warning(
            f"Der Orderflow dominiert mit dem Faktor {ratio:.2f}. Das Szenario ist stark "
            "endogen und kann selbstverstärkende Bewegungen erzeugen."
        )

    off_days = int(metrics["hft_off_days"])
    on_impact = float(metrics["mean_orderflow_hft_on"])
    off_impact = float(metrics["mean_orderflow_hft_off"])
    if off_days > 0:
        multiplier = off_impact / on_impact if on_impact > 0 else np.inf
        st.info(
            f"HFTs waren {off_days} Tage abgeschaltet. Der mittlere absolute "
            f"Orderflow-Kurseffekt lag an Aus-Tagen beim {multiplier:.2f}-Fachen der "
            "Aktiv-Tage. Diese Aussage wird direkt aus den simulierten Komponenten berechnet."
        )
    else:
        st.info("In diesem Lauf blieben die HFTs durchgehend aktiv.")


def render_events(result) -> None:
    data = result.data
    event_mask = (
        data["Sprungereignis"]
        | data["Zentralbank_Eingriff"]
        | (~data["HFT_aktiv"])
        | (data["Cap_Anpassung"].abs() > 1e-12)
    )
    events = data.loc[
        event_mask,
        [
            "Tag",
            "Kurs",
            "Rendite",
            "VIX",
            "Sprungereignis",
            "Sprung_Beitrag",
            "HFT_aktiv",
            "Fonds_Zwangsverkauf",
            "Zentralbank_Eingriff",
            "Zentralbank_Beitrag",
            "Cap_Anpassung",
        ],
    ].copy()
    if events.empty:
        st.info("Keine besonderen Ereignisse in diesem Lauf.")
    else:
        st.dataframe(
            events.style.format(
                {
                    "Kurs": "{:.2f}",
                    "Rendite": "{:.2%}",
                    "VIX": "{:.1f}",
                    "Sprung_Beitrag": "{:.2%}",
                    "Fonds_Zwangsverkauf": "{:.1f}",
                    "Zentralbank_Beitrag": "{:.2%}",
                    "Cap_Anpassung": "{:.2%}",
                }
            ),
            width="stretch",
            height=500,
        )


def render_exports(result) -> None:
    csv = result.data.to_csv(index=False).encode("utf-8")
    params_json = json.dumps(asdict(result.params), indent=2, ensure_ascii=False).encode("utf-8")
    col1, col2 = st.columns(2)
    col1.download_button(
        "📥 Simulationsdaten als CSV",
        data=csv,
        file_name="simulation_v2.csv",
        mime="text/csv",
    )
    col2.download_button(
        "📥 Parameter als JSON",
        data=params_json,
        file_name="simulation_parameter_v2.json",
        mime="application/json",
    )


def render_sensitivity(result) -> None:
    st.markdown(
        "Die niedrige und hohe Variante jedes Parameters verwendet paarweise denselben "
        "Zufalls-Seed. Dadurch misst der Test primär die Parameterwirkung und nicht neue Zufallspfade."
    )
    runs = st.slider("Gepaarte Läufe je Parameter", 10, 100, 30, 10)
    if st.button("🧪 Gepaarte Sensitivitätsanalyse starten"):
        progress = st.progress(0.0, text="Sensitivitätsanalyse wird vorbereitet …")

        def update_progress(done: int, total: int) -> None:
            progress.progress(done / total, text=f"Berechnung {done} von {total}")

        sensitivity = paired_sensitivity(result.params, runs=runs, progress_callback=update_progress)
        progress.empty()
        st.session_state["sensitivity_result"] = sensitivity

    sensitivity = st.session_state.get("sensitivity_result")
    if sensitivity is not None:
        display = sensitivity.copy()
        percent_columns = [
            "Rendite_niedrig_Mittel",
            "Rendite_hoch_Mittel",
            "Mittlerer_Effekt",
            "Median_Effekt",
            "P10_Effekt",
            "P90_Effekt",
            "Anteil_positiver_Effekte",
        ]
        st.dataframe(
            display.style.format({column: "{:.2%}" for column in percent_columns}),
            width="stretch",
        )
        fig = px.bar(
            sensitivity,
            x="Parameter",
            y="Mittlerer_Effekt",
            error_y=sensitivity["P90_Effekt"] - sensitivity["Mittlerer_Effekt"],
            error_y_minus=sensitivity["Mittlerer_Effekt"] - sensitivity["P10_Effekt"],
            title="Mittlere Renditeänderung: hoher minus niedriger Parameterwert",
        )
        fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, width="stretch")


initialize_state()

st.title("🧠 Marktpsychologie-Simulator 2.0")
st.caption(
    "Reproduzierbares Agentenmodell mit expliziter Renditezerlegung, echter HFT-Liquidität, "
    "Fonds-Zwangsverkäufen und protokollierten Zentralbankinterventionen."
)
render_model_explanation()
submitted = render_sidebar()

if submitted:
    try:
        params = current_params()
        progress = st.progress(0.0, text="Simulation wird vorbereitet …")

        def update_progress(day: int, total: int) -> None:
            if day == total or day % max(1, total // 100) == 0:
                progress.progress(day / total, text=f"Tag {day} von {total}")

        result = simulate(params, progress_callback=update_progress)
        progress.empty()
        st.session_state["last_result"] = result
        st.session_state.pop("sensitivity_result", None)
        st.success("Simulation abgeschlossen und vollständig protokolliert.")
    except ValueError as exc:
        st.error(f"Ungültige Parameterkombination: {exc}")

result = st.session_state.get("last_result")
if result is None:
    st.info("Wähle ein Szenario oder passe die manuellen Parameter an und starte die Simulation.")
else:
    render_metrics(result)
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Verlauf", "🔍 Ursachen", "📋 Ereignisse", "🧪 Sensitivität"]
    )
    with tab1:
        render_overview(result)
    with tab2:
        render_causal_analysis(result)
    with tab3:
        render_events(result)
    with tab4:
        render_sensitivity(result)
    render_exports(result)
