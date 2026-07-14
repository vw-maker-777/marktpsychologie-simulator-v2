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
    page_title="Marktpsychologie-Simulator 2.2",
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

PRESET_DESCRIPTIONS = {
    "Benutzerdefiniert": (
        "Alle Regler sind frei einstellbar. Diese Variante eignet sich zum Experimentieren."
    ),
    "1. Panik-Crash / Liquiditätsentzug": (
        "Anleger reagieren schnell panisch, Fonds müssen verkaufen und HFT-Liquidität "
        "verschwindet früh. Das Szenario ist bewusst krisenanfällig."
    ),
    "2. Aggressive Zentralbank": (
        "Die Zentralbank greift schon bei kleinen Verlusten ein und stützt den Kurs deutlich."
    ),
    "3. Gefährlicher Fondshebel": (
        "Fonds arbeiten mit hohem Hebel. Gewinne können verstärkt werden, Verluste und "
        "Zwangsverkäufe aber ebenfalls."
    ),
    "4. Ruhiger Aufwärtstrend": (
        "Niedrige Schwankungen, robuste Liquidität und zurückhaltende Marktteilnehmer "
        "begünstigen einen stabileren Verlauf."
    ),
}


def explain_slider(
    left: str,
    right: str,
    effect: str,
    current: str | None = None,
    warning: str | None = None,
) -> None:
    """Zeigt eine einsteigerfreundliche Erklärung direkt unter einem Regler."""

    if not st.session_state.get("beginner_mode", True):
        return
    st.caption(f"⬅️ **Links:** {left}  ·  **Rechts:** {right} ➡️")
    st.caption(f"**Typische Wirkung:** {effect}")
    if current:
        st.caption(f"**Aktuell:** {current}")
    if warning:
        st.warning(warning)


def market_regime_text(params: ModelParams) -> list[str]:
    """Übersetzt die wichtigsten Parameter in verständliche Alltagssprache."""

    if params.fundamental_drift > 0.00015:
        drift = "Das Marktumfeld besitzt einen leichten positiven Grundtrend."
    elif params.fundamental_drift < -0.00015:
        drift = "Das Marktumfeld besitzt einen negativen Grundtrend."
    else:
        drift = "Das Marktumfeld hat nahezu keinen Grundtrend."

    if params.fundamental_volatility < 0.003:
        vola = "Die normalen täglichen Schwankungen sind niedrig."
    elif params.fundamental_volatility < 0.008:
        vola = "Die normalen täglichen Schwankungen sind mittel."
    else:
        vola = "Die normalen täglichen Schwankungen sind hoch."

    if params.retail_panic_threshold > -0.03 or params.retail_panic_sell > 0.35:
        retail = "Privatanleger sind panikanfällig und können Rückgänge deutlich verstärken."
    elif params.retail_panic_threshold < -0.08 and params.retail_panic_sell < 0.15:
        retail = "Privatanleger reagieren vergleichsweise gelassen auf Kursverluste."
    else:
        retail = "Privatanleger reagieren moderat auf Angst und Gier."

    if params.fund_leverage_limit >= 1.6:
        funds = "Fonds dürfen einen hohen Hebel nutzen; Gewinne und Verluste werden verstärkt."
    elif params.fund_leverage_limit <= 1.1:
        funds = "Fonds sind defensiv eingestellt und nutzen kaum zusätzlichen Hebel."
    else:
        funds = "Fonds nutzen einen mittleren Hebel."

    if params.hft_capital * params.hft_liquidity_factor > 700 and params.hft_vix_shutdown >= 50:
        hft = "Die Marktliquidität ist robust; HFTs bleiben auch bei größerer Angst aktiv."
    elif params.hft_vix_shutdown <= 30:
        hft = "HFTs schalten früh ab; in Stressphasen kann die Liquidität schnell sinken."
    else:
        hft = "Die HFT-Liquidität ist mittelstark."

    if params.cb_intervention_threshold <= 0.05:
        cb = "Die Zentralbank greift sehr früh ein und begrenzt kleinere Rückgänge."
    elif params.cb_intervention_threshold >= 0.18:
        cb = "Die Zentralbank wartet lange und greift erst bei schweren Verlusten ein."
    else:
        cb = "Die Zentralbank greift bei deutlichen, aber noch nicht extremen Verlusten ein."

    return [drift, vola, retail, funds, hft, cb]


def render_plain_language_summary(params: ModelParams) -> None:
    with st.expander("🧭 Was bedeuten diese Einstellungen in Alltagssprache?", expanded=True):
        st.markdown("\n".join(f"- {line}" for line in market_regime_text(params)))
        st.caption(
            "Diese Aussagen beschreiben die eingestellten Mechanismen. Sie garantieren kein "
            "bestimmtes Ergebnis, weil der Zufallspfad weiterhin eine Rolle spielt."
        )


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
    st.sidebar.toggle(
        "🧭 Einsteiger-Erklärungen anzeigen",
        value=True,
        key="beginner_mode",
        help="Blendet unter jedem Regler eine Erklärung für die linke und rechte Richtung ein.",
    )
    st.sidebar.selectbox(
        "Szenario",
        options=list(PRESETS),
        key="scenario",
        on_change=apply_selected_scenario,
        help="Ein Szenario lädt eine vollständige, in sich abgestimmte Parameterkombination.",
    )
    st.sidebar.info(PRESET_DESCRIPTIONS[st.session_state["scenario"]])

    is_preset = st.session_state["scenario"] != "Benutzerdefiniert"
    if is_preset:
        st.sidebar.caption(
            "🔒 Im Preset sind die Modellregler gesperrt. Simulationslänge und Seed bleiben editierbar."
        )

    with st.sidebar:
        if st.session_state.get("beginner_mode", True):
            st.markdown(
                "**So liest du die Regler:** Links bedeutet den kleineren Wert, rechts den "
                "größeren Wert. Entscheidend ist nicht immer ›gut oder schlecht‹, sondern "
                "welcher Mechanismus dadurch stärker oder schwächer wird."
            )

        with st.expander("0. Laufsteuerung", expanded=True):
            st.slider(
                "Simulationslänge (Tage)",
                100,
                2_000,
                step=50,
                key="days",
                help="Anzahl der simulierten Handelstage. Rund 250 Tage entsprechen ungefähr einem Börsenjahr.",
            )
            explain_slider(
                "kürzerer Lauf, weniger Ereignisse",
                "längerer Lauf, mehr Krisen- und Erholungsphasen möglich",
                "Die Länge verändert nicht die Regeln, sondern wie lange sie beobachtet werden.",
                f"{int(st.session_state['days'])} Tage ≈ {st.session_state['days'] / 250:.1f} Börsenjahre",
            )

            st.number_input(
                "Zufalls-Seed",
                0,
                2_147_483_647,
                step=1,
                key="seed",
                help="Gleicher Seed plus gleiche Parameter erzeugen denselben Zufallspfad.",
            )
            explain_slider(
                "anderer Zufallspfad",
                "anderer Zufallspfad",
                "Der Seed verändert nur die konkrete Folge zufälliger Ereignisse, nicht die Modellregeln.",
                f"Seed {int(st.session_state['seed'])}",
            )

        with st.expander("1. Exogenes Marktumfeld", expanded=True):
            st.slider(
                "Fundamentale Tagesdrift",
                -0.0010,
                0.0010,
                step=0.00005,
                key="fundamental_drift",
                format="%.5f",
                disabled=is_preset,
                help="Langfristiger täglicher Grundtrend vor Psychologie und Zufallsschwankungen.",
            )
            explain_slider(
                "negativer Grundtrend",
                "positiver Grundtrend",
                "Mehr Drift verschiebt die langfristige Tendenz nach oben; weniger Drift nach unten.",
                format_percent(float(st.session_state['fundamental_drift']), 3) + " pro Tag",
            )

            st.slider(
                "Normale Tagesvolatilität",
                0.0000,
                0.0300,
                step=0.0005,
                key="fundamental_volatility",
                format="%.4f",
                disabled=is_preset,
                help="Stärke des normalen täglichen Zufallsrauschens.",
            )
            explain_slider(
                "ruhigere Kursbewegungen",
                "stärkere tägliche Ausschläge",
                "Höhere Volatilität erhöht sowohl Gewinn- als auch Verlustrisiken und treibt meist den VIX.",
                format_percent(float(st.session_state['fundamental_volatility']), 2) + " typische Tagesstreuung",
            )

            st.slider(
                "Sprungwahrscheinlichkeit",
                0.000,
                0.100,
                step=0.001,
                key="jump_probability",
                format="%.3f",
                disabled=is_preset,
                help="Wahrscheinlichkeit eines außergewöhnlichen Sprungs an einem einzelnen Tag.",
            )
            explain_slider(
                "seltene Sonderschocks",
                "häufige Sonderschocks",
                "Der Regler bestimmt, wie oft außergewöhnliche Ereignisse auftreten, nicht deren Richtung.",
                format_percent(float(st.session_state['jump_probability']), 1) + " pro Tag",
            )

            st.slider(
                "Mittlere Sprungrendite",
                -0.100,
                0.100,
                step=0.005,
                key="jump_mean",
                format="%.3f",
                disabled=is_preset,
                help="Durchschnittliche Richtung seltener Sprungereignisse.",
            )
            explain_slider(
                "Sprünge sind im Mittel negativ",
                "Sprünge sind im Mittel positiv",
                "Negative Werte erzeugen einen Crash-Bias; positive Werte einen Rallye-Bias.",
                format_percent(float(st.session_state['jump_mean']), 1) + " mittlere Sprungrendite",
            )

            st.slider(
                "Sprungvolatilität",
                0.000,
                0.150,
                step=0.005,
                key="jump_volatility",
                format="%.3f",
                disabled=is_preset,
                help="Streuung und Unberechenbarkeit der seltenen Sprünge.",
            )
            explain_slider(
                "ähnlich große Sprünge",
                "sehr unterschiedlich große Sprünge",
                "Höhere Werte machen Extremereignisse unberechenbarer und können sehr große Ausschläge erzeugen.",
                format_percent(float(st.session_state['jump_volatility']), 1) + " Sprungstreuung",
            )

            st.slider(
                "Hartes Tageslimit",
                0.010,
                0.300,
                step=0.010,
                key="daily_return_cap",
                format="%.3f",
                disabled=is_preset,
                help="Maximal erlaubte absolute Tagesrendite nach Addition aller Modellkomponenten.",
            )
            explain_slider(
                "Tagesbewegungen werden früh begrenzt",
                "größere Tagesbewegungen sind erlaubt",
                "Ein niedriges Limit dämpft Extremtage künstlich; ein hohes Limit lässt das Modell freier laufen.",
                "maximal ±" + format_percent(float(st.session_state['daily_return_cap']), 1) + " pro Tag",
            )

            st.slider(
                "Orderflow-Preiswirkung",
                0.000,
                0.300,
                step=0.005,
                key="orderflow_impact",
                format="%.3f",
                disabled=is_preset,
                help="Wie stark Käufe und Verkäufe der simulierten Anleger den Kurs bewegen.",
            )
            explain_slider(
                "Psychologie bewegt den Kurs kaum",
                "Käufe und Verkäufe bewegen den Kurs stark",
                "Dieser Regler bestimmt, wie wichtig das Verhalten der Marktteilnehmer gegenüber dem Zufall ist.",
                f"Wirkungsfaktor {float(st.session_state['orderflow_impact']):.3f}",
                warning=(
                    "Bei einem Wert nahe null wird aus dem Psychologie-Simulator überwiegend ein Zufallsmodell."
                    if float(st.session_state['orderflow_impact']) < 0.02
                    else None
                ),
            )

        with st.expander("2. Privatanleger"):
            st.slider(
                "Start-Aktienquote",
                0.0,
                1.0,
                step=0.05,
                key="retail_start",
                disabled=is_preset,
                help="Anteil des vorgesehenen Privatanlegerkapitals, der zu Beginn in Aktien investiert ist.",
            )
            explain_slider(
                "viel Bargeld, wenig Aktien",
                "fast vollständig investiert",
                "Eine hohe Startquote bedeutet mehr anfängliche Aktiennachfrage, aber weniger spätere Kaufreserve.",
                format_percent(float(st.session_state['retail_start']), 0) + " investiert",
            )

            st.slider(
                "Gier-Schwelle (5-Tage-Anstieg)",
                0.005,
                0.150,
                step=0.005,
                key="retail_greed_threshold",
                disabled=is_preset,
                help="Ab welchem 5-Tage-Anstieg Privatanleger zusätzliche Aktien kaufen.",
            )
            explain_slider(
                "Gier wird schon bei kleinen Anstiegen ausgelöst",
                "erst starke Rallyes lösen Gier aus",
                "Weiter links kaufen Privatanleger häufiger prozyklisch; weiter rechts reagieren sie seltener.",
                "Kaufreaktion ab +" + format_percent(float(st.session_state['retail_greed_threshold']), 1),
            )

            st.slider(
                "Panik-Schwelle (5-Tage-Verlust)",
                -0.200,
                -0.005,
                step=0.005,
                key="retail_panic_threshold",
                disabled=is_preset,
                help="Ab welchem 5-Tage-Verlust Privatanleger panisch verkaufen.",
            )
            explain_slider(
                "Panik erst nach sehr großem Verlust",
                "Panik schon nach kleinem Verlust",
                "Achtung: Weil die Werte negativ sind, bedeutet weiter rechts eine empfindlichere Panikreaktion.",
                "Verkaufsreaktion ab " + format_percent(float(st.session_state['retail_panic_threshold']), 1),
            )

            st.slider(
                "Panikverkauf (Quotenabbau)",
                0.0,
                0.80,
                step=0.05,
                key="retail_panic_sell",
                disabled=is_preset,
                help="Um wie viel die Aktienquote bei einer Panikreaktion reduziert wird.",
            )
            explain_slider(
                "kleiner Verkauf",
                "massiver Verkauf",
                "Ein hoher Wert erzeugt bei Panik starken Verkaufsdruck und kann Abwärtsbewegungen verstärken.",
                format_percent(float(st.session_state['retail_panic_sell']), 0) + " Quotenabbau je Panikreaktion",
            )

            st.slider(
                "Gierkauf (Quotenaufbau)",
                0.0,
                0.50,
                step=0.02,
                key="retail_greed_buy",
                disabled=is_preset,
                help="Um wie viel die Aktienquote bei einer Gierreaktion erhöht wird.",
            )
            explain_slider(
                "kleiner Zusatzkauf",
                "massiver Zusatzkauf",
                "Ein hoher Wert verstärkt Rallyes, kann aber auch Blasenbildung fördern.",
                format_percent(float(st.session_state['retail_greed_buy']), 0) + " Quotenaufbau je Gierreaktion",
            )

            st.slider(
                "Rückkehr zur Normalquote",
                0.0,
                0.20,
                step=0.01,
                key="retail_reversion",
                disabled=is_preset,
                help="Geschwindigkeit, mit der die Aktienquote ohne Panik oder Gier zum Ausgangswert zurückkehrt.",
            )
            explain_slider(
                "veränderte Stimmung hält lange an",
                "schnelle Rückkehr zum Ausgangszustand",
                "Ein hoher Wert stabilisiert die Anlegerquote nach Schocks schneller.",
                format_percent(float(st.session_state['retail_reversion']), 0) + " Rückkehr pro ruhigem Tag",
            )

        with st.expander("3. Fonds"):
            st.slider(
                "Start-Hebel",
                0.30,
                2.50,
                step=0.05,
                key="fund_start",
                disabled=is_preset,
                help="Anfängliche Aktienexponierung der Fonds. 1,0 entspricht ungefähr ungehebelt.",
            )
            explain_slider(
                "defensiv oder unterinvestiert",
                "stark gehebelt",
                "Ein höherer Starthebel verstärkt sowohl Gewinne als auch Verluste der Fondspositionen.",
                f"{float(st.session_state['fund_start']):.2f}× Exponierung",
            )

            st.slider(
                "Maximaler Hebel",
                0.30,
                2.50,
                step=0.05,
                key="fund_leverage_limit",
                disabled=is_preset,
                help="Obergrenze für die Fondsexponierung.",
            )
            explain_slider(
                "Fonds werden früh begrenzt",
                "Fonds dürfen sehr große Risiken eingehen",
                "Ein hohes Limit erhöht die mögliche prozyklische Verstärkung und das Deleveraging-Risiko.",
                f"maximal {float(st.session_state['fund_leverage_limit']):.2f}×",
                warning=(
                    "Der Start-Hebel darf nicht über dem maximalen Hebel liegen."
                    if float(st.session_state['fund_start']) > float(st.session_state['fund_leverage_limit'])
                    else None
                ),
            )

            st.slider(
                "VIX-Schwelle für Fondsstress",
                15.0,
                70.0,
                step=1.0,
                key="fund_vix_threshold",
                disabled=is_preset,
                help="Ab welchem VIX Fonds Mittelabflüsse und zusätzliches Deleveraging erleben.",
            )
            explain_slider(
                "Fonds geraten schon bei geringer Angst unter Druck",
                "erst extreme Angst löst Fondsstress aus",
                "Eine niedrige Schwelle führt häufiger zu Abflüssen und Zwangsverkäufen.",
                f"Stress ab VIX {float(st.session_state['fund_vix_threshold']):.0f}",
            )

            st.slider(
                "Tägliche Abflussrate",
                0.0,
                0.60,
                step=0.01,
                key="fund_outflow_rate",
                disabled=is_preset,
                help="Anteil des Fondsvermögens, den Anleger an einem Stresstag abziehen.",
            )
            explain_slider(
                "kaum Mittelabflüsse",
                "sehr starke Mittelabflüsse",
                "Höhere Abflüsse erzwingen größere Verkäufe und reduzieren das Fondsvermögen schneller.",
                format_percent(float(st.session_state['fund_outflow_rate']), 0) + " des Fondsvermögens je Stresstag",
            )

            st.slider(
                "Trendfolge-Schritt",
                0.0,
                0.30,
                step=0.01,
                key="fund_trend_step",
                disabled=is_preset,
                help="Wie stark Fonds ihre Aktienexponierung nach einer Rallye erhöhen.",
            )
            explain_slider(
                "Fonds folgen Trends kaum",
                "Fonds kaufen Rallyes aggressiv",
                "Ein hoher Wert verstärkt Aufwärtstrends, erhöht aber späteres Rückschlagrisiko.",
                f"+{float(st.session_state['fund_trend_step']):.2f} Hebelschritt nach starkem Anstieg",
            )

            st.slider(
                "Deleveraging-Schritt",
                0.0,
                0.50,
                step=0.01,
                key="fund_deleverage_step",
                disabled=is_preset,
                help="Wie stark Fonds nach Verlusten oder bei hohem VIX ihre Exponierung abbauen.",
            )
            explain_slider(
                "langsamer Risikoabbau",
                "aggressiver Risikoabbau",
                "Ein hoher Wert schützt den einzelnen Fonds, erzeugt aber kurzfristig starken Verkaufsdruck.",
                f"−{float(st.session_state['fund_deleverage_step']):.2f} Hebelschritt bei Stress",
            )

        with st.expander("4. HFT / Liquidität"):
            st.number_input(
                "HFT-Kapital",
                0.0,
                50_000.0,
                step=500.0,
                key="hft_capital",
                disabled=is_preset,
                help="Kapitalbasis der automatisierten Liquiditätsanbieter.",
            )
            explain_slider(
                "wenig zusätzliche Liquidität",
                "viel zusätzliche Liquidität",
                "Mehr HFT-Kapital erhöht die Markttiefe und dämpft die Kurswirkung derselben Order.",
                f"{float(st.session_state['hft_capital']):,.0f} Kapitaleinheiten".replace(",", "."),
            )

            st.slider(
                "HFT-Abschaltung bei VIX",
                15.0,
                90.0,
                step=1.0,
                key="hft_vix_shutdown",
                disabled=is_preset,
                help="Ab welchem VIX die HFT-Liquidität vollständig aus dem Markt verschwindet.",
            )
            explain_slider(
                "HFTs schalten früh ab",
                "HFTs bleiben auch in Krisen aktiv",
                "Eine niedrige Schwelle erhöht die Gefahr eines plötzlichen Liquiditätsverlusts.",
                f"Abschaltung ab VIX {float(st.session_state['hft_vix_shutdown']):.0f}",
            )

            st.slider(
                "Liquidität pro Kapitaleinheit",
                0.0,
                0.20,
                step=0.005,
                key="hft_liquidity_factor",
                disabled=is_preset,
                help="Wie viel Markttiefe jede Einheit HFT-Kapital bereitstellt.",
            )
            explain_slider(
                "HFT-Kapital wirkt wenig",
                "HFT-Kapital erzeugt viel Markttiefe",
                "Gemeinsam mit dem HFT-Kapital bestimmt dieser Regler, wie stark Orders gedämpft werden.",
                f"Zusätzliche Markttiefe: {float(st.session_state['hft_capital']) * float(st.session_state['hft_liquidity_factor']):,.0f}".replace(",", "."),
            )

        with st.expander("5. Zentralbank"):
            st.slider(
                "Eingriffsschwelle (5-Tage-Verlust)",
                0.01,
                0.40,
                step=0.01,
                key="cb_intervention_threshold",
                disabled=is_preset,
                help="Ab welchem Verlust über fünf Tage die Zentralbank eingreift.",
            )
            explain_slider(
                "Zentralbank greift früh ein",
                "Zentralbank wartet auf einen schweren Einbruch",
                "Eine niedrige Schwelle führt zu häufigeren Stützungsmaßnahmen; eine hohe lässt mehr Marktbereinigung zu.",
                "Eingriff ab −" + format_percent(float(st.session_state['cb_intervention_threshold']), 0),
            )

            st.slider(
                "Kursimpuls beim Eingriff",
                0.0,
                0.15,
                step=0.005,
                key="cb_purchase_return",
                disabled=is_preset,
                help="Direkter positiver Renditebeitrag eines Zentralbankeingriffs.",
            )
            explain_slider(
                "schwache Kursstützung",
                "starker positiver Kurssprung",
                "Ein höherer Wert macht jeden Eingriff mächtiger und kann Verluste schneller ausgleichen.",
                "+" + format_percent(float(st.session_state['cb_purchase_return']), 1) + " am Eingriffstag",
            )

            st.slider(
                "VIX-Reduktion",
                0.0,
                0.80,
                step=0.05,
                key="cb_vix_reduction",
                disabled=is_preset,
                help="Anteil, um den die Zentralbank den VIX bei einem Eingriff reduziert.",
            )
            explain_slider(
                "Angst bleibt fast unverändert",
                "Angst wird stark reduziert",
                "Eine starke VIX-Reduktion hält HFTs eher aktiv und kann Fondsstress schneller beenden.",
                format_percent(float(st.session_state['cb_vix_reduction']), 0) + " VIX-Reduktion",
            )

        return st.button("🚀 Simulation mit diesen Einstellungen starten", type="primary", width="stretch")



def _return_label(value: float) -> tuple[str, str]:
    if value >= 0.20:
        return "starke Rallye", "Der Kurs ist über den gesamten Zeitraum deutlich gestiegen."
    if value >= 0.05:
        return "positiver Verlauf", "Der Markt hat insgesamt einen klaren Gewinn erzielt."
    if value > -0.05:
        return "weitgehend seitwärts", "Gewinne und Verluste hielten sich über den Gesamtzeitraum ungefähr die Waage."
    if value > -0.20:
        return "schwacher Verlauf", "Der Markt hat einen spürbaren, aber noch begrenzten Verlust erlitten."
    return "schwerer Einbruch", "Der Markt hat einen großen Teil seines Ausgangswerts verloren."


def _drawdown_label(value: float) -> str:
    severity = abs(value)
    if severity < 0.05:
        return "Die zwischenzeitlichen Rückgänge waren gering."
    if severity < 0.15:
        return "Zwischendurch gab es eine merkliche, aber überschaubare Verlustphase."
    if severity < 0.30:
        return "Der Markt erlebte zwischenzeitlich einen schweren Rückgang."
    return "Der Markt durchlief zwischenzeitlich eine sehr schwere Verlustphase."


def _vix_label(value: float) -> str:
    if value < 20:
        return "Die Angst im Markt blieb niedrig."
    if value < 30:
        return "Die Marktstimmung wurde zeitweise nervös, blieb aber kontrollierbar."
    if value < 45:
        return "Es gab mindestens eine ausgeprägte Stressphase."
    return "Es trat eine extreme Angst- oder Panikphase auf."


def build_plain_language_report(result) -> dict[str, str]:
    """Erzeugt eine datenbasierte Erklärung eines einzelnen Simulationslaufs."""

    data = result.data
    metrics = result.metrics
    params = result.params

    final_return = float(metrics["final_return"])
    max_drawdown = float(metrics["max_drawdown"])
    max_vix = float(metrics["max_vix"])
    label, return_sentence = _return_label(final_return)

    daily_returns = data.loc[1:, "Rendite"]
    best_idx = int(daily_returns.idxmax())
    worst_idx = int(daily_returns.idxmin())
    best_return = float(data.loc[best_idx, "Rendite"])
    worst_return = float(data.loc[worst_idx, "Rendite"])
    positive_share = float((daily_returns > 0).mean()) if len(daily_returns) else 0.0
    realized_vol = float(daily_returns.std(ddof=1) * np.sqrt(250)) if len(daily_returns) > 1 else 0.0

    prices = data["Kurs"].to_numpy(dtype=float)
    running_peak = np.maximum.accumulate(prices)
    drawdowns = prices / running_peak - 1.0
    drawdown_day = int(np.argmin(drawdowns))
    peak_day = int(np.argmax(prices[: drawdown_day + 1]))
    peak_price = float(prices[peak_day])
    trough_price = float(prices[drawdown_day])
    end_price = float(prices[-1])
    recovery_from_trough = end_price / trough_price - 1.0 if trough_price > 0 else 0.0

    shares = {
        "externe Markteinflüsse und Zufall": float(metrics["exogenous_share"]),
        "Käufe und Verkäufe der Marktteilnehmer": float(metrics["orderflow_share"]),
        "Zentralbankeingriffe": float(metrics["cb_share"]),
    }
    dominant_driver = max(shares, key=shares.get)
    dominant_share = shares[dominant_driver]

    exogenous_net = float(data["Exogener_Beitrag"].sum())
    orderflow_net = float(data["Orderflow_Beitrag"].sum())
    cb_net = float(data["Zentralbank_Beitrag"].sum())

    if orderflow_net > 0.01:
        orderflow_direction = "Die Käufe und Verkäufe der Anleger stützten den Markt in Summe."
    elif orderflow_net < -0.01:
        orderflow_direction = "Die Käufe und Verkäufe der Anleger belasteten den Markt in Summe."
    else:
        orderflow_direction = "Die Käufe und Verkäufe der Anleger waren über den gesamten Lauf annähernd ausgeglichen."

    panic_days = int((data["Rendite_5T"] <= params.retail_panic_threshold).sum())
    greed_days = int((data["Rendite_5T"] >= params.retail_greed_threshold).sum())
    retail_start = float(data["Retail_Quote"].iloc[0])
    retail_end = float(data["Retail_Quote"].iloc[-1])

    forced_sale_days = int((data["Fonds_Zwangsverkauf"] < -1e-12).sum())
    forced_sale_total = float(-data["Fonds_Zwangsverkauf"].clip(upper=0).sum())
    fund_aum_end = float(data["Fonds_AUM"].iloc[-1])
    fund_leverage_max = float(data["Fonds_Hebel"].max())

    hft_off_days = int(metrics["hft_off_days"])
    cb_interventions = int(metrics["cb_interventions"])
    jump_events = int(metrics["jump_events"])

    if final_return >= 0.05 and max_drawdown > -0.15:
        lesson = (
            "Der Lauf war insgesamt erfolgreich und die Verlustphasen blieben begrenzt. "
            "Trotzdem ist dies nur ein einzelner Zufallspfad und keine Prognose."
        )
    elif final_return >= 0 and max_drawdown <= -0.20:
        lesson = (
            "Das Endergebnis ist positiv, aber der Weg dorthin war riskant. Ein Anleger hätte "
            "zwischenzeitlich einen erheblichen Verlust aushalten müssen."
        )
    elif final_return < 0 and orderflow_net < -0.01:
        lesson = (
            "Der Verlust wurde nicht nur durch externe Schwankungen, sondern auch durch das Verhalten "
            "der Marktteilnehmer verstärkt. Panik, Deleveraging oder Abflüsse wirkten prozyklisch."
        )
    elif final_return < 0:
        lesson = (
            "Der Verlust kam überwiegend aus dem externen Marktpfad. Die eingestellten Teilnehmermechanismen "
            "konnten diesen Lauf nicht ausreichend stabilisieren."
        )
    else:
        lesson = (
            "Der Markt bewegte sich ohne eindeutigen langfristigen Gewinner. Für eine belastbare Aussage "
            "sollten mehrere Seeds und die Sensitivitätsanalyse betrachtet werden."
        )

    headline = f"**{label.capitalize()}: {format_percent(final_return)} Gesamtrendite.** {return_sentence}"

    course = (
        f"Der Kurs startete bei 100,00 und endete bei {end_price:.2f}. "
        f"{_drawdown_label(max_drawdown)} Der größte Rückgang vom vorherigen Höchststand betrug "
        f"{format_percent(max_drawdown)}: vom Hoch bei {peak_price:.2f} an Tag {peak_day} "
        f"bis auf {trough_price:.2f} an Tag {drawdown_day}. Danach erholte sich der Kurs "
        f"um {format_percent(recovery_from_trough)} vom Tief. {_vix_label(max_vix)} "
        f"Der beste Tag brachte {format_percent(best_return)}, der schlechteste Tag "
        f"{format_percent(worst_return)}. {positive_share * 100:.0f} % der Tage waren positiv."
    )

    causes = (
        f"Die größten durchschnittlichen Tagesbewegungen kamen aus **{dominant_driver}** "
        f"(gemessener Beitragsanteil: {dominant_share * 100:.0f} %). "
        f"{orderflow_direction} Additiv über alle Tage betrugen die protokollierten Beiträge ungefähr: "
        f"extern {format_percent(exogenous_net)}, Orderflow {format_percent(orderflow_net)} und "
        f"Zentralbank {format_percent(cb_net)}. Diese Summen dienen zur Ursachenanalyse; wegen "
        f"täglicher Verzinsung entsprechen sie nicht exakt der Gesamtrendite."
    )

    participants = (
        f"Privatanleger starteten mit einer Aktienquote von {format_percent(retail_start, 0)} und "
        f"endeten bei {format_percent(retail_end, 0)}. Das Modell registrierte {panic_days} Paniktage "
        f"und {greed_days} Giertage. Die Fonds erreichten maximal einen Hebel von {fund_leverage_max:.2f}×. "
        f"An {forced_sale_days} Tagen gab es Zwangsverkäufe; das verbleibende Fondsvermögen lag am Ende "
        f"bei {fund_aum_end * 100:.1f} % des Ausgangswerts. HFTs waren {hft_off_days} Tage abgeschaltet. "
        f"Es traten {jump_events} Sprungereignisse und {cb_interventions} Zentralbankeingriffe auf."
    )

    risk = (
        f"Die realisierte Schwankungsintensität lag annualisiert bei ungefähr "
        f"{format_percent(realized_vol)}. {lesson}"
    )

    full_text = "\n\n".join(
        [
            "ERGEBNIS DER SIMULATION",
            headline.replace("**", ""),
            "MARKTVERLAUF\n" + course,
            "URSACHEN\n" + causes.replace("**", ""),
            "VERHALTEN DER MARKTTEILNEHMER\n" + participants,
            "EINORDNUNG\n" + risk,
            (
                "HINWEIS\nDie Erklärung basiert ausschließlich auf diesem simulierten Lauf und den "
                "im Modell protokollierten Komponenten. Sie ist keine Anlageberatung und keine Prognose."
            ),
        ]
    )

    return {
        "headline": headline,
        "course": course,
        "causes": causes,
        "participants": participants,
        "risk": risk,
        "full_text": full_text,
    }


def render_plain_language_result(result) -> None:
    report = build_plain_language_report(result)
    st.subheader("🗣️ Ergebnis in verständlicher Sprache")
    st.success(report["headline"])

    with st.container(border=True):
        st.markdown("#### 1. Was ist mit dem Markt passiert?")
        st.write(report["course"])

        st.markdown("#### 2. Warum ist das passiert?")
        st.write(report["causes"])

        st.markdown("#### 3. Wie haben sich die Marktteilnehmer verhalten?")
        st.write(report["participants"])

        st.markdown("#### 4. Wie ist das Ergebnis einzuordnen?")
        st.write(report["risk"])

    st.caption(
        "Die Erklärung basiert auf den tatsächlich gespeicherten Modellbeiträgen dieses Laufs. "
        "Sie beschreibt keine reale Marktentwicklung und ist keine Anlageberatung."
    )
    st.download_button(
        "📥 Erklärung als Text herunterladen",
        data=report["full_text"].encode("utf-8"),
        file_name="simulation_erklaerung.txt",
        mime="text/plain",
    )

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

st.title("🧠 Marktpsychologie-Simulator 2.2")
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
    if current_params() != result.params:
        st.warning(
            "Die Regler wurden seit dem letzten Lauf verändert. Die unten angezeigten Ergebnisse "
            "gehören noch zu den vorherigen Einstellungen. Starte die Simulation erneut, um sie zu aktualisieren."
        )
    render_plain_language_summary(result.params)
    render_plain_language_result(result)
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
