# Marktpsychologie-Simulator 2.0

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Start

```bash
streamlit run marktpsychologie_app.py
```

## Tests

```bash
pytest -q
```

## Wesentliche Modellverbesserungen

- Zufalls-Seed für reproduzierbare Läufe
- vollständige, atomare Presets
- exakter 5-Tage-Rückblick
- echter Maximum Drawdown
- finales Tageslimit nach Addition aller Komponenten
- HFT-Kapital wirkt über die effektive Markttiefe
- Fondsabflüsse erzeugen protokollierte Zwangsverkaufsorders
- Zentralbankinterventionen werden als Ereignisse gespeichert
- tägliche Rendite wird exakt in Exogen, Orderflow, Zentralbank und Cap zerlegt
- Sensitivitätsanalyse verwendet identische Zufallspfade je Vergleichspaar
