# AI-conditioned OU pairs trading on Latin American ADRs — a proof of concept

A research notebook arc that modernizes a classical pairs-trading strategy in two layers: OU mean-reversion diagnostics (Elliott 2005) to select cointegrated pairs, traded with a rolling z-score, and an AI-derived conditioning overlay — earnings calendar + news sentiment — layered on top.

## TL;DR

Across 11 years of US-listed LatAm ADRs, applying OU mean-reversion diagnostics to select pairs (traded with a rolling z-score) lifts a near-zero cointegration-only baseline (**Sharpe 0.04**) to **0.72**; an earnings-gate overlay carries it to **~1.00** at the principled horizon K=33 (robust across K ∈ {7…33}). A sentiment-gate overlay does not survive coverage (most names too news-thin) and signal (on the covered slice, VADER tone slightly *reduced* risk-adjusted return). Drawing that boundary precisely — rather than claiming an edge the data cannot support — is the contribution.

## Pipeline

```mermaid
flowchart TD
    DATA["EODHD<br/>prices · news · earnings"] --> NB01
    NB01["NB01 — ADR Universe<br/>~11y point-in-time"] --> NB02
    NB02["NB02 — Pairs Engine<br/>cointegration · OU-selected"] --> NB03
    NB03["NB03 — AI Conditioning<br/>earnings + sentiment gates"] --> NB04
    NB04["NB04 — Portfolio Ablation<br/>K-sweep · 4 variants"]
    APP["Appendix 02b<br/>risk-norm toolkit"] -.->|promoted into| NB04
```

## Findings

![Four-variant ablation — cumulative risk-normalized PnL](docs/ablation_equity.png)

*Cointegration-only selection (black, dashed) produces no tradable edge; the OU dynamics filter (grey) lifts Sharpe to 0.72; the earnings gate (blue) carries it to ~1.00 with shallower drawdowns; sentiment (orange) sits marginally below.*

| Variant | Sharpe | Note |
|---|---:|---|
| Cointegration-only | 0.04 | indistinguishable from zero |
| OU-selected | 0.72 | tradability comes from mean-reversion speed/cleanliness, not cointegration alone |
| OU + earnings gate | ~1.00 | at K=33 (median holding period, fixed pre-hoc); robust across K ∈ {7, 14, 21, 33} |
| OU + earnings + sentiment | ~0.98 | sentiment fails twice — coverage cliff + on the covered slice, VADER tone slightly *reduced* risk-adjusted return |

The sentiment boundary is the contribution: neither this lexicon nor this feed is fit for purpose. A finance-tuned tone model (FinBERT, Loughran-McDonald) on the covered slice, or a quant-grade entity-resolved feed (RavenPack, Refinitiv MarketPsych), would extend the picture.

**Design note.** We evaluated the Avellaneda & Lee (2010) S-score (frozen IS equilibrium μ, σ_eq) as an alternative trade signal. On these pairs the frozen parameters did not survive multi-year OOS — OOS spread volatility ran ~3× the IS σ_eq and the mean drifted multiple σ_eq from IS μ, saturating the signal. Rolling-z adapts; frozen-S does not. (Consistent with cointegration breakdown — Gatev, Goetzmann & Rouwenhorst 2006.)

**Scope.** Proof of concept. Framework feasibility established; deployability not claimed.

## Reproducibility

The repo ships notebook code and narrative only — no rendered outputs, no vendor data. A fresh clone reads as code + prose; charts and tables materialize once you run it. Data isn't redistributed (EODHD terms of service), so reproducing the results requires your own EODHD API key.

```bash
# 1. Clone and enter
git clone https://github.com/<user>/ai-pairs-trading.git
cd ai-pairs-trading

# 2. Python env (3.11+ recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. EODHD key
echo "EODHD_API_KEY=your_key_here" > .env

# 4. Build the data snapshot (first run only)
jupyter lab notebooks/01_adr_universe.ipynb   # run all cells; populates ./data/processed/

# 5. Run the pipeline: NB02 → NB03 → NB04, in order
```

Subsequent runs of NB01 replay from the local `data/processed/` snapshot (set `OFFLINE_MODE=1` in `.env`); the EODHD key is only needed to *build* the snapshot, not to re-run downstream notebooks.

## Repo layout

```
ai-pairs-trading/
├── notebooks/
│   ├── 01_adr_universe.ipynb               # point-in-time universe & data snapshot
│   ├── 02_pairs_engine.ipynb               # cointegration → OU spread → trades
│   ├── 03_ai_conditioned_pairs.ipynb       # earnings gate + sentiment overlay
│   ├── 04_conditioned_portfolio.ipynb      # 4-variant ablation, coverage strata, K-sweep
│   └── appendix/
│       └── 02b_ou_portfolio_appendix.ipynb # risk-normalization toolkit (promoted into NB04)
├── docs/
│   └── ablation_equity.png                 # 4-variant cumulative-PnL figure (README)
├── requirements.txt
├── .gitignore
└── README.md
```

Local-only (gitignored): `data/`, `artifacts/`, `semantic_cache_v05/` — vendor data and intermediate parquets — plus `.venv/`, `.env`, and `notebooks/img/` (matplotlib figures regenerated on every NB01 run).

## References

- Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models." *arXiv preprint* arXiv:1908.10063.
- Do, B., & Faff, R. (2010). "Does Simple Pairs Trading Still Work?" *Financial Analysts Journal*, 66(4), 83–95.
- Elliott, R. J., van der Hoek, J., & Malcolm, W. P. (2005). "Pairs Trading." *Quantitative Finance*, 5(3), 271–276.
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies*, 19(3), 797–827.
- Hutto, C. J., & Gilbert, E. (2014). "VADER: A Parsimonious Rule-Based Model for Sentiment Analysis of Social Media Text." *Proceedings of the International AAAI Conference on Web and Social Media*, 8(1), 216–225.
- Loughran, T., & McDonald, B. (2011). "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks." *Journal of Finance*, 66(1), 35–65.
