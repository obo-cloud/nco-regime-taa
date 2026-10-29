# Market Regime-Based Multi-Asset Portfolio Optimization Using Nested Clustered Optimization for Tactical Asset Allocation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

Official open-source data and code repository for:
**"Market Regime-Based Multi-Asset Portfolio Optimization Using Nested Clustered Optimization for Tactical Asset Allocation"**  
*Authors:* **Supriadi** and **Bambang Siswoyo**

---

## Overview

This repository provides a modular, reproducible research pipeline examining the empirical interaction between **continuous Gaussian Hidden Markov Model (HMM)** regime detection and **Nested Clustered Optimization (NCO)** for tactical asset allocation across emerging multi-asset portfolios.

### Key Research Questions Investigated
1. **Marginal Value of Regime Conditioning:** Does conditioning the covariance matrix and hierarchical dendrogram on latent market states genuinely enhance risk-adjusted returns, or does parameter estimation error within sparse sub-samples offset the theoretical benefits?
2. **Performance Trade-Offs:** How does the proposed regime-aware framework compare against unconditional graph clustering (Static NCO, True HRP), classical risk-based benchmarks (Minimum Variance), and diversified heuristics (Equal Weight $1/N$, Buy-and-Hold)?
3. **Turnover & Rebalancing Dynamics:** What are the operational costs of dynamic regime adaptation versus passive weight drift under realistic transaction slippage?

---

## Repository Structure

```text
nco-regime-taa/
├── data/
│   ├── dataset_monthly_returns.csv   # Historical simple monthly returns (80 periods, 2019-2025)
│   └── data_dictionary.md            # Data sources, proxy descriptions, and accounting definitions
├── src/
│   ├── run_backtest.py               # Complete end-to-end walk-forward backtesting pipeline
│   └── run_analysis.py               # Analysis runner and figure generator
├── figures/                          # Publication-grade visualizations
│   ├── fig1_cumulative_value.png     # Out-of-sample portfolio trajectories (log-scale NAV)
│   ├── fig2_rolling_sharpe.png       # 12-month rolling Sharpe ratio comparison
│   ├── fig3_regimes_lq45.png         # Decoded latent market regimes overlaid on LQ45
│   ├── fig4_regime_correlations.png  # Regime-conditional cross-asset correlation heatmaps
│   ├── fig5_weight_evolution.png     # Time-varying portfolio asset allocation weights
│   └── fig6_drawdown.png             # Under-water drawdown curves across all strategies
├── results/                          # Tabular backtest outputs and transaction ledgers
│   ├── performance_metrics.csv       # Summary risk-adjusted metrics across all benchmarks
│   ├── monthly_ledger.csv            # Detailed monthly accounting ledger (NAV, weights, turnover, costs)
│   ├── rebalancing_audit.csv         # Breakdown of rebalance events (regime shift vs weight drift)
│   └── tc_sensitivity.csv            # Slippage sensitivity grid (0 to 20 bps)
├── sbn_duration_robustness.csv   # Duration-convexity sensitivity grid (D* in [5.0, 8.0] yr)
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT Open Source License
└── README.md                         # Documentation
```

---

## Dataset Description

The empirical universe comprises 80 monthly observations (January 2019 – August 2025) representing five core asset classes in the Indonesian capital market:

| Asset | Proxy / Underlying | Provider / Identifier | Currency | Return Type | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SBN** | Sovereign Bond Yield (10Y) | Investing.com | IDR | Yield Change (%) | Sovereign duration & yield stress proxy |
| **LQ45** | Large-Cap Liquid Equities | Yahoo Finance (`^JKLQ45`) | IDR | Simple Return (%) | Core domestic equity growth engine |
| **Gold** | Spot Gold in Rupiah | Investing.com (`XAU/IDR`) | IDR | Simple Return (%) | Safe-haven domestic inflation hedge |
| **Valas** | Foreign Exchange | Investing.com (`CNY/IDR`) | IDR | Simple Return (%) | Bilateral trade exposure & FX buffer |
| **BTC** | Cryptocurrency Asset | Investing.com (`BTC/IDR`) | IDR | Simple Return (%) | Convex growth momentum diversifier |

All data is pre-processed into monthly decimal returns in `data/dataset_monthly_returns.csv`.

---

## Audited Empirical Results (56-Month Out-of-Sample Window)

*Out-of-sample period: January 2021 to August 2025. Risk-free rate: $r_f = 0.479\%$ monthly ($5.75\%$ annualized).*

| Strategy | Cumulative Return (%) | Ann. Return (%) | Volatility (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) | Calmar Ratio | CVaR 95% (%) | Monthly Turnover (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NCO (No Regime)** | **29.97** | **5.78** | **4.65** | **+0.0063** | **+0.0128** | **2.18** | **2.6506** | **1.67** | **5.41** |
| Minimum Variance | 18.45 | 3.69 | 4.70 | -0.4371 | -0.6258 | 4.75 | 0.7773 | 2.64 | 1.36 |
| True HRP (Independent) | 8.27 | 1.72 | 3.42 | -1.1802 | -1.6007 | 4.21 | 0.4074 | 2.23 | 3.59 |
| **Regime-NCO (Proposed)** | **1.60** | **0.34** | **5.52** | **-0.9791** | **-1.3124** | **9.99** | **0.0340** | **3.85** | **24.58** |
| Equal Weight ($1/N$) | 76.70 | 12.97 | 13.02 | +0.5551 | +1.0176 | 15.27 | 0.8496 | 6.67 | 0.00 |
| Buy-and-Hold | 87.98 | 14.48 | 17.56 | +0.4975 | +0.8974 | 21.17 | 0.6841 | 9.47 | 0.00 |

### Key Takeaways:
- **Capital Preservation:** Regime-agnostic NCO provides superior downside protection, keeping maximum drawdown to **2.18%** and delivering a Calmar ratio of **2.65**, outperforming all benchmarks in risk efficiency.
- **Sample Starvation in Regime Conditioning:** Conditioning NCO on an unconstrained rolling HMM in a small multi-asset pool ($N=5$) partitions lookback windows into sparse sub-samples ($T_k < 2N$), amplifying parameter estimation noise and causing 47 rebalance events (26 regime shifts + 20 drift corrections) with an average monthly turnover of **24.58%**.
- **Statistical Significance:** The pairwise Jobson–Korkie test ($z = 2.684, p = 0.0073^{**}$) confirms the statistical superiority of static NCO over Regime-NCO at the 1% significance level.

---


---

## Sovereign Bond (SBN) Repricing & Duration Robustness (Table VII)

To evaluate the sensitivity of portfolio outcomes to sovereign fixed-income return specifications, we implement an analytical duration-convexity repricing model for the Indonesian 10-year sovereign bond (SBN):

\\Delta P_t / P_{t-1} \\approx - D^* \\Delta y_t + \\frac{1}{2} C_x (\\Delta y_t)^2 + \\frac{y_{t-1}}{12}

where $\\Delta y_t = y_t - y_{t-1}$ is the monthly shift in 10-year benchmark yields, ^*$ is the modified duration,  \\approx 60.0\\text{ yr}^2$ is bond convexity, and {t-1}/12$ represents accrued coupon carry. 

Walk-forward performance across a spectrum of modified duration specifications (^* \\in [5.0, 8.0]$ years) over the 56-month out-of-sample period confirms empirical invariance:

| Modified Duration Specification | Ann. Return (%) | Max Drawdown (%) | Excess Sharpe ( = 5.75\%$) | Calmar Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Panel A: Static NCO (Regime-Agnostic)** | | | | |
| ^* = 5.0\\text{ yr}$ | 7.16 | 4.19 | +0.2623 | 1.7107 |
| ^* = 6.0\\text{ yr}$ | 7.12 | 4.47 | +0.2425 | 1.5930 |
| ^* = 7.0\\text{ yr}$ (Base) | **7.01** | **4.68** | **+0.2205** | **1.4964** |
| ^* = 8.0\\text{ yr}$ | 6.87 | 4.84 | +0.1943 | 1.4188 |
| **Panel B: Regime-NCO (Adaptive Switching)** | | | | |
| ^* = 5.0\\text{ yr}$ | 4.75 | 7.20 | -0.1741 | 0.6596 |
| ^* = 6.0\\text{ yr}$ | 5.06 | 7.56 | -0.1206 | 0.6701 |
| ^* = 7.0\\text{ yr}$ (Base) | 3.87 | 7.88 | -0.3139 | 0.4910 |
| ^* = 8.0\\text{ yr}$ | 4.31 | 8.17 | -0.2307 | 0.5281 |

*Empirical Takeaway:* Static NCO maintains ~7% CAGR, ~4.5% MDD, and positive excess Sharpe ratios across the entire duration spectrum, whereas Regime-NCO consistently underperforms (Sharpe -0.12 to -0.31, turnover >20%/month). This establishes that the estimation noise pathology in regime-conditioned NCO is invariant to fixed-income modeling choices.

## Quickstart & Reproduction

### 1. Clone the repository
```bash
git clone https://github.com/obo-cloud/nco-regime-taa.git
cd nco-regime-taa
```

### 2. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### 3. Run the backtesting pipeline
```bash
python src/run_backtest.py
```
This script executes the complete walk-forward simulation, generates all tabular metrics in `results/`, and creates the 6 visualization figures in `figures/`.

---

## Citation

```bibtex
@article{supriadi2026regime,
  title   = {Market Regime-Based Multi-Asset Portfolio Optimization Using Nested Clustered Optimization for Tactical Asset Allocation},
  author  = {Supriadi and Siswoyo, Bambang},
  journal = {Quantitative Finance Working Paper},
  year    = {2026}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
