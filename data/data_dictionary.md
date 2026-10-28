# Data Dictionary: Multi-Asset Tactical Allocation Dataset

This dataset provides monthly simple returns across five heterogeneous asset classes in an emerging market (Indonesia).

## Temporal Coverage
- **Total Period:** January 2019 – August 2025 (80 monthly periods)
- **Warmup / Initial Estimation Window:** January 2019 – December 2020 (24 months)
- **Walk-Forward Out-of-Sample Horizon:** January 2021 – August 2025 (56 months)

## Asset Classes & Identifiers

| Asset Identifier | Asset Class | Source / Reference | Currency | Return Type | Role in Portfolio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SBN** | Sovereign Fixed Income | ID 10-Year Government Bond Yield | IDR | Monthly Yield Change (%) | Sovereign duration & yield stress proxy |
| **LQ45** | Large-Cap Domestic Equity | Jakarta Liquid 45 Index (`^JKLQ45`) | IDR | Monthly Simple Return (%) | Core domestic equity growth engine |
| **Gold** | Commodity (Precious Metal) | Spot Gold in Rupiah (`XAU/IDR`) | IDR | Monthly Simple Return (%) | Domestic currency safe-haven hedge |
| **Valas** | Foreign Exchange | Bilateral Currency Rate (`CNY/IDR`) | IDR | Monthly Simple Return (%) | Major trade-flow currency buffer |
| **BTC** | Digital Asset (Cryptocurrency) | Spot Bitcoin in Rupiah (`BTC/IDR`) | IDR | Monthly Simple Return (%) | Alternative convex growth diversifier |

## Modeling & Accounting Conventions
1. **Portfolio Net Asset Value (NAV):**
   - Simple monthly returns: $R_{p,t} = \mathbf{w}_t^\top \mathbf{R}_t - c \times \text{TO}_t$
   - Proportional transaction cost: $c = 10\text{ bps}$ ($0.10\%$) applied to one-way turnover.
2. **HMM State Features:**
   - Feature vector: $\mathbf{f}_t = [\bar{r}_t^{\text{eq}}, \sigma_t^{\text{eq}}, \bar{r}_t^{\text{bond}}, \Delta e_t]^\top$
   - Computed using rolling stationary log-returns $\ell_{i,t} = \ln(1 + R_{i,t})$ over a 24-month rolling lookback.
