# 131-Month Walk-Forward Autonomous Audit for Gold (GOLDBEES: 2015 to 2026)

**Model Architecture:** GOLD-BAQE-v1.0 (Calibrated for Currency Drift, Inflation & Geopolitics)
- **Historical Horizon:** 131 Months (Oct 2015 to Sep 2026)
- **Directional Hit Rate:** `60.3%` (79/131 months correctly forecasted)
- **Mean Absolute Error (MAE):** `3.717 pp`
- **Root Mean Sq Error (RMSE):** `5.021 pp`
- **80% Confidence Band Coverage:** `87.0%`

| Month | Origin Date | Target Date | Origin Price (₹) | Actual Price (₹) | Actual % | Predicted % | Target Level (₹) | Abs Error (pp) | Direction Hit | Inside 80% Band | Regime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 2015-10-29 | 2015-11-27 | ₹24.45 | ₹23.10 | -5.50% | +2.07% | ₹24.95 | 7.57 | NO | NO | Safe-Haven & Inflation Breakout |
| 2 | 2015-11-27 | 2015-12-29 | ₹23.10 | ₹22.99 | -0.48% | +1.61% | ₹23.47 | 2.09 | NO | YES | Gold Value Dip Accumulation |
| 3 | 2015-12-29 | 2016-01-29 | ₹22.99 | ₹24.23 | +5.40% | +1.74% | ₹23.39 | 3.66 | YES | YES | Gold Value Dip Accumulation |
| 4 | 2016-01-29 | 2016-02-29 | ₹24.23 | ₹26.49 | +9.31% | +2.39% | ₹24.81 | 6.92 | YES | NO | Safe-Haven & Inflation Breakout |
| 5 | 2016-02-29 | 2016-03-29 | ₹26.49 | ₹25.43 | -4.00% | +1.96% | ₹27.01 | 5.96 | NO | YES | Safe-Haven & Inflation Breakout |
| 6 | 2016-03-29 | 2016-04-29 | ₹25.43 | ₹27.17 | +6.84% | +0.97% | ₹25.68 | 5.87 | YES | YES | Secular Currency Drift |
| 7 | 2016-04-29 | 2016-05-27 | ₹27.17 | ₹26.81 | -1.32% | +1.95% | ₹27.70 | 3.28 | NO | YES | Safe-Haven & Inflation Breakout |
| 8 | 2016-05-27 | 2016-06-29 | ₹26.81 | ₹27.85 | +3.89% | +0.83% | ₹27.03 | 3.06 | YES | YES | Secular Currency Drift |
| 9 | 2016-06-29 | 2016-07-29 | ₹27.85 | ₹27.96 | +0.39% | +0.41% | ₹27.97 | 0.02 | YES | YES | Overbought Froth & Consolidation |
| 10 | 2016-07-29 | 2016-08-29 | ₹27.96 | ₹28.16 | +0.71% | +0.67% | ₹28.15 | 0.05 | YES | YES | Secular Currency Drift |
| 11 | 2016-08-29 | 2016-09-29 | ₹28.16 | ₹28.50 | +1.20% | +0.67% | ₹28.35 | 0.53 | YES | YES | Secular Currency Drift |
| 12 | 2016-09-29 | 2016-10-28 | ₹28.50 | ₹27.36 | -4.01% | +1.94% | ₹29.05 | 5.95 | NO | YES | Safe-Haven & Inflation Breakout |
| 13 | 2016-10-28 | 2016-11-29 | ₹27.36 | ₹26.54 | -2.97% | +0.97% | ₹27.62 | 3.94 | NO | YES | Secular Currency Drift |
| 14 | 2016-11-29 | 2016-12-29 | ₹26.54 | ₹25.47 | -4.06% | +1.20% | ₹26.86 | 5.26 | NO | YES | Secular Currency Drift |
| 15 | 2016-12-29 | 2017-01-27 | ₹25.47 | ₹26.22 | +2.94% | +1.91% | ₹25.95 | 1.03 | YES | YES | Gold Value Dip Accumulation |
| 16 | 2017-01-27 | 2017-02-28 | ₹26.22 | ₹27.02 | +3.08% | +1.82% | ₹26.69 | 1.25 | YES | YES | Gold Value Dip Accumulation |
| 17 | 2017-02-28 | 2017-03-29 | ₹27.02 | ₹26.29 | -2.71% | +2.57% | ₹27.72 | 5.29 | NO | YES | Safe-Haven & Inflation Breakout |
| 18 | 2017-03-29 | 2017-04-28 | ₹26.29 | ₹26.41 | +0.46% | +1.96% | ₹26.80 | 1.50 | YES | YES | Gold Value Dip Accumulation |
| 19 | 2017-04-28 | 2017-05-29 | ₹26.41 | ₹26.30 | -0.43% | +1.59% | ₹26.83 | 2.02 | NO | YES | Secular Currency Drift |
| 20 | 2017-05-29 | 2017-06-29 | ₹26.30 | ₹25.84 | -1.75% | +2.10% | ₹26.85 | 3.85 | NO | YES | Gold Value Dip Accumulation |
| 21 | 2017-06-29 | 2017-07-28 | ₹25.84 | ₹25.65 | -0.70% | +2.26% | ₹26.42 | 2.96 | NO | YES | Gold Value Dip Accumulation |
| 22 | 2017-07-28 | 2017-08-29 | ₹25.65 | ₹26.83 | +4.57% | +2.38% | ₹26.26 | 2.20 | YES | YES | Gold Value Dip Accumulation |
| 23 | 2017-08-29 | 2017-09-29 | ₹26.83 | ₹26.66 | -0.62% | +3.04% | ₹27.64 | 3.65 | NO | YES | Safe-Haven & Inflation Breakout |
| 24 | 2017-09-29 | 2017-10-27 | ₹26.66 | ₹26.28 | -1.45% | +1.87% | ₹27.16 | 3.32 | NO | YES | Secular Currency Drift |
| 25 | 2017-10-27 | 2017-11-29 | ₹26.28 | ₹26.50 | +0.85% | +1.99% | ₹26.80 | 1.14 | YES | YES | Secular Currency Drift |
| 26 | 2017-11-29 | 2017-12-29 | ₹26.50 | ₹26.27 | -0.88% | +1.97% | ₹27.02 | 2.85 | NO | YES | Secular Currency Drift |
| 27 | 2017-12-29 | 2018-01-29 | ₹26.27 | ₹27.15 | +3.34% | +2.07% | ₹26.81 | 1.27 | YES | YES | Secular Currency Drift |
| 28 | 2018-01-29 | 2018-02-28 | ₹27.15 | ₹27.30 | +0.58% | +3.21% | ₹28.02 | 2.63 | YES | YES | Safe-Haven & Inflation Breakout |
| 29 | 2018-02-28 | 2018-03-28 | ₹27.30 | ₹27.46 | +0.57% | +1.99% | ₹27.85 | 1.42 | YES | YES | Secular Currency Drift |
| 30 | 2018-03-28 | 2018-04-27 | ₹27.46 | ₹27.90 | +1.61% | +2.00% | ₹28.01 | 0.38 | YES | YES | Secular Currency Drift |
| 31 | 2018-04-27 | 2018-05-29 | ₹27.90 | ₹27.91 | +0.03% | +3.25% | ₹28.81 | 3.22 | YES | YES | Safe-Haven & Inflation Breakout |
| 32 | 2018-05-29 | 2018-06-29 | ₹27.91 | ₹27.13 | -2.79% | +2.07% | ₹28.49 | 4.86 | NO | YES | Secular Currency Drift |
| 33 | 2018-06-29 | 2018-07-27 | ₹27.13 | ₹26.54 | -2.17% | +2.27% | ₹27.75 | 4.44 | NO | YES | Secular Currency Drift |
| 34 | 2018-07-27 | 2018-08-29 | ₹26.54 | ₹26.90 | +1.37% | +2.43% | ₹27.19 | 1.06 | YES | YES | Secular Currency Drift |
| 35 | 2018-08-29 | 2018-09-28 | ₹26.90 | ₹27.03 | +0.46% | +3.70% | ₹27.90 | 3.23 | YES | YES | Safe-Haven & Inflation Breakout |
| 36 | 2018-09-28 | 2018-10-29 | ₹27.03 | ₹28.35 | +4.89% | +2.49% | ₹27.70 | 2.41 | YES | YES | Secular Currency Drift |
| 37 | 2018-10-29 | 2018-11-29 | ₹28.35 | ₹27.11 | -4.37% | +3.55% | ₹29.36 | 7.93 | NO | NO | Safe-Haven & Inflation Breakout |
| 38 | 2018-11-29 | 2018-12-28 | ₹27.11 | ₹28.15 | +3.83% | +2.59% | ₹27.81 | 1.24 | YES | YES | Secular Currency Drift |
| 39 | 2018-12-28 | 2019-01-29 | ₹28.15 | ₹29.17 | +3.61% | +3.74% | ₹29.20 | 0.13 | YES | YES | Safe-Haven & Inflation Breakout |
| 40 | 2019-01-29 | 2019-02-28 | ₹29.17 | ₹29.50 | +1.15% | +3.65% | ₹30.23 | 2.50 | YES | YES | Safe-Haven & Inflation Breakout |
| 41 | 2019-02-28 | 2019-03-28 | ₹29.50 | ₹28.25 | -4.24% | +2.41% | ₹30.21 | 6.65 | NO | NO | Secular Currency Drift |
| 42 | 2019-03-28 | 2019-04-26 | ₹28.25 | ₹28.16 | -0.34% | +2.67% | ₹29.01 | 3.00 | NO | YES | Secular Currency Drift |
| 43 | 2019-04-26 | 2019-05-29 | ₹28.16 | ₹28.20 | +0.16% | +2.75% | ₹28.93 | 2.59 | YES | YES | Secular Currency Drift |
| 44 | 2019-05-29 | 2019-06-28 | ₹28.20 | ₹29.98 | +6.30% | +4.09% | ₹29.36 | 2.21 | YES | YES | Safe-Haven & Inflation Breakout |
| 45 | 2019-06-28 | 2019-07-29 | ₹29.98 | ₹30.67 | +2.30% | +3.85% | ₹31.13 | 1.55 | YES | YES | Safe-Haven & Inflation Breakout |
| 46 | 2019-07-29 | 2019-08-29 | ₹30.67 | ₹34.22 | +11.59% | +3.83% | ₹31.84 | 7.76 | YES | NO | Safe-Haven & Inflation Breakout |
| 47 | 2019-08-29 | 2019-09-27 | ₹34.22 | ₹33.02 | -3.51% | +1.79% | ₹34.83 | 5.30 | NO | YES | Overbought Froth & Consolidation |
| 48 | 2019-09-27 | 2019-10-29 | ₹33.02 | ₹33.71 | +2.09% | +2.24% | ₹33.76 | 0.15 | YES | YES | Secular Currency Drift |
| 49 | 2019-10-29 | 2019-11-29 | ₹33.71 | ₹33.41 | -0.88% | +3.48% | ₹34.88 | 4.36 | NO | YES | Safe-Haven & Inflation Breakout |
| 50 | 2019-11-29 | 2019-12-27 | ₹33.41 | ₹34.15 | +2.20% | +2.33% | ₹34.19 | 0.12 | YES | YES | Secular Currency Drift |
| 51 | 2019-12-27 | 2020-01-29 | ₹34.15 | ₹35.61 | +4.28% | +3.54% | ₹35.36 | 0.73 | YES | YES | Safe-Haven & Inflation Breakout |
| 52 | 2020-01-29 | 2020-02-28 | ₹35.61 | ₹37.37 | +4.94% | +3.43% | ₹36.83 | 1.51 | YES | YES | Safe-Haven & Inflation Breakout |
| 53 | 2020-02-28 | 2020-03-27 | ₹37.37 | ₹38.17 | +2.14% | +3.28% | ₹38.60 | 1.14 | YES | YES | Safe-Haven & Inflation Breakout |
| 54 | 2020-03-27 | 2020-04-29 | ₹38.17 | ₹42.37 | +11.00% | +3.26% | ₹39.42 | 7.74 | YES | NO | Safe-Haven & Inflation Breakout |
| 55 | 2020-04-29 | 2020-05-29 | ₹42.37 | ₹40.99 | -3.26% | +2.84% | ₹43.57 | 6.10 | NO | YES | Safe-Haven & Inflation Breakout |
| 56 | 2020-05-29 | 2020-06-29 | ₹40.99 | ₹42.49 | +3.66% | +1.79% | ₹41.72 | 1.87 | YES | YES | Secular Currency Drift |
| 57 | 2020-06-29 | 2020-07-29 | ₹42.49 | ₹46.52 | +9.48% | +1.36% | ₹43.07 | 8.13 | YES | NO | Overbought Froth & Consolidation |
| 58 | 2020-07-29 | 2020-08-28 | ₹46.52 | ₹45.10 | -3.05% | +0.90% | ₹46.94 | 3.95 | NO | YES | Overbought Froth & Consolidation |
| 59 | 2020-08-28 | 2020-09-29 | ₹45.10 | ₹44.15 | -2.11% | +1.34% | ₹45.71 | 3.45 | NO | YES | Secular Currency Drift |
| 60 | 2020-09-29 | 2020-10-29 | ₹44.15 | ₹44.35 | +0.45% | +1.95% | ₹45.01 | 1.49 | YES | YES | Gold Value Dip Accumulation |
| 61 | 2020-10-29 | 2020-11-27 | ₹44.35 | ₹42.84 | -3.40% | +2.00% | ₹45.24 | 5.41 | NO | YES | Gold Value Dip Accumulation |
| 62 | 2020-11-27 | 2020-12-29 | ₹42.84 | ₹43.68 | +1.96% | +2.24% | ₹43.80 | 0.28 | YES | YES | Gold Value Dip Accumulation |
| 63 | 2020-12-29 | 2021-01-29 | ₹43.68 | ₹42.65 | -2.36% | +2.23% | ₹44.66 | 4.59 | NO | YES | Gold Value Dip Accumulation |
| 64 | 2021-01-29 | 2021-02-26 | ₹42.65 | ₹40.25 | -5.63% | +2.39% | ₹43.67 | 8.02 | NO | NO | Gold Value Dip Accumulation |
| 65 | 2021-02-26 | 2021-03-26 | ₹40.25 | ₹38.79 | -3.63% | +2.79% | ₹41.37 | 6.42 | NO | YES | Gold Value Dip Accumulation |
| 66 | 2021-03-26 | 2021-04-29 | ₹38.79 | ₹40.72 | +4.98% | +3.09% | ₹39.99 | 1.89 | YES | YES | Gold Value Dip Accumulation |
| 67 | 2021-04-29 | 2021-05-28 | ₹40.72 | ₹42.22 | +3.68% | +2.92% | ₹41.91 | 0.76 | YES | YES | Gold Value Dip Accumulation |
| 68 | 2021-05-28 | 2021-06-29 | ₹42.22 | ₹40.73 | -3.53% | +2.77% | ₹43.39 | 6.30 | NO | YES | Gold Value Dip Accumulation |
| 69 | 2021-06-29 | 2021-07-29 | ₹40.73 | ₹41.68 | +2.33% | +2.99% | ₹41.95 | 0.66 | YES | YES | Gold Value Dip Accumulation |
| 70 | 2021-07-29 | 2021-08-27 | ₹41.68 | ₹40.86 | -1.97% | +2.97% | ₹42.92 | 4.94 | NO | YES | Gold Value Dip Accumulation |
| 71 | 2021-08-27 | 2021-09-29 | ₹40.86 | ₹40.02 | -2.06% | +3.09% | ₹42.12 | 5.15 | NO | YES | Gold Value Dip Accumulation |
| 72 | 2021-09-29 | 2021-10-29 | ₹40.02 | ₹41.50 | +3.70% | +3.28% | ₹41.33 | 0.41 | YES | YES | Gold Value Dip Accumulation |
| 73 | 2021-10-29 | 2021-11-29 | ₹41.50 | ₹41.62 | +0.29% | +4.02% | ₹43.17 | 3.73 | YES | YES | Safe-Haven & Inflation Breakout |
| 74 | 2021-11-29 | 2021-12-29 | ₹41.62 | ₹41.50 | -0.29% | +2.80% | ₹42.78 | 3.09 | NO | YES | Secular Currency Drift |
| 75 | 2021-12-29 | 2022-01-28 | ₹41.50 | ₹41.43 | -0.17% | +2.87% | ₹42.69 | 3.03 | NO | YES | Secular Currency Drift |
| 76 | 2022-01-28 | 2022-02-28 | ₹41.43 | ₹43.59 | +5.21% | +2.88% | ₹42.62 | 2.34 | YES | YES | Secular Currency Drift |
| 77 | 2022-02-28 | 2022-03-29 | ₹43.59 | ₹44.18 | +1.35% | +3.95% | ₹45.31 | 2.59 | YES | YES | Safe-Haven & Inflation Breakout |
| 78 | 2022-03-29 | 2022-04-29 | ₹44.18 | ₹44.79 | +1.38% | +3.98% | ₹45.94 | 2.60 | YES | YES | Safe-Haven & Inflation Breakout |
| 79 | 2022-04-29 | 2022-05-27 | ₹44.79 | ₹44.08 | -1.59% | +4.01% | ₹46.59 | 5.60 | NO | YES | Safe-Haven & Inflation Breakout |
| 80 | 2022-05-27 | 2022-06-29 | ₹44.08 | ₹43.76 | -0.73% | +2.92% | ₹45.37 | 3.65 | NO | YES | Secular Currency Drift |
| 81 | 2022-06-29 | 2022-07-29 | ₹43.76 | ₹44.06 | +0.69% | +3.01% | ₹45.08 | 2.32 | YES | YES | Secular Currency Drift |
| 82 | 2022-07-29 | 2022-08-29 | ₹44.06 | ₹43.85 | -0.48% | +3.01% | ₹45.38 | 3.48 | NO | YES | Secular Currency Drift |
| 83 | 2022-08-29 | 2022-09-29 | ₹43.85 | ₹42.90 | -2.17% | +3.07% | ₹45.19 | 5.23 | NO | YES | Secular Currency Drift |
| 84 | 2022-09-29 | 2022-10-28 | ₹42.90 | ₹43.28 | +0.89% | +3.24% | ₹44.29 | 2.35 | YES | YES | Secular Currency Drift |
| 85 | 2022-10-28 | 2022-11-29 | ₹43.28 | ₹45.10 | +4.21% | +3.24% | ₹44.68 | 0.96 | YES | YES | Secular Currency Drift |
| 86 | 2022-11-29 | 2022-12-29 | ₹45.10 | ₹46.61 | +3.35% | +4.34% | ₹47.06 | 0.99 | YES | YES | Safe-Haven & Inflation Breakout |
| 87 | 2022-12-29 | 2023-01-27 | ₹46.61 | ₹48.78 | +4.66% | +4.24% | ₹48.59 | 0.41 | YES | YES | Safe-Haven & Inflation Breakout |
| 88 | 2023-01-27 | 2023-02-28 | ₹48.78 | ₹47.41 | -2.81% | +4.12% | ₹50.79 | 6.93 | NO | NO | Safe-Haven & Inflation Breakout |
| 89 | 2023-02-28 | 2023-03-29 | ₹47.41 | ₹50.69 | +6.92% | +3.09% | ₹48.88 | 3.83 | YES | YES | Secular Currency Drift |
| 90 | 2023-03-29 | 2023-04-28 | ₹50.69 | ₹51.25 | +1.10% | +4.07% | ₹52.75 | 2.96 | YES | YES | Safe-Haven & Inflation Breakout |
| 91 | 2023-04-28 | 2023-05-29 | ₹51.25 | ₹51.18 | -0.14% | +2.53% | ₹52.55 | 2.67 | NO | YES | Overbought Froth & Consolidation |
| 92 | 2023-05-29 | 2023-06-28 | ₹51.18 | ₹49.55 | -3.18% | +2.84% | ₹52.63 | 6.03 | NO | YES | Secular Currency Drift |
| 93 | 2023-06-28 | 2023-07-28 | ₹49.55 | ₹50.67 | +2.26% | +3.06% | ₹51.06 | 0.80 | YES | YES | Secular Currency Drift |
| 94 | 2023-07-28 | 2023-08-29 | ₹50.67 | ₹49.98 | -1.36% | +4.27% | ₹52.83 | 5.63 | NO | YES | Safe-Haven & Inflation Breakout |
| 95 | 2023-08-29 | 2023-09-29 | ₹49.98 | ₹49.29 | -1.38% | +3.16% | ₹51.56 | 4.55 | NO | YES | Secular Currency Drift |
| 96 | 2023-09-29 | 2023-10-27 | ₹49.29 | ₹51.64 | +4.77% | +3.27% | ₹50.90 | 1.49 | YES | YES | Secular Currency Drift |
| 97 | 2023-10-27 | 2023-11-29 | ₹51.64 | ₹53.05 | +2.73% | +4.35% | ₹53.89 | 1.62 | YES | YES | Safe-Haven & Inflation Breakout |
| 98 | 2023-11-29 | 2023-12-29 | ₹53.05 | ₹53.66 | +1.15% | +4.29% | ₹55.33 | 3.14 | YES | YES | Safe-Haven & Inflation Breakout |
| 99 | 2023-12-29 | 2024-01-29 | ₹53.66 | ₹52.97 | -1.29% | +3.06% | ₹55.30 | 4.34 | NO | YES | Secular Currency Drift |
| 100 | 2024-01-29 | 2024-02-29 | ₹52.97 | ₹52.78 | -0.36% | +3.16% | ₹54.65 | 3.52 | NO | YES | Secular Currency Drift |
| 101 | 2024-02-29 | 2024-03-28 | ₹52.78 | ₹56.61 | +7.26% | +3.23% | ₹54.48 | 4.03 | YES | YES | Secular Currency Drift |
| 102 | 2024-03-28 | 2024-04-29 | ₹56.61 | ₹61.30 | +8.28% | +4.21% | ₹58.99 | 4.07 | YES | YES | Safe-Haven & Inflation Breakout |
| 103 | 2024-04-29 | 2024-05-29 | ₹61.30 | ₹61.20 | -0.16% | +2.31% | ₹62.72 | 2.48 | NO | YES | Overbought Froth & Consolidation |
| 104 | 2024-05-29 | 2024-06-28 | ₹61.20 | ₹60.63 | -0.93% | +2.60% | ₹62.79 | 3.53 | NO | YES | Secular Currency Drift |
| 105 | 2024-06-28 | 2024-07-29 | ₹60.63 | ₹58.78 | -3.05% | +2.68% | ₹62.26 | 5.73 | NO | YES | Secular Currency Drift |
| 106 | 2024-07-29 | 2024-08-29 | ₹58.78 | ₹60.57 | +3.05% | +2.90% | ₹60.49 | 0.14 | YES | YES | Secular Currency Drift |
| 107 | 2024-08-29 | 2024-09-27 | ₹60.57 | ₹63.48 | +4.80% | +4.08% | ₹63.04 | 0.72 | YES | YES | Safe-Haven & Inflation Breakout |
| 108 | 2024-09-27 | 2024-10-29 | ₹63.48 | ₹66.30 | +4.44% | +3.91% | ₹65.96 | 0.53 | YES | YES | Safe-Haven & Inflation Breakout |
| 109 | 2024-10-29 | 2024-11-29 | ₹66.30 | ₹64.29 | -3.03% | +3.78% | ₹68.80 | 6.81 | NO | NO | Safe-Haven & Inflation Breakout |
| 110 | 2024-11-29 | 2024-12-27 | ₹64.29 | ₹64.21 | -0.12% | +2.74% | ₹66.05 | 2.87 | NO | YES | Secular Currency Drift |
| 111 | 2024-12-27 | 2025-01-29 | ₹64.21 | ₹67.86 | +5.68% | +2.81% | ₹66.02 | 2.87 | YES | YES | Secular Currency Drift |
| 112 | 2025-01-29 | 2025-02-28 | ₹67.86 | ₹71.10 | +4.77% | +2.27% | ₹69.40 | 2.50 | YES | YES | Overbought Froth & Consolidation |
| 113 | 2025-02-28 | 2025-03-28 | ₹71.10 | ₹74.16 | +4.30% | +3.62% | ₹73.67 | 0.68 | YES | YES | Safe-Haven & Inflation Breakout |
| 114 | 2025-03-28 | 2025-04-29 | ₹74.16 | ₹80.04 | +7.93% | +1.92% | ₹75.58 | 6.01 | YES | YES | Overbought Froth & Consolidation |
| 115 | 2025-04-29 | 2025-05-29 | ₹80.04 | ₹79.10 | -1.17% | +3.14% | ₹82.55 | 4.31 | NO | YES | Safe-Haven & Inflation Breakout |
| 116 | 2025-05-29 | 2025-06-27 | ₹79.10 | ₹79.81 | +0.90% | +2.00% | ₹80.68 | 1.10 | YES | YES | Secular Currency Drift |
| 117 | 2025-06-27 | 2025-07-29 | ₹79.81 | ₹81.91 | +2.63% | +1.98% | ₹81.39 | 0.65 | YES | YES | Secular Currency Drift |
| 118 | 2025-07-29 | 2025-08-29 | ₹81.91 | ₹85.34 | +4.19% | +1.87% | ₹83.44 | 2.32 | YES | YES | Secular Currency Drift |
| 119 | 2025-08-29 | 2025-09-29 | ₹85.34 | ₹95.90 | +12.37% | +1.39% | ₹86.53 | 10.98 | YES | NO | Overbought Froth & Consolidation |
| 120 | 2025-09-29 | 2025-10-29 | ₹95.90 | ₹99.95 | +4.22% | +0.80% | ₹96.67 | 3.42 | YES | YES | Overbought Froth & Consolidation |
| 121 | 2025-10-29 | 2025-11-28 | ₹99.95 | ₹104.62 | +4.67% | +2.18% | ₹102.13 | 2.49 | YES | YES | Safe-Haven & Inflation Breakout |
| 122 | 2025-11-28 | 2025-12-29 | ₹104.62 | ₹113.03 | +8.04% | +2.02% | ₹106.73 | 6.02 | YES | YES | Safe-Haven & Inflation Breakout |
| 123 | 2025-12-29 | 2026-01-29 | ₹113.03 | ₹146.53 | +29.64% | +0.13% | ₹113.18 | 29.51 | YES | NO | Overbought Froth & Consolidation |
| 124 | 2026-01-29 | 2026-02-27 | ₹146.53 | ₹131.60 | -10.19% | -1.26% | ₹144.69 | 8.93 | YES | NO | Overbought Froth & Consolidation |
| 125 | 2026-02-27 | 2026-03-27 | ₹131.60 | ₹117.53 | -10.69% | -0.04% | ₹131.55 | 10.66 | YES | NO | Gold Value Dip Accumulation |
| 126 | 2026-03-27 | 2026-04-29 | ₹117.53 | ₹121.58 | +3.45% | +0.53% | ₹118.15 | 2.92 | YES | YES | Gold Value Dip Accumulation |
| 127 | 2026-04-29 | 2026-05-29 | ₹121.58 | ₹128.65 | +5.82% | +0.45% | ₹122.13 | 5.36 | YES | YES | Gold Value Dip Accumulation |
| 128 | 2026-05-29 | 2026-06-29 | ₹128.65 | ₹116.43 | -9.50% | +0.20% | ₹128.90 | 9.70 | NO | NO | Gold Value Dip Accumulation |
| 129 | 2026-06-29 | 2026-07-29 | ₹116.43 | ₹116.85 | +0.36% | +0.74% | ₹117.29 | 0.38 | YES | YES | Gold Value Dip Accumulation |
| 130 | 2026-07-29 | 2026-08-28 | ₹116.85 | ₹130.85 | +11.98% | +0.82% | ₹117.81 | 11.16 | YES | NO | Gold Value Dip Accumulation |
| 131 | 2026-08-28 | 2026-09-28 | ₹130.85 | ₹121.17 | -7.40% | +0.25% | ₹131.17 | 7.64 | NO | NO | Gold Value Dip Accumulation |
