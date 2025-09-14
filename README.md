# GARCH-ML-Volatility-Forecasting
GARCH &amp; ML volatility forecasting engine for the FTSE 100 (2018–2025), reusing the same Monte Carlo VaR dataset, benchmarking econometric (GARCH(1,1)) and tree-based ML models (RF/GBRT) against realised variance with MSE and QLIKE


# GARCH & ML Volatility Forecasting Engine (Independent Research)

Period: Aug 2025 – Sep 2025  
Author: Trevor Higgins   

## Overview
This project develops a volatility forecasting engine for FTSE 100 (2018–2025) data, reusing the same cleaned dataset from the Monte Carlo VaR Engine.  
It benchmarks econometric modelling (GARCH(1,1)) against machine learning approaches (Random Forest, Gradient Boosting) and then it evaluates forecasts against realised variance.

## Workflow
1. **Data Preparation**  
   - Source: FTSE 100 historical data from [Investing.com](https://uk.investing.com/indices/uk-100-historical-data?cid=27517).  
   - Data set was already cleaned from the Monte Carlo VaR engine

2. **Econometric Baseline (GARCH(1,1))**  
   - Rolling 1-day-ahead conditional variance forecasts.  
   - Captures volatility clustering and persistence.  

3. **Machine Learning Benchmarks**  
   - Random Forest (RF) and Gradient Boosting (GBRT).  
   - Features: lagged returns, moving averages, rolling volatilities, trading volume.  
   - Target: next-day realised variance.  

4. **Evaluation Metrics**  
   - MSE (Mean Squared Error).  
   - QLIKE (robust variance loss function).  
   - Naïve baseline: yesterday’s realised variance.  

5. **Backtesting (2018–2025)**  
   - Rolling forecasts with periodic refitting.  
   - Benchmarked GARCH vs RF vs GBRT across calm and volatile regimes.  
   - Results:  
     - GARCH → strong gains (≈44% MSE, ≈100% QLIKE improvement vs naïve).  
     - RF/GBRT → weaker, less stable forecasts.  

## Key Results
| Model         | ΔMSE vs Naïve | ΔQLIKE vs Naïve | Notes                  |
|---------------|---------------|-----------------|------------------------|
| Naïve         | 0%            | 0%              | Benchmark              |
| GARCH(1,1)    | **~44%**      | **~100%**       | Best performer         |
| Random Forest | Small gains   | Weak            | Limited predictive use |
| GBRT          | Small gains   | Weak            | Less stable forecasts  |

*Dataset: FTSE 100 daily returns (2018–2025); horizon: 1-day.*  

## Installation
```bash
git clone <your-repo>
cd vol-forecast-engine
pip install -r requirements.txt

