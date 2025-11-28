"""Generate final project report"""
import pandas as pd
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path("results")
OUTPUT_FILE = "FINAL_REPORT.md"

def generate_report():
    """Generate markdown report"""
    
    # Load results
    baseline_df = pd.read_csv(RESULTS_DIR / "tables" / "baseline_results.csv")
    boosting_df = pd.read_csv(RESULTS_DIR / "tables" / "boosting_results.csv")
    
    report = f"""# AI-Dominated Market Regime Detection - Final Report

**Team:** Ankit Paudel & Sohan Lama  
**Course:** CS 4771 - Machine Learning  
**Date:** {datetime.now().strftime('%B %d, %Y')}

---

## Executive Summary

This project developed machine learning models to predict short-horizon equity direction (1, 3, 5 days) for large-cap tech stocks (AAPL, GOOGL, MSFT, TSLA, NVDA) using technical indicators.

**Key Results:**
- **Best Model:** Logistic Regression for 1-day predictions (F1=0.415)
- **Data:** 13,615 observations spanning 2015-2025
- **Features:** 17 technical indicators (RSI, MACD, Bollinger, Volume, etc.)

---

## 1. Problem Statement

**Objective:** Predict equity price direction (BUY/SELL/HOLD) at 1, 3, and 5-day horizons.

**Labels:**
- BUY: Forward return > +2%
- SELL: Forward return < -2%
- HOLD: Forward return between -2% and +2%

**Challenge:** Stock prediction is inherently difficult due to market noise, non-stationarity, and class imbalance.

---

## 2. Data

### 2.1 Data Sources
- **Price Data:** Yahoo Finance (yfinance)
- **Tickers:** AAPL, GOOGL, MSFT, TSLA, NVDA
- **Period:** 2015-01-01 to 2025-10-30 (~11 years)
- **Total Observations:** 13,615 daily records

### 2.2 Features (17 total)
- **Returns:** 1, 3, 5, 10-day percentage changes
- **Volatility:** 10, 20-day realized volatility
- **Technical Indicators:** RSI(14), MACD, Bollinger Bands
- **Volume:** Z-score normalized volume
- **Price Action:** Gap percentage, range/ATR ratio

### 2.3 Label Distribution

**1-Day Horizon:**
- HOLD: 71.8% (9,704 samples)
- BUY: 15.1% (2,038 samples)
- SELL: 13.1% (1,773 samples)

**3-Day Horizon:**
- HOLD: 49.2%
- BUY: 29.3%
- SELL: 21.5%

**5-Day Horizon:**
- HOLD: 38.2%
- BUY: 36.6%
- SELL: 25.1%

**Observation:** Class imbalance decreases with longer horizons as price movements become larger.

---

## 3. Methodology

### 3.1 Models Evaluated
1. **Logistic Regression** (L2 regularization, class weights)
2. **Random Forest** (100 trees, class weights)
3. **LightGBM** (Gradient boosting)
4. **XGBoost** (Gradient boosting with sample weights)

### 3.2 Validation Strategy
- **Time Series Cross-Validation** (5 folds)
- Chronological splits to prevent look-ahead bias
- Training on past data, testing on future data

### 3.3 Evaluation Metrics
- **Primary:** Macro-averaged F1 Score (handles class imbalance)
- **Secondary:** Accuracy

---

## 4. Results

### 4.1 Model Performance Comparison

{baseline_df.to_markdown(index=False)}

{boosting_df.to_markdown(index=False)}

### 4.2 Best Models by Horizon

| Horizon | Best Model | F1-Macro | Accuracy |
|---------|-----------|----------|----------|
| 1-day | Logistic Regression | 0.415 | 0.552 |
| 3-day | Logistic Regression | 0.397 | 0.457 |
| 5-day | XGBoost | 0.392 | 0.426 |

### 4.3 Key Findings

1. **Logistic Regression outperforms complex models** on shorter horizons
   - Simpler models better for noisy, short-term data
   - Less prone to overfitting

2. **Performance decreases with longer horizons**
   - 1-day: F1 ≈ 0.41
   - 3-day: F1 ≈ 0.40
   - 5-day: F1 ≈ 0.39
   - Longer predictions are inherently harder

3. **Class imbalance is a major challenge**
   - HOLD class dominates, especially at 1-day
   - Models struggle with minority classes (BUY/SELL)

4. **XGBoost improves on longer horizons**
   - Better at capturing complex patterns in 5-day predictions

---

## 5. Challenges & Limitations

### 5.1 Challenges Encountered
- **Class Imbalance:** 72% HOLD labels at 1-day horizon
- **Market Noise:** Short-term price movements are highly stochastic
- **Feature Engineering:** Technical indicators have limited predictive power
- **Non-stationarity:** Market regimes change over time

### 5.2 Limitations
- **No sentiment data:** NewsAPI and Reddit data not implemented due to time constraints
- **No AI-Intensity Index:** Regime detection component not completed
- **Limited feature set:** Only technical indicators used
- **No transaction costs:** Backtesting not implemented

---

## 6. Conclusions

### 6.1 Project Outcomes
✅ Successfully built end-to-end ML pipeline for stock prediction  
✅ Evaluated 4 different model architectures  
✅ Achieved reasonable F1 scores (0.39-0.42) given task difficulty  
✅ Demonstrated proper time-series validation methodology  

### 6.2 Key Takeaways
1. **Stock prediction is hard** - F1 scores around 0.40 are reasonable
2. **Simple models can outperform complex ones** - especially on noisy data
3. **Shorter horizons are slightly more predictable** than longer ones
4. **Class imbalance matters** - must use proper weighting/metrics

### 6.3 Future Work
- Add sentiment features (news headlines, social media)
- Implement AI-Intensity Index for regime detection
- Build meta-labeling gate for selective execution
- Add cost-aware backtesting with realistic transaction costs
- Explore deep learning models (LSTM, Transformers)
- Test on more diverse asset classes

---

## 7. Reproducibility

### 7.1 Environment
- Python 3.x
- Key packages: pandas, scikit-learn, lightgbm, xgboost, yfinance

### 7.2 Code Structure
```
src/
├── data_ingestion/    # Data collection scripts
├── features/          # Feature engineering
├── models/            # Model training
├── evaluation/        # Evaluation & visualization
└── utils/             # Helper functions
```

### 7.3 Running the Pipeline
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch data
python src/data_ingestion/price_data.py

# 3. Create labels
python src/data_ingestion/create_labels.py

# 4. Engineer features
python src/features/technical.py

# 5. Train models
python src/models/baselines.py
python src/models/boosting.py
```

---

## 8. References

1. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
2. Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
3. Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. KDD.

---

**End of Report**

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open('report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_report()