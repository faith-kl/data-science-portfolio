# House Price Prediction: Multiple Regression Analysis

A comprehensive machine learning project comparing OLS, Ridge, and Lasso regression models to predict residential property prices using 500 recent home sales data.

## Project Overview

This project analyzes the VertexVault house prices dataset to build accurate prediction models for real estate valuation. Through systematic feature engineering, model comparison, and diagnostic analysis, we identify the optimal approach for predicting house prices based on property characteristics.

## Business Objective

Develop a robust predictive model that real estate professionals can use to:
- Estimate fair market value for residential properties
- Identify key price drivers in the housing market
- Support data-driven pricing decisions for buyers and sellers

## Dataset

**Source:** VertexVault_House_Prices2.csv  
**Size:** 500 observations, 11 features  
**Target Variable:** Sale price (in thousands of dollars)

### Features:
- **Continuous:** sqft, lot_size, age, school_dist, park_dist
- **Discrete:** bedrooms, bathrooms, condition (1-5 rating), garage
- **Binary:** has_basement (0/1)

## Analysis Pipeline

### 1. **Exploratory Data Analysis**
- Statistical summaries and distribution analysis
- Missing value assessment (0% missing data)
- Outlier detection using IQR method
- Correlation analysis and multicollinearity checks

### 2. **Feature Engineering**
- Log transformation of target variable to reduce skewness
- StandardScaler normalization for continuous features
- Creation of interaction features (sqft², sqft×bedrooms, sqft×bathrooms)
- Train-test split (80/20) with random state for reproducibility

### 3. **Model Development**
Three regression approaches were implemented and compared:
- **Ordinary Least Squares (OLS)** - Baseline linear regression
- **Ridge Regression (L2)** - Regularization to handle multicollinearity
- **Lasso Regression (L1)** - Feature selection through coefficient shrinkage

### 4. **Model Evaluation**
Performance metrics across all models:
- R² and Adjusted R² for explanatory power
- RMSE and MAE for prediction accuracy
- Residual analysis for assumption validation
- Cross-validation for generalization assessment

## Key Results

### Model Performance Comparison

| Model | Test R² | Test RMSE | Test MAE | Key Characteristics |
|-------|---------|-----------|----------|---------------------|
| **OLS** | 0.923 | 0.0313 | 0.0253 | Strong baseline, interpretable |
| **Ridge** | 0.922 | 0.0313 | 0.0253 | Stable, handles multicollinearity |
| **Lasso** | 0.755 | 0.0556 | 0.0445 | Feature selection, underfits |
| **Enhanced Ridge** | **0.926** | **0.0305** | **0.0248** | Best performance with interactions |

### Most Influential Features
1. **sqft** (coefficient: 0.0683) - Largest positive effect
2. **sqft²** (nonlinear relationship captured in enhanced model)
3. **has_basement** (coefficient: 0.0391) - Significant premium
4. **condition** (coefficient: 0.0289) - Quality rating impact
5. **bathrooms** (coefficient: 0.0312) - Strong positive correlation

### Model Improvements
- Enhanced Ridge with interaction features improved test R² by 0.4%
- RMSE reduced by 2.6% through polynomial feature engineering
- Residual heteroskedasticity substantially reduced
- Better generalization: Adjusted R² improved from 0.914 to 0.917

## Recommendation

**Ridge Regression with Interaction Features** is recommended for production deployment because:
- Highest predictive accuracy (R² = 0.926)
- Robust to multicollinearity through L2 regularization
- Stable predictions across train/test sets
- Captures non-linear relationships via engineered features
- Lower prediction errors (RMSE = 0.0305)

## Technologies Used

- **Python 3.9+**
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn, statsmodels
- **Statistical Analysis:** scipy

## Project Structure

```
house-price-prediction/
│
├── data/
│   └── VertexVault_House_Prices2.csv
│
├── notebooks/
│   └── house_price_analysis.ipynb
│
├── figures/
│   ├── price_distribution.png
│   ├── correlation_heatmap.png
│   ├── residual_plots.png
│   └── feature_importance.png
│
├── models/
│   ├── scaler.pkl
│   └── ridge_model.pkl
│
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels scipy
```

### Running the Analysis
```python
# Load and preprocess data
df = pd.read_csv('data/VertexVault_House_Prices2.csv')

# Train the enhanced Ridge model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Create interaction features
X['sqft_squared'] = X['sqft'] ** 2
X['sqft_bedrooms'] = X['sqft'] * X['bedrooms']
X['sqft_bathrooms'] = X['sqft'] * X['bathrooms']

# Scale and train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = Ridge(alpha=1.0)
model.fit(X_scaled, np.log(y))

# Predict
prediction = np.exp(model.predict(new_house_scaled))
```

### Example Prediction
For a house with:
- 2,500 sqft, 4 bedrooms, 2.5 bathrooms
- 15 years old, 8,000 sqft lot
- 1.2 miles to school, 0.8 miles to park
- Condition: 4/5, 2-car garage, has basement

**Predicted Price:** ~$1,050,000

## Visualizations

Key plots generated in this analysis:
- Price distribution (raw vs. log-transformed)
- Feature correlation heatmap
- Scatter plots: price vs. all numerical features
- Residual plots for model diagnostics
- Q-Q plots for normality assessment

## Model Diagnostics

### Assumptions Validated
- **Linearity:** Confirmed through residual scatter patterns
- **Homoskedasticity:** Improved with log transformation and interactions
- **Normality:** Residuals approximately normal (Q-Q plot)
- **Independence:** No autocorrelation detected (Durbin-Watson ≈ 2.0)
- **Multicollinearity:** Managed through Ridge regularization

## Future Enhancements

- [ ] Implement ensemble methods (Random Forest, Gradient Boosting)
- [ ] Add geospatial features (neighborhood, ZIP code)
- [ ] Include temporal features (market trends, seasonality)
- [ ] Cross-validation with multiple random states
- [ ] Deploy model as REST API for real-time predictions
- [ ] A/B testing framework for model updates

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Last Updated:** January 2026  
**Status:** In progress for improvement.
