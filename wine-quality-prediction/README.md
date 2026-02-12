# Wine Quality Classification: Multinomial Logistic Regression

A statistical modeling project predicting Portuguese Vinho Verde red wine quality tiers using physicochemical properties through multinomial logistic regression with stepwise AIC-based feature selection.

## Project Overview

This project addresses a practical challenge in wine production: efficiently categorizing wine quality without relying solely on time-intensive sensory evaluation by sommeliers. By leveraging 11 laboratory-measurable physicochemical features, we develop a predictive model that classifies wines into three business-relevant quality tiers: **Low**, **Medium**, and **Premium**.

### Business Context
- **Premium wines** (8-10% of production) command 3-5× higher prices
- **Medium-quality wines** (65-70%) require competitive pricing accuracy  
- **Low-quality wines** (20-25%) need immediate identification to prevent market release
- Misclassification carries significant financial consequences and brand reputation risks

## Objectives

1. Build a multinomial logistic regression model to classify wine quality into three ordered categories
2. Handle severe class imbalance in the dataset using SMOTE and class weighting techniques
3. Identify the most predictive physicochemical features through stepwise AIC selection
4. Achieve interpretable, actionable insights for winery production decisions

## Dataset

**Source:** [UCI Machine Learning Repository - Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)

**Characteristics:**
- **Observations:** 1,143 Portuguese Vinho Verde red wines (expanded to 4,256 post-SMOTE)
- **Features:** 11 continuous physicochemical predictors
- **Target Variable:** Quality scores (0-10) recoded into Low (3-5), Medium (6-7), Premium (8+)
- **Missingness:** 0% across all variables
- **Key Challenge:** Severe class imbalance with sparse observations at quality extremes

### Features
**Main Effects (7 variables):**
- `volatile acidity` – acetic acid content (g/L)
- `fixed acidity` – non-volatile acids (g/L)  
- `residual sugar` – remaining sugar post-fermentation (g/L)
- `chlorides` – salt content (g/L)
- `sulphates` – sulfate additives (g/L)
- `alcohol` – alcohol by volume (%)
- `dcolor` – binary indicator for color intensity

**Engineered Interaction Terms (2 variables):**
- `free sulfur dioxide × alcohol` – joint preservation and alcohol effect
- `chlorides × total sulfur dioxide` – salt and SO₂ interaction

## Methodology

### Data Preprocessing
- **Target transformation:** Collapsed quality scores into three ordered categories aligned with business decisions
- **Feature standardization:** Z-score normalization applied to all predictors (essential given vastly different measurement scales)
- **Train-test split:** 80-20 stratified split preserving class proportions

### Handling Class Imbalance
- **SMOTE (Synthetic Minority Over-sampling Technique):** Generated synthetic observations for underrepresented Low and Premium classes through feature-space interpolation
- **Class weighting:** Applied inverse-frequency penalties during model training (10× penalty for misclassifying rare Premium wines)

### Feature Selection: Stepwise AIC
- **Method:** Bidirectional stepwise selection with Akaike Information Criterion (AIC)
- **Convergence:** 20 iterations until no further AIC improvement
- **Final model:** 16 features selected (2 main effects + 14 interaction terms)
- **Rationale:** Prioritized predictive accuracy over extreme parsimony given high-stakes business context

**AIC vs BIC Trade-off:**
- AIC model: 16 features, AIC=3321.63, Pseudo R²=0.2604
- BIC model: 6 features, BIC=3530.55, Pseudo R²=0.2240
- **Decision:** Selected AIC for 16.2% more variance explained, justified by financial consequences of misclassification

### Model Specification: Multinomial Logistic Regression (GLM)

**Log-Odds Form:**
```
log(P(Yi = j) / P(Yi = Premium)) = β_j0 + β_j1*X_i1 + β_j2*X_i2 + ... + β_jp*X_ip
```

**Probability Transformation:**
```
P(Yi = j) = exp(Xi'β_j) / [1 + Σ exp(Xi'β_k)]  for k ∈ {Low, Medium}
P(Yi = Premium) = 1 / [1 + Σ exp(Xi'β_k)]
```

Where:
- `j ∈ {Low, Medium}` (Premium serves as reference category)
- `β_j` = coefficient vector for category j
- `Xi` = vector of predictor variables for observation i

**Estimation:** Maximum Likelihood Estimation (MLE)

### Model Validation
- **Cross-validation:** Stratified 5-fold CV maintaining original class proportions
- **Metrics:** Macro-averaged F1-score, per-class precision/recall, confusion matrix (accuracy avoided due to imbalance)
- **Multicollinearity assessment:** VIF scores monitored; elevated VIFs (alcohol=40.18, fixed acidity=32.47) retained due to domain importance and statistical significance

## Results

### Model Performance
- **Final AIC:** 3321.63
- **Pseudo R² (McFadden):** 0.2604 (26% reduction in deviance vs. null model)
- **Log-Likelihood:** -1626.82
- **LLR p-value:** < 0.001 (highly significant improvement over intercept-only model)
- **Convergence:** Achieved in 20 stepwise iterations

### Key Predictors (Selected Features)

**Top Predictors for Low vs Premium:**
| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `volatile acidity` | +28.58*** | Strong predictor of lower quality (acetic acid defects) |
| `alcohol` | -0.92*** | Higher alcohol associates with premium quality |
| `chlorides` | +14.51** | Higher salt content predicts lower quality |
| `total SO₂ × pH` | -0.035*** | pH-dependent preservation effectiveness |

**Top Predictors for Medium vs Premium:**
| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `volatile acidity` | +19.61*** | Still predicts lower quality, weaker effect than Low tier |
| `alcohol` | -0.85*** | Strongest single predictor across both categories |
| `free SO₂ × alcohol` | -0.002*** | Negative interaction—combined effect non-additive |

*Significance: ***p<0.001, **p<0.01, *p<0.05*

### Insights
1. **Volatile acidity** is the strongest chemical defect indicator—higher acetic acid strongly predicts lower quality tiers
2. **Alcohol content** consistently discriminates premium wines—each 1% increase substantially favors Premium classification
3. **Interaction terms dominate** the final model (14 of 16 features)—wine quality emerges from complex chemical interactions, not isolated properties
4. **Sulfur dioxide preservation** effectiveness is pH-dependent, captured through interaction terms
5. **Acid-sugar balance** (fixed acidity × residual sugar) significantly impacts quality perception

## Technologies Used

- **Python 3.9+**
- **Statistical Modeling:** `statsmodels` (MNLogit for multinomial logistic regression)
- **Data Manipulation:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Resampling:** `imblearn` (SMOTE for class imbalance)
- **Multicollinearity:** `statsmodels.stats.outliers_influence` (VIF calculation)

## Key Learnings

1. **Real-world class imbalance:** Successfully addressed severe imbalance using combined SMOTE + class weighting strategy
2. **Feature interactions matter:** Domain-driven interaction terms (14/16 features) captured complex chemical synergies
3. **Model selection trade-offs:** AIC vs BIC comparison demonstrated the importance of aligning statistical criteria with business objectives
4. **Interpretability in production:** Multinomial logistic regression provided transparent, auditable predictions critical for stakeholder buy-in
5. **Multicollinearity nuance:** Elevated VIF scores acceptable when features are statistically significant and domain-essential

## Future Work

- [ ] **Ensemble methods:** Compare performance against Random Forest, XGBoost to assess non-linear modeling gains
- [ ] **Feature importance:** Apply SHAP values for more granular feature contribution analysis
- [ ] **Threshold optimization:** ROC/PR curve analysis to optimize classification thresholds for business cost functions
- [ ] **External validation:** Test model generalization on white Vinho Verde wines or other wine regions
- [ ] **Deployment:** Build Flask API for real-time quality prediction in production environments
- [ ] **Temporal validation:** If vintage data available, test model stability across different harvest years

## Acknowledgments

- **Dataset:** UCI Machine Learning Repository, P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis (2009)
- **Inspiration:** Real-world winery quality control challenges and operational efficiency needs
- **References:** 
  - Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547-553.
  - Agresti, A. (2015). *Foundations of Linear and Generalized Linear Models*. Wiley.
