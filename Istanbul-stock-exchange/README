# Istanbul Stock Exchange (ISE) Index Forecasting

## Project Overview

This project analyzes and forecasts the Istanbul Stock Exchange (ISE) index using regression-based modeling techniques. The goal was to explore the relationship between macroeconomic indicators and the ISE index and evaluate the effectiveness of statistical models for short-term forecasting.

The analysis includes data preprocessing, exploratory data analysis (EDA), correlation assessment, model building, and prediction using linear regression techniques.

## Objectives

* Explore relationships between the ISE index and financial indicators
* Identify significant predictors using correlation analysis
* Build a regression model to forecast future ISE values
* Evaluate model performance
* Generate a 10-day forecast using the trained model

## Dataset

The dataset includes:

* ISE Index (target variable)
* SP (S&P 500 Index)
* DAX
* FTSE
* NIKKEI
* BOVESPA
* Exchange rates and other macroeconomic indicators

The target variable:

> **ISE – Istanbul Stock Exchange index value**

## Exploratory Data Analysis

* Checked for missing values and cleaned the dataset
* Calculated Pearson correlations between variables
* Visualized relationships between ISE and global market indices
* Assessed trends and potential multicollinearity

Key insight:
Global market indices (e.g., SP, DAX, FTSE) show meaningful correlation with the ISE index.

## Model Development

A **Linear Regression model** was used to forecast the ISE index.

### Steps:

1. Selected relevant predictors based on correlation
2. Split data into training and testing sets
3. Fit a regression model using `lm`
4. Evaluated model performance using error metrics
5. Generated a 10-day forecast using `predict()`

## Forecasting

The model was used to predict the ISE index for the next 10 days based on the most recent available data.

Important note:
Because the regression model does not inherently model time dependence (like ARIMA or LSTM would), forecasts assume predictor stability unless new projected input values are provided.


## Tools & Libraries
* R

## Limitations
* Linear regression does not capture seasonality or time-dependent structure.
* Forecast quality depends heavily on predictor stability.
* Market volatility can reduce predictive accuracy.

Future improvements could include:
* ARIMA modeling
* VAR models
* LSTM neural networks
* Feature engineering for lag variables

## Key Skills Demonstrated

* Time series analysis
* Financial data interpretation
* Regression modeling
* Model evaluation
* Forecast generation
* Data visualization
* Critical thinking about model limitations
