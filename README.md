
[![TimeSeriesYenforFuture](https://github.com/benjaminweymouth/Time-Series-analysis-using-ARIMA/blob/main/images/headerimagev3.png)](https://github.com/benjaminweymouth/Time-Series-analysis-using-ARIMA)



# Time Series: A Yen for the Future

### Introduction 
This repository holds 2 Jupyter notebooks and one csv file on  Time Series analysis for the A Yen for the Future exercises. The purpose of this code is to demonstrate understanding of time series work in Python: ARMA, ARIMA and related concepts.

## Background

The financial departments of large companies often have to make foreign currency transactions when doing international business, while hedge funds are also interested in anything that will provide an edge in predicting currency movements. Hence, both are always eager to gain a better understanding of the future direction and risk of various currencies. 

This repository uses time series analysis to predict future movements in the value of the Canadian dollar versus the Japanese yen.

Specifically, this repo reflects two main parts:

1. Time series forecasting
2. Linear regression modelling

## Steps (in 2 main parts) 

## Part 1: Time-Series Forecasting
The first step is to load historical CAD-JPY exchange rate data and apply time series analysis and modelling to determine if there is any predictable behaviour.

Following that, these steps are executed:

1. Plotting the Settle price to check for long or short-term patterns.

2. Decomposition using a Hodrick-Prescott filter (decompose the settle price into trend and noise).

3. Forecasting returns using an ARMA model. Based on the p-value, we can address if the model is a good fit.

4. Forecasting the exchange rate price using an ARIMA model. The forecast may indicated what will happen to the Japanese Yen in the near term.

5. Forecasting volatility with GARCH.

## Part 2: Linear Regression Forecasting

In this notebook, you will build a Scikit-Learn linear regression model to predict CAD/JPY returns with *lagged* CAD/JPY futures returns and categorical calendar seasonal effects (e.g., day-of-week or week-of-year seasonal effects).

Follow the steps outlined in the regression_analysis starter notebook to complete the following:

1. Data preparation (creating returns and lagged returns, and splitting the data into training and testing data)
2. Fitting a linear regression model.
3. Making predictions using the testing data.
4. Out-of-sample performance.
5. In-sample performance.

Use the results of the linear regression analysis and modelling to answer the following question:

* Does this model perform better or worse on out-of-sample data compared to in-sample data?
