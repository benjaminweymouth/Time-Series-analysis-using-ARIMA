
[![TimeSeriesYenforFuture](https://github.com/benjaminweymouth/Time-Series-analysis-using-ARIMA/blob/main/images/headerimagev3.png)](https://github.com/benjaminweymouth/Time-Series-analysis-using-ARIMA)



# Time Series: A Yen for the Future

### Introduction 
This repository holds 2 Jupyter notebooks and one csv file on  Time Series analysis for the A Yen for the Future exercises. The purpose of this code is to demonstrate understanding of time series work in Python: ARMA, ARIMA and related concepts.

## Background

The financial departments of large companies often have to make foreign currency transactions when doing international business, while hedge funds are also interested in anything that will provide an edge in predicting currency movements. Hence, both are always eager to gain a better understanding of the future direction and risk of various currencies. 

This repository uses time series analysis to predict future movements in the value of the Canadian dollar versus the Japanese yen.

Specifically, this repo reflects two core items:

1. Time series forecasting
2. Linear regression modelling

## Steps 

The first step is to load historical CAD-JPY exchange rate data and apply time series analysis and modelling to determine if there is any predictable behaviour.

Following that, these steps are executed:

1. Plotting the Settle price to check for long or short-term patterns.

2. Decomposition using a Hodrick-Prescott filter (decompose the settle price into trend and noise).

3. Forecasting returns using an ARMA model. Based on the p-value, we can address if the model is a good fit.

4. Forecasting the exchange rate price using an ARIMA model. The forecast may indicated what will happen to the Japanese Yen in the near term.

5. Forecasting volatility with GARCH.
