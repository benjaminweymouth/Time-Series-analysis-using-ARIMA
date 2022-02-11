
[![TimeSeriesYenforFuture](https://github.com/benjaminweymouth/Time-Series-analysis-using-ARIMA/blob/main/images/headerimagev3.png)](https://github.com/benjaminweymouth/Time-Series-analysis-using-ARIMA)

# Time Series: A Yen for the Future

### Introduction 
This repository holds 2 Jupyter notebooks and one csv file on  Time Series analysis for the A Yen for the Future exercises. The purpose of this code is to demonstrate understanding of time series work in Python: ARMA, ARIMA and related concepts.

### Deployed Live Link (live pages for each jupyter notebook) 

https://benjaminweymouth.github.io/Time-Series-analysis-using-ARIMA/

## Background

The financial departments of large companies often have to make foreign currency transactions when doing international business, while hedge funds are also interested in anything that will provide an edge in predicting currency movements. Hence, both are always eager to gain a better understanding of the future direction and risk of various currencies. 

This repository uses time series analysis to predict future movements in the value of the Canadian dollar versus the Japanese yen.

Specifically, this repo reflects two main parts:

1. Time series forecasting
2. Linear regression modelling

## Steps (in 2 main parts) 

## Part 1: Time-Series Forecasting

Live Link for time series: https://benjaminweymouth.github.io/Time-Series-analysis-using-ARIMA/TimeSeries/

The first step is to load historical CAD-JPY exchange rate data and apply time series analysis and modelling to determine if there is any predictable behaviour.

Following that, these steps are executed:

1. Plotting the Settle price to check for long or short-term patterns.

![image](https://user-images.githubusercontent.com/47256041/153657740-c9a7d573-ee0b-4ea6-92d6-4d19e4ce8ef3.png)


2. Decomposition using a Hodrick-Prescott filter (decompose the settle price into trend and noise).

![image](https://user-images.githubusercontent.com/47256041/153657808-ae98de00-086c-469f-84ac-5795de2dbd30.png)


3. Forecasting returns using an ARMA model. Based on the p-value, we can address if the model is a good fit.

![image](https://user-images.githubusercontent.com/47256041/153657853-946c4485-67d4-44ff-a2c2-200625b619dc.png)


4. Forecasting the exchange rate price using an ARIMA model. The forecast may indicated what will happen to the Japanese Yen in the near term.

![image](https://user-images.githubusercontent.com/47256041/153657951-98137082-5eb1-4a61-b1f3-eab426afd5e0.png)


5. Forecasting volatility with GARCH.

![image](https://user-images.githubusercontent.com/47256041/153658005-62299bc3-75cf-4097-9613-ca309433e9e0.png)


## Part 2: Linear Regression Forecasting

Live link for linear regression: https://benjaminweymouth.github.io/Time-Series-analysis-using-ARIMA/LinearRegression/

In this notebook, you will build a Scikit-Learn linear regression model to predict CAD/JPY returns with *lagged* CAD/JPY futures returns and categorical calendar seasonal effects (e.g., day-of-week or week-of-year seasonal effects).

Follow the steps outlined in the regression_analysis starter notebook to complete the following:

1. Data preparation (creating returns and lagged returns, and splitting the data into training and testing data)

![image](https://user-images.githubusercontent.com/47256041/153658151-b8b7bc62-dd2f-49b2-95c8-4b63b2cc1970.png)


3. Fitting a linear regression model.

![image](https://user-images.githubusercontent.com/47256041/153658259-7967737e-4e7c-40fc-b312-9109202bb3f5.png)

5. Making predictions using the testing data.

![image](https://user-images.githubusercontent.com/47256041/153658312-f7a42024-7424-45c0-aa2f-fd03b60734af.png)


7. Out-of-sample performance.

![image](https://user-images.githubusercontent.com/47256041/153658394-d8f59155-7408-4d75-af86-06e975fa6a98.png)


9. In-sample performance.

![image](https://user-images.githubusercontent.com/47256041/153658434-a38474a2-fef3-4845-8e4d-1ccd75eb61c9.png)



## Conclusions: 
 
Based on your time series analysis, would you buy the yen now?

Answer: The volatility of the Yen indicated by the GARCH model indicates that purchasing the yen now would not be a wise investment option.

Is the risk of the yen expected to increase or decrease?

Answer: The volatility of the Yen predicts that the risk associated with the Yen is on the rise. However, it should be noted that this only a short term conclusion. In the future the risk may vary upwards or downwards depending on a variety of factors.

Based on the model evaluation, would you feel confident in using these models for trading?

Answer: The fit of a model should be determined by p-value >α. These models have shown that they are not a good fit, and would therefore require further modifications and calibrations to be fit for trading purposes. They could be tweaked, and over time may be suitable. But at the present time these p-values indicate that the models are not a good fit- and therefore not suitable for trading purposes.
