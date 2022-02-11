# Return Forecasting: Time Series Analysis & Modelling with CAD-PHY Exchange rate data.
In this notebook, you will load historical Canadian Dollar-Yen exchange rate futures data and apply time series analysis and modeling to determine whether there is any predictable behavior.


```python
import numpy as np
import pandas as pd
from pathlib import Path
%matplotlib inline

import warnings
warnings.simplefilter(action='ignore', category=Warning)
```


```python
# Currency pair exchange rates for CAD/JPY
cad_jpy_df = pd.read_csv(
    Path("cad_jpy.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
cad_jpy_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1982-01-05</th>
      <td>184.65</td>
      <td>184.65</td>
      <td>184.65</td>
      <td>184.65</td>
    </tr>
    <tr>
      <th>1982-01-06</th>
      <td>185.06</td>
      <td>185.06</td>
      <td>185.06</td>
      <td>185.06</td>
    </tr>
    <tr>
      <th>1982-01-07</th>
      <td>186.88</td>
      <td>186.88</td>
      <td>186.88</td>
      <td>186.88</td>
    </tr>
    <tr>
      <th>1982-01-08</th>
      <td>186.58</td>
      <td>186.58</td>
      <td>186.58</td>
      <td>186.58</td>
    </tr>
    <tr>
      <th>1982-01-11</th>
      <td>187.64</td>
      <td>187.64</td>
      <td>187.64</td>
      <td>187.64</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Trim the dataset to begin on January 1st, 1990
cad_jpy_df = cad_jpy_df.loc["1990-01-01":, :]
cad_jpy_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990-01-02</th>
      <td>126.37</td>
      <td>126.31</td>
      <td>126.37</td>
      <td>126.31</td>
    </tr>
    <tr>
      <th>1990-01-03</th>
      <td>125.30</td>
      <td>125.24</td>
      <td>125.30</td>
      <td>125.24</td>
    </tr>
    <tr>
      <th>1990-01-04</th>
      <td>123.46</td>
      <td>123.41</td>
      <td>123.46</td>
      <td>123.41</td>
    </tr>
    <tr>
      <th>1990-01-05</th>
      <td>124.54</td>
      <td>124.48</td>
      <td>124.54</td>
      <td>124.48</td>
    </tr>
    <tr>
      <th>1990-01-08</th>
      <td>124.27</td>
      <td>124.21</td>
      <td>124.27</td>
      <td>124.21</td>
    </tr>
  </tbody>
</table>
</div>



# Initial Time-Series Plotting

 Start by plotting the "Settle" price. Do you see any patterns, long-term and/or short?


```python
# Plot just the "Price" column from the dataframe:
cad_jpy_df[["Price"]].plot(figsize=(15,10), title="CAD/JPY Exchange Rates")
```




    <AxesSubplot:title={'center':'CAD/JPY Exchange Rates'}, xlabel='Date'>




    
![png](output_6_1.png)
    


**Question:** Do you see any patterns, long-term and/or short? 

### Answer: 
    
Let us review the trends in the 1990s first. There is a significant drop in the price, but by the end of the 1990s there was a marginal resurgance. From 2000 until approximately 2008, we can see a significant upwards trend, with many smaller peaks and valleys. In the short term, therefore, we can see volatility abound. In the long term, we see a sharp decline and a long-standing resurgence of the price. Around the present year, we see almost the same pricing as in the 1990s. 

# Decomposition Using a Hodrick-Prescott Filter

 Using a Hodrick-Prescott Filter, decompose the exchange rate price into trend and noise.


```python
import statsmodels.api as sm

# Apply the Hodrick-Prescott Filter by decomposing the exchange rate price into two separate series:
ts_noise, ts_trend = sm.tsa.filters.hpfilter(cad_jpy_df['Price'])
```


```python
# Plot the trend
ts_trend.plot()
```




    <AxesSubplot:xlabel='Date'>




    
![png](output_12_1.png)
    



```python

```


```python
# Create a dataframe of just the exchange rate price, and add columns for "noise" and "trend" series from above:
combined_df = cad_jpy_df[["Price"]]
combined_df['noise'] = ts_noise
combined_df['trend'] = ts_trend
combined_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>noise</th>
      <th>trend</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990-01-02</th>
      <td>126.37</td>
      <td>0.519095</td>
      <td>125.850905</td>
    </tr>
    <tr>
      <th>1990-01-03</th>
      <td>125.30</td>
      <td>-0.379684</td>
      <td>125.679684</td>
    </tr>
    <tr>
      <th>1990-01-04</th>
      <td>123.46</td>
      <td>-2.048788</td>
      <td>125.508788</td>
    </tr>
    <tr>
      <th>1990-01-05</th>
      <td>124.54</td>
      <td>-0.798304</td>
      <td>125.338304</td>
    </tr>
    <tr>
      <th>1990-01-08</th>
      <td>124.27</td>
      <td>-0.897037</td>
      <td>125.167037</td>
    </tr>
  </tbody>
</table>
</div>




```python
 combined_df[["Price", "trend"]]["2015-01-01":"2020-06-04"].plot(figsize=(15,10), title="Price vs. Trend")
```




    <AxesSubplot:title={'center':'Price vs. Trend'}, xlabel='Date'>




    
![png](output_15_1.png)
    


**Question:** Do you see any patterns, long-term and/or short?

### Answer: 
    
Let us review the trends after applying the HP Filter. First, as may be predicted, the HP filter will smooth out the peaks and valleys. That is predicted because the purpose of the HP filter is to remove the fluctuations that do not add salience or relevance to our analysis. In the short term, we again see annual dips and peaks that correspond slightly with the months of the year- especially in 2018 and 2019. In the long term, we see a significant decline with a slight increase. The price has not yet regained its 2015 value as of the 2020 price indicators. Thus we can conclude an overall decline.  


```python
# Plot the settle noise
ts_noise.plot(figsize=(15,10), title="Noise")
```




    <AxesSubplot:title={'center':'Noise'}, xlabel='Date'>




    
![png](output_18_1.png)
    


---

# Forecasting Returns using an ARMA Model

Using exchange rate *Returns*, estimate an ARMA model

1. ARMA: Create an ARMA model and fit it to the returns data. Note: Set the AR and MA ("p" and "q") parameters to p=2 and q=1: order=(2, 1).
2. Output the ARMA summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
3. Plot the 5-day forecast of the forecasted returns (the results forecast from ARMA model)


```python
# Create a series using "Price" percentage returns, drop any nan"s, and check the results:
# (Make sure to multiply the pct_change() results by 100)
# In this case, you may have to replace inf, -inf values with np.nan"s
returns = (cad_jpy_df[["Price"]].pct_change() * 100)
returns = returns.replace(-np.inf, np.nan).dropna()
returns.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-29</th>
      <td>0.076697</td>
    </tr>
    <tr>
      <th>2020-06-01</th>
      <td>1.251756</td>
    </tr>
    <tr>
      <th>2020-06-02</th>
      <td>1.425508</td>
    </tr>
    <tr>
      <th>2020-06-03</th>
      <td>0.373134</td>
    </tr>
    <tr>
      <th>2020-06-04</th>
      <td>0.012392</td>
    </tr>
  </tbody>
</table>
</div>




```python
returns.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990-01-03</th>
      <td>-0.846720</td>
    </tr>
    <tr>
      <th>1990-01-04</th>
      <td>-1.468476</td>
    </tr>
    <tr>
      <th>1990-01-05</th>
      <td>0.874777</td>
    </tr>
    <tr>
      <th>1990-01-08</th>
      <td>-0.216798</td>
    </tr>
    <tr>
      <th>1990-01-09</th>
      <td>0.667901</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Import the ARMA model
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
```


```python
# use order=(2, 1).

# Estimate and ARMA model using statsmodels (use order=(2, 1))
model = ARIMA(returns.values, order=(2, 1,0))

# Fit the model and assign it to a variable called results
results = model.fit()

print(results.params)
```

    [-0.68605843 -0.33420741  0.92386488]
    


```python
# Output model summary results:
results.summary()
```




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>           <td>y</td>        <th>  No. Observations:  </th>    <td>7928</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARIMA(2, 1, 0)</td>  <th>  Log Likelihood     </th> <td>-10934.368</td>
</tr>
<tr>
  <th>Date:</th>            <td>Fri, 11 Feb 2022</td> <th>  AIC                </th>  <td>21874.736</td>
</tr>
<tr>
  <th>Time:</th>                <td>14:20:12</td>     <th>  BIC                </th>  <td>21895.670</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th>  <td>21881.904</td>
</tr>
<tr>
  <th></th>                      <td> - 7928</td>     <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>  <td>   -0.6861</td> <td>    0.006</td> <td> -110.065</td> <td> 0.000</td> <td>   -0.698</td> <td>   -0.674</td>
</tr>
<tr>
  <th>ar.L2</th>  <td>   -0.3342</td> <td>    0.006</td> <td>  -51.510</td> <td> 0.000</td> <td>   -0.347</td> <td>   -0.321</td>
</tr>
<tr>
  <th>sigma2</th> <td>    0.9239</td> <td>    0.008</td> <td>  121.627</td> <td> 0.000</td> <td>    0.909</td> <td>    0.939</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>59.79</td> <th>  Jarque-Bera (JB):  </th> <td>11829.47</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.00</td>  <th>  Prob(JB):          </th>   <td>0.00</td>  
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.83</td>  <th>  Skew:              </th>   <td>0.27</td>  
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td>  <th>  Kurtosis:          </th>   <td>8.96</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
print(results.forecast(steps=5)[:])
```

    [0.61159318 0.32106877 0.32012787 0.4178688  0.35112727]
    


```python
# Plot the 5 Day Returns Forecast
pd.DataFrame(results.forecast(steps=5)[:]).plot(title="5 Day Returns Forecast")
```




    <AxesSubplot:title={'center':'5 Day Returns Forecast'}>




    
![png](output_28_1.png)
    


### **Question:** Based on the p-value, is the model a good fit?



### Answer: 

Since our p-value >α, we can determine that the model is not a good fit. Specifically, in this case, 2 > 0.5, thus p-value >α. 

---

# Forecasting the Exchange Rate Price using an ARIMA Model

 1. Using the *raw* CAD/JPY exchange rate price, estimate an ARIMA model.
     1. Set P=5, D=1, and Q=1 in the model (e.g., ARIMA(df, order=(5,1,1))
     2. P= # of Auto-Regressive Lags, D= # of Differences (this is usually =1), Q= # of Moving Average Lags
 2. Output the ARIMA summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
 3. Plot a 5 day forecast for the Exchange Rate Price. What does the model forecast predict will happen to the Japanese Yen in the near term?


```python
# Currency pair exchange rates for CAD/JPY
cad_jpy_new_df = pd.read_csv(
    Path("cad_jpy.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
cad_jpy_new_df.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1982-01-05</th>
      <td>184.65</td>
      <td>184.65</td>
      <td>184.65</td>
      <td>184.65</td>
    </tr>
    <tr>
      <th>1982-01-06</th>
      <td>185.06</td>
      <td>185.06</td>
      <td>185.06</td>
      <td>185.06</td>
    </tr>
    <tr>
      <th>1982-01-07</th>
      <td>186.88</td>
      <td>186.88</td>
      <td>186.88</td>
      <td>186.88</td>
    </tr>
    <tr>
      <th>1982-01-08</th>
      <td>186.58</td>
      <td>186.58</td>
      <td>186.58</td>
      <td>186.58</td>
    </tr>
    <tr>
      <th>1982-01-11</th>
      <td>187.64</td>
      <td>187.64</td>
      <td>187.64</td>
      <td>187.64</td>
    </tr>
  </tbody>
</table>
</div>




```python
returns2 = (cad_jpy_new_df[["Price"]].pct_change())
returns2 = returns.replace(-np.inf, np.nan).dropna()
returns2.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-29</th>
      <td>0.076697</td>
    </tr>
    <tr>
      <th>2020-06-01</th>
      <td>1.251756</td>
    </tr>
    <tr>
      <th>2020-06-02</th>
      <td>1.425508</td>
    </tr>
    <tr>
      <th>2020-06-03</th>
      <td>0.373134</td>
    </tr>
    <tr>
      <th>2020-06-04</th>
      <td>0.012392</td>
    </tr>
  </tbody>
</table>
</div>




```python
 y = cad_jpy_df["Price"].to_frame()

 y.dtypes
```




    Price    float64
    dtype: object




```python
from statsmodels.tsa.arima.model import ARIMA

#utilize order=(5,1,1))

# Estimate and ARIMA Model:
# Hint: ARIMA(df, order=(p, d, q))

model = ARIMA( y, order=(5, 1, 1))

# Fit the model
results2 = model.fit()
```

    C:\Users\benja\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:593: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      warnings.warn('A date index has been provided, but it has no'
    C:\Users\benja\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:593: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      warnings.warn('A date index has been provided, but it has no'
    C:\Users\benja\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:593: ValueWarning: A date index has been provided, but it has no associated frequency information and so will be ignored when e.g. forecasting.
      warnings.warn('A date index has been provided, but it has no'
    


```python
print(results2.params)
```

    ar.L1     0.430330
    ar.L2     0.017827
    ar.L3    -0.011751
    ar.L4     0.010993
    ar.L5    -0.019068
    ma.L1    -0.458295
    sigma2    0.531769
    dtype: float64
    


```python
# Output model summary results:
results2.summary()
```




<table class="simpletable">
<caption>SARIMAX Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Price</td>      <th>  No. Observations:  </th>   <td>7929</td>   
</tr>
<tr>
  <th>Model:</th>            <td>ARIMA(5, 1, 1)</td>  <th>  Log Likelihood     </th> <td>-8745.898</td>
</tr>
<tr>
  <th>Date:</th>            <td>Fri, 11 Feb 2022</td> <th>  AIC                </th> <td>17505.796</td>
</tr>
<tr>
  <th>Time:</th>                <td>14:20:16</td>     <th>  BIC                </th> <td>17554.643</td>
</tr>
<tr>
  <th>Sample:</th>                  <td>0</td>        <th>  HQIC               </th> <td>17522.523</td>
</tr>
<tr>
  <th></th>                      <td> - 7929</td>     <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>        <td>opg</td>       <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>       <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>ar.L1</th>  <td>    0.4303</td> <td>    0.331</td> <td>    1.299</td> <td> 0.194</td> <td>   -0.219</td> <td>    1.080</td>
</tr>
<tr>
  <th>ar.L2</th>  <td>    0.0178</td> <td>    0.012</td> <td>    1.459</td> <td> 0.145</td> <td>   -0.006</td> <td>    0.042</td>
</tr>
<tr>
  <th>ar.L3</th>  <td>   -0.0118</td> <td>    0.009</td> <td>   -1.313</td> <td> 0.189</td> <td>   -0.029</td> <td>    0.006</td>
</tr>
<tr>
  <th>ar.L4</th>  <td>    0.0110</td> <td>    0.008</td> <td>    1.299</td> <td> 0.194</td> <td>   -0.006</td> <td>    0.028</td>
</tr>
<tr>
  <th>ar.L5</th>  <td>   -0.0191</td> <td>    0.007</td> <td>   -2.706</td> <td> 0.007</td> <td>   -0.033</td> <td>   -0.005</td>
</tr>
<tr>
  <th>ma.L1</th>  <td>   -0.4583</td> <td>    0.332</td> <td>   -1.381</td> <td> 0.167</td> <td>   -1.109</td> <td>    0.192</td>
</tr>
<tr>
  <th>sigma2</th> <td>    0.5318</td> <td>    0.004</td> <td>  118.418</td> <td> 0.000</td> <td>    0.523</td> <td>    0.541</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Ljung-Box (L1) (Q):</th>     <td>0.00</td> <th>  Jarque-Bera (JB):  </th> <td>9233.72</td>
</tr>
<tr>
  <th>Prob(Q):</th>                <td>0.97</td> <th>  Prob(JB):          </th>  <td>0.00</td>  
</tr>
<tr>
  <th>Heteroskedasticity (H):</th> <td>0.78</td> <th>  Skew:              </th>  <td>-0.58</td> 
</tr>
<tr>
  <th>Prob(H) (two-sided):</th>    <td>0.00</td> <th>  Kurtosis:          </th>  <td>8.16</td>  
</tr>
</table><br/><br/>Warnings:<br/>[1] Covariance matrix calculated using the outer product of gradients (complex-step).




```python
# Plot the 5 Day Price Forecast
pd.DataFrame(results2.forecast(steps=5)[:]).plot(title="5 Day Futures Price Forecast")
```

    C:\Users\benja\anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:390: ValueWarning: No supported index is available. Prediction results will be given with an integer index beginning at `start`.
      warnings.warn('No supported index is available.'
    




    <AxesSubplot:title={'center':'5 Day Futures Price Forecast'}>




    
![png](output_40_2.png)
    


**Question:** What does the model forecast will happen to the Japanese Yen in the near term?



## Answer: 

My model forecases that the Japanese Yen will weaken in the near term. The plot titled "5 Day Futures Price Forecast" clearly shows a steep decline. 

---

# Volatility Forecasting with GARCH

Rather than predicting returns, let's forecast near-term **volatility** of Japanese Yen exchange rate returns. Being able to accurately predict volatility will be extremely useful if we want to trade in derivatives or quantify our maximum loss.
 
Using exchange rate *Returns*, estimate a GARCH model. **Hint:** You can reuse the `returns` variable from the ARMA model section.

1. GARCH: Create an GARCH model and fit it to the returns data. Note: Set the parameters to p=2 and q=1: order=(2, 1).
2. Output the GARCH summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
3. Plot the 5-day forecast of the volatility.


```python
import arch as arch
from arch import arch_model

```


```python
returns = (cad_jpy_df[["Price"]].pct_change() * 100)
returns = returns.replace(-np.inf, np.nan).dropna()
returns.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Price</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990-01-03</th>
      <td>-0.846720</td>
    </tr>
    <tr>
      <th>1990-01-04</th>
      <td>-1.468476</td>
    </tr>
    <tr>
      <th>1990-01-05</th>
      <td>0.874777</td>
    </tr>
    <tr>
      <th>1990-01-08</th>
      <td>-0.216798</td>
    </tr>
    <tr>
      <th>1990-01-09</th>
      <td>0.667901</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Estimate a GARCH model:
model = arch_model(returns, mean="Zero", vol="GARCH", p=2, q=1)

# Fit the model
res = model.fit(disp="off")
```


```python
# Summarize the model results
res.summary()
```




<table class="simpletable">
<caption>Zero Mean - GARCH Model Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>Price</td>       <th>  R-squared:         </th>  <td>   0.000</td> 
</tr>
<tr>
  <th>Mean Model:</th>         <td>Zero Mean</td>     <th>  Adj. R-squared:    </th>  <td>   0.000</td> 
</tr>
<tr>
  <th>Vol Model:</th>            <td>GARCH</td>       <th>  Log-Likelihood:    </th> <td>  -8911.02</td>
</tr>
<tr>
  <th>Distribution:</th>        <td>Normal</td>       <th>  AIC:               </th> <td>   17830.0</td>
</tr>
<tr>
  <th>Method:</th>        <td>Maximum Likelihood</td> <th>  BIC:               </th> <td>   17858.0</td>
</tr>
<tr>
  <th></th>                        <td></td>          <th>  No. Observations:  </th>    <td>7928</td>   
</tr>
<tr>
  <th>Date:</th>           <td>Fri, Feb 11 2022</td>  <th>  Df Residuals:      </th>    <td>7928</td>   
</tr>
<tr>
  <th>Time:</th>               <td>14:20:17</td>      <th>  Df Model:          </th>      <td>0</td>    
</tr>
</table>
<table class="simpletable">
<caption>Volatility Model</caption>
<tr>
      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>       <th>P>|t|</th>      <th>95.0% Conf. Int.</th>   
</tr>
<tr>
  <th>omega</th>    <td>9.0733e-03</td> <td>2.545e-03</td> <td>    3.566</td> <td>3.628e-04</td>  <td>[4.086e-03,1.406e-02]</td>
</tr>
<tr>
  <th>alpha[1]</th> <td>    0.0624</td> <td>1.835e-02</td> <td>    3.402</td> <td>6.682e-04</td>  <td>[2.647e-02,9.841e-02]</td>
</tr>
<tr>
  <th>alpha[2]</th>   <td>0.0000</td>   <td>2.010e-02</td>   <td>0.000</td>   <td>    1.000</td> <td>[-3.940e-02,3.940e-02]</td>
</tr>
<tr>
  <th>beta[1]</th>  <td>    0.9243</td> <td>1.229e-02</td> <td>   75.205</td>   <td>0.000</td>      <td>[  0.900,  0.948]</td>  
</tr>
</table><br/><br/>Covariance estimator: robust



**Note:** Our p-values for GARCH and volatility forecasts tend to be much lower than our ARMA/ARIMA return and price forecasts. In particular, here we have all p-values of less than 0.05, except for alpha(2), indicating overall a much better model performance. In practice, in financial markets, it's easier to forecast volatility than it is to forecast returns or prices. (After all, if we could very easily predict returns, we'd all be rich!)


```python
# Find the last day of the dataset
last_day = returns.index.max().strftime('%Y-%m-%d')
last_day
```




    '2020-06-04'




```python
# Create a 5 day forecast of volatility
forecast_horizon = 5

# Start the forecast using the last_day calculated above
forecasts = res.forecast(start='2020-06-04', horizon=forecast_horizon, reindex=False)
forecasts
```




    <arch.univariate.base.ARCHModelForecast at 0x1f995aa9bb0>




```python
# Annualize the forecast
intermediate = np.sqrt(forecasts.variance.dropna() * 252)
intermediate.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>h.1</th>
      <th>h.2</th>
      <th>h.3</th>
      <th>h.4</th>
      <th>h.5</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-06-04</th>
      <td>12.566029</td>
      <td>12.573718</td>
      <td>12.581301</td>
      <td>12.588778</td>
      <td>12.596153</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Transpose the forecast so that it is easier to plot
final = intermediate.dropna().T
final.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Date</th>
      <th>2020-06-04</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>h.1</th>
      <td>12.566029</td>
    </tr>
    <tr>
      <th>h.2</th>
      <td>12.573718</td>
    </tr>
    <tr>
      <th>h.3</th>
      <td>12.581301</td>
    </tr>
    <tr>
      <th>h.4</th>
      <td>12.588778</td>
    </tr>
    <tr>
      <th>h.5</th>
      <td>12.596153</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the final forecast
final.plot(title="5 Day Forecast of Volatility")
```




    <AxesSubplot:title={'center':'5 Day Forecast of Volatility'}>




    
![png](output_54_1.png)
    


**Question:** What does the model forecast will happen to volatility in the near term?

## Answer: 

The model indicates that the volatility will increase in the near term. The five day forecast clearly shows the increase from h1 to h5. 

---

# Conclusions

Based on your time series analysis, would you buy the yen now?

## Answer: 

The volatility of the Yen indicated by the GARCH model indicates that purchasing the yen now would not be a wise investment option. 

Is the risk of the yen expected to increase or decrease?

## Answer: 

The volatility of the Yen predicts that the risk associated with the Yen is on the rise. However, it should be noted that this only a short term conclusion. In the future the risk may vary upwards or downwards depending on a variety of factors. 

Based on the model evaluation, would you feel confident in using these models for trading?
 
## Answer: 

The fit of a model should be determined by p-value >α. These models have shown that they are not a good fit, and would therefore require further modifications and calibrations to be fit for trading purposes. They could be tweaked, and over time may be suitable. But at the present time these p-values indicate that the models are not a good fit- and therefore not suitable for trading purposes.   


```python

```
