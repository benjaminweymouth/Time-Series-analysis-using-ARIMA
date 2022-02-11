# Regression Analysis: Currency Analysis with Sklearn Linear Regression
In this notebook, you will build a SKLearn linear regression model to predict Yen futures ("settle") returns with *lagged* CAD/JPY exchange rate returns. 


```python
import numpy as np
import pandas as pd
from pathlib import Path
%matplotlib inline
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



# Data Preparation

### Returns


```python
# Create a series using "Price" percentage returns, drop any nan"s, and check the results:
# (Make sure to multiply the pct_change() results by 100)
# In this case, you may have to replace inf, -inf values with np.nan"s
cad_jpy_df["Return"] = cad_jpy_df[["Price"]].pct_change() * 100
cad_jpy_df = cad_jpy_df.replace(-np.inf, np.nan).dropna()
cad_jpy_df.tail()
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
      <th>Return</th>
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
      <th>2020-05-29</th>
      <td>78.29</td>
      <td>78.21</td>
      <td>78.41</td>
      <td>77.75</td>
      <td>0.076697</td>
    </tr>
    <tr>
      <th>2020-06-01</th>
      <td>79.27</td>
      <td>78.21</td>
      <td>79.36</td>
      <td>78.04</td>
      <td>1.251756</td>
    </tr>
    <tr>
      <th>2020-06-02</th>
      <td>80.40</td>
      <td>79.26</td>
      <td>80.56</td>
      <td>79.15</td>
      <td>1.425508</td>
    </tr>
    <tr>
      <th>2020-06-03</th>
      <td>80.70</td>
      <td>80.40</td>
      <td>80.82</td>
      <td>79.96</td>
      <td>0.373134</td>
    </tr>
    <tr>
      <th>2020-06-04</th>
      <td>80.71</td>
      <td>80.80</td>
      <td>80.89</td>
      <td>80.51</td>
      <td>0.012392</td>
    </tr>
  </tbody>
</table>
</div>



### Lagged Returns 


```python
# Create a lagged return using the shift function
cad_jpy_df['Lagged_Return'] = cad_jpy_df.Return.shift()
cad_jpy_df =cad_jpy_df.dropna()
cad_jpy_df.tail()
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
      <th>Return</th>
      <th>Lagged_Return</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-29</th>
      <td>78.29</td>
      <td>78.21</td>
      <td>78.41</td>
      <td>77.75</td>
      <td>0.076697</td>
      <td>-0.114913</td>
    </tr>
    <tr>
      <th>2020-06-01</th>
      <td>79.27</td>
      <td>78.21</td>
      <td>79.36</td>
      <td>78.04</td>
      <td>1.251756</td>
      <td>0.076697</td>
    </tr>
    <tr>
      <th>2020-06-02</th>
      <td>80.40</td>
      <td>79.26</td>
      <td>80.56</td>
      <td>79.15</td>
      <td>1.425508</td>
      <td>1.251756</td>
    </tr>
    <tr>
      <th>2020-06-03</th>
      <td>80.70</td>
      <td>80.40</td>
      <td>80.82</td>
      <td>79.96</td>
      <td>0.373134</td>
      <td>1.425508</td>
    </tr>
    <tr>
      <th>2020-06-04</th>
      <td>80.71</td>
      <td>80.80</td>
      <td>80.89</td>
      <td>80.51</td>
      <td>0.012392</td>
      <td>0.373134</td>
    </tr>
  </tbody>
</table>
</div>



### Train Test Split


```python
# Create a train/test split for the data using 2018-2019 for testing and the rest for training
train = cad_jpy_df[:'2017']
test = cad_jpy_df['2018':]
```


```python
# Create four dataframes:
# X_train (training set using just the independent variables), X_test (test set of of just the independent variables)
# Y_train (training set using just the "y" variable, i.e., "Futures Return"), Y_test (test set of just the "y" variable):
X_train = train["Lagged_Return"].to_frame()
X_test = test["Lagged_Return"].to_frame()
y_train = train["Return"]
y_test = test["Return"]
```


```python
# Preview the X_train data
X_train.head()
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
      <th>Lagged_Return</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1990-01-04</th>
      <td>-0.846720</td>
    </tr>
    <tr>
      <th>1990-01-05</th>
      <td>-1.468476</td>
    </tr>
    <tr>
      <th>1990-01-08</th>
      <td>0.874777</td>
    </tr>
    <tr>
      <th>1990-01-09</th>
      <td>-0.216798</td>
    </tr>
    <tr>
      <th>1990-01-10</th>
      <td>0.667901</td>
    </tr>
  </tbody>
</table>
</div>



# Linear Regression Model


```python
# Create a Linear Regression model and fit it to the training data
from sklearn.linear_model import LinearRegression

# Fit a SKLearn linear regression using  just the training set (X_train, Y_train):
model = LinearRegression()
model.fit(X_train, y_train)
```




    LinearRegression()



# Make predictions using the Testing Data

**Note:** We want to evaluate the model using data that it has never seen before, in this case: `X_test`.


```python
# Make a prediction of "y" values using just the test dataset
predictions = model.predict(X_test)
```


```python
# Assemble actual y data (Y_test) with predicted y data (from just above) into two columns in a dataframe:
Results = y_test.to_frame()
Results["Predicted Return"] = predictions
Results.head(5)
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
      <th>Return</th>
      <th>Predicted Return</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>0.245591</td>
      <td>0.005434</td>
    </tr>
    <tr>
      <th>2018-01-02</th>
      <td>-0.055679</td>
      <td>-0.007317</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>0.011142</td>
      <td>0.000340</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>0.601604</td>
      <td>-0.001358</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>0.919158</td>
      <td>-0.016366</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot the first 20 predictions vs the true values

# The trends lines should be similar
Results[:20].plot(subplots=True, figsize=(15,15))

```




    array([<AxesSubplot:xlabel='Date'>, <AxesSubplot:xlabel='Date'>],
          dtype=object)




    
![png](output_18_1.png)
    


# Out-of-Sample Performance

Evaluate the model using "out-of-sample" data (`X_test` and `y_test`)


```python
from sklearn.metrics import mean_squared_error
# Calculate the mean_squared_error (MSE) on actual versus predicted test "y" 
# (Hint: use the dataframe from above)
mse = mean_squared_error(
    Results["Return"],
    Results["Predicted Return"]
)

# Using that mean-squared-error, calculate the root-mean-squared error (RMSE):
rmse = np.sqrt(mse)
print(f"Out-of-Sample Root Mean Squared Error (RMSE): {rmse}")
```

    Out-of-Sample Root Mean Squared Error (RMSE): 0.6445805658569028
    

# In-Sample Performance

Evaluate the model using in-sample data (X_train and y_train)


```python
# Construct a dataframe using just the "y" training data:
in_sample_results = y_train.to_frame()

# Add a column of "in-sample" predictions to that dataframe:  
in_sample_results["In-sample Predictions"] = model.predict(X_train)

# Calculate in-sample mean_squared_error (for comparison to out-of-sample)
in_sample_mse = mean_squared_error(
    in_sample_results["Return"],
    in_sample_results["In-sample Predictions"]
)

# Calculate in-sample root mean_squared_error (for comparison to out-of-sample)
in_sample_rmse = np.sqrt(in_sample_mse)
print(f"In-sample Root Mean Squared Error (RMSE): {in_sample_rmse}")
```

    In-sample Root Mean Squared Error (RMSE): 0.841994632894117
    

# Conclusions

**Question:** Does this model perform better or worse on out-of-sample data as compared to in-sample data?

**Answer:** With a lower RMSE, we have a better model. So with that in mind, we can conclude that the out-of-sample performance is better than our in-sample performance. The In-Sample performance RMSE is close to 1, whereas the RMSE for the out-of-sample is closer to 0.5. Thus, this model performs better on the out-of-sample performance. 


```python

```
