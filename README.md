## Name : Kishore S
## Reg No : 212222240050
# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```python
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset
data = pd.read_csv('/content/NFLX.csv')
data['Date'] = pd.to_datetime(data['Date'])  # Automatically infer date format
data.set_index('Date', inplace=True)

# Filter data from 2010 onward
data = data[data.index >= '2010-01-01']

# Convert 'Close' column to numeric and remove missing values
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data['Close'].fillna(method='ffill', inplace=True)

# Plot the Close price to inspect for trends
plt.figure(figsize=(10, 5))
plt.plot(data['Close'], label='Stock Close Price')
plt.title('Time Series of Stock Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Check stationarity with ADF test
result = adfuller(data['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Apply differencing if data is not stationary
if result[1] > 0.05:
    data['Close_diff'] = data['Close'].diff().dropna()
    result_diff = adfuller(data['Close_diff'].dropna())
    print('Differenced ADF Statistic:', result_diff[0])
    print('Differenced p-value:', result_diff[1])

    # Plot ACF and PACF for differenced data
    plot_acf(data['Close_diff'].dropna())
    plt.title('ACF of Differenced Close Price')
    plt.show()

    plot_pacf(data['Close_diff'].dropna())
    plt.title('PACF of Differenced Close Price')
    plt.show()
else:
    print("Data is already stationary.")

# Choose initial ARIMA parameters based on ACF and PACF (adjust as needed)
p = 1  # Based on PACF plot
d = 1  # Data was differenced once
q = 1  # Based on ACF plot

# Fit the ARIMA model
model = sm.tsa.ARIMA(data['Close'], order=(p, d, q))
fitted_model = model.fit()
print(fitted_model.summary())

# Forecast the next 30 days
forecast = fitted_model.forecast(steps=30)
forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')

# Plot actual vs forecasted values
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Actual')
plt.plot(forecast_index, forecast, label='Forecast', color='orange')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('ARIMA Forecast of Stock Price')
plt.legend()
plt.show()

# Evaluate the model with MAE and RMSE
predictions = fitted_model.predict(start=0, end=len(data['Close']) - 1)
mae = mean_absolute_error(data['Close'], predictions)
rmse = np.sqrt(mean_squared_error(data['Close'], predictions))
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)

```

### OUTPUT:

![image](https://github.com/user-attachments/assets/a88ac5a8-b921-4909-8874-584298eed248)

![image](https://github.com/user-attachments/assets/615f95d4-5450-448d-97a0-3b8d5a24c3b6)

![image](https://github.com/user-attachments/assets/2cddc7c5-a2e1-4c19-b0da-fbd4adc5ae85)

![image](https://github.com/user-attachments/assets/0250eaf3-d7c0-445f-a126-aca1c9b34083)

![image](https://github.com/user-attachments/assets/9cf70ba8-878c-41f0-b418-20286837c330)


### RESULT:
Thus the Time series analysis on Google stock prediction using the ARIMA model completed successfully.
