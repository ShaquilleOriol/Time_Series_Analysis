import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import time, json
from datetime import date
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

seaplaneDataframe = pd.read_csv('SeaPlaneTravel.csv')
seaplaneDataframe.head(6)

seaplaneDataframe.info()
seaplaneDataframe['Month'] = pd.to_datetime(seaplaneDataframe['Month'])

indexDataframe = seaplaneDataframe.set_index('Month')
ts = indexDataframe['#Passengers']
ts.head(10)

plot.figure(figsize=(15,5))
plot.plot(ts)
plot.show(block=False)

def test_stationarity(timeseries):
    rollingStd = timeseries.rolling(window=12,center=False).mean()
    rollingMean = timeseries.rolling(window=12,center=False).mean()
    original = plot.plot(timeseries, color='blue', label='Original')
    mean = plot.plot(rollingMean, color='red', label='Rolling Mean')
    std = plot.plot(rollingStd, color='black', label='Rolling Std')
    
    plot.legend(loc='best')
    plot.title('Rolling mean and standard deviation')
    plot.show()
    

    print('Results of dickey fuller test')
    test = adfuller(timeseries, autolag = 'AIC')
    output = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in test[4].items():
        output['Critical Value (%s)'%key] = value
    print(output)
    
test_stationarity(ts) 


plot.figure(figsize=(15,5))
tsLog = np.log(ts)
tsLogDiff = tsLog - tsLog.shift()
plot.plot(tsLogDiff)   


lag_acf = acf(tsLogDiff, nlags = 10)
lag_pacf = pacf(tsLogDiff, nlags = 10, method='ols')

print(lag_acf)
print(lag_pacf)

plot.figure(figsize=(15,5))
plot.plot(lag_acf)
plot.axhline(y=0,linestyle='--', color='gray')
plot.axhline(y=-7.96/np.sqrt(len(tsLogDiff)),linestyle='--', color='gray')
plot.axhline(y=7.96/np.sqrt(len(tsLogDiff)),linestyle='--', color='gray')
plot.title('Autocorrelation Function')
plot.show()


plot.figure(figsize=(15,5))
plot.plot(lag_pacf)
plot.axhline(y=0,linestyle='--', color='gray')
plot.axhline(y=-7.96/np.sqrt(len(tsLogDiff)),linestyle='--', color='gray')
plot.axhline(y=7.96/np.sqrt(len(tsLogDiff)),linestyle='--', color='gray')
plot.title('Partial Autocorrelation Function')
plot.show()

p = 2
q = 1
d = 1

testsize=15
train = tsLog[0:-15]
test = tsLog[-15:]
print(test)

history = [x for x in train]
prediction = []
print(history)


history = [x for x in train]
prediction = []
for t in range(len(test)):
    model = ARIMA(history, order= (p,d,q))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    prediction.append(float(yhat))
    actual_log = test[t]
    history.append(actual_log)
    print("Predicted Value", np.exp(yhat), "Actual Value", np.exp(actual_log))

mean_squared_error(test,prediction)

predictions_series = pd.Series(prediction, index = test.index)


fig, ax = plot.subplots()
ax.set(title='Forcasted data', xlabel='Date', ylabel='Passengers')
ax.plot(ts[-60:], 'o', label='observed')
ax.plot(np.exp(predictions_series), 'g', label='rolling one-step out-of-sample forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')
