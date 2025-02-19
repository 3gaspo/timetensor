import matplotlib.pyplot as plt
import numpy as np

show_axis = True

def h(offset, amp):
    return offset + np.random.normal(0,amp)

S = [100]
for k in range(1,100):
    S.append(h(S[k-1],1))

plt.plot(S)
if show_axis:
    plt.xlabel("Time")
else:
    plt.axis('off')
    plt.title(None)
plt.savefig("auto.pdf")


print("Starting arima")

import statsmodels.api as sm

#1 step
lookback = S[0:70]
horizon = S[70:]
mod = sm.tsa.arima.ARIMA(lookback, order=(1, 0, 0)).fit()

forecast_steps = 30
forecast = mod.forecast(steps = forecast_steps)

plt.clf()
plt.figure(figsize=(10, 5))
plt.plot(lookback, label="Historical Data")
plt.plot(range(len(lookback), len(lookback) + forecast_steps), horizon, label="True horizon", linestyle="--")
plt.plot(range(len(lookback), len(lookback) + forecast_steps), forecast, label="Forecast", color="red")
plt.legend()
plt.title("ARIMA Forecast")
plt.savefig("arima.pdf")




#2 step

def h2(offset1, offset2,  amp):
    return (offset1 + offset2)/2 + np.random.normal(0,amp)

S = [100, 80]

for k in range(2,100):
    S.append(h2(S[k-1], S[k-2], 10))

lookback = S[0:70]
horizon = S[70:]
mod = sm.tsa.arima.ARIMA(lookback, order=(1, 0, 0)).fit()

forecast_steps = 30
forecast = mod.forecast(steps = forecast_steps)

plt.clf()
plt.figure(figsize=(10, 5))
plt.plot(lookback, label="Historical Data")
plt.plot(range(len(lookback), len(lookback) + forecast_steps), horizon, label="True horizon", linestyle="--")
plt.plot(range(len(lookback), len(lookback) + forecast_steps), forecast, label="Forecast", color="red")
plt.legend()
plt.title("ARIMA Forecast")
plt.savefig("arima2.pdf")