import matplotlib.pyplot as plt
import numpy as np

show_axis = True #to show axis in plots
T = 900
t = np.arange(0, T)
zoom_idx = 24 * 7
year_freq = 900
day_freq = 24
N = 10

#temperature

def cyclic(x, period, offset, amplitude, phase):
    """returns a sinusoidal wave"""
    return offset + amplitude * np.cos((x-phase) * 2 * np.pi / period)

temperatures = cyclic(t, day_freq, 0, 5 , day_freq//2) + cyclic(t, year_freq, 15, 10, year_freq//2) + np.random.normal(0, 0.5, size=T)

plt.clf()
plt.plot(t, temperatures)
if show_axis:
    plt.xlabel("Time")
    plt.ylabel("Temperature")
else:
    plt.axis('off')
    plt.title(None)
plt.savefig("temperature.pdf")

plt.clf()
plt.plot(t[0:zoom_idx], temperatures[0:zoom_idx])
if show_axis:
    plt.xlabel("Time")
    plt.ylabel("Temperature")
else:
    plt.axis('off')
    plt.title(None)
plt.savefig("zoom_temperature.pdf")

# X = f(temp)

def f(T, amp=1):
    """heating based on temperature"""
    return cyclic(T, 40, 100, 100*amp, 0)

temp_range = np.arange(40)
plt.clf()
plt.plot(temp_range, f(temp_range))
if show_axis:
    plt.xlabel("T")
    plt.ylabel("X")
else:
    plt.axis('off')
    plt.title(None)
plt.savefig("partial_T.pdf")


#week

def g(t, amp=1):
    """weekly consumption"""
    weekend_mask = (t > 24*5)
    week_cycle =  cyclic(t, day_freq//2, 100 ,100*amp, 8)
    weekend_cycle = cyclic(t, day_freq, 150, 150*amp, 14)
    return week_cycle * (1-weekend_mask) + weekend_cycle * weekend_mask

plt.clf()
week_range = np.arange(1,24*7)
plt.plot(week_range, g(week_range))
for k in range(1,8):
    plt.axvline(24*k, color="red", linestyle="--")
if show_axis:
    plt.xlabel("t")
    plt.ylabel("X")
else:
    plt.axis('off')
    plt.title(None)
plt.savefig("partial_t.pdf")


#clients

def get_client():
    """returns random amplitude modifiers for each clients"""
    amp_t = np.random.normal(0,0.1) + 1
    amp_T = np.random.normal(0,0.1) + 1
    return amp_T, amp_t

clients = [get_client() for k in range(N)]


plt.clf()
plt.plot(temp_range, f(temp_range, clients[0][0]))
if show_axis:
    plt.xlabel("T")
    plt.ylabel("X")
else:
    plt.axis('off')
    plt.title(None)
plt.savefig("client_partial_T.pdf")

plt.clf()
plt.plot(week_range, g(week_range, clients[0][1]))
if show_axis:
    plt.xlabel("t")
    plt.ylabel("X")
else:
    plt.axis('off')
    plt.title(None)
plt.savefig("client_partial_t.pdf")


#simulation

loads = []
for k in range(N):
    X = g(t, clients[k][1]) + f(temperatures, clients[k][0]) + np.random.normal(0, 10, size=T)
    loads.append(X)
loads = np.array(loads)
aggregated = np.sum(loads, axis=0)


#individual

plt.clf()
plt.plot(t, loads[0])
if show_axis:
    plt.xlabel("Time")
    plt.ylabel("Consumption")
else:
    plt.axis('off')
    plt.title(None)
plt.savefig("client_load.pdf")

plt.clf()
plt.plot(t[0:zoom_idx], loads[0, 0:zoom_idx])
if show_axis:
    plt.xlabel("Time")
    plt.ylabel("Consumption")
else:
    plt.axis('off')
    plt.title(None)
plt.savefig("zoomed_client_load.pdf")

#aggregated 

plt.clf()
plt.plot(t, aggregated)
if show_axis:
    plt.xlabel("Time") 
    plt.ylabel("Consumption")
else:
    plt.axis('off')
    plt.title(None)
plt.savefig("load.pdf")

plt.clf()
plt.plot(t[0:zoom_idx], aggregated[0:zoom_idx])
if show_axis:
    plt.xlabel("Time")
    plt.ylabel("Consumption")
else:
    plt.axis('off')
    plt.title(None)
plt.savefig("zoomed_load.pdf")

plt.clf()
plt.scatter(np.mean(loads, axis=1), np.max(loads, axis=1))
if show_axis:
    plt.title("Distribution of clients")
    plt.xlabel("Mean")
    plt.ylabel("Max")
else:
    plt.axis('off')
    plt.title(None)
plt.savefig("distribution.pdf")


print("starting gam")
from pygam import LinearGAM, s, f

pos_day = np.array([k % day_freq for k in t])
pos_week = np.array([(k % (day_freq*7)) // day_freq for k in t])
features = np.array([temperatures, pos_day, pos_week]).T
#gam = LinearGAM(s(0) + f(1)).fit(features, aggregated)
gam = LinearGAM(s(0) + s(1) + f(2)).fit(features, aggregated)

pred = gam.predict(features)
plt.clf()
plt.plot(t, pred)
plt.savefig("gam.pdf")

plt.clf()
plt.plot(t[0:zoom_idx], pred[0:zoom_idx])
plt.savefig("zoomed_gam.pdf")

for i, term in enumerate(gam.terms):
    if term.isintercept:
        continue

    XX = gam.generate_X_grid(term=i)
    pdep, confi = gam.partial_dependence(term=i, X=XX, width=0.95)

    plt.figure()
    plt.plot(XX[:, term.feature], pdep)
    plt.plot(XX[:, term.feature], confi, c='r', ls='--')
    plt.title(repr(term))
    plt.savefig(repr(term) + ".pdf")