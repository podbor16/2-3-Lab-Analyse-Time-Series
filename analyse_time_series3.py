import numpy as np
import pandas as pd
from numpy.linalg import svd
import matplotlib.pyplot as plt


def components(a, b):
    u = U[:, range(a, b)]
    s = np.diag(S)[a:b, a:b]
    vt = VT[range(a, b), :]
    res_a = u @ s @ vt
    return res_a.reshape(n * m)


dataset = pd.read_csv('WTI Price FOB.csv')
dataset.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
dataset.set_index('Date', inplace=True)
dataset.index = pd.to_datetime(dataset.index)
dataset.rename(columns={'WTI Spot Price, Monthly (Dollars per Million Btu)': 'Sale'},  inplace=True)
dataset.index = pd.to_datetime(dataset.index)
data = dataset
data = data.dropna()

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(data['Sale'])

data['Trend'] = data['Sale'].rolling(window=12, min_periods=1, center=True).mean()
data['signal'] = data['Sale'] - data['Trend']
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(data['signal'])
plt.plot(data['Sale'])
plt.plot(data['Trend'])

nm = len(data['Sale'])
# nm

n = 18
m = int(nm / n)
# n, m, n * m

A_x = data['Sale'][:n * m].values.reshape((n, m))
A_s = data['signal'][:n * m].values.reshape((n, m))

U, S, VT = svd(A_x, full_matrices=False)

a, b = (0, 1)
Trend = components(a, b)
plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(data['Trend'])
plt.plot

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(Trend)
plt.plot



signal = data['Sale'][:n * m].values - Trend
A_x_s = signal.reshape((n, m))
U, S, VT = svd(A_x_s, full_matrices=False)



f, axarr = plt.subplots(2, 2, figsize=(16, 16))
for i, (a, b) in enumerate([(1, 2), (2, 4), (1, 6), (5, 6)]):
    axarr[i//2, i%2].plot(components(a, b))
    axarr[i//2, i%2].set_title(f'компоненты {a}-{b}')
plt.show()


U, S, VT = svd(A_s, full_matrices=False)
plt.plot(S)

f, axarr = plt.subplots(2, 2, figsize=(16, 16))
for i, (a, b) in enumerate([(1, 2), (2, 4), (1, 6), (5, 6)]):
    axarr[i//2, i%2].plot(components(a, b))
    axarr[i//2, i%2].set_title(f'компоненты {a}-{b}')
plt.show()