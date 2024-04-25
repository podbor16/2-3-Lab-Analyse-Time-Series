import warnings
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pywt


def adfuller_test(data):
    result = adfuller(data, autolag='AIC')
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')
    if result[1] <= 0.05:
        print('Временной ряд стационарен (отвергаем нулевую гипотезу)')
    else:
        print('Временной ряд нестационарен (не отвергаем нулевую гипотезу)')


warnings.filterwarnings("ignore")

# Загрузка данных
dataset = pd.read_csv('WTI Price FOB.csv')
dataset.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
dataset.set_index('Date', inplace=True)
dataset.index = pd.to_datetime(dataset.index)
dataset.rename(columns={'WTI Spot Price, Monthly (Dollars per Million Btu)': 'Sale'}, inplace=True)
dataset.index = pd.to_datetime(dataset.index)

# Шум
coeffs = pywt.wavedec(dataset['Sale'], "haar")
threshold = 5
coeffs_filtered = [coeff if idx == 0 else pywt.threshold(coeff, threshold, mode='soft') for idx, coeff in
                   enumerate(coeffs)]
reconstructed_data_noise = pywt.waverec(coeffs_filtered, "haar")

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(dataset)
plt.title("Оригинальный ряд")
plt.subplot(2, 1, 2)
plt.plot(reconstructed_data_noise)
plt.title("Восстановленный ряд без шума")
plt.show()

adfuller_test(reconstructed_data_noise)

plot_acf(reconstructed_data_noise, alpha=0.05, lags=50)
plt.show()


plot_pacf(reconstructed_data_noise)
plt.show()


# Тренд
coeff1s = pywt.wavedec(dataset['Sale'], "haar")
threshold = 5
coeffs_filtered = [coeff if idx == 0 else pywt.threshold(coeff, threshold, mode='soft') for idx, coeff in
                   enumerate(coeff1s)]
reconstructed_data_trend = pywt.waverec(coeffs_filtered[:6], "haar")

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(dataset)
plt.title("Оригинальный ряд")
plt.subplot(2, 1, 2)
plt.plot(reconstructed_data_trend)
plt.title("Восстановленный тренд")
plt.show()

adfuller_test(reconstructed_data_trend)

plot_pacf(reconstructed_data_trend)
plt.show()

plot_acf(reconstructed_data_trend, alpha=0.05, lags=50)
plt.show()
