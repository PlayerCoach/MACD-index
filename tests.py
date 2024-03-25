from import_data import *
import numpy as np


def calculate_ema(data, window):

    ema = []
    alpha = 2 / (window + 1)

    # Initialize values for the first window as the simple moving average (SMA)
    # because EMA requires at least window number of elements to be calculated properlu
    ema = np.zeros(len(data))
    sma = np.mean(data[:window])
    for i in range(window):
        ema[i] = sma

    # Calculate the EMA for the rest of the data
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]

    return ema


def create_macd(data, short_window, long_window):
    data['ShortEMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['LongEMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['ShortEMA'] - data['LongEMA']

    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data
def own_macd(data,short_window,long_window):
    data['ShortEMA'] = calculate_ema(data['Close'].tolist(), short_window)
    data['LongEMA'] = calculate_ema(data['Close'].tolist(), long_window)

    # Calculate the MACD
    data['MACD'] = data['ShortEMA'] - data['LongEMA']

    # Calculate the Signal line (9-period EMA of the MACD)
    data['Signal'] = calculate_ema(data['MACD'].tolist(), 9)

    return data

weekly = get_weekly()
data = weekly.copy(deep=True)
short_window = 12
long_window = 26
data = create_macd(data, short_window, long_window)
weekly = own_macd(weekly, short_window, long_window)
if(data['MACD'].equals(weekly['MACD'])):
    print("MACD is correct")
print(data['MACD'])
print(weekly['MACD'])
