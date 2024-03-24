import matplotlib.pyplot as plt
import numpy as np

from import_data import *


def create_macd(data, short_window, long_window):
    data['ShortEMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['LongEMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['ShortEMA'] - data['LongEMA']

    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    return data


def select_macd_points(data):
    buy_signals = []
    sell_signals = []
    flag = -1

    for i in range(0, len(data)):
        if data['MACD'][i] > data['Signal'][i]:
            sell_signals.append(np.nan)
            if flag != 1:
                buy_signals.append(data['Close'][i])
                flag = 1
            else:
                buy_signals.append(np.nan)
        elif data['MACD'][i] < data['Signal'][i]:
            buy_signals.append(np.nan)
            if flag != 0:
                sell_signals.append(data['Close'][i])
                flag = 0
            else:
                sell_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    return buy_signals, sell_signals


def select_intersection_points(data):
    buy_signals = []
    sell_signals = []
    flag = -1

    for i in range(0, len(data)):
        if data['MACD'][i] > data['Signal'][i]:
            sell_signals.append(np.nan)
            if flag != 1:
                buy_signals.append(data['Signal'][i])
                flag = 1
            else:
                buy_signals.append(np.nan)
        elif data['MACD'][i] < data['Signal'][i]:
            buy_signals.append(np.nan)
            if flag != 0:
                sell_signals.append(data['Signal'][i])
                flag = 0
            else:
                sell_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    return buy_signals, sell_signals


if __name__ == '__main__':
    weekly = get_weekly()
    data = weekly.copy(deep=True)
    short_window = 12
    long_window = 26
    data = create_macd(data, short_window, long_window)

    # Plot the data
    weekly['Date'] = pd.to_datetime(weekly['Date'])
    plt.figure(figsize=(12, 6))
    plt.plot(weekly['Date'], weekly['Close'])
    plt.scatter(data['Date'], select_macd_points(data)[0], color='green', marker='^', alpha=1)
    plt.scatter(data['Date'], select_macd_points(data)[1], color='red', marker='v', alpha=1)
    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.title('Netflix Stock Price')
    plt.show()

    # Create MACD indicator

    # Plot MACD
    data['Date'] = pd.to_datetime(data['Date'])
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['MACD'], label='MACD', color='red', linewidth=0.5)
    plt.plot(data['Date'], data['Signal'], label='Signal Line', color='blue', linewidth=0.5)
    plt.scatter(data['Date'], select_intersection_points(data)[0], color='green', marker='^', alpha=1)
    plt.scatter(data['Date'], select_intersection_points(data)[1], color='red', marker='v', alpha=1)


    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.title('MACD Indicator')
    plt.legend()
    plt.show()




