from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from import_data import *


def calculate_ema(closure: List[float], window: int) -> np.ndarray:
    alpha = 2 / (window + 1)
    # Initialize values for the first window as the simple moving average (SMA)
    # because EMA requires at least window number of elements to be calculated properly
    # Inspiration:
    # https://dayanand-shah.medium.com/exponential-moving-average-and-implementation-with-python-1890d1b880e6
    ema = np.zeros(len(closure))
    sma = np.mean(closure[:window])

    for i in range(window):
        ema[i] = sma

    # Calculate the EMA for the rest of the data
    for i in range(1, len(closure)):
        ema[i] = alpha * closure[i] + (1 - alpha) * ema[i - 1]

    return ema


def create_macd_and_signal_in_data(_data: pd.DataFrame, _short_window: int, _long_window: int) -> pd.DataFrame:
    _data['ShortEMA'] = calculate_ema(_data['Close'].tolist(), _short_window)
    _data['LongEMA'] = calculate_ema(_data['Close'].tolist(), _long_window)
    _data['MACD'] = _data['ShortEMA'] - _data['LongEMA']
    _data['Signal'] = calculate_ema(_data['MACD'].tolist(), 9)
    return _data


def choose_buy_sell_points_for_initial_plot(_data: pd.DataFrame, _long_window: int) -> (np.ndarray, np.ndarray):
    _buy_signals = []
    _sell_signals = []
    switch_transaction = -1  # -1 - no transaction, 0 - sell, 1 - buy

    for i in range(0, len(_data)):
        if i > long_window:
            if _data['MACD'][i] > _data['Signal'][i]:
                _sell_signals.append(np.nan)
                if switch_transaction != 1:
                    _buy_signals.append(_data['Close'][i])
                    switch_transaction = 1
                else:
                    _buy_signals.append(np.nan)
            elif _data['MACD'][i] < _data['Signal'][i]:
                _buy_signals.append(np.nan)
                if switch_transaction != 0:
                    _sell_signals.append(_data['Close'][i])
                    switch_transaction = 0
                else:
                    _sell_signals.append(np.nan)
            else:
                _buy_signals.append(np.nan)
                _sell_signals.append(np.nan)
        else:
            _buy_signals.append(np.nan)
            _sell_signals.append(np.nan)

    return _buy_signals, _sell_signals


def choose_macd_and_signal_intersection_points(_data: pd.DataFrame, _long_window: int) -> (np.ndarray, np.ndarray):
    _buy_signals = []
    _sell_signals = []
    flag = -1  # -1 - no transaction, 0 - sell, 1 - buy

    for i in range(0, len(_data)):
        if i > long_window:
            if _data['MACD'][i] > _data['Signal'][i]:
                _sell_signals.append(np.nan)
                if flag != 1:
                    _buy_signals.append(_data['MACD'][i])
                    flag = 1
                else:
                    _buy_signals.append(np.nan)
            elif _data['MACD'][i] < _data['Signal'][i]:
                _buy_signals.append(np.nan)
                if flag != 0:
                    _sell_signals.append(_data['MACD'][i])
                    flag = 0
                else:
                    _sell_signals.append(np.nan)
            else:
                _buy_signals.append(np.nan)
                _sell_signals.append(np.nan)
        else:
            _buy_signals.append(np.nan)
            _sell_signals.append(np.nan)

    return _buy_signals, _sell_signals


def get_indexes_of_buy_sell_signals(_data: pd.DataFrame, _long_window: int) -> (List[int], List[int]):
    buy_signals_indexes = []
    sell_signals_indexes = []
    flag = -1

    for i in range(0, len(_data)):
        if i > long_window:
            if _data['MACD'][i] > _data['Signal'][i]:
                sell_signals_indexes.append(np.nan)
                if flag != 1:
                    buy_signals_indexes.append(i)
                    flag = 1
                else:
                    buy_signals_indexes.append(np.nan)
            elif _data['MACD'][i] < _data['Signal'][i]:
                buy_signals_indexes.append(np.nan)
                if flag != 0:
                    sell_signals_indexes.append(i)
                    flag = 0
                else:
                    sell_signals_indexes.append(np.nan)
            else:
                buy_signals_indexes.append(np.nan)
                sell_signals_indexes.append(np.nan)
        else:
            buy_signals_indexes.append(np.nan)
            sell_signals_indexes.append(np.nan)

    return buy_signals_indexes, sell_signals_indexes


def simulate_investment(_data: pd.DataFrame, buy_signals_indexes: List[int], sell_signals_indexes: List[int]) \
        -> (float, List[float], List[float]):
    _wallet = []
    _shares_wallet = []
    money = 1000
    shares = 0
    #  shares_wallet.append(shares)
    first_buy_flag = False
    #  wallet.append(investment)
    for i in range(0, len(_data)):
        if i in buy_signals_indexes:
            shares = money / _data["Close"].iloc[i]
            money = money - shares * _data["Close"].iloc[i]
            _wallet.append(money)
            _shares_wallet.append(shares)
            first_buy_flag = True
        elif i in sell_signals_indexes and first_buy_flag:
            money = shares * _data["Close"].iloc[i]
            _wallet.append(money)
            shares = 0
            _shares_wallet.append(shares)
        else:
            _wallet.append(money)
            _shares_wallet.append(shares)
    money = money + shares * _data["Close"].iloc[-1]
    _wallet[-1] = money
    shares = 0
    _shares_wallet[-1] = shares
    return money, _wallet, _shares_wallet


def choose_heights_of_wallet_and_stocks(_data: pd.DataFrame, _buy_signals: List[float], _sell_signals: List[float]) \
        -> (float, List[float], List[float]):
    _wallet = []
    _shares_wallet = []
    _investment = 1000
    shares = 0
    # shares_wallet.append(shares)
    first_buy_flag = False
    # wallet.append(investment)
    for i in range(0, len(_data)):
        if i in _buy_signals:
            shares = _investment / _data["Close"].iloc[i]
            _investment = _investment - shares * _data["Close"].iloc[i]
            _wallet.append(np.nan)
            _shares_wallet.append(shares)
            first_buy_flag = True
        elif i in _sell_signals and first_buy_flag:
            _investment = shares * _data["Close"].iloc[i]
            _wallet.append(_investment)
            shares = 0
            _shares_wallet.append(np.nan)
        else:
            _wallet.append(np.nan)
            _shares_wallet.append(np.nan)
    _investment = _investment + shares * _data["Close"].iloc[-1]
    _wallet[-1] = _investment
    shares = 0
    _shares_wallet[-1] = shares
    return _investment, _wallet, _shares_wallet


def simulate_buy_and_hold_to_the_end(_data: pd.DataFrame) -> (float, List[float]):
    _investment = 1000
    buy_and_hold_wallet = [np.nan] * len(data)
    buy_and_hold_wallet[0] = _investment
    shares = _investment / data["Close"].iloc[0]
    _investment = _investment - shares * data["Close"].iloc[0]
    _investment = _investment + shares * data["Close"].iloc[-1]
    buy_and_hold_wallet[-1] = _investment

    buy_and_hold_wallet_series = pd.Series(buy_and_hold_wallet)
    buy_and_hold_wallet_interpolated = buy_and_hold_wallet_series.interpolate(method='linear')

    return _investment, buy_and_hold_wallet_interpolated


def simulate_buy_and_hold(_data: pd.DataFrame) -> (float, List[float]):
    _investment = 1000
    buy_and_hold_wallet = [np.nan] * len(data)
    buy_and_hold_wallet[0] = _investment
    shares_at_the_start = _investment / data["Close"].iloc[0]
    _investment_at_the_start = _investment - shares_at_the_start * data["Close"].iloc[0]
    for i in range(1, len(data)):
        money = shares_at_the_start * data["Close"].iloc[i] + _investment_at_the_start
        buy_and_hold_wallet[i] = money

    buy_and_hold_wallet_series = pd.Series(buy_and_hold_wallet)
    buy_and_hold_wallet_interpolated = buy_and_hold_wallet_series.interpolate(method='linear')

    return buy_and_hold_wallet[-1], buy_and_hold_wallet_interpolated


def plot_initial_data() -> None:
    weekly['Date'] = pd.to_datetime(weekly['Date'])
    plt.figure(figsize=(12, 6))
    plt.plot(weekly['Date'], weekly['Close'], color="black", label='Close Price', linewidth=0.5)

    buy_legend = Line2D([0], [0], color='green', marker='^', linestyle='None', label='Buy')
    sell_legend = Line2D([0], [0], color='red', marker='v', linestyle='None', label='Sell')
    close_price_legend = Line2D([0], [0], color='black', linestyle='-', label='Close Price')

    plt.scatter(data['Date'], choose_buy_sell_points_for_initial_plot(data, long_window)[0], color='green', marker='^',
                alpha=1)
    plt.scatter(data['Date'], choose_buy_sell_points_for_initial_plot(data, long_window)[1], color='red', marker='v',
                alpha=1)

    plt.xlabel('Date')
    plt.ylabel('Close')
    plt.title('Netflix Stock Price')
    plt.legend(handles=[close_price_legend, buy_legend, sell_legend], loc='upper left')
    plt.show()


def plot_macd_and_signal() -> None:
    data['Date'] = pd.to_datetime(data['Date'])
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['MACD'], label='MACD', color='blue', linewidth=0.5)
    plt.plot(data['Date'], data['Signal'], label='Signal Line', color='red', linewidth=0.5)
    buy_legend = Line2D([0], [0], color='green', marker='^', linestyle='None', label='Buy')
    sell_legend = Line2D([0], [0], color='red', marker='v', linestyle='None', label='Sell')
    signal_legend = Line2D([0], [0], color='red', linestyle='-', label='Signal')
    macd_legend = Line2D([0], [0], color='blue', linestyle='-', label='MACD')
    plt.scatter(data['Date'], choose_macd_and_signal_intersection_points(data, long_window)[0], color='green',
                marker='^', alpha=1)
    plt.scatter(data['Date'], choose_macd_and_signal_intersection_points(data, long_window)[1], color='red', marker='v',
                alpha=1)

    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.title('MACD Indicator')
    plt.legend(handles=[signal_legend, macd_legend, buy_legend, sell_legend], loc='upper left')
    plt.show()


def plot_common_wallet_simulation(_ax1, _ax2, _data, _wallet_interpolated, _brute_force_wallet):

    _color = 'tab:blue'
    _ax1.set_xlabel('Date')
    _ax1.set_ylabel('MACD Wallet', color=_color)
    _ax1.plot(_data['Date'], _wallet_interpolated, label='macd_wallet', color=_color, linewidth=0.5)
    _ax1.tick_params(axis='y', labelcolor=_color)
    _color = 'tab:red'
    _ax2.set_ylabel('Hold Wallet', color=_color)
    _ax2.plot(_data['Date'], _brute_force_wallet, label='hold_wallet', color=_color, linewidth=0.5)
    _ax2.tick_params(axis='y', labelcolor=_color)
    fig.tight_layout()
    plt.title('Gain from both strategies')
    macd_wallet_legend = Line2D([0], [0], color='blue', linestyle='-', label=' MACD Wallet')
    hold_wallet_legend = Line2D([0], [0], color='red', linestyle='-', label='Hold Wallet')
    plt.legend(handles=[macd_wallet_legend, hold_wallet_legend], loc='upper left')


def plot_macd_and_hold_investment() -> None:
    print(f"Final investment: {investment}")
    _fig, _ax1 = plt.subplots(figsize=(12, 6))
    _ax2 = _ax1.twinx()
    plot_common_wallet_simulation(_ax1, _ax2, data, wallet_interpolated, brute_force_wallet)
    plt.show()


def plot_macd_and_hold_investment_real_scale() -> None:
    _fig, _ax1 = plt.subplots(figsize=(12, 6))
    _ax2 = _ax1.twinx()
    plot_common_wallet_simulation(_ax1, _ax2, data, wallet_interpolated, brute_force_wallet)
    lim_max = max(_ax1.get_ylim()[1], _ax2.get_ylim()[1])
    _ax1.set_ylim(0, lim_max)
    _ax2.set_ylim(0, lim_max)
    plt.show()





if __name__ == '__main__':
    weekly = get_weekly()
    data = weekly.copy(deep=True)
    short_window = 12
    long_window = 26
    data = create_macd_and_signal_in_data(data, short_window, long_window)

    # Plot the data
    plot_initial_data()

    # Plot the MACD and Signal Line
    plot_macd_and_signal()

    buy_signals, sell_signals = get_indexes_of_buy_sell_signals(data, long_window)

    # Filter out NaN values
    filtered_buy_signals = [x for x in buy_signals if not pd.isnull(x)]
    filtered_sell_signals = [x for x in sell_signals if not pd.isnull(x)]

    investment, wallet, shares_wallet = simulate_investment(data, filtered_buy_signals, filtered_sell_signals)
    print(f"Final investment: {investment}")


    investment, wallet, shares_wallet = choose_heights_of_wallet_and_stocks(data, filtered_buy_signals,
                                                                            filtered_sell_signals)
    print(f"Final investment: {investment}")
    wallet_series = pd.Series(wallet)
    shares_wallet_series = pd.Series(shares_wallet)

    # Interpolate NaN values in wallet and shares_wallet
    wallet_interpolated = wallet_series.interpolate(method='linear')
    shares_wallet_interpolated = shares_wallet_series.interpolate(method='linear')

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the wallet values on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Wallet', color=color)
    ax1.plot(data['Date'], wallet_interpolated, label='Wallet', color=color, linewidth=0.5)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot the shares values on the second y-axis
    color = 'tab:red'
    ax2.set_ylabel('Shares', color=color)
    ax2.plot(data['Date'], shares_wallet_interpolated, label='Shares', color=color, linewidth=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    # Adjust the layout to prevent the y-labels from overlapping
    fig.tight_layout()

    # Display the plot
    plt.title('Wallet and Shares value')
    plt.legend()
    plt.show()

    investment, brute_force_wallet = simulate_buy_and_hold(data)
    plot_macd_and_hold_investment_real_scale()
    plot_macd_and_hold_investment()



