from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from pandas import Series

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


def plot_common_wallet_simulation(
        _ax1: Axes,
        _ax2: Axes,
        _data: pd.DataFrame,
        _wallet_interpolated: Series,
        _brute_force_wallet: Series,
        _fig: plt.Figure
) -> None:
    _color = 'tab:blue'
    _ax1.set_xlabel('Date')
    _ax1.set_ylabel('MACD Wallet', color=_color)
    _ax1.plot(_data['Date'], _wallet_interpolated, label='macd_wallet', color=_color, linewidth=0.5)
    _ax1.tick_params(axis='y', labelcolor=_color)
    _color = 'tab:red'
    _ax2.set_ylabel('Hold Wallet', color=_color)
    _ax2.plot(_data['Date'], _brute_force_wallet, label='hold_wallet', color=_color, linewidth=0.5)
    _ax2.tick_params(axis='y', labelcolor=_color)
    _fig.tight_layout()
    plt.title('Gain from both strategies')
    macd_wallet_legend = Line2D([0], [0], color='blue', linestyle='-', label=' MACD Wallet')
    hold_wallet_legend = Line2D([0], [0], color='red', linestyle='-', label='Hold Wallet')
    plt.legend(handles=[macd_wallet_legend, hold_wallet_legend], loc='upper left')


def plot_macd_and_hold_investment() -> None:
    print(f"Final investment: {investment}")
    _fig, _ax1 = plt.subplots(figsize=(12, 6))
    _ax2 = _ax1.twinx()
    plot_common_wallet_simulation(_ax1, _ax2, data, wallet_interpolated, brute_force_wallet, _fig)
    plt.show()


def plot_macd_and_hold_investment_real_scale() -> None:
    _fig, _ax1 = plt.subplots(figsize=(12, 6))
    _ax2 = _ax1.twinx()
    plot_common_wallet_simulation(_ax1, _ax2, data, wallet_interpolated, brute_force_wallet, _fig)
    lim_max = max(_ax1.get_ylim()[1], _ax2.get_ylim()[1])
    _ax1.set_ylim(0, lim_max)
    _ax2.set_ylim(0, lim_max)
    plt.show()


def generate_macd_transactions_csv(_data: pd.DataFrame, _buy_signals: List[int], _sell_signals: List[int],
                                   output_file: str) -> None:
    # Create copies of the buy and sell signals lists to avoid modifying the original lists
    buy_signals_copy = _buy_signals.copy()
    sell_signals_copy = _sell_signals.copy()

    transactions = []
    for buy_index in buy_signals_copy:
        buy_date = _data['Date'].iloc[buy_index]
        buy_cost = _data['Close'].iloc[buy_index]
        for sell_index in sell_signals_copy:
            if sell_index > buy_index:
                sell_date = _data['Date'].iloc[sell_index]
                sell_cost = _data['Close'].iloc[sell_index]
                profit = sell_cost - buy_cost
                transactions.append({
                    'Buy Date': buy_date,
                    'Buy Cost': buy_cost,
                    'Sell Date': sell_date,
                    'Sell Cost': sell_cost,
                    'Profit': profit
                })
                # Remove the sell signal to avoid processing it again
                sell_signals_copy.remove(sell_index)
                break  # Exit the inner loop after processing a sell signal

    # Convert the transactions list to a DataFrame
    transactions_df = pd.DataFrame(transactions)

    # Write the DataFrame to a CSV file
    transactions_df.to_csv(output_file, index=False)

    print(f"MACD transactions CSV file '{output_file}' has been created.")


def find_most_profitable_transaction(_data: pd.DataFrame, _buy_signals: List[int], _sell_signals: List[int]) -> dict:
    sell_signals_copy = _sell_signals.copy()
    most_profitable = {'Profit': float('-inf'), 'Transaction': None, 'Buy Index': None, 'Sell Index': None}
    for buy_index in _buy_signals:
        for sell_index in sell_signals_copy:
            if sell_index > buy_index:  # Ensure sell signal occurs after buy signal
                buy_date = _data['Date'].iloc[buy_index]
                buy_cost = _data['Close'].iloc[buy_index]
                sell_date = _data['Date'].iloc[sell_index]
                sell_cost = _data['Close'].iloc[sell_index]
                profit = sell_cost - buy_cost
                if profit > most_profitable['Profit']:
                    most_profitable = {
                        'Buy Date': buy_date,
                        'Buy Cost': buy_cost,
                        'Sell Date': sell_date,
                        'Sell Cost': sell_cost,
                        'Profit': profit,
                        'Buy Index': buy_index,
                        'Sell Index': sell_index
                    }
                # Remove the sell signal to avoid processing it again
                sell_signals_copy.remove(sell_index)
                break  # Exit the inner loop after processing a sell signal
    return most_profitable


def find_least_profitable_transaction(_data: pd.DataFrame, _buy_signals: List[int], _sell_signals: List[int]) -> dict:
    sell_signals_copy = _sell_signals.copy()

    least_profitable = {'Profit': float('inf'), 'Transaction': None, 'Buy Index': None, 'Sell Index': None}
    for buy_index in _buy_signals:
        for sell_index in sell_signals_copy:  # Use the copy for iteration
            if sell_index > buy_index:  # Ensure sell signal occurs after buy signal
                buy_date = _data['Date'].iloc[buy_index]
                buy_cost = _data['Close'].iloc[buy_index]
                sell_date = _data['Date'].iloc[sell_index]
                sell_cost = _data['Close'].iloc[sell_index]
                profit = sell_cost - buy_cost
                if profit < least_profitable['Profit']:
                    least_profitable = {
                        'Buy Date': buy_date,
                        'Buy Cost': buy_cost,
                        'Sell Date': sell_date,
                        'Sell Cost': sell_cost,
                        'Profit': profit,
                        'Buy Index': buy_index,
                        'Sell Index': sell_index
                    }
                # Remove the sell signal from the copy to avoid processing it again
                sell_signals_copy.remove(sell_index)
                break  # Exit the inner loop after processing a sell signal
    return least_profitable


def plot_transaction(_data: pd.DataFrame, transaction: dict, title: str) -> None:
    # Calculate the start and end indices for the plot
    start_index = max(0, transaction['Buy Index'] - 5)
    end_index = min(len(_data) - 1, transaction['Sell Index'] + 5)

    # Slice the data to include the relevant segment
    relevant_data = _data.iloc[start_index:end_index + 1]

    plt.figure(figsize=(12, 6))

    # Plot the entire stock price range in a lighter color
    plt.plot(relevant_data['Date'], relevant_data['Close'], label='Close Price', color='lightgray', alpha=0.5)

    # Overlay the transaction period in a darker color
    plt.plot(relevant_data['Date'], relevant_data['Close'], label='Transaction Period', color='black')

    # Plot the buy and sell signals within the relevant data segment
    plt.scatter(relevant_data['Date'].iloc[transaction['Buy Index'] - start_index], transaction['Buy Cost'],
                color='green', marker='^', label='Buy Signal')
    plt.scatter(relevant_data['Date'].iloc[transaction['Sell Index'] - start_index], transaction['Sell Cost'],
                color='red', marker='v', label='Sell Signal')

    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_most_profitable_transaction(_data: pd.DataFrame, _buy_signals: List[int], _sell_signals: List[int]) -> None:
    most_profitable = find_most_profitable_transaction(_data, _buy_signals, _sell_signals)
    print(f"Most Profitable Transaction Profit: {most_profitable['Profit']:.2f}")
    print(f"Buy Date: {most_profitable['Buy Date']}")
    print(f"Sell Date: {most_profitable['Sell Date']}")
    plot_transaction(_data, most_profitable, 'Most Profitable Transaction')


def plot_least_profitable_transaction(_data: pd.DataFrame, _buy_signals: List[int], _sell_signals: List[int]) -> None:
    least_profitable = find_least_profitable_transaction(_data, _buy_signals, _sell_signals)
    if 'Buy Date' in least_profitable:
        print(f"Least Profitable Transaction Profit: {least_profitable['Profit']:.2f}")
        print(f"Buy Date: {least_profitable['Buy Date']}")
        print(f"Sell Date: {least_profitable['Sell Date']}")
        plot_transaction(_data, least_profitable, 'Least Profitable Transaction')
    else:
        print("No least profitable transaction found.")


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

    generate_macd_transactions_csv(data, filtered_buy_signals, filtered_sell_signals, 'macd_transactions.csv')

    investment, wallet, shares_wallet = choose_heights_of_wallet_and_stocks(data, filtered_buy_signals,
                                                                            filtered_sell_signals)
    print(f"Final investment: {investment}")
    wallet_series = pd.Series(wallet)
    shares_wallet_series = pd.Series(shares_wallet)
    wallet_interpolated = wallet_series.interpolate(method='linear')
    shares_wallet_interpolated = shares_wallet_series.interpolate(method='linear')

    investment, brute_force_wallet = simulate_buy_and_hold(data)
    plot_macd_and_hold_investment_real_scale()
    plot_macd_and_hold_investment()
    plot_most_profitable_transaction(data, filtered_buy_signals, filtered_sell_signals)
    plot_least_profitable_transaction(data, filtered_buy_signals, filtered_sell_signals)

    data = pd.read_csv('macd_transactions.csv')

    # Calculate the sum of the 'Profit' column
    total_profit = data['Profit'].sum()

    print(f"Total Profit: {total_profit}")
