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
                buy_signals.append(data['MACD'][i])
                flag = 1
            else:
                buy_signals.append(np.nan)
        elif data['MACD'][i] < data['Signal'][i]:
            buy_signals.append(np.nan)
            if flag != 0:
                sell_signals.append(data['MACD'][i])
                flag = 0
            else:
                sell_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    return buy_signals, sell_signals


def get_index_of_buy_sell_signals(data):
    buy_signals = []
    sell_signals = []
    flag = -1

    for i in range(0, len(data)):
        if data['MACD'][i] > data['Signal'][i]:
            sell_signals.append(np.nan)
            if flag != 1:
                buy_signals.append(i)
                flag = 1
            else:
                buy_signals.append(np.nan)
        elif data['MACD'][i] < data['Signal'][i]:
            buy_signals.append(np.nan)
            if flag != 0:
                sell_signals.append(i)
                flag = 0
            else:
                sell_signals.append(np.nan)
        else:
            buy_signals.append(np.nan)
            sell_signals.append(np.nan)

    return buy_signals, sell_signals


def simulate_investment(data, buy_signals, sell_signals):
    wallet = []
    shares_wallet = []
    investment = 1000
    shares = 0
    #shares_wallet.append(shares)
    first_buy_flag = False
    #wallet.append(investment)
    for i in range(0, len(data)):
        if i in buy_signals:
            shares = investment / data["Close"].iloc[i]
            investment = investment - shares * data["Close"].iloc[i]
            wallet.append(investment)
            shares_wallet.append(shares)
            first_buy_flag = True
        elif i in sell_signals and first_buy_flag:
            investment = shares * data["Close"].iloc[i]
            wallet.append(investment)
            shares = 0
            shares_wallet.append(shares)
        else:
            wallet.append(investment)
            shares_wallet.append(shares)
    investment = investment + shares * data["Close"].iloc[-1]
    wallet[-1] = investment
    shares = 0
    shares_wallet[-1] = shares
    return investment, wallet, shares_wallet

def plot_nicer_wallet_and_shares(data,buy_singal,sell_singals):

    wallet = []
    shares_wallet = []
    investment = 1000
    shares = 0
    # shares_wallet.append(shares)
    first_buy_flag = False
    # wallet.append(investment)
    for i in range(0, len(data)):
        if i in buy_signals:
            shares = investment / data["Close"].iloc[i]
            investment = investment - shares * data["Close"].iloc[i]
            wallet.append(np.nan)
            shares_wallet.append(shares)
            first_buy_flag = True
        elif i in sell_signals and first_buy_flag:
            investment = shares * data["Close"].iloc[i]
            wallet.append(investment)
            shares = 0
            shares_wallet.append(np.nan)
        else:
            wallet.append(np.nan)
            shares_wallet.append(np.nan)
    investment = investment + shares * data["Close"].iloc[-1]
    wallet[-1] = investment
    shares = 0
    shares_wallet[-1] = shares
    return investment, wallet, shares_wallet

def simulate_investment_bruteforce(data, buy_signals, sell_signals):
    investment = 1000
    shares = 0
    shares = investment / data["Close"].iloc[0]
    investment = investment - shares * data["Close"].iloc[0]
    investment = investment + shares * data["Close"].iloc[-1]
    shares = 0

    return investment


if __name__ == '__main__':
    weekly = get_weekly()
    data = weekly.copy(deep=True)
    short_window = 12
    long_window = 26
    data = create_macd(data, short_window, long_window)

    # Plot the data
    weekly['Date'] = pd.to_datetime(weekly['Date'])
    plt.figure(figsize=(12, 6))
    plt.plot(weekly['Date'], weekly['Close'], color="black", label='Close Price', linewidth=0.5)
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
    plt.plot(data['Date'], data['MACD'], label='MACD', color='blue', linewidth=0.5)
    plt.plot(data['Date'], data['Signal'], label='Signal Line', color='red', linewidth=0.5)
    plt.scatter(data['Date'], select_intersection_points(data)[0], color='green', marker='^', alpha=1)
    plt.scatter(data['Date'], select_intersection_points(data)[1], color='red', marker='v', alpha=1)

    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.title('MACD Indicator')
    plt.legend()
    plt.show()

    buy_signals, sell_signals = get_index_of_buy_sell_signals(data)

    # Filter out NaN values from buy_signals
    filtered_buy_signals = [x for x in buy_signals if not pd.isnull(x)]

    # Filter out NaN values from sell_signals
    filtered_sell_signals = [x for x in sell_signals if not pd.isnull(x)]
    # data = data.reset_index(drop=True)
    # print("Buy signals:")
    # print(filtered_buy_signals)
    # print("Sell signals:")
    # print(filtered_sell_signals)
    # print(data["Close"].iloc[filtered_buy_signals[0]])
    #
    # print(data["Close"].iloc[filtered_buy_signals[1]])
    # print(data["Close"].iloc[filtered_sell_signals[0]])
    investment, wallet, shares_wallet = simulate_investment(data, filtered_buy_signals, filtered_sell_signals)
    print(f"Final investment: {investment}")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the wallet values on the first y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Wallet', color=color)
    ax1.plot(data['Date'], wallet, label='Wallet', color=color, linewidth=0.5)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot the shares values on the second y-axis
    color = 'tab:red'
    ax2.set_ylabel('Shares', color=color)
    ax2.plot(data['Date'], shares_wallet, label='Shares', color=color, linewidth=0.5)
    ax2.tick_params(axis='y', labelcolor=color)

    # Adjust the layout to prevent the y-labels from overlapping
    fig.tight_layout()

    # Display the plot
    plt.title('Wallet and Shares value')
    plt.legend()
    plt.show()

    investment, wallet, shares_wallet = plot_nicer_wallet_and_shares(data, filtered_buy_signals, filtered_sell_signals)
    print(f"Final investment: {investment}")
    wallet_series = pd.Series(wallet)
    shares_wallet_series = pd.Series(shares_wallet)

    # Interpolate NaN values in wallet and shares_wallet
    wallet_interpolated = wallet_series.interpolate(method='linear')
    shares_wallet_interpolated = shares_wallet_series.interpolate(method='linear')

    # Create a new figure with specified size
    print(f"Final investment: {investment}")
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






