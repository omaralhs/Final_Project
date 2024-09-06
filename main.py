import os
import pickle
import random
import tkinter as tk
from PIL import Image, ImageTk
import customtkinter
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import backtrader as bt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator
from ta.volume import VolumeWeightedAveragePrice
import numpy as np
from deap import base, creator, tools, algorithms

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")


def evaluate(individual, data, return_actions=False):
    num_of_trades = 0
    monthly_profits = []
    trades = 0
    days = 0
    initial_cash = 3000  # Starting cash
    cash = initial_cash
    stock = 0
    buy_sell_actions = []
    buy_prices = []  # Track buy prices for the condition
    last_trade_day = None

    # Extracting columns to arrays for vectorized operations
    close_prices = data['Close'].values
    dates = data.index.values.astype('datetime64[D]').astype(object)
    current_month = dates[0].month
    month_start_value = initial_cash

    for i in range(len(data)):
        row = data.iloc[i]

        # Prevent multiple trades on the same day
        if last_trade_day is not None and dates[i] == last_trade_day:
            continue

        if dates[i].month != current_month:
            # Calculate profit for the previous month
            portfolio_value = cash + stock * close_prices[i - 1]
            monthly_profit = portfolio_value - month_start_value
            monthly_profits.append(monthly_profit)

            # Reset for new month
            month_start_value = portfolio_value
            current_month = dates[i].month

        # Calculate decision value as weighted sum of indicators
        decision = (
            individual[0] * row['McGinley_Dynamic'] +
            individual[1] * row['FVMA'] +
            individual[2] * row['%K'] +
            individual[3] * row['%D'] +
            individual[4] * row['WMA'] +
            individual[5] * row['T3'] +
            individual[7] * row['RSI'] +
            individual[8] * row['VIX'] +
            individual[9] * row['VWAP']
        )

        # Scale decision to [-1, 1]
        decision = max(min(decision, 1), -1)

        # Sell if current price is 10% higher than any of the buy prices
        if stock > 0 and buy_prices:
            highest_buy_price = max(buy_prices)  # Use the highest buy price to compare
            if close_prices[i] >= 1.1 * highest_buy_price:
                num_stocks_to_sell = int(0.8 * stock)  # Sell 80% of stocks
                if num_stocks_to_sell > 0:
                    total_gain = num_stocks_to_sell * close_prices[i]
                    cash += total_gain
                    stock -= num_stocks_to_sell
                    buy_sell_actions.append((dates[i], 'sell', num_stocks_to_sell, close_prices[i]))
                    trades += 1
                    num_of_trades += 1
                    last_trade_day = dates[i]  # Update last trade day
                continue  # Skip the rest of the loop to avoid making another decision

        # Buy decision
        if decision > 0:
            num_stocks_to_buy = int(((cash) / close_prices[i]) * 0.5)  # Fractional buying based on decision
            if num_stocks_to_buy > 0:
                total_cost = num_stocks_to_buy * close_prices[i]
                cash -= total_cost
                stock += num_stocks_to_buy
                buy_prices.append(close_prices[i])  # Track the buy price
                buy_sell_actions.append((dates[i], 'buy', num_stocks_to_buy, close_prices[i]))
                trades += 1
                num_of_trades += 1
                last_trade_day = dates[i]  # Update last trade day

        # Sell decision
        elif decision < 0:
            if stock > 0 and buy_prices:
                lowest_buy_price = min(buy_prices)  # Use the lowest buy price to compare
                if close_prices[i] > lowest_buy_price:  # Check if the current price is higher than the buy price
                    num_stocks_to_sell = int(0.6 * stock)  # Fractional selling based on decision
                    if num_stocks_to_sell > 0:
                        total_gain = num_stocks_to_sell * close_prices[i]
                        cash += total_gain
                        stock -= num_stocks_to_sell
                        buy_prices = []  # Clear the buy prices as stocks are sold
                        buy_sell_actions.append((dates[i], 'sell', num_stocks_to_sell, close_prices[i]))
                        trades += 1
                        num_of_trades += 1
                        last_trade_day = dates[i]

        days += 1

    # Calculate profit for the last month
    portfolio_value = cash + stock * close_prices[-1]
    monthly_profit = portfolio_value - month_start_value
    monthly_profits.append(monthly_profit)

    # Calculate overall fitness
    average_monthly_profit = sum(monthly_profits)
    fitness = average_monthly_profit
    profit = sum(monthly_profits)

    if return_actions:
        return fitness, buy_sell_actions, profit, num_of_trades
    else:
        return fitness,

def plot_stock_with_signals(data, buy_sell_actions):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot the closing price
    ax.plot(data.index, data['Close'], color='blue', label='Close Price')

    # Plot buy and sell signals
    buy_points = []
    sell_points = []
    annotations = []

    for action in buy_sell_actions:
        if action[1] == 'buy':
            buy_point = ax.scatter(action[0], action[3], color='green', marker='^', alpha=1, label='Buy')
            buy_points.append((buy_point, action[0], f'Buy: {action[2]} At ${action[3]:.2f}'))
        elif action[1] == 'sell':
            sell_point = ax.scatter(action[0], action[3], color='red', marker='v', alpha=1, label='Sell')
            sell_points.append((sell_point, action[0], f'Sell: {action[2]} At ${action[3]:.2f}'))

    ax.set_title('Stock Price with Buy and Sell Signals')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend(loc='best')
    ax.grid(True)

    # Rotate and format the dates on the x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    # Define font properties for the annotations
    font = FontProperties()
    font.set_family('serif')
    font.set_style('italic')
    font.set_size(10)

    # Annotate buy/sell points on hover
    def hover(event):
        if event.inaxes == ax:
            for point, date, text in buy_points + sell_points:
                cont, _ = point.contains(event)
                if cont:
                    # Remove previous annotations
                    while annotations:
                        annotations.pop().remove()
                    # Add the new annotation
                    annotation = ax.annotate(
                        text,
                        xy=(mdates.date2num(date), point.get_offsets()[0][1]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        fontproperties=font,
                        bbox=dict(facecolor='green' if 'Buy' in text else 'red', alpha=0.8),
                        arrowprops=dict(facecolor='black', arrowstyle='->')
                    )
                    annotations.append(annotation)
                    fig.canvas.draw_idle()
                    return

        # If no point is hovered, remove the annotation
        while annotations:
            annotations.pop().remove()
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('motion_notify_event', hover)

    plt.show()


def run_genetic_algorithm(stock_name):
    start = '2010-01-01'
    end = '2020-01-01'
    stock = stock_name

    data = read_stock_data(stock)

    # Calculate indicators
    construct_signals(data)

    # Fill NaN values
    data.fillna(method='bfill', inplace=True)

    # Create the DEAP toolbox
    # (remaining code for DEAP toolbox setup omitted for brevity)
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register('attr_float', random.uniform, -1, 1)
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.attr_float, n=20)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    toolbox.register('evaluate', lambda ind: (evaluate(ind, data)[0],))
    toolbox.register('mate', tools.cxBlend, alpha=0.5)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register('select', tools.selTournament, tournsize=3)

    # Genetic Algorithm parameters
    population_size = 100
    generations = 5
    crossover_prob = 0.7
    mutation_prob = 0.2

    def main():
        population = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)

        algorithms.eaSimple(population, toolbox, cxpb=crossover_prob, mutpb=mutation_prob,
                            ngen=generations, stats=stats, halloffame=hof, verbose=True)

        return population, stats, hof

    if __name__ == '__main__':
        # Execute the GA
        population, stats, hof = main()
        best_individual = hof[0]
        best_fitness = best_individual.fitness.values[0]
        print('Best individual:', best_individual, 'with fitness:', best_fitness)

        # Save the best individual to a file
        with open('best_individual.pkl', 'wb') as f:
            pickle.dump(best_individual, f)

        # Load new data
        start = '2019-01-01'
        end = '2021-01-01'

        new_data = download_data(stock,start,end)

        # Calculate indicators for new data
        construct_signals(new_data)

        # Fill NaN values
        new_data.fillna(method='bfill', inplace=True)

        # Load the best individual from the file
        with open('best_individual.pkl', 'rb') as f:
            best_individual = pickle.load(f)

        # Evaluate the best individual on the new data
        fitness_new_data, buy_sell_actions, profit, num_of_trades = evaluate(best_individual, new_data, return_actions=True)
        print(f'number of trades on this data : {num_of_trades}')

        print('Fitness on new data:', fitness_new_data)
        print('Profit made:', profit)

        # Print buy and sell actions
        for action in buy_sell_actions:
            date, action_type, quantity, price = action
            print(f"{action_type.capitalize()} {quantity} shares on {pd.to_datetime(date).date()} at ${price:.2f}")

        # Plot the stock price with buy and sell signals
        plot_stock_with_signals(new_data, buy_sell_actions)

# Now, we need to set up the callback function for the AI button
def ai_button_callback(stock):
    run_genetic_algorithm(stock)


class BollingerBandsReversionStrategy(bt.Strategy):
    params = (
        ("period", 20),  # Lookback period for Bollinger Bands
        ("num_std", 2),   # Number of standard deviations for Bollinger Bands
    )

    def __init__(self):
        self.stock = self.datas[0]  # Assuming the only data is for the stock
        self.bollinger_bands = bt.indicators.BollingerBands(self.stock.close, period=self.params.period, devfactor=self.params.num_std)

    def next(self):
        if self.stock.close < self.bollinger_bands.lines.bot and not self.position:
            self.buy(size=100)  # Buying fixed size for simplicity
        elif self.stock.close > self.bollinger_bands.lines.top and self.position:
            self.sell(size=100)  # Selling fixed size for simplicity



class RSIOverboughtOversoldStrategy(bt.Strategy):
    params = (
        ("rsi_period", 14),  # RSI period
        ("overbought_threshold", 70),  # RSI level indicating overbought condition
        ("oversold_threshold", 30),  # RSI level indicating oversold condition
    )

    def __init__(self):
        self.stock = self.datas[0]  # Assuming the only data is for the stock
        self.rsi = bt.indicators.RSI(self.stock.close, period=self.params.rsi_period)

    def next(self):
        if self.rsi > self.params.overbought_threshold and not self.position:
            self.sell(size=100)  # Short selling for simplicity
        elif self.rsi < self.params.oversold_threshold and self.position:
            self.buy(size=100)  # Buying to cover the short position


class SimpleMovingAverageCrossoverStrategy(bt.Strategy):
    params = (
        ("short_period", 50),  # Short moving average period
        ("long_period", 200),  # Long moving average period
    )

    def __init__(self):
        self.stock = self.datas[0]  # Assuming the only data is for the stock
        self.short_sma = bt.indicators.SimpleMovingAverage(self.stock.close, period=self.params.short_period)
        self.long_sma = bt.indicators.SimpleMovingAverage(self.stock.close, period=self.params.long_period)

    def next(self):
        if self.short_sma > self.long_sma and not self.position:
            self.buy(size=100)  # Buying fixed size for simplicity
        elif self.short_sma < self.long_sma and self.position:
            self.sell(size=100)  # Selling fixed size for simplicity



class SimpleMomentumStrategy(bt.Strategy):
    params = (
        ("lookback_period", 90),  # Lookback period for momentum calculation
        ("buy_threshold", 0.05),  # Momentum threshold for buying
        ("sell_threshold", 0),    # Momentum threshold for selling
    )

    def __init__(self):
        self.stock = self.datas[0]  # Assuming the only data is for MSFT
        self.trades = []  # List to store trades

    def next(self):
        momentum = (self.stock.close[0] - self.stock.close[-self.params.lookback_period]) / self.stock.close[-self.params.lookback_period]

        if momentum > self.params.buy_threshold and not self.position:
            self.buy(size=100)  # Buying fixed size for simplicity
            self.trades.append(('Buy', self.data.datetime.date()))  # Append buy trade
            print(f"Buying at {self.data.datetime.date()}")
        elif momentum < self.params.sell_threshold and self.position:
            self.sell(size=100)  # Selling fixed size for simplicity
            self.trades.append(('Sell', self.data.datetime.date()))  # Append sell trade
            print(f"Selling at {self.data.datetime.date()}")

strategies = [
    ("Simple Momentum Strategy", SimpleMomentumStrategy),
    ("Simple Moving Average Crossover Strategy", SimpleMovingAverageCrossoverStrategy),
    ("RSI Overbought/Oversold Strategy", RSIOverboughtOversoldStrategy),
    ("Bollinger Bands Reversion Strategy", BollingerBandsReversionStrategy),
]

def display_strategies():
    # Destroy all widgets in the current frame
    for widget in new_frame.winfo_children():
        widget.destroy()

    # Create a new frame for displaying strategies
    strategy_selection_frame = customtkinter.CTkFrame(master=new_frame)
    strategy_selection_frame.pack(pady=20, padx=60, fill="both", expand=True)

    # Function to run the selected strategy
    def run_selected_strategy(strategy_class):
        stock_name = selected_stock_name[selected_stock_name.index("(") + 1:-1]
        run_strategy(stock_name, strategy_class)

    # Add logo
    logo_label = tk.Label(strategy_selection_frame, image=logo_photo)
    logo_label.pack()

    # Add back button
    def back_to_main_frame():
        strategy_selection_frame.pack_forget()
        on_Stock_select(selected_stock_name)  # Go back to the main frame

    back_button = customtkinter.CTkButton(master=strategy_selection_frame, text="Back", command=back_to_main_frame)
    back_button.pack(pady=12, padx=10)

    # Display each strategy as a button
    for strategy_name, strategy_class in strategies:
        strategy_button = customtkinter.CTkButton(master=strategy_selection_frame, text=strategy_name,
                                                  command=lambda s=strategy_class: run_selected_strategy(s))
        strategy_button.pack(pady=5)



def read_stock_data(stock_name ):
    file_path = os.path.join('sp500_data', f"{stock_name}.csv")
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        return df
    else:
        print(f"File for {stock_name} not found.")
        return None


def run_strategy(stock_name, strategy_class):
    # Creating the strategy frame
    strategy_frame = customtkinter.CTkFrame(master=root)
    strategy_frame.pack(pady=20, padx=60, fill="both", expand=True)

    # Download stock data
    df = read_stock_data(stock_name)
    feed = bt.feeds.PandasData(dataname=df)

    # Add strategy
    cerebro = bt.Cerebro()
    cerebro.adddata(feed)
    cerebro.addstrategy(BollingerBandsReversionStrategy)
    cerebro.broker.set_cash(10000)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    global previous_frame



    def back_to_list():
        strategy_frame.pack_forget()
        previous_frame.pack(pady=20, padx=60, fill="both", expand=True)

    previous_frame = new_frame
    new_frame.pack_forget()

    # Clear the frame
    for widget in strategy_frame.winfo_children():
        widget.destroy()

    # Close all existing figures
    plt.close('all')

    # Plot the strategy's performance without volume in the frame
    fig = cerebro.plot(style='line', volume=False)[0][0]  # Get the figure

    # Embed the figure in the Tkinter frame
    canvas = FigureCanvasTkAgg(fig, master=strategy_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)

    # Add navigation toolbar
    toolbar = NavigationToolbar2Tk(canvas, strategy_frame)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)


    new_frame.pack_forget()

    # Add back button
    back_button = customtkinter.CTkButton(master=strategy_frame, text="Back", command=back_to_list)
    back_button.pack(pady=12, padx=10)


def download_data(stock, start, end):
    data = yf.download(stock, start, end)
    df = pd.DataFrame(data)

    return df





def calculate_rsi(data, period=14):
        data['move'] = data['Close'] - data['Close'].shift(1)
        data['up'] = np.where(data['move'] > 0, data['move'], 0)
        data['down'] = np.where(data['move'] < 0, data['move'], 0)
        data['average_gain'] = data['up'].rolling(period).mean()
        data['average_loss'] = data['down'].abs().rolling(period).mean()
        data['relative_strength'] = data['average_gain'] / data['average_loss']
        data['RSI'] = 100.0 - (100.0 / (1.0 + data['relative_strength']))

def calculate_ema(data, period=20):
        data['EMA'] = data['Close'].ewm(span=period, adjust=False).mean()

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
        data['ShortEMA'] = data['Close'].ewm(span=short_period, adjust=False).mean()
        data['LongEMA'] = data['Close'].ewm(span=long_period, adjust=False).mean()
        data['MACD'] = data['ShortEMA'] - data['LongEMA']
        data['Signal_Line'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()

def calculate_bollinger_bands(data, period=20, num_std=2):
        data['SMA'] = data['Close'].rolling(window=period).mean()
        data['STD'] = data['Close'].rolling(window=period).std()
        data['Upper_Band'] = data['SMA'] + (data['STD'] * num_std)
        data['Lower_Band'] = data['SMA'] - (data['STD'] * num_std)

def calculate_vix(data, period=14):
        data['typical_price'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['log_returns'] = np.log(data['typical_price'] / data['typical_price'].shift(1))
        data['volatility'] = data['log_returns'].rolling(window=period).std() * np.sqrt(252)
        data['VIX'] = data['volatility'] * 100

def calculate_atr(data, period=14):
        data['TR'] = np.maximum(np.maximum(data['High'] - data['Low'], abs(data['High'] - data['Close'].shift(1))),
                                abs(data['Low'] - data['Close'].shift(1)))
        data['ATR'] = data['TR'].rolling(window=period).mean()

def calculate_stochastic_oscillator(data, k_period=14, d_period=3):
        data['%K'] = ((data['Close'] - data['Low'].rolling(window=k_period).min()) /
                      (data['High'].rolling(window=k_period).max() - data['Low'].rolling(window=k_period).min())) * 100
        data['%D'] = data['%K'].rolling(window=d_period).mean()

def calculate_vwap(data):
        data['Volume_Price'] = data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3
        data['Cumulative_Volume_Price'] = data['Volume_Price'].cumsum()
        data['Cumulative_Volume'] = data['Volume'].cumsum()
        data['VWAP'] = data['Cumulative_Volume_Price'] / data['Cumulative_Volume']

def calculate_adx(data, period=14):
        data['tr'] = data[['High', 'Low', 'Close']].max(axis=1) - data[['High', 'Low', 'Close']].min(axis=1)
        data['atr'] = data['tr'].rolling(window=period).mean()
        data['up_move'] = data['High'] - data['High'].shift(1)
        data['down_move'] = data['Low'].shift(1) - data['Low']
        data['plus_dm'] = np.where((data['up_move'] > data['down_move']) & (data['up_move'] > 0), data['up_move'], 0)
        data['minus_dm'] = np.where((data['down_move'] > data['up_move']) & (data['down_move'] > 0), data['down_move'],
                                    0)
        data['plus_di'] = 100 * (data['plus_dm'].ewm(alpha=1 / period).mean() / data['atr'])
        data['minus_di'] = 100 * (data['minus_dm'].ewm(alpha=1 / period).mean() / data['atr'])
        data['dx'] = 100 * np.abs((data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di']))
        data['ADX'] = data['dx'].rolling(window=period).mean()

def calculate_fvma(data, period=20):
        data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
        data['Volume_Typical'] = data['Volume'] * data['Typical_Price']
        data['FVMA'] = data['Volume_Typical'].rolling(window=period).sum() / data['Volume'].rolling(window=period).sum()

def calculate_wma(data, window):
        weights = np.arange(1, window + 1)
        return data.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

def calculate_mcginley_dynamic(data, period=10):
        mcg = [0] * len(data)
        for i in range(period, len(data)):
            if mcg[i - 1] == 0:  # Avoid division by zero
                mcg[i] = data['Close'][i]
            else:
                mcg[i] = mcg[i - 1] + (data['Close'][i] - mcg[i - 1]) / (period * (data['Close'][i] / mcg[i - 1]) ** 4)
        data['McGinley_Dynamic'] = mcg

def calculate_t3(data, period):
        e1 = data['Close'].ewm(span=period, adjust=False).mean()
        e2 = e1.ewm(span=period, adjust=False).mean()
        e3 = e2.ewm(span=period, adjust=False).mean()
        e4 = e3.ewm(span=period, adjust=False).mean()
        e5 = e4.ewm(span=period, adjust=False).mean()
        t3 = 3 * (e1 - e2) + 3 * (e3 - e4) + e5
        return t3

def construct_signals(data, ma_period=60, rsi_period=14, vix_period=14, atr_period=14):
        data['SMA'] = data['Close'].rolling(window=ma_period).mean()
        data['trend'] = (data['Open'] - data['SMA']) * 100
        calculate_rsi(data, rsi_period)
        calculate_ema(data)
        calculate_macd(data)
        calculate_bollinger_bands(data)
        calculate_vix(data, vix_period)
        calculate_atr(data, atr_period)
        calculate_stochastic_oscillator(data)
        calculate_vwap(data)
        calculate_adx(data)
        calculate_fvma(data)
        data['WMA'] = calculate_wma(data['Close'], 10)
        data['T3'] = calculate_t3(data, period=20)
        calculate_mcginley_dynamic(data, 5)









def switch_to_new_frame():
    main_frame.pack_forget()
    new_frame.pack(pady=20, padx=60, fill="both", expand=True)

def switch_to_previous_frame():
    new_frame.pack_forget()
    main_frame.pack(pady=20, padx=60, fill="both", expand=True)

def TESTING():
    print('TESTING.....')

def update_listbox(event):
    search_query = search_var.get().lower()
    canvas.delete("all")
    for i, item in enumerate(stocks):
        bg_color = "gray15" if i % 2 == 0 else "gray20"
        canvas.create_rectangle(0, i * 25, 1000, (i + 1) * 25, fill=bg_color)
        canvas.create_text(10, i * 25 + 12, anchor=tk.W, text=item, fill="white", font=("Arial", 12))

def on_search(event):
    search_query = search_var.get().lower()
    canvas.delete("all")
    displayed_items = [item for item in stocks if search_query.lower() in item.lower()]
    for i, item in enumerate(displayed_items):
        bg_color = "gray15" if i % 2 == 0 else "gray20"
        canvas.create_rectangle(0, i * 25, 1000, (i + 1) * 25, fill=bg_color)
        canvas.create_text(10, i * 25 + 12, anchor=tk.W, text=item, fill="white", font=("Arial", 12))


def on_select2(event):
    global selected_stock_name
    y = event.y
    index = y // 25
    if index < len(stocks):
        selected_stock_name = stocks[index]
        stock_name = selected_stock_name[selected_stock_name.index("(") + 1:-1]
        on_Stock_select(stock_name)

def on_Stock_select(stock_name):
    for widget in new_frame.winfo_children():
        widget.destroy()

    switch_to_new_frame()

    logo_label = tk.Label(new_frame, image=logo_photo)
    logo_label.pack()

    back_button = customtkinter.CTkButton(master=new_frame, text="Back", command=switch_to_previous_frame)
    back_button.pack(pady=12, padx=10)

    Strategy_button = customtkinter.CTkButton(master=new_frame, text="Try Strategies", command=display_strategies)
    Strategy_button.pack(pady=12, padx=10)

    Graph_button = customtkinter.CTkButton(master=new_frame, text="See graphs", command=lambda: on_select(stock_name))
    Graph_button.pack(pady=12, padx=10)

    AI_button = customtkinter.CTkButton(master=new_frame, text="USE AI", command=lambda: ai_button_callback(stock_name))
    AI_button.pack(pady=12, padx=10)



def on_select(stock_name):
    global previous_frame

    def back_to_list():
        stock_frame.pack_forget()
        previous_frame.pack(pady=20, padx=60, fill="both", expand=True)

    previous_frame = new_frame
    new_frame.pack_forget()

    stock_frame = customtkinter.CTkFrame(master=root)
    label = customtkinter.CTkLabel(master=stock_frame, text=f"Stock: {stock_name}")
    label.pack(pady=12, padx=10)
    back_button = customtkinter.CTkButton(master=stock_frame, text="Back to OPTIONS", command=back_to_list)
    back_button.pack(pady=12, padx=10)
    stock_frame.pack(pady=20, padx=60, fill="both", expand=True)

    # Function to update the plot based on checkbox selection
    def update_plot():
        ax.clear()
        ax.plot(stock_data.index, stock_data['Close'], label='Close Price')

        if var_ma.get():
            ax.plot(stock_data.index, stock_data['ma'], label='MA 20', linestyle='--')
        if var_bollinger.get():
            ax.plot(stock_data.index, stock_data['upper_band'], label='Upper Band', linestyle='-.')
            ax.plot(stock_data.index, stock_data['lower_band'], label='Lower Band', linestyle='-.')
            ax.fill_between(stock_data.index, stock_data['lower_band'], stock_data['upper_band'], color='gray', alpha=0.3)
        if var_ma200.get():
            ax.plot(stock_data.index, stock_data['ma200'], label='MA 200', linestyle='--')
        if T3.get():
            ax.plot(stock_data.index, stock_data['T3'], label='T3', linestyle='-.')

        if var_FVMA.get():
            ax.plot(stock_data.index, stock_data['FVMA'], label='FVMA', linestyle='-')
        if var_WMA.get():
            ax.plot(stock_data.index, stock_data['WMA'], label='WMA', linestyle='-')
        if var_McGinley.get():
            ax.plot(stock_data.index, stock_data['McGinley_Dynamic'], label='McGinley Dynamic', linestyle='-')

        ax.legend()
        canvas.draw()

    # Create a frame for checkboxes
    checkboxes_frame = customtkinter.CTkFrame(master=stock_frame)

    T3 = tk.BooleanVar()
    check_T3 = tk.Checkbutton(checkboxes_frame, text="T3", variable=T3, command=update_plot)
    check_T3.pack(side=tk.LEFT)  # Pack horizontally within the checkboxes_frame


    var_ma200 = tk.BooleanVar()
    check_ma200 = tk.Checkbutton(checkboxes_frame, text="MA 200", variable=var_ma200, command=update_plot)
    check_ma200.pack(side=tk.LEFT)  # Pack horizontally within the checkboxes_frame

    var_ma = tk.BooleanVar()
    check_ma = tk.Checkbutton(checkboxes_frame, text="MA 20", variable=var_ma, command=update_plot)
    check_ma.pack(side=tk.LEFT)  # Pack horizontally within the checkboxes_frame

    var_bollinger = tk.BooleanVar()
    check_bollinger = tk.Checkbutton(checkboxes_frame, text="Bollinger Bands", variable=var_bollinger, command=update_plot)
    check_bollinger.pack(side=tk.LEFT)  # Pack horizontally within the checkboxes_frame

    var_FVMA = tk.BooleanVar()
    check_FVMA = tk.Checkbutton(checkboxes_frame, text="FVMA", variable=var_FVMA, command=update_plot)
    check_FVMA.pack(side=tk.LEFT)

    var_WMA = tk.BooleanVar()
    check_WMA = tk.Checkbutton(checkboxes_frame, text="WMA", variable=var_WMA, command=update_plot)
    check_WMA.pack(side=tk.LEFT)

    var_McGinley = tk.BooleanVar()
    check_McGinley = tk.Checkbutton(checkboxes_frame, text="McGinley Dynamic", variable=var_McGinley,
                                    command=update_plot)
    check_McGinley.pack(side=tk.LEFT)
    # Pack the checkboxes_frame
    checkboxes_frame.pack(pady=(0, 10))  # Add some padding below the button

    start_date = dt.datetime(2011, 1, 1)
    end_date = dt.datetime(2012, 1, 1)
    stock_data = read_stock_data(stock_name)
    construct_signals(stock_data)
    stock_data['tp'] = (stock_data['Close'] + stock_data['Low'] + stock_data['High']) / 3
    stock_data['std'] = stock_data['tp'].rolling(20).std()
    stock_data['ma'] = stock_data['tp'].rolling(20).mean()
    stock_data['ma200'] = stock_data['tp'].rolling(200).mean()
    stock_data['upper_band'] = stock_data['ma'] + 2 * stock_data['std']
    stock_data['lower_band'] = stock_data['ma'] - 2 * stock_data['std']
    stock_data['sma20'] = stock_data['Close'].rolling(window=20).mean()  # Adding SMA 20
    stock_data['sma200'] = stock_data['Close'].rolling(window=200).mean()  # Adding SMA 200
    stock_data = stock_data.dropna()

    # Code for creating and packing the plot_frame remains unchanged
    plot_frame = customtkinter.CTkFrame(master=stock_frame)
    plot_frame.pack(pady=20, padx=60, fill="both", expand=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(stock_data.index, stock_data['Close'], label='Close Price')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title(f'Bollinger Bands for {stock_name}')
    plt.legend()

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    toolbar = NavigationToolbar2Tk(canvas, stock_frame)
    toolbar.update()
    toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    update_plot()  # Update the plot initially


stocks = [
    "3M Company (MMM)", "A.O. Smith Corporation (AOS)", "Abbott Laboratories (ABT)",
    "AbbVie Inc. (ABBV)", "Accenture plc (ACN)", "Activision Blizzard, Inc. (ATVI)",
    "Adobe Inc. (ADBE)", "Advance Auto Parts, Inc. (AAP)", "AES Corporation (AES)",
    "Aflac Incorporated (AFL)", "Agilent Technologies, Inc. (A)", "Air Products and Chemicals, Inc. (APD)",
    "Akamai Technologies, Inc. (AKAM)", "Alaska Air Group, Inc. (ALK)", "Albemarle Corporation (ALB)",
    "Alexandria Real Estate Equities, Inc. (ARE)", "Align Technology, Inc. (ALGN)", "Allegion plc (ALLE)",
    "Alliant Energy Corporation (LNT)", "Allstate Corporation (ALL)", "Alphabet Inc. (GOOGL)",
    "Alphabet Inc. (GOOG)", "Altria Group, Inc. (MO)", "Amazon.com, Inc. (AMZN)",
    "Amcor plc (AMCR)", "AMD (AMD)", "American Airlines Group Inc. (AAL)",
    "American Electric Power Company, Inc. (AEP)", "American Express Company (AXP)",
    "American International Group, Inc. (AIG)", "American Tower Corporation (AMT)",
    "American Water Works Company, Inc. (AWK)", "Ameriprise Financial, Inc. (AMP)",
    "AmerisourceBergen Corporation (ABC)", "Ametek, Inc. (AME)", "Amgen Inc. (AMGN)",
    "Amphenol Corporation (APH)", "Analog Devices, Inc. (ADI)", "Ansys, Inc. (ANSS)",
    "Aon plc (AON)", "APA Corporation (APA)", "Apple Inc. (AAPL)",
    "Applovin Corporation (APP)", "Applied Materials, Inc. (AMAT)",
    "Aptiv PLC (APTV)", "Arch Capital Group Ltd. (ACGL)", "Archer-Daniels-Midland Company (ADM)",
    "Arista Networks, Inc. (ANET)", "Arthur J. Gallagher & Co. (AJG)", "Assurant, Inc. (AIZ)",
    "AT&T Inc. (T)", "Atmos Energy Corporation (ATO)", "Autodesk, Inc. (ADSK)",
    "AutoZone, Inc. (AZO)", "AvalonBay Communities, Inc. (AVB)", "Avery Dennison Corporation (AVY)",
    "Baker Hughes Company (BKR)", "Ball Corporation (BALL)", "Bank of America Corporation (BAC)",
    "The Bank of New York Mellon Corporation (BK)", "Baxter International Inc. (BAX)",
    "Becton, Dickinson and Company (BDX)", "Berkshire Hathaway Inc. (BRK.B)",
    "Best Buy Co., Inc. (BBY)", "Bio-Rad Laboratories, Inc. (BIO)", "Bio-Techne Corporation (TECH)",
    "Biogen Inc. (BIIB)", "BlackRock, Inc. (BLK)", "Boeing Company (BA)",
    "Booking Holdings Inc. (BKNG)", "BorgWarner Inc. (BWA)", "Boston Properties, Inc. (BXP)",
    "Boston Scientific Corporation (BSX)", "Bristol-Myers Squibb Company (BMY)",
    "Broadcom Inc. (AVGO)", "Broadridge Financial Solutions, Inc. (BR)", "Brown & Brown, Inc. (BRO)",
    "Brown-Forman Corporation (BF.B)", "C.H. Robinson Worldwide, Inc. (CHRW)",
    "Cadence Design Systems, Inc. (CDNS)", "Caesars Entertainment, Inc. (CZR)",
    "Camden Property Trust (CPT)", "Campbell Soup Company (CPB)", "Capital One Financial Corporation (COF)",
    "Cardinal Health, Inc. (CAH)", "CarMax, Inc. (KMX)", "Carnival Corporation & plc (CCL)",
    "Carrier Global Corporation (CARR)", "Catalent, Inc. (CTLT)", "Caterpillar Inc. (CAT)",
    "Cboe Global Markets, Inc. (CBOE)", "CBRE Group, Inc. (CBRE)", "CDW Corporation (CDW)",
    "Celanese Corporation (CE)", "Centene Corporation (CNC)", "CenterPoint Energy, Inc. (CNP)",
    "Ceridian HCM Holding Inc. (CDAY)", "CF Industries Holdings, Inc. (CF)",
    "Charles River Laboratories International, Inc. (CRL)", "Charles Schwab Corporation (SCHW)",
    "Charter Communications, Inc. (CHTR)", "Chevron Corporation (CVX)", "Chipotle Mexican Grill, Inc. (CMG)",
    "Chubb Limited (CB)", "Church & Dwight Co., Inc. (CHD)", "Cigna Group (CI)",
    "Cincinnati Financial Corporation (CINF)", "Cintas Corporation (CTAS)", "Cisco Systems, Inc. (CSCO)",
    "Citigroup Inc. (C)", "Citizens Financial Group, Inc. (CFG)", "Clorox Company (CLX)",
    "CME Group Inc. (CME)", "CMS Energy Corporation (CMS)", "Coca-Cola Company (KO)",
    "Cognizant Technology Solutions Corporation (CTSH)", "Colgate-Palmolive Company (CL)",
    "Comcast Corporation (CMCSA)", "Comerica Incorporated (CMA)", "Conagra Brands, Inc. (CAG)",
    "ConocoPhillips (COP)", "Consolidated Edison, Inc. (ED)", "Constellation Brands, Inc. (STZ)",
    "Constellation Energy Corporation (CEG)", "CooperCompanies Inc. (COO)", "Copart, Inc. (CPRT)",
    "Corning Inc. (GLW)", "Corteva, Inc. (CTVA)", "Costco Wholesale Corporation (COST)",
    "Coterra Energy Inc. (CTRA)", "Crown Castle Inc. (CCI)", "CSX Corporation (CSX)",
    "Cummins Inc. (CMI)", "CVS Health Corporation (CVS)", "D.R. Horton, Inc. (DHI)",
    "Danaher Corporation (DHR)", "Darden Restaurants, Inc. (DRI)", "DaVita Inc. (DVA)",
    "Deere & Company (DE)", "Delta Air Lines, Inc. (DAL)", "DENTSPLY SIRONA Inc. (XRAY)",
    "Devon Energy Corporation (DVN)", "DexCom, Inc. (DXCM)", "Diamondback Energy, Inc. (FANG)",
    "Digital Realty Trust, Inc. (DLR)", "Discover Financial Services (DFS)", "Dish Network Corporation (DISH)",
    "Disney (DIS)", "DISH Network Corporation (DISH)", "Dollar General Corporation (DG)",
    "Dollar Tree, Inc. (DLTR)", "Dominion Energy, Inc. (D)", "Dover Corporation (DOV)",
    "Dow Inc. (DOW)", "DTE Energy Company (DTE)", "Duke Energy Corporation (DUK)",
    "DuPont de Nemours, Inc. (DD)", "DXC Technology Company (DXC)", "Eastman Chemical Company (EMN)",
    "Eaton Corporation plc (ETN)", "eBay Inc. (EBAY)", "Ecolab Inc. (ECL)",
    "Edison International (EIX)", "Edwards Lifesciences Corporation (EW)", "Electronic Arts Inc. (EA)",
    "Elevance Health, Inc. (ELV)", "Eli Lilly and Company (LLY)", "Emerson Electric Co. (EMR)",
    "Enphase Energy, Inc. (ENPH)", "Entergy Corporation (ETR)", "EOG Resources, Inc. (EOG)",
    "EPAM Systems, Inc. (EPAM)", "Equifax Inc. (EFX)", "Equinix, Inc. (EQIX)",
    "Equity Residential (EQR)", "Essex Property Trust, Inc. (ESS)", "Estee Lauder Companies Inc. (EL)",
    "Eversource Energy (ES)", "Exelon Corporation (EXC)", "Expedia Group, Inc. (EXPE)",
    "Expeditors International of Washington, Inc. (EXPD)", "Extra Space Storage Inc. (EXR)",
    "Exxon Mobil Corporation (XOM)", "F5, Inc. (FFIV)", "Fastenal Company (FAST)",
    "Federal Realty Investment Trust (FRT)", "FedEx Corporation (FDX)", "Fidelity National Information Services, Inc. (FIS)",
    "Fifth Third Bancorp (FITB)", "First Republic Bank (FRC)", "First Solar, Inc. (FSLR)",
    "FirstEnergy Corp. (FE)", "Fiserv, Inc. (FISV)", "Fleetcor Technologies, Inc. (FLT)",
    "FMC Corporation (FMC)", "Ford Motor Company (F)", "Fortive Corporation (FTV)",
    "Fox Corporation (FOXA)", "Fox Corporation (FOX)", "Franklin Resources, Inc. (BEN)"
]

root = customtkinter.CTk()
root.geometry("1000x750")

main_frame = customtkinter.CTkFrame(master=root)
main_frame.pack(pady=20, padx=60, fill="both", expand=True)

logo_image = Image.open("img.png")
logo_photo = ImageTk.PhotoImage(logo_image)
logo_label = tk.Label(main_frame, image=logo_photo)
logo_label.pack()

new_frame = customtkinter.CTkFrame(master=root)
label = customtkinter.CTkLabel(master=new_frame, text="Choose an option")
label.pack(pady=12, padx=10)

search_var = tk.StringVar()
search_entry = tk.Entry(master=main_frame, textvariable=search_var, font=("Arial", 12), bd=2, relief=tk.FLAT)
search_entry.pack(pady=5, padx=10, fill=tk.X)
search_entry.bind("<KeyRelease>", on_search)

canvas = tk.Canvas(master=main_frame, bd=0, highlightthickness=0, height=500, width=1000)
canvas.pack(pady=12, padx=10, fill=tk.BOTH, expand=True)
canvas.bind("<Double-Button-1>", on_select2)

update_listbox(None)
root.mainloop()
