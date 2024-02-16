import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.stattools import coint

class PairsHunter:
    def __init__(self, data, p_value_threshold = 0.05):
        self.data = data 
        self.corr_matrix = self.data.corr()
        
        self.p_value_threshold = p_value_threshold
        self.pairs_trading_ideas = self.find_cointegrated_pairs()  
    
    def plot_heatmap(self):
        
        data = self.data
        corr_matrix = self.corr_matrix

        columns = data.columns

        heatmap_trace = go.Heatmap(z=corr_matrix,
                                x=columns,
                                y=columns,
                                colorscale='geyser',
                                text=corr_matrix)
        layout = go.Layout(title='Correlation Matrix',
                        xaxis=dict(title='Asset'),
                        yaxis=dict(title='Asset'))
        fig = go.Figure(data=[heatmap_trace], layout=layout)

        return fig

    def plot_heatmap_sns(self):
        corr_matrix = self.corr_matrix        
        plt.figure(figsize = (15, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths = 0.75)
        plt.show()
    
    def find_cointegrated_pairs(self):
        pairs_list = set()
        for asset1 in self.data.columns:
            for asset2 in self.data.columns:
                if asset1 != asset2:
                    p_value = self.test_cointegration(asset1, asset2)
                    if p_value < self.p_value_threshold:
                        pairs_list.add(tuple(sorted([asset1, asset2])))

        pairs_list = list(pairs_list)
        pairs_list = [list(pairs) for pairs in pairs_list]
        return pairs_list
                   
    
    def test_cointegration(self, asset1, asset2):
        prices1 = self.data[asset1].values
        prices2 = self.data[asset2].values
        _, p_value, _ = coint(prices1, prices2)
        return p_value

    
@st.cache_resource        
class PairsTrading:
    def __init__(self, df, k, window_size):
        self.k = k
        self.window_size = window_size
        self.df = self.set_limits(df)
        self.signals_df = self.pairs_trading_strategy_signals()
        self.backtest_df = self.pairs_trading_backtest()
        self.measures = self.strategy_measures()
        # print(self.signals_df.query('Signal == "Sell"'))
        
    def set_limits(self, df):
        ativo_1 = df.columns[0]
        ativo_2 = df.columns[1]
        df['Spread'] = df[ativo_1] - df[ativo_2]

        window_size = self.window_size
        k = self.k

        df['Spread_Mean'] = df['Spread'].rolling(window=window_size).mean()
        df['Spread_Std'] = df['Spread'].rolling(window=window_size).std()

        df['Upper_Limit'] = df['Spread_Mean'] + k * df['Spread_Std']
        df['Lower_Limit'] = df['Spread_Mean'] - k * df['Spread_Std']
        
        return df


    def spread_chart(self):
        # Retrieve instance information
        df = self.df
        plt.figure(figsize=(12, 8))
        sns.lineplot(x=df.index, y=df['Spread'], label='Spread', color='blue')
        sns.lineplot(x=df.index, y=df['Spread_Mean'], label='Mean', color='orange')
        sns.lineplot(x=df.index, y=df['Upper_Limit'], label='Upper Limit', linestyle='--', color='red')
        sns.lineplot(x=df.index, y=df['Lower_Limit'], label='Lower Limit', linestyle='--', color='green')

        plt.title('Pairs Trading - Spread e Dynamics Limits')
        plt.xlabel('Data')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
    
    def spread_chart_plotly(self):
        # Retrieve instance information
        df = self.df

        # Create traces for Spread, Mean, Upper Limit, and Lower Limit
        spread_trace = go.Scatter(x=df.index, y=df['Spread'], mode='lines', name='Spread', line=dict(color='blue'))
        mean_trace = go.Scatter(x=df.index, y=df['Spread_Mean'], mode='lines', name='Mean', line=dict(color='orange'))
        upper_limit_trace = go.Scatter(x=df.index, y=df['Upper_Limit'], mode='lines', name='Upper Limit', line=dict(color='red', dash='dot'))
        lower_limit_trace = go.Scatter(x=df.index, y=df['Lower_Limit'], mode='lines', name='Lower Limit', line=dict(color='green', dash='dot'))

        # Create subplot figure
        fig = make_subplots(rows=1, cols=1, subplot_titles=('Pairs Trading - Spread and Dynamic Limits'))

        # Add traces to subplot
        fig.add_trace(spread_trace)
        fig.add_trace(mean_trace)
        fig.add_trace(upper_limit_trace)
        fig.add_trace(lower_limit_trace)

        # Update layout
        fig.update_layout(title_text='Pairs Trading - Spread and Dynamic Limits', xaxis_title='Date', yaxis_title='Value', showlegend=True)

        # Return the figure
        return fig

    
    def pairs_trading_strategy_signals(self):        
        # Retrieve instance information
        df = self.df
        ativo_1 = df.columns[0]
        ativo_2 = df.columns[1]
        signals = []

        position_open = False  # Flag to track if a position is open

        for i in range(1, len(df)):
            spread = df[ativo_1][i] - df[ativo_2][i]
            upper_limit = df['Upper_Limit'].iloc[i]
            lower_limit = df['Lower_Limit'].iloc[i]

            previous_spread = df[ativo_1].iloc[i - 1] - df[ativo_2].iloc[i - 1]

            # Signals for Ativo_1
            if previous_spread < lower_limit and spread >= lower_limit:
                signals.append(('Buy', df.index[i], ativo_1, df[ativo_1].iloc[i]))
                position_open = True

            elif previous_spread > upper_limit and spread <= upper_limit:
                signals.append(('Sell', df.index[i], ativo_1, df[ativo_1].iloc[i]))
                position_open = True

            elif position_open and ((previous_spread < upper_limit and previous_spread > lower_limit) and (spread >= upper_limit or spread <= lower_limit)):
                signals.append(('Close', df.index[i], ativo_1, df[ativo_1].iloc[i]))
                position_open = False  # Reset flag

            # Signals for Ativo_2 (assuming opposite positions for Ativo_2)
            if previous_spread < lower_limit and spread >= lower_limit:
                signals.append(('Sell', df.index[i], ativo_2, df[ativo_2].iloc[i]))
                position_open = True

            elif previous_spread > upper_limit and spread <= upper_limit:
                signals.append(('Buy', df.index[i], ativo_2, df[ativo_2].iloc[i]))
                position_open = True

            elif position_open and ((previous_spread < upper_limit and previous_spread > lower_limit) and (spread >= upper_limit or spread <= lower_limit)):
                signals.append(('Close', df.index[i], ativo_2, df[ativo_2].iloc[i]))
                position_open = False  # Reset flag

        signals_df = pd.DataFrame(signals, columns=['Signal', 'Date', 'Ativo', 'Price'])

        return signals_df 
    
    
    
    def pairs_trading_backtest(self):
        # Retrieve instance information
        signals_df = self.signals_df
        df = self.df
        ativo_1 = df.columns[0]
        ativo_2 = df.columns[1]

        backtest_df = pd.DataFrame(index=df.index)
        backtest_df['Spread'] = df[ativo_1] - df[ativo_2]
        backtest_df[ativo_1] =  df[ativo_1]
        backtest_df[ativo_2] =  df[ativo_2]

        # Initialize variables
        position = 0  # Initialize position
        entry_price = 0  # Initialize entry price
        cumulative_returns = [0]  # Initialize cumulative returns list

        # Simulate the trading strategy based on signals
        for i in range(1, len(df)):
            spread = backtest_df['Spread'].iloc[i]
            signal = signals_df[signals_df['Date'] == backtest_df.index[i]]
            trade_returns = 0
            if signal.empty:
                cumulative_returns.append(cumulative_returns[-1] + trade_returns)
                continue

            else:
                signal_type = signal['Signal'].iloc[0]

                if signal_type == 'Buy' and position == 0:
                    position = 1
                    entry_price = spread
    #                 print(f"Buy Signal: {backtest_df.index[i]}, Entry Price: {entry_price}")
                elif signal_type == 'Sell' and position == 0:
                    position = -1
                    entry_price = spread
    #                 print(f"Sell Signal: {backtest_df.index[i]}, Entry Price: {entry_price}")
                elif signal_type == 'Close' and position != 0:
                    if position == 1:
                        exit_price = spread
                        trade_returns = exit_price - entry_price
                        entry_price = 0  # Reset entry price after closing a position
                        # print(f"Close Signal for {ativo_1}: Exit Price: {exit_price}, Trade Returns: {trade_returns}, Cumulative Returns: {cumulative_returns[-1]}")
                    elif position == -1:
                        exit_price = spread
                        trade_returns = entry_price - exit_price  # Reverse trade returns calculation for short position
                        entry_price = 0  # Reset entry price after closing a position
                        # print(f"Close Signal for {ativo_2}: Exit Price: {exit_price}, Trade Returns: {trade_returns}, Cumulative Returns: {cumulative_returns[-1]}")
                    position = 0  # Reset position after closing a position

    #                 print(f"Close Signal: {backtest_df.index[i]}, Exit Price: {exit_price}, Trade Returns: {trade_returns}, Cumulative Returns: {cumulative_returns[-1]}")

                cumulative_returns.append(cumulative_returns[-1] + trade_returns)

        # Add cumulative returns to the DataFrame
        backtest_df['Cumulative_Returns'] = cumulative_returns

        return backtest_df


    def plot_backtest(self):
        # Retrieve instance information
        backtest_df = self.backtest_df
        signals_df = self.signals_df
        
        # Plot the spread and cumulative returns
        plt.figure(figsize=(12, 8))

        # Plot Spread
        plt.subplot(2, 1, 1)
        plt.plot(backtest_df[backtest_df.columns[:-1]], label=backtest_df.columns[:-1])

        # Mark Buy signals
        plt.scatter(signals_df.loc[signals_df['Signal'] == 'Buy', 'Date'],
                    signals_df.loc[signals_df['Signal'] == 'Buy', 'Price'],
                    marker='^', color='green', label='Buy Signal')

        # Mark Sell signals
        plt.scatter(signals_df.loc[signals_df['Signal'] == 'Sell', 'Date'],
                    signals_df.loc[signals_df['Signal'] == 'Sell', 'Price'],
                    marker='v', color='red', label='Sell Signal')

        # Mark Close signals
        plt.scatter(signals_df.loc[signals_df['Signal'] == 'Close', 'Date'],
                    signals_df.loc[signals_df['Signal'] == 'Close', 'Price'],
                    marker='x', color='black', label='Close Signal')

        plt.title('Pairs Trading Backtest - Spread')
        plt.xlabel('Date')
        plt.ylabel('Spread')
        plt.legend()

        # Plot Cumulative Returns
        plt.subplot(2, 1, 2)
        plt.plot(backtest_df['Cumulative_Returns'], label='Cumulative Returns', color='green')
        plt.title('Pairs Trading Backtest - Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_backtest_plotly(self):
        # Retrieve instance information
        backtest_df = self.backtest_df
        signals_df = self.signals_df
        
        # Create traces for Ativo_1, Ativo_2, Spread, and Cumulative Returns
        ativo_1_trace = go.Scatter(x=backtest_df.index, y=backtest_df[backtest_df.columns[1]], mode='lines', name=backtest_df.columns[1])
        ativo_2_trace = go.Scatter(x=backtest_df.index, y=backtest_df[backtest_df.columns[2]], mode='lines', name=backtest_df.columns[2])
        spread_trace = go.Scatter(x=backtest_df.index, y=backtest_df[backtest_df.columns[0]], mode='lines', name='Spread')
        cumulative_returns_trace = go.Scatter(x=backtest_df.index, y=backtest_df['Cumulative_Returns'], mode='lines', name='Cumulative Returns')

        # Create traces for Buy and Sell signals
        buy_signals = signals_df[signals_df['Signal'] == 'Buy']
        sell_signals = signals_df[signals_df['Signal'] == 'Sell']

        buy_trace = go.Scatter(x=buy_signals['Date'], y=buy_signals['Price'], mode='markers', marker=dict(symbol='triangle-up', size=8, color='green'), name='Buy Signal')
        sell_trace = go.Scatter(x=sell_signals['Date'], y=sell_signals['Price'], mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'), name='Sell Signal')

       # Retrieve datetime index for Close signals
        close_signals = signals_df[signals_df['Signal'] == 'Close']
        close_dates = close_signals['Date']

        # Project Close signals onto the Spread line
        close_indices = []
        for date in close_dates:
            idx = backtest_df.index.get_loc(date)
            close_indices.append(idx)

        close_trace = go.Scatter(x=backtest_df.index[close_indices], y=backtest_df.iloc[close_indices][backtest_df.columns[0]], mode='markers', marker=dict(symbol='hexagon2', size=10, color='black'), name='Close Signal')



        # Create subplot figure
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=('Pairs Trading Backtest - Prices and Spread', 'Pairs Trading Backtest - Cumulative Returns'))

        # Add traces to subplot
        fig.add_trace(ativo_1_trace, row=1, col=1)
        fig.add_trace(ativo_2_trace, row=1, col=1)
        fig.add_trace(spread_trace, row=1, col=1)
        fig.add_trace(buy_trace, row=1, col=1)
        fig.add_trace(sell_trace, row=1, col=1)
        fig.add_trace(close_trace, row=1, col=1)

        fig.add_trace(cumulative_returns_trace, row=2, col=1)

        # Update layout
        fig.update_layout(title_text="Pairs Trading Backtest", showlegend=True)

        return fig

    
    def strategy_measures(self):
        cumulative_returns = self.backtest_df['Cumulative_Returns']
        total_trading_days = len(cumulative_returns.index)

        # Calculate annualized returns
        annualized_returns = cumulative_returns.iloc[-1] / total_trading_days

        # Calculate annualized volatility
        daily_volatility = cumulative_returns.std()
        annualized_volatility = daily_volatility * np.sqrt(total_trading_days)

        # Calculate Sharpe ratio
        sharpe_ratio = annualized_returns / daily_volatility * np.sqrt(total_trading_days)

        # Calculate max drawdown
        max_drawdown = -cumulative_returns.min()

        # Calculate Calmar ratio
        calmar_ratio = annualized_returns / max_drawdown

        # Create measures dictionary
        measures_dict = {
            'Cumulative Returns': cumulative_returns.iloc[-1],
            'Annual Returns': annualized_returns,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Annual Vol': annualized_volatility,
            'Calmar Ratio': calmar_ratio
        }

        
        return measures_dict