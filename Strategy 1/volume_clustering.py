import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator



class VolumeClustering:
    
    def __init__(self, dfs: dict, params: dict, coin: str):
        
        c = coin.upper()
        
        self.df = pd.concat({
            "open": dfs["open"][c],
            "high": dfs["high"][c],
            "low": dfs["low"][c],
            "close": dfs["close"][c],
            "volume": dfs["volume"][c],
            "returns": dfs["return"][c]
        }, axis=1).astype(float).sort_index()

        self.df.index = pd.to_datetime(self.df.index)
        
        self.open = self.df["open"]
        self.high = self.df["high"]
        self.low = self.df["low"]
        self.close = self.df["close"]
        self.volume = self.df["volume"]
        self.returns = self.df["returns"]

        self.params = params
    
    def get_hist_window(self, window, n_bins = None):

        half_life = self.params["half_life"]

        if n_bins == None: 
            n_bins = 10

        datetime_ref = pd.to_datetime(window.index[-1])
        datetime_index = pd.to_datetime(window.index)
        datetime_diff = (datetime_ref - datetime_index).total_seconds()
        window_weight = np.exp(-np.log(2) * datetime_diff / pd.Timedelta(half_life).total_seconds())

        edges = np.linspace(window['open'].min(), window['open'].max(), n_bins + 1)
        bins = pd.cut(window["open"], edges)

        histogram = (window["volume"] * window_weight).groupby(bins, observed=False).sum()

        return histogram

    def get_sup_res_window(self, histogram, spot):

        below_zone = histogram[[(iv.mid < spot) and (spot not in iv) for iv in histogram.index]]
        above_zone = histogram[[(iv.mid > spot) and (spot not in iv) for iv in histogram.index]]

        support = below_zone.idxmax().mid if not below_zone.empty else np.nan
        resistance = above_zone.idxmax().mid if not above_zone.empty else np.nan

        return support, resistance

    def get_sup_res_df(self, lookback = None): 

        if lookback == None: 
            lookback = 2

        sup_res_df = pd.DataFrame(index = self.df.index)
        sup_res_df[['support', 'spot', 'resistance']] = np.nan

        for i in range(lookback, len(self.df)):

            window = self.df.iloc[i - lookback: i]
            spot = self.df.loc[window.index[-1] + pd.Timedelta("1D"), 'open']
            histogram = self.get_hist_window(window)
            support, resistance = self.get_sup_res_window(histogram, spot)

            sup_res_df.loc[self.df.index[i], ['support', 'spot', 
                                              'resistance']] = support, spot, resistance
        
        return sup_res_df 

    def plot_sup_res(self, sup_res_df): 
        
        _, ax = plt.subplots(figsize = (14, 7))
            
        sup_res_df.plot(ax=ax)
        ax.grid(True, alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.legend()

        plt.suptitle('Spot & Support & Resistance')  
        plt.tight_layout()
        plt.show()

    def get_avg_true_range(self):

        true_range_df = self.df['high'] - self.df['low']
        atr_df = true_range_df.rolling(window = 5).mean()
        
        return atr_df

    def get_weight(self, sup_res_df, atr_df, shift = 2): 

        sup_res_df['avg_true_range'] = atr_df
        sup_res_df[['support', 'resistance', 'avg_true_range']] = sup_res_df[['support', 'resistance', 'avg_true_range']].shift(shift)
        sup_res_df['entry'] = (sup_res_df['spot'] > (sup_res_df['resistance'] + sup_res_df['avg_true_range'])).astype(int)

        sup_res_df['entry_price'] = (sup_res_df['entry'] * sup_res_df['spot']).replace(0, np.nan)
        sup_res_df['entry_price'] = sup_res_df['entry_price'].ffill()

        sup_res_df['exit'] = (sup_res_df['spot'] < (sup_res_df['entry_price'] - sup_res_df['avg_true_range'])).astype(int)

        weight_df = (sup_res_df['entry'] - sup_res_df['exit']).replace(0, np.nan)
        weight_df = weight_df.ffill().fillna(0)
        weight_df = weight_df.replace(-1, 0)

        return weight_df

    def plot_weight(self, weight_df): 

        _, ax = plt.subplots(figsize = (14, 7))

        weight_df.plot(ax=ax)

        ax.set_title('Weight over Time')
        ax.set_ylabel('Weight')
        ax.set_xlabel('Time')

        ax.grid(True, alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.legend()

        plt.show()

    def plot_position(self, weight_df):
  
        _, ax = plt.subplots(figsize=(14, 7))

        ax.plot(self.open.index, self.open, label=f"Open Price (USD)", color='tab:blue')
        ax.fill_between(self.open.index, self.open.min(), self.open.max(),
                            where=weight_df != 0, color='green', alpha=0.2, label='Long Exposure')
        ax.set_ylabel('Price (USD)')
        ax.grid(True, alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.legend()

        plt.suptitle('Strategy Long Entry & Exit', fontsize=15, y=0.9)  
        plt.show()

    def plot_histogram(self, histogram, spot):
        
        bin_centers = [interval.mid for interval in histogram.index]
        bar_width = histogram.index[0].right - histogram.index[0].left

        plt.bar(bin_centers, histogram.values, width=bar_width, align='center', edgecolor='black')
        plt.xlim(0.97 * spot, 1.03 * spot)
        plt.axvline(x=spot, color='red', linewidth=2, label="BTC Open, 2022-12-17")

        plt.xlabel("BTC Price Bins over 45-day lookback")
        plt.ylabel("Cumulative Volume Traded")
        plt.title("Histogram of Cumulative Volume Traded, weighted with " +
        "Exponential Decay over 45-days loockack on BTC", fontsize=14)

        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

    symbols = ['ADA', 'AVAX', 'BNB', 'BTC', 'DOGE', 'ETH', 'SOL', 'XRP']
    training_df_dic, validation_df_dic = data.get_data()
  
    dic_param = {
        'half_life': '2D', 
        'n_bins': 5, 
        'band_pct': 0.05, 
        'lookback': 10, 
        'fee': 0.002
    }

    # Get performance metrics using strategy-toolbox.py
    strat = VolumeClustering(training_df_dic, '1T')
    weight_df = strat.get_weight(dic_param)
  
    metric, perf = st.strat_eval(training_df_dic, weight_df)
    indicator = st.get_strat_metrics(metric, 'strat')
    st.plot_equity_curve(training_df_dic, perf)
    st.plot_drawdown(perf)
    plt.show()
