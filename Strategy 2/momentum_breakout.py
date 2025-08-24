import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import seaborn as sns
import strategy_toolbox as st
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

class MomentumBreakout: 

    def __init__(self, dfs, time): 

        self.open = dfs['open'].resample(time).first()
        self.high = dfs['high'].resample(time).max()
        self.low = dfs['low'].resample(time).min()

    def get_avg_true_range(self, atr_window):

        high_low = self.high - self.low
        high_open = (self.high - self.open).abs()
        low_open = (self.low - self.open).abs()

        true_range_df = pd.concat([high_low, high_open, low_open], axis=0).groupby(level=0).max()
        avg_true_range_df = true_range_df.rolling(window = atr_window).mean()

        return avg_true_range_df

    def get_entry_exit(self, atr_window, atr_multiple, mtd_shift): 

        entry_line = self.high.cummax().shift(mtd_shift)
        avg_true_range_df = self.get_avg_true_range(atr_window)
        exit_line = entry_line - atr_multiple * avg_true_range_df

        return entry_line, exit_line

    def get_weight(self, dic_param): 

        atr_window = dic_param['atr_window']
        atr_multiple = dic_param['atr_multiple']
        mtd_shift = dic_param['mtd_shift']

        entry_line, exit_line = self.get_entry_exit(atr_window, atr_multiple, mtd_shift)

        entry_df = (self.open > entry_line).astype(int)
        exit_df = (self.open < exit_line).astype(int)

        weight_df = (entry_df - exit_df).replace(0, np.nan)
        weight_df = weight_df.ffill().fillna(0)
        weight_df = weight_df.replace(-1, 0)

        total_weight = weight_df.sum(axis=1)
        weight_df = weight_df.div(total_weight.where(total_weight != 0), axis = 0).fillna(0)

        return weight_df

    def plot_weight(self, weight_df): 

        _, ax = plt.subplots(figsize = (14, 7))

        weight_df.iloc[:, :-1].plot(ax=ax)

        ax.set_title('Portfolio Weights')
        ax.set_ylabel('Fraction of Budget in Each Coin')
        ax.set_xlabel('Time')

        ax.grid(True, alpha=0.6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.legend()

        plt.show()

    def plot_position(self, weight_df, symbols):
  
        _, axs = plt.subplots(4, 2, figsize=(18, 20))
        axs = axs.flatten() 

        for i, coin in enumerate(symbols):
            COIN = coin.upper()
            axs[i].plot(self.open.index, self.open[COIN], label=f"{COIN} Open Price (USD)", color='tab:blue')
            axs[i].fill_between(self.open.index, self.open[COIN].min(), self.open[COIN].max(),
                                where=weight_df[COIN] != 0, color='green', alpha=0.2, label=f'{COIN} Long Exposure')
            axs[i].set_ylabel(f'{COIN} Price (USD)')
            axs[i].grid(True, alpha=0.6)
            axs[i].xaxis.set_major_locator(MaxNLocator(nbins=4))
            axs[i].legend()

        plt.suptitle('Strategy Long Entry & Exit for Each Individual Coin', fontsize=15, y=0.9)  
        plt.show()

    def get_metric(self, metric): 

        metric_df = pd.DataFrame(list(metric.items()), columns=['metric', 'value'])
        metric_df['value'] = metric_df['value'].round(3)

        return metric_df

    def random_search(self, dfs):

        results = []
        n = 1000

        for i in range(n):
            atr_window = random.randint(5, 20)
            atr_multiple = random.choice([round(x * 0.1, 1) for x in range(10, 51)])
            mtd_shift = random.randint(1, 10)

            entry, exit = self.get_entry_exit(atr_window, atr_multiple, mtd_shift)
            weight_df = self.get_weight(entry, exit)
            metric,_ = st.eval_strat_multi(dfs, weight_df)
            metric['atr_window'] = atr_window
            metric['atr_multiple'] = atr_multiple
            metric['mtd_shift'] = mtd_shift

            results.append(metric)
        
        result_df = pd.DataFrame(results)
        result_df = result_df[['sharpe_fee', 'atr_window', 'atr_multiple', 'mtd_shift']]

        return result_df.sort_values(by= 'sharpe_fee', ascending=False).head(20)

    def sharpe_heatmap(self, dfs, atr_window_range, mtd_shift_range):

        atr_multiple = self.params['atr_multiple'] 

        heatmap_data = []

        for atr_window in atr_window_range:
            for mtd_shift in mtd_shift_range:

                entry, exit = self.get_entry_exit(atr_window, atr_multiple, mtd_shift)
                weight_df = self.get_weight(entry, exit)
                metrics, _ = st.eval_strat_multi(dfs, weight_df)

                sharpe = metrics['sharpe_fee']
                heatmap_data.append({'atr_window': atr_window, 'mtd_shift': mtd_shift, 'sharpe_ratio': sharpe})
        
        df_heatmap = pd.DataFrame(heatmap_data)
        heatmap_matrix = df_heatmap.pivot(index='atr_window', columns='mtd_shift', values='sharpe_ratio')

        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0, cbar_kws={'label': 'Sharpe Ratio'})
        plt.title('Sharpe Ratio Sensitivity to Parameters')
        plt.xlabel('Average True Range Window')
        plt.ylabel('Max-to-Date Shift')
        plt.tight_layout()
        plt.show()

    def sharpe_surface(self, dfs, atr_window_range, mtd_shift_range):

        atr_multiple = self.params['atr_multiple'] 

        heatmap_data = []

        for atr_window in atr_window_range:
            for mtd_shift in mtd_shift_range:
                entry, exit = self.get_entry_exit(atr_window, atr_multiple, mtd_shift)
                weight_df = self.get_weight(entry, exit)
                metrics, _ = st.eval_strat_multi(dfs, weight_df)

                sharpe = metrics['sharpe_fee']
                heatmap_data.append({'atr_window': atr_window, 'mtd_shift': mtd_shift, 'sharpe_ratio': sharpe})
        
        df_heatmap = pd.DataFrame(heatmap_data)

        X, Y = np.meshgrid(sorted(df_heatmap['atr_window'].unique()), 
                        sorted(df_heatmap['mtd_shift'].unique()))

        Z = df_heatmap.pivot(index='mtd_shift', columns='atr_window', values='sharpe_ratio').values

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, edgecolor='k', alpha=0.8)

        ax.set_title('Sharpe Ratio Sensitivity Surface')
        ax.set_xlabel('ATR Window')
        ax.set_ylabel('MTD Shift')
        ax.set_zlabel('Sharpe Ratio')

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Sharpe Ratio')

        plt.tight_layout()
        plt.show()

    def check_forward_looking(self):
      
        n = self.open.shape[0]
        index_list = self.open.index
        weight_df = self.strategy_breakout()
        weight_list = np.zeros(weight_df.shape)

        for k in tqdm(range(n)):
            index = index_list[k]
            df_open_tmp = self.open.loc[:index]
            df_high_tmp = self.high.loc[:index]
            df_low_tmp = self.low.loc[:index]
            weight_dfs_tmp = self.strategy_breakout(df_open_tmp, df_high_tmp, df_low_tmp, params)
            weight_list[k] = weight_dfs_tmp.values[-1]

        return (weight_df.values == weight_list).all()

if __name__ == '__main__':

    symbols = ['ADA', 'AVAX', 'BNB', 'BTC', 'DOGE', 'ETH', 'SOL', 'XRP']
    training_df_dic, validation_df_dic = data.get_data()
  
    dic_param = {
      'atr_window': 15,
      'atr_multiple': 1.2,
      'mtd_shift': 3
    }
  
    strat = MomentumBreakout(training_df_dic, '1T')
    weight_df = strat.get_weight(dic_param)
  
    metric, perf = st.strat_eval(training_df_dic, weight_df)
    indicator = st.get_strat_metrics(metric, 'strat')
    st.plot_equity_curve(training_df_dic, perf)
    st.plot_drawdown(perf)
    plt.show()
