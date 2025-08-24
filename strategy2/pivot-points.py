import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import random
import seaborn as sns
import strategy_toolbox as st

class PivotPoints:

    def __init__(self, dfs: dict):
        required = {'open', 'high', 'low', 'close'}
        missing = required - set(dfs.keys())
        if missing:
            raise ValueError(f"dfs missing required keys: {missing}")

        self.open_df = dfs['open'].copy()
        self.high_df = dfs['high'].copy()
        self.low_df = dfs['low'].copy()
        self.close_df = dfs['close'].copy()

        for key in ['high', 'low', 'close']:
            if not self.open_df.index.equals(dfs[key].index):
                raise ValueError(f"Index mismatch between open and {key}.")
            if list(self.open_df.columns) != list(dfs[key].columns):
                raise ValueError(f"Column mismatch between open and {key}.")

        self.tickers = list(self.open_df.columns)
        self.index = self.open_df.index
        self.dfs = dfs 

    @staticmethod
    def _param_for_ticker(dic_param, ticker):

        if dic_param is None:
            raise ValueError("dic_param must be provided.")
        if isinstance(dic_param, dict) and ticker in dic_param and isinstance(dic_param[ticker], dict):
            p = dic_param[ticker]
        else:
            p = dic_param

        p = {
            'shift_signal': int(p.get('shift_signal', 1)),
            'take_profit': float(p.get('take_profit', 0.10)),
            'stop_loss'  : float(p.get('stop_loss', 0.03)),
        }
        if p['shift_signal'] < 0:
            raise ValueError("shift_signal must be >= 0")
        if p['take_profit'] <= 0 or p['stop_loss'] <= 0:
            raise ValueError("take_profit and stop_loss must be > 0")
        return p

    def get_weight(self, dic_param: dict) -> pd.DataFrame:

        signal_df = pd.DataFrame(0, index=self.index, columns=self.tickers, dtype=float)

        for ticker in self.tickers:
            p = self._param_for_ticker(dic_param, ticker)

            high = self.high_df[ticker]
            low = self.low_df[ticker]
            close = self.close_df[ticker]
            open_ = self.open_df[ticker]

            center = (high + low + close) / 3.0
            support = (2 * center - high).shift(p['shift_signal'])
            resistance = (2 * center - low).shift(p['shift_signal'])
          
            entry = (open_ > resistance).astype(int)
            entry_price = (entry * open_).replace(0, np.nan).ffill().fillna(0.0)

            stop = (open_ < entry_price * (1.0 - p['stop_loss'])).astype(int)
            take_profit = (open_ > entry_price * (1.0 + p['take_profit'])).astype(int)
            exit_ = np.sign(take_profit + stop) 

            raw = (entry - exit_).replace(0, np.nan)
            pos = raw.ffill().fillna(0.0).replace(-1, 0.0)

            signal_df[ticker] = pos

        row_sum = signal_df.sum(axis=1)
        weight_df = signal_df.div(row_sum.replace(0, np.nan), axis=0).shift(1).fillna(0.0)

        return weight_df

    def sharpe_heatmap(self,
                       shift_signal_range,
                       take_profit_range,
                       stop_loss=0.02,
                       eval_module=None):

        if eval_module is None:
            import strategy_toolbox as st
            eval_module = st

        heatmap_data = []

        shift_signal_range = list(shift_signal_range)
        take_profit_range = list(take_profit_range)

        for shift_signal in shift_signal_range:
            for take_profit in take_profit_range:
                param = {
                    'shift_signal': int(shift_signal),
                    'take_profit': float(take_profit),
                    'stop_loss': float(stop_loss)
                }
                weight_df = self.get_weight(param)
                metric, _ = eval_module.strat_eval(self.dfs, weight_df)
                indicator = eval_module.get_strat_metrics(metric, 'strat')

                sharpe = indicator.get('sharpe_fees', np.nan)
                heatmap_data.append({
                    'shift_signal': shift_signal,
                    'take_profit': take_profit,
                    'sharpe_ratio': sharpe
                })

        df_heatmap = pd.DataFrame(heatmap_data)
        heatmap_matrix = df_heatmap.pivot(index='shift_signal',
                                          columns='take_profit',
                                          values='sharpe_ratio')

        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_matrix, annot=True, fmt=".2f",
                    cmap='RdBu_r', center=0,
                    cbar_kws={'label': 'Sharpe Ratio'})
        plt.title('Sharpe Ratio Sensitivity to Parameters')
        plt.xlabel('Take Profit')
        plt.ylabel('Shift Signal')
        plt.tight_layout()
        plt.show()

        return df_heatmap, heatmap_matrix

    def random_search(self,
                      n: int = 1000,
                      shift_min: int = 1,
                      shift_max: int = 100,
                      tp_grid=None,
                      sl_grid=None,
                      eval_module=None,
                      top_k: int = 20,
                      seed: int = None) -> pd.DataFrame:

        if eval_module is None:
            import strategy_toolbox as st
            eval_module = st

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if tp_grid is None:
            tp_grid = np.round(np.linspace(0.01, 0.30, 50), 3).tolist()
        if sl_grid is None:
            sl_grid = np.round(np.linspace(0.01, 0.30, 50), 3).tolist()

        results = []

        for _ in range(int(n)):
            shift_signal = random.randint(int(shift_min), int(shift_max))
            take_profit = float(random.choice(tp_grid))
            stop_loss = float(random.choice(sl_grid))

            param = {
                'shift_signal': shift_signal,
                'take_profit': take_profit,
                'stop_loss': stop_loss
            }

            weight_df = self.get_weight(param)
            metric, _ = eval_module.strat_eval(self.dfs, weight_df)
            indicator = eval_module.get_strat_metrics(metric, 'strat')

            row = {
                'sharpe_fees': indicator.get('sharpe_fees', np.nan),
                'shift_signal': shift_signal,
                'take_profit': take_profit,
                'stop_loss': stop_loss
            }
            results.append(row)

        result_df = pd.DataFrame(results)
        result_df = result_df[['sharpe_fees', 'shift_signal', 'take_profit', 'stop_loss']]

        return result_df.sort_values(by='sharpe_fees', ascending=False).head(top_k)

if __name__ == '__main__':
  
    symbols = ['ADA', 'AVAX', 'BNB', 'BTC', 'DOGE', 'ETH', 'SOL', 'XRP']

    # Download data using data.py and define parameters
    training_df_dic, validation_df_dic = data.get_data()
  
    dic_param = {
      'ADA': {'shift_signal': 1, 'stop_loss': 0.02, 'take_profit': 0.05},
      'AVAX': {'shift_signal': 2, 'stop_loss': 0.03, 'take_profit': 0.06},
      'BNB': {'shift_signal': 1, 'stop_loss': 0.01, 'take_profit': 0.04},
      'BTC': {'shift_signal': 3, 'stop_loss': 0.015, 'take_profit': 0.07},
      'DOGE': {'shift_signal': 2, 'stop_loss': 0.025, 'take_profit': 0.05},
      'ETH': {'shift_signal': 2, 'stop_loss': 0.02, 'take_profit': 0.06},
      'SOL': {'shift_signal': 1, 'stop_loss': 0.03, 'take_profit': 0.08},
      'XRP': {'shift_signal': 2, 'stop_loss': 0.02, 'take_profit': 0.05}
    }
  
    # Get performance metrics using strategy_toolbox.py
    strat = PivotPoints(training_df_dic, '1T')
    weight_df = strat.get_weight(dic_param)
  
    metric, perf = st.strat_eval(training_df_dic, weight_df)
    indicator = st.get_strat_metrics(metric, 'strat')
    st.plot_equity_curve(training_df_dic, perf)
    st.plot_drawdown(perf)
    plt.show()

    # Optimize parameters using heatmap and random search 
    sharpe_heatmap(range(60, 80), np.round(np.linspace(0.01, 0.2, 20), 3).tolist())
    random_search()



    
