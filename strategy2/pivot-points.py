
  dic_param_2 = {
    'ADA': {'shift_signal': 1, 'stop_loss': 0.02, 'take_profit': 0.05},
    'AVAX': {'shift_signal': 2, 'stop_loss': 0.03, 'take_profit': 0.06},
    'BNB': {'shift_signal': 1, 'stop_loss': 0.01, 'take_profit': 0.04},
    'BTC': {'shift_signal': 3, 'stop_loss': 0.015, 'take_profit': 0.07},
    'DOGE': {'shift_signal': 2, 'stop_loss': 0.025, 'take_profit': 0.05},
    'ETH': {'shift_signal': 2, 'stop_loss': 0.02, 'take_profit': 0.06},
    'SOL': {'shift_signal': 1, 'stop_loss': 0.03, 'take_profit': 0.08},
    'XRP': {'shift_signal': 2, 'stop_loss': 0.02, 'take_profit': 0.05}
}

# Parameter Optimization

def random_search():

results = []
n = 1000

for i in range(n):

shift_signal = random.randint(1, 100)
take_profit = random.choice(np.round(np.linspace(0.01, 0.3, 50), 3).tolist())
stop_loss = random.choice(np.round(np.linspace(0.01, 0.3, 50), 3).tolist())

param = {'shift_signal': shift_signal,
'take_profit': take_profit,
'stop_loss': stop_loss
}

weight_df = main(training_df_dic, param)
metric, _ = u.strat_eval(training_df_dic, weight_df)
indicator = u.get_strat_metrics(metric, 'strat')

indicator['shift_signal'] = shift_signal
indicator['take_profit'] = take_profit
indicator['stop_loss'] = stop_loss

results.append(indicator)
result_df = pd.DataFrame(results)
result_df = result_df[['sharpe_fees', 'shift_signal', 'take_profit', 'stop_loss']]

return result_df.sort_values(by= 'sharpe_fees', ascending=False).head(20)

random_search()

def sharpe_heatmap(shift_signal_range, take_profit_range):

heatmap_data = []

for shift_signal in shift_signal_range:
for take_profit in take_profit_range:

param = {'shift_signal': shift_signal,
'take_profit': take_profit,
'stop_loss': 0.02
}

weight_df = main(training_df_dic, param)
metric, _ = u.strat_eval(training_df_dic, weight_df)
indicator = u.get_strat_metrics(metric, 'strat')

indicator['shift_signal'] = shift_signal
indicator['take_profit'] = take_profit

sharpe = indicator['sharpe_fees']
heatmap_data.append({'shift_signal': shift_signal, 'take_profit': take_profit, 'sharpe_ratio': sharpe})
df_heatmap = pd.DataFrame(heatmap_data)
heatmap_matrix = df_heatmap.pivot(index='shift_signal', columns='take_profit', values='sharpe_ratio')

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0, cbar_kws={'label': 'Sharpe Ratio'})
plt.title('Sharpe Ratio Sensitivity to Parameters')
plt.xlabel('Shift Signal')
plt.ylabel('Take Profit')
plt.tight_layout()
plt.show()

sharpe_heatmap(range(60, 80), np.round(np.linspace(0.01, 0.2, 20), 3).tolist())




def main_multi_support_resistance(dfs, param_dict):
    open_df = dfs['open']
    high_df = dfs['high']
    low_df = dfs['low']
    close_df = dfs['close']
    tickers = open_df.columns

    signal_df = pd.DataFrame(index=open_df.index, columns=tickers)

    for ticker in tickers:
        param = param_dict[ticker]

        center = (high_df[ticker] + low_df[ticker] + close_df[ticker]) / 3
        support = (2 * center - high_df[ticker]).shift(param['shift_signal'])
        resistance = (2 * center - low_df[ticker]).shift(param['shift_signal'])

        entry = (open_df[ticker] > resistance).astype(int)

        entry_price = (entry * open_df[ticker]).replace(0, np.nan).ffill().fillna(0)
        stop = (open_df[ticker] < entry_price * (1 - param['stop_loss'])).astype(int)
        take_profit = (open_df[ticker] > entry_price * (1 + param['take_profit'])).astype(int)
        exit_ = np.sign(take_profit + stop)

        signal = (entry - exit_).replace(0, np.nan).ffill().replace(-1, 0).fillna(0)
        signal_df[ticker] = signal

    weight_df = signal_df.div(signal_df.sum(axis=1), axis=0).shift().fillna(0)
    return weight_df
