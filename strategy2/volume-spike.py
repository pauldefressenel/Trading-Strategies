
param = {'window': 30,
'shift': 4,
'scalar': 8,
'take_profit': 0.10,
'stop_loss': 0.05
}

def main(df_dic, param):

window = param['window']
shift = param['shift']
scalar = param['scalar']
take_profit = param['take_profit']
stop_loss = param['stop_loss']

open_df, high_df, volume_df = df_dic['open'], df_dic['high'], df_dic['volume']

max_high_df = high_df.rolling(window = window, min_periods = window).max()
mean_volume_df = volume_df.rolling(window = window, min_periods = window).mean()

moment_df = (high_df > max_high_df.shift(shift)).astype(int)
volume_spike_df = (volume_df > scalar * mean_volume_df.shift(shift)).astype(int)
entry_df = moment_df * volume_spike_df

entry_price_df = (entry_df.shift() * open_df).replace(0, np.nan).ffill().fillna(0)
stop_df = (open_df < entry_price_df * (1 - stop_loss)).astype(int)
take_profit_df = (open_df > entry_price_df * (1 + take_profit)).astype(int)
exit_df = np.sign(take_profit_df + stop_df)

signal_df = (entry_df - exit_df).replace(0, np.nan).ffill().replace(-1, 0).fillna(0)
weight_df = (signal_df / signal_df.shape[1]).shift().fillna(0)

return weight_df

def random_search():

results = []
n = 1000

for i in range(n):

window = random.randint(1, 100)
shift = random.randint(1, 100)
scalar = random.randint(1, 20)

param = {'window': window,
'shift': shift,
'scalar': scalar,
'take_profit': 0.10,
'stop_loss': 0.05
}
weight_df = main(training_df_dic, param)

metric, _ = u.strat_eval(training_df_dic, weight_df)
indicator = u.get_strat_metrics(metric, 'strat')

indicator['window'] = window
indicator['shift'] = shift
indicator['scalar'] = scalar

results.append(indicator)
result_df = pd.DataFrame(results)
result_df = result_df[['sharpe_fees', 'window', 'shift', 'scalar']]

return result_df.sort_values(by = 'sharpe_fees', ascending=False).head(20)

random_search()

def sharpe_heatmap(window_vol_range, window_high_range):

heatmap_data = []
return_df = training_df_dic['return']
window = 25

for window_vol in window_vol_range:
for window_high in window_high_range:

param = {'window': window,
'window_vol': window_vol,
'window_high': window_high
}

weight_df = main(training_df_dic, param)
metric, _ = u.strat_eval_hourly(return_df, weight_df)
indicator = u.get_strat_metrics(metric)

indicator['window'] = window
indicator['window_vol'] = window_vol
indicator['window_high'] = window_high

sharpe = indicator['sharpe_fees']
heatmap_data.append({'window_vol': window_vol, 'window_high': window_high, 'sharpe_ratio': sharpe})
df_heatmap = pd.DataFrame(heatmap_data)
heatmap_matrix = df_heatmap.pivot(index='window_vol', columns='window_high', values='sharpe_ratio')

plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_matrix, annot=True, fmt=".2f", cmap='RdBu_r', center=0, cbar_kws={'label': 'Sharpe Ratio'})
plt.title('Sharpe Ratio Sensitivity to Parameters')
plt.xlabel('Average True Range Window')
plt.ylabel('Max-to-Date Shift')
plt.tight_layout()
plt.show()

sharpe_heatmap(np.round(np.linspace(1, 10, 20), 1).tolist(), range(2, 20))
