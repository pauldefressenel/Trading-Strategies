import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def calculate_fees(df_weights, params):
    fees = params['fees']
    df_fees = (df_weights.diff().abs() * fees)
    df_fees = df_fees.sum(axis=1).fillna(0)
    return df_fees

def strat_eval(returns, weights, params):
    fees = params['fees']
    df_fees = (weights.diff().abs() * fees)
    df_fees = df_fees.sum(axis=1).fillna(0)

    df_return_strat = (weights * returns).sum(axis=1).fillna(0)
    df_return_fees_strat = df_return_strat - df_fees

    PNL = (df_return_strat + 1).cumprod()
    PNL_fees = (df_return_fees_strat + 1).cumprod()

    drawdowns = (PNL_fees / PNL_fees.cummax()) - 1
    max_drawdown = drawdowns.min(skipna=True) * 100
    mean_ret = df_return_fees_strat.mean()
    std_ret = df_return_fees_strat.std()
    if std_ret == 0 or np.isnan(std_ret):
        sharpe = np.nan
    else:
        sharpe = mean_ret / std_ret * np.sqrt(365.25)

    annualized_return = PNL_fees.iloc[-1] ** (365.25 / len(df_return_fees_strat)) - 1 if PNL_fees.iloc[-1] > 0 else 0
    annualized_volatility = df_return_fees_strat.std() * np.sqrt(365.25)
    downside_std = df_return_fees_strat[df_return_fees_strat < 0].std()
    sortino_ratio = df_return_fees_strat.mean() / downside_std * np.sqrt(365.25)
    winning_rate = (df_return_fees_strat > 0).mean() * 100

    strategy_changed = (weights.diff().abs().sum(axis=1) != 0)
    total_trades = strategy_changed.astype(int).sum()
    strategy_ids = (~strategy_changed).cumsum()
    strategy_holding_periods = strategy_ids.value_counts().sort_index()
    avg_strategy_holding = strategy_holding_periods.mean()
    turnover = weights.diff().abs().sum().sum() #verif mettre en temps div par nb trades
    for_the_plots = {
        'equity_curve': PNL,
        'equity_curve_fees': PNL_fees,
        'drawdowns_fees': drawdowns
    }

    performance_metrics = {
        'max_drawdown_fees': max_drawdown,
        'sharpe_fees': sharpe,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sortino_ratio': sortino_ratio,
        'winning_rate': winning_rate,
        'total_trades': total_trades,
        'avg_strategy_holding': avg_strategy_holding,
        'turnover' : turnover
    }

    return for_the_plots, performance_metrics

def plot(for_the_plots, df_close_returns):
    fig, ax1 = plt.subplots()

    cumulative_returns = (1 + df_close_returns.fillna(0)).cumprod()
    cumulative_returns.plot(ax=ax1, alpha=0.4, linestyle='--', label='Benchmark')
    ax1.set_ylabel("Cumulative Returns")
    ax1.set_xlabel("Date")

    ax2 = ax1.twinx()

    ax2.plot(for_the_plots['equity_curve'], label='Strategy (No Fees)', color='blue', linewidth=2)
    ax2.plot(for_the_plots['equity_curve_fees'], label='Strategy (With Fees)', color='black', linewidth=2)
    ax2.set_ylabel("Equity Curve")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    # rajouter locscale et mettre un subplot

    plt.title("Strategy Performance vs Benchmark")
    plt.tight_layout()
    plt.show()

def plot_drawdowns(for_the_plots):
        drawdowns_fees = for_the_plots['drawdowns_fees']
        plt.figure(figsize=(14, 4))
        plt.plot(drawdowns_fees, label='Drawdown with fees', color='blue', alpha=1)
        plt.fill_between(drawdowns_fees.index, drawdowns_fees, 0, color='blue', alpha=0.3)

        plt.title('Strategy Drawdowns Over Time')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

def calculate_fees_per_coin(df_weights, params):
    fees = params['fees']
    df_fees = (df_weights.diff().abs() * fees)
    return df_fees

def strat_eval_per_coin(returns, weights, params):
    df_fees = (weights.diff().abs() * params['fees'])

    df_return_strat = (weights * returns).fillna(0)
    df_return_fees_strat = df_return_strat - df_fees.shift(1)

    PNL = (df_return_strat + 1).cumprod()
    PNL_fees = (df_return_fees_strat + 1).cumprod()

    drawdowns = (PNL_fees / PNL_fees.cummax()) - 1
    max_drawdown = drawdowns.min(skipna=True) * 100
    sharpe = df_return_fees_strat.mean() / df_return_fees_strat.std() * np.sqrt(365.25)
    annualized_return = PNL_fees.iloc[-1] ** (365 / len(df_return_fees_strat)) - 1
    annualized_volatility = df_return_fees_strat.std() * np.sqrt(365)
    downside_std = df_return_fees_strat[df_return_fees_strat < 0].std()
    sortino_ratio = df_return_fees_strat.mean() / downside_std * np.sqrt(365)
    winning_rate = (df_return_fees_strat > 0).mean() * 100

    strategy_changed = (weights.diff().abs() != 0)
    total_trades = np.ceil(strategy_changed.sum())
    strategy_ids = (~strategy_changed).cumsum()
    strategy_holding_periods = strategy_ids.value_counts().sort_index()
    avg_strategy_holding = strategy_holding_periods.mean()
    turnover = weights.diff().abs().sum().sum() / (total_trades) #verif mettre en temps div par nb trades

    for_the_plots = {
        'equity_curve': PNL,
        'equity_curve_fees': PNL_fees,
        'drawdowns_fees': drawdowns
    }

    performance_metrics = {
        'max_drawdown_fees': max_drawdown,
        'sharpe_fees': sharpe,
        'annualized_return': annualized_return,
        'annualized_volatility': annualized_volatility,
        'sortino_ratio': sortino_ratio,
        'winning_rate': winning_rate,
        'total_trades': total_trades,
        'avg_strategy_holding': avg_strategy_holding,
        'turnover' : turnover
    }

    return for_the_plots, performance_metrics
