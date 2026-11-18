"""Vectorized single-asset backtester."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    stats: dict

    def __str__(self) -> str:
        width = max(len(k) for k in self.stats)
        lines = []
        for k, v in self.stats.items():
            if isinstance(v, float):
                v = f"{v:.2%}" if k.endswith(("return", "CAGR", "drawdown", "vol", "rate")) else f"{v:.2f}"
            lines.append(f"{k:<{width}}  {v}")
        return "\n".join(lines)


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def run(
    df: pd.DataFrame,
    position: pd.Series,
    cost_bps: float = 5.0,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Backtest a target-position series against close prices.

    The position decided on bar t is held over bar t+1 (no look-ahead).
    `cost_bps` is charged on each unit of position change.
    """
    close = df["close"]
    asset_ret = close.pct_change().fillna(0.0)
    held = position.shift(1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    costs = turnover.shift(1).fillna(0.0) * cost_bps / 10_000
    strat_ret = held * asset_ret - costs
    equity = initial_capital * (1 + strat_ret).cumprod()

    years = len(df) / TRADING_DAYS
    total = float(equity.iloc[-1] / initial_capital - 1)
    vol = float(strat_ret.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(strat_ret.mean() / strat_ret.std() * np.sqrt(TRADING_DAYS)) if strat_ret.std() > 0 else 0.0
    downside = strat_ret[strat_ret < 0].std()
    sortino = float(strat_ret.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0.0

    # Trade-level stats: a trade is a contiguous run of non-zero held position.
    trade_id = (held != held.shift(1)).cumsum()[held != 0]
    trade_rets = (1 + strat_ret[held != 0]).groupby(trade_id).prod() - 1
    bh_total = float(close.iloc[-1] / close.iloc[0] - 1)

    stats = {
        "Total return": total,
        "CAGR": (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan"),
        "Annual vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max drawdown": max_drawdown(equity),
        "Trades": int(len(trade_rets)),
        "Win rate": float((trade_rets > 0).mean()) if len(trade_rets) else 0.0,
        "Exposure": float((held != 0).mean()),
        "Buy & hold return": bh_total,
    }
    return BacktestResult(equity.rename("equity"), strat_ret.rename("returns"), position, stats)
