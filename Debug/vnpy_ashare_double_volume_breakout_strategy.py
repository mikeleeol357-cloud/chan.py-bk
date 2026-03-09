"""
基于 vn.py CTA 模板实现的 A 股短线“倍量突破”多买点策略。

策略思路（仅做多）：
1. 倍量突破追涨买点：
   - 收盘价突破 N 日新高
   - 当日成交量 >= 最近 M 日均量 * 倍量系数
   - 均线多头（快线 > 慢线）过滤
2. 回踩确认买点：
   - 发生过一次有效突破后，价格在若干日内回踩突破位附近
   - 回踩不破前低并重新收回短均线
3. 二次放量突破买点：
   - 首次突破后经过一段整理，再次创出阶段新高且再次放量

风控：
- 固定止损（以最新入场均价为基准）
- 浮盈回撤止盈（追踪最高价）
- 信号冷却，避免连续追单
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from vnpy.trader.object import BarData
from vnpy_ctastrategy import ArrayManager, BarGenerator, CtaTemplate


@dataclass
class BreakoutState:
    """记录首次突破后的状态，供回踩与二次突破使用。"""

    price: float = 0.0
    dt: Optional[datetime] = None
    valid_days: int = 0


class AshareDoubleVolumeBreakoutStrategy(CtaTemplate):
    """A股短线：倍量突破后多买点策略。"""

    author = "Codex"

    # ===== 参数（可在 vn.py 策略界面动态配置） =====
    breakout_window = 20            # 新高观察窗口
    volume_ma_window = 10           # 均量窗口
    double_volume_factor = 2.0      # 倍量系数

    fast_ma_window = 5              # 趋势过滤：快均线
    slow_ma_window = 20             # 趋势过滤：慢均线

    pullback_days = 8               # 回踩买点有效天数
    pullback_tolerance = 0.02       # 回踩容忍度（相对突破价）

    rebreak_window = 15             # 二次突破观察窗口
    cooldown_bars = 3               # 开仓后冷却K线数

    risk_per_trade = 0.3            # 每次信号默认下单仓位（手数请结合合约乘数/资金管理调整）
    stop_loss_pct = 0.03            # 固定止损比例
    trailing_take_profit_pct = 0.05 # 浮盈回撤止盈

    parameters = [
        "breakout_window",
        "volume_ma_window",
        "double_volume_factor",
        "fast_ma_window",
        "slow_ma_window",
        "pullback_days",
        "pullback_tolerance",
        "rebreak_window",
        "cooldown_bars",
        "risk_per_trade",
        "stop_loss_pct",
        "trailing_take_profit_pct",
    ]

    # ===== 变量（界面可见） =====
    breakout_price = 0.0
    cooldown_count = 0
    highest_since_entry = 0.0
    last_entry_price = 0.0

    variables = [
        "breakout_price",
        "cooldown_count",
        "highest_since_entry",
        "last_entry_price",
    ]

    def __init__(self, cta_engine, strategy_name: str, vt_symbol: str, setting: dict):
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager(size=200)

        self.breakout_state = BreakoutState()

    def on_init(self) -> None:
        self.write_log("策略初始化")
        self.load_bar(50)

    def on_start(self) -> None:
        self.write_log("策略启动")

    def on_stop(self) -> None:
        self.write_log("策略停止")

    def on_tick(self, tick) -> None:
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData) -> None:
        self.cancel_all()

        self.am.update_bar(bar)
        if not self.am.inited:
            return

        # 冷却计数
        if self.cooldown_count > 0:
            self.cooldown_count -= 1

        # 先做持仓风控
        self._manage_position(bar)

        # 只做多，已有持仓时不重复加仓（可按需扩展分批加仓）
        if self.pos > 0:
            self.put_event()
            return

        trend_ok = self._trend_filter()
        breakout_signal = self._first_breakout_signal()
        pullback_signal = self._pullback_confirmation_signal()
        rebreak_signal = self._second_breakout_signal()

        if self.cooldown_count == 0 and trend_ok:
            if breakout_signal:
                self._open_long(bar.close_price, reason="首次倍量突破")
                self.breakout_state = BreakoutState(
                    price=bar.close_price,
                    dt=bar.datetime,
                    valid_days=self.pullback_days,
                )
            elif pullback_signal:
                self._open_long(bar.close_price, reason="突破后回踩确认")
            elif rebreak_signal:
                self._open_long(bar.close_price, reason="二次放量突破")

        self._update_breakout_state(bar)
        self.put_event()

    # ===== 信号逻辑 =====
    def _trend_filter(self) -> bool:
        fast_ma = self.am.sma(self.fast_ma_window)
        slow_ma = self.am.sma(self.slow_ma_window)
        return fast_ma > slow_ma

    def _first_breakout_signal(self) -> bool:
        """买点1：倍量突破新高。"""
        prev_high = max(self.am.high_array[-self.breakout_window - 1 : -1])
        volume_ma = self.am.sma(self.volume_ma_window, array=True)[-2]
        cur_volume = self.am.volume_array[-1]
        cur_close = self.am.close_array[-1]

        return (
            cur_close > prev_high
            and volume_ma > 0
            and cur_volume >= volume_ma * self.double_volume_factor
        )

    def _pullback_confirmation_signal(self) -> bool:
        """买点2：突破后回踩支撑再转强。"""
        if self.breakout_state.valid_days <= 0 or self.breakout_state.price <= 0:
            return False

        breakout_price = self.breakout_state.price
        cur_close = self.am.close_array[-1]
        cur_low = self.am.low_array[-1]
        prev_close = self.am.close_array[-2]
        fast_ma = self.am.sma(self.fast_ma_window)

        near_breakout = abs(cur_low - breakout_price) / breakout_price <= self.pullback_tolerance
        reclaim_strength = prev_close < fast_ma <= cur_close

        return near_breakout and reclaim_strength

    def _second_breakout_signal(self) -> bool:
        """买点3：整理后的二次放量再突破。"""
        if self.breakout_state.price <= 0:
            return False

        recent_high = max(self.am.high_array[-self.rebreak_window - 1 : -1])
        cur_close = self.am.close_array[-1]

        volume_ma = self.am.sma(self.volume_ma_window, array=True)[-2]
        cur_volume = self.am.volume_array[-1]

        return (
            cur_close > recent_high
            and cur_close > self.breakout_state.price
            and volume_ma > 0
            and cur_volume >= volume_ma * self.double_volume_factor
        )

    # ===== 交易与风控 =====
    def _open_long(self, price: float, reason: str) -> None:
        self.buy(price, self.risk_per_trade)
        self.cooldown_count = self.cooldown_bars
        self.last_entry_price = price
        self.highest_since_entry = price
        self.write_log(f"开多触发: {reason}, price={price:.2f}")

    def _manage_position(self, bar: BarData) -> None:
        if self.pos <= 0:
            return

        self.highest_since_entry = max(self.highest_since_entry, bar.high_price)
        stop_price = self.last_entry_price * (1 - self.stop_loss_pct)
        trail_price = self.highest_since_entry * (1 - self.trailing_take_profit_pct)

        if bar.close_price <= stop_price:
            self.sell(bar.close_price, abs(self.pos))
            self.write_log(f"触发止损平仓, close={bar.close_price:.2f}, stop={stop_price:.2f}")
        elif bar.close_price <= trail_price:
            self.sell(bar.close_price, abs(self.pos))
            self.write_log(f"触发回撤止盈平仓, close={bar.close_price:.2f}, trail={trail_price:.2f}")

    def _update_breakout_state(self, bar: BarData) -> None:
        if self.breakout_state.valid_days > 0:
            self.breakout_state.valid_days -= 1

        # 若价格明显跌破突破位，失效首次突破状态
        if self.breakout_state.price > 0 and bar.close_price < self.breakout_state.price * (1 - self.pullback_tolerance * 1.5):
            self.breakout_state = BreakoutState()

    # ===== vn.py 回报回调（最简实现） =====
    def on_order(self, order) -> None:
        pass

    def on_trade(self, trade) -> None:
        # 按成交更新入场价格锚点（防止滑点导致风控偏差）
        if trade.direction.value == "long" and trade.offset.value == "open":
            self.last_entry_price = trade.price
            self.highest_since_entry = max(self.highest_since_entry, trade.price)
        self.put_event()

    def on_stop_order(self, stop_order) -> None:
        pass
