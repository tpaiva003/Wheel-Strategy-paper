"""
Wheel Strategy bot for INTC on Alpaca paper trading.

Stages:
  1. SELL_PUT  - sell cash-secured puts ~10% OTM, 2-4 weeks out
  2. SELL_CALL - own shares, sell covered calls ~10% above cost basis, 2-4 weeks out

Rules:
  - Liquidity: never sell a put without enough cash to cover assignment
  - Cost basis: never sell a call below cost basis
  - 50% profit-take: close early if open option has gained 50% of premium
  - Track total premium collected across all cycles
  - Only act during market hours
  - Daily summary at market close
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetOptionContractsRequest,
    MarketOrderRequest,
    LimitOrderRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    AssetStatus,
    ContractType,
    OrderStatus,
    QueryOrderStatus,
)
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest

load_dotenv()

LOG = logging.getLogger("wheel")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

STATE_FILE = Path(os.environ.get("WHEEL_STATE_FILE", "wheel_state.json"))
SYMBOL = os.environ.get("SYMBOL", "INTC")

# Strategy parameters
PUT_OTM_PCT = 0.10           # 10% below current price
CALL_OTM_PCT = 0.10          # 10% above cost basis
MIN_DTE = 14                  # 2 weeks
MAX_DTE = 28                  # 4 weeks
PROFIT_TAKE_PCT = 0.50        # close at 50% of max profit
CONTRACT_MULTIPLIER = 100     # standard equity option


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------
@dataclass
class State:
    stage: str = "SELL_PUT"               # SELL_PUT or SELL_CALL
    open_option_symbol: Optional[str] = None
    open_option_side: Optional[str] = None  # "put" or "call"
    open_option_premium: float = 0.0       # credit per share when sold
    open_option_qty: int = 0
    cost_basis: Optional[float] = None     # per-share cost basis after assignment
    total_premium_collected: float = 0.0
    last_summary_date: Optional[str] = None
    cycles_completed: int = 0

    @classmethod
    def load(cls) -> "State":
        if STATE_FILE.exists():
            return cls(**json.loads(STATE_FILE.read_text()))
        return cls()

    def save(self) -> None:
        STATE_FILE.write_text(json.dumps(asdict(self), indent=2))


# ---------------------------------------------------------------------------
# Broker wrapper
# ---------------------------------------------------------------------------
class Broker:
    def __init__(self) -> None:
        api_key = os.environ["ALPACA_API_KEY"]
        secret = os.environ["ALPACA_SECRET_KEY"]
        paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"
        self.trading = TradingClient(api_key, secret, paper=paper)
        self.stock_data = StockHistoricalDataClient(api_key, secret)
        self.option_data = OptionHistoricalDataClient(api_key, secret)

    # --- market state ---
    def is_market_open(self) -> bool:
        return self.trading.get_clock().is_open

    def market_close_today(self) -> datetime:
        return self.trading.get_clock().next_close

    def minutes_to_close(self) -> float:
        clock = self.trading.get_clock()
        if not clock.is_open:
            return -1
        return (clock.next_close - clock.timestamp).total_seconds() / 60.0

    # --- account ---
    def cash(self) -> float:
        return float(self.trading.get_account().cash)

    def shares_held(self, symbol: str) -> int:
        try:
            pos = self.trading.get_open_position(symbol)
            return int(float(pos.qty))
        except Exception:
            return 0

    def position_avg_cost(self, symbol: str) -> Optional[float]:
        try:
            pos = self.trading.get_open_position(symbol)
            return float(pos.avg_entry_price)
        except Exception:
            return None

    # --- prices ---
    def stock_price(self, symbol: str) -> float:
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        return float(self.stock_data.get_stock_latest_trade(req)[symbol].price)

    def option_mid(self, occ_symbol: str) -> Optional[float]:
        req = OptionLatestQuoteRequest(symbol_or_symbols=occ_symbol)
        try:
            q = self.option_data.get_option_latest_quote(req)[occ_symbol]
            bid, ask = float(q.bid_price or 0), float(q.ask_price or 0)
            if bid <= 0 or ask <= 0:
                return None
            return round((bid + ask) / 2, 2)
        except Exception:
            return None

    # --- option chain ---
    def find_contract(
        self,
        underlying: str,
        side: ContractType,
        target_strike: float,
        prefer_above: bool,
    ) -> Optional[object]:
        """Find contract closest to target_strike with 14-28 DTE."""
        today = date.today()
        req = GetOptionContractsRequest(
            underlying_symbols=[underlying],
            status=AssetStatus.ACTIVE,
            type=side,
            expiration_date_gte=today + timedelta(days=MIN_DTE),
            expiration_date_lte=today + timedelta(days=MAX_DTE),
            limit=500,
        )
        contracts = self.trading.get_option_contracts(req).option_contracts or []
        if not contracts:
            return None

        by_exp: dict = {}
        for c in contracts:
            by_exp.setdefault(c.expiration_date, []).append(c)
        chosen_exp = sorted(by_exp.keys())[0]
        candidates = by_exp[chosen_exp]

        def keep(c) -> bool:
            s = float(c.strike_price)
            return s >= target_strike if prefer_above else s <= target_strike

        side_filtered = [c for c in candidates if keep(c)]
        if not side_filtered:
            side_filtered = candidates

        return min(side_filtered, key=lambda c: abs(float(c.strike_price) - target_strike))

    # --- orders ---
    def sell_to_open(self, occ_symbol: str, qty: int, limit_price: float) -> str:
        order = LimitOrderRequest(
            symbol=occ_symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
        )
        resp = self.trading.submit_order(order)
        return str(resp.id)

    def buy_to_close(self, occ_symbol: str, qty: int, limit_price: float) -> str:
        order = LimitOrderRequest(
            symbol=occ_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
        )
        resp = self.trading.submit_order(order)
        return str(resp.id)

    def has_open_position(self, occ_symbol: str) -> bool:
        try:
            self.trading.get_open_position(occ_symbol)
            return True
        except Exception:
            return False

    # --- safety: detect any short put on an underlying, regardless of state file ---
    def existing_short_put(self, underlying: str) -> Optional[dict]:
        """Return dict {symbol, qty, avg_entry_price} of any existing short put
        on the given underlying, or None. Used as a safety net so the bot never
        sells a duplicate put even if the persisted state file is missing or stale.
        """
        try:
            positions = self.trading.get_all_positions()
        except Exception as e:
            LOG.warning("Could not list positions for safety check: %s", e)
            return None
        for p in positions:
            symbol = getattr(p, "symbol", "") or ""
            if not symbol.startswith(underlying):
                continue
            # OCC symbol layout: {underlying}{YYMMDD}{P|C}{strike*1000 padded to 8}
            rest = symbol[len(underlying):]
            if len(rest) < 15 or rest[6] not in ("P", "C"):
                continue
            if rest[6] != "P":
                continue
            try:
                qty = float(p.qty)
            except Exception:
                continue
            if qty >= 0:
                continue
            try:
                avg = float(p.avg_entry_price)
            except Exception:
                avg = 0.0
            return {
                "symbol": symbol,
                "qty": int(abs(qty)),
                "avg_entry_price": avg,
            }
        return None


# ---------------------------------------------------------------------------
# Strategy core
# ---------------------------------------------------------------------------
class Wheel:
    def __init__(self, broker: Broker, state: State, symbol: str = SYMBOL) -> None:
        self.broker = broker
        self.state = state
        self.symbol = symbol

    # ----- reconciliation -----
    def reconcile(self) -> None:
        """Detect assignment / call-away by reconciling positions with state.
        Also adopts any orphan short put on the underlying that the state file
        does not track (safety net against lost state)."""
        shares = self.broker.shares_held(self.symbol)

        if self.state.open_option_symbol:
            still_open = self.broker.has_open_position(self.state.open_option_symbol)
            if not still_open:
                LOG.info("Tracked option %s no longer open - assuming closed/expired/assigned",
                         self.state.open_option_symbol)
                self.state.open_option_symbol = None
                self.state.open_option_side = None
                self.state.open_option_premium = 0.0
                self.state.open_option_qty = 0

        # Safety: if the state file says no open option but the broker shows a
        # short put on this underlying, adopt it. This prevents duplicate sells
        # when persisted state is missing or stale.
        if not self.state.open_option_symbol:
            orphan = self.broker.existing_short_put(self.symbol)
            if orphan:
                LOG.warning("Adopting orphan short put from broker: %s qty=%d avg=%.2f",
                            orphan["symbol"], orphan["qty"], orphan["avg_entry_price"])
                self.state.open_option_symbol = orphan["symbol"]
                self.state.open_option_side = "put"
                self.state.open_option_premium = orphan["avg_entry_price"]
                self.state.open_option_qty = orphan["qty"]
                self.state.total_premium_collected += (
                    orphan["avg_entry_price"] * orphan["qty"] * CONTRACT_MULTIPLIER
                )

        if shares >= 100 and self.state.stage != "SELL_CALL":
            avg = self.broker.position_avg_cost(self.symbol) or self.broker.stock_price(self.symbol)
            premium_per_share = self.state.total_premium_collected / 100.0
            self.state.cost_basis = round(avg - premium_per_share, 2)
            self.state.stage = "SELL_CALL"
            LOG.info("Assigned %d shares at avg %.2f. Effective cost basis %.2f. Stage -> SELL_CALL",
                     shares, avg, self.state.cost_basis)

        if shares < 100 and self.state.stage == "SELL_CALL":
            LOG.info("Shares called away. Cycle complete. Stage -> SELL_PUT")
            self.state.stage = "SELL_PUT"
            self.state.cost_basis = None
            self.state.cycles_completed += 1

        self.state.save()

    # ----- profit-take -----
    def maybe_profit_take(self) -> bool:
        if not self.state.open_option_symbol:
            return False
        mid = self.broker.option_mid(self.state.open_option_symbol)
        if mid is None:
            return False
        entry = self.state.open_option_premium
        if entry <= 0:
            return False
        if mid <= entry * (1 - PROFIT_TAKE_PCT):
            LOG.info("50%% profit-take: %s entry=%.2f now=%.2f -> buy to close",
                     self.state.open_option_symbol, entry, mid)
            self.broker.buy_to_close(
                self.state.open_option_symbol, self.state.open_option_qty, mid
            )
            self.state.total_premium_collected -= mid * self.state.open_option_qty * CONTRACT_MULTIPLIER
            self.state.open_option_symbol = None
            self.state.open_option_side = None
            self.state.open_option_premium = 0.0
            self.state.open_option_qty = 0
            self.state.save()
            return True
        return False

    # ----- stage 1: sell put -----
    def try_sell_put(self) -> None:
        if self.state.open_option_symbol:
            return

        # Safety net: even if state says empty, re-check the broker right before
        # selling. If a short put already exists on this underlying, adopt it
        # instead of creating a duplicate.
        orphan = self.broker.existing_short_put(self.symbol)
        if orphan:
            LOG.warning("try_sell_put aborting: broker already has short put %s "
                        "qty=%d avg=%.2f; adopting into state",
                        orphan["symbol"], orphan["qty"], orphan["avg_entry_price"])
            self.state.open_option_symbol = orphan["symbol"]
            self.state.open_option_side = "put"
            self.state.open_option_premium = orphan["avg_entry_price"]
            self.state.open_option_qty = orphan["qty"]
            self.state.total_premium_collected += (
                orphan["avg_entry_price"] * orphan["qty"] * CONTRACT_MULTIPLIER
            )
            self.state.save()
            return

        price = self.broker.stock_price(self.symbol)
        target_strike = price * (1 - PUT_OTM_PCT)
        contract = self.broker.find_contract(
            self.symbol, ContractType.PUT, target_strike, prefer_above=False
        )
        if contract is None:
            LOG.warning("No put contract found near %.2f", target_strike)
            return

        strike = float(contract.strike_price)
        cash_required = strike * CONTRACT_MULTIPLIER
        if self.broker.cash() < cash_required:
            LOG.warning("Insufficient cash %.2f to secure put strike %.2f (need %.2f)",
                        self.broker.cash(), strike, cash_required)
            return

        mid = self.broker.option_mid(contract.symbol)
        if mid is None or mid <= 0:
            LOG.warning("No valid quote for %s", contract.symbol)
            return

        LOG.info("Selling put %s strike=%.2f exp=%s mid=%.2f",
                 contract.symbol, strike, contract.expiration_date, mid)
        self.broker.sell_to_open(contract.symbol, 1, mid)

        self.state.open_option_symbol = contract.symbol
        self.state.open_option_side = "put"
        self.state.open_option_premium = mid
        self.state.open_option_qty = 1
        self.state.total_premium_collected += mid * CONTRACT_MULTIPLIER
        self.state.save()

    # ----- stage 2: sell call -----
    def try_sell_call(self) -> None:
        if self.state.open_option_symbol:
            return
        if self.state.cost_basis is None:
            LOG.warning("No cost basis recorded; skipping call sell")
            return
        target_strike = self.state.cost_basis * (1 + CALL_OTM_PCT)
        contract = self.broker.find_contract(
            self.symbol, ContractType.CALL, target_strike, prefer_above=True
        )
        if contract is None:
            LOG.warning("No call contract found near %.2f", target_strike)
            return

        strike = float(contract.strike_price)
        if strike < self.state.cost_basis:
            LOG.warning("Skipping call: strike %.2f below cost basis %.2f",
                        strike, self.state.cost_basis)
            return

        mid = self.broker.option_mid(contract.symbol)
        if mid is None or mid <= 0:
            LOG.warning("No valid quote for %s", contract.symbol)
            return

        LOG.info("Selling call %s strike=%.2f exp=%s mid=%.2f",
                 contract.symbol, strike, contract.expiration_date, mid)
        self.broker.sell_to_open(contract.symbol, 1, mid)

        self.state.open_option_symbol = contract.symbol
        self.state.open_option_side = "call"
        self.state.open_option_premium = mid
        self.state.open_option_qty = 1
        self.state.total_premium_collected += mid * CONTRACT_MULTIPLIER
        self.state.save()

    # ----- daily summary -----
    def daily_summary(self) -> str:
        shares = self.broker.shares_held(self.symbol)
        cash = self.broker.cash()
        try:
            price = self.broker.stock_price(self.symbol)
        except Exception:
            price = 0.0
        equity_value = shares * price
        unrealized = 0.0
        if shares and self.state.cost_basis:
            unrealized = (price - self.state.cost_basis) * shares

        total_return = self.state.total_premium_collected + unrealized

        lines = [
            f"=== Wheel daily summary {date.today()} ({self.symbol}) ===",
            f"Stage: {self.state.stage}",
            f"Cycles completed: {self.state.cycles_completed}",
            f"Cash: ${cash:,.2f}",
            f"Shares: {shares} @ market ${price:.2f} = ${equity_value:,.2f}",
            f"Cost basis: {self.state.cost_basis}",
            f"Open option: {self.state.open_option_symbol} "
            f"(entry credit ${self.state.open_option_premium:.2f})",
            f"Total premium collected: ${self.state.total_premium_collected:,.2f}",
            f"Unrealized stock P&L: ${unrealized:,.2f}",
            f"Total return: ${total_return:,.2f}",
        ]
        return "\n".join(lines)

    # ----- main tick -----
    def tick(self) -> None:
        if not self.broker.is_market_open():
            LOG.info("Market closed - nothing to do")
            return

        self.reconcile()

        if self.maybe_profit_take():
            return

        if self.state.stage == "SELL_PUT":
            self.try_sell_put()
        elif self.state.stage == "SELL_CALL":
            self.try_sell_call()

        mtc = self.broker.minutes_to_close()
        today_str = str(date.today())
        if 0 < mtc <= 16 and self.state.last_summary_date != today_str:
            summary = self.daily_summary()
            print(summary)
            LOG.info("\n%s", summary)
            self.state.last_summary_date = today_str
            self.state.save()


def run_once() -> None:
    broker = Broker()
    state = State.load()
    Wheel(broker, state).tick()


if __name__ == "__main__":
    run_once()
