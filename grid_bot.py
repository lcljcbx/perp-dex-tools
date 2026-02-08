"""
Grid Trading Bot

Implements a directional grid (long/short) with arithmetic/geometric spacing on a fixed price range.

Behavior (directional, one-sided open orders):
- Long grid:
  - Places BUY open orders on grid levels within [lower, upper]
  - When a BUY at level i fills, places a SELL close order at level i+1 (or upper for the last level)
  - When a SELL close fills, re-places the BUY open at level i to continue the cycle

- Short grid:
  - Places SELL open orders on grid levels within [lower, upper]
  - When a SELL at level i fills, places a BUY close order at level i-1 (or lower for the first level)
  - When a BUY close fills, re-places the SELL open at level i to continue the cycle

Note:
- This bot requires the exchange client to implement `place_limit_order` for explicit price placement.
"""

import asyncio
import traceback
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from exchanges import ExchangeFactory
from helpers import TradingLogger


@dataclass
class GridConfig:
    exchange: str
    ticker: str
    contract_id: str
    tick_size: Decimal
    quantity: Decimal

    # grid params
    direction: str  # "long" | "short"
    spacing: str    # "arith" | "geo"
    lower: Decimal
    upper: Decimal
    grids: int
    grid_size: Decimal  # per-grid order size

    # execution controls
    post_only: bool = True
    max_concurrent_orders: Optional[int] = None  # optional extra cap
    
    # exit conditions
    take_profit_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None

    @property
    def open_side(self) -> str:
        return "buy" if self.direction == "long" else "sell"

    @property
    def close_side(self) -> str:
        return "sell" if self.direction == "long" else "buy"

    @property
    def close_order_side(self) -> str:
        return self.close_side

class GridBot:
    def __init__(self, config: GridConfig):
        self.config = config
        self.logger = TradingLogger(config.exchange, config.ticker, log_to_console=True)

        self.exchange_client = ExchangeFactory.create_exchange(config.exchange, config)

        # runtime state
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.shutdown_requested = False

        # grid levels (ascending)
        self.levels: List[Decimal] = []

        # order_id -> (kind, level_index)
        # kind: "OPEN" | "CLOSE"
        self.order_map: Dict[str, Tuple[str, int]] = {}

        # per level active order ids
        self.open_order_ids: Dict[int, str] = {}
        self.close_order_ids: Dict[int, str] = {}
        
        # level state tracking: idx -> "OPEN" | "CLOSE"
        self.level_phases: Dict[int, str] = {}

        self._setup_order_handler()

    def _setup_order_handler(self):
        def handler(msg: dict):
            try:
                if msg.get("contract_id") != self.config.contract_id:
                    return

                order_id = msg.get("order_id")
                status = msg.get("status")
                side = (msg.get("side") or "").lower()
                filled_size = Decimal(str(msg.get("filled_size", "0")))

                if not order_id or order_id not in self.order_map:
                    return

                kind, idx = self.order_map[order_id]

                if status != "FILLED":
                    return

                # Only act on full fills (SDK emits FILLED); partials are ignored here.
                if kind == "OPEN":
                    # open filled -> place corresponding close
                    self.open_order_ids.pop(idx, None)
                    self.level_phases[idx] = "CLOSE"
                    self.logger.log(f"[GRID] OPEN filled at level {idx} ({self.levels[idx]}), placing CLOSE", "INFO")
                    if self.loop:
                        self.loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(self._place_close_for_level(idx, filled_size))
                        )
                elif kind == "CLOSE":
                    # close filled -> re-place open at same level
                    self.close_order_ids.pop(idx, None)
                    self.level_phases[idx] = "OPEN"
                    self.logger.log(f"[GRID] CLOSE filled for level {idx}, re-placing OPEN", "INFO")
                    if self.loop:
                        self.loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(self._place_open_for_level(idx))
                        )
                
                # cleanup order map
                self.order_map.pop(order_id, None)

            except Exception as e:
                self.logger.log(f"[GRID] handler error: {e}", "ERROR")
                self.logger.log(f"Traceback: {traceback.format_exc()}", "ERROR")

        self.exchange_client.setup_order_update_handler(handler)

    def _build_levels(self) -> List[Decimal]:
        if self.config.grids < 2:
            raise ValueError("--grid-grids must be >= 2")
        if self.config.lower <= 0 or self.config.upper <= 0:
            raise ValueError("--grid-lower/--grid-upper must be > 0")
        if self.config.lower >= self.config.upper:
            raise ValueError("--grid-lower must be < --grid-upper")

        lower = self.config.lower
        upper = self.config.upper
        n = self.config.grids

        if self.config.spacing == "arith":
            step = (upper - lower) / Decimal(n - 1)
            levels = [lower + step * Decimal(i) for i in range(n)]
        elif self.config.spacing == "geo":
            ratio = (upper / lower) ** (Decimal("1") / Decimal(n - 1))
            levels = [lower * (ratio ** Decimal(i)) for i in range(n)]
        else:
            raise ValueError("--grid-spacing must be 'arith' or 'geo'")

        # round to tick
        levels = [self.exchange_client.round_to_tick(p) for p in levels]
        # ensure sorted and unique-ish (tick rounding can collapse)
        levels = sorted(list(dict.fromkeys(levels)))
        if len(levels) < 2:
            raise ValueError("Grid levels collapsed after tick rounding; widen range or increase tick precision.")
        return levels

    def _close_price_for_level(self, idx: int) -> Decimal:
        # levels are ascending
        if self.config.direction == "long":
            # sell at next higher level
            if idx + 1 < len(self.levels):
                return self.levels[idx + 1]
            return self.levels[-1]
        else:
            # short: buy at next lower level
            if idx - 1 >= 0:
                return self.levels[idx - 1]
            return self.levels[0]

    async def _place_open_for_level(self, idx: int):
        if self.shutdown_requested:
            return

        # don't double-place
        if idx in self.open_order_ids:
            return
        # if a close is still working for this level, wait
        if idx in self.close_order_ids:
            return

        price = self.levels[idx]
        side = self.config.open_side
        qty = self.config.grid_size

        if not hasattr(self.exchange_client, "place_limit_order"):
            raise ValueError(f"Exchange {self.config.exchange} does not support place_limit_order required for grid")

        res = await self.exchange_client.place_limit_order(
            self.config.contract_id, qty, price, side, post_only=self.config.post_only, reduce_only=False
        )
        if not res.success:
            self.logger.log(f"[GRID] Failed to place OPEN at level {idx} {price}: {res.error_message}", "ERROR")
            return

        order_id = res.order_id
        self.open_order_ids[idx] = order_id
        self.order_map[order_id] = ("OPEN", idx)
        self.logger.log(f"[GRID] OPEN placed level {idx}: {side} {qty} @ {res.price}", "INFO")

    async def _place_close_for_level(self, idx: int, filled_size: Decimal):
        if self.shutdown_requested:
            return

        # clear open id for this level (it filled)
        self.open_order_ids.pop(idx, None)

        # don't double-place
        if idx in self.close_order_ids:
            return

        close_price = self._close_price_for_level(idx)
        side = self.config.close_side
        qty = filled_size if filled_size and filled_size > 0 else self.config.grid_size

        res = await self.exchange_client.place_limit_order(
            self.config.contract_id, qty, close_price, side, post_only=self.config.post_only, reduce_only=True
        )
        if not res.success:
            self.logger.log(f"[GRID] Failed to place CLOSE for level {idx} @ {close_price}: {res.error_message}", "ERROR")
            return

        order_id = res.order_id
        self.close_order_ids[idx] = order_id
        self.order_map[order_id] = ("CLOSE", idx)
        self.logger.log(f"[GRID] CLOSE placed for level {idx}: {side} {qty} @ {res.price}", "INFO")

    async def _seed_open_orders(self):
        # Place initial open orders at every level (directional one-side)
        for idx in range(len(self.levels)):
            self.level_phases[idx] = "OPEN"
            if self.config.max_concurrent_orders is not None:
                if len(self.open_order_ids) + len(self.close_order_ids) >= self.config.max_concurrent_orders:
                    break
            await self._place_open_for_level(idx)
            await asyncio.sleep(0.05)

    async def _monitor_exit_conditions(self):
        """Monitor price for TP/SL conditions."""
        self.logger.log("Starting exit condition monitor...", "INFO")
        while not self.shutdown_requested:
            try:
                # 1. Get current price
                best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
                if best_bid <= 0 or best_ask <= 0:
                    await asyncio.sleep(1)
                    continue
                
                mid_price = (best_bid + best_ask) / 2
                
                # 2. Check TP
                if self.config.take_profit_price and self.config.take_profit_price > 0:
                    triggered = False
                    if self.config.direction == "long" and mid_price >= self.config.take_profit_price:
                        triggered = True
                    elif self.config.direction == "short" and mid_price <= self.config.take_profit_price:
                        triggered = True
                        
                    if triggered:
                        self.logger.log(f"[GRID] Take Profit triggered at {mid_price} (Target: {self.config.take_profit_price})", "WARNING")
                        await self._execute_take_profit()
                        return

                # 3. Check SL
                if self.config.stop_loss_price and self.config.stop_loss_price > 0:
                    triggered = False
                    if self.config.direction == "long" and mid_price <= self.config.stop_loss_price:
                        triggered = True
                    elif self.config.direction == "short" and mid_price >= self.config.stop_loss_price:
                        triggered = True
                        
                    if triggered:
                        self.logger.log(f"[GRID] Stop Loss triggered at {mid_price} (Target: {self.config.stop_loss_price})", "WARNING")
                        await self._execute_stop_loss()
                        return

                await asyncio.sleep(1)
            except Exception as e:
                self.logger.log(f"[GRID] Monitor error: {e}", "ERROR")
                await asyncio.sleep(5)

    async def _execute_take_profit(self):
        """Execute TP logic: Cancel all orders and exit."""
        self.shutdown_requested = True # Stop grid logic
        self.logger.log("[GRID] Executing Take Profit...", "INFO")
        
        # Cancel orders
        await self._cancel_all_orders()
        
        self.logger.log("[GRID] Take Profit execution complete. Exiting.", "INFO")
        # Ensure we disconnect
        await self.exchange_client.disconnect()
        # Raise SystemExit or just let the run loop finish (shutdown_requested is True)
        # Since run loop waits on shutdown_requested, it will exit.

    async def _execute_stop_loss(self):
        """Execute SL logic: Cancel orders, close position, exit."""
        self.shutdown_requested = True # Stop grid logic
        self.logger.log("[GRID] Executing Stop Loss...", "WARNING")
        
        # 1. Cancel orders
        await self._cancel_all_orders()
        
        # 2. Close position
        try:
            position = await self.exchange_client.get_account_positions()
            if abs(position) > 0:
                self.logger.log(f"[GRID] Closing position {position}...", "WARNING")
                side = "sell" if position > 0 else "buy"
                qty = abs(position)
                
                # Try market close first
                if hasattr(self.exchange_client, "place_market_order"):
                    res = await self.exchange_client.place_market_order(
                        self.config.contract_id, qty, side
                    )
                else:
                    # Aggressive limit close
                    best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
                    # 5% slippage allowance
                    price = best_ask * Decimal("1.05") if side == "buy" else best_bid * Decimal("0.95")
                    res = await self.exchange_client.place_limit_order(
                        self.config.contract_id, qty, price, side, reduce_only=True
                    )
                
                if res.success:
                    self.logger.log("[GRID] Position closed successfully.", "INFO")
                else:
                    self.logger.log(f"[GRID] Failed to close position: {res.error_message}", "ERROR")
            else:
                self.logger.log("[GRID] No position to close.", "INFO")
                
        except Exception as e:
            self.logger.log(f"[GRID] Error closing position: {e}", "ERROR")

        self.logger.log("[GRID] Stop Loss execution complete. Exiting.", "INFO")
        await self.exchange_client.disconnect()

    async def _cancel_all_orders(self):
        """Cancel all active grid orders."""
        all_ids = list(self.open_order_ids.values()) + list(self.close_order_ids.values())
        if not all_ids:
            return
            
        self.logger.log(f"[GRID] Cancelling {len(all_ids)} orders...", "INFO")
        tasks = [self.exchange_client.cancel_order(oid) for oid in all_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.open_order_ids.clear()
        self.close_order_ids.clear()
        self.order_map.clear()

    async def _maintenance_loop(self):
        """Periodically check grid consistency and retry missing orders."""
        self.logger.log("Starting maintenance loop...", "INFO")
        while not self.shutdown_requested:
            try:
                await asyncio.sleep(10)
                
                # Check each level's state vs reality
                for idx in list(self.level_phases.keys()):
                    phase = self.level_phases[idx]
                    
                    if phase == "OPEN":
                        # We expect an OPEN order
                        if idx not in self.open_order_ids:
                            # Verify we don't have a close order (conflicting state)
                            if idx in self.close_order_ids:
                                self.logger.log(f"[GRID] Maintenance: Level {idx} has CLOSE order but phase is OPEN. Correcting phase.", "WARNING")
                                self.level_phases[idx] = "CLOSE"
                                continue
                            
                            self.logger.log(f"[GRID] Maintenance: Level {idx} phase is OPEN but no order. Retrying OPEN.", "WARNING")
                            await self._place_open_for_level(idx)
                            
                    elif phase == "CLOSE":
                        # We expect a CLOSE order
                        if idx not in self.close_order_ids:
                            # Verify we don't have an open order
                            if idx in self.open_order_ids:
                                self.logger.log(f"[GRID] Maintenance: Level {idx} has OPEN order but phase is CLOSE. Correcting phase.", "WARNING")
                                self.level_phases[idx] = "OPEN"
                                continue
                                
                            self.logger.log(f"[GRID] Maintenance: Level {idx} phase is CLOSE but no order. Retrying CLOSE.", "WARNING")
                            # Note: we lost filled_size info here if we crashed/restarted, default to grid_size
                            await self._place_close_for_level(idx, self.config.grid_size)
                            
            except Exception as e:
                self.logger.log(f"[GRID] Maintenance error: {e}", "ERROR")
                await asyncio.sleep(5)

    async def run(self):
        self.loop = asyncio.get_running_loop()

        self.config.contract_id, self.config.tick_size = await self.exchange_client.get_contract_attributes()

        self.levels = self._build_levels()
        self.logger.log(
            f"[GRID] levels={len(self.levels)} direction={self.config.direction} spacing={self.config.spacing} "
            f"range=[{self.levels[0]}, {self.levels[-1]}] size={self.config.grid_size}",
            "INFO",
        )

        await self.exchange_client.connect()
        await asyncio.sleep(2)

        await self._seed_open_orders()

        # Start exit condition monitor
        monitor_task = asyncio.create_task(self._monitor_exit_conditions())
        maintenance_task = asyncio.create_task(self._maintenance_loop())

        # idle loop; work is done by callbacks
        while not self.shutdown_requested:
            await asyncio.sleep(5)
            
        # If we broke out of loop, ensure monitor is cancelled if it's still running
        if not monitor_task.done():
            monitor_task.cancel()
        if not maintenance_task.done():
            maintenance_task.cancel()

    async def graceful_shutdown(self, reason: str = "Unknown"):
        self.logger.log(f"[GRID] shutdown: {reason}", "INFO")
        self.shutdown_requested = True
        try:
            await self.exchange_client.disconnect()
        except Exception as e:
            self.logger.log(f"[GRID] disconnect error: {e}", "ERROR")
