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
                    self.logger.log(f"[GRID] OPEN filled at level {idx} ({self.levels[idx]}), placing CLOSE", "INFO")
                    if self.loop:
                        self.loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(self._place_close_for_level(idx, filled_size))
                        )
                elif kind == "CLOSE":
                    # close filled -> re-place open at same level
                    self.logger.log(f"[GRID] CLOSE filled for level {idx}, re-placing OPEN", "INFO")
                    if self.loop:
                        self.loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(self._place_open_for_level(idx))
                        )

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
            if self.config.max_concurrent_orders is not None:
                if len(self.open_order_ids) + len(self.close_order_ids) >= self.config.max_concurrent_orders:
                    break
            await self._place_open_for_level(idx)
            await asyncio.sleep(0.05)

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

        # idle loop; work is done by callbacks
        while not self.shutdown_requested:
            await asyncio.sleep(5)

    async def graceful_shutdown(self, reason: str = "Unknown"):
        self.logger.log(f"[GRID] shutdown: {reason}", "INFO")
        self.shutdown_requested = True
        try:
            await self.exchange_client.disconnect()
        except Exception as e:
            self.logger.log(f"[GRID] disconnect error: {e}", "ERROR")
