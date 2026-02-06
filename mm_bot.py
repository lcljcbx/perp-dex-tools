"""
Market Maker Bot

Implements a market making strategy that:
1. Places buy/sell orders around the mid price based on configured spread and step.
2. Rebalances orders when price moves beyond a threshold.
3. Monitors for fills/positions and triggers an emergency close mechanism:
   - Cancel all orders
   - Market close position
   - Sleep for a duration
   - Restart
"""

import asyncio
import time
import traceback
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, List

from exchanges import ExchangeFactory
from helpers import TradingLogger


@dataclass
class MarketMakerConfig:
    exchange: str
    ticker: str
    contract_id: str
    tick_size: Decimal
    
    # Strategy parameters
    upper_price: Decimal
    lower_price: Decimal
    price_step: Decimal       # Absolute price step between levels
    grid_count: int           # Number of orders per side
    price_spread: Decimal     # Percentage spread from mid price for the first order (e.g. 0.0008)
    order_quantity: Decimal   # Quantity per order
    
    # Rebalance parameters
    rebalance_threshold: Decimal = Decimal("0.001") # Price change threshold to trigger rebalance (0.1%)
    
    # Safety parameters
    emergency_sleep_time: int = 600  # Seconds to sleep after emergency close

    @property
    def quantity(self) -> Decimal:
        """Alias for order_quantity to be compatible with ExchangeClient expectation."""
        return self.order_quantity

    @property
    def close_order_side(self) -> str:
        """
        Market Maker is bi-directional, so there is no fixed 'close' side.
        Return a dummy value so that ExtendedClient doesn't crash.
        All orders will be treated as 'OPEN' type by default in the client logic.
        """
        return "UNKNOWN"


class MarketMakerBot:
    def __init__(self, config: MarketMakerConfig):
        self.config = config
        self.logger = TradingLogger(config.exchange, config.ticker, log_to_console=True)
        
        # Create exchange client
        self.exchange_client = ExchangeFactory.create_exchange(config.exchange, config)
        
        # Runtime state
        self.shutdown_requested = False
        self.emergency_mode = False
        self.last_center_price = Decimal("0")
        self.active_order_ids: List[str] = []
        
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.position_check_event = asyncio.Event()

    async def _on_order_update(self, data: dict):
        """Callback for order updates from the exchange."""
        status = data.get('status')
        if status in ['FILLED', 'PARTIALLY_FILLED']:
            self.logger.log(f"Order filled: {data.get('order_id')} {status} {data.get('filled_size')} @ {data.get('price')}", "WARNING")
            # Trigger immediate check
            self.emergency_mode = True
            self.position_check_event.set()

    async def run(self):
        """Main entry point."""
        self.loop = asyncio.get_running_loop()
        self.logger.log(f"Starting Market Maker Bot for {self.config.ticker}...", "INFO")
        
        # Connect to exchange
        await self.exchange_client.connect()
        
        # Register callback if supported
        if hasattr(self.exchange_client, "setup_order_update_handler"):
            self.logger.log("Registering order update handler...", "INFO")
            self.exchange_client.setup_order_update_handler(self._on_order_update)
        
        # Get contract details
        self.config.contract_id, self.config.tick_size = await self.exchange_client.get_contract_attributes()
        self.logger.log(f"Contract ID: {self.config.contract_id}, Tick Size: {self.config.tick_size}", "INFO")
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_fills_and_position())
        
        try:
            # Main strategy loop
            while not self.shutdown_requested:
                if self.emergency_mode:
                    await asyncio.sleep(0.1)
                    continue
                
                await self._market_making_logic()
                await asyncio.sleep(1) # Check price every second (or rely on websocket updates if implemented)
                
        except Exception as e:
            self.logger.log(f"Bot crashed: {e}", "ERROR")
            self.logger.log(traceback.format_exc(), "ERROR")
        finally:
            self.shutdown_requested = True
            await self._cancel_all_orders()
            await self.exchange_client.disconnect()
            if not monitor_task.done():
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

    async def _market_making_logic(self):
        """Core logic to check price and update orders."""
        try:
            if self.emergency_mode:
                return

            # 0. Fail-safe: Check position in main loop to prevent fighting with emergency logic
            # This is critical to avoid "New order cost exceeds available balance" errors
            # when a fill happens but monitor hasn't reacted yet.
            position = await self.exchange_client.get_account_positions()
            if abs(position) > 0:
                self.logger.log(f"Position detected in main loop: {position}. Triggering Emergency.", "WARNING")
                await self._execute_emergency_protocol(position)
                return

            # 1. Get current price (Mid Price)
            best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
            if best_bid <= 0 or best_ask <= 0:
                self.logger.log("Waiting for valid BBO...", "WARNING")
                return

            mid_price = (best_bid + best_ask) / 2
            
            # Check if price is within global bounds
            if mid_price < self.config.lower_price or mid_price > self.config.upper_price:
                self.logger.log(f"Price {mid_price} out of bounds [{self.config.lower_price}, {self.config.upper_price}]. Pausing.", "WARNING")
                if self.active_order_ids:
                    await self._cancel_all_orders()
                return

            # 2. Check if we need to rebalance
            # If we have no orders, or price moved significantly
            should_rebalance = False
            if not self.active_order_ids:
                should_rebalance = True
            elif self.last_center_price > 0:
                price_change_pct = abs(mid_price - self.last_center_price) / self.last_center_price
                if price_change_pct > self.config.rebalance_threshold:
                    self.logger.log(f"Price changed by {price_change_pct:.4f} (thresh: {self.config.rebalance_threshold}). Rebalancing.", "INFO")
                    should_rebalance = True
            
            # 3. Rebalance if needed
            if should_rebalance:
                await self._place_grid_orders(mid_price)

        except Exception as e:
            self.logger.log(f"Error in market making logic: {e}", "ERROR")
            self.logger.log(traceback.format_exc(), "ERROR")

    async def _place_grid_orders(self, center_price: Decimal):
        """Cancel existing orders and place new ones around center_price."""
        if self.emergency_mode:
            return

        self.logger.log(f"Placing grid orders around {center_price}...", "INFO")
        
        # 1. Cancel all existing
        await self._cancel_all_orders()
        
        if self.emergency_mode:
            return

        # 2. Calculate levels
        buy_orders = []
        sell_orders = []
        
        # Spread is percentage of price, e.g. 0.0008 * 60000 = 48
        spread_value = center_price * self.config.price_spread
        
        for i in range(self.config.grid_count):
            # Calculate distance: spread + (i * step)
            distance = spread_value + (Decimal(i) * self.config.price_step)
            
            bid_price = center_price - distance
            ask_price = center_price + distance
            
            # Ensure tick size precision
            bid_price = self.exchange_client.round_to_tick(bid_price)
            ask_price = self.exchange_client.round_to_tick(ask_price)
            
            if bid_price > 0:
                buy_orders.append(bid_price)
            if ask_price > 0:
                sell_orders.append(ask_price)

        # 3. Place orders (Batch or Sequential)
        # Note: Ideally this should be done in parallel for speed
        tasks = []
        
        # Place Buys
        for p in buy_orders:
            if self.emergency_mode: break
            tasks.append(self.exchange_client.place_limit_order(
                self.config.contract_id,
                self.config.order_quantity,
                p,
                "buy",
                post_only=True
            ))
            
        # Place Sells
        for p in sell_orders:
            if self.emergency_mode: break
            tasks.append(self.exchange_client.place_limit_order(
                self.config.contract_id,
                self.config.order_quantity,
                p,
                "sell",
                post_only=True
            ))
            
        if self.emergency_mode:
            self.logger.log("Emergency mode detected during order placement. Aborting.", "WARNING")
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = 0
        for res in results:
            if isinstance(res, Exception):
                self.logger.log(f"Order placement failed with exception: {res}", "ERROR")
            elif res.success:
                self.active_order_ids.append(res.order_id)
                success_count += 1
            else:
                self.logger.log(f"Order placement failed: {res.error_message}", "WARNING")
                
        self.last_center_price = center_price
        self.logger.log(f"Placed {success_count} orders. Center: {center_price}", "INFO")

    async def _cancel_all_orders(self):
        """Cancel all tracked orders."""
        if not self.active_order_ids:
            return
            
        self.logger.log(f"Cancelling {len(self.active_order_ids)} active orders...", "INFO")
        tasks = [self.exchange_client.cancel_order(oid) for oid in self.active_order_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
        self.active_order_ids = []

    async def _monitor_fills_and_position(self):
        """Background task to monitor position and trigger emergency close."""
        self.logger.log("Position monitor started.", "INFO")
        while not self.shutdown_requested:
            try:
                # Wait for trigger or timeout
                try:
                    await asyncio.wait_for(self.position_check_event.wait(), timeout=1.0)
                    self.position_check_event.clear()
                except asyncio.TimeoutError:
                    pass # Check periodically anyway
                
                if self.shutdown_requested:
                    break

                # Check position
                # Note: get_account_positions usually returns a Decimal size
                position = await self.exchange_client.get_account_positions()
                
                # If position is non-zero (or larger than a tiny dust amount), trigger emergency
                if abs(position) > 0:
                    self.logger.log(f"Emergency Triggered! Position detected: {position}", "WARNING")
                    await self._execute_emergency_protocol(position)
                
            except Exception as e:
                self.logger.log(f"Monitor error: {e}", "ERROR")
                await asyncio.sleep(1)

    async def _execute_emergency_protocol(self, current_position: Decimal):
        """Execute the emergency close sequence."""
        self.emergency_mode = True
        
        # 1. Cancel all orders immediately
        self.logger.log("[EMERGENCY] Cancelling all orders...", "WARNING")
        await self._cancel_all_orders()
        
        # 2. Close position (Market Order)
        # If position is positive (Long), we need to SELL.
        # If position is negative (Short), we need to BUY.
        side = "sell" if current_position > 0 else "buy"
        qty = abs(current_position)
        
        self.logger.log(f"[EMERGENCY] Market closing position: {side} {qty}", "WARNING")
        
        # Try to close until successful
        max_retries = 5
        for i in range(max_retries):
            # Using place_market_order if available, otherwise simulate with limit
            if hasattr(self.exchange_client, "place_market_order"):
                res = await self.exchange_client.place_market_order(
                    self.config.contract_id,
                    qty,
                    side
                )
            else:
                # Fallback to aggressive limit order if market order not supported
                # Get current price
                best_bid, best_ask = await self.exchange_client.fetch_bbo_prices(self.config.contract_id)
                price = best_ask * Decimal("1.05") if side == "buy" else best_bid * Decimal("0.95")
                res = await self.exchange_client.place_limit_order(
                    self.config.contract_id,
                    qty,
                    price,
                    side,
                    post_only=False
                )
                
            if res.success:
                self.logger.log("[EMERGENCY] Position close order placed successfully.", "INFO")
                break
            else:
                self.logger.log(f"[EMERGENCY] Failed to close position (attempt {i+1}): {res.error_message}", "ERROR")
                await asyncio.sleep(1)
        
        # 3. Sleep
        self.logger.log(f"[EMERGENCY] Sleeping for {self.config.emergency_sleep_time} seconds...", "INFO")
        await asyncio.sleep(self.config.emergency_sleep_time)
        
        # 4. Resume
        self.logger.log("[EMERGENCY] Resuming normal operation.", "INFO")
        self.last_center_price = Decimal("0") # Force rebalance
        self.emergency_mode = False
