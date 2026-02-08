#!/usr/bin/env python3
"""
Modular Trading Bot - Supports multiple exchanges
"""

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import dotenv
from decimal import Decimal
from trading_bot import TradingBot, TradingConfig
from exchanges import ExchangeFactory
from grid_bot import GridBot, GridConfig
from mm_bot import MarketMakerBot, MarketMakerConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Modular Trading Bot - Supports multiple exchanges')

    # Exchange selection
    parser.add_argument('--exchange', type=str, default='edgex',
                        choices=ExchangeFactory.get_supported_exchanges(),
                        help='Exchange to use (default: edgex). '
                             f'Available: {", ".join(ExchangeFactory.get_supported_exchanges())}')

    # Trading parameters
    parser.add_argument('--ticker', type=str, default='ETH',
                        help='Ticker (default: ETH)')
    parser.add_argument('--quantity', type=Decimal, default=Decimal(0.1),
                        help='Order quantity (default: 0.1)')
    parser.add_argument('--take-profit', type=Decimal, default=Decimal(0.02),
                        help='Take profit in USDT (default: 0.02)')
    parser.add_argument('--direction', type=str, default='buy', choices=['buy', 'sell'],
                        help='Direction of the bot (default: buy)')
    parser.add_argument('--max-orders', type=int, default=40,
                        help='Maximum number of active orders (default: 40)')
    parser.add_argument('--wait-time', type=int, default=450,
                        help='Wait time between orders in seconds (default: 450)')
    parser.add_argument('--env-file', type=str, default=".env",
                        help=".env file path (default: .env)")
    parser.add_argument('--grid-step', type=str, default='-100',
                        help='The minimum distance in percentage to the next close order price (default: -100)')
    parser.add_argument('--stop-price', type=Decimal, default=-1,
                        help='Price to stop trading and exit. Buy: exits if price >= stop-price.'
                        'Sell: exits if price <= stop-price. (default: -1, no stop)')
    parser.add_argument('--pause-price', type=Decimal, default=-1,
                        help='Pause trading and wait. Buy: pause if price >= pause-price.'
                        'Sell: pause if price <= pause-price. (default: -1, no pause)')
    parser.add_argument('--boost', action='store_true',
                        help='Use the Boost mode for volume boosting')

    # Strategy selection
    parser.add_argument('--strategy', type=str, default='maker_tp', choices=['maker_tp', 'grid', 'mm'],
                        help="Strategy to run: 'maker_tp' (default), 'grid' or 'mm'")

    # Grid strategy parameters (used when --strategy grid)
    parser.add_argument('--grid-direction', type=str, default='long', choices=['long', 'short'],
                        help="Grid direction (default: long)")
    parser.add_argument('--grid-spacing', type=str, default='arith', choices=['arith', 'geo'],
                        help="Grid spacing mode: arith (equal difference) or geo (equal ratio) (default: arith)")
    parser.add_argument('--grid-lower', type=Decimal, default=Decimal('-1'),
                        help="Grid lower price bound (required for grid strategy)")
    parser.add_argument('--grid-upper', type=Decimal, default=Decimal('-1'),
                        help="Grid upper price bound (required for grid strategy)")
    parser.add_argument('--grid-grids', type=int, default=10,
                        help="Number of grid levels (default: 10)")
    parser.add_argument('--grid-size', type=Decimal, default=Decimal('-1'),
                        help="Per-grid order size (required for grid strategy)")
    parser.add_argument('--grid-tp', type=Decimal, default=Decimal('-1'),
                        help="Grid Take Profit Price (default: -1, no TP)")
    parser.add_argument('--grid-sl', type=Decimal, default=Decimal('-1'),
                        help="Grid Stop Loss Price (default: -1, no SL)")
    
    # Market Maker strategy parameters (used when --strategy mm)
    parser.add_argument('--mm-upper', type=Decimal, default=Decimal('-1'),
                        help="MM: Upper price bound")
    parser.add_argument('--mm-lower', type=Decimal, default=Decimal('-1'),
                        help="MM: Lower price bound")
    parser.add_argument('--mm-step', type=Decimal, default=Decimal('5'),
                        help="MM: Price step between levels")
    parser.add_argument('--mm-grids', type=int, default=5,
                        help="MM: Number of orders per side")
    parser.add_argument('--mm-spread', type=Decimal, default=Decimal('0.0008'),
                        help="MM: Spread from mid price (e.g. 0.0008 for 0.08%%)")
    parser.add_argument('--mm-quantity', type=Decimal, default=Decimal('0.002'),
                        help="MM: Order quantity per level")

    return parser.parse_args()


def setup_logging(log_level: str):
    """Setup global logging configuration."""
    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Clear any existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure root logger WITHOUT adding a console handler
    # This prevents duplicate logs when TradingLogger adds its own console handler
    root_logger.setLevel(level)

    # Suppress websockets debug logs unless DEBUG level is explicitly requested
    if log_level.upper() != 'DEBUG':
        logging.getLogger('websockets').setLevel(logging.WARNING)

    # Suppress other noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    # Suppress Lighter SDK debug logs
    logging.getLogger('lighter').setLevel(logging.WARNING)
    # Also suppress any root logger DEBUG messages that might be coming from Lighter
    if log_level.upper() != 'DEBUG':
        # Set root logger to WARNING to suppress DEBUG messages from Lighter SDK
        root_logger.setLevel(logging.WARNING)


async def main():
    """Main entry point."""
    args = parse_arguments()
    print("args:")
    print(args)
    print('*' * 50)

    # Setup logging first
    setup_logging("WARNING")

    # Validate boost-mode can only be used with aster and backpack exchange
    if args.boost and args.exchange.lower() != 'aster' and args.exchange.lower() != 'backpack':
        print(f"Error: --boost can only be used when --exchange is 'aster' or 'backpack'. "
              f"Current exchange: {args.exchange}")
        sys.exit(1)

    env_path = Path(args.env_file)
    if not env_path.exists():
        print(f"Env file not find: {env_path.resolve()}")
        sys.exit(1)
    dotenv.load_dotenv(args.env_file)

    # Create configuration
    if args.strategy == 'grid':
        if args.grid_lower <= 0 or args.grid_upper <= 0:
            raise ValueError("--grid-lower and --grid-upper must be set (> 0) when --strategy grid")
        if args.grid_size <= 0:
            raise ValueError("--grid-size must be set (> 0) when --strategy grid")

        grid_config = GridConfig(
            exchange=args.exchange.lower(),
            ticker=args.ticker.upper(),
            quantity=args.quantity,
            contract_id='',  # will be set in bot.run()
            tick_size=Decimal(0),
            direction=args.grid_direction,
            spacing=args.grid_spacing,
            lower=args.grid_lower,
            upper=args.grid_upper,
            grids=args.grid_grids,
            grid_size=args.grid_size,
            take_profit_price=args.grid_tp if args.grid_tp > 0 else None,
            stop_loss_price=args.grid_sl if args.grid_sl > 0 else None,
        )

        bot = GridBot(grid_config)
        await bot.run()
    elif args.strategy == 'mm':
        if args.mm_lower <= 0 or args.mm_upper <= 0:
            raise ValueError("--mm-lower and --mm-upper must be set (> 0) when --strategy mm")
        
        mm_config = MarketMakerConfig(
            exchange=args.exchange.lower(),
            ticker=args.ticker.upper(),
            contract_id='', # will be set in bot.run()
            tick_size=Decimal(0),
            upper_price=args.mm_upper,
            lower_price=args.mm_lower,
            price_step=args.mm_step,
            grid_count=args.mm_grids,
            price_spread=args.mm_spread,
            order_quantity=args.mm_quantity
        )
        
        bot = MarketMakerBot(mm_config)
        await bot.run()
    else:
        config = TradingConfig(
            ticker=args.ticker.upper(),
            contract_id='',  # will be set in the bot's run method
            tick_size=Decimal(0),
            quantity=args.quantity,
            take_profit=args.take_profit,
            direction=args.direction.lower(),
            max_orders=args.max_orders,
            wait_time=args.wait_time,
            exchange=args.exchange.lower(),
            grid_step=Decimal(args.grid_step),
            stop_price=Decimal(args.stop_price),
            pause_price=Decimal(args.pause_price),
            boost_mode=args.boost
        )

        bot = TradingBot(config)
        try:
            await bot.run()
        except Exception as e:
            print(f"Bot execution failed: {e}")
            return


if __name__ == "__main__":
    asyncio.run(main())
