import argparse
from decimal import Decimal

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--strategy', type=str, default='maker_tp',
                        choices=['maker_tp', 'grid'])

    parser.add_argument('--grid-lower', type=str, default='-1')

    return parser.parse_args()

args = parse_arguments()

print(args.strategy)
print(args.grid_lower)
print(Decimal(args.grid_lower))

