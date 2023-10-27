import argparse
import numpy as np

from config import yaml_config_hook
from runner import runner

if __name__ == '__main__':
    for i in range(15):
        parser = argparse.ArgumentParser(description='SLIDE')
        config = yaml_config_hook('config.yaml')
        for k, v in config.items():
            parser.add_argument(f"--{k}", default=v, type=type(v))

        args = parser.parse_args()
        args.sample = i
        runner(args)
