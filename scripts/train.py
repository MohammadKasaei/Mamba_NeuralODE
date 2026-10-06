#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config
from src.training.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    train(load_config(args.config, args.set))
