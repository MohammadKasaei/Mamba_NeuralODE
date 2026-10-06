import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.analysis.plotting import aggregate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results")
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = aggregate(args.root, args.output)
    print(f"Aggregated {len(rows)} runs")
