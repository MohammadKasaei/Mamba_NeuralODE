import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.tiny_shakespeare import download

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data_cache/tiny_shakespeare.txt")
    args = parser.parse_args()
    print(download(args.path))
