import argparse
from research.src.utils import load_config
from research.src.train import train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/configs/quick_test.yaml")
    cfg = load_config(parser.parse_args().config)
    for name in cfg["models"]:
        train(cfg, name)

if __name__ == "__main__":
    main()
