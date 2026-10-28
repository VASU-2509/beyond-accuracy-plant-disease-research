import argparse
from research.src.utils import load_config
from research.scripts.prepare_data import prepare
from research.src.train import train
from research.scripts.evaluate_all import evaluate_all

def main():
    parser = argparse.ArgumentParser(description="Prepare data, train, calibrate and evaluate both models offline")
    parser.add_argument("--config", default="research/configs/quick_test.yaml")
    cfg = load_config(parser.parse_args().config)
    prepare(cfg)
    for name in cfg["models"]:
        train(cfg, name)
    evaluate_all(cfg)

if __name__ == "__main__":
    main()
