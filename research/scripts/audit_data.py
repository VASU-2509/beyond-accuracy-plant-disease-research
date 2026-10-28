import argparse
from pathlib import Path
from research.src.data_audit import audit
from research.src.utils import save_json
from research.src.plots import distribution
from research.storage import paths, managed_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=paths()["artifacts_dir"] / "reports/manual_audit.json")
    args = parser.parse_args()
    args.output = managed_path(args.output)
    report = audit(args.directory)
    save_json(args.output, report)
    distribution(report["counts"], args.output.with_suffix(".png"))
    print(f"{report['count']} readable, {len(report['corrupt'])} corrupt; report: {args.output}")

if __name__ == "__main__":
    main()
