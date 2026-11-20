"""Command-line entry point.

    python -m fraud_detection eda        # statistics + EDA figures
    python -m fraud_detection compare    # imbalance-strategy comparison (5-fold CV)
    python -m fraud_detection tune       # randomized hyper-parameter search
    python -m fraud_detection train      # threshold selection, final fit, test evaluation
    python -m fraud_detection baselines  # benchmark models on the same split
    python -m fraud_detection report     # build the PDF report
    python -m fraud_detection all        # every stage in order
    python -m fraud_detection predict --input new.csv --output scored.csv
"""
from __future__ import annotations

import argparse
import importlib
import logging
import time

STAGES = {
    "eda": "fraud_detection.eda",
    "compare": "fraud_detection.imbalance",
    "tune": "fraud_detection.tuning",
    "train": "fraud_detection.train",
    "baselines": "fraud_detection.baselines",
    "report": "fraud_detection.report",
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fraud_detection", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[*STAGES, "all", "predict"])
    parser.add_argument("--input", help="CSV to score (predict stage)")
    parser.add_argument("--output", help="Where to write scores (predict stage)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("fraud_detection")

    if args.stage == "predict":
        from .predict import score_file
        score_file(args.input, args.output)
        return

    order = list(STAGES) if args.stage == "all" else [args.stage]
    for stage in order:
        t0 = time.perf_counter()
        log.info("=== stage: %s ===", stage)
        importlib.import_module(STAGES[stage]).run()
        log.info("=== %s finished in %.1f s ===", stage, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
