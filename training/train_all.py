from __future__ import annotations

import argparse
from pathlib import Path

from training.common import optional_int
from training.train_classifier import train_classifier
from training.train_regressor import train_regressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain classifier and conditional regressor.")
    parser.add_argument("--project-dir", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--metrics-dir", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--classifier-n-estimators", type=int, default=500)
    parser.add_argument("--classifier-max-depth", type=optional_int, default=6)
    parser.add_argument("--regressor-n-estimators", type=int, default=500)
    parser.add_argument("--regressor-max-depth", type=optional_int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classifier_result = train_classifier(
        project_dir=args.project_dir,
        data_path=args.data_path,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.classifier_n_estimators,
        max_depth=args.classifier_max_depth,
    )
    regressor_result = train_regressor(
        project_dir=args.project_dir,
        data_path=args.data_path,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        classifier_model=classifier_result.model,
        classifier_metadata=classifier_result.metadata,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.regressor_n_estimators,
        max_depth=args.regressor_max_depth,
    )

    print("Retraining complete.")
    print("Classifier:", classifier_result.model_path)
    print("Regressor:", regressor_result.model_path)
    print("Classifier metrics:")
    print(classifier_result.metrics.round(4).to_string(index=False))
    print("Regressor metrics:")
    print(regressor_result.metrics.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

