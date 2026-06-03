"""
SargaNet — CLI entry point for the ResNet-50 fine-tuning pipeline.

Usage:
    python run.py --mode train       Train the model (Phase 1 + Phase 2)
    python run.py --mode evaluate    Evaluate best checkpoint on validation set
    python run.py --mode predict     Generate submission CSV with TTA
    python run.py --mode all         Run train → evaluate → predict
"""

import argparse
import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="SargaNet — ResNet-50 Fine-Tuning for Sargassum Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --mode train
  python run.py --mode evaluate
  python run.py --mode predict
  python run.py --mode all
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "evaluate", "predict", "all"],
        help="Pipeline stage to run",
    )

    args = parser.parse_args()

    if args.mode == "train":
        from src.train import train
        train()

    elif args.mode == "evaluate":
        from src.evaluate import evaluate
        evaluate()

    elif args.mode == "predict":
        from src.predict import predict
        predict()

    elif args.mode == "all":
        print("Running full pipeline: train → evaluate → predict\n")

        from src.train import train
        train()

        from src.evaluate import evaluate
        evaluate()

        from src.predict import predict
        predict()

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
