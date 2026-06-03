import os
import argparse
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

def main():
    parser = argparse.ArgumentParser(description="Ensemble Kaggle submissions by averaging probabilities.")
    parser.add_argument("csv_files", nargs='+', help="Paths to test_probabilities_ensemble.csv files to combine.")
    parser.add_argument("--output", default="outputs/submission_mega_ensemble.csv", help="Output submission CSV path.")
    
    args = parser.parse_args()
    
    if len(args.csv_files) < 2:
        print("Error: Please provide at least two CSV files to ensemble.")
        sys.exit(1)
        
    print(f"Ensembling {len(args.csv_files)} models...")
    
    # Read the first one to establish the dataframe structure
    base_df = pd.read_csv(args.csv_files[0])
    image_names = base_df["image_name"]
    
    # Accumulator for probabilities
    all_probs = np.zeros((len(base_df), config.NUM_CLASSES), dtype=np.float64)
    
    for csv_file in args.csv_files:
        print(f" -> Reading: {csv_file}")
        df = pd.read_csv(csv_file)
        # Ensure it has the same images in the same order
        assert (df["image_name"] == image_names).all(), f"Image name mismatch in {csv_file}"
        
        # Extract just the probability columns
        probs = df[config.CLASS_NAMES].values
        all_probs += probs
        
    # Average
    all_probs /= len(args.csv_files)
    
    # Get final predictions
    predictions = np.argmax(all_probs, axis=1)
    predicted_labels = [config.IDX_TO_LABEL[p] for p in predictions]
    
    # Create final submission dataframe
    submission = pd.DataFrame({"image_name": image_names, "label": predicted_labels})
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    submission.to_csv(args.output, index=False)
    
    print(f"\n[Success] Mega-Ensemble saved to {args.output}")
    print(f"\n[Predict] Predicted class distribution:")
    for cls_name in config.CLASS_NAMES:
        count = (submission["label"] == cls_name).sum()
        pct = 100.0 * count / len(submission)
        bar = "█" * int(pct / 2)
        print(f"  {cls_name:12s}: {count:4d} ({pct:5.1f}%) {bar}")

if __name__ == "__main__":
    main()
