"""
Check label distribution in all processed CSV files.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

def check_csv_labels(csv_path):
    """Check label distribution in a single CSV file."""
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Find label column
        label_col = next((c for c in df.columns if c in {"sentiment", "label", "sentiment_label", "target"}), None)
        
        if label_col is None:
            return None, "No label column found"
        
        # Get label distribution
        label_counts = df[label_col].value_counts()
        return label_counts, None
        
    except Exception as e:
        return None, str(e)


def main():
    print("="*70)
    print("LABEL DISTRIBUTION ANALYSIS")
    print("="*70)
    print(f"\nChecking CSV files in: {DATA_DIR.resolve()}\n")
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found!")
        return
    
    for csv_file in sorted(csv_files):
        print(f"\n{'─'*70}")
        print(f"File: {csv_file.name}")
        print(f"{'─'*70}")
        
        label_counts, error = check_csv_labels(csv_file)
        
        if error:
            print(f"  ⚠ {error}")
        else:
            print(f"  Total rows: {label_counts.sum()}")
            print(f"\n  Label distribution:")
            for label, count in label_counts.items():
                percentage = (count / label_counts.sum()) * 100
                print(f"    {str(label):20s} : {count:6d}  ({percentage:5.1f}%)")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
