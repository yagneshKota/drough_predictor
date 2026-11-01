"""
Command-line inference script for Drought Predictor ML Model.

Usage:
    python scripts/predict.py --input examples/sample_input.json
    python scripts/predict.py --input examples/sample_input_batch.json --output results.json
"""

import sys
import json
import argparse
from pathlib import Path

# Add package root to sys.path
pkg_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pkg_root))

from src.inference import predict_drought

def parse_args():
    parser = argparse.ArgumentParser(description="Run 7d, 15d, 30d drought predictions from JSON input.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input JSON file (single record or list of records)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional path to save output JSON")
    return parser.parse_args()

def main():
    args = parse_args()
    input_p = Path(args.input)
    if not input_p.exists():
        print(f"[ERROR] Input file does not exist: {input_p}")
        sys.exit(1)
        
    with open(input_p, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"[PREDICT] Loaded input from: {input_p}")
    results = predict_drought(data)
    
    if args.output:
        out_p = Path(args.output)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[PREDICT] Results saved to: {out_p}")
    else:
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
