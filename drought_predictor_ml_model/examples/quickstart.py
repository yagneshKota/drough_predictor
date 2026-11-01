"""
Quickstart Example: Load models and generate multi-horizon drought predictions.
"""

import sys
import json
from pathlib import Path

# Add package root to sys.path
pkg_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pkg_root))

from src.inference import predict_drought

def main():
    sample_file = pkg_root / "examples" / "sample_input.json"
    print(f"[QUICKSTART] Loading sample observation from: {sample_file}")
    with open(sample_file, "r", encoding="utf-8") as f:
        sample_input = json.load(f)
        
    print(f"[QUICKSTART] Target District: {sample_input.get('district')} ({sample_input.get('state')}) on {sample_input.get('date')}")
    print("[QUICKSTART] Running 7-day, 15-day, and 30-day drought prediction...\n")
    
    result = predict_drought(sample_input)
    
    print("=" * 65)
    print("DROUGHT PREDICTION RESULTS")
    print("=" * 65)
    for horizon, pred in result["predictions"].items():
        print(f"\nHorizon: {pred['horizon_label']}")
        print(f"  Predicted Severity: Class {pred['predicted_class']} ({pred['predicted_label']})")
        print(f"  Confidence:         {pred['confidence'] * 100:.2f}%")
        print(f"  Condition:          {pred['description']}")
        print(f"  Action Advisory:    {pred['advisory']}")
        print("  Class Probabilities:")
        for c_name, p_val in pred["probabilities"].items():
            print(f"    - {c_name:<25}: {p_val * 100:.2f}%")
    print("=" * 65)

if __name__ == "__main__":
    main()
