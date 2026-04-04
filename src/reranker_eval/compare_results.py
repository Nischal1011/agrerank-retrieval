import json
from pathlib import Path
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("../results"),
                    help="Directory containing the JSON result files.")
    args = ap.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Results directory not found at '{args.results_dir}'")
        return

    all_metrics = {}
    
    result_files = sorted(args.results_dir.glob("results_*.json"))
    
    if not result_files:
        print(f"Error: No result files matching 'results_*.json' found in '{args.results_dir}'")
        return

    for f in result_files:
        method_name = f.stem.replace("results_", "")
        try:
            metrics = json.loads(f.read_text(encoding="utf-8"))
            all_metrics[method_name] = metrics
        except Exception as e:
            print(f"Warning: An error occurred while processing '{f}': {e}. Skipping.")

    if "baseline" not in all_metrics:
        print("\nError: 'results_baseline.json' not found. Cannot compute deltas.")
        return
        
    baseline_metrics = all_metrics["baseline"]

    print("\n\n--- Comparison Table (Value and Delta from Baseline) ---")
    
    data = []
    
    # --- Define a logical order for columns, including new timing metrics at the end ---
    metric_keys = sorted([k for k in baseline_metrics.keys() if not k.endswith('_s') and not k.endswith('_ms')])
    timing_keys = sorted([k for k in baseline_metrics.keys() if k.endswith('_s') or k.endswith('_ms')])
    all_keys = metric_keys + timing_keys
    
    method_order = ["baseline"] + sorted([m for m in all_metrics if m != "baseline"])
    
    for method_name in method_order:
        if method_name not in all_metrics:
            continue
            
        metrics = all_metrics[method_name]
        display_name = "AURA_PLUS" if method_name == "aura_prime_plus" else method_name.upper()
        row = {"Method": display_name}
        
        for key in all_keys:
            base_val = baseline_metrics.get(key, 0)
            method_val = metrics.get(key, 0)
            
            # Custom formatting for timing values
            if key in timing_keys:
                if method_name == "baseline":
                    row[key] = f"{method_val:.2f}"
                else:
                    delta = method_val - base_val
                    row[key] = f"{method_val:.2f} ({delta:+.2f})"
            else: # Standard formatting for performance metrics
                if method_name == "baseline":
                    row[key] = f"{method_val:.4f}"
                else:
                    delta = method_val - base_val
                    row[key] = f"{method_val:.4f} ({delta:+.4f})"
        data.append(row)
        
    df = pd.DataFrame(data)
    df = df[["Method"] + all_keys] 
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()


# python vectorrag/eval/compare_results.py --results-dir data/results_with_timing





