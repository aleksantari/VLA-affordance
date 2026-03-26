"""Step 6: Compare raw vs fine-tuned SigLIP weights."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
from pathlib import Path

from encoders.feature_extractor import UnifiedFeatureExtractor
from probing.weight_divergence import print_divergence_report
from evaluation.visualization import plot_weight_divergence


def run_weight_divergence(device="cuda", figures_dir="./results/figures",
                          tables_dir="./results/tables"):
    """Compare raw SigLIP weights against pi0 and pi0.5 extracted weights."""
    figures_dir = Path(figures_dir)
    tables_dir = Path(tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load raw SigLIP
    print("Loading raw SigLIP...")
    raw_extractor = UnifiedFeatureExtractor("raw_siglip", device=device)
    raw_model = raw_extractor.get_raw_model()

    comparisons = {
        "pi0_siglip": "pi0 SigLIP",
        "pi05_siglip": "pi0.5 SigLIP",
    }

    all_summaries = {}

    for encoder_name, display_name in comparisons.items():
        print(f"\nComparing raw SigLIP vs {display_name}...")
        try:
            ft_extractor = UnifiedFeatureExtractor(encoder_name, device=device)
            ft_model = ft_extractor.get_raw_model()

            results, summary = print_divergence_report(
                raw_model, ft_model,
                name_a="Raw SigLIP",
                name_b=display_name,
            )

            # Plot
            plot_weight_divergence(
                results,
                save_path=figures_dir / f"weight_divergence_{encoder_name}.png"
            )

            # Convert summary for JSON serialization
            json_summary = {}
            for k, v in summary.items():
                if k == 'most_changed_layers':
                    json_summary[k] = [(name, float(change)) for name, change in v]
                elif isinstance(v, (int, float, bool)):
                    json_summary[k] = v
                else:
                    json_summary[k] = str(v)

            all_summaries[encoder_name] = json_summary

            del ft_extractor
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            all_summaries[encoder_name] = {"error": str(e)}

    # Save results
    with open(tables_dir / "weight_divergence_results.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    del raw_extractor
    torch.cuda.empty_cache()

    return all_summaries


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_weight_divergence(device=device)
