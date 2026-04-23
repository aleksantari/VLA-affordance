"""
Script 11: Generate Axis 2 Report

Loads cached metrics from script 10 and generates publication-quality
figures and summary tables for the Axis 2 interaction affordance probing.

Usage:
    python scripts/11_generate_axis2_report.py
    python scripts/11_generate_axis2_report.py --results_dir ./results
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_aggregated_metrics(tables_dir: Path) -> dict:
    """Load aggregated metrics from JSON."""
    path = tables_dir / "axis2_metrics.json"
    if not path.exists():
        print(f"ERROR: Metrics file not found: {path}")
        print("Run script 10 first: python scripts/10_run_interaction_probing.py")
        sys.exit(1)
    
    with open(path) as f:
        return json.load(f)


def load_per_sample_metrics(tables_dir: Path) -> list:
    """Load per-sample metrics from CSV."""
    path = tables_dir / "axis2_per_sample.csv"
    if not path.exists():
        return []
    
    samples = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["kld"] = float(row["kld"])
            row["sim"] = float(row["sim"])
            row["nss"] = float(row["nss"])
            samples.append(row)
    return samples


def plot_bar_chart(
    aggregated: dict,
    metric: str,
    title: str,
    ylabel: str,
    save_path: str,
    higher_is_better: bool = True,
):
    """Plot a bar chart of a metric across affordance categories."""
    # Filter out OVERALL for the bar chart body
    categories = [k for k in sorted(aggregated.keys()) if k != "OVERALL"]
    values = [aggregated[k][f"{metric}_mean"] for k in categories]
    errors = [aggregated[k][f"{metric}_std"] for k in categories]
    
    fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.5), 6))
    
    # Color by performance
    colors = []
    if higher_is_better:
        vmin, vmax = min(values), max(values)
    else:
        vmin, vmax = max(values), min(values)  # Reverse for lower-is-better
    
    cmap = plt.cm.RdYlGn
    for v in values:
        if vmax != vmin:
            norm_v = (v - min(values)) / (max(values) - min(values))
            if not higher_is_better:
                norm_v = 1 - norm_v
        else:
            norm_v = 0.5
        colors.append(cmap(norm_v))
    
    bars = ax.bar(range(len(categories)), values, yerr=errors,
                  color=colors, capsize=3, edgecolor='#333', linewidth=0.5)
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Overall line
    if "OVERALL" in aggregated:
        overall = aggregated["OVERALL"][f"{metric}_mean"]
        ax.axhline(y=overall, color='red', linestyle='--', linewidth=1.5,
                   label=f'Overall: {overall:.3f}')
        ax.legend(fontsize=10)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_scatter_kld_sim(
    per_sample: list,
    save_path: str,
):
    """Scatter plot of KLD vs SIM colored by affordance category."""
    if not per_sample:
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Group by affordance
    by_aff = defaultdict(list)
    for s in per_sample:
        by_aff[s["affordance"]].append(s)
    
    cmap = plt.cm.tab20
    for idx, (aff, samples) in enumerate(sorted(by_aff.items())):
        klds = [s["kld"] for s in samples]
        sims = [s["sim"] for s in samples]
        color = cmap(idx / max(len(by_aff), 1))
        ax.scatter(klds, sims, c=[color], label=aff, alpha=0.6, s=20)
    
    ax.set_xlabel("KLD ↓ (lower is better)", fontsize=12)
    ax.set_ylabel("SIM ↑ (higher is better)", fontsize=12)
    ax.set_title("Verb-Spatial Binding: KLD vs SIM", fontsize=14, fontweight='bold')
    
    # Legend outside if too many categories
    if len(by_aff) > 10:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7,
                  ncol=2, markerscale=1.5)
    else:
        ax.legend(fontsize=8, markerscale=1.5)
    
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {save_path}")
    plt.close(fig)


def generate_latex_table(aggregated: dict, save_path: str):
    """Generate a LaTeX table of results."""
    categories = [k for k in sorted(aggregated.keys()) if k != "OVERALL"]
    
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Flux Verb-Spatial Binding: Per-Category Metrics on AGD20K}",
        r"\label{tab:axis2_metrics}",
        r"\begin{tabular}{l c c c c}",
        r"\toprule",
        r"Affordance & Count & KLD $\downarrow$ & SIM $\uparrow$ & NSS $\uparrow$ \\",
        r"\midrule",
    ]
    
    for cat in categories:
        s = aggregated[cat]
        lines.append(
            f"{cat} & {s['count']} & "
            f"{s['kld_mean']:.3f} $\\pm$ {s['kld_std']:.2f} & "
            f"{s['sim_mean']:.3f} $\\pm$ {s['sim_std']:.2f} & "
            f"{s['nss_mean']:.3f} $\\pm$ {s['nss_std']:.2f} \\\\"
        )
    
    if "OVERALL" in aggregated:
        s = aggregated["OVERALL"]
        lines.append(r"\midrule")
        lines.append(
            f"\\textbf{{Overall}} & {s['count']} & "
            f"\\textbf{{{s['kld_mean']:.3f}}} & "
            f"\\textbf{{{s['sim_mean']:.3f}}} & "
            f"\\textbf{{{s['nss_mean']:.3f}}} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    with open(save_path, 'w') as f:
        f.write("\n".join(lines))
    print(f"  Saved LaTeX table: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Axis 2 report")
    parser.add_argument(
        "--results_dir", type=str, default="./results",
        help="Results directory"
    )
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures" / "axis2"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("AXIS 2 — Report Generation")
    print("=" * 60)
    
    # Load metrics
    print("\nLoading metrics...")
    aggregated = load_aggregated_metrics(tables_dir)
    per_sample = load_per_sample_metrics(tables_dir)
    
    overall = aggregated.get("OVERALL", {})
    n_categories = len([k for k in aggregated if k != "OVERALL"])
    print(f"  {n_categories} affordance categories, {overall.get('count', 0)} total samples")
    
    # Generate plots
    print("\nGenerating plots...")
    
    # KLD bar chart
    plot_bar_chart(
        aggregated, "kld",
        title="Flux Cross-Attention: KL Divergence per Affordance",
        ylabel="KLD ↓ (lower is better)",
        save_path=str(figures_dir / "axis2_kld_bar.png"),
        higher_is_better=False,
    )
    
    # SIM bar chart
    plot_bar_chart(
        aggregated, "sim",
        title="Flux Cross-Attention: Similarity per Affordance",
        ylabel="SIM ↑ (higher is better)",
        save_path=str(figures_dir / "axis2_sim_bar.png"),
        higher_is_better=True,
    )
    
    # NSS bar chart
    plot_bar_chart(
        aggregated, "nss",
        title="Flux Cross-Attention: Normalized Scanpath Saliency per Affordance",
        ylabel="NSS ↑ (higher is better)",
        save_path=str(figures_dir / "axis2_nss_bar.png"),
        higher_is_better=True,
    )
    
    # Scatter plot
    plot_scatter_kld_sim(
        per_sample,
        save_path=str(figures_dir / "axis2_kld_sim_scatter.png"),
    )
    
    # LaTeX table
    generate_latex_table(
        aggregated,
        save_path=str(tables_dir / "axis2_table.tex"),
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("REPORT SUMMARY")
    print("=" * 60)
    
    from interaction.verb_spatial_binding import print_metrics_table
    print_metrics_table(aggregated)
    
    print(f"\nOutput files:")
    print(f"  Figures: {figures_dir}/")
    print(f"    - axis2_kld_bar.png")
    print(f"    - axis2_sim_bar.png")
    print(f"    - axis2_nss_bar.png")
    print(f"    - axis2_kld_sim_scatter.png")
    print(f"  Tables: {tables_dir}/")
    print(f"    - axis2_metrics.csv")
    print(f"    - axis2_metrics.json")
    print(f"    - axis2_table.tex")


if __name__ == "__main__":
    main()
