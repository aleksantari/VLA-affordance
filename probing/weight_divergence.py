"""Weight divergence analysis: compare encoder weights pre/post VLA training.

Critical question: Does pi0 actually update the SigLIP weights during training,
or does it freeze them? Verify by:
1. Comparing raw SigLIP weights vs pi0-extracted weights (L2 distance)
2. Checking per-layer relative change
3. If weights are identical -> SigLIP was frozen during pi0 training (important finding)
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_weight_divergence(model_a, model_b):
    """Compare weights between two models (e.g., raw SigLIP vs pi0 SigLIP).

    Returns per-layer L2 distances and cosine similarities.

    Args:
        model_a: first model (e.g., raw SigLIP)
        model_b: second model (e.g., pi0's SigLIP)

    Returns:
        dict mapping param_name -> {l2_distance, cosine_similarity, relative_change, param_count}
    """
    results = {}
    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    for name in params_a:
        if name in params_b:
            a = params_a[name].data.float()
            b = params_b[name].data.float()
            l2_dist = (a - b).norm().item()
            cosine_sim = F.cosine_similarity(
                a.flatten().unsqueeze(0),
                b.flatten().unsqueeze(0)
            ).item()
            relative_change = l2_dist / (a.norm().item() + 1e-8)
            results[name] = {
                'l2_distance': l2_dist,
                'cosine_similarity': cosine_sim,
                'relative_change': relative_change,
                'param_count': a.numel()
            }

    return results


def summarize_divergence(results):
    """Summarize weight divergence results.

    Args:
        results: output of compute_weight_divergence

    Returns:
        dict with summary statistics
    """
    if not results:
        return {"error": "No matching parameters found"}

    l2_distances = [r['l2_distance'] for r in results.values()]
    cosine_sims = [r['cosine_similarity'] for r in results.values()]
    relative_changes = [r['relative_change'] for r in results.values()]
    total_params = sum(r['param_count'] for r in results.values())

    # Weighted average by parameter count
    weighted_l2 = sum(
        r['l2_distance'] * r['param_count'] for r in results.values()
    ) / total_params

    summary = {
        'total_l2_distance': sum(l2_distances),
        'mean_l2_distance': np.mean(l2_distances),
        'max_l2_distance': max(l2_distances),
        'weighted_l2_distance': weighted_l2,
        'mean_cosine_similarity': np.mean(cosine_sims),
        'min_cosine_similarity': min(cosine_sims),
        'mean_relative_change': np.mean(relative_changes),
        'max_relative_change': max(relative_changes),
        'total_parameters': total_params,
        'num_layers': len(results),
        'weights_identical': all(r['l2_distance'] < 1e-6 for r in results.values()),
    }

    # Find most changed layers
    sorted_layers = sorted(results.items(), key=lambda x: x[1]['relative_change'], reverse=True)
    summary['most_changed_layers'] = [
        (name, data['relative_change']) for name, data in sorted_layers[:10]
    ]

    return summary


def print_divergence_report(model_a, model_b, name_a="Model A", name_b="Model B"):
    """Print a formatted divergence report."""
    results = compute_weight_divergence(model_a, model_b)
    summary = summarize_divergence(results)

    print(f"\n{'='*60}")
    print(f"Weight Divergence: {name_a} vs {name_b}")
    print(f"{'='*60}")
    print(f"Total parameters compared: {summary['total_parameters']:,}")
    print(f"Number of layers: {summary['num_layers']}")
    print(f"Weights identical: {summary['weights_identical']}")
    print(f"\nL2 Distance:")
    print(f"  Total: {summary['total_l2_distance']:.6f}")
    print(f"  Mean:  {summary['mean_l2_distance']:.6f}")
    print(f"  Max:   {summary['max_l2_distance']:.6f}")
    print(f"\nCosine Similarity:")
    print(f"  Mean: {summary['mean_cosine_similarity']:.6f}")
    print(f"  Min:  {summary['min_cosine_similarity']:.6f}")
    print(f"\nRelative Change:")
    print(f"  Mean: {summary['mean_relative_change']:.6f}")
    print(f"  Max:  {summary['max_relative_change']:.6f}")
    print(f"\nTop 10 Most Changed Layers:")
    for name, change in summary['most_changed_layers']:
        print(f"  {change:.6f}  {name}")
    print(f"{'='*60}\n")

    return results, summary
