"""
Verb-Spatial Binding Evaluation Metrics

Evaluates how well Flux cross-attention maps align with ground truth
affordance heatmaps using standard saliency metrics:
- KLD (Kullback-Leibler Divergence) — lower is better
- SIM (Similarity / Histogram Intersection) — higher is better  
- NSS (Normalized Scanpath Saliency) — higher is better

Reference:
    Bylinskii et al., "What do different evaluation metrics tell us
    about saliency models?", TPAMI 2019.
    
    Zhang et al., "Probing and Bridging Geometry-Interaction Cues
    for Affordance Reasoning in Vision Foundation Models", CVPR 2026.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BindingMetrics:
    """Evaluation metrics for a single prediction."""
    kld: float       # KL Divergence — lower is better
    sim: float       # Similarity — higher is better
    nss: float       # Normalized Scanpath Saliency — higher is better
    affordance: str = ""
    prompt: str = ""


def normalize_to_distribution(heatmap: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Normalize a heatmap to a probability distribution (sum to 1)."""
    heatmap = np.clip(heatmap, 0, None)  # Ensure non-negative
    total = heatmap.sum()
    if total > eps:
        return heatmap / total
    else:
        # Uniform distribution as fallback
        return np.ones_like(heatmap) / heatmap.size


def compute_kld(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """
    Kullback-Leibler Divergence.
    
    Measures information loss when pred_map is used to approximate gt_map.
    Lower is better (0 = perfect match).
    
    Both maps are normalized to sum to 1.
    
    Args:
        pred_map: Predicted attention heatmap (any shape)
        gt_map: Ground truth affordance heatmap (same shape)
        eps: Small constant to avoid log(0)
        
    Returns:
        KLD value (non-negative float)
    """
    pred = normalize_to_distribution(pred_map.flatten(), eps)
    gt = normalize_to_distribution(gt_map.flatten(), eps)
    
    return float(np.sum(gt * np.log((gt + eps) / (pred + eps))))


def compute_sim(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """
    Similarity (Histogram Intersection).
    
    Measures overlap between two normalized distributions.
    Higher is better (1 = identical distributions, 0 = no overlap).
    
    Both maps are normalized to sum to 1.
    
    Args:
        pred_map: Predicted attention heatmap
        gt_map: Ground truth affordance heatmap
        eps: Small constant for normalization stability
        
    Returns:
        SIM value in [0, 1]
    """
    pred = normalize_to_distribution(pred_map.flatten(), eps)
    gt = normalize_to_distribution(gt_map.flatten(), eps)
    
    return float(np.sum(np.minimum(pred, gt)))


def compute_nss(
    pred_map: np.ndarray,
    gt_fixations: np.ndarray,
    eps: float = 1e-7,
) -> float:
    """
    Normalized Scanpath Saliency.
    
    Evaluates how high the predicted saliency is at ground truth
    fixation locations. The predicted map is z-score normalized first.
    Higher is better.
    
    Args:
        pred_map: Predicted attention heatmap (any shape)
        gt_fixations: Binary fixation map (same shape). For affordance,
                     this is a binarized version of the GT heatmap
                     (e.g., top 20% pixels = 1, rest = 0).
        eps: Small constant for std stability
        
    Returns:
        NSS value (unbounded, typically 0-5 for good models)
    """
    pred = pred_map.flatten().astype(np.float64)
    fixations = gt_fixations.flatten().astype(np.float64)
    
    # Z-score normalize the predicted map
    mu = pred.mean()
    sigma = pred.std()
    if sigma < eps:
        return 0.0
    normed = (pred - mu) / (sigma + eps)
    
    # Average normalized saliency at fixation locations
    fixation_mask = fixations > 0
    if fixation_mask.sum() == 0:
        return 0.0
    
    return float(normed[fixation_mask].mean())


def heatmap_to_fixations(
    heatmap: np.ndarray,
    threshold_percentile: float = 80.0,
) -> np.ndarray:
    """
    Convert a continuous heatmap to a binary fixation map.
    
    Used to create binary GT for NSS computation from continuous
    affordance heatmaps.
    
    Args:
        heatmap: Continuous heatmap (any shape)
        threshold_percentile: Percentile threshold for binarization.
                             Top (100 - threshold_percentile)% pixels become 1.
                             
    Returns:
        Binary fixation map (same shape, dtype bool)
    """
    threshold = np.percentile(heatmap, threshold_percentile)
    return (heatmap > threshold).astype(np.float64)


def evaluate_single(
    pred_map: np.ndarray,
    gt_heatmap: np.ndarray,
    affordance: str = "",
    prompt: str = "",
    nss_threshold_percentile: float = 80.0,
) -> BindingMetrics:
    """
    Compute all metrics for a single prediction-GT pair.
    
    Args:
        pred_map: Predicted attention heatmap from Flux
        gt_heatmap: Ground truth affordance heatmap from AGD20K
        affordance: Affordance category name (for logging)
        prompt: Prompt used (for logging)
        nss_threshold_percentile: Percentile for NSS fixation binarization
        
    Returns:
        BindingMetrics with KLD, SIM, and NSS
    """
    # Resize pred_map to match gt_heatmap if needed
    if pred_map.shape != gt_heatmap.shape:
        from PIL import Image
        pred_pil = Image.fromarray(pred_map.astype(np.float32), mode='F')
        pred_pil = pred_pil.resize(
            (gt_heatmap.shape[1], gt_heatmap.shape[0]),
            Image.BILINEAR,
        )
        pred_map = np.array(pred_pil)
    
    # Compute metrics
    kld = compute_kld(pred_map, gt_heatmap)
    sim = compute_sim(pred_map, gt_heatmap)
    
    gt_fixations = heatmap_to_fixations(gt_heatmap, nss_threshold_percentile)
    nss = compute_nss(pred_map, gt_fixations)
    
    return BindingMetrics(
        kld=kld,
        sim=sim,
        nss=nss,
        affordance=affordance,
        prompt=prompt,
    )


def evaluate_verb_spatial_binding(
    pred_maps: Dict[str, np.ndarray],
    gt_heatmaps: Dict[str, np.ndarray],
    affordances: Dict[str, str],
    prompts: Optional[Dict[str, str]] = None,
) -> Dict[str, BindingMetrics]:
    """
    Evaluate verb-spatial binding for a collection of predictions.
    
    Args:
        pred_maps: Dict mapping sample_id -> predicted attention heatmap
        gt_heatmaps: Dict mapping sample_id -> GT affordance heatmap
        affordances: Dict mapping sample_id -> affordance category
        prompts: Optional dict mapping sample_id -> prompt text
        
    Returns:
        Dict mapping sample_id -> BindingMetrics
    """
    results = {}
    
    for sample_id in pred_maps:
        if sample_id not in gt_heatmaps:
            continue
        
        prompt = prompts.get(sample_id, "") if prompts else ""
        
        results[sample_id] = evaluate_single(
            pred_map=pred_maps[sample_id],
            gt_heatmap=gt_heatmaps[sample_id],
            affordance=affordances.get(sample_id, ""),
            prompt=prompt,
        )
    
    return results


def aggregate_metrics_by_affordance(
    results: Dict[str, BindingMetrics],
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics per affordance category.
    
    Returns:
        Dict mapping affordance_name -> {
            "kld_mean": float, "kld_std": float,
            "sim_mean": float, "sim_std": float,
            "nss_mean": float, "nss_std": float,
            "count": int,
        }
    """
    from collections import defaultdict
    
    by_affordance = defaultdict(list)
    for sample_id, metrics in results.items():
        by_affordance[metrics.affordance].append(metrics)
    
    aggregated = {}
    for affordance, metrics_list in sorted(by_affordance.items()):
        klds = [m.kld for m in metrics_list]
        sims = [m.sim for m in metrics_list]
        nsss = [m.nss for m in metrics_list]
        
        aggregated[affordance] = {
            "kld_mean": float(np.mean(klds)),
            "kld_std": float(np.std(klds)),
            "sim_mean": float(np.mean(sims)),
            "sim_std": float(np.std(sims)),
            "nss_mean": float(np.mean(nsss)),
            "nss_std": float(np.std(nsss)),
            "count": len(metrics_list),
        }
    
    # Add overall row
    all_klds = [m.kld for m in results.values()]
    all_sims = [m.sim for m in results.values()]
    all_nsss = [m.nss for m in results.values()]
    
    aggregated["OVERALL"] = {
        "kld_mean": float(np.mean(all_klds)),
        "kld_std": float(np.std(all_klds)),
        "sim_mean": float(np.mean(all_sims)),
        "sim_std": float(np.std(all_sims)),
        "nss_mean": float(np.mean(all_nsss)),
        "nss_std": float(np.std(all_nsss)),
        "count": len(results),
    }
    
    return aggregated


def print_metrics_table(aggregated: Dict[str, Dict[str, float]]):
    """Pretty-print aggregated metrics as a table."""
    print(f"\n{'Affordance':<20} {'Count':>6} {'KLD↓':>10} {'SIM↑':>10} {'NSS↑':>10}")
    print("-" * 60)
    
    for affordance, stats in aggregated.items():
        if affordance == "OVERALL":
            continue
        print(
            f"{affordance:<20} {stats['count']:>6} "
            f"{stats['kld_mean']:>7.3f}±{stats['kld_std']:.2f} "
            f"{stats['sim_mean']:>7.3f}±{stats['sim_std']:.2f} "
            f"{stats['nss_mean']:>7.3f}±{stats['nss_std']:.2f}"
        )
    
    # Overall
    if "OVERALL" in aggregated:
        stats = aggregated["OVERALL"]
        print("-" * 60)
        print(
            f"{'OVERALL':<20} {stats['count']:>6} "
            f"{stats['kld_mean']:>7.3f}±{stats['kld_std']:.2f} "
            f"{stats['sim_mean']:>7.3f}±{stats['sim_std']:.2f} "
            f"{stats['nss_mean']:>7.3f}±{stats['nss_std']:.2f}"
        )
