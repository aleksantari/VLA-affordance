# Axis 2: Interaction Affordance Probing
# Cross-attention extraction (Flux + Cosmos) and verb-spatial binding analysis

from .flux_attention import FluxVerbAttentionExtractor, VerbAttentionResult
from .cosmos_attention import CosmosVerbAttentionExtractor, CosmosVerbAttentionResult
from .verb_spatial_binding import compute_kld, compute_sim, compute_nss, evaluate_verb_spatial_binding
from .visualization import (
    plot_attention_overlay,
    plot_verb_comparison_grid,
    plot_gt_vs_predicted,
    plot_timestep_progression,
)
