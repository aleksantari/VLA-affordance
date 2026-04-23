# Axis 2: Interaction Affordance Probing
# Flux cross-attention extraction and verb-spatial binding analysis

from .flux_attention import FluxVerbAttentionExtractor
from .verb_spatial_binding import compute_kld, compute_sim, compute_nss, evaluate_verb_spatial_binding
from .visualization import (
    plot_attention_overlay,
    plot_verb_comparison_grid,
    plot_gt_vs_predicted,
    plot_timestep_progression,
)
