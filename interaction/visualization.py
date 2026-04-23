"""
Visualization utilities for Axis 2 interaction affordance probing.

Generates publication-quality figures for:
- Attention heatmap overlays on images
- Verb comparison grids (same object, different verbs)
- Ground truth vs. predicted side-by-side
- Timestep progression of verb-spatial binding
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab/server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from PIL import Image


# Custom colormap: transparent -> yellow -> red (for attention overlays)
ATTENTION_CMAP = LinearSegmentedColormap.from_list(
    'attention',
    [(0, 0, 0, 0), (1, 1, 0, 0.5), (1, 0, 0, 0.9)],
    N=256,
)


def _resize_heatmap(
    heatmap: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Resize a heatmap to match image dimensions."""
    from PIL import Image as PILImage
    heatmap_pil = PILImage.fromarray(heatmap.astype(np.float32), mode='F')
    heatmap_resized = heatmap_pil.resize(target_size, PILImage.BILINEAR)
    return np.array(heatmap_resized)


def plot_attention_overlay(
    image: Image.Image,
    attention_map: np.ndarray,
    title: str = "",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 6),
    alpha: float = 0.6,
    cmap: str = "jet",
) -> plt.Figure:
    """
    Overlay an attention heatmap on an image.
    
    Args:
        image: PIL Image (RGB)
        attention_map: 2D attention heatmap (any resolution, will be resized)
        title: Figure title
        save_path: Optional path to save figure
        figsize: Figure size
        alpha: Heatmap overlay opacity
        cmap: Colormap for the heatmap
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Resize attention to image resolution
    attn_resized = _resize_heatmap(attention_map, (w, h))
    
    # Normalize to [0, 1]
    vmin, vmax = attn_resized.min(), attn_resized.max()
    if vmax > vmin:
        attn_resized = (attn_resized - vmin) / (vmax - vmin)
    
    ax.imshow(img_array)
    ax.imshow(attn_resized, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_verb_comparison_grid(
    image: Image.Image,
    verb_attention_maps: Dict[str, np.ndarray],
    title: str = "Verb-Spatial Binding Comparison",
    save_path: Optional[str] = None,
    figsize_per_panel: Tuple[float, float] = (4, 4),
    cmap: str = "jet",
) -> plt.Figure:
    """
    Show the same image with attention maps for different verbs.
    
    Args:
        image: PIL Image (RGB)
        verb_attention_maps: Dict mapping verb -> attention heatmap
        title: Overall figure title
        save_path: Optional path to save
        figsize_per_panel: Size per sub-panel
        cmap: Colormap
        
    Returns:
        matplotlib Figure
    """
    n_verbs = len(verb_attention_maps)
    n_cols = min(n_verbs + 1, 5)  # +1 for original image
    n_rows = (n_verbs + n_cols) // n_cols
    
    fig, axes = plt.subplots(
        1, n_verbs + 1,
        figsize=(figsize_per_panel[0] * (n_verbs + 1), figsize_per_panel[1]),
    )
    
    if n_verbs == 0:
        return fig
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis('off')
    
    # Verb attention maps
    for idx, (verb, attn_map) in enumerate(verb_attention_maps.items()):
        ax = axes[idx + 1]
        
        attn_resized = _resize_heatmap(attn_map, (w, h))
        vmin, vmax = attn_resized.min(), attn_resized.max()
        if vmax > vmin:
            attn_resized = (attn_resized - vmin) / (vmax - vmin)
        
        ax.imshow(img_array)
        ax.imshow(attn_resized, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
        ax.set_title(f'"{verb}"', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_gt_vs_predicted(
    image: Image.Image,
    gt_heatmap: np.ndarray,
    pred_heatmap: np.ndarray,
    affordance: str = "",
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5),
    cmap: str = "jet",
) -> plt.Figure:
    """
    Side-by-side comparison: Original | GT Affordance | Predicted Attention
    
    Args:
        image: PIL Image (RGB)
        gt_heatmap: Ground truth affordance heatmap
        pred_heatmap: Predicted attention heatmap from Flux
        affordance: Affordance category name
        metrics: Optional dict with KLD, SIM, NSS values
        save_path: Optional path to save
        figsize: Figure size
        cmap: Colormap
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Panel 1: Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')
    
    # Panel 2: Ground truth affordance
    gt_resized = _resize_heatmap(gt_heatmap, (w, h))
    gt_norm = gt_resized / (gt_resized.max() + 1e-7)
    
    axes[1].imshow(img_array)
    axes[1].imshow(gt_norm, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
    axes[1].set_title(f"GT: {affordance}", fontsize=12)
    axes[1].axis('off')
    
    # Panel 3: Predicted attention
    pred_resized = _resize_heatmap(pred_heatmap, (w, h))
    pred_norm = pred_resized / (pred_resized.max() + 1e-7)
    
    # Metrics text
    metrics_text = ""
    if metrics:
        metrics_text = (
            f"KLD↓={metrics.get('kld', 0):.3f}  "
            f"SIM↑={metrics.get('sim', 0):.3f}  "
            f"NSS↑={metrics.get('nss', 0):.3f}"
        )
    
    axes[2].imshow(img_array)
    axes[2].imshow(pred_norm, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title(f"Flux Attention: {affordance}", fontsize=12)
    axes[2].axis('off')
    
    if metrics_text:
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11,
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_timestep_progression(
    image: Image.Image,
    verb: str,
    attention_per_timestep: np.ndarray,
    num_display_steps: int = 6,
    save_path: Optional[str] = None,
    figsize_per_panel: Tuple[float, float] = (3, 3),
    cmap: str = "jet",
) -> plt.Figure:
    """
    Show how verb-spatial binding evolves across denoising timesteps.
    
    Args:
        image: PIL Image (RGB)
        verb: Verb being analyzed
        attention_per_timestep: (T, H, W) attention maps per timestep
        num_display_steps: Number of timesteps to display (evenly sampled)
        save_path: Optional path to save
        figsize_per_panel: Size per sub-panel
        cmap: Colormap
        
    Returns:
        matplotlib Figure
    """
    T = attention_per_timestep.shape[0]
    
    # Sample evenly across timesteps
    if T <= num_display_steps:
        step_indices = list(range(T))
    else:
        step_indices = np.linspace(0, T - 1, num_display_steps, dtype=int).tolist()
    
    n = len(step_indices) + 1  # +1 for original image
    fig, axes = plt.subplots(
        1, n,
        figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]),
    )
    
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis('off')
    
    # Timestep panels
    for panel_idx, t in enumerate(step_indices):
        ax = axes[panel_idx + 1]
        
        attn = attention_per_timestep[t]
        attn_resized = _resize_heatmap(attn, (w, h))
        
        vmin, vmax = attn_resized.min(), attn_resized.max()
        if vmax > vmin:
            attn_resized = (attn_resized - vmin) / (vmax - vmin)
        
        ax.imshow(img_array)
        ax.imshow(attn_resized, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
        ax.set_title(f"t={t+1}/{T}", fontsize=10)
        ax.axis('off')
    
    fig.suptitle(
        f'Timestep Progression: "{verb}"',
        fontsize=13,
        fontweight='bold',
        y=1.02,
    )
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def plot_affordance_summary_grid(
    results: List[Dict],
    n_cols: int = 4,
    save_path: Optional[str] = None,
    figsize_per_panel: Tuple[float, float] = (4, 4),
    cmap: str = "jet",
) -> plt.Figure:
    """
    Summary grid showing best examples per affordance category.
    
    Args:
        results: List of dicts with keys:
            - image: PIL Image
            - pred_heatmap: np.ndarray
            - affordance: str
            - metrics: Dict[str, float] (kld, sim, nss)
        n_cols: Number of columns in the grid
        save_path: Optional save path
        figsize_per_panel: Per-panel size
        cmap: Colormap
        
    Returns:
        matplotlib Figure
    """
    n = len(results)
    n_rows = (n + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
    )
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, result in enumerate(results):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        img_array = np.array(result["image"])
        h, w = img_array.shape[:2]
        
        pred = _resize_heatmap(result["pred_heatmap"], (w, h))
        pred = pred / (pred.max() + 1e-7)
        
        ax.imshow(img_array)
        ax.imshow(pred, cmap=cmap, alpha=0.55, vmin=0, vmax=1)
        
        affordance = result["affordance"]
        metrics = result.get("metrics", {})
        sim = metrics.get("sim", 0)
        ax.set_title(f"{affordance}\nSIM={sim:.3f}", fontsize=10)
        ax.axis('off')
    
    # Hide empty axes
    for idx in range(n, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(
        "Flux Verb-Spatial Binding — Per Affordance",
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig
