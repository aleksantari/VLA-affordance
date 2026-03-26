"""Visualization utilities: PCA colormaps, similarity heatmaps, bar charts."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path


def plot_pca_colormap(projected, title="PCA Projection", save_path=None):
    """Visualize PCA-projected patch features as an RGB colormap.

    The first 3 PCA components are mapped to RGB channels.

    Args:
        projected: (H, W, 3) array of PCA-projected features
        title: plot title
        save_path: optional path to save figure
    """
    # Normalize each component to [0, 1] for RGB display
    rgb = projected.copy()
    for c in range(3):
        vmin, vmax = rgb[:, :, c].min(), rgb[:, :, c].max()
        if vmax - vmin > 0:
            rgb[:, :, c] = (rgb[:, :, c] - vmin) / (vmax - vmin)
        else:
            rgb[:, :, c] = 0.5

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(rgb, interpolation='nearest')
    ax.set_title(title)
    ax.axis('off')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return fig


def plot_pca_comparison(projections_dict, save_path=None):
    """Plot PCA projections for multiple encoders side by side.

    Args:
        projections_dict: dict mapping encoder_name -> (16, 16, 3) PCA projections
        save_path: optional path to save figure
    """
    n = len(projections_dict)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, proj) in zip(axes, projections_dict.items()):
        rgb = proj.copy()
        for c in range(min(3, proj.shape[-1])):
            vmin, vmax = rgb[:, :, c].min(), rgb[:, :, c].max()
            if vmax - vmin > 0:
                rgb[:, :, c] = (rgb[:, :, c] - vmin) / (vmax - vmin)
            else:
                rgb[:, :, c] = 0.5
        if proj.shape[-1] < 3:
            rgb = np.pad(rgb, ((0, 0), (0, 0), (0, 3 - proj.shape[-1])))
        ax.imshow(rgb[:, :, :3], interpolation='nearest')
        ax.set_title(name, fontsize=14)
        ax.axis('off')

    plt.suptitle("PCA Subspace Projections", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return fig


def plot_similarity_heatmap(similarity_map, query_patch_idx=None,
                            title="Cosine Similarity", save_path=None):
    """Plot a cosine similarity heatmap.

    Args:
        similarity_map: (H, W) array of similarity values
        query_patch_idx: optional (row, col) to mark the query patch
        title: plot title
        save_path: optional path to save
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(similarity_map, cmap='hot', interpolation='nearest',
                   vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)

    if query_patch_idx is not None:
        row, col = query_patch_idx
        ax.plot(col, row, 'c*', markersize=15, markeredgecolor='white',
                markeredgewidth=1.5)

    ax.set_title(title)
    ax.axis('off')

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return fig


def plot_miou_comparison(miou_dict, save_path=None):
    """Bar chart comparing mIoU across encoders.

    Args:
        miou_dict: dict mapping encoder_name -> mIoU value
        save_path: optional path to save
    """
    names = list(miou_dict.keys())
    values = list(miou_dict.values())

    # Color coding by encoder family
    colors = []
    for name in names:
        if 'dino' in name.lower():
            colors.append('#2ecc71')  # green for DINOv2 family
        elif 'siglip' in name.lower() or 'pi0' in name.lower():
            colors.append('#3498db')  # blue for SigLIP family
        else:
            colors.append('#95a5a6')  # gray for others

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('mIoU', fontsize=14)
    ax.set_title('Affordance Segmentation mIoU Comparison', fontsize=16)
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    ax.axhline(y=0.670, color='red', linestyle='--', alpha=0.7,
               label='Zhang et al. DINOv2 (0.670)')
    ax.legend(fontsize=11)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return fig


def plot_weight_divergence(divergence_results, save_path=None):
    """Visualize per-layer weight divergence.

    Args:
        divergence_results: output of compute_weight_divergence()
        save_path: optional path to save
    """
    names = list(divergence_results.keys())
    changes = [divergence_results[n]['relative_change'] for n in names]

    # Shorten names for display
    short_names = [n.split('.')[-1] if len(n) > 30 else n for n in names]

    fig, ax = plt.subplots(figsize=(14, max(6, len(names) * 0.2)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, changes, color='steelblue', edgecolor='black', linewidth=0.3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=6)
    ax.set_xlabel('Relative Change (L2 / ||w||)', fontsize=12)
    ax.set_title('Per-Layer Weight Divergence', fontsize=14)
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return fig
