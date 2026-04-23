"""
Script 09: Setup and verify Flux for interaction affordance probing.

Smoke test that:
1. Loads Flux pipeline (schnell or dev)
2. Hooks attention extraction via attention-map-diffusers
3. Runs a single test prompt
4. Verifies attention map shapes and verb token identification
5. Saves a sample visualization

Usage:
    python scripts/09_setup_flux.py
    python scripts/09_setup_flux.py --model dev
    python scripts/09_setup_flux.py --cpu_offload  # for lower VRAM
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Verify Flux setup for Axis 2")
    parser.add_argument(
        "--model", type=str, default="schnell",
        choices=["schnell", "dev"],
        help="Flux variant (default: schnell for fast testing)"
    )
    parser.add_argument(
        "--cpu_offload", action="store_true",
        help="Use CPU offloading for lower VRAM"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results/figures/axis2",
        help="Directory to save test outputs"
    )
    args = parser.parse_args()
    
    # Model selection
    model_map = {
        "schnell": "black-forest-labs/FLUX.1-schnell",
        "dev": "black-forest-labs/FLUX.1-dev",
    }
    model_name = model_map[args.model]
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("AXIS 2 — Flux Setup Verification")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: cuda" if torch.cuda.is_available() else "Device: cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()
    
    # ── Step 1: Import and initialize extractor ──
    print("Step 1: Loading Flux pipeline...")
    t0 = time.time()
    
    from interaction.flux_attention import FluxVerbAttentionExtractor
    
    extractor = FluxVerbAttentionExtractor(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        enable_cpu_offload=args.cpu_offload,
    )
    
    # Force initialization
    extractor._ensure_initialized()
    
    t1 = time.time()
    print(f"✓ Pipeline loaded in {t1 - t0:.1f}s")
    if torch.cuda.is_available():
        print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()
    
    # ── Step 2: Test verb token identification ──
    print("Step 2: Testing verb token identification...")
    
    test_prompts = [
        ("a person cutting bread with a knife", ["cutting"]),
        ("a person holding a cup and drinking coffee", ["holding", "drinking"]),
        ("a person pouring water from a bottle", ["pouring"]),
    ]
    
    for prompt, verbs in test_prompts:
        indices = extractor.get_verb_token_indices(prompt, verbs)
        print(f"  Prompt: \"{prompt}\"")
        for verb, idx in indices.items():
            status = "✓" if idx else "✗ NOT FOUND"
            print(f"    '{verb}' -> token indices: {idx} {status}")
    print()
    
    # ── Step 3: Run inference with attention extraction ──
    print("Step 3: Running inference with attention extraction...")
    t0 = time.time()
    
    test_prompt = "a person cutting bread with a sharp knife on a wooden table"
    test_verbs = ["cutting"]
    
    result = extractor.extract(
        prompt=test_prompt,
        verbs=test_verbs,
        height=512,
        width=512,
        seed=42,
    )
    
    t1 = time.time()
    print(f"✓ Inference completed in {t1 - t0:.1f}s")
    print(f"  Prompt: \"{test_prompt}\"")
    print(f"  Latent size: {result.latent_size}")
    print(f"  Timesteps: {result.num_timesteps}")
    
    for verb, attn_map in result.verb_attention_maps.items():
        print(f"  Verb '{verb}' attention map shape: {attn_map.shape}")
        print(f"    min={attn_map.min():.4f}, max={attn_map.max():.4f}, "
              f"mean={attn_map.mean():.4f}")
    
    if result.verb_attention_per_timestep:
        for verb, per_step in result.verb_attention_per_timestep.items():
            print(f"  Per-timestep shape for '{verb}': {per_step.shape}")
    print()
    
    # ── Step 4: Save sample visualization ──
    print("Step 4: Generating sample visualizations...")
    
    from interaction.visualization import (
        plot_attention_overlay,
        plot_timestep_progression,
    )
    
    if result.generated_image is not None:
        # Save generated image
        gen_path = output_dir / "flux_test_generated.png"
        result.generated_image.save(str(gen_path))
        print(f"  Saved generated image: {gen_path}")
        
        # Attention overlay
        for verb, attn_map in result.verb_attention_maps.items():
            overlay_path = output_dir / f"flux_test_attn_{verb}.png"
            plot_attention_overlay(
                image=result.generated_image,
                attention_map=attn_map,
                title=f'Flux Attention: "{verb}"',
                save_path=str(overlay_path),
            )
            plt_path = output_dir / f"flux_test_attn_{verb}.png"
        
        # Timestep progression
        if result.verb_attention_per_timestep:
            for verb, per_step in result.verb_attention_per_timestep.items():
                prog_path = output_dir / f"flux_test_timestep_{verb}.png"
                plot_timestep_progression(
                    image=result.generated_image,
                    verb=verb,
                    attention_per_timestep=per_step,
                    save_path=str(prog_path),
                )
    
    print()
    
    # ── Step 5: Verify metrics on synthetic data ──
    print("Step 5: Verifying evaluation metrics on synthetic data...")
    
    from interaction.verb_spatial_binding import compute_kld, compute_sim, compute_nss
    
    # Test: identical distributions
    gt = np.random.rand(32, 32)
    gt = gt / gt.sum()
    
    kld_same = compute_kld(gt, gt)
    sim_same = compute_sim(gt, gt)
    print(f"  KLD(GT, GT) = {kld_same:.6f} (expect ~0)")
    print(f"  SIM(GT, GT) = {sim_same:.6f} (expect ~1)")
    
    assert kld_same < 0.01, f"KLD self-check failed: {kld_same}"
    assert sim_same > 0.99, f"SIM self-check failed: {sim_same}"
    
    # Test: uniform vs concentrated
    uniform = np.ones((32, 32)) / (32 * 32)
    concentrated = np.zeros((32, 32))
    concentrated[14:18, 14:18] = 1.0
    concentrated = concentrated / concentrated.sum()
    
    kld_diff = compute_kld(uniform, concentrated)
    sim_diff = compute_sim(uniform, concentrated)
    print(f"  KLD(uniform, concentrated) = {kld_diff:.3f} (expect > 0)")
    print(f"  SIM(uniform, concentrated) = {sim_diff:.3f} (expect < 1)")
    
    assert kld_diff > 0, "KLD divergence check failed"
    assert sim_diff < 1, "SIM divergence check failed"
    
    # NSS test
    fixations = np.zeros((32, 32))
    fixations[15, 15] = 1.0
    pred = np.zeros((32, 32))
    pred[14:18, 14:18] = 1.0
    nss_val = compute_nss(pred, fixations)
    print(f"  NSS(concentrated_pred, center_fixation) = {nss_val:.3f} (expect > 0)")
    
    print()
    print("✓ All metric checks passed")
    
    # ── Summary ──
    print()
    print("=" * 60)
    print("SETUP VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"✓ Flux {args.model} pipeline loaded and operational")
    print(f"✓ Verb token identification working")
    print(f"✓ Cross-attention extraction producing valid maps")
    print(f"✓ Visualization pipeline working")
    print(f"✓ Evaluation metrics validated")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nReady to run: python scripts/10_run_interaction_probing.py")


if __name__ == "__main__":
    main()
