"""
Script 09b: Setup and verify Cosmos for interaction affordance probing.

Fail-fast smoke test (~2-3 min on A100) that:
1. Loads the Cosmos pipeline (Predict2 V2W or Policy)
2. Verifies cross-attention recorder hooks register correctly
3. Runs ONE inference at low resolution
4. Verifies attention map shapes are sensible
5. Verifies metric computations work

Run this BEFORE committing to a multi-hour pilot. If it fails, debugging
is cheap; if it passes, the pilot will succeed.

Usage:
    python scripts/09b_setup_cosmos.py --system cosmos_predict2_v2w
    python scripts/09b_setup_cosmos.py --system cosmos_policy
    python scripts/09b_setup_cosmos.py --system cosmos_predict2_t2i
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from PIL import Image


SYSTEM_CONFIG = {
    "cosmos_predict2_t2i": {
        "model_name": "nvidia/Cosmos-Predict2-2B-Text2Image",
        "pipeline_type": "text2image",
        "needs_image_input": False,
    },
    "cosmos_predict2_v2w": {
        "model_name": "nvidia/Cosmos-Predict2-2B-Video2World",
        "pipeline_type": "video2world",
        "needs_image_input": True,
    },
    "cosmos_policy": {
        "model_name": "nvidia/Cosmos-Policy-ALOHA-Predict2-2B",
        "pipeline_type": "video2world",
        "needs_image_input": True,
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--system", type=str, required=True, choices=list(SYSTEM_CONFIG.keys()),
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=9)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument(
        "--output_dir", type=str, default="./results/figures/axis2/smoke",
    )
    args = parser.parse_args()

    cfg = SYSTEM_CONFIG[args.system]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"AXIS 2 — Cosmos Smoke Test: {args.system}")
    print("=" * 60)
    print(f"Model: {cfg['model_name']}")
    print(f"Pipeline type: {cfg['pipeline_type']}")
    print(f"Resolution: {args.height}×{args.width}")
    if cfg["pipeline_type"] == "video2world":
        print(f"Num frames: {args.num_frames}")
    print(f"Inference steps: {args.num_inference_steps}")
    print()

    print(f"Device: cuda" if torch.cuda.is_available() else "Device: cpu")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name()
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu}  VRAM: {vram:.1f} GB")
    print()

    # ── Step 1: Initialize extractor ──
    print("Step 1: Loading pipeline (this may take 2-5 minutes on first run)...")
    t0 = time.time()

    from interaction.cosmos_attention import CosmosVerbAttentionExtractor

    extractor = CosmosVerbAttentionExtractor(
        model_name=cfg["model_name"],
        pipeline_type=cfg["pipeline_type"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.bfloat16,
        enable_cpu_offload=args.cpu_offload,
    )
    extractor._ensure_initialized()

    n_recorders = len(extractor._recorder_modules)
    print(f"✓ Loaded in {time.time() - t0:.1f}s")
    print(f"  Cross-attention recorders registered: {n_recorders}")
    if n_recorders == 0:
        print("❌ FAIL: Zero attn2 modules found. The model architecture may have changed.")
        sys.exit(1)
    if torch.cuda.is_available():
        print(f"  VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print()

    # ── Step 2: Verify verb token detection ──
    print("Step 2: Testing verb token identification...")
    test_prompts = [
        ("a person cutting bread with a knife", ["cutting"]),
        ("a person holding a cup and drinking coffee", ["holding", "drinking"]),
        ("a person pouring water from a bottle", ["pouring"]),
    ]
    for prompt, verbs in test_prompts:
        idx = extractor.get_verb_token_indices(prompt, verbs)
        for verb, ids in idx.items():
            status = "✓" if ids else "✗"
            print(f"  {status} '{verb}' in \"{prompt[:50]}...\" -> tokens {ids}")
    print()

    # ── Step 3: Run a single inference ──
    print("Step 3: Running one inference + attention extraction...")
    t0 = time.time()

    # Build a dummy image for video2world / policy
    dummy_image = None
    if cfg["needs_image_input"]:
        dummy_image = Image.new("RGB", (args.width, args.height), color=(127, 127, 127))

    test_prompt = "a person cutting bread with a sharp knife on a wooden table"
    test_verbs = ["cutting"]

    try:
        result = extractor.extract(
            prompt=test_prompt,
            verbs=test_verbs,
            image=dummy_image,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            num_frames=args.num_frames,
            seed=42,
            store_per_timestep=True,
        )
    except Exception as e:
        print(f"❌ FAIL: extract() raised {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"✓ Inference completed in {time.time() - t0:.1f}s")
    print(f"  Latent size: {result.latent_size}")
    print(f"  Num blocks averaged: {result.num_blocks}")
    for verb, m in result.verb_attention_maps.items():
        print(
            f"  Verb '{verb}' attention map: shape={m.shape} "
            f"min={m.min():.4f} max={m.max():.4f} mean={m.mean():.4f}"
        )
    if result.verb_attention_per_timestep:
        for verb, ts in result.verb_attention_per_timestep.items():
            print(f"  Per-timestep '{verb}': shape={ts.shape}")
    print()

    # Sanity: attention map should not be all-zero or all-constant
    failed = False
    for verb, m in result.verb_attention_maps.items():
        if m.std() < 1e-6:
            print(f"❌ WARNING: '{verb}' attention map is constant — extraction may be broken")
            failed = True
        if not (m.min() >= 0 and m.max() <= 1):
            print(f"⚠ '{verb}' attention map outside [0, 1] (got {m.min():.3f}-{m.max():.3f})")
            failed = True

    # ── Step 4: Verify metrics on the produced map ──
    print("Step 4: Sanity-checking metric computation...")
    from interaction.verb_spatial_binding import compute_kld, compute_sim, compute_nss, heatmap_to_fixations

    pred = list(result.verb_attention_maps.values())[0]
    # Construct a synthetic GT — a Gaussian blob in the center
    gh, gw = pred.shape
    yy, xx = np.mgrid[:gh, :gw]
    gt = np.exp(-((yy - gh / 2) ** 2 + (xx - gw / 2) ** 2) / (2 * (gh / 4) ** 2))
    gt = gt / gt.sum()

    kld = compute_kld(pred, gt)
    sim = compute_sim(pred, gt)
    nss = compute_nss(pred, heatmap_to_fixations(gt))
    print(f"  vs synthetic Gaussian-center GT: KLD={kld:.3f}  SIM={sim:.3f}  NSS={nss:.3f}")
    print()

    # ── Step 5: Save the smoke-test artifact ──
    smoke_path = output_dir / f"smoke_{args.system}_pred.png"
    pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    Image.fromarray((pred_norm * 255).astype(np.uint8)).save(smoke_path)
    if result.generated_image is not None:
        result.generated_image.save(output_dir / f"smoke_{args.system}_img.png")
    print(f"✓ Saved attention map to {smoke_path}")
    print()

    print("=" * 60)
    if failed:
        print(f"⚠ SMOKE TEST FAILED — extraction produced unusable output.")
        print(f"   Fix bugs before running pilot (would waste hours).")
        sys.exit(2)
    print(f"✓ SMOKE TEST PASSED — Cosmos extraction is working for {args.system}.")
    print("=" * 60)
    print("Ready to run: python scripts/10b_run_cosmos_probing.py --system "
          f"{args.system}")


if __name__ == "__main__":
    main()
