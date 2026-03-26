# SigLIP Encoder Progression: Raw → PaliGemma → π0 → π0.5

## Why this matters

Our probing experiment compares how VLA fine-tuning changes SigLIP's geometric affordance features. But SigLIP doesn't jump directly from contrastive pretraining to robot training — there's an intermediate stage: **PaliGemma's multimodal pretraining**. Understanding each stage tells us exactly what gradient regime shaped the features we're probing.

---

## The four encoder states

```
Raw SigLIP (224)  →  PaliGemma SigLIP  →  π0 SigLIP  →  π0.5 SigLIP
  contrastive         + multimodal VL      + robot          + robot
  on WebLi            1B examples          end-to-end       gradient-insulated
```

### 1. Raw SigLIP — contrastive pretraining only

- **Model:** `google/siglip-so400m-patch14-224` (224-native)
- **Training:** Contrastive loss on WebLi (image-text pairs). Global sigmoid loss per pair — no per-patch supervision.
- **Position embeddings:** 256, native to 224×224 (16×16 grid)
- **What SigLIP patch tokens learned:** Contribute to a single global image embedding via MAP (multi-head attention pooling). Individual patches were never directly supervised — they exist only as keys/values for the global contrastive query.
- **Expected affordance features:** Weak. Patch tokens capture broad semantic categories (object identity) but lack fine-grained spatial decomposition into functional parts.

### 2. PaliGemma SigLIP — multimodal VL joint training

- **Model:** `google/paligemma-3b-pt-224` → extract `model.vision_tower`
- **Training:** PaliGemma Stage 1 — SigLIP + linear projector (1152→2048) + Gemma-2B, trained jointly on 1B multimodal examples at 224×224. **SigLIP is fully unfrozen.** Slow linear LR warm-up protects SigLIP from initially misaligned LLM gradients.
- **Tasks providing gradients to SigLIP:** Image captioning, VQA, OCR, object detection, segmentation, grounded captioning
- **Position embeddings:** 256, jointly fine-tuned during Stage 1
- **Key fact:** The PaliGemma paper states "not freezing any part of the model is indeed advantageous." SigLIP weights are **definitively different** from raw SigLIP.
- **Expected affordance features:** Potentially stronger than raw SigLIP. VL tasks like grounded captioning and detection require localizing objects and parts — this could teach patch tokens to encode spatial/functional structure. However, the tasks are still primarily semantic (language-driven), not geometric.

**Source:** PaliGemma paper (arXiv 2407.07726), confirmed unfrozen + LR warm-up.

### 3. π0 SigLIP — robot action end-to-end fine-tuning

- **Model:** `lerobot/pi0_base` → extract `policy.model.paligemma_with_expert.paligemma.vision_tower`
- **Training:** PaliGemma backbone + 315M action expert (randomly initialized Gemma-based transformer). Trained on robot manipulation data via flow matching loss. **VLM backbone (including SigLIP) is fine-tuned end-to-end** — the π0 paper reports "2.291B parameters to be fine-tuned" for PaliGemma.
- **Gradient flow:** Action expert gradients flow backward through shared attention layers into the full VLM backbone, including SigLIP. The action expert is randomly initialized, so early gradients are noisy.
- **Position embeddings:** 256, inherited from PaliGemma, further fine-tuned
- **Risk:** The Knowledge Insulation paper (π0.5) was specifically motivated by the problem that "gradients from the action expert lead to unfavorable learning dynamics." This suggests π0's end-to-end training may **damage** the carefully learned VL representations.
- **Expected affordance features:** Could go either way. Robot manipulation requires understanding grasp affordances and contact geometry — which could improve spatial features. But noisy gradients from a randomly initialized expert could degrade the representations PaliGemma learned.

**Source:** π0 paper (arXiv 2410.24164) — "2.291B to be fine-tuned"; Knowledge Insulation paper's motivation.

### 4. π0.5 SigLIP — gradient-insulated robot training

- **Model:** `lerobot/pi05_base` → extract `policy.model.paligemma_with_expert.paligemma.vision_tower`
- **Training:** Same architecture as π0, but with **knowledge insulation**: `stop_gradient` is applied to the keys and values from the VLM backbone when action expert tokens attend to them. This blocks the action expert's flow-matching gradients from reaching SigLIP.
- **Gradients SigLIP receives:**
  - ✅ Discrete action tokens (FAST-tokenized, predicted via next-token prediction — a native LLM objective)
  - ✅ VLM co-training data (captioning, VQA — explicitly stated as "particularly important for generalization to novel objects")
  - ❌ Continuous action expert flow-matching loss (blocked by stop_gradient)
- **Position embeddings:** 256, inherited from PaliGemma, further fine-tuned
- **Expected affordance features:** Should preserve or improve upon PaliGemma's representations. The VLM co-training maintains general VL capability, while discrete action tokens provide robot-relevant signal through a safer gradient path.

**Source:** Knowledge Insulation paper (arXiv 2505.23705) — stop_gradient mechanism, co-training, frozen=0%.

---

## Resolution and position embedding comparison

| Encoder | HuggingFace model | Native resolution | Pos embeddings | Interpolation needed? |
|---------|-------------------|-------------------|----------------|----------------------|
| Raw SigLIP (224) | `google/siglip-so400m-patch14-224` | 224×224 | 256 native | **No** |
| Raw SigLIP (384) | `google/siglip-so400m-patch14-384` | 384×384 | 729 native | Yes (729→256 bicubic) |
| PaliGemma SigLIP | `google/paligemma-3b-pt-224` | 224×224 | 256 fine-tuned | **No** |
| π0 SigLIP | `lerobot/pi0_base` | 224×224 | 256 fine-tuned | **No** |
| π0.5 SigLIP | `lerobot/pi05_base` | 224×224 | 256 fine-tuned | **No** |

### Critical fix: use the 224-native raw SigLIP

The original code uses `google/siglip-so400m-patch14-384` forced to 224×224 with `interpolate_pos_encoding=True`. This introduces bicubic interpolation artifacts — the model receives position embeddings it never saw during training.

**Fix:** Switch to `google/siglip-so400m-patch14-224`, which was independently pretrained at 224×224 with native 256 position embeddings. This eliminates interpolation artifacts and provides a fair baseline.

Both the 224 and 384 checkpoints use the same SigLIP-So400m architecture (27 layers, 1152 hidden dim, ~400M params). The only difference is the resolution they were trained at.

**Note:** The 224 and 384 models are **separately trained** — the 224 model is NOT derived from the 384 model by interpolation. They are independent runs of the same architecture on the same data (WebLi) at different resolutions.

---

## What PaliGemma changes about SigLIP

PaliGemma doesn't just attach a projector and LLM — it **retrains SigLIP's weights** on 1 billion multimodal examples. The changes are:

1. **Patch token representations shift** from purely contrastive features to features useful for fine-grained VL tasks (detection, segmentation, grounded captioning)
2. **Position embeddings are jointly fine-tuned** with the rest of the model
3. **Attention patterns may change** — instead of attending primarily to support the global MAP pooling, patches learn to attend in ways useful for spatial tasks the LLM decoder needs
4. **Layer features evolve** — earlier layers may retain more raw spatial detail while later layers adapt to the LLM's expectations

This is why PaliGemma as a probing target is essential. Without it, we can't distinguish between:
- "Robot training changed SigLIP" (comparing π0 vs raw)
- "Multimodal VL training already changed SigLIP, and robot training changed it further" (comparing π0 vs PaliGemma vs raw)

---

## Impact on experimental design

### The progression tells a story

Each comparison answers a distinct question:

| Comparison | Question |
|-----------|----------|
| PaliGemma vs Raw SigLIP | Does multimodal VL training improve geometric affordance features? |
| π0 vs PaliGemma | Does end-to-end robot training help or hurt affordance features? |
| π0.5 vs PaliGemma | Does gradient-insulated robot training preserve affordance features? |
| π0.5 vs π0 | Does knowledge insulation protect representations better than naive fine-tuning? |
| DINOv2 vs all SigLIP variants | How large is the architecture gap (iBOT patch-level training vs contrastive)? |

### Hypothetical result scenarios

**Scenario A — VL training helps, robot training hurts:**
```
Raw SigLIP (0.35) < PaliGemma (0.50) > π0 (0.42) ≈ π0.5 (0.48)  <<  DINOv2 (0.67)
```
Interpretation: Multimodal VL training teaches useful spatial features, but robot action gradients partially corrupt them. Knowledge insulation preserves most of the gain.

**Scenario B — Progressive improvement:**
```
Raw SigLIP (0.35) < PaliGemma (0.45) < π0 (0.50) < π0.5 (0.55)  <  DINOv2 (0.67)
```
Interpretation: Each training stage adds geometric understanding. Robot experience is beneficial. The gap to DINOv2 is structural (iBOT patch-level supervision vs global contrastive).

**Scenario C — VL training is the key, robot training is neutral:**
```
Raw SigLIP (0.35) < PaliGemma (0.50) ≈ π0 (0.49) ≈ π0.5 (0.51)  <<  DINOv2 (0.67)
```
Interpretation: Multimodal VL training provides the geometric improvement. Robot training neither helps nor hurts appreciably.

---

## Loading PaliGemma's SigLIP (implementation)

```python
"""Extract SigLIP vision encoder from PaliGemma-3B pretrained checkpoint."""
import torch

def load_paligemma_siglip(model_name="google/paligemma-3b-pt-224", device="cuda"):
    from transformers import PaliGemmaForConditionalGeneration, SiglipImageProcessor

    # PaliGemma is gated — requires HuggingFace access approval
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_name)
    vision_tower = model.vision_tower  # SiglipVisionModel
    vision_tower = vision_tower.to(device).eval()

    # PaliGemma pt-224 uses SigLIP at 224x224 natively
    processor = SiglipImageProcessor.from_pretrained(model_name)

    # Free the LLM and projector
    del model
    torch.cuda.empty_cache()

    return vision_tower, processor


def extract_features(model, processor, images, device="cuda"):
    if not isinstance(images, list):
        images = [images]
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        patch_tokens = outputs.last_hidden_state  # (B, 256, 1152)
    return patch_tokens
```

**Requires:** HuggingFace access to `google/paligemma-3b-pt-224` (gated model, free approval).

---

## Updated encoder table (6 encoders)

| # | Encoder | Source | Type | Feature dim | Fused dim (4 layers) | Resolution |
|---|---------|--------|------|-------------|---------------------|------------|
| 1 | Raw SigLIP | `google/siglip-so400m-patch14-224` | siglip | 1152 | 4608 | 224 native |
| 2 | PaliGemma SigLIP | `google/paligemma-3b-pt-224` | siglip | 1152 | 4608 | 224 native |
| 3 | π0 SigLIP | `lerobot/pi0_base` | siglip | 1152 | 4608 | 224 native |
| 4 | π0.5 SigLIP | `lerobot/pi05_base` | siglip | 1152 | 4608 | 224 native |
| 5 | DINOv2 ViT-B | `facebook/dinov2-base` | dinov2 | 768 | 3072 | 224 native |
| 6 | DINO-WM | mazpie/dino-wm | dinov2 | 768 | 3072 | 224 native |

All 6 encoders now operate at 224×224 natively with no position embedding interpolation.

---

## Confirmed vs inferred

### Confirmed (from papers/code)
- PaliGemma SigLIP is **unfrozen** during Stage 1 multimodal pretraining (paper: "not freezing is advantageous")
- PaliGemma Stage 1 trains at **224×224** producing 256 tokens
- SigLIP gets **slow linear LR warm-up** during Stage 1
- π0 has "2.291B to be fine-tuned" for PaliGemma component
- π0.5 uses `stop_gradient` on keys/values from VLM backbone in action expert attention
- π0.5 **co-trains on VLM data** (captioning, VQA)
- Freezing VLM backbone = **0% performance** (must adapt)
- `google/siglip-so400m-patch14-224` exists as a **separately trained** 224-native checkpoint

### Inferred (strong evidence)
- π0 fine-tunes SigLIP end-to-end (based on "2.291B to fine-tune" + Knowledge Insulation paper's problem statement)
- PaliGemma likely initializes from the 224-native SigLIP checkpoint (trains at 224, uses "off-the-shelf" checkpoints)

### Unknown
- Exact per-component learning rates in π0 and π0.5
- Whether π0 uses VLM co-training data (π0.5 explicitly does)
- Precise magnitude of representation change at each stage (this is what our probing measures)
