# SigLIP Through the PaliGemma -> pi0 -> pi0.5 Pipeline: Deep Research Report

**Date:** 2026-03-26
**Purpose:** Inform probing experiment design for geometric affordance research

---

## 1. PaliGemma's SigLIP Integration

### Architecture

PaliGemma is a 3B-parameter VLM combining three components:

1. **Vision Encoder:** SigLIP-So400m/14 (~400M params) -- a "shape-optimized" ViT
2. **Linear Projector:** A single linear layer projecting SigLIP's 1152-dim patch embeddings to Gemma's 2048-dim token space. **Initialized to zero** at the start of training.
3. **Text Decoder:** Gemma-2B (~2B params)

The projector is deliberately simple -- the PaliGemma paper tested linear vs MLP and found equivalent performance (77.2 vs 77.1 points), confirming a linear projection suffices.

### Training Stages

**Stage 0 -- Unimodal Pretraining (off-the-shelf):**
- Uses publicly available, independently pre-trained SigLIP and Gemma checkpoints
- Google explicitly states: "We do not perform any custom unimodal pretraining, instead relying on existing publicly available checkpoints"
- The SigLIP checkpoint is SigLIP-So400m/14, contrastively pre-trained on WebLi

**Stage 1 -- Multimodal Pretraining (224px, 1B examples):**
- Resolution: 224x224 (producing 256 image tokens)
- Sequence length: 128 text tokens
- **CRITICAL: Nothing is frozen.** The paper states: "not freezing any part of the model is indeed advantageous"
- The entire model (SigLIP + projector + Gemma) is trained jointly on diverse VL tasks
- SigLIP receives a **slow linear learning rate warm-up** to prevent degradation from initially misaligned LLM gradients
- This is explicitly contrasted with models like Flamingo that freeze the vision backbone

**Stage 2 -- Resolution Increase:**
- Takes the Stage 1 checkpoint and continues training at 448px (1024 tokens, 50M examples) and 896px (4096 tokens, 10M examples)
- Same task mixture but upweighted toward high-resolution tasks
- Position embeddings and patch embeddings are resized for the new resolution
- **All components remain unfrozen**
- The paper tested RoPE interpolation for image tokens during this stage but "saw no benefit"

**Stage 3 -- Transfer/Fine-tuning:**
- Fine-tuning on individual downstream tasks
- **Full model fine-tuning** of all parameters

### Key Implication: PaliGemma's SigLIP Has Different Weights Than Raw SigLIP

Since SigLIP is **unfrozen and jointly trained** during Stage 1 on 1 billion multimodal examples, the SigLIP weights inside PaliGemma are **definitively different** from the standalone pretrained SigLIP. The vision encoder has been adapted to work with the language model, receiving gradients from VL tasks including captioning, VQA, OCR, detection, and segmentation.

---

## 2. SigLIP Position Embedding Resolution Issue

### The Problem

Raw SigLIP (`google/siglip-so400m-patch14-384`) was pre-trained at 384x384:
- 384 / 14 = 27.43 -> 27 patches per side (last 6 pixels truncated due to `padding="valid"`)
- 27 x 27 = 729 learned position embeddings
- Note: 384 not being divisible by 14 was acknowledged by Google as "an inattention mistake" -- 378 would have been correct

PaliGemma/pi0/pi0.5 run at 224x224:
- 224 / 14 = 16 patches per side (exactly divisible)
- 16 x 16 = 256 position embeddings needed

### How Google Handled This

**CONFIRMED: There exists a separate SigLIP checkpoint natively trained at 224x224.**

HuggingFace hosts `google/siglip-so400m-patch14-224`, which was independently pre-trained on WebLi at 224x224 resolution with native 256 position embeddings. This is a separately trained model, not an interpolated version of the 384 checkpoint.

**Most likely scenario for PaliGemma:** PaliGemma's Stage 1 starts at 224px. Given that:
1. Google releases both 224 and 384 SigLIP checkpoints
2. Stage 1 trains at 224px producing 256 tokens
3. Google says they use "off-the-shelf" SigLIP checkpoints
4. The paper mentions no position embedding interpolation for Stage 1

It is highly probable that PaliGemma initializes from the **224-native SigLIP checkpoint** (not the 384 one with interpolated embeddings). Then for Stage 2 (448px, 896px), position embeddings are resized and training continues.

**However, this cannot be 100% confirmed** from the paper text. The paper does not explicitly name which resolution SigLIP checkpoint is used. It is also possible they used a 384-native checkpoint and resized position embeddings down to 256, since the SigLIP 2 paper describes a general strategy of resuming training at 95% with resized position embeddings.

### Impact of Using `interpolate_pos_encoding=True` on Raw SigLIP at 224x224

When using HuggingFace's `interpolate_pos_encoding=True` on the 384 model at 224x224:
- The 729 learned position embeddings (27x27 grid) are reshaped to (1, 27, 27, 1152)
- Bicubic interpolation downsamples to (1, 16, 16, 1152) = 256 positions
- `align_corners=False` is used, no antialiasing

**This will introduce artifacts:**
- Bicubic interpolation introduces smoothing -- nearby position embeddings get blended
- The model never saw these interpolated embeddings during training
- Spatial precision could degrade, particularly at boundaries between patches
- The effect is likely modest for a downsampling operation (27->16), but it is not zero

---

## 3. pi0's Treatment of SigLIP

### Architecture

pi0 = PaliGemma (3B, including SigLIP) + Action Expert (315M, randomly initialized)
- Total: ~3.3B parameters
- The PaliGemma backbone ("VLM expert") processes images and language
- The Action Expert processes robot state and action tokens
- They interact **only through shared self-attention layers** (mixture-of-experts style)
- Block-wise causal masking: VLM attends to itself, proprioception attends to VLM + itself, actions attend to all

### VLM Backbone Training Status: CONFLICTING EVIDENCE

This is the most ambiguous point in the entire pipeline. There are two conflicting accounts:

**Evidence that pi0 fine-tunes the VLM backbone (including SigLIP):**
- The pi0 paper says the model has "2.291B [parameters] to be fine-tuned" for the PaliGemma component, suggesting the full VLM backbone is trainable
- The Knowledge Insulation paper (pi0.5's paper) explicitly states that the problem they solve is that "gradients from the action expert lead to unfavorable learning dynamics" -- implying that in pi0 (the predecessor), gradients DO flow from the action expert into the VLM backbone
- The Knowledge Insulation paper shows freezing achieves "0% performance", demonstrating that VLM adaptation is necessary
- The pi0 paper describes "further train[ing]" the model on robot data

**Evidence that pi0 freezes the VLM backbone:**
- One analysis blog states: "This model is frozen during pi0 training, reducing compute but capping generalization"
- The openpi source code runs SigLIP with `train=False` (though this likely controls batch norm/dropout mode, not gradient flow)

**My assessment based on the evidence:** The weight of evidence strongly suggests that **pi0 fine-tunes the entire VLM backbone end-to-end**, including SigLIP. The "2.291B to be fine-tuned" statement is the most direct evidence, and the entire motivation of the Knowledge Insulation paper is to fix the problem of unconstrained gradient flow from the action expert into the VLM backbone -- which only makes sense if pi0 allowed that gradient flow. The `train=False` in the code likely just disables dropout/stochastic depth in the vision encoder, which is standard practice even for fine-tuned components.

### Image Resolution

pi0 processes images at **224x224** (confirmed from openpi preprocessing code: `ResizeImages(224, 224)`).

### Learning Rates

The pi0 paper does not disclose per-component learning rates. The openpi code uses a cosine decay schedule (e.g., `peak_lr=5e-5`), but whether SigLIP gets a different learning rate than the action expert is not specified in the open-source code.

---

## 4. pi0.5's Treatment of SigLIP (Knowledge Insulation)

### The Core Innovation

pi0.5 introduces **knowledge insulation** -- a gradient-stopping mechanism that prevents the action expert's gradients from flowing back into the VLM backbone:

```
Standard pi0:      Action Expert gradients --> flow into --> VLM backbone (SigLIP + Gemma)
Knowledge-insulated: Action Expert gradients --X STOPPED X--> VLM backbone
```

### Technical Mechanism

The gradient stopping is implemented in the **attention layers** via `stop_gradient` (sg) operations:

1. When action expert tokens attend to VLM backbone tokens, the **keys and values** from the backbone have `stop_gradient` applied:
   - `Q_action * sg(K_backbone^T)` -- action queries attending to backbone keys (gradient stopped)
   - `P_ab * sg(V_backbone)` -- action expert using backbone values (gradient stopped)
2. The VLM backbone's own self-attention operates normally (no gradient stopping)
3. This ensures the action expert can *read* backbone representations but cannot *modify* them through its gradients

### What SigLIP Receives Gradients From in pi0.5

**SigLIP IS still trained in pi0.5**, receiving gradients from two sources:

1. **FAST-tokenized discretized actions:** The VLM backbone is trained to predict discrete action tokens via next-token prediction (language-model style), and these gradients flow through the full backbone including SigLIP
2. **VLM co-training data:** The model is co-trained on vision-language tasks (captioning, VQA, bounding boxes). The paper states: "Co-training on VLM data is particularly important for generalization to novel objects"

**SigLIP does NOT receive gradients from:**
- The continuous action expert's flow-matching loss (blocked by stop_gradient)

### Comparison: pi0 vs pi0.5 SigLIP Training

| Aspect | pi0 | pi0.5 |
|--------|-----|-------|
| **SigLIP frozen?** | No (fine-tuned end-to-end) | No (fine-tuned, but gradient-insulated from action expert) |
| **Gradients from action expert?** | Yes (full backprop) | No (stop_gradient) |
| **Gradients from VLM tasks?** | Unclear (may not co-train on VLM data) | Yes (captioning, VQA, discrete actions) |
| **Risk of representation damage** | High (action gradients can corrupt VLM features) | Low (backbone protected) |

### Implication for SigLIP Weights

pi0.5's SigLIP has been fine-tuned on both robotics data (via discrete action prediction) and VLM data, but **protected from the potentially destructive gradients** of the continuous action expert. This should produce cleaner, more linguistically grounded representations than pi0's SigLIP.

---

## 5. PaliGemma as a Separate Probing Target

### How to Load PaliGemma's SigLIP

PaliGemma is gated on HuggingFace (requires access approval), but once approved:

```python
from transformers import PaliGemmaForConditionalGeneration

# Load full PaliGemma model
model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224")

# Extract just the vision tower (this IS a SiglipVisionModel)
vision_tower = model.vision_tower
vision_tower.eval()
```

**HuggingFace model IDs:**
- `google/paligemma-3b-pt-224` -- pretrained at 224px (Stage 1 output), best for our comparison
- `google/paligemma-3b-pt-448` -- pretrained at 448px (Stage 2)
- `google/paligemma-3b-mix-224` -- fine-tuned on task mixture (Stage 3)

### Would PaliGemma's SigLIP Have Different Weights?

**Yes, definitively.** PaliGemma trains SigLIP jointly with Gemma on 1 billion multimodal examples during Stage 1. The weights have been shaped by:
- Image captioning gradients
- Visual question answering gradients
- OCR gradients
- Object detection and segmentation gradients
- Grounded captioning gradients

This gives us a clean progression for probing:

```
Raw SigLIP (224)  -->  PaliGemma SigLIP  -->  pi0 SigLIP  -->  pi0.5 SigLIP
  (contrastive       (+ multimodal VL      (+ robot action    (+ robot action,
   pretraining)        joint training)       end-to-end)        gradient-insulated)
```

### Position Embedding Advantage

PaliGemma's `pt-224` checkpoint natively operates at 224x224 with 256 position embeddings. There is no interpolation needed -- this is the resolution it was trained at. This makes it a much cleaner comparison target than raw SigLIP-384 run at 224 with interpolated position embeddings.

---

## 6. Impact on Probing Results and Experimental Design

### Current Problem: Raw SigLIP at 224x224 with Interpolated Positions

The current pipeline (`encoders/raw_siglip.py`) uses:
```python
model_name = "google/siglip-so400m-patch14-384"  # 384-native model
processor.size = {"height": 224, "width": 224}    # forced to 224
```

This requires `interpolate_pos_encoding=True` (or equivalent), which:
- Downsamples 729 position embeddings to 256 via bicubic interpolation
- Introduces embeddings the model has never seen during training
- Could degrade spatial features, making raw SigLIP look artificially worse

### Recommended Experimental Design

**Option A: Use the 224-native SigLIP checkpoint (RECOMMENDED)**

Change `raw_siglip.py` to use `google/siglip-so400m-patch14-224`:
- Natively trained at 224x224
- Has 256 learned position embeddings
- No interpolation artifacts
- Fair comparison with PaliGemma (also 224-native) and pi0/pi0.5 (also 224)

**Option B: Also probe at 384 native resolution**

Keep the 384 model at its native resolution for comparison:
- Produces 729 patches (27x27 grid) instead of 256 (16x16)
- Cannot be directly compared patch-for-patch with pi0/pi0.5 (different grid sizes)
- But gives the "truest" raw SigLIP features
- Useful as a sanity check: if 224-native SigLIP underperforms 384-native SigLIP, position embeddings matter

**Option C: Add PaliGemma as an intermediate probing target (HIGHLY RECOMMENDED)**

This gives the cleanest experimental progression:

| Encoder | Resolution | Pos Embeddings | Training |
|---------|-----------|----------------|----------|
| Raw SigLIP (224-native) | 224x224 | 256 native | Contrastive on WebLi |
| PaliGemma SigLIP | 224x224 | 256 (jointly trained) | + Multimodal VL (1B examples) |
| pi0 SigLIP | 224x224 | 256 (fine-tuned) | + Robot actions (end-to-end) |
| pi0.5 SigLIP | 224x224 | 256 (fine-tuned) | + Robot actions (insulated) + VLM co-training |

### What Results Would Mean

**If PaliGemma SigLIP >> Raw SigLIP on affordance probing:**
- Multimodal VL training teaches geometric/functional understanding
- The improvement comes from language grounding, not robot experience

**If pi0 SigLIP < PaliGemma SigLIP:**
- Robot action gradients flowing through SigLIP *damage* general geometric understanding
- Supports the knowledge insulation hypothesis

**If pi0.5 SigLIP >= PaliGemma SigLIP:**
- Knowledge insulation successfully preserves VLM representations
- Robot training can coexist with good geometric features when gradients are managed

**If pi0 SigLIP > PaliGemma SigLIP:**
- Robot manipulation experience teaches additional geometric understanding
- End-to-end gradients from action prediction are beneficial

---

## Summary of Confirmed vs Inferred Facts

### Confirmed (directly from papers/code)

- PaliGemma's SigLIP is **unfrozen and jointly trained** during multimodal pretraining (Stage 1)
- PaliGemma Stage 1 trains at **224x224** producing **256 image tokens**
- SigLIP uses a **slow linear LR warm-up** during Stage 1
- The linear projector is **zero-initialized** and projects 1152 -> 2048 dims
- pi0.5 uses **stop_gradient** to block action expert gradients from reaching VLM backbone
- pi0.5 **co-trains on VLM data** (captioning, VQA) in addition to robot data
- Freezing the VLM backbone produces **0% performance** -- adaptation is necessary
- A **224-native SigLIP checkpoint** exists (`google/siglip-so400m-patch14-224`)
- HuggingFace's position embedding interpolation uses **bicubic mode** without antialiasing
- The 384x384 resolution for SigLIP-patch14 was "an inattention mistake" (should be 378)

### Inferred (strong evidence but not directly stated)

- pi0 **fine-tunes the VLM backbone end-to-end** including SigLIP (based on "2.291B to be fine-tuned" and the knowledge insulation paper's problem statement)
- PaliGemma likely initializes from the **224-native SigLIP checkpoint** (not 384 with interpolation)
- pi0's SigLIP receives gradients from the action expert's flow-matching loss (this is what knowledge insulation was designed to prevent)
- SigLIP in pi0.5 still receives gradients from discrete action tokens and VLM co-training data

### Unknown

- Exact per-component learning rates in pi0 and pi0.5
- Whether pi0 uses any VLM co-training data (pi0.5 explicitly does, pi0 is unclear)
- The precise magnitude of representation damage from action expert gradients in pi0
- Whether PaliGemma used the 224 or 384 SigLIP checkpoint for initialization (strong inference: 224)

---

## Sources

- [PaliGemma paper (arXiv 2407.07726)](https://arxiv.org/html/2407.07726v1) -- Training stages, SigLIP unfrozen, learning rate warm-up
- [PaliGemma big_vision README](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md) -- Stage descriptions, resolution details
- [Google Developers Blog: PaliGemma Architecture](https://developers.googleblog.com/gemma-explained-paligemma-architecture/) -- Architecture overview, position embeddings
- [HuggingFace PaliGemma Blog](https://huggingface.co/blog/paligemma) -- SigLIP integration, linear projector, fine-tuning guidance
- [pi0 paper (arXiv 2410.24164)](https://arxiv.org/html/2410.24164v1) -- Architecture, 2.291B fine-tunable params
- [Knowledge Insulation paper (arXiv 2505.23705)](https://arxiv.org/html/2505.23705v1) -- Gradient stopping mechanism, frozen backbone = 0%, co-training
- [Physical Intelligence Knowledge Insulation page](https://www.pi.website/research/knowledge_insulation) -- Gradient flow in pi0 vs pi0.5
- [VLA Models blog (Zihan)](https://loveaiblog.github.io/2025/05/14/pi/) -- pi0 frozen backbone claim, pi0.5 unfrozen during pretraining
- [OpenPI GitHub](https://github.com/Physical-Intelligence/openpi) -- Source code, training configs, `train=False` on SigLIP
- [OpenPI DeepWiki](https://deepwiki.com/Physical-Intelligence/openpi/4.2-p-model-family) -- Architecture diagrams, AdaRMS details
- [HuggingFace SigLIP-So400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) -- 384 model card, 729 patches
- [HuggingFace SigLIP-So400m-patch14-224](https://huggingface.co/google/siglip-so400m-patch14-224) -- 224-native model card
- [SigLIP 384 discussion: 384 not divisible by 14](https://huggingface.co/google/siglip-so400m-patch14-384/discussions/4) -- "Inattention mistake", padding behavior
- [HuggingFace SigLIP modeling code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py) -- interpolate_pos_encoding implementation
- [SigLIP 2 paper (arXiv 2502.14786)](https://arxiv.org/html/2502.14786v1) -- Position embedding resize strategy at 95% training
- [HuggingFace PaliGemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) -- Model card, architecture
