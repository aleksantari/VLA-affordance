# Cosmos Predict2 — Literature Notes

## Models Used in Axis 2

| Model | HF ID | VRAM | Pipeline | Role |
|---|---|---|---|---|
| Cosmos-Predict2-2B-Text2Image | `nvidia/Cosmos-Predict2-2B-Text2Image` | 26 GB | `Cosmos2TextToImagePipeline` | Optional Flux-comparable variant |
| **Cosmos-Predict2-2B-Video2World** | `nvidia/Cosmos-Predict2-2B-Video2World` | 32.5 GB | `Cosmos2VideoToWorldPipeline` | **H2b — base video diffusion** |
| **Cosmos-Policy-ALOHA-Predict2-2B** | `nvidia/Cosmos-Policy-ALOHA-Predict2-2B` | ~32 GB | custom (cosmos-policy repo) | **H2c — manipulation fine-tuned** |

## Architecture (Predict2 family)

- Diffusion Transformer in latent space
- Interleaved self-attention + cross-attention + feedforward layers
- Cross-attention layers condition on text throughout denoising
- Adaptive layer normalization for time embedding
- Precision: BF16 only (FP16/FP32 not supported)
- 2B parameters

## Critical detail — base architecture

Cosmos Policy is fine-tuned from **Video2World** (NOT Text2Image). For a clean H2b vs H2c comparison, both must use the same Video2World base. The Text2Image variant is a *separate* Predict2 model and would not isolate fine-tuning effects.

## Conditioning differences

| System | Inputs |
|---|---|
| Flux | text only |
| Cosmos T2I | text only |
| Cosmos Video2World | text + first frame image |
| Cosmos Policy | text + 3 multi-view images (224×224) + proprioception (14-dim) |

## Implication for protocol

For H2b/H2c on AGD20K, the AGD20K image serves as the conditioning frame for Video2World (and a stand-in front-cam frame for Policy). For Flux, no image conditioning — only text. This is methodologically OK because we're probing the verb→spatial-latent cross-attention map *given* the model's natural conditioning, not the conditional generation quality.

## Attention extraction

`attention-map-diffusers` library does NOT support Cosmos. Need custom hook via diffusers `AttentionProcessor` API:

```python
class AttentionMapProcessor:
    def __call__(self, attn, hidden_states, encoder_hidden_states, ...):
        # standard QKV
        ...
        # store attention probs
        self.attn_maps.append(attention_probs.detach())
        return out
```

Registration:
```python
for name, module in pipe.transformer.named_modules():
    if name.endswith("attn2"):  # cross-attention
        module.processor = AttentionMapProcessor()
```

## Key papers

- NVIDIA. "Cosmos World Foundation Model Platform for Physical AI." arXiv:2501.03575, 2025.
- Kim et al. "Cosmos Policy: Fine-Tuning Video Models for Visuomotor Control and Planning." arXiv:2601.16163, 2026.
- HF blog: https://huggingface.co/blog/nvidia/cosmos-predict-2

## Sources
- [Cosmos-Predict2-2B-Text2Image](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Text2Image)
- [Cosmos-Predict2-2B-Video2World](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World)
- [Cosmos-Policy-ALOHA-Predict2-2B](https://huggingface.co/nvidia/Cosmos-Policy-ALOHA-Predict2-2B)
- [attention-map-diffusers](https://github.com/wooyeolbaek/attention-map-diffusers)
- [Introducing Cosmos Predict-2 blog](https://huggingface.co/blog/nvidia/cosmos-predict-2)
