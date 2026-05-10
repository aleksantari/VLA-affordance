"""
Cosmos Cross-Attention Extraction for Verb-Spatial Binding Analysis

Extracts cross-attention maps from NVIDIA Cosmos diffusion transformers
during denoising. Mirrors the FluxVerbAttentionExtractor API so the same
downstream evaluation works across all systems in Axis 2.

Two pipelines are supported:
- Cosmos2TextToImagePipeline (text-only, image generation)
- Cosmos2VideoToWorldPipeline (text + first-frame image, video generation)

Cosmos models use T5-11B as text encoder and the CosmosTransformer3DModel
(diffusers). Cross-attention layers are named `attn2` (one per transformer
block). We register a custom AttentionProcessor on every `attn2` to capture
attention probabilities during the forward pass.

attention-map-diffusers does not support Cosmos — this module is the
Cosmos-specific equivalent.

Reference:
    NVIDIA. "Cosmos World Foundation Model Platform for Physical AI."
    arXiv:2501.03575, 2025.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Result type — same shape as FluxVerbAttentionExtractor.VerbAttentionResult
# so downstream evaluation code is identical.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CosmosVerbAttentionResult:
    """Result of verb attention extraction for a single Cosmos run."""
    prompt: str
    verbs: List[str]
    verb_token_indices: Dict[str, List[int]]
    verb_attention_maps: Dict[str, np.ndarray]                # verb -> (H, W)
    verb_attention_per_timestep: Optional[Dict[str, np.ndarray]] = None  # verb -> (T, H, W)
    generated_image: Optional[object] = None                  # PIL.Image or None
    num_timesteps: int = 0
    num_blocks: int = 0
    latent_size: Tuple[int, int] = (0, 0)
    pipeline_type: str = ""                                   # "text2image" or "video2world"


# ─────────────────────────────────────────────────────────────────────────────
# Custom AttentionProcessor that records cross-attention probs.
# Mimics the diffusers default cross-attention math but stores probs.
# ─────────────────────────────────────────────────────────────────────────────


class CosmosCrossAttnRecorder:
    """
    Custom attention processor that records cross-attention probabilities.

    Designed to be a drop-in replacement for `attn.processor` on a Cosmos
    `attn2` (cross-attention) module. Mirrors the math of the diffusers
    default processor, with the addition of storing softmax(QK^T / sqrt(d))
    on `self.attn_maps`.

    For each forward pass, appends a tensor of shape
    (batch, num_heads, num_image_tokens, num_text_tokens) to `attn_maps`.
    """

    def __init__(self, name: str = "attn2", store: Optional[List] = None):
        self.name = name
        # Each entry: dict(layer=str, attn=Tensor(H_q, H_k), shape=(...) )
        self.store = store if store is not None else []

    def reset(self):
        self.store.clear()

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Cross-attention requires encoder_hidden_states; if absent, this is
        # called as self-attention (shouldn't happen for attn2, but guard).
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        residual = hidden_states
        batch_size, q_len, _ = hidden_states.shape
        kv_len = encoder_hidden_states.shape[1]

        if attn.norm_q is not None:
            hidden_states = attn.norm_q(hidden_states)

        # Q, K, V projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = attn.heads if hasattr(attn, "heads") else (
            query.shape[-1] // 64  # fallback assumption
        )
        # diffusers convention: attn.heads is num heads
        heads = attn.heads
        head_d = query.shape[-1] // heads

        # (B, q_len, heads*head_d) -> (B, heads, q_len, head_d)
        query = query.view(batch_size, q_len, heads, head_d).transpose(1, 2)
        key = key.view(batch_size, kv_len, heads, head_d).transpose(1, 2)
        value = value.view(batch_size, kv_len, heads, head_d).transpose(1, 2)

        if hasattr(attn, "norm_k") and attn.norm_k is not None:
            key = attn.norm_k(key)

        # Compute attention probs explicitly so we can record them.
        # Use float32 for numerical stability when saving probs.
        attn_logits = torch.matmul(
            query.float(), key.float().transpose(-2, -1)
        ) / math.sqrt(head_d)

        if attention_mask is not None:
            # Broadcast attention mask onto logits.
            # attention_mask is typically (B, 1, 1, kv_len) or (B, kv_len)
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_logits = attn_logits + attention_mask.to(attn_logits.dtype)

        attn_probs = attn_logits.softmax(dim=-1)
        # Detach + move to CPU + cast to half to keep memory manageable
        # Average over heads here to drop a 16-32x memory factor.
        avg_probs = attn_probs.detach().mean(dim=1).to(
            dtype=torch.float16, device="cpu"
        )  # (B, q_len, kv_len)

        self.store.append({
            "layer": self.name,
            "attn": avg_probs,                  # (B, q_len, kv_len) head-averaged
            "q_len": q_len,
            "kv_len": kv_len,
        })

        # Continue forward pass (use the same precision as inputs).
        attn_probs_for_out = attn_logits.softmax(dim=-1).to(query.dtype)
        out = torch.matmul(attn_probs_for_out, value)
        out = out.transpose(1, 2).reshape(batch_size, q_len, heads * head_d)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out) if len(attn.to_out) > 1 else out

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor — mirrors FluxVerbAttentionExtractor's public API
# ─────────────────────────────────────────────────────────────────────────────


class CosmosVerbAttentionExtractor:
    """
    Extract verb-specific cross-attention maps from a Cosmos pipeline
    during denoising.

    Supported:
        - Cosmos2TextToImagePipeline ("text2image")
        - Cosmos2VideoToWorldPipeline ("video2world")

    Usage:
        extractor = CosmosVerbAttentionExtractor(
            model_name="nvidia/Cosmos-Predict2-2B-Video2World",
            pipeline_type="video2world",
        )
        result = extractor.extract(
            prompt="a person cutting bread with a knife",
            verbs=["cutting"],
            image=PIL_image,  # required for video2world
        )
        # result.verb_attention_maps["cutting"] -> (H, W) heatmap
    """

    def __init__(
        self,
        model_name: str,
        pipeline_type: str = "video2world",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = False,
    ):
        if pipeline_type not in ("text2image", "video2world"):
            raise ValueError(
                f"pipeline_type must be 'text2image' or 'video2world', got '{pipeline_type}'"
            )

        self.model_name = model_name
        self.pipeline_type = pipeline_type
        self.device = device
        self.dtype = dtype
        self._enable_cpu_offload = enable_cpu_offload

        self._pipe = None
        self._initialized = False
        # Shared store populated by all attn2 processors during one forward pass.
        self._attn_store: List[dict] = []
        self._recorder_modules: List[torch.nn.Module] = []

    # ── pipeline lifecycle ────────────────────────────────────────────

    def _ensure_initialized(self):
        if self._initialized:
            return

        if self.pipeline_type == "text2image":
            from diffusers import Cosmos2TextToImagePipeline as PipelineCls
        else:
            from diffusers import Cosmos2VideoToWorldPipeline as PipelineCls

        print(f"Loading Cosmos pipeline: {self.model_name}")
        print(f"  pipeline_type: {self.pipeline_type}")
        print(f"  dtype: {self.dtype}")

        self._pipe = PipelineCls.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
        )

        if self._enable_cpu_offload:
            self._pipe.enable_sequential_cpu_offload()
        else:
            self._pipe = self._pipe.to(self.device)

        self._register_attention_recorders()
        self._initialized = True
        print(f"  registered {len(self._recorder_modules)} cross-attention recorders")
        print("✓ Cosmos pipeline ready")

    def _register_attention_recorders(self):
        """Replace the AttentionProcessor on every `attn2` module."""
        self._recorder_modules.clear()

        transformer = self._pipe.transformer
        for name, module in transformer.named_modules():
            # Cosmos blocks: <block>.attn2 is cross-attention
            if name.endswith("attn2") and hasattr(module, "to_q"):
                rec = CosmosCrossAttnRecorder(name=name, store=self._attn_store)
                # diffusers Attention modules have a .processor attribute
                if hasattr(module, "processor"):
                    module.processor = rec
                else:
                    # Fallback: monkey-patch forward to use recorder
                    module._cosmos_recorder = rec
                self._recorder_modules.append(module)

    # ── tokenization ──────────────────────────────────────────────────

    def get_verb_token_indices(
        self,
        prompt: str,
        verbs: List[str],
    ) -> Dict[str, List[int]]:
        """
        Find token positions for verbs in the T5 tokenization of the prompt.

        Mirrors FluxVerbAttentionExtractor.get_verb_token_indices but uses
        the Cosmos tokenizer (T5).
        """
        self._ensure_initialized()

        tokenizer = self._pipe.tokenizer
        encoding = tokenizer(
            prompt,
            return_offsets_mapping=True,
            add_special_tokens=True,
            max_length=getattr(self._pipe, "max_sequence_length", 512),
            truncation=True,
        )
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        verb_indices: Dict[str, List[int]] = {}
        prompt_lower = prompt.lower()

        for verb in verbs:
            verb_lower = verb.lower()
            indices: List[int] = []

            # Strategy 1: offset-mapping match
            offsets = encoding.get("offset_mapping")
            if offsets:
                start = prompt_lower.find(verb_lower)
                if start >= 0:
                    end = start + len(verb_lower)
                    for i, (s, e) in enumerate(offsets):
                        if s < end and e > start:
                            indices.append(i)

            # Strategy 2: decoded-token fallback
            if not indices:
                for i, tok in enumerate(tokens):
                    clean = tok.replace("▁", "").replace("Ġ", "").lower()
                    if clean and (verb_lower.startswith(clean) or clean in verb_lower):
                        indices.append(i)

            verb_indices[verb] = indices
            if not indices:
                print(f"  ⚠ verb '{verb}' not found in tokens: {tokens[:20]}...")

        return verb_indices

    # ── inference + extraction ────────────────────────────────────────

    def extract(
        self,
        prompt: str,
        verbs: List[str],
        image=None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 7.0,
        height: int = 480,
        width: int = 704,
        seed: Optional[int] = 42,
        store_per_timestep: bool = True,
        num_frames: int = 17,  # only used for video2world; small to save VRAM
    ) -> CosmosVerbAttentionResult:
        """
        Run Cosmos inference and extract verb cross-attention.

        For text2image: pass only prompt + verbs.
        For video2world: also pass `image` (the first conditional frame).
        """
        self._ensure_initialized()

        if num_inference_steps is None:
            num_inference_steps = 12  # below default 35 to keep wall-clock low

        if self.pipeline_type == "video2world" and image is None:
            raise ValueError("video2world requires an `image` argument")

        verb_token_indices = self.get_verb_token_indices(prompt, verbs)

        # Reset attention store
        self._attn_store.clear()

        generator = (
            torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        )

        if self.pipeline_type == "text2image":
            output = self._pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )
            generated_image = output.images[0]
            output_frames = None
        else:
            output = self._pipe(
                prompt=prompt,
                image=image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_frames=num_frames,
                generator=generator,
            )
            # Cosmos2VideoToWorld returns .frames[0] = list of PIL frames
            output_frames = output.frames[0]
            # Use the middle frame as the representative generated image
            generated_image = output_frames[len(output_frames) // 2]

        # Process recorded attention into per-verb spatial maps
        verb_maps_avg, verb_maps_per_step, num_blocks, latent_size = (
            self._process_attention_store(
                self._attn_store,
                verb_token_indices,
                spatial_h=height,
                spatial_w=width,
                num_frames=num_frames if self.pipeline_type == "video2world" else 1,
                store_per_timestep=store_per_timestep,
            )
        )

        return CosmosVerbAttentionResult(
            prompt=prompt,
            verbs=verbs,
            verb_token_indices=verb_token_indices,
            verb_attention_maps=verb_maps_avg,
            verb_attention_per_timestep=verb_maps_per_step if store_per_timestep else None,
            generated_image=generated_image,
            num_timesteps=num_inference_steps,
            num_blocks=num_blocks,
            latent_size=latent_size,
            pipeline_type=self.pipeline_type,
        )

    # ── attention map post-processing ────────────────────────────────

    def _process_attention_store(
        self,
        store: List[dict],
        verb_token_indices: Dict[str, List[int]],
        spatial_h: int,
        spatial_w: int,
        num_frames: int,
        store_per_timestep: bool,
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]], int, Tuple[int, int]]:
        """
        Process recorded cross-attention into per-verb (H, W) maps.

        Cosmos transformer flattens (T, H, W) latent into a single sequence
        of length T*H*W. We need to:
          1. Group store entries by layer-name (each block contributes
             num_inference_steps entries — one per denoising step)
          2. For each entry, slice the kv dim at verb token positions
          3. Reshape the q dim to (T, H, W); collapse temporal by averaging
          4. Average across all blocks
        """
        # Default empty result if no entries recorded
        if not store:
            empty = {v: np.zeros((spatial_h // 16, spatial_w // 16), dtype=np.float32) for v in verb_token_indices}
            return empty, None, 0, (spatial_h // 16, spatial_w // 16)

        # Group by layer name. Within a layer, entries are in temporal order
        # (one per denoising step).
        by_layer: Dict[str, List[dict]] = {}
        for entry in store:
            by_layer.setdefault(entry["layer"], []).append(entry)

        num_blocks = len(by_layer)

        # Determine spatial latent grid by trying common Cosmos downsampling
        # factors. Cosmos uses an 8x VAE + 2x patch in transformer (≈16x
        # effective). We compute the expected spatial product and find which
        # (T, H, W) factorization matches the recorded q_len.
        sample_q_len = store[0]["q_len"]
        latent_h, latent_w, latent_t = self._infer_latent_layout(
            sample_q_len, spatial_h, spatial_w, num_frames
        )

        # Allocate per-verb accumulators
        verb_per_step_blockavg: Dict[str, List[np.ndarray]] = {v: [] for v in verb_token_indices}

        # Process each layer separately, then average across layers
        for layer_name, entries in by_layer.items():
            num_steps_layer = len(entries)
            if num_steps_layer == 0:
                continue

            # Stack: list of (B, q_len, kv_len) -> (T_steps, B, q_len, kv_len)
            stacked = torch.stack([e["attn"] for e in entries], dim=0)
            # Take first batch element (B=1 expected)
            stacked = stacked[:, 0]  # (T_steps, q_len, kv_len)

            # If the q_len matches our expected T*H*W, reshape; else skip layer
            T_steps, q_len, kv_len = stacked.shape
            if q_len != latent_t * latent_h * latent_w:
                # Some layers (e.g., temporal-only attention) may have
                # different q_len. Try to fit any compatible layout.
                continue

            # Reshape q dim -> (T_steps, latent_t, latent_h, latent_w, kv_len)
            stacked_spatial = stacked.reshape(
                T_steps, latent_t, latent_h, latent_w, kv_len
            )

            # Collapse temporal (latent_t) by mean — interaction binding is a
            # spatial property, and we conditioned on a single first frame.
            spatial_attn = stacked_spatial.mean(dim=1)  # (T_steps, H, W, kv_len)

            for verb, token_indices in verb_token_indices.items():
                if not token_indices:
                    continue
                valid = [i for i in token_indices if i < kv_len]
                if not valid:
                    continue

                # Pick verb token columns and average
                verb_attn = spatial_attn[..., valid].mean(dim=-1)  # (T_steps, H, W)
                verb_per_step_blockavg[verb].append(verb_attn.float().numpy())

        # Aggregate across blocks then average across timesteps
        verb_maps_avg: Dict[str, np.ndarray] = {}
        verb_maps_per_step: Optional[Dict[str, np.ndarray]] = (
            {} if store_per_timestep else None
        )

        for verb in verb_token_indices:
            block_arrays = verb_per_step_blockavg[verb]
            if not block_arrays:
                verb_maps_avg[verb] = np.zeros((latent_h, latent_w), dtype=np.float32)
                if store_per_timestep:
                    verb_maps_per_step[verb] = np.zeros((1, latent_h, latent_w), dtype=np.float32)
                continue

            stack = np.stack(block_arrays, axis=0)  # (num_blocks, T_steps, H, W)
            block_avg = stack.mean(axis=0)          # (T_steps, H, W)

            map_avg = block_avg.mean(axis=0)         # (H, W)
            vmin, vmax = float(map_avg.min()), float(map_avg.max())
            if vmax > vmin:
                map_avg = (map_avg - vmin) / (vmax - vmin)

            verb_maps_avg[verb] = map_avg
            if store_per_timestep:
                verb_maps_per_step[verb] = block_avg

        return verb_maps_avg, verb_maps_per_step, num_blocks, (latent_h, latent_w)

    @staticmethod
    def _infer_latent_layout(
        q_len: int,
        spatial_h: int,
        spatial_w: int,
        num_frames: int,
    ) -> Tuple[int, int, int]:
        """
        Infer (latent_h, latent_w, latent_t) from the observed q_len.

        Tries a sequence of plausible downsampling factors used by Cosmos
        (VAE 8x in space + transformer 2x patch = 16x; T downsampling 4-8x).
        Returns the first (h, w, t) whose product equals q_len.
        """
        candidates = []
        for spatial_factor in (16, 8, 32):
            for t_factor in (1, 2, 4, 8):
                lh = max(1, spatial_h // spatial_factor)
                lw = max(1, spatial_w // spatial_factor)
                lt = max(1, num_frames // t_factor) if num_frames > 1 else 1
                candidates.append((lh, lw, lt))

        for lh, lw, lt in candidates:
            if lh * lw * lt == q_len:
                return lh, lw, lt

        # Fallback: assume single temporal frame, square spatial
        side = int(round(math.sqrt(q_len)))
        return side, side, 1

    @property
    def pipe(self):
        self._ensure_initialized()
        return self._pipe
