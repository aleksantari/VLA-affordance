"""
Flux Cross-Attention Extraction for Verb-Spatial Binding Analysis

REWRITTEN 2026-05-10 to drop the attention-map-diffusers dependency.

attention-map-diffusers patches diffusers internals via monkey-patching and
breaks every time diffusers ships a signature change (Colab installed a
diffusers where FluxSingleTransformerBlock.forward gained an
encoder_hidden_states positional, but the library still calls block(...)
without it — TypeError at runtime).

Instead, this module installs a custom AttentionProcessor on every
FluxTransformerBlock.attn (and FluxSingleTransformerBlock.attn) and mirrors
the math of diffusers' FluxAttnProcessor while recording the post-softmax
attention map. This is the same pattern used in cosmos_attention.py.

Reference:
    Zhang et al., "Probing and Bridging Geometry-Interaction Cues
    for Affordance Reasoning in Vision Foundation Models", CVPR 2026.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VerbAttentionResult:
    """Result of verb attention extraction for a single image."""
    prompt: str
    verbs: List[str]
    verb_token_indices: Dict[str, List[int]]
    verb_attention_maps: Dict[str, np.ndarray]                 # verb -> (H, W)
    verb_attention_per_timestep: Optional[Dict[str, np.ndarray]] = None
    generated_image: Optional[object] = None
    num_timesteps: int = 0
    num_blocks: int = 0
    latent_size: Tuple[int, int] = (64, 64)


# ─────────────────────────────────────────────────────────────────────────────
# Custom AttentionProcessor — records cross-attention probabilities
# Mirrors diffusers' FluxAttnProcessor with explicit softmax for capture.
# ─────────────────────────────────────────────────────────────────────────────


class FluxCrossAttnRecorder:
    """
    Custom processor that mirrors `FluxAttnProcessor` (diffusers) and records
    head-averaged cross-attention probabilities (image_q × text_k).

    Joint-attention layout (per diffusers' transformer_flux.py):
      Q = concat([encoder_q, image_q], dim=seq)
      K = concat([encoder_k, image_k], dim=seq)
      V = concat([encoder_v, image_v], dim=seq)
    Text comes first, image second, along the sequence dim.

    For each forward pass, appends a record of shape
        (batch, num_image_tokens, num_text_tokens)
    to `self.store`, head-averaged and dtype=float16, device=CPU.
    """

    def __init__(self, name: str = "attn", store: Optional[List] = None):
        self.name = name
        self.store = store if store is not None else []

    def reset(self):
        self.store.clear()

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Mirrors diffusers' current FluxAttnProcessor exactly. Layout note:
        # the diffusers FluxAttention uses (B, S, H, D) — i.e. unflatten the
        # last dim into (heads, head_d) WITHOUT transposing to put heads
        # before sequence. RoPE and attention then operate over sequence_dim=1.
        # The ONLY thing we add is recording the post-softmax (image_q × text_k)
        # slice for verb-spatial-binding analysis.

        # ── Q/K/V projections — image stream (called `to_q/to_k/to_v`) ──
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)

        # ── Text stream — only for double-stream blocks (added_kv_proj_dim set) ──
        txt_len = 0
        if encoder_hidden_states is not None and getattr(attn, "added_kv_proj_dim", None) is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            if getattr(attn, "norm_added_q", None) is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if getattr(attn, "norm_added_k", None) is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            txt_len = encoder_hidden_states.shape[1]

            # Joint: text first, then image, along sequence dim=1
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # ── RoPE on (B, S, H, D), sequence_dim=1 ──
        if image_rotary_emb is not None:
            try:
                from diffusers.models.embeddings import apply_rotary_emb
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
            except TypeError:
                from diffusers.models.embeddings import apply_rotary_emb
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)
            except Exception:
                pass

        # ── Compute attention probs explicitly (need transpose to (B, H, S, D)) ──
        q_h = query.permute(0, 2, 1, 3).contiguous()
        k_h = key.permute(0, 2, 1, 3).contiguous()
        v_h = value.permute(0, 2, 1, 3).contiguous()

        head_d = q_h.shape[-1]
        scale = head_d ** -0.5
        attn_logits = torch.matmul(q_h.float(), k_h.float().transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attn_logits = attn_logits + attention_mask.to(attn_logits.dtype)

        attn_probs = attn_logits.softmax(dim=-1)  # (B, H, S, S) fp32

        # ── Record cross-attention slice (image_q × text_k) ──
        if txt_len > 0:
            img_len = q_h.shape[2] - txt_len
            cross = attn_probs[:, :, txt_len:txt_len + img_len, :txt_len]
            avg = cross.detach().mean(dim=1).to(dtype=torch.float16, device="cpu")
            self.store.append({
                "layer": self.name,
                "attn": avg,
                "img_len": img_len,
                "txt_len": txt_len,
            })

        # ── Forward: probs @ V → reshape back to (B, S, H*D) ──
        out_h = torch.matmul(attn_probs.to(q_h.dtype), v_h)         # (B, H, S, D)
        out = out_h.permute(0, 2, 1, 3).contiguous()                 # (B, S, H, D)
        out = out.flatten(2, 3).to(query.dtype)                      # (B, S, H*D)

        # ── Output projections: ONLY for double-stream blocks ──
        # Single-stream / pre_only blocks have no `to_out`; the block
        # itself handles the projection downstream.
        if encoder_hidden_states is not None and txt_len > 0:
            encoder_out, hidden_out = out.split_with_sizes(
                [txt_len, out.shape[1] - txt_len], dim=1
            )
            if hasattr(attn, "to_out") and not getattr(attn, "pre_only", False):
                hidden_out = attn.to_out[0](hidden_out.contiguous())
                if len(attn.to_out) > 1:
                    hidden_out = attn.to_out[1](hidden_out)
            if hasattr(attn, "to_add_out"):
                encoder_out = attn.to_add_out(encoder_out.contiguous())
            return hidden_out, encoder_out
        else:
            return out


# ─────────────────────────────────────────────────────────────────────────────
# Public extractor — same API as before so script 10 doesn't change
# ─────────────────────────────────────────────────────────────────────────────


class FluxVerbAttentionExtractor:
    """
    Extract verb-specific cross-attention maps from Flux during denoising.

    Drop-in replacement for the previous (broken) attention-map-diffusers
    based extractor. Public API unchanged.
    """

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-schnell",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = False,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.is_schnell = "schnell" in model_name.lower()
        self._enable_cpu_offload = enable_cpu_offload

        self._pipe = None
        self._initialized = False
        self._attn_store: List[dict] = []
        self._recorder_modules: List[torch.nn.Module] = []

    def _ensure_initialized(self):
        if self._initialized:
            return

        from diffusers import FluxPipeline

        print(f"Loading Flux pipeline: {self.model_name}")
        print(f"  dtype: {self.dtype}")
        print(f"  device: {self.device}")

        self._pipe = FluxPipeline.from_pretrained(
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
        print(f"✓ Flux pipeline ready")

    def _register_attention_recorders(self):
        """Replace the AttentionProcessor on every Flux block's `attn`."""
        self._recorder_modules.clear()
        transformer = self._pipe.transformer

        # Double-stream blocks expose `transformer_blocks[i].attn` (the
        # text+image joint attention). Single-stream blocks expose
        # `single_transformer_blocks[i].attn` (concatenated stream).
        for name, module in transformer.named_modules():
            short = name.rsplit(".", 1)[-1]
            if short == "attn" and hasattr(module, "to_q"):
                rec = FluxCrossAttnRecorder(name=name, store=self._attn_store)
                if hasattr(module, "processor"):
                    module.processor = rec
                self._recorder_modules.append(module)

    # ── tokenization ──────────────────────────────────────────────────

    def get_verb_token_indices(
        self,
        prompt: str,
        verbs: List[str],
    ) -> Dict[str, List[int]]:
        self._ensure_initialized()

        # Flux uses two tokenizers; T5 (tokenizer_2) is the one feeding
        # cross-attention text features.
        tokenizer = self._pipe.tokenizer_2
        encoding = tokenizer(
            prompt,
            return_offsets_mapping=True,
            add_special_tokens=True,
            max_length=getattr(self._pipe, "tokenizer_max_length", 512),
            truncation=True,
        )
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

        verb_indices: Dict[str, List[int]] = {}
        prompt_lower = prompt.lower()

        for verb in verbs:
            verb_lower = verb.lower()
            indices: List[int] = []

            offsets = encoding.get("offset_mapping")
            if offsets:
                start = prompt_lower.find(verb_lower)
                if start >= 0:
                    end = start + len(verb_lower)
                    for i, (s, e) in enumerate(offsets):
                        if s < end and e > start:
                            indices.append(i)

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
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 0.0,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = 42,
        store_per_timestep: bool = True,
    ) -> VerbAttentionResult:
        self._ensure_initialized()

        if num_inference_steps is None:
            num_inference_steps = 4 if self.is_schnell else 20
        if self.is_schnell:
            guidance_scale = 0.0  # schnell ignores CFG

        verb_token_indices = self.get_verb_token_indices(prompt, verbs)
        self._attn_store.clear()

        generator = (
            torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        )

        output = self._pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        generated_image = output.images[0]

        # Flux packs 2 spatial patches per token; latent grid = (H/16, W/16).
        latent_h = height // 16
        latent_w = width // 16

        verb_maps_avg, verb_maps_per_step, num_blocks = self._process_attention_store(
            self._attn_store,
            verb_token_indices,
            latent_h,
            latent_w,
            store_per_timestep,
        )

        return VerbAttentionResult(
            prompt=prompt,
            verbs=verbs,
            verb_token_indices=verb_token_indices,
            verb_attention_maps=verb_maps_avg,
            verb_attention_per_timestep=verb_maps_per_step if store_per_timestep else None,
            generated_image=generated_image,
            num_timesteps=num_inference_steps,
            num_blocks=num_blocks,
            latent_size=(latent_h, latent_w),
        )

    def _process_attention_store(
        self,
        store: List[dict],
        verb_token_indices: Dict[str, List[int]],
        latent_h: int,
        latent_w: int,
        store_per_timestep: bool,
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]], int]:
        """
        Group recorded entries by layer name, then for each verb:
          - select kv columns corresponding to verb tokens
          - reshape image dim to (H, W)
          - average across heads (already done at capture) and timesteps and blocks
        """
        if not store:
            empty = {v: np.zeros((latent_h, latent_w), dtype=np.float32) for v in verb_token_indices}
            return empty, None, 0

        by_layer: Dict[str, List[dict]] = {}
        for entry in store:
            by_layer.setdefault(entry["layer"], []).append(entry)
        num_blocks = len(by_layer)

        verb_block_lists: Dict[str, List[np.ndarray]] = {v: [] for v in verb_token_indices}

        # Determine the correct image-token reshape. Flux uses 2x2 packed
        # patches, so img_len = (H/16) * (W/16) = latent_h * latent_w.
        # If observed img_len doesn't match, try 4x packing.
        expected = latent_h * latent_w

        for layer_name, entries in by_layer.items():
            num_steps = len(entries)
            if num_steps == 0:
                continue

            # Stack across timesteps: (T, B, img_len, txt_len)
            stacked = torch.stack([e["attn"] for e in entries], dim=0)[:, 0]
            T_steps, img_len, txt_len = stacked.shape

            # Figure out spatial reshape
            if img_len == expected:
                lh, lw = latent_h, latent_w
            else:
                side = int(round(math.sqrt(img_len)))
                if side * side == img_len:
                    lh, lw = side, side
                else:
                    continue  # skip layers whose img_len doesn't factor

            for verb, token_indices in verb_token_indices.items():
                if not token_indices:
                    continue
                valid = [i for i in token_indices if i < txt_len]
                if not valid:
                    continue
                # (T, img_len)
                verb_attn = stacked[..., valid].mean(dim=-1)
                verb_spatial = verb_attn.reshape(T_steps, lh, lw)
                verb_block_lists[verb].append(verb_spatial.float().numpy())

        verb_maps_avg: Dict[str, np.ndarray] = {}
        verb_maps_per_step: Optional[Dict[str, np.ndarray]] = (
            {} if store_per_timestep else None
        )

        for verb in verb_token_indices:
            arrays = verb_block_lists[verb]
            if not arrays:
                verb_maps_avg[verb] = np.zeros((latent_h, latent_w), dtype=np.float32)
                if store_per_timestep:
                    verb_maps_per_step[verb] = np.zeros((1, latent_h, latent_w), dtype=np.float32)
                continue
            stack = np.stack(arrays, axis=0)  # (B_blocks, T_steps, H, W)
            block_avg = stack.mean(axis=0)
            map_avg = block_avg.mean(axis=0)
            vmin, vmax = float(map_avg.min()), float(map_avg.max())
            if vmax > vmin:
                map_avg = (map_avg - vmin) / (vmax - vmin)
            verb_maps_avg[verb] = map_avg
            if store_per_timestep:
                verb_maps_per_step[verb] = block_avg

        return verb_maps_avg, verb_maps_per_step, num_blocks

    def extract_batch(
        self,
        prompts: List[str],
        verbs_per_prompt: List[List[str]],
        **kwargs,
    ) -> List[VerbAttentionResult]:
        results = []
        for i, (prompt, verbs) in enumerate(zip(prompts, verbs_per_prompt)):
            print(f"  [{i+1}/{len(prompts)}] {prompt}")
            result = self.extract(prompt, verbs, **kwargs)
            results.append(result)
        return results

    @property
    def pipe(self):
        self._ensure_initialized()
        return self._pipe
