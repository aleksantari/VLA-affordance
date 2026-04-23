"""
Flux Cross-Attention Extraction for Verb-Spatial Binding Analysis

Extracts cross-attention maps from Flux text-to-image diffusion model
during denoising, isolating verb-token attention to measure spatial
binding to functional object regions.

Flux uses MMDiT (Multimodal Diffusion Transformer) with:
- Double-Stream blocks: separate text/image streams with cross-connections
- Single-Stream blocks: concatenated text+image joint attention

We hook into both block types to extract the Image-Query × Text-Key
attention slice, focusing on verb token positions.

Dependencies:
    pip install diffusers attention-map-diffusers accelerate

Reference:
    Zhang et al., "Probing and Bridging Geometry-Interaction Cues
    for Affordance Reasoning in Vision Foundation Models", CVPR 2026.
"""

import re
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class VerbAttentionResult:
    """Result of verb attention extraction for a single image."""
    prompt: str
    verbs: List[str]
    verb_token_indices: Dict[str, List[int]]
    # Verb -> (H_latent, W_latent) attention heatmap, averaged across timesteps/layers
    verb_attention_maps: Dict[str, np.ndarray]
    # Verb -> (num_timesteps, H_latent, W_latent) per-timestep maps
    verb_attention_per_timestep: Optional[Dict[str, np.ndarray]] = None
    # Generated image (PIL)
    generated_image: Optional[object] = None
    # Raw metadata
    num_timesteps: int = 0
    num_blocks: int = 0
    latent_size: Tuple[int, int] = (64, 64)


class FluxVerbAttentionExtractor:
    """
    Extract verb-specific cross-attention maps from Flux during denoising.
    
    Usage:
        extractor = FluxVerbAttentionExtractor(model_name="black-forest-labs/FLUX.1-schnell")
        result = extractor.extract("a person cutting bread with a knife", verbs=["cutting"])
        # result.verb_attention_maps["cutting"] -> (64, 64) heatmap
    """
    
    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.1-schnell",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model ID for Flux
            device: Device to load model on
            dtype: Model precision (bfloat16 recommended for A100)
            enable_cpu_offload: Use sequential CPU offload for low VRAM
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.is_schnell = "schnell" in model_name.lower()
        
        # Lazy load — don't load model until first extraction
        self._pipe = None
        self._initialized = False
        self._enable_cpu_offload = enable_cpu_offload
    
    def _ensure_initialized(self):
        """Lazy initialization of Flux pipeline with attention hooks."""
        if self._initialized:
            return
        
        from diffusers import FluxPipeline
        from attention_map_diffusers import init_pipeline
        
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
        
        # Register attention extraction hooks
        self._pipe = init_pipeline(self._pipe)
        
        self._initialized = True
        print(f"✓ Flux pipeline ready")
    
    def get_verb_token_indices(
        self,
        prompt: str,
        verbs: List[str],
    ) -> Dict[str, List[int]]:
        """
        Find token positions for specified verbs in the tokenized prompt.
        
        Flux uses two tokenizers (CLIP + T5). We focus on the T5 tokenizer
        since that's what drives the cross-attention in the transformer blocks.
        
        Args:
            prompt: The full text prompt
            verbs: List of verb strings to find (e.g., ["cutting", "holding"])
            
        Returns:
            Dict mapping each verb to its token indices in the T5 encoding
        """
        self._ensure_initialized()
        
        # Flux has tokenizer (CLIP) and tokenizer_2 (T5)
        tokenizer = self._pipe.tokenizer_2  # T5 tokenizer
        
        # Encode the full prompt
        encoding = tokenizer(
            prompt,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        
        verb_indices = {}
        prompt_lower = prompt.lower()
        
        for verb in verbs:
            verb_lower = verb.lower()
            indices = []
            
            # Strategy 1: Match via offset mapping if available
            if "offset_mapping" in encoding and encoding["offset_mapping"]:
                offsets = encoding["offset_mapping"]
                # Find where the verb appears in the original text
                verb_start = prompt_lower.find(verb_lower)
                if verb_start >= 0:
                    verb_end = verb_start + len(verb_lower)
                    for tok_idx, (start, end) in enumerate(offsets):
                        if start < verb_end and end > verb_start:
                            indices.append(tok_idx)
            
            # Strategy 2: Fallback — match decoded tokens
            if not indices:
                for tok_idx, token in enumerate(tokens):
                    # T5 tokens often have ▁ prefix for word boundaries
                    clean_token = token.replace("▁", "").lower()
                    if clean_token and (
                        verb_lower.startswith(clean_token) or
                        clean_token in verb_lower
                    ):
                        indices.append(tok_idx)
            
            verb_indices[verb] = indices
            
            if not indices:
                print(f"  ⚠ Could not find verb '{verb}' in tokens: {tokens}")
        
        return verb_indices
    
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
        """
        Run Flux inference and extract verb-specific cross-attention maps.
        
        Args:
            prompt: Text prompt for image generation
            verbs: Verbs to extract attention for
            num_inference_steps: Denoising steps (default: 4 for schnell, 20 for dev)
            guidance_scale: CFG scale (0.0 for schnell, 3.5 for dev)
            height: Image height
            width: Image width
            seed: Random seed for reproducibility
            store_per_timestep: Whether to store attention maps per timestep
            
        Returns:
            VerbAttentionResult with attention maps and metadata
        """
        self._ensure_initialized()
        
        from attention_map_diffusers import attn_maps
        
        # Default inference settings
        if num_inference_steps is None:
            num_inference_steps = 4 if self.is_schnell else 20
        if self.is_schnell:
            guidance_scale = 0.0  # schnell doesn't use CFG
        
        # Get verb token positions
        verb_token_indices = self.get_verb_token_indices(prompt, verbs)
        
        # Clear any previous attention maps
        attn_maps.clear()
        
        # Run inference
        generator = torch.Generator(device=self.device).manual_seed(seed) if seed else None
        
        output = self._pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        generated_image = output.images[0]
        
        # Latent spatial dimensions
        latent_h = height // 16  # Flux uses 16x downsampling for patch embedding
        latent_w = width // 16
        # But the actual latent is packed differently — 
        # Flux packs 2x2 patches, so effective grid is height//16 x width//16
        # Let's compute from the attention map shapes
        
        # Extract verb-specific attention from stored maps
        verb_maps_avg, verb_maps_per_step = self._process_attention_maps(
            attn_maps,
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
            latent_size=(latent_h, latent_w),
        )
    
    def _process_attention_maps(
        self,
        attn_maps_store,
        verb_token_indices: Dict[str, List[int]],
        latent_h: int,
        latent_w: int,
        store_per_timestep: bool,
    ) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """
        Process raw attention maps from attention-map-diffusers hooks.
        
        The stored attention maps have structure depending on the library version.
        We handle the common formats:
        - Dict[layer_name] -> Tensor of shape (num_heads, seq_len, seq_len)
        - Or accumulated across timesteps
        
        We need to:
        1. Identify which part of seq_len is image vs text tokens
        2. Slice out image_query x text_key (cross-attention portion)
        3. Index into text_key at verb token positions
        4. Reshape image dimension to spatial grid
        5. Average across heads and blocks
        """
        verb_maps_all_steps = {verb: [] for verb in verb_token_indices}
        
        # Process attention maps — the exact structure depends on 
        # attention-map-diffusers version. We try multiple formats.
        
        if hasattr(attn_maps_store, 'items'):
            # Dict-like access
            raw_maps = dict(attn_maps_store)
        elif hasattr(attn_maps_store, '__iter__'):
            raw_maps = {f"block_{i}": m for i, m in enumerate(attn_maps_store)}
        else:
            print("⚠ Unexpected attention map format. Returning empty maps.")
            empty = {v: np.zeros((latent_h, latent_w)) for v in verb_token_indices}
            return empty, None
        
        for layer_name, attn_tensor in raw_maps.items():
            if not isinstance(attn_tensor, torch.Tensor):
                continue
            
            attn = attn_tensor.float().cpu()
            
            # Expected shapes vary:
            # (heads, total_seq, total_seq) — single timestep
            # (timesteps, heads, total_seq, total_seq) — accumulated
            # (total_seq, total_seq) — already head-averaged
            
            if attn.dim() == 2:
                attn = attn.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S)
            elif attn.dim() == 3:
                attn = attn.unsqueeze(0)  # (1, heads, S, S)
            
            # attn: (T, H, S, S) where S = num_image_tokens + num_text_tokens
            T, H, S, _ = attn.shape
            
            # Determine image/text split
            # Flux: image tokens come first, then text tokens
            num_image_tokens = latent_h * latent_w
            num_text_tokens = S - num_image_tokens
            
            if num_text_tokens <= 0:
                # This might be a self-attention-only block, skip
                continue
            
            # Extract cross-attention: image_query x text_key
            # attn[:, :, :num_image, num_image:] = how image tokens attend to text
            cross_attn = attn[:, :, :num_image_tokens, num_image_tokens:]
            # cross_attn: (T, H, num_image_tokens, num_text_tokens)
            
            # Average across heads
            cross_attn = cross_attn.mean(dim=1)  # (T, num_image_tokens, num_text_tokens)
            
            for verb, token_indices in verb_token_indices.items():
                if not token_indices:
                    continue
                
                # Clamp indices to valid range
                valid_indices = [
                    idx for idx in token_indices 
                    if idx < num_text_tokens
                ]
                
                if not valid_indices:
                    continue
                
                # Extract attention for verb tokens and average
                verb_attn = cross_attn[:, :, valid_indices].mean(dim=-1)
                # verb_attn: (T, num_image_tokens)
                
                # Reshape to spatial grid
                verb_spatial = verb_attn.reshape(T, latent_h, latent_w)
                # verb_spatial: (T, latent_h, latent_w)
                
                verb_maps_all_steps[verb].append(verb_spatial.numpy())
        
        # Aggregate across blocks
        verb_maps_avg = {}
        verb_maps_per_step = {} if store_per_timestep else None
        
        for verb in verb_token_indices:
            if verb_maps_all_steps[verb]:
                # Stack across blocks and average
                stacked = np.stack(verb_maps_all_steps[verb], axis=0)
                # stacked: (num_blocks, T, H, W)
                
                # Average across blocks
                block_avg = stacked.mean(axis=0)  # (T, H, W)
                
                # Average across timesteps for the summary map
                verb_maps_avg[verb] = block_avg.mean(axis=0)  # (H, W)
                
                # Normalize to [0, 1]
                vmin, vmax = verb_maps_avg[verb].min(), verb_maps_avg[verb].max()
                if vmax > vmin:
                    verb_maps_avg[verb] = (verb_maps_avg[verb] - vmin) / (vmax - vmin)
                
                if store_per_timestep:
                    verb_maps_per_step[verb] = block_avg  # (T, H, W)
            else:
                verb_maps_avg[verb] = np.zeros((latent_h, latent_w))
                if store_per_timestep:
                    verb_maps_per_step[verb] = np.zeros((1, latent_h, latent_w))
        
        return verb_maps_avg, verb_maps_per_step
    
    def extract_batch(
        self,
        prompts: List[str],
        verbs_per_prompt: List[List[str]],
        **kwargs,
    ) -> List[VerbAttentionResult]:
        """
        Extract verb attention for multiple prompts sequentially.
        
        Note: Flux doesn't support true batch inference for attention extraction
        (each prompt needs its own attention maps), so this is a convenience
        wrapper for sequential extraction.
        """
        results = []
        for i, (prompt, verbs) in enumerate(zip(prompts, verbs_per_prompt)):
            print(f"  [{i+1}/{len(prompts)}] {prompt}")
            result = self.extract(prompt, verbs, **kwargs)
            results.append(result)
        return results
    
    @property
    def pipe(self):
        """Access the underlying Flux pipeline."""
        self._ensure_initialized()
        return self._pipe
