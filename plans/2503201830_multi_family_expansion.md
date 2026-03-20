# Plan: Multi-Family Model Expansion

**Date:** 2026-03-20 18:30
**Status:** Active
**Branch:** multi-family-models

## Goal

Expand the hybrid BP head analysis beyond the GPT-2 family to models
from different companies and training paradigms. Test whether the
three-phase structure and boolean-AND scaling behavior are universal
properties of transformer language models or GPT-2-specific artifacts.

## Target Models

### Family 1: GPT-2 (OpenAI) — already done
    gpt2        124M  ✓ collected
    gpt2-medium 345M  ✓ collected
    gpt2-xl     1.5B  pending (zero new code needed)

### Family 2: Mistral (Mistral AI, France)
    Mistral-7B-v0.1   7B   ~14GB  probably too big for Mac
    mistral-7b-v0.3   7B   ~14GB  too big
    -- better option --
    Mistral-Nemo-Base-2407  12B  too big
    -- small option --
    Looking for a <4GB Mistral variant

### Family 3: Qwen (Alibaba, China)
    Qwen2.5-0.5B   0.5B  ~1GB   fits easily
    Qwen2.5-1.5B   1.5B  ~3GB   fits
    Qwen2.5-3B     3B    ~6GB   fits
    Architecture: Qwen2, different from GPT-2, needs new forward pass

### Family 4: DeepSeek (DeepSeek AI, China)
    DeepSeek-R1-Distill-Qwen-1.5B  1.5B  ~3.5GB  fits
    Architecture: Qwen2 (same as Qwen family above)
    Training: reasoning distillation from R1 — very different objective
    This is the most interesting comparison: same architecture as Qwen
    but trained for reasoning. Does the head typology differ?

## Architecture Notes

**GPT-2 family:** forward pass already implemented. Zero new code.

**Qwen2 family** (covers both Qwen2.5 and DeepSeek-R1-Distill):
- Different weight naming convention
- Rotary positional embeddings (RoPE) instead of learned positional
- Grouped query attention (GQA) — fewer KV heads than Q heads
- SiLU activation instead of GELU
- RMSNorm instead of LayerNorm
- New forward pass needed but single implementation covers both
  Qwen2.5 and DeepSeek-R1-Distill (same architecture)

**Mistral family:**
- Also uses RoPE, GQA, SiLU, RMSNorm — very similar to Qwen2
- Sliding window attention in some variants
- May share forward pass implementation with Qwen2

## Implementation Plan

### Trail 1: GPT-2 XL
- Zero new code
- Just download and collect
- Completes GPT-2 family scaling curve
- Expected: boolean-AND stays at 9

### Trail 2: Qwen2 forward pass
- Implement Qwen2/LLaMA-style forward pass in core/forward_qwen2.py
- Add model configs for Qwen2.5-0.5B and Qwen2.5-1.5B
- Test with forward command
- Covers both Qwen and DeepSeek with one implementation

### Trail 3: Qwen2.5-0.5B and Qwen2.5-1.5B
- Download, collect, analyze
- First cross-company comparison
- Does three-phase structure appear in Alibaba model?

### Trail 4: DeepSeek-R1-Distill-Qwen-1.5B
- Same architecture as Qwen2, same forward pass
- Different training: reasoning distillation
- Compare head typology to Qwen2.5-1.5B at same size
- Key question: does reasoning training change the boolean-AND count?

### Trail 5: Mistral (if small variant found)
- Find a Mistral variant under 4GB
- Implement if architecture differs from Qwen2
- European model, different training data

## Key Questions This Plan Answers

1. Does boolean-AND = 9 hold across GPT-2 XL (1.5B)?
2. Does the three-phase structure appear in Qwen2 models?
3. Does reasoning training (DeepSeek) change the head typology
   vs standard language modeling (Qwen)?
4. Is the pattern consistent across companies and training paradigms?

## Success Criteria

- At least 3 model families analyzed
- Same 88-prompt all index used for every model
- Cross-family comparison report
- Clear answer on boolean-AND scaling behavior

## Disk Budget

    Current used:   175GB / 228GB
    Available:       27GB
    
    GPT-2 XL:        ~6GB weights + 80MB patterns
    Qwen2.5-0.5B:    ~1GB weights + 10MB patterns
    Qwen2.5-1.5B:    ~3GB weights + 30MB patterns
    DeepSeek-1.5B:   ~3.5GB weights + 30MB patterns
    Total new:       ~14GB — fits comfortably

## Notes

The Qwen2 forward pass is the main engineering task. Once it works
for Qwen2.5-0.5B it works for Qwen2.5-1.5B and DeepSeek-R1-Distill
at no extra cost — same architecture. This is the high-leverage
implementation investment.

Mistral is lower priority — if Qwen2 forward pass is clean, Mistral
may share enough of the architecture to reuse it with minor changes.