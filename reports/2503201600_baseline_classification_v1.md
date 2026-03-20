# Baseline Head Classification: GPT-2 Small v1

**Date:** 2026-03-20 16:00
**Model:** GPT-2 small (124M parameters, 12 layers, 12 heads, 144 heads total)
**Prompts:** 88 prompts across 9 categories (all index)
**Method:** Per-prompt classification, aggregate type distribution

## Summary Distribution

    Type          Heads    Pct
    first-token   103      72%
    diffuse        28      19%
    self-attn       5       3%
    prev-token      5       3%
    sharp           3       2%

## Stable Specialized Heads

These heads show consistent behavior across all 88 prompts.

**Prev-token heads:**

    Layer 2, Head 2  — 100% prev-token
    Layer 3, Head 2  —  91% prev-token
    Layer 3, Head 7  —  61% prev-token
    Layer 4, Head 11 — 100% prev-token

**Self-attn heads (layer 0 only):**

    Layer 0, Head 1  — 100% self-attn
    Layer 0, Head 3  — 100% self-attn
    Layer 0, Head 5  — 100% self-attn

**Sharp heads:**

    Layer 4, Head 0  —  50% sharp
    Layer 4, Head 3  —  47% sharp

**Persistent diffuse heads:**

    Layer 11, Head 0  —  99% diffuse
    Layer 11, Head 8  — 100% diffuse
    Layer 0, Head 10  —  99% diffuse
    Layer 1, Head 10  — 100% diffuse

## Stability Check

Distribution at 38 prompts vs 88 prompts:

    Type          38 prompts    88 prompts    Delta
    first-token       76%           72%        -4%
    diffuse           16%           19%        +3%
    self-attn          3%            3%         0%
    prev-token         3%            3%         0%
    sharp              1%            2%        +1%

Distribution is stable. Results are believable at this sample size.
Adding more prompts is unlikely to change the dominant picture.

## Connection to BP Framework

**Prev-token heads** are the clearest match to BP routing. They implement
a sharp gather step — attending to exactly one neighbor (the previous token)
and copying its value into the residual stream. This is precisely the
projectDim/crossProject construction from the transformer-bp-lean proof.

**Self-attn heads in layer 0** are consistent with local feature detection —
reading the current token's own embedding before any cross-token communication
has occurred. In BP terms this is a self-loop or prior update.

**Persistent diffuse heads** (especially layers 0 and 11) are the open
question. Layer 0 diffuse heads appear before any meaningful cross-token
communication has happened. Layer 11 diffuse heads appear at the output.
These may correspond to the function-vector heads identified in the
interpretability literature — doing something other than sharp BP routing.

**First-token heads** (72%) are most likely the global workspace pattern —
using position 0 as a shared communication hub. In BP terms, position 0
functions as a hub factor node connected to all positions. These heads
are either writing local information to position 0 or reading global
context from it. Distinguishing readers from writers requires OV circuit
analysis (value matrix inspection), not just attention pattern analysis.

## What This Study Does Not Yet Tell Us

1. **Reader vs writer first-token heads.** 72% first-token is a single
   category that likely contains two distinct functional types. Need OV
   circuit analysis to split them.

2. **Induction heads.** Not found with current prompts. Induction heads
   require repeated token patterns [A][B]...[A] to fire. Current prompt
   set has no deliberate repetition.

3. **Layer-by-layer information flow.** We know which heads do what but
   not how information flows between layers. A full picture requires
   understanding what each head reads from and writes to the residual
   stream.

4. **Per-category variation.** Do reasoning prompts produce different
   head behavior than narrative prompts? We have the infrastructure to
   test this (indexes) but have not yet done the comparison.

## Methodology Notes

- Analysis functions: entropy, max_attn, prev_tok, first_tok, diagonal
- Classification thresholds: prev_tok > 0.4, first_tok > 0.5,
  diagonal > 0.5, max_attn > 0.6, else diffuse
- Thresholds were set by inspection, not optimized
- No ground truth labels — classification is our own typology

## Repo State

All code in `src/psychic/`. Prompts in `src/psychic/prompts/data/`.
Indexes in `src/psychic/prompts/indexes/`. Forward pass implemented
from scratch in numpy, no torch dependency. Weights loaded directly
from safetensors format.