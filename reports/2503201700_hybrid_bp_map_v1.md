# Hybrid BP Map: GPT-2 Small v1

**Date:** 2026-03-20 17:00
**Model:** GPT-2 small (12 layers, 12 heads, 144 heads total)
**Collection:** 88 prompts, all index
**Method:** Per-prompt classification mapped to hybrid BP roles

## Summary

    Role            Heads    Pct
    hub              103     72%
    continuous-OR     28     19%
    boolean-AND        9      6%
    mixed              8      6%

## Layer Summary

    Layer   boolean-AND   continuous-OR   hub   mixed   character
    0             3             8          0      1     continuous-heavy
    1             1             4          7      0     continuous-heavy
    2             1             7          3      1     continuous-heavy
    3             2             2          6      2     mixed
    4             1             0          8      3     hub-dominated
    5             0             0         12      0     hub-dominated
    6             0             0         11      1     hub-dominated
    7             0             0         12      0     hub-dominated
    8             0             0         12      0     hub-dominated
    9             0             0         12      0     hub-dominated
    10            0             0         12      0     hub-dominated
    11            0             2         10      0     hub-dominated

## The Main Finding

GPT-2 small has a clear three-phase structure:

**Phase 1 — Early processing (layers 0-2): continuous-heavy.**
Dominated by continuous-OR heads doing real-valued aggregation.
Boolean-AND heads are present but sparse — a few sharp routing
heads doing local feature detection. This is where raw token
embeddings get processed into useful representations before any
structured cross-token reasoning begins.

**Phase 2 — Transition (layer 3): mixed.**
Boolean, continuous, hub, and mixed heads all appear. The network
is shifting from local feature extraction to global communication.
The prev-token heads in this layer (3.2, 3.7) are doing BP-style
routing at the exact moment the network transitions to hub mode.

**Phase 3 — Hub-dominated (layers 4-11): global workspace.**
The global workspace takes over almost completely. Nearly all heads
communicate through position 0. Once early processing is done,
the model's primary mechanism is reading from and writing to the
shared communication channel at position 0.

## Boolean-AND Heads: The Complete List

All sharp routing (BP gather step) heads in GPT-2 small:

    Layer 0, Head 1  — self-attn  100%  (local feature detection)
    Layer 0, Head 3  — self-attn  100%  (local feature detection)
    Layer 0, Head 5  — self-attn  100%  (local feature detection)
    Layer 1, Head 11 — self-attn  100%  (local feature detection)
    Layer 2, Head 2  — prev-token 100%  (BP gather, neighbor routing)
    Layer 3, Head 2  — prev-token  91%  (BP gather, neighbor routing)
    Layer 3, Head 7  — prev-token  61%  (BP gather, neighbor routing)
    Layer 4, Head 11 — prev-token 100%  (BP gather, neighbor routing)
    (Layer 4, Head 0 — sharp       50%  borderline, classified mixed)

After layer 4: zero boolean-AND heads. All sharp routing happens
in the first half of the network.

## Persistent Continuous-OR Heads

Most continuous-OR heads appear in layers 0-2. Two persist to
the output:

    Layer 11, Head 0  — diffuse 99%
    Layer 11, Head 8  — diffuse 100%

These late-layer diffuse heads are doing something different from
the hub pattern at the output stage. Possible roles: output
preparation, continuous quantity summarization, or the function
vector style computation identified in the interpretability
literature as dominant at larger scales.

## Connection to Hybrid BP Theory

The layer structure maps cleanly onto the hybrid BP framework
from the interp repo:

**Boolean nodes** (sharp, discrete, exact posteriors):
Concentrated in layers 0-4. These are the heads the formal BP
proofs apply to directly — the projectDim/crossProject construction,
the no-hallucination corollary, the exact inference guarantee.
9 heads total (6%).

**Continuous nodes** (diffuse, real-valued, weighted aggregation):
Concentrated in layers 0-2, two survivors in layer 11.
These are the heads the hybrid BP extension needs to cover —
what is the updateBelief analog for a continuous node?
28 heads total (19%).

**Hub nodes** (global workspace, position 0 communication):
Dominant from layer 1 onward. These are doing something the
current theory does not fully characterize — reading vs writing,
what information flows through position 0 and when.
103 heads total (72%).

## What This Does Not Yet Tell Us

1. **Hub head distinction.** 72% hub is a single category that
   likely contains readers (gathering from position 0) and writers
   (sending to position 0). Need OV circuit analysis to split.

2. **The transition at layer 3.** Why does the network shift from
   continuous-heavy to hub-dominated at layer 3-4? What changes
   at that depth?

3. **The late diffuse heads.** Layer 11 heads 0 and 8 are
   persistently diffuse at the output. What are they computing?
   Are these the function vector heads that dominate at scale?

4. **No induction heads found.** The classic two-layer induction
   circuit (Olsson et al.) was not surfaced. Current prompts lack
   repeated token patterns needed to activate induction behavior.

## Methodology

- 88 prompts across 9 categories (general, factual, narrative,
  logic, questions, dialogue, causal, temporal, spatial)
- Per-prompt classification into 5 types, aggregated to distribution
- Hybrid role assigned by dominant type (>60% threshold)
- Heads below 60% dominance classified as mixed
- Analysis functions: entropy, max_attn, prev_tok, first_tok, diagonal
- No ground truth labels — fully unsupervised

## Repo State

Collection saved to ~/.cache/psychic/gpt2_all_patterns/ (16MB).
All analysis runs from disk in seconds. Code in src/psychic/.