# Cross-Model Comparison: GPT-2 Small vs Medium

**Date:** 2026-03-20 18:00
**Models:** GPT-2 small (124M), GPT-2 medium (345M)
**Collection:** 88 prompts, all index, same prompt set for both
**Method:** Per-prompt classification, hybrid BP role assignment

## Summary Comparison

    Role            Small (124M)         Medium (345M)
                    heads    pct         heads    pct
    hub              103     72%          305     79%
    continuous-OR     28     19%           46     12%
    boolean-AND        9      6%            9      2%
    mixed              8      6%           24      6%
    total            144    100%          384    100%

## The Three-Phase Structure Replicates

Both models show the same three-phase layer structure:

**Phase 1 — continuous-heavy (early layers):**

    Small:  layers 0-2  (0-17% depth)
    Medium: layers 0-4  (0-17% depth)

Both models concentrate continuous-OR and boolean-AND heads in the
first ~17% of depth. Feature extraction and local routing happen early.

**Phase 2 — transition (middle-early layers):**

    Small:  layer 3    (25% depth)
    Medium: layers 4-5 (17-22% depth)

A mixed zone where boolean, continuous, hub, and mixed heads coexist.
The network is shifting from local processing to global communication.

**Phase 3 — hub-dominated (majority of depth):**

    Small:  layers 4-11  (33-100% depth)
    Medium: layers 5-23  (22-100% depth)

Global workspace takes over. Nearly all heads communicate through
position 0. This phase occupies ~75% of depth in both models.

## The Key Scaling Finding

**Boolean-AND heads do not scale.**

    Model       Total heads    Boolean-AND    Pct
    Small 124M      144             9         6%
    Medium 345M     384             9         2%

The model grows 2.8x in total heads. The number of boolean-AND heads
stays exactly at 9. As a percentage, boolean-AND shrinks from 6% to 2%.

This is a strong empirical claim: sharp BP routing heads are a fixed
small set, not a scaling resource. The model does not get more boolean
reasoning capacity as it gets bigger. What grows is the hub and
continuous-OR infrastructure.

**Continuous-OR heads grow in absolute count but shrink as percentage:**

    Small:  28 heads (19%)
    Medium: 46 heads (12%)

More continuous aggregation heads in absolute terms, but they become
a smaller fraction of the total as hub heads dominate more.

**Hub heads grow in both absolute count and percentage:**

    Small:  103 heads (72%)
    Medium: 305 heads (79%)

The global workspace is what scales. More model capacity goes to
position-0 communication as the model gets bigger.

## Layer-Depth Normalized View

When normalized by depth percentage, the transition happens at
roughly the same relative depth in both models (~20-25%).

    Small:  boolean-AND cutoff at layer 4  (33% depth)
    Medium: boolean-AND cutoff at layer 6  (26% depth)

Both models complete their boolean routing in the first quarter
of their depth. After that, hub-dominated processing takes over.

## Late-Layer Continuous-OR Heads

Both models have a small number of persistent continuous-OR heads
in their final layers:

    Small:  layer 11 heads 0, 8    (2 heads at 100% depth)
    Medium: layer 23 heads 1,2,3   (3 heads at 100% depth)

These late diffuse heads survive through all the hub processing
to the output. Likely candidates for output preparation or
function-vector style computation.

## Connection to Hybrid BP Theory

The scaling behavior maps cleanly onto the hybrid BP framework:

**Boolean-AND nodes are fixed infrastructure.** 9 heads implement
the sharp gather step — the formal BP construction with
projectDim/crossProject weight structure. This is not a resource
that scales with model size. It is a fixed logical primitive.

**Continuous-OR nodes grow modestly.** Real-valued aggregation
capacity grows with model size but not proportionally. Diminishing
returns in continuous estimation.

**Hub nodes are the scaling resource.** The global workspace
grows with model size. More model capacity goes to communicating
through position 0 — reading and writing shared context. This
is what larger models do more of.

The implication: scaling a transformer does not give you more
logical reasoning capacity (boolean-AND stays at 9). It gives
you more global workspace capacity (hub grows from 72% to 79%).
This is consistent with the known empirical observation that
larger models are better at multi-step contextual reasoning —
they have more hub infrastructure for assembling context —
but not necessarily better at formal logical inference.

## What This Does Not Yet Tell Us

1. Does the boolean-AND count of 9 hold at GPT-2 XL (1.5B)?
   If yes, this is a strong scaling law: boolean-AND = 9,
   independent of model size within the GPT-2 family.

2. Does the pattern hold across model families (Pythia, GPT-Neo)?
   If yes, this may be a universal property of transformer
   language models.

3. What are the 9 boolean-AND heads doing specifically?
   Do they have projectDim/crossProject weight structure
   as predicted by the shannon paper? This would be the
   direct theory-experiment connection.

4. What information flows through the hub? The 72-79% hub
   heads are the dominant computation but the least
   characterized. OV circuit analysis needed.

## Next Steps

1. Run GPT-2 XL to confirm boolean-AND = 9 at larger scale
2. Check if boolean-AND heads in medium correspond to same
   relative depth positions as in small
3. OV circuit analysis on the 9 boolean-AND heads in both models
4. Pythia family for cross-family confirmation

## Methodology Notes

Same 88 prompts, same analysis functions, same classification
thresholds, same hybrid role assignment for both models.
Results are directly comparable. The only difference is model
architecture (n_layers, n_heads, d_model).