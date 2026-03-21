# Qwen 0.5B Analysis: First Cross-Family Comparison

**Date:** 2026-03-20 22:00
**Server:** psychic1 (Vultr small, persistent disk psychic-data)
**Models:** Qwen 2.5 0.5B (500M)
**Collection:** 88 prompts, all index
**Method:** Per-prompt classification, hybrid BP role assignment

## Context

This is the first result from the new persistent experiment server.
GPT-2 small and medium results are from earlier today (local machine).
Qwen 0.5B is the first cross-family data point.

## Qwen 0.5B Profile

    Role            Heads    Pct
    hub               203    60%
    continuous-OR      59    18%
    boolean-AND        29     9%
    mixed              45    13%
    total             336   100%

    Architecture: 24 layers, 14 heads per layer

## Cross-Family Comparison

    Role            Small (124M)   Medium (345M)   Qwen 0.5B (500M)
    hub               103 (72%)      305 (79%)        203 (60%)
    continuous-OR      28 (19%)       46 (12%)         59 (18%)
    boolean-AND          9  (6%)        9  (2%)         29  (9%)
    mixed                8  (6%)       24  (6%)         45 (13%)
    total              144            384              336

## Key Finding: boolean-AND Does Not Transfer Across Families

Within the GPT-2 family, boolean-AND = 9 held exactly across
two model sizes (124M and 345M). Qwen 0.5B breaks this: 29
boolean-AND heads, more than 3x the GPT-2 count.

Two interpretations:

1. Architecture-specific: Qwen was trained with different
   objectives or initialization and genuinely develops more
   sharp routing heads.

2. Threshold sensitivity: The classification boundary may
   behave differently on Qwen weight distributions, inflating
   the boolean-AND count.

Distinguishing these requires either OV circuit analysis on
the 29 heads or running Qwen 1.5B to check if the count
scales within the Qwen family.

## Layer Structure

    Layer   Depth%   boolean-AND   continuous-OR   hub   mixed   character
        0       0%             5               9     0       0   boolean-heavy
        1       4%             2              11     0       1   continuous-heavy
        2       9%             2               6     0       6   continuous-heavy
        3      13%             1               1    12       0   hub-dominated
        4      17%             2               3     9       0   continuous-heavy
        5      22%             1               1    11       1   hub-dominated
        6      26%             2               2    10       0   hub-dominated
        7      30%             1               1    11       1   hub-dominated
        8      35%             2               2     5       5   mixed
        9      39%             0               1    12       1   hub-dominated
       10      43%             2               1    10       1   hub-dominated
       11      48%             0               0    13       1   hub-dominated
       12      52%             1               4     8       1   continuous-heavy
       13      57%             1               2    11       0   hub-dominated
       14      61%             2               3     9       0   continuous-heavy
       15      65%             1               1    11       1   hub-dominated
       16      70%             0               0    14       0   hub-dominated
       17      74%             0               1    13       0   hub-dominated
       18      78%             0               0    11       3   hub-dominated
       19      83%             1               4     8       1   continuous-heavy
       20      87%             0               2    12       0   hub-dominated
       21      91%             1               0    13       0   hub-dominated
       22      96%             0               2     0      12   mixed
       23     100%             2               2     0      10   mixed

## Three Structural Features

**Early boolean/continuous zone (layers 0-2):**
Same pattern as GPT-2. First ~10% of depth does feature
extraction and local routing. This appears universal.

**Hub highway (layers 3-21):**
Same as GPT-2 but less dominant — 60% hub vs 72-79% in GPT-2.
Qwen distributes more capacity to boolean-AND and mixed roles
throughout the middle layers rather than collapsing fully to hub.

**Late-layer collapse to mixed (layers 22-23):**
Unique to Qwen so far. GPT-2 models end with a small number of
continuous-OR heads at final depth. Qwen ends with 22 mixed heads
across the last two layers. This is a qualitatively different
output preparation strategy.

## Connection to Hybrid BP Theory

The three-phase structure (early extraction, hub highway, late
output preparation) replicates across families. This supports
the hybrid BP framework as a universal description.

The boolean-AND divergence is the open question. If Qwen genuinely
has more sharp routing heads, it would suggest the number of
boolean-AND heads is a training/architecture choice rather than
a fixed logical primitive. If it is a threshold artifact, the
GPT-2 finding of boolean-AND = 9 may still generalize.

## Open Questions

1. Does Qwen 1.5B have ~29 boolean-AND heads or does it scale?
   If stable at 29, boolean-AND is a Qwen family constant like
   GPT-2's 9.

2. Does the late-layer mixed collapse replicate in Qwen 1.5B?
   If yes, this is a Qwen architectural signature.

3. OV circuit analysis on Qwen's 29 boolean-AND heads vs GPT-2's 9.
   Do they have the same projectDim/crossProject weight structure?

4. GPT-2 XL (1.5B): does boolean-AND stay at 9?
   If yes, boolean-AND = 9 is a strong GPT-2 family constant
   across 124M, 345M, and 1.5B.

## Next Steps

    1. Collect Qwen 1.5B on server
    2. Collect GPT-2 XL on server
    3. Update cross-family comparison with all four data points
    4. OV circuit analysis on boolean-AND heads
