# Classification Results: First Attempt

**Date:** 2026-03-20 14:30

## What We Tried

Two approaches to classifying attention heads in GPT-2 small (12 layers,
12 heads, 144 heads total) using 38 prompts across 4 categories: general,
factual, logic, narrative.

### Approach 1: Average-then-classify

For each head, compute analysis scores averaged across all prompts and all
token positions, then classify the head based on the average.

Result: 82% first-token, 10% diffuse, 6% self-attn, 3% prev-token.
The attention sink dominated everything.

### Approach 2: Classify-then-count (per-prompt)

For each (head, prompt) pair, compute scores on the full attention pattern
matrix, classify into a type, then aggregate counts across prompts.
Report as a distribution.

Result: much better signal. Summary of dominant types:

    prev-token:  5 heads  (3.5%)  — layers 2, 3, 4
    self-attn:   4 heads  (2.8%)  — layer 0 only
    sharp:       2 heads  (1.4%)  — layers 4
    first-token: 110 heads (76%)  — layers 1-11
    diffuse:     23 heads (16%)   — layers 0, 6, 11

## Key Findings

**Prev-token heads found:**
- Layer 2, head 2 — 100% prev-token
- Layer 3, head 2 — 87% prev-token
- Layer 3, head 7 — 58% prev-token
- Layer 4, head 11 — 100% prev-token

These are consistent with the literature. Olsson et al. 2022 identified
previous-token heads as the prerequisite for induction heads, appearing
in early-to-mid layers.

**Self-attn heads in layer 0:**
Heads 0.1, 0.3, 0.4, 0.5 classify as self-attn (diagonal score > 0.5).
These are early-layer local feature detection heads, consistent with
Elhage et al. 2021.

**Late-layer diffuse heads:**
Layer 11 heads 0 and 8 are consistently diffuse across all prompts.
These may be the function-vector style heads identified in the literature
as dominant at larger scales. At GPT-2 scale they appear in small numbers.

**The attention sink problem:**
76% of heads classify as first-token. This is almost certainly the
attention sink phenomenon — heads that have nothing specific to attend
to default to position 0. The literature does not report 76% first-token.
We are not distinguishing genuine first-token semantic heads from
attention sink behavior.

## What Is Missing

**Induction heads not found.** Olsson et al. found induction heads in
layers 1-2 of GPT-2 small (specifically heads 1.4 and 1.10 in their
notation). We find no induction heads. Reason: induction heads require
repeated token patterns to fire — [A][B]...[A] -> [B]. Our prompts
contain no such patterns.

**Attention sink vs semantic first-token not distinguished.** A head
that genuinely uses position 0 as a global context store looks identical
to a head that defaults to position 0 when it has nothing to do.
We need a way to distinguish these.

## Next Steps

1. Add `repeated.txt` prompt file with induction-triggering patterns
   to surface induction heads and compare to Olsson et al.

2. Add attention sink detection — measure whether a head's first-token
   score drops when position 0 is semantically irrelevant vs relevant.
   A true sink head attends to position 0 regardless of content.
   A semantic first-token head attends there because the content matters.

3. Compare final distribution to literature typology once induction
   heads are surfaced.

## Comparison to Literature

    Type          Our results    Literature (GPT-2 small)
    prev-token    5 heads        ~2-4 heads in layers 0-1  (partial match)
    self-attn     4 heads        documented in layer 0     (match)
    induction     0 heads        ~2-4 heads in layers 1-2  (missing)
    first-token   110 heads      not reported at this rate (mismatch)
    diffuse       23 heads       function vector style      (partial match)

The prev-token and self-attn results are encouraging. The missing
induction heads and the attention sink domination are the two main
gaps to address.