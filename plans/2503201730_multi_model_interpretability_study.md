# Plan: Multi-Model Interpretability Study

**Date:** 2026-03-20 17:30
**Status:** Active
**Repo:** psychic

## The Core Insight

Open model weights are free. Forward passes are cheap. We can run a full
head typology study on any open model in minutes on a Mac. This gives us
an empirical interpretability lab with near-zero marginal cost per model.

## What We Have Now

- GPT-2 small fully analyzed: 88 prompts, stable classification, hybrid
  BP map, three-phase layer structure documented
- Infrastructure: download, collect, classify, analyze commands
- Prompt library: 88 prompts across 9 categories with index system
- Reports: 5 timestamped reports documenting methodology and findings

## The Main Finding So Far

GPT-2 small has a three-phase layer structure:

    Phase 1 (layers 0-2):  continuous-heavy — feature extraction
    Phase 2 (layer 3):     mixed — transition zone
    Phase 3 (layers 4-11): hub-dominated — global workspace

Boolean-AND heads (sharp BP routing) disappear after layer 4.
All 9 boolean-AND heads are in the first half of the network.

## Plan Elements

### Element 1: Scale Study (Pythia Family)
Run the same analysis on the Pythia model family:
70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B — all same architecture,
same training data, different scale. Ask: does the three-phase structure
replicate? Does the boolean-AND cutoff layer shift with depth? Does the
ratio of boolean-AND to hub heads change with scale?

This requires: multi-model download support, generalized forward pass
for different architectures (Pythia uses GPT-NeoX architecture, slightly
different from GPT-2).

Effort: medium. New architecture support needed.
Payoff: high. First cross-scale comparison of hybrid BP structure.

### Element 2: Deepen GPT-2 Analysis
Add OV circuit analysis — look at the value (V) and output projection
matrices of each head to distinguish hub readers from hub writers.
This would split the 72% first-token category into two meaningful types.

Also: add more prompts (target 500+), run reasoning vs language index
comparison, look at per-category variation.

Effort: low-medium. No new architecture needed.
Payoff: medium. Better characterization of GPT-2.

### Element 3: Replicate Known Results
Add repeated-token prompts to surface induction heads. Try to match
Olsson et al.'s specific heads (1.4 and 1.10). Validate methodology
against known ground truth.

Effort: low. Just new prompt files.
Payoff: medium. Validates methodology, grounds our results in literature.

### Element 4: Cross-Architecture Comparison
Download GPT-Neo, Mistral-7B, or a small Llama variant. Compare
head typology across architectures with similar parameter counts.
Ask: is the three-phase structure architecture-specific or universal?

Effort: medium-high. New tokenizers and architectures.
Payoff: high if universal, interesting either way.

### Element 5: Connect to Shannon Paper
Write a formal connection between the psychic empirical results and
the shannon paper's theoretical claims. Specifically: do the 9
boolean-AND heads we found show projectDim/crossProject weight
structure as predicted by the BP construction? This would be the
first empirical confirmation of the specific weight structure
predicted by the formal proof.

Effort: medium. Requires OV weight analysis.
Payoff: very high. Direct theory-experiment connection.

### Element 6: Paper Draft
Once we have scale study results (Element 1) and OV analysis
(Element 2), draft a short paper: "Hybrid BP Structure of Open
Language Models." Main claims: three-phase structure, boolean-AND
cutoff, scale behavior. Connect to shannon paper as theoretical
foundation.

Effort: high.
Payoff: very high.

## Suggested Order

1. Element 3 (validate) — quick win, grounds methodology
2. Element 2 (deepen GPT-2) — OV circuit analysis, more prompts
3. Element 1 (scale study) — Pythia family, main new result
4. Element 5 (connect to shannon) — theory-experiment link
5. Element 6 (paper) — when 1-4 are done
6. Element 4 (cross-architecture) — parallel or after paper

## Open Questions

- Does the three-phase structure replicate on Pythia?
- Does the boolean-AND cutoff layer scale linearly with depth?
- Can we distinguish hub readers from hub writers via OV circuits?
- Do the boolean-AND heads show projectDim weight structure?
- Is 88 prompts enough or do we need 500+?

## Notes

Prior work confirms: early layers do feature extraction, late layers
do contextual integration. Our contribution is the hybrid BP framing —
boolean-AND vs continuous-OR vs hub — and the specific claim about
the boolean-AND cutoff at layer 4. This framing appears novel.

The Pythia family is the highest-leverage next experiment because it
lets us make a scaling claim, which is what gets attention in the
current ML landscape.