# psychic

Empirical inspection of transformer weights through the lens of the
Transformers are Bayesian Networks framework (Coppola 2026, shannon repo).

The central question: can we look at real model weights and identify
which attention heads correspond to which types in the BP typology?

## The Plan

1. Download a pretrained model (GPT-2 small to start)
2. Iterate through the weights and understand their structure
3. For each attention head, extract the Q and K matrices
4. Classify each head into a type from the typology:
   - Sharp routing (previous token, induction, name mover, retrieval)
   - Diffuse routing (function vector — the open problem)
   - Positional, inhibition, redundant
5. Build a visitor that walks every head and assigns it a type
6. See what the distribution looks like empirically

## Connection to the Broader Program

The `interp` repo reviewed the mechanistic interpretability literature
and concluded that sharp routing heads are fully covered by the boolean
BP account, while diffuse (function vector) heads are not. This repo
goes empirical — looking at actual weights rather than reading papers.

The hypothesis: Q and K matrices of sharp routing heads should show
projectDim-like structure (rank-1, single-dimension peak). Diffuse
heads should look different. We want to see if we can tell them apart
just from the weights.

## Related Repos

- `interp` — literature meta-review, attention head typology
- `shannon` — the transformers are Bayesian networks paper
- `bayes-learner` — gradient descent finds BP weights from scratch

## Setup

    pip install transformers torch