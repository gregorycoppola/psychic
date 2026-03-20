# The Global Workspace Hypothesis

**Date:** 2026-03-20 15:00

## Observation

76% of heads in GPT-2 small classify as first-token across our 38 prompts.
Initial interpretation was "attention sink" — heads defaulting to position 0
when they have nothing specific to attend to.

## Revised Interpretation

Position 0 is not a sink. It is a global workspace.

GPT-2 is a causal language model. Position 0 is the only token position
that every subsequent position can attend to without violating the causal
mask. It is the natural location for a shared communication channel.

The pattern makes sense architecturally:
- Some heads write local information to position 0's residual stream
- Other heads read from position 0 to gather global context
- Position 0 accumulates a running summary of the sequence

This is not pathological. It is an efficient use of the residual stream
as a shared scratchpad — exactly the "global workspace" framing from
cognitive science and from Elhage et al. 2021.

## Connection to BP Framework

In BP terms, position 0 functions as a global factor node:

- **Writers** send messages to the global node (local -> global)
- **Readers** gather from the global node (global -> local)
- The global node aggregates evidence across the full sequence

This is a form of loopy BP where position 0 is a hub node connected
to all other positions. The loops run through position 0.

This is consistent with the BP framework — it just means the implicit
factor graph of GPT-2 has a hub structure centered on position 0,
not a tree or chain structure.

## Implication for Head Typology

"First-token" is probably not a single head type. It is at least two:

- **Readers** — gather global context from position 0
- **Writers** — send local information to position 0

Distinguishing these requires looking at the value (V) and output
projection (OV) matrices, not just the attention pattern (QK).
A reader head attends to position 0 and copies its content forward.
A writer head attends to position 0 and writes new information there
via the residual stream addition.

This is the OV circuit analysis from Elhage et al. 2021 — a natural
next step once we have the attention pattern picture settled.

## What This Does Not Change

The specialized heads we found are still the most interesting:
- Prev-token heads (layers 2-4) — clear BP gather step
- Self-attn heads (layer 0) — local feature detection
- Diffuse heads (layers 0, 11) — open question, possible function vector

These are the heads doing something other than global workspace
communication, and they are the ones most directly relevant to the
BP typology question.

## Next Steps

1. Add repeated prompts to surface induction heads
2. Look at OV circuits to distinguish reader vs writer first-token heads
3. Map the full head typology against BP predictions