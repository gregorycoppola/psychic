<!-- 📄 psychic/reports/2503202300_and_hub_theoretical_v1.md -->
# Hub as Positional-AND: A Theoretical Note

**Date:** 2026-03-20 23:00
**Context:** Discussion following Qwen 0.5B analysis and cross-family comparison

## The Observation

The four-type empirical taxonomy (boolean-AND, continuous-OR, hub, mixed) may
collapse to two theoretical types at the BP level:

    sharp  = AND  (discrete lookup, one neighbor)
    diffuse = OR  (continuous aggregation, many neighbors)

Hub heads are sharp — they have peaked attention distributions just like
boolean-AND heads. The difference is what they attend to:

    boolean-AND  →  content-matched token  (semantic address)
    hub          →  position 0             (positional address)

## The Argument

If AND means "sharp discrete lookup" then hub IS AND, just with a trivial
address function. Both are performing the same formal operation:

    copy one token's representation into the residual stream

The boolean-AND heads do this with a content-determined address.
The hub heads do this with a fixed positional address (always token 0).

This gives a cleaner theoretical vocabulary:

    semantic-AND   =  what the paper calls boolean-AND
    positional-AND =  what the classifier calls hub
    OR             =  continuous-OR
    mixed          =  mixed (no change)

## Why Hub Got Its Own Category Empirically

Position-0 attention is trivially detectable — the classifier can identify
it without any semantic analysis. It is also overwhelmingly dominant:
60-79% of all heads across every model studied. Separating it out made
the empirical taxonomy cleaner and more useful for analysis.

But empirical convenience is not the same as theoretical distinction.
The formal BP construction does not have a separate concept for
"positional AND" vs "semantic AND" — both are sharp gather operations
that copy one value into one dimension of the residual stream.

## Implications

**For the paper:** The four-type taxonomy is the right empirical description.
The theoretical claim should be the two-type reduction:

    All sharp heads (hub + boolean-AND) implement the AND primitive.
    All diffuse heads (continuous-OR) implement the OR primitive.
    Mixed heads are context-dependent combinations of both.

Hub dominance then becomes a specific empirical finding about how
transformers implement AND in practice — they prefer the simplest
possible address function (position 0) for the majority of their
AND capacity, reserving content-based addressing for a small fixed
set of heads.

**For the interpretability section:** The four types remain the right
display taxonomy. But the page should note that hub and boolean-AND
are both AND at the theoretical level, distinguished only by whether
the address is positional or semantic.

**For the scaling finding:** The boolean-AND = 9 result now has a
sharper interpretation. It is not "AND heads do not scale" — it is
"semantic-AND heads do not scale." The positional-AND (hub) heads
scale aggressively. What the model acquires as it grows is more
positional routing capacity, not more semantic routing capacity.

## Refined Scaling Statement

    Type              GPT-2 small   GPT-2 medium   interpretation
    semantic-AND            9              9        fixed logical primitive
    positional-AND        103            305        scales with model size
    OR                     28             46        grows modestly
    mixed                   8             24        grows modestly

The model does not get more logical reasoning capacity as it scales.
It gets more positional routing capacity — more global workspace
infrastructure for assembling and communicating context.

## Open Questions

1. Do the 9 semantic-AND heads in GPT-2 correspond to specific
   factor graph edges in the implicit QBBN? If yes, they are
   directly interpretable as logical relations.

2. Is position 0 special by construction (BOS token) or does the
   model learn to use it as a hub? The BOS token is always present
   and never predicted — a natural candidate for a global workspace.

3. Does the semantic-AND / positional-AND distinction hold under
   OV circuit analysis? Both should show projectDim/crossProject
   weight structure if the BP construction is correct.
