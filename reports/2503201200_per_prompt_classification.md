# Per-Prompt Classification Design

**Date:** 2025-03-20 12:00

## Problem

Current `patterns` command averages attention scores across all prompts and
all token positions before classifying each head. This collapses the signal —
the attention sink (first-token) dominates the average and we get 82%
first-token heads, which does not match the literature.

## Root Cause

Averaging across positions and prompts before classification loses the
per-prompt behavioral signal. A head that strongly attends to the previous
token on 60% of prompts but defaults to the attention sink on 40% looks
like a weak first-token head in the average.

## Proposed Fix

Classify each head *per prompt* first, then aggregate the classifications
into a distribution.

### Per-prompt classification

For each (head, prompt) pair:
1. Compute scores as now (entropy, max_attn, prev_tok, first_tok, diagonal)
2. Classify into one type based on those scores
3. Store the classification, not just the scores

### Aggregation

For each head, count how many prompts classified as each type.
Report as percentages. The dominant type is the plurality.

## New Table Format

    Layer  Head  prev-tok  first-tok  self-attn  sharp  diffuse  dominant
    2      2     65%       20%        5%         5%     5%       prev-token
    0      1     2%        5%         90%        3%     0%       self-attn
    4      11    98%       1%         0%         1%     0%       prev-token

## Expected Improvement

Per-prompt classification should reveal the head typology that the literature
reports — previous token heads in layer 0, induction-adjacent heads in
layers 1-2, and a more diverse distribution overall.

## Implementation

New command `psychic classify`. Core logic in `psychic/core/classify.py`.
Prompt files unchanged. Analysis functions unchanged.