# Design

## Structure

    src/psychic/
      core/
        loader.py      -- load_safetensors, shared by all commands
        forward.py     -- GPT-2 forward pass, returns logits + attention patterns
        analysis.py    -- analysis functions over attention patterns
        tokenizer.py   -- minimal BPE tokenizer
      prompts/
        general.txt    -- one prompt per line, general text
        factual.txt    -- factual statements
        narrative.txt  -- narrative/story text
        logic.txt      -- logical reasoning sentences
      cli/
        commands/
          download.py       -- download model weights
          download_vocab.py -- download tokenizer files
          forward.py        -- run one forward pass on text
          heads.py          -- weight-space analysis (rank, singular values)
          patterns.py       -- attention pattern analysis across prompts
          clear.py          -- clear cached files

## Adding Prompts

Add a line to any file in `src/psychic/prompts/` or create a new `.txt` file.
All `.txt` files in that directory are loaded automatically by the `patterns` command.

## Adding Analyses

Add a function to `src/psychic/core/analysis.py` with signature:

    def my_analysis(pattern: np.ndarray) -> float:
        ...

Then register it in the ANALYSES dict at the bottom of that file.
The `patterns` command picks up all registered analyses automatically.

## Batching

Not yet implemented. The controller loop in `patterns.py` will print
a summary every N prompts when added. Core code will not change.

## The Research Question

For each attention head in GPT-2, is it sharp (BP-style routing,
concentrated attention on one or few positions) or diffuse (function
vector style, spread across many positions)?

The hypothesis from the `interp` repo: sharp heads are fully covered
by the boolean BP account. Diffuse heads are not. If we can classify
every head in GPT-2, we get an empirical picture of how much of the
model is operating in BP mode vs something else.