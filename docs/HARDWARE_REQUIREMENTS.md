# Hardware Requirements

## Local Mac (Current)

Sufficient for GPT-2 small and medium. Getting tight for larger models.

    Disk available:  ~27GB
    Models done:     gpt2 (500MB), gpt2-medium (1.5GB)
    Models pending:  gpt2-xl (6GB), deepseek (3.5GB), qwen (1GB + 3.5GB)
    Total pending:   ~14GB weights + ~150MB patterns
    Verdict:         fits but tight, no room for mistakes

## Server Requirements (Vultr or equivalent)

For running the full multi-model study comfortably.

### Minimum Viable

    CPU:   2 vCPUs
    RAM:   8GB
    Disk:  80GB SSD
    Cost:  ~$24/month, ~$0.036/hour

Handles all models up to 1.5B parameters. Forward pass in numpy
is CPU-only — no GPU needed. Peak RAM usage when loading a 1.5B
model is ~6-8GB including numpy overhead.

### Recommended

    CPU:   4 vCPUs
    RAM:   16GB
    Disk:  160GB SSD
    Cost:  ~$48/month, ~$0.072/hour

Comfortable headroom. Can load GPT-2 XL (6GB weights) and have
room for OS, code, and intermediate data. Forward pass on 4 vCPUs
is roughly 2-3x faster than Mac due to better numpy BLAS.

### If Adding Larger Models Later

    CPU:   8 vCPUs
    RAM:   32GB
    Disk:  320GB SSD
    Cost:  ~$96/month, ~$0.144/hour

Needed if adding 7B+ models (Mistral-7B = ~14GB weights,
needs ~16GB RAM minimum). Not needed for current plan.

## Disk Budget (Current Plan)

    Model                           Weights    Patterns
    gpt2                            500MB      16MB
    gpt2-medium                     1.5GB      43MB
    gpt2-xl                         6GB        80MB
    deepseek-r1-qwen-1.5b           3.5GB      30MB
    qwen2.5-0.5b                    1GB        10MB
    qwen2.5-1.5b                    3.5GB      30MB
    ─────────────────────────────────────────────────
    Total weights:                  16GB
    Total patterns:                 209MB
    OS + code + misc:               10GB
    ─────────────────────────────────────────────────
    Total:                          ~26GB

    Recommended disk:               80GB (3x headroom)

## Recommended Workflow

Spin up → run → copy results → terminate. Not a persistent server.

1. Spin up Vultr Cloud Compute (4 vCPU, 16GB RAM, 160GB SSD)
2. SSH in, git clone psychic, uv sync
3. Download all model weights (~16GB, takes ~30 min at 10MB/s)
4. Run all collects (~30 min total for all models)
5. rsync pattern files back to Mac (~250MB, fast)
6. Terminate instance

Total cloud cost per full study run: ~$0.15-0.25

## Runtime Estimates (4 vCPU server)

    Model               Params   Time/prompt   88 prompts
    gpt2                124M     ~0.1s         ~10s
    gpt2-medium         345M     ~0.3s         ~25s
    gpt2-xl             1558M    ~1.5s         ~2min
    deepseek-r1-1.5b    1500M    ~1.5s         ~2min
    qwen2.5-0.5b        500M     ~0.2s         ~18s
    qwen2.5-1.5b        1500M    ~1.5s         ~2min
    ──────────────────────────────────────────────────
    Total:                                     ~8min

## Setup Script

See docs/SERVER_SETUP.sh for one-command server setup.