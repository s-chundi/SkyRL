# Plan: Addressing SkyRL Issue #1423

**Issue:** `tp2_cp2_policy_seq_packing_no_entropy_loss` CI test fails with `assert 0.3345... < 0.25` on `policy_loss` — a numerical parity mismatch between the Megatron and FSDP backends.

**Hardware:** 8×H100 is plenty. The test pins `gpus_per_node=4` (TP=2 × CP=2 × PP=1 × DP=1). CI runs it on 4×L4. You'll have massive headroom — use the extra GPUs to run parallel configs while debugging.

---

## Phase 0 — Setup (half a day)

```bash
cd /Users/suhas.chundi/Documents/SkyRL
uv sync --extra megatron --extra dev
uv run examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
export CI=true _SKYRL_USE_NEW_INFERENCE=1
```

Reproduce the failure:

```bash
uv run --isolated --extra dev --extra megatron pytest -s \
  tests/backends/skyrl_train/gpu/gpu_ci/test_megatron_worker.py::test_megatron_train \
  -k "tp2_cp2_policy_seq_packing_no_entropy_loss"
```

Expected: `assert 0.3345... < 0.25` on `policy_loss`. If it passes, something is already different in your env — track that down before proceeding.

---

## Phase 1 — Targeted doc reading (1 day max)

Four concepts. Skim; you'll re-read after the code forces questions.

From https://github.com/NVIDIA/Megatron-LM/tree/main/docs:

1. **`parallelisms.md`** — just TP and CP sections. Key distinction: TP shards *weights* (ranks see the same tokens), CP shards the *sequence dimension* (ranks see different tokens, all-gather for attention).
2. **`source/api-guide/tensor_parallel.rst`** — glance at `ColumnParallelLinear` / `RowParallelLinear`. No need for the math.
3. **The verl PR from issue #794: https://github.com/volcengine/verl/pull/4510** — read carefully. It fixes this exact class of bug (loss normalization under CP/DP). Highest-leverage read.

Skip PP, EP, MoE, distributed optimizer until a specific question forces you in.

---

## Phase 2 — Code tracing (2–3 days)

Hypothesis to falsify: *under CP=2, each rank holds half the sequence, so any "mean over tokens" must be a weighted mean using global token count, not a local mean then all-reduce-mean.*

**Trace path:**

1. `tests/backends/skyrl_train/gpu/gpu_ci/test_megatron_worker.py` (~line 540) — the FSDP re-run. See exactly what `policy_loss` it compares.
2. `skyrl-train/.../workers/megatron/megatron_worker.py` — find where `policy_loss` is returned, trace upstream.
3. `skyrl-train/.../workers/megatron/megatron_model_wrapper.py` — forward under TP/CP. How are logits and losses reduced across CP ranks?
4. `skyrl-train/.../distributed/megatron/megatron_utils.py` — sequence packing. Packing + CP interact: packing multiple short sequences and then CP-splitting the long one gives *uneven* tokens per rank.
5. `workers/fsdp/fsdp_worker.py` — the reference computation. The test is parity-with-FSDP.

**Instrument with prints, not a debugger.** Per-rank log: local token count, local loss sum, mask sum, the denominator in the mean. Do it in both backends. Diff the numbers.

**Narrow with the sibling parametrizations** (same test file):

- `tp2_policy_seq_packing` (CP=1) — pass? (likely yes → CP-specific)
- `cp2_policy_seq_packing` (TP=1) — pass? (likely no → confirms CP)
- `tp2_cp2_policy` (no packing) — pass? (tells you if packing interacts with CP)

Put the results in a small table. That table becomes your PR description.

---

## Phase 3 — Fix and validate (3–5 days)

The fix is likely small (tens of lines). Most of the work is confidence you haven't broken other parametrizations. Run the full megatron test suite before opening a PR. If the fix overlaps issue #794, mention both.

---

## Other considerations

- **Comment on the issue first.** Before sinking a week in, post on #1423 that you're taking it. Tag `@SumanthRH` (reporter) and `@erictang000` (Megatron maintainer). Avoids duplicate work, gets maintainer eyes early.
- **Ask early, ask cheap.** Once you have a concrete hypothesis (not before), post it on the issue or in a draft PR. Maintainers can redirect you in hours.
- **Nondeterminism is real.** FSDP/Megatron parity can flake from RNG/seed handling. If repro passes once, run 3×. If it fails by a *different* number each time, the tolerance itself may be the issue.
- **megatron-bridge version.** HF→Megatron conversion goes through `megatron-bridge`. Check `git log` on `uv.lock` / `pyproject.toml` — a recent bump could masquerade as a loss bug.
- **Don't over-read verl.** Reference, not template. Their parallelism layout differs. Use PR #4510 for the *class* of bug, not to copy code.
- **Keep a scratch log.** In `scratch/issue_1423/`, save per-rank prints, the pass/fail table, and your evolving hypothesis. It becomes the PR description.

---

## Key files

- `tests/backends/skyrl_train/gpu/gpu_ci/test_megatron_worker.py` (models at 36–40; params at 438–456; assertion at end of `test_megatron_train`)
- `tests/backends/skyrl_train/gpu/utils.py` (`get_test_actor_config`, `get_test_training_batch`, `init_worker_with_type`)
- `ci/gpu_ci_run_skyrl_train_megatron.sh`
- `.github/workflows/gpu_skyrl_train_megatron.yaml`
