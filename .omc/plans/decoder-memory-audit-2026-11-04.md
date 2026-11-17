# Decoder and tiled-memory audit

Owner request: check decoder bugs, explain parameter-count versus tiled memory,
and explain how the learned architecture affects decoding. Keep the active
training run under its existing pinned executable and supervisor.

Independent source audit identified explicit story text being passed to a corpus
prefix helper that injects another START byte for short prefixes, plus synthetic END observation
during warmup of a one-byte prompt. Correct story prefix construction and prompt
warmup, reuse the established finite lowest-index greedy selector, and test
boundary behavior. These findings do not establish the cause of space loops.

Add bounded, opt-in completion traces for at most sixteen steps, with actual
logits, chosen byte, probability/margin, route and visible-prefix evidence, plus
a scalar tied-readout cross-check. A clearly labeled diagnostic prompt-only
feedback mode freezes all post-prompt memory evolution without changing the
default model training policy. Independent review precedes one coordinated test sweep.

Runtime checks use a hash-verified copy of preserved checkpoint a000000186164 in
work/core-runs/connected-decoder-audit-20261104. Preserve the original completion
binary (SHA-256 1e5f5119668ed9134ee07dc5b2c99717f466583ab20d42563e72c6f062e3ec0a)
and compare bounded before/after completions and feedback variants under the
existing 512 MiB child Job cap. Hash input checkpoint files before and after.
Do not stop or change the running study; its generation measurements retain the
earlier pinned policy, and its training arithmetic remains unchanged.

Memory audit distinguishes total tile-bank disk storage from resident tile/Adam
RAM. At fixed tile size, resident cap, context and global-state sizes, heavy tile
RAM is bounded as more tiles are added; per-tile metadata still grows. Widening
tiles, increasing workers/context, or enlarging shared embeddings/predictor/memory
does increase RAM. Current implementation fixes seven tiles; unlimited tile-bank
growth has not been implemented or benchmarked.

Completed implementation and independent source review, followed by one integrated
sweep at work/validation/ea495053cfda433b9fb0b701dc6e6c24: 130 Rust tests passed
(three existing ignored), 25 Python passed, format/Clippy/structure, three planning
profiles and smoke passed. Offline release build succeeded.

The bounded runtime comparison completed 18 requests (six prompts, three modes,
16 new bytes each). All 192 corrected/ablation trace rows passed independent
Python prefix, greedy-selection, probability and emission checks. Scalar versus
packed readout maximum absolute error was zero. Five prompts repeated spaces;
the one-byte X prompt changed from R/dash to repeated J with the fixes. Default
and prompt-only outputs matched in all six cases, so feedback deltas compare
identical generated prefixes. Input hashes stayed identical, resident bundles
never exceeded four and each scratch directory was removed. These results do
not establish useful generation or correctness of every upstream computation.
Evidence: work/inference/decoder-audit-20261104/{comparison,analysis}.json.
Current training remained on its original pinned executable throughout.
