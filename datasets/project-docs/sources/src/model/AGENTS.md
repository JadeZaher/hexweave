# Numerical fixture contract

This directory is a newly written Candle 0.11 CPU reference. `HexModel::forward`
is the entry point. `tile.rs` owns causal attention and FFN transforms,
`controller.rs` owns pulse recurrence and hard route scores, and `ops.rs`
contains local tensor primitives and deterministic fixture randomness.

## Exact implemented semantics

The tiny model ties input embeddings to vocabulary readout and adds fixed
sinusoidal positions. Each tile has full multihead causal attention and a gated
FFN. Attention reads the complete immutable incoming sequence. Its mask blocks
future positions. Residual attention/FFN branches have fixed 0.2 scales.

There is one shared route for the whole sequence, starting at the center tile.
The controller reads ONLY the first token's hidden state plus its recurrent
eight-value pulse. Conditioning a shared sequence route on pooled or final-token
state would leak future inputs into earlier predictions. Legal destinations are
hex neighbors and halt, with learned signed axial displacement preferences.
Revisits are allowed, depth and total visits are bounded, and at least one tile
runs. Budget-forced termination contributes no sampled halt score.

The pulse is updated before each tile and conditions its three-knot continuous
activation. Knots use bounded tanh values with linear interpolation between
-2, 0, and 2 and linear endpoint extrapolation. Knots and pulse receive ordinary
pathwise derivatives. The hard route uses the score-function term
`detached(task_loss - moving_baseline) * sum(log_probability)` in the
minimized objective. See Schulman et al., Stochastic Computation Graphs:
https://proceedings.neurips.cc/paper/2015/hash/de03beffeed9da5f3639a621bcab5dd4-Abstract.html.
The scalar baseline starts at log(vocabulary size) and subsequently uses only
previous training losses. Held-out evaluation never feeds the update. The score
term uses coefficient one, matching the ordinary score-function estimator for
the current example with a fixed incoming history; clipped SGD and detached
cross-example history do not claim a full history-dependent meta-gradient.

The terminal pulse is detached and carried into the next training example as
one bounded history state. Evaluation resets history to zero. This is NOT a
corrected-demonstration FIFO, predicted local targets, or causal error messages.
Weights remain fixed for each forward/backward computation; clipped SGD stages
finite updates before writing them. A failed fixture discards its model. This
is not an atomic persistent checkpoint protocol.

## Differences from the source experiments

This is F-inspired, not a numerical F port: F routes per token, samples detached
continuous activation knots and learns their policy scores, includes post-actions
and private-memory/calibration/counterfactual objectives, and uses fake ternary
operators. This fixture directly differentiates its knots, uses sequence routing,
and has none of those additional objectives. It uses FP32 and keeps every tile,
parameter, gradient, and activation resident. It does not demonstrate bounded
large-model residency. Fanout exists only in the independent planner.

The synthetic token-ID task has no text tokenizer and no natural-language
quality claim. Reported initial/final greedy losses use the same disjoint fixture
sequence and reset pulse; a decrease is not guaranteed by sampled policy updates.
Fixtures test prefix causality, gradient connectivity, parameter changes, and
allocation bounds. They do not benchmark a trained model against Granite.

API reference: https://docs.rs/candle-core/0.11.0/candle_core/struct.Var.html.
Keep source modules below 300 lines and functions below 90 lines. Put rationale
here, with short source comments. Batch fixes before the integrated test sweep.
