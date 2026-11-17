# Connected v5.4: anchor-route credit modes and input/tile-seeded dynamic FFN slicing (plan)

Status: owner-approved 2026-11-11 after Q&A decision gate.

## Motivation and evidence

1. Five v5/v5.1/v5.2 pilots and the v5.3 full-window pilot (`connected-fp32-v53-window-pilot-s7-20261111`,
   15,750 anchors, held-out 4.332 at 8,192 and 4.180 at 15,750) showed that label-conditioned synthetic
   gradients solve the local credit alignment problem (prefit cosine 0.89–0.97, trust 0.69–0.87), yet
   continued lane-based tile updates still trail frozen tiles after 5,000–7,000 anchors.
2. Telemetry from `connected-fp32-v53-window-pilot-s7-20261111` shows severe exposure skew: center tile 3
   received 20,051 anchor visits and 160,285 local visits (hitting the 1.500 master RMS cap with 25.5%
   knot overflow), while outer tiles 5 and 6 received only 1,414 and 1,363 anchor visits.
3. To isolate the learning bottleneck and test input/tile-seeded dynamic perspective shifting without
   changing on-disk checkpoint shapes or resident bundle caps, v5.4 adds two orthogonal, backwards-compatible
   controls on `connected`: `credit_mode` and `slice_mode`.

## Design

### 1. Credit and pathing modes (`connected.credit_mode`)

`CreditMode` (`lanes` default, omitted from serialized identity when default):
- `lanes`: existing v5 behavior (anchor reverse pass trains `E`, readout bias, predictor and routes; eight
  unscored local lanes train tile transformations from predicted credit).
- `anchor_exact`: exact reverse-pass tile gradients collected during `connected_anchor::gather` along the
  realized anchor route are accumulated once per visited tile per anchor and applied at
  `local_rate * local_decay.factor(anchor)` with `gain = 1.0`. No unscored local lanes run in training or
  evaluation replay.
- `anchor_predicted`: predicted credit is evaluated and differentiated only on the realized anchor route's
  frames (no unscored local lanes), scaled by `Lambert * trust` at `local_rate * local_decay.factor(anchor)`.
- `hybrid`: combines exact anchor-route gradients from `gather` with local lane gradients in `Accumulator`,
  applying at unit base trust so anchored supervision and local exploration co-adapt.

### 2. Dynamic active-slice sizing and `(input, tile_id)` entry offsets (`connected.slice_mode`)

`SliceMode` (`full` default, omitted from serialized identity when default):
- `full`: all `m = geometry.ffn` channels active (`mask = 1.0`, `down_scale = 1 / sqrt(m)`), bit-for-bit
  identical to v5.3.
- `windowed`: deterministic seed computed in `Tile::cache` from `tile.id`, the last 4 visible prefix rows in
  `memory`, coarse sign-bits of `input`, and `history`. Selects:
  - Entry offset `offset = (seed as usize) % m` into FFN matrices (`G`, `A`, `CURVE`, `DOWN`).
  - Dynamic active slice width `m_active` from `{m / 2, 3 * m / 4, 7 * m / 8}` (`{192, 288, 336}` at
    `m = 384`), shifting effective tile capacity and routing perspective per `(input, tile_id, history)`.
  - Channels in `[offset, offset + m_active) mod m` have `mask = 1.0` (`0.0` outside) and
    `down_scale = 1 / sqrt(m_active)`.
- `hybrid`: active window `[offset, offset + m_active) mod m` has `mask = 1.0`, while outside channels keep
  a cyclic `tile_id`-shifted background weight in `{0.25, 0.5}` with `down_scale = 1 / sqrt(sum(mask^2))`,
  preserving full-bank gradient flow alongside windowed specialization.

Because the seed is a pure function of `(tile.id, input, memory, history)`, `Tile::forward` and
`Tile::backward` reconstruct the exact same mask, and training, evaluation and `hexweave complete` share
the exact same entry distribution without changing on-disk matrix shapes or resident memory bounds.

## Pilot execution sequence (one run at a time, 30 minutes each, seed 7, FP32)

1. **Pilot 1a (`connected-fp32-v54-anchor-exact-s7-20261111`):**
   `credit_mode: "anchor_exact"`, `slice_mode: "full"` (Experiment 1 without offset entry).
2. **Pilot 1b (`connected-fp32-v54-anchor-exact-window-s7-20261111`):**
   `credit_mode: "anchor_exact"`, `slice_mode: "windowed"` (Experiment 1 with `(input, tile_id)` offset
   entry and dynamic active-slice sizing).
3. **Pilot 2 (`connected-fp32-v54-anchor-pred-window-s7-20261111`):**
   `credit_mode: "anchor_predicted"`, `slice_mode` set to the stronger mode from 1a/1b (Experiment 2:
   predicted credit on the anchor route only).
4. **Pilot 3 (`connected-fp32-v54-hybrid-s7-20261111`):**
   `credit_mode: "hybrid"`, `slice_mode: "hybrid"` (combining anchored supervision with local lanes and
   hybrid projection pathing).
