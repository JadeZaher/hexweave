# Causal-prefix-v1 implementation contract

2026-10-24 (America/Denver). Authored implementation contract for new
`causal-prefix-v1` under **src/causal/**. At authoring, independent review,
integrated checks and training admission were pending. Selected direction:
byte codec, immutable prefix/current target,
attention per visited tile, ternary weights, two correction phases and experimental
route-to-U detachment. The equations below settle implementation choices; they
do not claim learning validation. Preserve text v4 unchanged. The 81M/287M
targets remain separate. Directory-level `src/causal/AGENTS.md` documents the
implemented state interfaces, lifetimes and corrections to the initial proposal.

## Fixed choices

| Field | Numerical fixture | Unadmitted candidate |
|---|---:|---:|
| V / d / m / heads | 256 / 16 / 48 / 2 | 256 / 128 / 384 / 4 |
| Context cap | 8 bytes | 1,024 bytes |
| Mesh / visits | radius 1, seven tiles / 4 | same |
| Anchor interval / history cap | 4 opportunities / 8 records | same |

Only these profiles initially validate. Keep the existing 1,048,576-coefficient
matrix bound. All learned values are ternary; calculations are f32, checked
finite, with fixed accumulation order and pinned resolved kernel. Reference
derivatives can use f64. No floating master weights, learned norms, dropout or
hidden task-gradient clipping.

Use persistent i16 sign evidence, threshold 1, existing adjacent-trit update/reset
rule. This optimizer remains experimental. A tile/E/predictor version increments
once when its weights or evidence change; predictor has a separate version.
Initialize E with seed S, predictor S+1; tile t matrices with wrapping
S+16+16*t plus offsets Q,K,V,O,G,A,D,curve,R,U,P = 0..10 respectively.

## Prefix and normalization

E is d×256, target state x is d, memory M is L×d. Visible bytes exclude the target;
1<=L<=context. No BOS/padding. Positions start at zero within an example:

```text
omega_j = 10000^(-2*j/d)
pos[i,2j] = sin(i*omega_j); pos[i,2j+1] = cos(i*omega_j)
a_i = E[:,byte_i] + pos[i,:]
M_i = rms(a_i); x0 = M[L-1]

rms(v): inv=1/sqrt(mean(v*v)+1e-5); z=v*inv
rms_back(z,inv,b): inv*(b-z*mean(b*z))
```

Compute sinusoidal values in f64, then cast completed values to f32; compute
positions on demand without a retained context×width table. Cache M and its L
inverse RMS scalars. M stays immutable through the entire opportunity.

## Tile forward

Tile matrices: Q/K/V/O d×d; G/A m×d; D d×m; curve C 3×m;
router R r×2d; pulse U d×2d; pulse P (2d+r)×d.
Center r=7, each outer tile r=4. h is detached compatible correction history.

```text
n = rms(x)
q = Qn/sqrt(d); k_i=KM_i/sqrt(d); v_i=VM_i/sqrt(d)
For each contiguous head of width k=d/heads:
  score_i=dot(q_head,k_i_head)/sqrt(k)
  alpha_i=softmax_i(score)
  c_head=sum_i(alpha_i*v_i_head)
c=concat(c_head)
u=x+0.25*Oc/sqrt(d)
rms_u=rms(u)
gate=sigmoid(G*rms_u/sqrt(d)); z=A*rms_u/sqrt(d)
curved=curve(z,C); w=gate*curved; ffn=Dw/sqrt(m)
f=concat(rms_u,h)
latent=tanh(Uf/sqrt(2d)); p=P*latent/sqrt(d)
gain=1+0.125*tanh(p[0:d]); shift=0.125*tanh(p[d:2d])
y=u+0.25*(gain*ffn+shift)
```

Softmax subtracts its maximum. Sigmoid uses 1/(1+exp(-v)) for v>=0,
exp(v)/(1+exp(v)) otherwise. Curve knots have abscissae [-2,0,2] and
ordinates [-1,0,1]+C[:,channel]/4. For z<0, s=0,t=(z+2)/2; otherwise
s=1,t=z/2. Curve=(1-t)*ordinate_s+t*ordinate_(s+1), slope is their
difference/2. Extrapolate on the endpoint segment; choose the right derivative
at zero. Knot derivatives are (1-t)/4 and t/4. Derivative reference cases avoid
the kink.

## Routing and readout

Start at center. Legal actions: ascending neighbor IDs then halt ID 7. Revisits
are legal. Fourth visit forces exit with no route call, draw or policy term.

```text
rho=concat(rms(y),h)
route_logits=R*rho/sqrt(2d)+0.25*p[2d:]
pi=softmax(route_logits)
t=rms(final_y)
logits=E^T*t/sqrt(d)
loss=logsumexp(logits)-logits[target]
delta=softmax(logits)-one_hot(target)
grad_E_readout=outer(t,delta)/sqrt(d)
b_final_y=rms_back(t,inv_y,E*delta/sqrt(d))
```

Training route RNG: existing SplitMix64 and existing 23-bit midpoint f32 uniform
conversion/final-index fallback, initial state
S.wrapping_add(global_opportunity.wrapping_mul(7919)); one draw per actual route.
Inference uses greedy routes, ties by first legal action index.

Baseline starts ln(256); after each completed anchor set baseline=0.9*baseline
+0.1*loss. With A=loss-previous_baseline:

```text
policy=A*(one_hot(selected)-pi)
grad_R+=outer(policy,rho)/sqrt(2d)
grad_P_route_rows+=outer(0.25*policy,latent)/sqrt(d)
```

rho and latent are detached for policy credit. **No route gradient enters U,
rms(y), earlier state, memory, E, attention, FFN, curve or history.** Task credit
still trains U and gain/shift P rows. Local visits have no policy term.

## Exact conditional task backward

Given b_y, initialize b_u=b_y. All operations below use frozen weights:

```text
b_ffn=0.25*gain*b_y
b_p_gain=0.25*ffn*b_y*0.125*(1-tanh(p_gain)^2)
b_p_shift=0.25*b_y*0.125*(1-tanh(p_shift)^2)
b_p_route=0
grad_P_task+=outer(b_p,latent)/sqrt(d)
b_latent_pre=(P^T*b_p)/sqrt(d)*(1-latent^2)
grad_U+=outer(b_latent_pre,f)/sqrt(2d)
b_f=U^T*b_latent_pre/sqrt(2d)

grad_D+=outer(b_ffn,w)/sqrt(m)
b_w=D^T*b_ffn/sqrt(m)
b_gate_pre=b_w*curved*gate*(1-gate)
b_curved=b_w*gate
b_z=b_curved*curve_slope
grad_G+=outer(b_gate_pre,rms_u)/sqrt(d)
grad_A+=outer(b_z,rms_u)/sqrt(d)
grad_C[s,j]+=b_curved[j]*(1-t[j])/4
grad_C[s+1,j]+=b_curved[j]*t[j]/4
b_rms_u=(G^T*b_gate_pre+A^T*b_z)/sqrt(d)+b_f[0:d]
b_u+=rms_back(rms_u,inv_u,b_rms_u)  // discard b_f history half

b_o=0.25*b_u
grad_O+=outer(b_o,c)/sqrt(d); b_c=O^T*b_o/sqrt(d)
For each head:
  b_alpha_i=dot(b_c_head,v_i_head)
  b_score_i=alpha_i*(b_alpha_i-sum_j(alpha_j*b_alpha_j))
  b_v_i_head=alpha_i*b_c_head
  b_q_head=sum_i(b_score_i*k_i_head)/sqrt(k)
  b_k_i_head=b_score_i*q_head/sqrt(k)
grad_Q+=outer(b_q,n)/sqrt(d)
grad_K+=sum_i outer(b_k_i,M_i)/sqrt(d)
grad_V+=sum_i outer(b_v_i,M_i)/sqrt(d)
b_M_i=(K^T*b_k_i+V^T*b_v_i)/sqrt(d)
b_x=b_u+rms_back(n,inv_x,Q^T*b_q/sqrt(d))
```

Anchor phase one gathers every frame's b_y and summed b_M before ANY commits.
Phase two groups repeated visits by tile, recomputes frozen frames, aggregates
task+policy parameter gradients and commits each tile once. Do not resample
actions or add b_M again in phase two. Hold only one decoded tile and one tile
gradient. Finally add returned b_x0 into b_M[last], then:

```text
grad_E = grad_E_readout
for each prefix occurrence i:
  grad_E[:,byte_i] += rms_back(M_i,prefix_inv[i],b_M_i)
apply E once, after all frozen-anchor corrections/gradients exist
```

Repeated byte occurrences and both tied E uses accumulate before this single
update. This is a continuous surrogate conditional on the realized route, not
differentiation through discrete trit transitions/actions.

## Local predictor and history

Hpred is d×F, F=3d+2. Detached features=[rms(x);rms(y);h;axial_q;axial_r].
unit_clip(v)=v/max(1,||v||2), with finite checks.

```text
true_target=unit_clip(real_anchor_b_y)
prediction=tanh(Hpred*features/sqrt(F))
MSE=mean((prediction-true_target)^2)
grad_Hpred=outer((2/d)*(prediction-true_target)*(1-prediction^2),
                features)/sqrt(F)
local_correction=unit_clip(prediction)
```

Train predictor against the raw tanh prediction; clipping is not a hidden
straight-through derivative. Local correction trains only the current tile.
Local visits never update E, reveal a target, read later tiles or invent routing
advantage. Commit locally, then carry the already-computed y onward.

Retain eight history records FIFO: features, clipped real b_y, seven tile versions,
E version, global opportunity, tile ID. Prune age>8 or any dependency ahead of
current state/lagging by more than one version. h=mean(retained corrections), or
zero; ||h||<=1. Anchors capture h once. Local opportunities prune and replay the
latest compatible predictor record once, then prune/recompute h before each
visit. Capture anchor records at frozen versions; after all corrections/commits,
train predictor and push records in forward visitation order. Count replay and
real-anchor predictor updates separately.

## QA schema and exact exposure schedule

Literal prompt bytes (each displayed \n means one newline):

```text
Use evidence only; otherwise UNKNOWN.\nSOURCE {citation}\n{evidence}\nQ: {question}\nA: 
```

Completion is compact UTF-8 JSON in key order answer,citation, then one newline:

```text
{"answer":"supported text","citation":"supplied citation"}\n
{"answer":"UNKNOWN","citation":null}\n
```

No JSON spaces or ordinary-Unicode ASCII escaping; standard control/quote/
backslash escapes. Pin serialized bytes and masks. Prompt+completion<=1024.
Reject oversized examples; do not truncate their facts. Targets are completion
bytes only, including JSON syntax/newline. Record masks partitioning answer-value,
citation-value and format targets. Prompt bytes are conditioning, not targets.

Cursor={document/example ID, completion target offset, epoch, manifest order
index}. Iterate fixed pinned order; repeat that order next epoch. Pending prefix
is prompt+completion[..offset]. At global g%4==0, execute one anchor, reveal
completion[offset] only for loss, finish updates/history/baseline/counters and
advance cursor exactly once. Otherwise run local updates on the NEXT pending
prefix, never expose its target and never advance cursor. Increment g only at a
complete opportunity boundary. Thus local g=1,2,3 share the prefix whose target
anchor g=4 consumes. Earlier teacher-forced answer bytes are valid causal input.
Partial UTF-8 byte prefixes are legal; do not decode them.

Count committed anchors/completion targets, answer/citation/format target masks,
local opportunities/updates, prefix bytes including repetitions, unique source
bytes/examples, epochs, visits and repeated supervised exposure. Separate logical
committed exposure from physical work replayed after recovery; preserve unknown
physical exposure on failed attempts.

## Parameter and activation ledger

For r routes; seven tiles have sum r=31:

```text
P_tile(r)=8d²+3dm+3m+3dr
P_E=256d; P_predictor=d(3d+2); P_shared=P_E+P_predictor
P_total=59d²+21dm+21m+351d
W_matrix=ceil(P_matrix/4); W_tile/shared=sum individual W_matrix
Evidence=2P bytes; dense gradient=4P bytes
```

| Payload | fixture | candidate |
|---|---:|---:|
| Total coefficients | 37,856 | 2,051,840 |
| Shared coefficients | 4,896 | 82,176 |
| Largest tile coefficients | 4,832 | 282,368 |
| Stored packed weights total | 9,464 B | 512,960 B |
| Resident shared packed weights | 1,224 B | 20,544 B |
| Largest tile packed weights | 1,208 B | 70,592 B |
| Shared evidence | 9,792 B | 164,352 B |
| Largest tile evidence | 9,664 B | 564,736 B |
| Largest tile gradient | 19,328 B | 1,129,472 B |
| E gradient | 16,384 B | 131,072 B |
| Predictor gradient | 3,200 B | 197,632 B |

Largest candidate matrix is predictor49,408, below existing bound. Forward frames
store x only plus tile/version/action/depth metadata; replay internal activations.
No frame owns weights or a persistent projected prefix.

```text
B_prefix=4Ld+4L+L                // M, inverse RMS, raw bytes
B_tape=4Dd+32D+8d               // inputs, metadata, h and final y
B_corrections=4Dd; B_prefix_adjoint=4Ld; B_readout=8V
B_history=8*(4*(F+d)+80)+16
B_arena=4*(2Ld+2HL+9m+64d+8r+4)
```

Scratch payload capacities: K/V=2Ld; attention probabilities/score adjoints=2HL;
FFN gate,z,curved,w,b_w,b_gate,b_curved,b_gate_pre,b_z=9m;
target/pulse/adjoint scratch=64d; route scratch=8r; scalar RMS caches=4.
64d/8r are explicit reserved payload bounds, not necessary-work claims. The
implementation uses named bounded vectors; the directory ledger inventories
their overlapping lifetimes and separate metadata. Stream K/V
adjoints one prefix position at a time. Parameter gradients live outside arena.

```text
B_anchor_core=W_shared+2P_shared
  +max_tile(W_tile+2P_tile+4P_tile)+4Vd
  +B_prefix+B_tape+B_corrections+B_prefix_adjoint
  +B_readout+max_tile(B_arena)+B_history
B_predictor_phase=W_shared+2P_shared+B_history
  +4dF+4(F+4d)+B_prefix+B_tape+B_corrections
B_core_peak=max(B_anchor_core,B_predictor_phase)
```

Predictor update phase holds no tile gradient; local/inference reuse subsets of
these arenas. Candidate L1024 gives B_arena=1,128,176 B and prefix+adjoint
1,053,696 B. These are planned payload bounds, NOT heap/RSS measurements.
The original tape metadata allowance is insufficient on 64-bit targets: Frame
headers are expected to occupy 64 bytes rather than 32, with additional Vec,
record and mesh metadata. Separately account actual size_of metadata, allocator,
stacks, code, OS cache, bounded data buffers, observers and checkpoint I/O.
The model sequence is bounded to 1,024 bytes; the serialized provenance-bearing
QA row has an 8 KiB bound, the manifest 64 KiB, and streaming hash buffers 8 KiB.
Initial tile I/O reserve=2*max(W_tile+2P_tile)+4096 bytes; verify the codec actually
fits it. Use bounded packed binary+evidence payloads, not large JSON integer
arrays. Admission/retention requires the complete measured ledger.

## Immutable generation and complete state

`generate(checkpoint_view,prompt_utf8,max_new_bytes)->raw bytes,valid_utf8,
stop_reason,route_counts,timings,before/after identity`.

Reject empty/invalid-UTF8 prompt, zero byte budget or prompt.len()+budget>1024.
No truncation. Zero training history, greedy routes and greedy byte readout
(lowest-byte tie). Recompute prefix each byte; stop on first generated newline
or budget. Return invalid UTF8/JSON unchanged and score it as invalid, never
repair it invisibly. No predictor/evidence/cursor/baseline updates, sampling
temperature, server or KV cache.

Checkpoint only after complete opportunities: schema/protocol/constants; all
eleven tile matrices and evidence/version refs; E and predictor/evidence/versions;
history ordering/dependencies; baseline; global g; exact QA cursor/order/epoch/
masks and data hashes; S/RNG algorithm; all work/exposure/transition counters;
immutable payload hashes/lengths; owner attempt lineage, consumed aggregate
budgets, STOP and observer sequence. Persist threshold1 evidence or validate an
explicit canonical-zero encoding. Manifest publication is atomic after payload
sync; strict resume rejects missing/mismatched dependencies and resumes into a
fresh owned attempt. No in-flight tape is resumable. The owned supervisor also
requires exact process release and durable terminal observer evidence, preserving
consumed aggregate budgets on continuation.

## Author split in new src/causal

- Numerical: geometry, norm, positions, attention, ffn, pulse, routing, tile,
  tile_gradients, shared and predictor.
- Execution: prefix, execution and credit, implementing frozen gather/commit.
- Data: data, cursor and evaluation, plus pinned manifest/QA preparation.
- Persistence: state, history, checkpoint and storage; new owned supervisor and
  receipt path using reviewed Windows primitives.
- Integrator only: mod, runner, report, generation, new src/causal_cli.rs,
  src/lib.rs/main.rs wiring. Coordinate pure generation with numerical author.

Do not change src/text/**, text CLI/owners/campaign/receipts/tests for this model.
Reuse Mesh and ternary kernels without increasing bounds. Modules can merge
small helpers while preserving single-author boundaries. T3/T4/T5 proceed
with this experimental candidate after the completed T2 diagnostic GO; training still needs
independent numerical/resource/ownership review, all fixes, one final integrated
check and a separately bounded authorized owned pilot. No forecast from fixture
speed or NLL alone.

