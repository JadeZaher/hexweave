# Packed kernels

`head_rows.rs` implements checked contiguous output-row products for factored
single-query attention. It reads packed coefficients without decoded matrices
or zero-padded full-width inputs. Forward reduces ascending input columns;
transpose reduces ascending selected rows using the existing add/subtract/skip
primitive. These small block methods are scalar regardless of the resolved
full-matrix family. Existing full-matrix kernel arithmetic is unchanged.

`kernels.rs` validates vector shapes and finite inputs, dispatches operations,
and rejects nonfinite outputs. `HEXWEAVE_KERNEL=auto|scalar|avx2` is resolved
once per process. The environment must be set before the first dispatch; changing
it later has no effect. Auto selects AVX2 on supported x86-64 CPUs and scalar
elsewhere. Explicit AVX2 requests fail on unsupported CPUs. Separate processes
are required when benchmarking different environment choices.
Selected AVX2 dispatch falls back directly to scalar when forward has fewer than
eight rows or transpose has fewer than eight columns. Such shapes contain no
complete SIMD output group. Explicit AVX2 methods still exercise their scalar
tails for parity tests; `kernel_name` names the selected family, not every call.

`avx2.rs` uses eight independent output lanes. Forward keeps the original
column reduction order; transpose keeps the original row reduction order.
There is no horizontal reassociation, floating-point multiplication, FMA, or
decoded weight cache. Zero trits blend the old accumulator unchanged, including
its sign bit. Vector tails use the scalar add/subtract/skip rule. Intermediate
arithmetic stays f32; weights remain packed ternary throughout.

Every AVX2 entry point is behind runtime detection and validated dimensions.
The only pointer accesses load a complete eight-i32 stack array or store a
complete eight-f32 output region. Both use unaligned intrinsics and are bounded
by the complete-lane count. Packed extraction uses checked byte reads; absent
high bytes contribute zero only beyond the final valid trit. No cross-row
padding or aligned allocation is assumed. Target-feature requirements follow
[Rust's architecture documentation](https://doc.rust-lang.org/core/arch/index.html),
[variable shifts](https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_srlv_epi32.html),
and [masked selection](https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_blendv_ps.html).

`from_packed` accepts exactly ceil(rows * cols / 4) bytes, codes 0/1/2 for
trits -1/0/+1, and zero unused high bits in the final byte. Code 3, noncanonical
padding, invalid dimensions, and extra or missing bytes are rejected before
construction. `column` returns unscaled trits for a selected column and does not
decode the rest of the matrix. Neither operation loads model-wide caches.

SIMD is an implementation mechanism, not a performance claim. Tiny or strided
products can be slower than the scalar reference. Compare release builds with
identical seeds and receipts before selecting a default based on measured cost.

## Update experiments

A threshold-one packed-byte specialization was rejected after measured runtime
regression. The original general update remains the production rule for every
threshold. Its adversarial tests remain in `update_contracts.rs`: inherited
evidence, saturating i16 arithmetic, zero gradients after lowering a threshold,
partial-byte padding, and whole-gradient validation before mutation.
