# Authored byte fixture

These short UTF-8 files were authored for this repository's bounded text
experiment. They do not copy a parent dataset, private user documents, or an
external corpus. The text is provided under CC0-1.0 for reuse as a test fixture.

`train.txt` alone supplies update inputs and next-byte targets. `valid.txt` is a
distinct document used only for read-only before/after evaluation. Shared words
and byte sequences are expected in language; this is document separation, not
a claim of zero overlapping n-grams. Evaluation samples at most 64 successive
eight-byte contexts, resetting history for every example.

`development-t2.txt` is a fresh document frozen before T2 implementation and
observation. The explicit T2 profile requires this file and the unchanged training
file by content identity; it cannot silently evaluate `valid.txt`. The first 72
ASCII bytes supply the 64 evaluated contexts/targets. Full-document and consumed
prefix identities are in [the T2 contract](../../docs/T2-CREDIT-DIAGNOSTIC.md).
The remaining development bytes are not evaluated by this bounded fixture.

The vocabulary is all 256 byte values, without a learned tokenizer. Each input
file is capped at 64 KiB. Results include exact byte counts and deterministic
FNV-1a identities; those are reproducibility checks, not cryptographic hashes.
This small artificial corpus cannot establish useful language capability or
predict the time needed to train the planned 81M/287M attention models.
