# Synthetic software inputs

`tiny.jsonl` exists to exercise parsing and result schemas. It is not a quality
benchmark, held-out research corpus or evidence of language-model capability.
The test suite creates separate fake weight/tokenizer/template files inside
temporary directories and labels their manifests `software_fixture`.

No ready-to-run model manifest is included. `../granite.template.json` must fail
validation until its identities, artifact paths and dataset are pinned.
