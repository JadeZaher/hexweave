# Source provenance

Checked 2026-10-23. These sources establish API and model identities, not
Hexweave's performance or convergence.

| Source | What it supports |
| --- | --- |
| [Candle repository](https://github.com/huggingface/candle) | Rust tensor framework and training examples |
| [Candle 0.11.0 Var API](https://docs.rs/candle-core/0.11.0/candle_core/struct.Var.html) | Trainable variables, tensor construction and mutation |
| [Candle MNIST training example](https://github.com/huggingface/candle/blob/main/candle-examples/examples/mnist-training/main.rs) | Official automatic-differentiation training usage; main is not an immutable source pin |
| [IBM Granite 4.0 350M](https://huggingface.co/ibm-granite/granite-4.0-350m) | Official small instruct comparison candidate and license |
| [IBM Granite 4.0 H 350M](https://huggingface.co/ibm-granite/granite-4.0-h-350m) | Official hybrid small instruct comparison candidate and license |
| [IBM Granite 4.0 H 1B](https://huggingface.co/ibm-granite/granite-4.0-h-1b) | Larger local comparison anchor, not the seed's size target |

The three Granite cards describe instruction-tuned models. Comparing them with
a new scratch base model is a deployment comparison, not evidence that one
architecture learns more efficiently. Pin immutable revisions, local artifact
hashes and templates before measuring. No Granite artifact was downloaded or
loaded while creating this scaffold.

Local design inspection used the source repository's
`routing_research/model/{recurrent.py,bank.py,AGENTS.md}`,
`routing_research/activations.py`, `hex_lab/{GUIDE.md,ARCHITECTURE.md}` and
`wave_lab/{README.md,ARCHITECTURE.md}`. Those are design references, not runtime
imports. No source experiment history, weights, corpus or environment is copied.

The bumblebee social-learning illustration is an owner-provided inspiration,
not a biological result newly verified by this software task.
