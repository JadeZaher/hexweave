# Contributing

Read [AGENTS.md](AGENTS.md) and [the architecture](docs/ARCHITECTURE.md) first.
Preserve the distinction between a software fixture and a training experiment.

Work in focused file sets: configuration/topology, numerical operations, learning,
benchmark protocol or runtime. Name the entry point and invariants in that
directory's documentation. Aim for source files under 300 lines and functions
under 90 lines; split by responsibility, not arbitrary numbered fragments.
Use a blank line between functions and short doc-comments. Put explanatory
essays in directory docs, and cite the algorithm where it is implemented.

Every consequential change needs a reviewer other than its author. A supportive
review checks whether it advances the owner's hypothesis; an adversarial review
tries to falsify its correctness, safety and claims. Neither may replace tests.
Apply all review fixes before the final integrated validation sweep.

```powershell
.\scripts\validate.ps1 -Offline
```

Do not silently start servers, fetch model weights or download datasets.
Runtime files and result JSON belong under ignored `work/`, with new paths per
run. Never delete or terminate unrelated processes to make a run fit.
Full-size training needs a separate reviewed run protocol and explicit request.

No remote is configured by this scaffold. A future license or publication
decision belongs to the owner; dependency and model licenses remain separate.
