# Paper

`main.tex` is the lead write-up of Hexweave's CPU results: bounded-residency, per-tile online
training of a paged byte-level language model. `main.pdf` is the compiled copy.

- `main.tex`: the paper. Figures are drawn by TikZ/pgfplots directly from `data/*.csv`.
- `refs.bib`: bibliography (natbib, `plainnat`).
- `data/`: exported held-out evaluation series (`anchors,heldout_bpb`) and 12-evaluation moving means
  (`anchors,mean_bpb`) for the two long runs and the 30-minute pilots.

Values that depend on the long run still in progress are marked with the red `\pending{...}`
macro (defined in the preamble). Search for `\pending` to fill them in when that run ends.

## Build

Requires a TeX distribution with `pgfplots`, `tikz`, `booktabs`, `natbib`, `hyperref`, `amsmath`,
`enumitem`, `caption`, `microtype` and `lmodern` (MiKTeX and TeX Live both ship these). From this
directory:

```sh
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

With MiKTeX, add `--disable-installer` to the `pdflatex` calls to prevent automatic package
downloads.

## Build by-products

Not deliverables; suggested `.gitignore` entries for this directory:

```gitignore
paper/*.aux
paper/*.bbl
paper/*.blg
paper/*.log
paper/*.out
paper/*.toc
paper/*.synctex.gz
paper/*.fdb_latexmk
paper/*.fls
```
