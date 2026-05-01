# PrefBench arXiv Draft

This directory contains the LaTeX source for the PrefBench technical report.
The paper uses the local `arxiv.sty` template and BibTeX references from
`data/UG_Thesis.bib`.

Build locally with:

```bash
cd arxiv_paper
make
```

Or run the equivalent command directly:

```bash
latexmk -pdf -interaction=nonstopmode main.tex
```

Clean generated files with:

```bash
make clean
```

The arXiv draft is organized under `sections/` and compiled through `main.tex`.
The `data/` directory only keeps the bibliography file and figures used by the
current draft.
