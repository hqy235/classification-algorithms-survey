# Reproduction Materials for "A Survey of Classification Algorithms in Pattern Recognition"

This repository contains the code and data that reproduce the experimental
figures in the survey paper *A Survey of Classification Algorithms in Pattern
Recognition*.

## Contents

| File | Description |
|------|-------------|
| `gen_figs.py` | Generates every experimental figure and prints the exact metrics reported in the paper. |
| `iris.csv` | A self-contained snapshot (150 rows) of the real Iris dataset used for all numbers. |
| `fig_boundaries.png` | Decision boundaries of four classifiers on the Iris data (PCA, 2-D). |
| `fig_clustdr.png` | Clustering-after-dimensionality-reduction comparison (raw / PCA / t-SNE). |
| `requirements.txt` | Python dependencies. |

## Reproduce

```bash
pip install -r requirements.txt
python gen_figs.py
```
