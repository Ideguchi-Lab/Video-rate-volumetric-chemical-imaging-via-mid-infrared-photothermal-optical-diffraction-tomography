# Data

Raw measurements are **not** tracked in git. Place them here (or anywhere and
point `MIPODT_DATA_DIR` at it) with the layout below, or run
`uv run scripts/download_data.py` once the dataset is published.

```
data/
├── fig1/
├── fig2/
├── fig3/
└── figS1/
```

Each `figures/<fig>/README.md` lists the exact files it needs.
