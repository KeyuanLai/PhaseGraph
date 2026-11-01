# Using PhaseGraph

## Images and extraction

`phasegraph extract --images INPUT --out OUTPUT` processes a folder of PNG or JPEG images. The geometry pipeline detects thin coloured boundaries and uses dark-interference neighbourhoods for local repair. Light-background diagrams with blue or red boundaries are the supported input convention. Dense coloured backgrounds are rejected.

For each image, extraction saves `graph.json`, `nodes.csv`, `adjacency_edges.csv`, region labels, boundary masks and extraction settings. `graph.json` is the preferred interchange format because it preserves isolated nodes. Edge CSV files alone cannot represent them.

Repair can be selected with `--repair-strategy local`, `none` or `global`; local is the default. `--canonical-width 1024` restores the pixel scale before morphology while preserving the aspect ratio. All settings are written with the outputs.

## Distances and galleries

`phasegraph matrix --graphs GRAPHS --out OUTPUT` constructs pairwise distances. A graph folder may contain extraction subfolders; graph JSONs are found recursively. `phasegraph fit` creates a frozen gallery for later queries.

The default uses 30 finite shortest-path bins, diagonal-normalised geometric costs and weights 0.45/0.45/0.10. `--encoding continuous` enables interpolated path assignment. Use the same coordinate convention for the gallery and queries. An optional `--coordinate-metric axes` during extraction removes independent axis scaling for a fixed graph.

The default overflow policy rejects query paths beyond the fitted finite support. `--overflow tail` during fitting adds a finite overflow channel. This keeps finite overflow separate from disconnected nodes.

```bash
phasegraph fit --graphs output/graphs --overflow tail --out output/gallery
phasegraph query --model output/gallery/model.json --graph output/graphs/fixture_48/graph.json --top-k 3 --out output/query
```

A query from the gallery itself is included among the candidates. When checking neighbours, exclude its own identifier as appropriate.

## Clustering

```bash
phasegraph cluster --matrix output/matrix/distance_matrix.csv --groups 2 --out output/clusters
```

Clustering uses positive PCoA coordinates followed by Ward or k-means. Outputs include assignments, coordinates and embedding diagnostics. Choose the group count with `--groups`; the tool does not infer it automatically.

## Outputs and troubleshooting

Use a fresh output directory for each run. Commands save `command.txt`, `config.yaml`, `result.csv` and a run log. Run `phasegraph doctor` to check installed dependency versions, or append `--help` to a command for its options.

If no regions are recovered, inspect `raw.png`, `fixed.png` and `wall.png`, check the image colour convention and frame, and confirm that narrow boundaries remain visible at the processing resolution. OCR text labels are not required for the current geometry workflow.
