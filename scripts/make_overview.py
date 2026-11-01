"""Render the README example from quick-start outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from phasegraph import read_graph

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "output/quickstart"
plt.rcParams.update({"font.family": "Arial", "font.size": 22, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.9), gridspec_kw={"width_ratios": [1, 1, 1.05]})
fig.subplots_adjust(left=.02, right=.96, top=.81, bottom=.16, wspace=.29)
for ax, title in zip(axes, ["Image", "Region graph", "Pairwise distances"]):
    fig.text(ax.get_position().x0, .86, title, fontsize=25, color="#173449", weight="bold")
axes[0].imshow(Image.open(ROOT / "examples/images/fixture_48.png")); axes[0].set_axis_off()
graph = read_graph(DATA / "graphs/fixture_48/graph.json")
wall = Image.open(DATA / "graphs/fixture_48/wall.png").convert("L")
axes[1].imshow(np.where(np.asarray(wall)>0, 225, 255), cmap="gray", vmin=0, vmax=255)
pos = {n: a["position"] for n, a in graph.nodes(data=True)}
nx.draw_networkx_edges(graph, pos, ax=axes[1], edge_color="#d88742", width=3.5)
nx.draw_networkx_nodes(graph, pos, ax=axes[1], node_color="#237c8d", node_size=250, edgecolors="white", linewidths=2)
axes[1].set_axis_off()
matrix = pd.read_csv(DATA / "matrix/distance_matrix.csv", index_col=0)
im = axes[2].imshow(matrix, cmap="YlGnBu", vmin=0, vmax=.55)
axes[2].set_xticks(range(3), ["48", "49", "50"]); axes[2].set_yticks(range(3), ["48", "49", "50"])
for i in range(3):
    for j in range(3):
        axes[2].text(j, i, f"{matrix.iloc[i,j]:.2f}", ha="center", va="center", color="white" if matrix.iloc[i,j]>.33 else "#173449", fontsize=21)
axes[2].set_xlabel("Synthetic fixture", fontsize=20)
fig.text(.02,.04,"Bundled example · 3 synthetic partitions · no OCR model required",fontsize=18,color="#586775")
target=ROOT / "docs/assets/overview.png";target.parent.mkdir(exist_ok=True,parents=True)
fig.savefig(target, dpi=140, facecolor="white");plt.close(fig)
print(target)
