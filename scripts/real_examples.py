"""Extract the four real-diagram examples and draw their region graphs."""
import argparse
import colorsys
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from phasegraph.io import read_graph

ROOT = Path(__file__).resolve().parent.parent


def font(size):
    for path in ("Arial.ttf", "C:/Windows/Fonts/arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def preview(image_path, folder, destination):
    graph = read_graph(folder / "graph.json")
    with np.load(folder / "regions.npz") as data:
        labels = data["labels"]
    height, width = labels.shape
    palette = {}
    for index, node in enumerate(sorted(graph, key=int)):
        rgb = colorsys.hsv_to_rgb((index * 0.618034 + .08) % 1, .32, .92)
        palette[node] = tuple(round(c * 255) for c in rgb)
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    for node, color in palette.items():
        pixels[labels == int(node)] = color
    wall = np.asarray(Image.open(folder / "wall.png")) > 0
    pixels[wall] = (80, 88, 96)
    regions = Image.fromarray(pixels)
    network = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(network)
    points = {node: tuple(graph.nodes[node]["position"]) for node in graph}
    for u, v in graph.edges:
        draw.line((points[u], points[v]), fill="#b8804a", width=3)
    for node, (x, y) in points.items():
        draw.ellipse((x-7, y-7, x+7, y+7), fill=palette[node],
                     outline="#33485e", width=2)
    canvas = Image.new("RGB", (1800, 560), "white")
    draw = ImageDraw.Draw(canvas)
    images = [Image.open(image_path).convert("RGB"), regions, network]
    headings = [graph.graph["name"], "Recovered regions", "Region graph"]
    for index, (image, heading) in enumerate(zip(images, headings)):
        tile = ImageOps.contain(image, (564, 440), Image.Resampling.LANCZOS)
        canvas.paste(tile, (index*600 + (600-tile.width)//2,
                            64 + (440-tile.height)//2))
        draw.text((index*600+22, 16), heading, font=font(28), fill="#24374b")
    draw.text((622, 520), f"{len(graph)} regions", font=font(24), fill="#4d5f70")
    draw.text((1222, 520), f"{len(graph)} nodes, {graph.number_of_edges()} edges",
              font=font(24), fill="#4d5f70")
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/real_examples")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Choose an empty output directory")
    images = ROOT / "examples/real/images"
    subprocess.run([sys.executable, "-m", "phasegraph", "extract", "--images",
                    str(images), "--repair-strategy", "local", "--canonical-width",
                    "1024", "--out", str(out / "graphs")], check=True)
    for path in sorted(images.glob("*.png")):
        preview(path, out / "graphs" / path.stem, out / f"{path.stem}.png")
    print(f"Examples saved to {out}")


if __name__ == "__main__":
    main()
