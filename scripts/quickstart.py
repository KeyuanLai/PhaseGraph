"""Run the bundled synthetic example through the command-line interface."""
from pathlib import Path
import argparse
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/quickstart")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Choose an empty output directory")
    commands = [
        ["doctor"],
        ["extract", "--images", str(ROOT / "examples/images"), "--out", str(out / "graphs")],
        ["matrix", "--graphs", str(out / "graphs"), "--out", str(out / "matrix")],
        ["fit", "--graphs", str(out / "graphs"), "--out", str(out / "gallery")],
        ["query", "--model", str(out / "gallery/model.json"), "--graph", str(out / "graphs/fixture_48/graph.json"), "--top-k", "3", "--out", str(out / "query")],
        ["cluster", "--matrix", str(out / "matrix/distance_matrix.csv"), "--groups", "2", "--out", str(out / "clusters")],
    ]
    for command in commands:
        subprocess.run([sys.executable, "-m", "phasegraph", *command], check=True, cwd=ROOT)
    matrix = pd.read_csv(out / "matrix/distance_matrix.csv", index_col=0).to_numpy()
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(matrix), 0, atol=1e-7)
    query = pd.read_csv(out / "query/result.csv")
    assert query.iloc[0]["name"] == "fixture_48" and query.iloc[0]["distance"] < 1e-7
    print(f"Example completed: {out}")

if __name__ == "__main__":
    main()
