"""Plot complete GPU operator measurements, normalized to the SRAM baseline."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
import numpy as np


def plot(source, output):
    data = json.loads(source.read_text())
    if data.get("measurement_scope") != "gpu_kernel" or not data.get("complete"):
        raise ValueError("Complete GPU-kernel results are required")
    cases = data["cases"]
    operators = data["operators"]
    values = []
    for op in operators:
        rows = [op["results"][case["id"]] for case in cases]
        for row in rows:
            for key in ("latency_ns", "gpu_power_mw"):
                if not math.isfinite(row[key]) or row[key] <= 0:
                    raise ValueError(f"Invalid GPU metric: {key}")
        baseline = rows[0]
        values.append([[r["latency_ns"] / baseline["latency_ns"],
                        r["gpu_power_mw"] / baseline["gpu_power_mw"],
                        baseline["latency_ns"] * baseline["gpu_power_mw"] /
                        (r["latency_ns"] * r["gpu_power_mw"])] for r in rows])
    findfont("Arial", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Arial", "font.size": 13,
                         "axes.titlesize": 20, "axes.titleweight": "bold",
                         "axes.labelsize": 15, "axes.linewidth": 1.6,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.subplots_adjust(left=.055, right=.99, bottom=.17, top=.73, wspace=.27)
    colors = ["#D2D2D2", "#A5D5E8", "#245882", "#E45D85"]
    values = np.asarray(values)
    x = np.arange(len(operators))
    width = .78 / len(cases)
    for metric, (ax, title) in enumerate(zip(axes, ["(a) Latency", "(b) Power", "(c) FoM"])):
        for index, case in enumerate(cases):
            ax.bar(x + (index - (len(cases) - 1) / 2) * width,
                   values[:, index, metric], width * .94, label=case["label"],
                   color=colors[index % len(colors)], edgecolor="#202020", linewidth=1.1)
        ax.set_title(title, pad=13)
        ax.set_ylabel("Normalized " + ["latency", "power", "FoM"][metric])
        ax.set_xticks(x, [op["label"] for op in operators])
        ax.set_ylim(0, max(1.0, values[:, :, metric].max()) * 1.15)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", color="#DADADA")
        ax.axhline(1, color="#888888", linewidth=.8, linestyle="--")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .99),
               ncol=len(cases), frameon=False, fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output.with_suffix("." + extension), dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.source, args.output)
