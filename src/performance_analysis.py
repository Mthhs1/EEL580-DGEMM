from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-degemm")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "Dados_Coletados.ods"
OUTPUT_DIR = ROOT_DIR / "src" / "analysis_outputs"

VERSION_ORDER = [
    "Python",
    "NumPy",
    "C",
    "C AVX",
    "C AVX + Unroll",
    "C AVX + Unroll + Blocking",
]

VERSION_RENAME = {
    "Python": "Python",
    "Numpy": "NumPy",
    "C": "C",
    "C (avx)": "C AVX",
    "C (Block)": "C AVX + Unroll + Blocking",
}

VERSION_SLUG = {
    "Python": "python",
    "NumPy": "numpy",
    "C": "c",
    "C AVX": "c_avx",
    "C AVX + Unroll": "c_avx_unroll",
    "C AVX + Unroll + Blocking": "c_avx_block",
}

SIZE_COLUMNS = {
    512: (2, 3),
    1024: (4, 5),
    2048: (6, 7),
    4096: (8, 9),
    8192: (10, 11),
}


def _clean_number(value: object) -> float:
    if pd.isna(value):
        return np.nan

    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan

    if number <= 0:
        return np.nan

    return number


def _section_starts(raw_df: pd.DataFrame) -> list[int]:
    starts: list[int] = []

    for row_index in range(len(raw_df) - 2):
        name = raw_df.iat[row_index, 2]
        next_label = raw_df.iat[row_index + 1, 1]

        if isinstance(name, str) and isinstance(next_label, str) and next_label.strip() == "Iterações":
            starts.append(row_index)

    return starts


def _normalise_version(raw_name: str, repeated_lpi_count: int) -> str:
    if raw_name == "C (LPI)" and repeated_lpi_count == 0:
        return "C AVX + Unroll"

    if raw_name == "C (LPI)" and repeated_lpi_count == 1:
        return "C AVX + Unroll + Blocking"

    return VERSION_RENAME.get(raw_name, raw_name)


def load_measurements(data_path: Path = DATA_PATH) -> pd.DataFrame:
    raw_df = pd.read_excel(data_path, sheet_name="Sheet1", engine="odf", header=None)
    rows: list[dict[str, object]] = []
    lpi_count = 0

    for start in _section_starts(raw_df):
        raw_name = str(raw_df.iat[start, 2]).strip()
        version = _normalise_version(raw_name, lpi_count)

        if raw_name == "C (LPI)":
            lpi_count += 1

        for test_offset in range(3, 8):
            iteration = raw_df.iat[start + test_offset, 1]

            if pd.isna(iteration):
                continue

            for matrix_size, (wall_col, cpu_col) in SIZE_COLUMNS.items():
                rows.append(
                    {
                        "version": version,
                        "version_slug": VERSION_SLUG[version],
                        "matrix_size": matrix_size,
                        "iteration": int(iteration),
                        "wall_time_s": _clean_number(raw_df.iat[start + test_offset, wall_col]),
                        "cpu_time_s": _clean_number(raw_df.iat[start + test_offset, cpu_col]),
                    }
                )

    measurements = pd.DataFrame(rows)
    measurements["version"] = pd.Categorical(measurements["version"], VERSION_ORDER, ordered=True)
    measurements = measurements.sort_values(["version", "matrix_size", "iteration"]).reset_index(drop=True)
    return measurements


def build_summary(measurements: pd.DataFrame) -> pd.DataFrame:
    grouped = measurements.groupby(["version", "version_slug", "matrix_size"], observed=True)
    summary = grouped.agg(
        n_tests=("wall_time_s", "count"),
        mean_wall_s=("wall_time_s", "mean"),
        median_wall_s=("wall_time_s", "median"),
        std_wall_s=("wall_time_s", "std"),
        min_wall_s=("wall_time_s", "min"),
        max_wall_s=("wall_time_s", "max"),
        mean_cpu_s=("cpu_time_s", "mean"),
        std_cpu_s=("cpu_time_s", "std"),
    ).reset_index()

    summary["std_wall_s"] = summary["std_wall_s"].fillna(0.0)
    summary["std_cpu_s"] = summary["std_cpu_s"].fillna(0.0)
    summary["operations"] = 2.0 * summary["matrix_size"].astype(float) ** 3
    summary["gflops_wall"] = summary["operations"] / (summary["mean_wall_s"] * 1e9)
    summary["gflops_cpu"] = summary["operations"] / (summary["mean_cpu_s"] * 1e9)
    summary["ci95_wall_s"] = 1.96 * summary["std_wall_s"] / np.sqrt(summary["n_tests"])

    python_baseline = summary[summary["version"].astype(str) == "Python"][
        ["matrix_size", "mean_wall_s", "std_wall_s", "n_tests"]
    ].rename(
        columns={
            "mean_wall_s": "python_mean_wall_s",
            "std_wall_s": "python_std_wall_s",
            "n_tests": "python_n_tests",
        }
    )

    summary = summary.merge(python_baseline, on="matrix_size", how="left")
    summary["speedup_vs_python"] = summary["python_mean_wall_s"] / summary["mean_wall_s"]

    version_cv = (summary["std_wall_s"] / summary["mean_wall_s"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    python_cv = (
        summary["python_std_wall_s"] / summary["python_mean_wall_s"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    summary["speedup_std"] = summary["speedup_vs_python"] * np.sqrt(version_cv**2 + python_cv**2)
    summary["speedup_ci95"] = 1.96 * summary["speedup_std"] / np.sqrt(
        np.minimum(summary["n_tests"], summary["python_n_tests"])
    )

    summary["previous_mean_wall_s"] = summary.groupby("matrix_size", observed=True)["mean_wall_s"].shift(1)
    summary["speedup_vs_previous"] = summary["previous_mean_wall_s"] / summary["mean_wall_s"]

    return summary.sort_values(["version", "matrix_size"]).reset_index(drop=True)


def _plot_time(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    for version in VERSION_ORDER:
        data = summary[summary["version"].astype(str) == version]
        ax.plot(data["matrix_size"], data["mean_wall_s"], marker="o", linewidth=2, label=version)

    ax.set_title("Tempo medio de execucao por tamanho de matriz")
    ax.set_xlabel("Tamanho da matriz (N x N)")
    ax.set_ylabel("Wall time medio (s)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "01_tempo_medio.png", dpi=200)
    plt.close(fig)


def _plot_gflops(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    for version in VERSION_ORDER:
        data = summary[summary["version"].astype(str) == version]
        ax.plot(data["matrix_size"], data["gflops_wall"], marker="o", linewidth=2, label=version)

    ax.set_title("Desempenho em GFLOPS")
    ax.set_xlabel("Tamanho da matriz (N x N)")
    ax.set_ylabel("GFLOPS")
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "02_gflops.png", dpi=200)
    plt.close(fig)


def _plot_speedup(summary: pd.DataFrame, output_dir: Path) -> None:
    versions = [version for version in VERSION_ORDER if version != "Python"]
    sizes = sorted(summary["matrix_size"].unique())
    x = np.arange(len(sizes))
    width = 0.13

    fig, ax = plt.subplots(figsize=(12, 6))

    for index, version in enumerate(versions):
        data = summary[summary["version"].astype(str) == version].set_index("matrix_size").loc[sizes]
        offset = (index - (len(versions) - 1) / 2) * width
        ax.bar(x + offset, data["speedup_vs_python"], width=width, label=version)

    ax.set_title("Speedup em relacao ao Python")
    ax.set_xlabel("Tamanho da matriz (N x N)")
    ax.set_ylabel("Speedup")
    ax.set_xticks(x)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.set_yscale("log")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "03_speedup_vs_python.png", dpi=200)
    plt.close(fig)


def _plot_speedup_evolution(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(VERSION_ORDER))

    for matrix_size in sorted(summary["matrix_size"].unique()):
        data = summary[summary["matrix_size"] == matrix_size].set_index("version").loc[VERSION_ORDER]
        ax.errorbar(
            x,
            data["speedup_vs_python"],
            yerr=data["speedup_ci95"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=f"{matrix_size}x{matrix_size}",
        )

    ax.set_title("Evolucao do speedup por versao")
    ax.set_xlabel("Versao")
    ax.set_ylabel("Speedup vs Python (IC 95%)")
    ax.set_xticks(x)
    ax.set_xticklabels(VERSION_ORDER, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "04_evolucao_speedup.png", dpi=200)
    plt.close(fig)


def _plot_gflops_heatmap(summary: pd.DataFrame, output_dir: Path) -> None:
    heatmap = summary.pivot(index="version", columns="matrix_size", values="gflops_wall").loc[VERSION_ORDER]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    image = ax.imshow(heatmap.to_numpy(), aspect="auto", cmap="viridis")

    ax.set_title("Mapa de calor de GFLOPS")
    ax.set_xlabel("Tamanho da matriz")
    ax.set_ylabel("Versao")
    ax.set_xticks(np.arange(len(heatmap.columns)))
    ax.set_xticklabels([str(size) for size in heatmap.columns])
    ax.set_yticks(np.arange(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index.astype(str))

    for row_index in range(heatmap.shape[0]):
        for col_index in range(heatmap.shape[1]):
            value = heatmap.iat[row_index, col_index]
            ax.text(col_index, row_index, f"{value:.1f}", ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(image, ax=ax, label="GFLOPS")
    fig.tight_layout()
    fig.savefig(output_dir / "05_heatmap_gflops.png", dpi=200)
    plt.close(fig)


def save_outputs(measurements: pd.DataFrame, summary: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements.to_csv(output_dir / "measurements_long.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)

    _plot_time(summary, output_dir)
    _plot_gflops(summary, output_dir)
    _plot_speedup(summary, output_dir)
    _plot_speedup_evolution(summary, output_dir)
    _plot_gflops_heatmap(summary, output_dir)


def print_summary(summary: pd.DataFrame) -> None:
    columns = [
        "version",
        "matrix_size",
        "n_tests",
        "mean_wall_s",
        "std_wall_s",
        "gflops_wall",
        "speedup_vs_python",
        "speedup_vs_previous",
    ]
    display_df = summary[columns].copy()
    display_df["mean_wall_s"] = display_df["mean_wall_s"].round(6)
    display_df["std_wall_s"] = display_df["std_wall_s"].round(6)
    display_df["gflops_wall"] = display_df["gflops_wall"].round(3)
    display_df["speedup_vs_python"] = display_df["speedup_vs_python"].round(2)
    display_df["speedup_vs_previous"] = display_df["speedup_vs_previous"].round(2)
    print(display_df.to_string(index=False))


def main() -> None:
    measurements = load_measurements()
    summary = build_summary(measurements)
    save_outputs(measurements, summary)
    print_summary(summary)
    print(f"\nArquivos gerados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
