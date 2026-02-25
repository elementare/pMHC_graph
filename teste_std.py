import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Sequence, List


@dataclass
class Stats:
    mean: float
    std: float
    median: float
    mad: float
    iqr: float


def compute_mad(data: np.ndarray) -> float:
    """
    Median Absolute Deviation (escala bruta, sem constante 1.4826).
    """
    median = np.median(data)
    abs_dev = np.abs(data - median)
    return float(np.median(abs_dev))


def compute_iqr(data: np.ndarray) -> float:
    """
    Interquartile Range Q3 - Q1.
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    return float(q3 - q1)


def compute_stats(data: Sequence[float]) -> Stats:
    data = np.asarray(data, dtype=float)
    return Stats(
        mean=float(np.mean(data)),
        std=float(np.std(data, ddof=1)),
        median=float(np.median(data)),
        mad=compute_mad(data),
        iqr=compute_iqr(data),
    )


def print_stats(label: str, stats: Stats):
    print(f"=== {label} ===")
    print(f"Mean   : {stats.mean:.4f}")
    print(f"Std    : {stats.std:.4f}")
    print(f"Median : {stats.median:.4f}")
    print(f"MAD    : {stats.mad:.4f}")
    print(f"IQR    : {stats.iqr:.4f}")
    print()


def simulate_outlier_effect(
    base_data: Sequence[float],
    outlier_values: Sequence[float],
    plot: bool = True,
):
    """
    Para um conjunto base de distancias, adiciona diferentes valores
    de outlier e mostra como Std e MAD se comportam.
    """
    base_data = np.asarray(base_data, dtype=float)
    base_stats = compute_stats(base_data)
    print_stats("Base data", base_stats)

    std_list: List[float] = []
    mad_list: List[float] = []
    mean_list: List[float] = []
    median_list: List[float] = []

    outlier_values = np.asarray(outlier_values, dtype=float)

    for v in outlier_values:
        augmented = np.concatenate([base_data, [v]])
        s = compute_stats(augmented)
        std_list.append(s.std)
        mad_list.append(s.mad)
        mean_list.append(s.mean)
        median_list.append(s.median)

    print("Outlier  Std      MAD      Mean     Median")
    for v, sd, md, m, med in zip(outlier_values, std_list, mad_list, mean_list, median_list):
        print(f"{v:7.2f}  {sd:7.4f}  {md:7.4f}  {m:7.4f}  {med:7.4f}")

    if plot:
        plt.figure()
        plt.title("Effect of outlier on Std and MAD")
        plt.plot(outlier_values, std_list, label="Std")
        plt.plot(outlier_values, mad_list, label="MAD")
        plt.xlabel("Outlier value")
        plt.ylabel("Scale")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Exemplo: distancias de uma triade relativamente coerente
    base_d1 = [4.5, 4.7, 4.6, 4.8, 4.4, 4.7, 4.5, 4.6]

    # Vamos variar um outlier de 5 ate 20
    outlier_values = np.linspace(5.0, 20.0, 16)

    simulate_outlier_effect(base_d1, outlier_values, plot=True)
