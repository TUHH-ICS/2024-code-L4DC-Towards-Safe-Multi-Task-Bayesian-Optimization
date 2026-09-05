"""Recreate the comparison figure from the bundled experiment data."""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import scipy.io
import torch

PRIMARY_FILES = [
    Path("data_paper/data/num_laser2_dist0_lengthscale010.obj"),
    Path("data_paper/data/num_laser2_dist1_lengthscale010.obj"),
    Path("data_paper/data/num_laser2_dist3_lengthscale010.obj"),
    Path("data_paper/data/num_laser2_dist5_lengthscale010.obj"),
    Path("data_paper/data/num_laser2_dist7_lengthscale010.obj"),
]

COMPARISON_FILES = [
    Path("data_paper/data/num_laser5_pure_bo.obj"),
    Path("data_paper/data/matlabSim_num_laser5.mat"),
    Path("data_paper/data/num_laser5_dist1_lengthscale010_2.obj"),
]


def configure_plotting():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": "Helvetica",
            "font.size": 12,
            "axes.grid": True,
        }
    )


def plot_disturbance_result(ax, path, color_index):
    with path.open("rb") as file:
        data = pickle.load(file)

    mean_objective = -torch.tensor([result[-1] for result in data["bests"]]).mean(dim=0)
    source = str(path)
    disturbance_start = source.find("dist") + len("dist")
    disturbance_end = source.find("_lengthscale")
    disturbance_percent = int(source[disturbance_start:disturbance_end]) * 10
    ax.plot(
        mean_objective,
        f"--C{color_index}",
        label=rf"$\pm {disturbance_percent}\%$",
    )


def plot_comparison_result(ax, path, color_index):
    if path.suffix == ".obj":
        with path.open("rb") as file:
            data = pickle.load(file)
        std_objective, mean_objective = torch.std_mean(
            -torch.tensor([result[-1] for result in data["bests"]]), dim=0
        )
    else:
        data = scipy.io.loadmat(path)
        std_objective = data["std_Y"].squeeze()
        mean_objective = data["Y"].squeeze()

    evaluations = torch.arange(len(mean_objective))
    fill = ax.fill_between(
        evaluations,
        mean_objective + std_objective,
        mean_objective - std_objective,
        color=f"C{color_index}",
        alpha=0.3,
    )
    (line,) = ax.plot(evaluations, mean_objective, f"C{color_index}")
    return fill, line


def main():
    configure_plotting()
    figure, axes = plt.subplots(1, 2, figsize=(9, 2))

    for label, axis in zip(("(a)", "(b)"), axes):
        transform = mtransforms.ScaledTranslation(
            0 / 72, 3 / 72, figure.dpi_scale_trans
        )
        axis.text(
            0.0,
            1.0,
            label,
            transform=axis.transAxes + transform,
            fontsize=14,
            va="bottom",
            ha="center",
        )

    for index, path in enumerate(PRIMARY_FILES):
        plot_disturbance_result(axes[0], path, index)
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_xlabel(r"Evaluations of main task $n$")
    axes[0].set_ylabel(r"$J_{opt}(n)$")
    axes[0].set_xlim(0, 50)
    axes[0].set_yticks([5, 10, 15, 20])
    axes[0].set_ylim(5, 25)

    handles = [
        plot_comparison_result(axes[1], path, index)
        for index, path in enumerate(COMPARISON_FILES)
    ]
    axes[1].legend(
        handles,
        [r"\texttt{SafeBO} + EI", r"\texttt{MoSaOpt}", r"\texttt{SaMSBO} (our)"],
    )
    axes[1].set_xlabel(r"Evaluations of main task $n$")
    axes[1].set_xlim(0, 200)
    axes[1].set_yticks([14, 16, 18, 20])
    axes[1].set_ylim(14, 22)

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)
    figure.savefig(
        output_dir / "comparison.pdf",
        bbox_inches="tight",
        pad_inches=0.01,
        format="pdf",
    )
    plt.show()


if __name__ == "__main__":
    main()
