"""Render the figures used in the README: the two-stage scale-space reverse process.

Run after training `small` and `big` (see the makefile), e.g.

    python3 make_figures.py --small_name small --big_name big
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import utils
from models.DDPM import DDPM
from models.DDPM_big import DDPM_big

OUT_DIR = "figures"

# Intermediate states live in the normalized space the models operate in, where clean MNIST
# spans about (-0.42, 2.82) and the noise is standard gaussian. A fixed window keeps
# brightness comparable across a trajectory instead of auto-scaling every frame.
NOISY_RANGE = (-2.0, 3.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Render README figures")
    parser.add_argument("--small_name", type=str, default="small")
    parser.add_argument("--big_name", type=str, default="big")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def read_run_args(run_name):
    """Recover the hyperparameters a run was trained with from its args.txt."""
    values = {}
    with open("checkpoints/{}/args.txt".format(run_name)) as f:
        for line in f:
            key, value = line.split()
            key = key.replace("--", "", 1)
            try:
                values[key] = eval(value)
            except (NameError, SyntaxError):
                values[key] = value
    return values


def load_models(small_name, big_name, device):
    s = read_run_args(small_name)
    b = read_run_args(big_name)

    small = DDPM(s["img_size"], 11, s["markov_states"], s["unet_stages"], s["noise_power"], device=device)
    small = utils.load_checkpoint(small, small_name).to(device)
    small.eval()

    big = DDPM_big(b["img_size"], 11, b["markov_states"], b["unet_stages"], b["noise_power"], device=device)
    big = utils.load_checkpoint(big, big_name).to(device)
    big.eval()

    return small, big


def upscale_bridge(small_samples, small_size, big_size):
    """Hand a low-res sample to the upscaler in the space it was trained on.

    The upscaler saw conditioning images normalized with its own resolution's statistics, so
    the low-res sample goes back to the [0, 1] image domain before being resized and
    re-normalized. Resizing normalized values directly would inflate their spread.
    """
    unit = utils.denormalize_images(small_samples, small_size, clamp=False)
    resized = torchvision.transforms.Resize(big_size, antialias=True)(unit)
    return utils.normalize_images(resized, big_size), resized.clamp(0.0, 1.0)


@torch.no_grad()
def run_cascade(small, big, labels, device):
    """Sample the full two-stage chain, keeping every intermediate state."""
    labels = labels.to(device)
    small_traj = small.sample(len(labels), labels, keep_intermediate=True)
    small_final = small_traj[:, -1]

    condition, bridge_unit = upscale_bridge(small_final, small.image_size, big.image_size)
    big_traj = big.sample(len(labels), labels, condition, keep_intermediate=True)

    return small_traj, bridge_unit, big_traj


# Panels are meant to read as a filmstrip, so leave almost no gutter between them.
TIGHT = {"wspace": 0.06, "hspace": 0.06}


def blank_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def frame_label(c, n_states):
    """Frame c of a kept trajectory holds x_t for t = n_states-2-c; the last one is the sample."""
    return "$x_0$" if c >= n_states - 1 else f"t={n_states - 2 - c}"


def show(ax, img, vmin, vmax, interpolation="nearest"):
    ax.imshow(img.squeeze().cpu().numpy(), cmap="gray", vmin=vmin, vmax=vmax, interpolation=interpolation)
    blank_axis(ax)


def figure_lowres_reverse(small, labels, small_traj, path):
    """Every state of the 14x14 reverse process, one row per sample."""
    n_samples, n_states = small_traj.shape[0], small_traj.shape[1]
    fig, axes = plt.subplots(n_samples, n_states, figsize=(n_states * 0.42, n_samples * 0.52), gridspec_kw=TIGHT)
    for r in range(n_samples):
        for c in range(n_states):
            ax = axes[r, c]
            show(ax, small_traj[r, c], *NOISY_RANGE)
            if r == 0 and (c % 3 == 0 or c == n_states - 1):
                ax.set_title(frame_label(c, n_states), fontsize=7, pad=3)
        axes[r, 0].set_ylabel(str(int(labels[r])), fontsize=9, rotation=0, labelpad=8, va="center")
        axes[r, 0].yaxis.set_visible(True)
        axes[r, 0].set_yticks([])

    fig.suptitle(
        f"Stage 1 — reverse process at {small.image_size}x{small.image_size}, "
        f"{n_states - 1} denoising steps (row label = conditioning digit)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def figure_cascade_strip(small, big, labels, small_traj, bridge_unit, big_traj, path):
    """The whole chain end to end: low-res denoising, the upscale, then high-res refinement."""
    n_samples = small_traj.shape[0]
    n_small, n_big = small_traj.shape[1], big_traj.shape[1]
    n_cols = n_small + 1 + n_big

    widths = [1] * n_small + [1.35] + [1] * n_big
    fig, axes = plt.subplots(
        n_samples, n_cols, figsize=(n_cols * 0.40, n_samples * 0.52), gridspec_kw={"width_ratios": widths, **TIGHT}
    )

    for r in range(n_samples):
        for c in range(n_small):
            show(axes[r, c], small_traj[r, c], *NOISY_RANGE)

        bridge_ax = axes[r, n_small]
        show(bridge_ax, bridge_unit[r], 0.0, 1.0, interpolation="bilinear")
        for spine in bridge_ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#d1495b")
            spine.set_linewidth(1.4)

        for c in range(n_big):
            show(axes[r, n_small + 1 + c], big_traj[r, c], *NOISY_RANGE)

    # tight_layout cannot handle these fixed-aspect image axes, so place the headings by hand
    fig.subplots_adjust(left=0.01, right=0.99, top=0.78, bottom=0.02)

    def band_center(first, last):
        return 0.5 * (axes[0, first].get_position().x0 + axes[0, last].get_position().x1)

    label_y = 0.83
    fig.text(
        band_center(0, n_small - 1),
        label_y,
        f"stage 1: {small.image_size}x{small.image_size}, {n_small - 1} steps",
        ha="center",
        fontsize=9,
    )
    fig.text(
        band_center(n_small, n_small),
        label_y,
        "bilinear\nupscale",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#d1495b",
    )
    fig.text(
        band_center(n_small + 1, n_cols - 1),
        label_y,
        f"stage 2: {big.image_size}x{big.image_size}, {n_big - 1} steps",
        ha="center",
        fontsize=9,
    )

    fig.suptitle("Scale-space DDPM — one continuous reverse process across two resolutions", fontsize=12, y=0.99)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def figure_cascade_annotated(small, big, labels, small_traj, bridge_unit, big_traj, path):
    """A curated, larger-panel version of the chain for readers skimming the README."""
    n_small, n_big = small_traj.shape[1], big_traj.shape[1]
    small_idx = np.linspace(0, n_small - 1, 5).round().astype(int)
    big_idx = np.linspace(0, n_big - 1, 4).round().astype(int)
    n_cols = len(small_idx) + 1 + len(big_idx)
    n_samples = min(3, small_traj.shape[0])

    fig, axes = plt.subplots(
        n_samples, n_cols, figsize=(n_cols * 1.05, n_samples * 1.18), gridspec_kw={"wspace": 0.10, "hspace": 0.10}
    )
    for r in range(n_samples):
        for j, c in enumerate(small_idx):
            show(axes[r, j], small_traj[r, c], *NOISY_RANGE)
            if r == 0:
                axes[r, j].set_title(frame_label(c, n_small), fontsize=8)

        bridge_ax = axes[r, len(small_idx)]
        show(bridge_ax, bridge_unit[r], 0.0, 1.0, interpolation="bilinear")
        for spine in bridge_ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#d1495b")
            spine.set_linewidth(1.6)
        if r == 0:
            bridge_ax.set_title("upscaled", fontsize=8, color="#d1495b")

        for j, c in enumerate(big_idx):
            ax = axes[r, len(small_idx) + 1 + j]
            show(ax, big_traj[r, c], *NOISY_RANGE)
            if r == 0:
                ax.set_title(frame_label(c, n_big), fontsize=8)

    fig.text(
        0.5,
        0.955,
        "Coarse structure is decided at low resolution, detail is added at full resolution",
        ha="center",
        fontsize=12,
    )
    fig.text(0.24, 0.02, f"stage 1 — {small.image_size}x{small.image_size} DDPM", ha="center", fontsize=10)
    fig.text(0.76, 0.02, f"stage 2 — {big.image_size}x{big.image_size} upscaling DDPM", ha="center", fontsize=10)
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def figure_upscaler_vs_bilinear(small, big, labels, small_traj, bridge_unit, big_traj, path):
    """What the learned upscaler adds over just interpolating the low-res sample."""
    n = small_traj.shape[0]
    small_unit = utils.denormalize_images(small_traj[:, -1], small.image_size)
    big_unit = utils.denormalize_images(big_traj[:, -1], big.image_size)

    rows = [
        (small_unit, f"{small.image_size}x{small.image_size}\nstage-1 sample", "nearest"),
        (bridge_unit, "bilinear\nupscale only", "bilinear"),
        (big_unit, "cascaded\n(learned upscaler)", "nearest"),
    ]

    fig, axes = plt.subplots(
        len(rows), n, figsize=(n * 0.85, len(rows) * 1.0), gridspec_kw={"wspace": 0.08, "hspace": 0.08}
    )
    for r, (imgs, label, interp) in enumerate(rows):
        for c in range(n):
            show(axes[r, c], imgs[c], 0.0, 1.0, interpolation=interp)
        axes[r, 0].yaxis.set_visible(True)
        axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=34)
    for c in range(n):
        axes[0, c].set_title(str(int(labels[c])), fontsize=9)

    fig.suptitle("The second DDPM sharpens rather than just interpolating", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


@torch.no_grad()
def figure_x0_predictions(small, labels, small_traj, path, device):
    """Each noisy state beside the model's instantaneous guess at the clean image."""
    n_samples, n_states = small_traj.shape[0], small_traj.shape[1]
    n_samples = min(n_samples, 4)
    usable = n_states - 1  # the last frame is already x_0

    fig, axes = plt.subplots(2 * n_samples, usable, figsize=(usable * 0.42, 2 * n_samples * 0.50), gridspec_kw=TIGHT)
    for r in range(n_samples):
        x_t = small_traj[r, :usable].to(device)
        t = torch.tensor([max(n_states - 2 - c, 0) for c in range(usable)], device=device)
        lbl = labels[r].repeat(usable).to(device)
        x0_hat = small.insta_predict_from_t(x_t, t, lbl)

        for c in range(usable):
            show(axes[2 * r, c], x_t[c], *NOISY_RANGE)
            show(axes[2 * r + 1, c], x0_hat[c], *NOISY_RANGE)
            if r == 0 and c % 3 == 0:
                axes[0, c].set_title(f"t={int(t[c])}", fontsize=7, pad=3)

        for offset, label in ((0, "$x_t$"), (1, "$\\hat{x}_0$")):
            ax = axes[2 * r + offset, 0]
            ax.yaxis.set_visible(True)
            ax.set_yticks([])
            ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=12, va="center")

    fig.suptitle("Noisy state $x_t$ vs the model's instantaneous estimate $\\hat{x}_0$", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def figure_forward_process(small, big, path, device):
    """The forward process at both scales, which is what the two stages learn to invert."""
    n = 12
    fig, axes = plt.subplots(2, 1, figsize=(n * 0.62, 2.9))

    for ax, model, size, title in (
        (
            axes[0],
            small,
            small.image_size,
            f"stage 1 target: {small.image_size}x{small.image_size}, " f"{small.markov_states} states",
        ),
        (
            axes[1],
            big,
            big.image_size,
            f"stage 2 target: {big.image_size}x{big.image_size}, " f"{big.markov_states} states",
        ),
    ):
        images, labels = utils.load_mnist_split(True, size)
        x0 = utils.normalize_images(images[:n], size).to(device)
        states = model.markov_states
        idx = np.linspace(0, states - 1, n).round().astype(int)
        noise = torch.randn_like(x0)
        frames = [
            model.forward_diffusion(
                x0[i : i + 1], noise[i : i + 1], torch.tensor([idx[i]], device=device), keep_intermediate=False
            )
            for i in range(n)
        ]
        strip = torch.cat([f.squeeze(0).squeeze(0) for f in frames], dim=1)
        ax.imshow(strip.cpu().numpy(), cmap="gray", vmin=NOISY_RANGE[0], vmax=NOISY_RANGE[1])
        blank_axis(ax)
        ax.set_title(title + "   (t increases left to right)", fontsize=10)

    fig.suptitle("Forward processes the two stages are trained to reverse", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


@torch.no_grad()
def figure_class_grid(small, big, path, device, per_class=8):
    """Class-conditional grid of finished cascaded samples."""
    labels = torch.arange(10).repeat_interleave(per_class)
    _, _, big_traj = run_cascade(small, big, labels, device)
    final = utils.denormalize_images(big_traj[:, -1], big.image_size)

    fig, axes = plt.subplots(10, per_class, figsize=(per_class * 0.62, 10 * 0.64), gridspec_kw=TIGHT)
    for r in range(10):
        for c in range(per_class):
            show(axes[r, c], final[r * per_class + c], 0.0, 1.0)
        axes[r, 0].yaxis.set_visible(True)
        axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(str(r), fontsize=9, rotation=0, labelpad=8, va="center")

    fig.suptitle("Class-conditional samples from the cascaded model", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path)


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    device = utils.pick_device()
    print("Rendering on", device)
    small, big = load_models(args.small_name, args.big_name, device)

    torch.manual_seed(args.seed)
    labels = torch.arange(10)
    small_traj, bridge_unit, big_traj = run_cascade(small, big, labels, device)

    figure_cascade_annotated(small, big, labels, small_traj, bridge_unit, big_traj, f"{OUT_DIR}/cascade_annotated.png")
    figure_cascade_strip(
        small, big, labels, small_traj[:6], bridge_unit[:6], big_traj[:6], f"{OUT_DIR}/cascade_full_strip.png"
    )
    figure_lowres_reverse(small, labels[:6], small_traj[:6], f"{OUT_DIR}/lowres_reverse_process.png")
    figure_upscaler_vs_bilinear(
        small, big, labels, small_traj, bridge_unit, big_traj, f"{OUT_DIR}/upscaler_vs_bilinear.png"
    )
    figure_x0_predictions(small, labels, small_traj, f"{OUT_DIR}/x0_predictions.png", device)
    figure_forward_process(small, big, f"{OUT_DIR}/forward_processes.png", device)
    figure_class_grid(small, big, f"{OUT_DIR}/cascaded_class_grid.png", device)


if __name__ == "__main__":
    main()
