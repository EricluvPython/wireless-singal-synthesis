import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np

def load_mnist_example(target_digit=3, data_root="./data"):
    """
    Load a single MNIST example of the specified digit.
    Returns:
        img: tensor of shape [28, 28], values in [0, 1]
        label: int
    """
    transform = T.Compose([
        T.ToTensor(),         # [1, 28, 28], in [0,1]
    ])
    mnist = torchvision.datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    for img, label in mnist:
        if label == target_digit:
            # squeeze channel dimension -> [28, 28]
            return img.squeeze(0), int(label)

    raise RuntimeError(f"No example of digit {target_digit} found (this should not happen).")


def compute_vertical_profile(img):
    """
    Vertical attenuation analogue: mean pixel value per row.
    img: [28, 28]
    returns: row_means of shape [28]
    """
    # mean over columns (dim=1)
    row_means = img.mean(dim=1)
    return row_means


def compute_column_occupancy(img, threshold=0.5):
    """
    Column occupancy analogue: fraction of bright pixels per column.
    img: [28, 28]
    returns: col_bright_frac of shape [28]
    """
    bright = (img > threshold).float()
    # fraction of bright pixels per column -> mean over rows
    col_bright_frac = bright.mean(dim=0)
    return col_bright_frac


def compute_gradients_and_tau(dataset, num_samples=1000, threshold_quantile=0.95):
    """
    Compute spatial gradients |Δ| over a subset of MNIST and estimate tau
    as the given quantile.

    Returns:
        tau: float (95th percentile)
    """
    all_diffs = []

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
    )

    seen = 0
    for imgs, _ in loader:
        # imgs: [B, 1, 28, 28]
        imgs = imgs.squeeze(1)  # [B, 28, 28]
        # horizontal diffs: x[:, :, 1:] - x[:, :, :-1]
        dh = imgs[:, :, 1:] - imgs[:, :, :-1]
        # vertical diffs: x[:, 1:, :] - x[:, :-1, :]
        dv = imgs[:, 1:, :] - imgs[:, :-1, :]

        abs_dh = dh.abs().reshape(-1)
        abs_dv = dv.abs().reshape(-1)

        all_diffs.append(abs_dh)
        all_diffs.append(abs_dv)

        seen += imgs.size(0)
        if seen >= num_samples:
            break

    all_diffs = torch.cat(all_diffs, dim=0)
    tau = torch.quantile(all_diffs, threshold_quantile).item()
    return tau


def compute_image_gradients(img):
    """
    Compute horizontal and vertical differences for a single image.
    img: [28, 28]
    returns:
        grad_mag: [28, 28] (with zero-padding on edges)
    """
    # horizontal diffs
    dh = torch.zeros_like(img)
    dh[:, 1:] = img[:, 1:] - img[:, :-1]

    # vertical diffs
    dv = torch.zeros_like(img)
    dv[1:, :] = img[1:, :] - img[:-1, :]

    grad_mag = torch.sqrt(dh ** 2 + dv ** 2)
    return grad_mag


def compute_center_row_transitions(img, threshold=0.5, row_index=14):
    """
    Handover analogue: transitions in binarized center row.
    img: [28, 28]
    returns:
        bin_row: [28] tensor of 0/1
        transitions: int
    """
    row = img[row_index, :]           # [28]
    bin_row = (row > threshold).int() # [28]
    # count changes between adjacent pixels
    transitions = (bin_row[1:] != bin_row[:-1]).sum().item()
    return bin_row, transitions


def main():
    # 1. Load one example image
    target_digit = 3
    img, label = load_mnist_example(target_digit=target_digit)
    print(f"Loaded example of digit {label}")

    # 2. Compute priors on MNIST train set for tau (smoothness)
    transform = T.Compose([T.ToTensor()])
    mnist_train = torchvision.datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    tau = compute_gradients_and_tau(mnist_train, num_samples=2000, threshold_quantile=0.95)
    print(f"Estimated tau (95th percentile of |Δ|) ≈ {tau:.4f}")

    # 3. Compute per-image quantities
    row_means = compute_vertical_profile(img)            # vertical profile
    col_bright = compute_column_occupancy(img, 0.5)      # column occupancy
    grad_mag = compute_image_gradients(img)              # gradient magnitude
    bin_row, transitions = compute_center_row_transitions(
        img, threshold=0.5, row_index=14
    )

    # 4. Plot everything
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"MNIST digit {label}: Visualizing 'physics' constraints", fontsize=16)

    # (0,0) Original image
    ax = axes[0, 0]
    ax.imshow(img.numpy(), cmap="gray")
    ax.set_title("Original image (x)")
    ax.axis("off")
    ax.text(
        0.5, -0.1,
        "This is the 'sequence × towers' grid.\n"
        "All other plots summarize this image\n"
        "for the different loss terms.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    # (0,1) Vertical attenuation analogue
    ax = axes[0, 1]
    rows = np.arange(28)
    ax.plot(row_means.numpy(), rows)
    ax.invert_yaxis()
    ax.set_xlabel("Mean intensity per row")
    ax.set_ylabel("Row index")
    ax.set_title("Vertical profile (L_vert)")
    ax.text(
        0.5, -0.2,
        "L_vert: keeps this curve close to the\n"
        "digit-specific average vertical profile.\n"
        "Analogue of vertical attenuation over floors.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    # (0,2) Column occupancy analogue
    ax = axes[0, 2]
    cols = np.arange(28)
    ax.bar(cols, col_bright.numpy())
    ax.set_xlabel("Column index")
    ax.set_ylabel("Bright fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Column 'occupancy' (L_occ)")
    ax.text(
        0.5, -0.25,
        "L_occ: matches where bright pixels tend\n"
        "to occur horizontally (per digit).\n"
        "Analogue of tower occupancy distribution.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    # (1,0) Gradient magnitude (smoothness)
    ax = axes[1, 0]
    im = ax.imshow(grad_mag.numpy(), cmap="magma")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Gradient magnitude (L_smooth)")
    ax.axis("off")
    ax.text(
        0.5, -0.2,
        f"L_smooth: penalizes |Δ| > τ (τ≈{tau:.3f}).\n"
        "Discourages unrealistically sharp jumps\n"
        "compared to real MNIST images.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    # (1,1) Center row binarization + transitions
    ax = axes[1, 1]
    x_axis = np.arange(28)
    row_values = img[14, :].numpy()
    ax.plot(x_axis, row_values, label="Row 14 intensity")
    threshold = 0.5
    ax.axhline(threshold, color="red", linestyle="--", label="threshold")
    ax.step(x_axis, bin_row.numpy(), where="mid", label="Binarized (0/1)", alpha=0.7)
    ax.set_xlabel("Column index (center row)")
    ax.set_ylabel("Value")
    ax.set_title(f"Center row transitions (L_handover)\nN = {transitions}")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(
        0.5, -0.3,
        "L_handover: keeps number of 0↔1 flips along\n"
        "this scanline close to digit-specific stats.\n"
        "Analogue of CID handover transitions.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    # (1,2) Text-only panel summarizing all constraints
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = (
        "How the four losses act on an image:\n\n"
        "• L_vert: enforces a typical vertical brightness profile.\n"
        "• L_occ: enforces typical horizontal occupancy of bright pixels.\n"
        "• L_smooth: discourages overly sharp spatial edges.\n"
        "• L_handover: enforces a typical number of foreground/background\n"
        "  transitions along a 1D scanline.\n\n"
        "Together, they push generated images toward\n"
        "the global statistics of real MNIST digits,\n"
        "just like physics priors do for RSS sequences."
    )
    ax.text(0.0, 0.5, summary_text, fontsize=9, va="center")

    plt.tight_layout()
    plt.savefig("mnist_physics_losses_visualization.png", dpi=300)


if __name__ == "__main__":
    main()
