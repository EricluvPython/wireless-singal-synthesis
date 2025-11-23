import pandas as pd
import matplotlib.pyplot as plt
import os

# Read data from CSV file
csv_path = "./work_dir/dit_data_fidelity_results.csv"
df = pd.read_csv(csv_path)

# Output directory
output_dir = "./work_dir"
os.makedirs(output_dir, exist_ok=True)

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

methods_style = {
    "Real_Only":         {"label": "Real Only",       "marker": "s", "linestyle": "--", "color": "#4C78A8", "markersize": 5},
    "DiT_Augmented":     {"label": "DiT Aug.",   "marker": "o", "linestyle": "-",  "color": "#F58518", "markersize": 5},
    "GAN_Augmented":     {"label": "GAN Aug,",   "marker": "^", "linestyle": "-.",  "color": "#54A24B", "markersize": 5},
    "Kriging_Augmented": {"label": "Kriging Aug.",    "marker": "D", "linestyle": ":",  "color": "#E45756", "markersize": 5},
}

def make_dataset_plot(dataset_name, filename):
    sub = df[df["dataset"] == dataset_name].copy()
    
    # Filter out 100% real data ratio (no point showing it)
    sub = sub[sub["train_ratio"] < 1.0]
    
    ratios = sorted(sub["train_ratio"].unique())

    fig, ax = plt.subplots(figsize=(4.5, 1.6))  # Half height for stacking

    for method, style in methods_style.items():
        mdata = sub[sub["method"] == method]
        if mdata.empty:
            continue
        ax.plot(
            mdata["train_ratio"] * 100,
            mdata["accuracy"] * 100,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=2.0,
            markersize=style["markersize"],
            label=style["label"],
            alpha=0.9,
        )

    # Oracle line (still show for reference, but filtered to same x-range)
    oracle = df[(df["dataset"] == dataset_name) & (df["method"] == "Oracle_100%")]
    if not oracle.empty:
        ax.axhline(
            y=oracle["accuracy"].iloc[0] * 100,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            label="Oracle",
            alpha=0.7,
        )

    ax.set_xlabel("Real Data Used (%)", fontsize=10)
    ax.set_ylabel("Test Accuracy (%)", fontsize=10)
    ax.set_xticks([r * 100 for r in ratios])
    
    # Auto-scale y-axis to show differences better
    # Get min/max accuracy for this dataset (excluding oracle)
    non_oracle = sub[sub["method"] != "Oracle_100%"]
    if not non_oracle.empty:
        min_acc = non_oracle["accuracy"].min() * 100
        max_acc = non_oracle["accuracy"].max() * 100
        # Add 5% padding
        padding = (max_acc - min_acc) * 0.1
        y_min = max(0, min_acc - padding)
        y_max = min(100, max_acc + padding)
        ax.set_ylim(y_min, y_max)
    else:
        ax.set_ylim(0, 105)
    
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_title(dataset_name.replace("_", " "), pad=2, fontsize=9, fontweight='bold')
    ax.legend(frameon=True, ncol=1, loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {filename}")
    plt.close(fig)

def make_stacked_plot(filename):
    """Create stacked plot with both datasets"""
    # Get both datasets
    datasets = df["dataset"].unique()
    
    # Filter out 100% real data ratio
    plot_df = df[df["train_ratio"] < 0.8].copy()
    
    fig, axes = plt.subplots(2, 1, figsize=(4.5, 3.5), sharex=False)
    
    for idx, dataset_name in enumerate(sorted(datasets)):
        ax = axes[idx]
        sub = plot_df[plot_df["dataset"] == dataset_name].copy()
        ratios = sorted(sub["train_ratio"].unique())
        
        for method, style in methods_style.items():
            mdata = sub[sub["method"] == method]
            if mdata.empty:
                continue
            ax.plot(
                mdata["train_ratio"] * 100,
                mdata["accuracy"] * 100,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2.0,
                markersize=style["markersize"],
                label=style["label"],
                alpha=0.9,
            )
        
        # Oracle line
        oracle = df[(df["dataset"] == dataset_name) & (df["method"] == "Oracle_100%")]
        if not oracle.empty:
            ax.axhline(
                y=oracle["accuracy"].iloc[0] * 100,
                color="gray",
                linestyle=":",
                linewidth=1.5,
                label="Oracle",
                alpha=0.7,
            )
        
        # Set labels and formatting
        if idx == 1:  # Bottom plot
            ax.set_xlabel("Real Data Used (%)", fontsize=10)
        ax.set_ylabel("Test Accuracy (%)", fontsize=10)
        ax.set_xticks([r * 100 for r in ratios])
        
        # Auto-scale y-axis
        non_oracle = sub[sub["method"] != "Oracle_100%"]
        if not non_oracle.empty:
            min_acc = non_oracle["accuracy"].min() * 100
            max_acc = non_oracle["accuracy"].max() * 100
            padding = (max_acc - min_acc) * 0.1
            y_min = max(0, min_acc - padding)
            y_max = min(100, max_acc + padding)
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(0, 105)
        
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_title(dataset_name.replace("_", " "), pad=2, fontsize=9, fontweight='bold')
        
        ax.legend(frameon=True, ncol=2, loc="lower right", fontsize=7)
    
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved stacked plot: {filename}")
    plt.close(fig)

# Create individual plots
make_dataset_plot("UCI_Indoor", os.path.join(output_dir, "uci_indoor_results.png"))
make_dataset_plot("POWDER_Outdoor", os.path.join(output_dir, "powder_outdoor_results.png"))

# Create stacked plot
make_stacked_plot(os.path.join(output_dir, "combined_results.png"))
