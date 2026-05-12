import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_metrics(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    metrics = {}
    for row in rows:
        for key, value in row.items():
            if value is None or value == "":
                continue
            try:
                metrics.setdefault(key, []).append(float(value))
            except ValueError:
                pass
    return metrics, len(rows)


def plot_group(ax, metrics, x_key, y_keys, title):
    x = metrics.get(x_key)
    if not x:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No epoch rows yet", ha="center", va="center")
        return

    for key in y_keys:
        y = metrics.get(key)
        if y and len(y) == len(x):
            ax.plot(x, y, marker="o", linewidth=1.5, label=key)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def main(args):
    csv_path = Path(args.csv_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {csv_path}")

    metrics, row_count = read_metrics(csv_path)
    if row_count == 0:
        print(f"No epoch rows yet in {csv_path}. Try again after epoch 1 finishes.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    plot_group(
        axes[0, 0],
        metrics,
        "Epoch",
        ["Train_Total_Loss", "Train_Recon_MSE", "Val_Recon_MSE"],
        "Reconstruction / Total Loss",
    )
    plot_group(
        axes[0, 1],
        metrics,
        "Epoch",
        ["Train_Active_SmoothL1", "Val_Active_SmoothL1", "Train_Contact_Focal", "Val_Contact_Focal"],
        "Active Contact Losses",
    )
    plot_group(
        axes[1, 0],
        metrics,
        "Epoch",
        ["Val_Contact_Micro_F1", "Val_Contact_Macro_F1", "Val_Contact_Micro_Precision", "Val_Contact_Micro_Recall"],
        "Validation Contact Quality",
    )
    plot_group(
        axes[1, 1],
        metrics,
        "Epoch",
        ["Gate_Exp_1", "Gate_Exp_2", "Gate_Exp_3"],
        "MoE Gate Usage",
    )
    plot_group(
        axes[2, 0],
        metrics,
        "Epoch",
        ["Train_KLD_Loss", "Train_Raw_KLD_Loss", "Beta"],
        "KLD / Beta",
    )
    plot_group(
        axes[2, 1],
        metrics,
        "Epoch",
        ["Grad_Norm", "Train_Gate_Balance"],
        "Grad / Gate Balance",
    )

    fig.suptitle(csv_path.parent.name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=args.dpi)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot IIW training metrics CSV")
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./proposed/TGCN_cVAE_MoE/weights_ver2/training_metrics.csv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./proposed/TGCN_cVAE_MoE/weights_ver2/training_metrics.png",
    )
    parser.add_argument("--dpi", type=int, default=160)
    main(parser.parse_args())
