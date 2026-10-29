import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load loss history
csv_path = Path("/home/e92/School/CS396/Oppurtunistic_Detection_of_Type_II_Diabetes/malek_and_angelo/malek/cxr_vision_loss_history.csv")
df = pd.read_csv(csv_path)

# Average across folds for each epoch
avg = df.groupby(["model", "epoch"])[["train_loss", "val_loss", "val_auroc"]].mean().reset_index()

base = avg[avg["model"] == "Baseline_Scratch_CNN"]
enh = avg[avg["model"] == "Enhanced_Residual_CXRNet"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor="#1e293b")

# Plot 1: Validation Loss Curve
ax1.set_facecolor("#0f172a")
ax1.plot(base["epoch"], base["val_loss"], marker='o', color="#38bdf8", label="Baseline Scratch CNN", linewidth=2.5)
ax1.plot(enh["epoch"], enh["val_loss"], marker='s', color="#34d399", label="Enhanced Residual CXRNet", linewidth=2.5)
ax1.set_title("Validation Loss Tracking (Lower is Better)", color="white", fontsize=12, fontweight="bold", pad=12)
ax1.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
ax1.set_ylabel("Validation Loss (Cross-Entropy)", color="#94a3b8", fontsize=10)
ax1.tick_params(colors="white")
ax1.grid(True, linestyle="--", alpha=0.3, color="#94a3b8")
ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)

# Plot 2: Validation AUROC Curve
ax2.set_facecolor("#0f172a")
ax2.plot(base["epoch"], base["val_auroc"], marker='o', color="#38bdf8", label="Baseline Scratch CNN", linewidth=2.5)
ax2.plot(enh["epoch"], enh["val_auroc"], marker='s', color="#34d399", label="Enhanced Residual CXRNet", linewidth=2.5)
ax2.set_title("Validation AUROC (Higher is Better)", color="white", fontsize=12, fontweight="bold", pad=12)
ax2.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
ax2.set_ylabel("AUROC", color="#94a3b8", fontsize=10)
ax2.tick_params(colors="white")
ax2.grid(True, linestyle="--", alpha=0.3, color="#94a3b8")
ax2.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=9)

plt.tight_layout()
out_png = Path("/home/e92/School/CS396/Oppurtunistic_Detection_of_Type_II_Diabetes/malek_and_angelo/malek/cxr_loss_graph.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Chart saved successfully at {out_png}")
