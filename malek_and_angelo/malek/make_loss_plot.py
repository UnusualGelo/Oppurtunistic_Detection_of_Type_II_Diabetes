import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load loss history from 3-way split (569 Train / 190 Val / 190 Test)
csv_path = Path("/home/e92/School/CS396/Oppurtunistic_Detection_of_Type_II_Diabetes/malek_and_angelo/malek/cxr_vision_loss_history.csv")
df = pd.read_csv(csv_path)

base = df[df["model"] == "Baseline_Scratch_CNN"]
enh = df[df["model"] == "Enhanced_Residual_CXRNet"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="#1e293b")

# Plot 1: Training vs. Validation Loss Curves
ax1.set_facecolor("#0f172a")
# Baseline lines
ax1.plot(base["epoch"], base["train_loss"], linestyle="--", marker="o", markersize=4, color="#7dd3fc", label="Baseline: Train Loss", alpha=0.85, linewidth=1.8)
ax1.plot(base["epoch"], base["val_loss"], linestyle="-", marker="o", markersize=5, color="#38bdf8", label="Baseline: Val Loss", linewidth=2.4)
# Enhanced lines
ax1.plot(enh["epoch"], enh["train_loss"], linestyle="--", marker="^", markersize=5, color="#6ee7b7", label="Enhanced: Train Loss", alpha=0.85, linewidth=1.8)
ax1.plot(enh["epoch"], enh["val_loss"], linestyle="-", marker="s", markersize=5, color="#10b981", label="Enhanced: Val Loss", linewidth=2.4)

# Sweet spot annotation at Epoch 6 for Enhanced
ax1.annotate("Sweet Spot\n(Val Loss: 0.5898)", xy=(6, 0.5898), xytext=(3.5, 0.42),
             arrowprops=dict(arrowstyle="->", color="#f59e0b", lw=1.5),
             color="#f59e0b", fontsize=8.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#f59e0b", alpha=0.9))

ax1.set_title("3-Way Split: Training vs. Validation Loss", color="white", fontsize=12, fontweight="bold", pad=12)
ax1.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
ax1.set_ylabel("Cross-Entropy Loss (569 Train / 190 Val)", color="#94a3b8", fontsize=10)
ax1.set_ylim(0.12, 0.75)
ax1.tick_params(colors="white")
ax1.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")
ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=8.5, loc="lower left")

# Plot 2: Validation AUROC Curve
ax2.set_facecolor("#0f172a")
ax2.plot(base["epoch"], base["val_auroc"], marker="o", markersize=5, color="#38bdf8", label="Baseline Scratch CNN (Peak: 0.6695)", linewidth=2.4)
ax2.plot(enh["epoch"], enh["val_auroc"], marker="s", markersize=5, color="#10b981", label="Enhanced Residual CXRNet (Peak: 0.6914)", linewidth=2.4)

ax2.annotate("Peak Val AUROC: 0.6914", xy=(6, 0.6914), xytext=(4.2, 0.62),
             arrowprops=dict(arrowstyle="->", color="#34d399", lw=1.5),
             color="#34d399", fontsize=8.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e293b", edgecolor="#34d399", alpha=0.9))

ax2.set_title("3-Way Split: Validation AUROC Tracking", color="white", fontsize=12, fontweight="bold", pad=12)
ax2.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
ax2.set_ylabel("Validation AUROC", color="#94a3b8", fontsize=10)
ax2.set_ylim(0.55, 0.72)
ax2.tick_params(colors="white")
ax2.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")
ax2.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=8.5, loc="lower right")

plt.tight_layout()
out_png = Path("/home/e92/School/CS396/Oppurtunistic_Detection_of_Type_II_Diabetes/malek_and_angelo/malek/cxr_loss_graph.png")
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Chart saved successfully at {out_png}")
