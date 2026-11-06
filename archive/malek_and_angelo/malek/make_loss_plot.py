import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load comparison data
base_dir = Path("/home/e92/School/CS396/Oppurtunistic_Detection_of_Type_II_Diabetes/malek_and_angelo/malek")
df = pd.read_csv(base_dir / "anti_memorization_comparison.csv")

raw = df[df["model"] == "Enhanced_Residual_CXRNet"]
anti = df[df["model"] == "Anti_Memorization_CXRNet"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="#1e293b")

# ---------------------------------------------------------------------------
# Plot 1: The Memorization Gap vs. Anti-Memorization Solution
# ---------------------------------------------------------------------------
ax1.set_facecolor("#0f172a")

# Standard Run (Without Augmentation - Memorization Trap)
ax1.plot(raw["epoch"], raw["train_loss"], linestyle=":", marker="x", markersize=5, color="#f43f5e", label="Standard Train Loss (Overfitting)", alpha=0.85, linewidth=1.8)
ax1.plot(raw["epoch"], raw["val_loss"], linestyle="-", marker="x", markersize=5, color="#fb7185", label="Standard Val Loss", alpha=0.85, linewidth=2.0)

# Anti-Memorization Run (With Augmentation & Regularization)
ax1.plot(anti["epoch"], anti["train_loss"], linestyle="--", marker="o", markersize=5, color="#6ee7b7", label="Anti-Memorize Train Loss", linewidth=2.2)
ax1.plot(anti["epoch"], anti["val_loss"], linestyle="-", marker="s", markersize=5, color="#10b981", label="Anti-Memorize Val Loss", linewidth=2.5)

# Annotations
ax1.annotate("Memorization Gap (0.50)\nTrain drops to 0.17!", xy=(12, 0.1735), xytext=(7.0, 0.30),
             arrowprops=dict(arrowstyle="->", color="#f43f5e", lw=1.5),
             color="#f43f5e", fontsize=8.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e293b", edgecolor="#f43f5e", alpha=0.9))

ax1.annotate("Zero Gap (~0.03)!\nReal Learning", xy=(11.5, 0.65), xytext=(5.5, 0.53),
             arrowprops=dict(arrowstyle="->", color="#34d399", lw=1.5),
             color="#34d399", fontsize=8.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e293b", edgecolor="#34d399", alpha=0.9))

ax1.set_title("Loss Curves: Memorization Trap vs. Anti-Memorization", color="white", fontsize=11.5, fontweight="bold", pad=12)
ax1.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
ax1.set_ylabel("Cross-Entropy Loss", color="#94a3b8", fontsize=10)
ax1.set_ylim(0.12, 0.85)
ax1.tick_params(colors="white")
ax1.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")
ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=8, loc="upper right")

# ---------------------------------------------------------------------------
# Plot 2: Anti-Memorization AUROC & Diabetic Catch Rate
# ---------------------------------------------------------------------------
ax2.set_facecolor("#0f172a")

ax2.plot(anti["epoch"], anti["val_auroc"], marker="s", markersize=5, color="#38bdf8", label="Validation AUROC (Climbs to 0.644)", linewidth=2.4)
ax2.plot(anti["epoch"], anti["val_recall"] / 100.0, marker="^", markersize=5, color="#f59e0b", label="Diabetic Catch Rate (Recall)", linewidth=2.2, linestyle="--")

ax2.annotate("Peak Discrimination: 0.644 AUROC\n& 56.5% Catch Rate", xy=(11, 0.6444), xytext=(3.5, 0.35),
             arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.5),
             color="#38bdf8", fontsize=8.5, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e293b", edgecolor="#38bdf8", alpha=0.9))

ax2.set_title("Anti-Memorization: AUROC & Diabetic Catch Rate", color="white", fontsize=11.5, fontweight="bold", pad=12)
ax2.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
ax2.set_ylabel("Validation Score (0.0 to 1.0)", color="#94a3b8", fontsize=10)
ax2.set_ylim(0.0, 1.0)
ax2.tick_params(colors="white")
ax2.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")
ax2.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="white", fontsize=8, loc="lower right")

plt.tight_layout()
out_png = base_dir / "cxr_loss_graph.png"
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Chart saved successfully at {out_png}")
