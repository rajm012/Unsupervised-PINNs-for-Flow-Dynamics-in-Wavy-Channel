# import re
# import matplotlib.pyplot as plt

# # Path to your log file
# log_file = "runs/relobralo/train.log"

# epochs = []
# betas = []

# # Regex pattern
# pattern = re.compile(r"Beta for the epoch\s+(\d+)\s+is\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")

# with open(log_file, "r") as f:
#     for line in f:
#         match = pattern.search(line)
#         if match:
#             epochs.append(int(match.group(1)))
#             betas.append(float(match.group(2)))

# print(f"Found {len(epochs)} beta values.")
# print(f"Epoch range: {min(epochs)} - {max(epochs)}")

# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(epochs, betas, linewidth=1.5)
# plt.xlabel("Epoch")
# plt.ylabel("Beta")
# plt.title("Relobralo(1, 0.2) based Adaptive Beta vs Epoch")
# plt.grid(True, alpha=0.3)
# plt.xlim(1, 20000)
# plt.tight_layout()

# plt.savefig("beta_vs_epoch.png", dpi=900)
# plt.show()


# ======================================================================

import pandas as pd
import matplotlib.pyplot as plt

file_path = "runs/periodic_b1_R10_p300/loss_history.csv"
df = pd.read_csv(file_path)
epoch = df["epoch"]

governing_losses = ["Lg1_continuity", "Lg2_xmom", "Lg3_ymom", "Lg4_energy", "Lg5_constitutive"]
boundary_losses = ["LBC_u", "LBC_v", "LBC_p"]
all_losses = governing_losses + boundary_losses

plt.figure(figsize=(15, 8))
for col in ["Lp", "Lg", "LBC"]:
    plt.plot(epoch, df[col], label=col, linewidth=2)

plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Overall PINN Loss")
plt.legend()
plt.grid(True, which="both", alpha=0.1)
plt.tight_layout()
plt.savefig('Overall_loss.png', dpi=900)
# plt.show()


plt.figure(figsize=(15, 8))
for col in governing_losses:
    plt.plot(epoch, df[col], label=col, linewidth=1.5)

plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Governing Equation Losses")
plt.legend()
plt.grid(True, which="both", alpha=0.1)
plt.tight_layout()
plt.savefig('Governing_Losses.png', dpi=900)
# plt.show()


plt.figure(figsize=(15, 8))
for col in boundary_losses:
    plt.plot(epoch, df[col], label=col, linewidth=1.5)


plt.yscale("log")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Boundary Condition Losses")
plt.legend()
plt.grid(True, which="both", alpha=0.1)
plt.tight_layout()
plt.savefig('Boundary_Losses.png', dpi=900)
# plt.show()


plt.figure(figsize=(15, 8))
for col in all_losses:
    normalized = df[col] / df[col].iloc[0]
    plt.plot(epoch, normalized, label=col, linewidth=1.5)

plt.yscale("log")
plt.axhline(1, color="black", linestyle="--", linewidth=0.8)
plt.xlabel("Epoch")
plt.ylabel(r"$L_i/L_i(0)$")
plt.title("Normalized Loss Reduction")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(True, which="both", alpha=0.1)
plt.tight_layout()
plt.savefig('Normalized_Loss.png', dpi=900)
# plt.show()


plt.figure(figsize=(15, 8))
plt.plot(epoch, df["lr"], color="black", linewidth=1.5)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.grid(True, which="both", alpha=0.1)
plt.tight_layout()
plt.savefig('Learning_Rate.png', dpi=900)
# plt.show()

beta = 2
beta_LBC = df["LBC"]/beta

plt.figure(figsize=(15, 8))
plt.plot(epoch, df["Lg"], label=r"$L_g$", linewidth=2)
plt.plot(epoch, beta_LBC, label=r"$L_{BC}/ \beta $", linewidth=2)
plt.plot(epoch, df["Lp"], label=r"$L_p$", linewidth=2, linestyle="--")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Composition")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig('Loss_Composition.png', dpi=900)
# plt.show()

total = df["Lg"] + beta_LBC
governing_fraction = df["Lg"] / total
boundary_fraction = beta_LBC / total


plt.figure(figsize=(15, 8))
plt.plot(epoch, governing_fraction, label="Governing", linewidth=2)
plt.plot(epoch, boundary_fraction, label="Boundary", linewidth=2)
plt.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
plt.ylabel("Contribution fraction")
plt.title("Relative Contribution to Total Loss")
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Fraction_Contri.png', dpi=900)
# plt.show()

