import re
import matplotlib.pyplot as plt

# Path to your log file
log_file = "runs/ratioNorm/train.log"

epochs = []
betas = []

# Regex pattern
pattern = re.compile(r"Beta for the epoch\s+(\d+)\s+is\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)")

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            betas.append(float(match.group(2)))

print(f"Found {len(epochs)} beta values.")
print(f"Epoch range: {min(epochs)} - {max(epochs)}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(epochs, betas, linewidth=1.5)
plt.xlabel("Epoch")
plt.ylabel("Beta")
plt.title("Ratio Norm(0.05) based Adaptive Beta vs Epoch")
plt.grid(True, alpha=0.3)
plt.xlim(1, 20000)
plt.tight_layout()

plt.savefig("beta_vs_epoch.png", dpi=900)
plt.show()


