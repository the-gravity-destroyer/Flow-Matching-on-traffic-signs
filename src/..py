import torch

stats = torch.load("class_stats/gtsrb.pt")
print("Mittlere Std gesamt:", stats["stds"].mean().item())
print("\nMittlere Std pro Klasse:")
for c, std in enumerate(stats["stds"].mean(dim=[1,2,3])):
    print(f"  Klasse {c:3d}: {std:.4f}")