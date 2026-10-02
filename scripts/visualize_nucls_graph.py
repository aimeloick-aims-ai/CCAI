from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

GRAPH_DIR = Path("processed/nucls/graphs")
RGB_DIR = Path("data/nucls/NuCLS_dataset/rgb")

# Prendre le premier graphe
graph_file = next(GRAPH_DIR.glob("*.npz"))

graph = np.load(graph_file, allow_pickle=True)

image_id = graph["image_id"].item()
pos = graph["pos"]
edge_index = graph["edge_index"]

print("Graph:", graph_file.name)
print("Image ID:", image_id)
print("Nodes:", pos.shape[0])
print("Edges:", edge_index.shape[1])

# Chercher automatiquement l'image correspondante
possible_images = list(RGB_DIR.glob(f"{image_id}.*"))

if not possible_images:
    raise FileNotFoundError(
        f"Image not found for {image_id}"
    )

image_file = possible_images[0]

print("Image:", image_file)

image = Image.open(image_file)

# -------------------------
# Visualisation
# -------------------------

plt.figure(figsize=(10, 10))

plt.imshow(image)

# Dessiner les arêtes
for source, target in edge_index.T:
    x1, y1 = pos[source]
    x2, y2 = pos[target]

    plt.plot(
        [x1, x2],
        [y1, y2],
        linewidth=0.5,
        alpha=0.35
    )

# Dessiner les cellules
plt.scatter(
    pos[:, 0],
    pos[:, 1],
    s=18
)

plt.title(
    f"NuCLS graph overlay\n"
    f"{pos.shape[0]} cells | "
    f"{edge_index.shape[1]} directed edges"
)

plt.axis("off")
plt.tight_layout()
plt.show()