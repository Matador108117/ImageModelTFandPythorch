import os
import json
import torch
import timm
import numpy as np

from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================================================
# CONFIG
# =========================================================

TRAIN_DIR = "dataset/train"

MODEL_PATH = "plant_classifier.pth"

OUTPUT_PATH = "known_embeddings.npy"

CLASSES_PATH = "classes.json"

MODEL_NAME = "convnext_tiny"

IMG_SIZE = 224

BATCH_SIZE = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# LOAD CLASSES
# =========================================================

with open(CLASSES_PATH, "r") as f:
    CLASSES = json.load(f)

print("\nCLASES DETECTADAS:\n")

for c in CLASSES:
    print("-", c)

# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(IMG_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# DATASET
# =========================================================

dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# =========================================================
# FEATURE EXTRACTOR
# =========================================================

print("\nCARGANDO MODELO...\n")

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=0
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    state_dict,
    strict=False
)

model.to(DEVICE)

model.eval()

print("MODELO CARGADO\n")

# =========================================================
# EMBEDDINGS STORAGE
# =========================================================

embeddings_by_class = {
    class_name: []
    for class_name in CLASSES
}

# =========================================================
# GENERATE EMBEDDINGS
# =========================================================

print("GENERANDO EMBEDDINGS...\n")

with torch.no_grad():

    loop = tqdm(loader)

    for images, labels in loop:

        images = images.to(DEVICE)

        embeddings = model(images)

        embeddings = embeddings.cpu().numpy()

        labels = labels.numpy()

        for emb, label_idx in zip(
            embeddings,
            labels
        ):

            class_name = CLASSES[label_idx]

            embeddings_by_class[
                class_name
            ].append(emb)

# =========================================================
# AVERAGE EMBEDDINGS
# =========================================================

print("\nCALCULANDO CENTROIDES...\n")

known_embeddings = []

for class_name in CLASSES:

    class_embeddings = np.array(
        embeddings_by_class[class_name]
    )

    print(
        f"{class_name}: "
        f"{len(class_embeddings)} embeddings"
    )

    centroid = np.mean(
        class_embeddings,
        axis=0
    )

    # =====================================================
    # NORMALIZATION
    # =====================================================

    norm = np.linalg.norm(
        centroid
    )

    if norm > 0:
        centroid = centroid / norm

    known_embeddings.append(
        centroid
    )

known_embeddings = np.array(
    known_embeddings,
    dtype=np.float32
)

# =========================================================
# SAVE
# =========================================================

np.save(
    OUTPUT_PATH,
    known_embeddings
)

print("\n===================================")
print("EMBEDDINGS GENERADOS")
print("===================================\n")

print(
    f"Shape: "
    f"{known_embeddings.shape}"
)

print(
    f"Archivo guardado en:\n"
    f"{OUTPUT_PATH}"
)