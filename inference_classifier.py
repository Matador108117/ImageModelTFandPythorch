import json
import cv2
import timm
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224

UNKNOWN_THRESHOLD = 0.55

MODEL_NAME = "convnext_tiny"

# =========================================================
# CLASSES
# =========================================================

with open("classes.json", "r") as f:
    CLASSES = json.load(f)

print("\nCLASES:")
for c in CLASSES:
    print("-", c)

# =========================================================
# MODEL
# =========================================================

classifier = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=len(CLASSES)
)

classifier.load_state_dict(
    torch.load(
        "plant_classifier.pth",
        map_location=DEVICE
    )
)

classifier.to(DEVICE)
classifier.eval()

# =========================================================
# FEATURE EXTRACTOR
# =========================================================

feature_extractor = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=0
)

feature_extractor.load_state_dict(
    torch.load(
        "plant_classifier.pth",
        map_location=DEVICE
    ),
    strict=False
)

feature_extractor.to(DEVICE)
feature_extractor.eval()

# =========================================================
# EMBEDDINGS
# =========================================================

known_embeddings = np.load(
    "known_embeddings.npy"
)

# =========================================================
# TRANSFORM
# =========================================================

transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(
        IMG_SIZE
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# PREDICT
# =========================================================

def predict(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = classifier(tensor)

        probs = torch.softmax(
            logits,
            dim=1
        )

        confidence, pred_idx = torch.max(
            probs,
            dim=1
        )

        confidence = confidence.item()
        pred_idx = pred_idx.item()

        embedding = feature_extractor(
            tensor
        )

        embedding = embedding.cpu().numpy()

    sims = cosine_similarity(
        embedding,
        known_embeddings
    )[0]

    best_similarity = np.max(sims)

    print("\n===================")
    print("SIMILARITIES\n")

    for label, sim in zip(
        CLASSES,
        sims
    ):
        print(
            f"{label}: {sim:.4f}"
        )

    print("\n===================")

    if best_similarity < UNKNOWN_THRESHOLD:

        print(
            "\nPLANTA DESCONOCIDA"
        )

        print(
            f"Similarity: "
            f"{best_similarity:.4f}"
        )

        return

    print(
        f"\nPredicción: "
        f"{CLASSES[pred_idx]}"
    )

    print(
        f"Confidence: "
        f"{confidence:.4f}"
    )

    print(
        f"Similarity: "
        f"{best_similarity:.4f}"
    )

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    predict("test.jpg")