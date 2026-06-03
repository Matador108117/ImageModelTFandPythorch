import os
import cv2
import torch
import numpy as np
import torch.nn as nn

from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

import timm
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# CONFIG
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 123
CLASSES = [
    "tomato",
    "cactus",
    "rose",
    "sunflower"
]

UNKNOWN_THRESHOLD = 0.55

# =========================================================
# YOLO DETECTOR
# =========================================================

detector = YOLO("best.pt")

# =========================================================
# CLASSIFIER
# =========================================================

classifier = timm.create_model(
    "convnext_tiny",
    pretrained=True,
    num_classes=len(CLASSES)
)

classifier.load_state_dict(
    torch.load("plant_classifier.pth", map_location=DEVICE)
)

classifier.to(DEVICE)
classifier.eval()

# =========================================================
# FEATURE EXTRACTOR
# =========================================================

feature_extractor = timm.create_model(
    "convnext_tiny",
    pretrained=False,
    num_classes=0
)

feature_extractor.load_state_dict(
    torch.load("plant_classifier.pth", map_location=DEVICE),
    strict=False
)

feature_extractor.to(DEVICE)
feature_extractor.eval()

# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# EMBEDDINGS CONOCIDOS
# =========================================================

known_embeddings = np.load("known_embeddings.npy")

# shape:
# [num_classes, embedding_size]

# =========================================================
# PREDICCIÓN
# =========================================================

def classify_crop(crop_bgr):

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(crop_rgb)

    tensor = transform(pil).unsqueeze(0).to(DEVICE)

    # logits
    with torch.no_grad():

        logits = classifier(tensor)

        probs = torch.softmax(logits, dim=1)

        confidence, pred_idx = torch.max(probs, dim=1)

        confidence = confidence.item()
        pred_idx = pred_idx.item()

        # embedding
        embedding = feature_extractor(tensor)

        embedding = embedding.cpu().numpy()

    # =====================================================
    # COSINE SIMILARITY
    # =====================================================

    sims = cosine_similarity(
        embedding,
        known_embeddings
    )[0]

    best_similarity = np.max(sims)

    # =====================================================
    # UNKNOWN DETECTION
    # =====================================================

    if best_similarity < UNKNOWN_THRESHOLD:

        return {
            "label": "unknown_plant",
            "confidence": float(best_similarity)
        }

    return {
        "label": CLASSES[pred_idx],
        "confidence": confidence
    }

# =========================================================
# MAIN INFERENCE
# =========================================================

def analyze_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise Exception("No se pudo leer la imagen")

    results = detector.predict(
        source=image,
        conf=0.4,
        verbose=False
    )

    boxes = results[0].boxes

    if len(boxes) == 0:

        print("\nNO SE DETECTÓ NINGUNA PLANTA")

        return

    print(f"\nPlantas detectadas: {len(boxes)}")

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        crop = image[y1:y2, x1:x2]

        result = classify_crop(crop)

        print("\n======================")
        print(f"Planta #{i+1}")
        print(f"Clase: {result['label']}")
        print(f"Confianza: {result['confidence']:.4f}")

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0,255,0),
            2
        )

        cv2.putText(
            image,
            result["label"],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    cv2.imshow("Resultado", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    analyze_image("test.jpg")
