import json
import torch
import timm
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay
)

# =========================================================
# CONFIG
# =========================================================

TEST_DIR = "dataset/test"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 224

MODEL_NAME = "convnext_tiny"

# =========================================================
# CLASSES
# =========================================================

with open("classes.json") as f:
    CLASSES = json.load(f)

# =========================================================
# TRANSFORM
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
    TEST_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)

# =========================================================
# MODEL
# =========================================================

model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=len(CLASSES)
)

model.load_state_dict(
    torch.load(
        "plant_classifier.pth",
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

# =========================================================
# EVALUATE
# =========================================================

y_true = []
y_pred = []

with torch.no_grad():

    for images, labels in tqdm(loader):

        images = images.to(DEVICE)

        outputs = model(images)

        preds = torch.argmax(
            outputs,
            dim=1
        )

        y_true.extend(labels.numpy())

        y_pred.extend(
            preds.cpu().numpy()
        )

# =========================================================
# METRICS
# =========================================================

acc = accuracy_score(
    y_true,
    y_pred
)

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASSES,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    "classification_report.csv"
)

# =========================================================
# MARKDOWN REPORT
# =========================================================

with open("report.md", "w") as f:

    f.write("# Plant Classifier Report\n\n")

    f.write(
        f"## Accuracy\n\n"
        f"{acc:.4f}\n\n"
    )

    f.write(
        "## Classification Report\n\n"
    )

    f.write(
        report_df.to_markdown()
    )

# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)

fig, ax = plt.subplots(
    figsize=(12,12)
)

disp.plot(
    ax=ax,
    xticks_rotation=45
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

print("\nREPORT GENERATED")