import os
import argparse
import torch
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from sklearn.preprocessing import label_binarize

# =====================================================
# ARGUMENTOS
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--data_dir",
    type=str,
    required=True
)

parser.add_argument(
    "--weights",
    type=str,
    required=True
)

parser.add_argument(
    "--model",
    type=str,
    required=True
)

parser.add_argument(
    "--img_size",
    type=int,
    default=224
)

args = parser.parse_args()

# =====================================================
# DIRECTORIO RESULTADOS
# =====================================================

RESULTS_DIR = "results"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"\nDevice: {device}")

# =====================================================
# NORMALIZACION IMAGENET
# =====================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =====================================================
# TRANSFORMS
# =====================================================

transform = transforms.Compose([

    transforms.Resize(256),

    transforms.CenterCrop(
        args.img_size
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

])

# =====================================================
# DATASET
# =====================================================

test_dataset = datasets.ImageFolder(
    os.path.join(
        args.data_dir,
        "test"
    ),
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4
)

classes = test_dataset.classes

num_classes = len(classes)

print("\nClases detectadas:")
print(classes)

# =====================================================
# MODELOS SOPORTADOS
# =====================================================

MODEL_MAP = {

    "efficientnetv2_m":
        "tf_efficientnetv2_m.in21k_ft_in1k",

    "swin_s":
        "swin_small_patch4_window7_224",

    "convnext_base":
        "convnext_base.fb_in22k_ft_in1k"
}

if args.model not in MODEL_MAP:

    raise ValueError(
        f"Modelo no soportado: {args.model}"
    )

# =====================================================
# CARGA MODELO
# =====================================================
def build_model(model_name, num_classes):

    if model_name == "efficientnetv2_m":

        model = timm.create_model(
            "tf_efficientnetv2_m.in21k_ft_in1k",
            pretrained=False,
            num_classes=num_classes
        )

    elif model_name == "swin_s":

        model = timm.create_model(
            "swin_small_patch4_window7_224",
            pretrained=False,
            num_classes=num_classes
        )

    elif model_name == "convnext_base":

        model = timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=False,
            num_classes=num_classes
        )

        in_features = model.head.fc.in_features

        model.head.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(
                in_features,
                num_classes
            )
        )

    else:

        raise ValueError(
            f"Modelo no soportado: {model_name}"
        )

    return model


model = build_model(
    args.model,
    num_classes
)
checkpoint = torch.load(
    args.weights,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

model.eval()

# =====================================================
# PREDICCIONES
# =====================================================

y_true = []
y_pred = []
y_probs = []

print("\nEvaluando modelo...\n")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probs = torch.softmax(
            outputs,
            dim=1
        )

        _, preds = torch.max(
            outputs,
            1
        )

        y_true.extend(
            labels.numpy()
        )

        y_pred.extend(
            preds.cpu().numpy()
        )

        y_probs.extend(
            probs.cpu().numpy()
        )

y_probs = np.array(
    y_probs
)

# =====================================================
# METRICAS GENERALES
# =====================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

print("\n========== METRICAS ==========")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# =====================================================
# GUARDAR METRICAS CSV
# =====================================================

metrics_df = pd.DataFrame([{

    "model": args.model,

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1

}])

metrics_csv = os.path.join(

    RESULTS_DIR,

    f"{args.model}_metrics.csv"

)

metrics_df.to_csv(
    metrics_csv,
    index=False
)

print(
    f"\nMetricas guardadas: {metrics_csv}"
)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

report_dict = classification_report(

    y_true,

    y_pred,

    target_names=classes,

    output_dict=True

)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_csv = os.path.join(

    RESULTS_DIR,

    f"{args.model}_classification_report.csv"

)

report_df.to_csv(
    report_csv
)

print(
    f"Classification Report guardado: {report_csv}"
)

# =====================================================
# MATRIZ DE CONFUSION
# =====================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(12, 10)
)

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=classes,

    yticklabels=classes,

    linewidths=0.5

)

plt.title(
    f"Confusion Matrix - {args.model}"
)

plt.xlabel("Predicted")

plt.ylabel("True")

plt.xticks(rotation=45)

plt.yticks(rotation=0)

plt.tight_layout()

cm_path = os.path.join(

    RESULTS_DIR,

    f"{args.model}_confusion_matrix.png"

)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Confusion Matrix guardada: {cm_path}"
)

# =====================================================
# ROC CURVE
# =====================================================

y_true_bin = label_binarize(

    y_true,

    classes=range(num_classes)

)

auc_rows = []

plt.figure(
    figsize=(12, 10)
)

sns.set_style(
    "whitegrid"
)

for i in range(num_classes):

    fpr, tpr, _ = roc_curve(

        y_true_bin[:, i],

        y_probs[:, i]

    )

    roc_auc = auc(
        fpr,
        tpr
    )

    auc_rows.append({

        "model":
            args.model,

        "class":
            classes[i],

        "auc":
            roc_auc

    })

    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label=f"{classes[i]} | AUC={roc_auc:.3f}"

    )

plt.plot(

    [0, 1],

    [0, 1],

    linestyle="--",

    linewidth=2

)

plt.title(
    f"ROC Curve - {args.model}"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.legend(
    loc="lower right",
    fontsize=9
)

plt.grid(True)

plt.tight_layout()

roc_path = os.path.join(

    RESULTS_DIR,

    f"{args.model}_roc_curve.png"

)

plt.savefig(

    roc_path,

    dpi=300,

    bbox_inches="tight"

)

plt.close()

print(
    f"ROC Curve guardada: {roc_path}"
)

# =====================================================
# GUARDAR AUC POR CLASE
# =====================================================

auc_df = pd.DataFrame(
    auc_rows
)

auc_csv = os.path.join(

    RESULTS_DIR,

    f"{args.model}_auc_per_class.csv"

)

auc_df.to_csv(
    auc_csv,
    index=False
)

print(
    f"AUC por clase guardado: {auc_csv}"
)

# =====================================================
# RESUMEN FINAL
# =====================================================

print("\n===================================")
print("EVALUACION FINALIZADA")
print("===================================")

print(f"Modelo : {args.model}")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nArchivos generados:")

print(metrics_csv)
print(report_csv)
print(auc_csv)
print(cm_path)
print(roc_path)