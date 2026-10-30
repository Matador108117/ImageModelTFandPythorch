import os
import copy
import time
import random
import torch
import numpy as np
import timm

from tqdm import tqdm
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler

# =========================================================
# CONFIG
# =========================================================

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"

MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"

IMAGE_SIZE = 224

BATCH_SIZE = 8

EPOCHS = 20
MIN_EPOCHS = 10

PATIENCE = 5

FREEZE_EPOCHS = 5

LEARNING_RATE = 1e-4

SEED = 42

WEIGHT_DECAY = 1e-4

LABEL_SMOOTHING = 0.1

GRAD_CLIP = 1.0

NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

OUTPUT_DIR = "weights"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_PATH = os.path.join(
    OUTPUT_DIR,
    "convnext_base_best.pth"
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =========================================================
# DATA AUGMENTATION
# =========================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE,
        scale=(0.6, 1.0),
        ratio=(0.75, 1.33)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomVerticalFlip(p=0.2),

    transforms.RandomRotation(30),

    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),

    transforms.RandomPerspective(
        distortion_scale=0.3,
        p=0.3
    ),

    transforms.RandomGrayscale(p=0.05),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),

    transforms.RandomErasing(
        p=0.3,
        scale=(0.02, 0.2),
        value='random'
    )
])

val_transform = transforms.Compose([
    transforms.Resize(256),

    transforms.CenterCrop(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

# =========================================================
# DATASETS
# =========================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

num_classes = len(train_dataset.classes)

print("\n===================================")
print("CLASES DETECTADAS")
print("===================================")
print(train_dataset.classes)
print(f"Total clases: {num_classes}")

# =========================================================
# MODELO
# =========================================================

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=num_classes
)

# =========================================================
# DROPOUT EXTRA
# =========================================================

if hasattr(model, "head") and hasattr(model.head, "fc"):

    in_features = model.head.fc.in_features

    model.head.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )

model = model.to(DEVICE)

# =========================================================
# FREEZE BACKBONE
# =========================================================

def freeze_backbone(model):

    for name, param in model.named_parameters():

        if "head" not in name:
            param.requires_grad = False

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nParámetros entrenables (head): {trainable:,}"
    )

# =========================================================
# UNFREEZE
# =========================================================

def unfreeze_all(model):

    for param in model.parameters():
        param.requires_grad = True

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nParámetros entrenables (full): {trainable:,}"
    )

freeze_backbone(model)

# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss(
    label_smoothing=LABEL_SMOOTHING
)

# =========================================================
# OPTIMIZER
# =========================================================

def get_optimizer(model, phase="head"):

    decay_params = []
    no_decay_params = []
    head_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if "head" in name:

            head_params.append(param)

        elif (
            "bias" in name
            or "norm" in name
            or "bn" in name
        ):

            no_decay_params.append(param)

        else:

            decay_params.append(param)

    if phase == "head":

        groups = [
            {
                "params": head_params,
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY
            }
        ]

    else:

        groups = [

            {
                "params": decay_params,
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY
            },

            {
                "params": no_decay_params,
                "lr": LEARNING_RATE,
                "weight_decay": 0.0
            },

            {
                "params": head_params,
                "lr": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY
            }
        ]

    return AdamW(groups)

optimizer = get_optimizer(
    model,
    phase="head"
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)

# =========================================================
# MIXED PRECISION
# =========================================================

scaler = GradScaler()

# =========================================================
# TRAIN
# =========================================================

def train_one_epoch():

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(
        train_loader,
        desc="Train"
    ):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast():

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP
        )

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    return (
        running_loss / len(train_loader),
        100 * correct / total
    )

# =========================================================
# VALIDATE
# =========================================================

def validate():

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in tqdm(
            val_loader,
            desc="Val"
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total += labels.size(0)

            correct += (
                predicted == labels
            ).sum().item()

    return (
        running_loss / len(val_loader),
        100 * correct / total
    )

# =========================================================
# TRAIN LOOP
# =========================================================

best_val_loss = float("inf")

best_model_weights = copy.deepcopy(
    model.state_dict()
)

epochs_no_improve = 0

phase = "head"

print("\n===================================")
print("FASE 1 - HEAD")
print("===================================\n")

for epoch in range(EPOCHS):

    if (
        epoch == FREEZE_EPOCHS
        and phase == "head"
    ):

        print(
            "\n==================================="
        )
        print(
            "FASE 2 - FULL FINETUNING"
        )
        print(
            "===================================\n"
        )

        unfreeze_all(model)

        optimizer = get_optimizer(
            model,
            phase="full"
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3
        )

        phase = "full"

    print("\n" + "=" * 60)

    print(
        f"EPOCH {epoch+1}/{EPOCHS}"
    )

    print(
        f"PHASE: {phase}"
    )

    print("=" * 60)

    start = time.time()

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc = validate()

    scheduler.step(val_loss)

    gap = train_acc - val_acc

    print(
        f"\nTrain Loss : {train_loss:.4f}"
    )

    print(
        f"Train Acc  : {train_acc:.2f}%"
    )

    print(
        f"Val Loss   : {val_loss:.4f}"
    )

    print(
        f"Val Acc    : {val_acc:.2f}%"
    )

    print(
        f"Gap        : {gap:.2f}%"
    )

    print(
        f"LR         : {optimizer.param_groups[-1]['lr']:.8f}"
    )

    print(
        f"Tiempo     : {time.time()-start:.2f}s"
    )

    if val_loss < best_val_loss:

        print(
            "\n✓ Mejor modelo encontrado"
        )

        best_val_loss = val_loss

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save({

            "model_state_dict":
                model.state_dict(),

            "classes":
                train_dataset.classes,

            "model_name":
                MODEL_NAME,

            "image_size":
                IMAGE_SIZE,

            "val_acc":
                val_acc,

            "val_loss":
                val_loss

        }, SAVE_PATH)

        epochs_no_improve = 0

    else:

        epochs_no_improve += 1

        print(
            f"\nSin mejora: {epochs_no_improve}/{PATIENCE}"
        )

    if (
        (epoch + 1) >= MIN_EPOCHS
        and epochs_no_improve >= PATIENCE
    ):

        print(
            "\nEarly Stopping activado"
        )

        break

# =========================================================
# CARGAR MEJOR MODELO
# =========================================================

model.load_state_dict(
    best_model_weights
)

print("\n===================================")
print("ENTRENAMIENTO FINALIZADO")
print("===================================")

print(
    f"Modelo guardado: {SAVE_PATH}"
)

print(
    f"Mejor Val Loss: {best_val_loss:.4f}"
)