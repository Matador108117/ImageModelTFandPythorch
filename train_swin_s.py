import os
import torch
import timm
import argparse

from tqdm import tqdm
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

# =====================================================
# ARGUMENTOS
# =====================================================

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--output_dir", type=str, default="weights")

args = parser.parse_args()

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nDevice: {device}")

# =====================================================
# TRANSFORMS
# =====================================================

train_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor()
])

# =====================================================
# DATASETS
# =====================================================

train_dataset = datasets.ImageFolder(
    os.path.join(args.data_dir, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(args.data_dir, "val"),
    transform=val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4
)

classes = train_dataset.classes
num_classes = len(classes)

print(f"\nClases detectadas: {classes}")

# =====================================================
# MODELO
# =====================================================

model_name = "swin_small_patch4_window7_224"

model = timm.create_model(
    model_name,
    pretrained=True,
    num_classes=num_classes
)

model.to(device)

# =====================================================
# LOSS / OPTIMIZER
# =====================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=args.lr
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    patience=3,
    factor=0.5
)

scaler = GradScaler()

# =====================================================
# OUTPUT
# =====================================================

os.makedirs(args.output_dir, exist_ok=True)

model_save_path = os.path.join(
    args.output_dir,
    "swin_s_best.pth"
)

# =====================================================
# EARLY STOPPING
# =====================================================

best_val_loss = float("inf")
patience = 5
counter = 0

# =====================================================
# TRAINING
# =====================================================

for epoch in range(args.epochs):

    print(f"\n========== EPOCH {epoch+1}/{args.epochs} ==========")

    # ======================
    # TRAIN
    # ======================

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    train_loop = tqdm(train_loader)

    for images, labels in train_loop:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)

        train_correct += (preds == labels).sum().item()

        train_total += labels.size(0)

        train_loop.set_postfix(loss=loss.item())

    train_acc = train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    # ======================
    # VALIDATION
    # ======================

    model.eval()

    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()

            val_total += labels.size(0)

    val_acc = val_correct / val_total
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step(avg_val_loss)

    print(f"\nTrain Loss : {avg_train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.4f}")

    print(f"Val Loss   : {avg_val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.4f}")

    # ======================
    # SAVE BEST MODEL
    # ======================

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss
        counter = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": classes
        }, model_save_path)

        print(f"\nModelo guardado en:")
        print(model_save_path)

    else:

        counter += 1

        if counter >= patience:

            print("\nEarly stopping activado")
            break
        