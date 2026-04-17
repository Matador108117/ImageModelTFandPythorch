import os
import numpy as np

from sklearn.metrics import classification_report

import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -------------------------------------------------------
# Límite de memoria RAM (~6 GB)
# -------------------------------------------------------
#imports para la documentacion 
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import label_binarize
from datetime import datetime
import pandas as pd


# -------------------------------------------------------
# Configuración
# -------------------------------------------------------

IMG_SIZE = 224
BATCH_SIZE = 16  # más pequeño para ir tranqui con la RAM

# Carpeta base donde están las subcarpetas de cada planta
PLANT_DATA_PATH = os.path.join(os.getcwd(), "plant_data")

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DISABLE_MKL"] = "1"

# Limitar memoria GPU (crecimiento bajo demanda)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# -------------------------------------------------------
# Generadores: NO cargamos todo a memoria
# -------------------------------------------------------

# Data augmentation SOLO para entrenamiento
train_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    validation_split=0.2,
    preprocessing_function=preprocess_input  # MobileNetV2
)

# Validación sin aumentación agresiva
val_datagen = ImageDataGenerator(
    validation_split=0.2,
    preprocessing_function=preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    PLANT_DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="training",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    PLANT_DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    subset="validation",
    shuffle=False  # importante para alinear y_true y y_pred
)

CLASSES = list(train_gen.class_indices.keys())
print("[INFO] Clases detectadas:", CLASSES)

# Guardar mapping de clases
np.save("plant_label_classes.npy", np.array(CLASSES))
print("[INFO] Clases guardadas en plant_label_classes.npy")

# -------------------------------------------------------
# Modelo con MobileNetV2 (fase 1: base congelada)
# -------------------------------------------------------

base = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base.trainable = False  # Fase 1: no tocamos la base

model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base,
    GlobalAveragePooling2D(),

    Dense(256, activation="relu"),
    Dropout(0.4),

    Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7
    )
]

print("[INFO] Entrenando fase 1 (base congelada)...")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------------------------------
# Fine-tuning (fase 2: descongelar parte de la base)
# -------------------------------------------------------

print("[INFO] Configurando fine-tuning...")

base.trainable = True

# Descongelar solo la parte “alta” de MobileNetV2
fine_tune_from = 100  # ajustable si quieres revisando base.summary()

for i, layer in enumerate(base.layers):
    layer.trainable = (i >= fine_tune_from)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("[INFO] Entrenando fase 2 (fine-tuning)...")

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------------------------------
# Evaluación con classification_report
# -------------------------------------------------------

print("[INFO] Evaluando en conjunto de validación...")

# Número de pasos para recorrer completamente val_gen
val_steps = int(np.ceil(val_gen.samples / BATCH_SIZE))

y_pred_probs = model.predict(val_gen, steps=val_steps)
y_pred = np.argmax(y_pred_probs, axis=1)

y_true = val_gen.classes  # etiquetas verdaderas en el mismo orden

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=CLASSES))
print("[INFO] Calculando métricas adicionales...")

# -----------------------------
# Métricas globales
# -----------------------------

accuracy = accuracy_score(y_true, y_pred)
precision_macro = precision_score(y_true, y_pred, average="macro")
recall_macro = recall_score(y_true, y_pred, average="macro")
f1_macro = f1_score(y_true, y_pred, average="macro")
mcc = matthews_corrcoef(y_true, y_pred)

# -----------------------------
# ROC AUC (multiclase OVR)
# -----------------------------

y_true_bin = label_binarize(y_true, classes=range(len(CLASSES)))

try:
    roc_auc = roc_auc_score(
        y_true_bin,
        y_pred_probs,
        multi_class="ovr",
        average="macro"
    )
except Exception:
    roc_auc = None

# -----------------------------
# Matriz de confusión
# -----------------------------

cm = confusion_matrix(y_true, y_pred)

# -----------------------------
# Reporte detallado por clase
# -----------------------------

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASSES,
    output_dict=True
)

report_df = pd.DataFrame(report_dict).transpose()
print("[INFO] Generando archivo Markdown con métricas...")

md_lines = []
md_lines.append("# 📊 Model Performance Report\n")
md_lines.append(f"**Fecha:** {datetime.now()}\n")
md_lines.append("---\n")

md_lines.append("## 🔹 Métricas Globales\n")
md_lines.append(f"- **Accuracy:** {accuracy:.4f}")
md_lines.append(f"- **Precision (macro):** {precision_macro:.4f}")
md_lines.append(f"- **Recall (macro):** {recall_macro:.4f}")
md_lines.append(f"- **F1-score (macro):** {f1_macro:.4f}")
md_lines.append(f"- **Matthews CorrCoef (MCC):** {mcc:.4f}")

if roc_auc is not None:
    md_lines.append(f"- **ROC AUC (OVR macro):** {roc_auc:.4f}")
else:
    md_lines.append("- **ROC AUC:** No disponible")

md_lines.append("\n---\n")

# Tabla métricas por clase
md_lines.append("## 🔹 Métricas por Clase\n")
md_lines.append(report_df.to_markdown())
md_lines.append("\n---\n")

# Matriz de confusión
md_lines.append("## 🔹 Matriz de Confusión\n")
md_lines.append("```\n")
md_lines.append(np.array2string(cm))
md_lines.append("\n```")

with open("model_metrics_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("[INFO] Reporte guardado en model_metrics_report.md")

# -------------------------------------------------------
# Guardar modelo
# -------------------------------------------------------

# Para reusar localmente con Keras 3
model.save("plants_model.keras")
print("[INFO] Modelo guardado en plants_model.keras")

# Para TensorFlow Serving / Docker
EXPORT_DIR = os.path.join(os.getcwd(), "plantmodel", "1")
os.makedirs(EXPORT_DIR, exist_ok=True)
model.export(EXPORT_DIR)
print(f"[INFO] Modelo exportado a: {EXPORT_DIR}")

# -------------------------------------------------------
# Prueba con una imagen individual
# --------------------------------------------l-----------

import cv2

# Tomamos una imagen de la primera clase como demo
first_class = CLASSES[0]
first_class_dir = os.path.join(PLANT_DATA_PATH, first_class)
test_file = os.listdir(first_class_dir)[0]
test_path = os.path.join(first_class_dir, test_file)

print(f"[INFO] Probando imagen: {test_path}")

test_bgr = cv2.imread(test_path)
if test_bgr is None:
    raise FileNotFoundError(f"No se pudo leer la imagen de prueba: {test_path}")

test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
test_rgb = cv2.resize(test_rgb, (IMG_SIZE, IMG_SIZE))
test_rgb = test_rgb.astype("float32")
test_proc = preprocess_input(test_rgb)
test_proc = np.expand_dims(test_proc, axis=0)

pred = model.predict(test_proc)[0]
pred_idx = int(np.argmax(pred))
pred_label = CLASSES[pred_idx]

print("\n===== PREDICCIÓN LOCAL (PLANTA) =====")
print(f"Predicción: {pred_label}")
print("Probabilidades por clase:")
for cls, prob in zip(CLASSES, pred):
    print(f"  {cls}: {prob:.4f}")
