import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

RESULTS_DIR = "results"

MODEL_METRICS = {
    "convnext_base": os.path.join(
        RESULTS_DIR,
        "convnext_base_metrics.csv"
    ),
    "efficientnetv2_m": os.path.join(
        RESULTS_DIR,
        "efficientnetv2_m_metrics.csv"
    ),
    "swin_s": os.path.join(
        RESULTS_DIR,
        "swin_s_metrics.csv"
    )
}

missing_files = [
    file
    for file in MODEL_METRICS.values()
    if not os.path.isfile(file)
]

if missing_files:
    raise FileNotFoundError(
        "Faltan archivos de metricas: "
        + ", ".join(missing_files)
    )

metric_files = list(MODEL_METRICS.values())

dfs = []

for file in metric_files:

    df = pd.read_csv(file)

    dfs.append(df)

comparison_df = pd.concat(
    dfs,
    ignore_index=True
)

comparison_df = comparison_df[
    comparison_df["model"].isin(MODEL_METRICS)
].drop_duplicates(
    subset=["model"],
    keep="last"
).reset_index(drop=True)

if set(comparison_df["model"]) != set(MODEL_METRICS):
    raise ValueError(
        "La comparativa debe contener exactamente los tres modelos esperados"
    )

comparison_csv = os.path.join(
    RESULTS_DIR,
    "comparison_metrics.csv"
)

comparison_df.to_csv(
    comparison_csv,
    index=False
)

print("\nResumen:")
print(comparison_df)

# ==========================================
# BARPLOT
# ==========================================

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]

plot_df = comparison_df.melt(
    id_vars=["model"],
    value_vars=metrics,
    var_name="metric",
    value_name="score"
)

plt.figure(figsize=(12, 8))

sns.barplot(
    data=plot_df,
    x="metric",
    y="score",
    hue="model"
)

plt.ylim(0, 1.05)

plt.title(
    "Comparacion de Modelos"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "comparison_barplot.png"
    ),
    dpi=300
)

# ==========================================
# RADAR CHART
# ==========================================

labels = metrics

num_metrics = len(labels)

angles = np.linspace(
    0,
    2*np.pi,
    num_metrics,
    endpoint=False
)

angles = np.concatenate(
    (angles, [angles[0]])
)

fig = plt.figure(
    figsize=(10, 10)
)

ax = plt.subplot(
    111,
    polar=True
)

for _, row in comparison_df.iterrows():

    values = [
        row["accuracy"],
        row["precision"],
        row["recall"],
        row["f1_score"]
    ]

    values.append(values[0])

    ax.plot(
        angles,
        values,
        linewidth=2,
        label=row["model"]
    )

    ax.fill(
        angles,
        values,
        alpha=0.1
    )

ax.set_xticks(
    angles[:-1]
)

ax.set_xticklabels(labels)

plt.title(
    "Radar Chart de Modelos"
)

plt.legend(
    loc="upper right"
)

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "comparison_radar.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

print(
    "\nGraficas comparativas generadas"
)