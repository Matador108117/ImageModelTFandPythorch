# 📊 Model Performance Report

**Fecha:** 2026-03-19 09:38:08.903984

---

## 🔹 Métricas Globales

- **Accuracy:** 0.9938
- **Precision (macro):** 0.9942
- **Recall (macro):** 0.9938
- **F1-score (macro):** 0.9937
- **Matthews CorrCoef (MCC):** 0.9933
- **ROC AUC (OVR macro):** 1.0000

---

## 🔹 Métricas por Clase

|               |   precision |   recall |   f1-score |    support |
|:--------------|------------:|---------:|-----------:|-----------:|
| albahaca      |    1        |  1       |   1        |  160       |
| dieffenbachia |    1        |  1       |   1        |  120       |
| floripondio   |    1        |  0.925   |   0.961039 |  120       |
| hierbaMaestra |    1        |  1       |   1        |  120       |
| hierbabuena   |    1        |  1       |   1        |  120       |
| martanche     |    1        |  1       |   1        |  140       |
| oregano       |    1        |  1       |   1        |  120       |
| ruda          |    1        |  1       |   1        |  121       |
| sacatelimon   |    1        |  1       |   1        |  120       |
| tlanepa       |    0.930233 |  1       |   0.963855 |  120       |
| tomillo       |    1        |  1       |   1        |  100       |
| violeta       |    1        |  1       |   1        |  100       |
| accuracy      |    0.99384  |  0.99384 |   0.99384  |    0.99384 |
| macro avg     |    0.994186 |  0.99375 |   0.993741 | 1461       |
| weighted avg  |    0.99427  |  0.99384 |   0.993831 | 1461       |

---

## 🔹 Matriz de Confusión

```

[[160   0   0   0   0   0   0   0   0   0   0   0]
 [  0 120   0   0   0   0   0   0   0   0   0   0]
 [  0   0 111   0   0   0   0   0   0   9   0   0]
 [  0   0   0 120   0   0   0   0   0   0   0   0]
 [  0   0   0   0 120   0   0   0   0   0   0   0]
 [  0   0   0   0   0 140   0   0   0   0   0   0]
 [  0   0   0   0   0   0 120   0   0   0   0   0]
 [  0   0   0   0   0   0   0 121   0   0   0   0]
 [  0   0   0   0   0   0   0   0 120   0   0   0]
 [  0   0   0   0   0   0   0   0   0 120   0   0]
 [  0   0   0   0   0   0   0   0   0   0 100   0]
 [  0   0   0   0   0   0   0   0   0   0   0 100]]

```
