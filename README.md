# Data Mining · TUIA

Tres trabajos prácticos de **Minería de Datos** (Tecnicatura Universitaria en Inteligencia Artificial, 2026), realizados en Python con scikit-learn. Cubren el recorrido típico de un proyecto de ciencia de datos: análisis exploratorio, limpieza y preparación, modelado no supervisado y supervisado, ajuste de hiperparámetros y comparación de modelos.

**Autores:** Aldana Desirée Sánchez · Gastón Rinaldi

| TP | Tema | Tipo de problema | Dataset |
|----|------|------------------|---------|
| [TP1](tp1-reduccion-dimensionalidad-clustering) | Reducción de dimensionalidad y clustering | No supervisado | Pingüinos (3 especies) |
| [TP2](tp2-arboles-naive-bayes-knn) | Árboles de decisión, Naive Bayes y k-NN | Regresión y clasificación | 1000 empresas · Prescripción de fármacos |
| [TP3](tp3-svm-random-forest) | SVM y Random Forest | Clasificación multiclase | Recomendación de cultivos (22 clases) |

Cada carpeta contiene el notebook con el análisis completo, los datos y el enunciado en PDF.

---

## TP1 · Reducción de dimensionalidad y clustering

Análisis de un dataset biométrico de pingüinos para descubrir su estructura latente sin usar la etiqueta de especie.

- **Reducción de dimensionalidad:** PCA (varianza acumulada, regla de Kaiser y scree plot para elegir componentes), Isomap (variando `n_neighbors`) y t-SNE (variando `perplexity` y `max_iter`).
- **Clustering:** K-Means y clustering jerárquico aglomerativo con dendrograma.
- **Selección de K:** Silhouette score y GAP statistic.
- **Hallazgo:** las tres técnicas separan visualmente las especies; las métricas de clustering indican K=2 como partición más natural, porque dos de las especies son muy similares en sus medidas.

## TP2 · Árboles de decisión, Naive Bayes y k-NN

**Parte A, regresión:** predicción de `Profit` a partir del gasto de 1000 empresas.
- EDA con detección de outliers, correlaciones de Pearson y Spearman, y descarte de una variable colineal.
- `DecisionTreeRegressor` con búsqueda de hiperparámetros, particiones train/validación/prueba (70/15/15, comparada con 80/20) y métricas MAE y MSE.

**Parte B, clasificación:** predicción de la droga adecuada para cada paciente (200 registros, clases desbalanceadas).
- **Árbol de decisión:** `GridSearchCV` con validación cruzada estratificada, pre-poda y post-poda por costo-complejidad (`ccp_alpha`).
- **Naive Bayes:** `GaussianNB` frente a `CategoricalNB` con discretización por cuantiles.
- **k-NN:** estandarización, variación de `k` y comparación de distancias Euclidiana y Manhattan.
- Comparación con métricas macro (precisión, recall, F1) y matrices de confusión. El árbol de decisión fue el mejor equilibrio entre rendimiento e interpretabilidad.

## TP3 · SVM y Random Forest para recomendación de cultivos

Clasificación multiclase (22 cultivos) con 16 variables de suelo y clima, sobre un dataset con problemas de calidad.

- **Limpieza:** filas con valores faltantes, 198 duplicados (evitando fuga de datos entre train y test) y valores físicamente imposibles (p. ej. pH de 25).
- **Modelos:** SVM lineal, SVM con kernel RBF y Random Forest, con búsqueda en grilla de `C`, `gamma`, `n_estimators` y `max_depth` mediante validación cruzada estratificada.
- **Resultados (accuracy en test):** SVM lineal ≈ 0,61 · SVM RBF ≈ 0,88 · **Random Forest ≈ 0,91**.
- **Interpretación:** importancia de variables (las climáticas dominan) y matrices de confusión normalizadas por clase.
- **GPU:** los modelos se entrenaron con **RAPIDS cuML** en una GPU T4 de Google Colab. Con scikit-learn en CPU es posible, pero mucho más lento.

---

## Algoritmos

| Categoría | Técnicas |
|-----------|----------|
| Reducción de dimensionalidad | PCA, Isomap, t-SNE |
| Clustering | K-Means, clustering jerárquico (Ward/dendrograma) |
| Regresión | Árbol de decisión |
| Clasificación | Árbol de decisión (con poda), Naive Bayes (Gaussiano y Categórico), k-NN, SVM (lineal y RBF), Random Forest |
| Validación y selección | Hold-out train/val/test, K-Fold estratificado, `GridSearchCV`, Silhouette, GAP statistic |
| Métricas | Accuracy, precisión/recall/F1 macro, MAE, MSE, matriz de confusión |

## Tecnologías

- **Lenguaje y entorno:** Python, Jupyter Notebook, Google Colab
- **Datos:** pandas, NumPy, SciPy
- **Machine learning:** scikit-learn, RAPIDS cuML (GPU)
- **Visualización:** Matplotlib, Seaborn

## Estructura

```
data-mining-tuia/
├── tp1-reduccion-dimensionalidad-clustering/
├── tp2-arboles-naive-bayes-knn/
└── tp3-svm-random-forest/
```

Cada carpeta incluye `*.ipynb`, los `*.csv` usados y el enunciado `*.pdf`.

## Cómo ejecutarlo

```bash
git clone https://github.com/aldanasanchezasz/data-mining-tuia.git
cd data-mining-tuia
pip install -r requirements.txt
jupyter notebook
```

Abrí el notebook desde su propia carpeta: los `.csv` se leen con rutas relativas. En Google Colab alcanza con subir el notebook y sus CSV al panel de archivos.

**TP3 y la GPU:** el notebook está pensado para Google Colab con GPU T4 y usa **RAPIDS cuML** (`cuml.svm.SVC` y `cuml.ensemble.RandomForestClassifier`); las salidas guardadas corresponden a esa ejecución. En Colab, activá la GPU y ejecutá la celda de instalación de cuML. Para correrlo en CPU con scikit-learn, comentá en la celda de librerías los imports de `cuml` y descomentá el bloque "Alternativa con scikit-learn": la búsqueda de hiperparámetros será mucho más lenta (el dataset tiene ~50.000 filas).
