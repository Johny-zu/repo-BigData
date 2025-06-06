import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

# 1. Cargar datos desde PySpark (lectura directa de Hadoop)
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("BigDataCredito").getOrCreate()
df_spark = spark.read.csv("file:///home/u21051432/Credit_Risk_Benchmark_Dataset.csv", header=True, inferSchema=True)
df = df_spark.toPandas()

# Renombrar columnas a español
df = df.rename(columns={
    'rev_util': 'uso_de_credito',
    'age': 'edad',
    'debt_ratio': 'radio_deuda',
    'real_estate': 'num_otros_prestamos',
    'dependents': 'num_dependientes',
    'late_30_59': 'atraso_30_59',
    'late_60_89': 'atraso_60_89',
    'late_90': 'atraso_90',
    'monthly_inc': 'ingreso_mensual',
    'open_credit': 'cuentas_abiertas',
    'dlq_2yr': 'delincuencia_reciente',
})

def clasificar_riesgo(row):
    if row['atraso_90'] > 0:
        return 'Alto'
    elif row['atraso_60_89'] > 0:
        return 'Medio'
    elif row['atraso_30_59'] > 0:
        return 'Bajo'
    else:
        return 'Sin riesgo'

df['nivel_riesgo'] = df.apply(clasificar_riesgo, axis=1)
df = df.drop(['atraso_30_59', 'atraso_60_89', 'atraso_90'], axis=1)

# 2. Panorama general de los datos
print("Forma del dataset:", df.shape)
print("Tipos de datos:\n", df.dtypes)
print("Nulos por columna:\n", df.isnull().sum())
print("Duplicados:", df.duplicated().sum())
print("Valores únicos por columna:\n", df.nunique())

# 3. Análisis exploratorio de datos
print("\nEstadísticas básicas:\n", df.describe(include='all'))

# 4. Preparación de datos para modelado
if 'ID' in df.columns:
    df = df.drop(['ID'], axis=1)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category').cat.codes
df = df.dropna()

# Separar variable objetivo antes de escalar
y = df['nivel_riesgo']
X_features = df.drop('nivel_riesgo', axis=1)

# 5. Escalamiento
scaler = StandardScaler()
X = scaler.fit_transform(X_features)

# 6. Reducción de dimensionalidad
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 7. Método del codo
inertia = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# 8. Generación de gráficas y PDF
with PdfPages("graficas_credito.pdf") as pdf:
    # 1. Histograma
    df.hist(figsize=(12, 8))
    plt.suptitle("Histogramas de las variables numéricas")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig()
    plt.close()

    # Página en blanco para interpretación
    plt.figure(figsize=(8, 2))
    plt.axis('off')
    plt.text(0.5, 0.5, "Interpretación de los histogramas:\n\n", ha='center', va='center', fontsize=12)
    pdf.savefig()
    plt.close()

    # 2. Boxplots
    df.plot(kind='box', subplots=True, layout=(2, int(np.ceil(len(df.select_dtypes(include=np.number).columns) / 2))), figsize=(12, 6))
    plt.suptitle("Boxplots de las variables numéricas")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig()
    plt.close()

    # Página en blanco para interpretación
    plt.figure(figsize=(8, 2))
    plt.axis('off')
    plt.text(0.5, 0.5, "Interpretación de los boxplots:\n\n", ha='center', va='center', fontsize=12)
    pdf.savefig()
    plt.close()

    # 3. Heatmap de correlación
    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Heatmap de correlación')
    pdf.savefig()
    plt.close()

    # Interpretación del heatmap
    plt.figure(figsize=(8, 2))
    plt.axis('off')
    plt.text(0.5, 0.5, "Interpretación del heatmap de correlación:\n\nAquí puedes analizar qué variables numéricas están más relacionadas entre sí. Observa los valores altos (positivos o negativos) para identificar relaciones fuertes.", ha='center', va='center', fontsize=12)
    pdf.savefig()
    plt.close()

    # 4. Gráficas de barras para variables categóricas vs numéricas
    for col in df.select_dtypes(include='category').columns:
        if df[col].nunique() < 10:
            for num_col in df.select_dtypes(include=np.number).columns:
                plt.figure(figsize=(8, 4))
                sns.barplot(x=col, y=num_col, data=df)
                plt.title(f'Relación entre {col} y {num_col}')
                pdf.savefig()
                plt.close()
                # Interpretación
                plt.figure(figsize=(8, 2))
                plt.axis('off')
                plt.text(0.5, 0.5, f"Interpretación de la relación entre {col} y {num_col}:\n\nDescribe si hay diferencias notables entre categorías.", ha='center', va='center', fontsize=12)
                pdf.savefig()
                plt.close()

    # 5. Método del codo
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 10), inertia, marker='o')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inercia')
    plt.title('Método del codo')
    pdf.savefig()
    plt.close()
    # Interpretación
    plt.figure(figsize=(8, 2))
    plt.axis('off')
    plt.text(0.5, 0.5, "Interpretación del método del codo:\n\nEl punto donde la curva deja de bajar abruptamente sugiere el número óptimo de clusters.", ha='center', va='center', fontsize=12)
    pdf.savefig()
    plt.close()

    # 6. PCA
    plt.figure(figsize=(8, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.title('Reducción de dimensionalidad con PCA')
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')
    pdf.savefig()
    plt.close()
    # Interpretación
    plt.figure(figsize=(8, 2))
    plt.axis('off')
    plt.text(0.5, 0.5, "Interpretación de PCA:\n\nObserva si los grupos de riesgo se separan visualmente en el espacio reducido.", ha='center', va='center', fontsize=12)
    pdf.savefig()
    plt.close()

    # 7. t-SNE
    plt.figure(figsize=(8, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.title('Reducción de dimensionalidad con t-SNE')
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    pdf.savefig()
    plt.close()
    # Interpretación
    plt.figure(figsize=(8, 2))
    plt.axis('off')
    plt.text(0.5, 0.5, "Interpretación de t-SNE:\n\nPermite visualizar agrupamientos no lineales. Compara con PCA.", ha='center', va='center', fontsize=12)
    pdf.savefig()
    plt.close()

# 9. Modelos supervisados y no supervisados
# Para modelos supervisados, la variable objetivo debe ser numérica
y_lr = y
X_lr = X

# Regresión Logística (solo si es binaria)
if len(np.unique(y_lr)) == 2:
    logreg = LogisticRegression().fit(X_lr, y_lr)
    print("Accuracy Regresión Logística:", logreg.score(X_lr, y_lr))

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_lr, y_lr)
print("Accuracy LDA:", lda.score(X_lr, y_lr))

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_lr, y_lr)
print("Accuracy KNN:", knn.score(X_lr, y_lr))

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
print("Silhouette KMeans:", silhouette_score(X, kmeans.labels_))

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
if len(set(dbscan_labels)) > 1 and -1 in dbscan_labels:
    mask = dbscan_labels != -1
    print("Silhouette DBSCAN:", silhouette_score(X[mask], dbscan_labels[mask]))
elif len(set(dbscan_labels)) > 1:
    print("Silhouette DBSCAN:", silhouette_score(X, dbscan_labels))
else:
    print("DBSCAN no encontró clusters válidos.")

# GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X)
print("Silhouette GMM:", silhouette_score(X, gmm_labels))

# 10. Selección del mejor modelo
scores = {
    "KMeans": silhouette_score(X, kmeans.labels_),
    "GMM": silhouette_score(X, gmm_labels)
}
if len(set(dbscan_labels)) > 1 and -1 in dbscan_labels:
    scores["DBSCAN"] = silhouette_score(X[mask], dbscan_labels[mask])
elif len(set(dbscan_labels)) > 1:
    scores["DBSCAN"] = silhouette_score(X, dbscan_labels)
best_model = max(scores, key=scores.get)
print("Mejor modelo según silhouette score:", best_model, "con score:", scores[best_model])

# 11. Conclusiones
print("\nConclusiones:")
print(f"El modelo con mejor desempeño de acuerdo al silhouette score fue: {best_model}.")
print("Se recomienda analizar más a fondo las variables que más influyen en la segmentación y considerar la recolección de más datos si es posible.")

# 12. Fin
print("\nAnálisis completo. Gráficas guardadas en archivo PDF: graficas_credito.pdf")