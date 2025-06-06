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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn_extra.cluster import KMedoids

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

# t-SNE solo con una muestra si hay muchos datos
if X.shape[0] > 2000:
    idx = np.random.choice(X.shape[0], 2000, replace=False)
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X[idx])
    y_tsne = y.iloc[idx]
else:
    X_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
    y_tsne = y

# 7. Método del codo
inertia = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# 8. Modelos supervisados y no supervisados
# Regresión Lineal Simple (ejemplo: edad -> ingreso_mensual)
linreg_simple = LinearRegression().fit(df[['edad']], df['ingreso_mensual'])
simple_score = linreg_simple.score(df[['edad']], df['ingreso_mensual'])

# Regresión Lineal Múltiple (todas menos ingreso_mensual y nivel_riesgo)
X_mult = df.drop(['ingreso_mensual', 'nivel_riesgo'], axis=1)
y_mult = df['ingreso_mensual']
linreg_multi = LinearRegression().fit(X_mult, y_mult)
multi_score = linreg_multi.score(X_mult, y_mult)

# Regresión Logística (solo si es binaria)
if len(np.unique(y)) == 2:
    logreg = LogisticRegression().fit(X, y)
    logreg_score = logreg.score(X, y)
else:
    logreg_score = None

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
lda_score = lda.score(X, y)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
knn_score = knn.score(X, y)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
kmeans_silhouette = silhouette_score(X, kmeans.labels_)

# K-Medoids
kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids_labels = kmedoids.fit_predict(X)
kmedoids_silhouette = silhouette_score(X, kmedoids_labels)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
if len(set(dbscan_labels)) > 1 and -1 in dbscan_labels:
    mask = dbscan_labels != -1
    dbscan_silhouette = silhouette_score(X[mask], dbscan_labels[mask])
elif len(set(dbscan_labels)) > 1:
    dbscan_silhouette = silhouette_score(X, dbscan_labels)
else:
    dbscan_silhouette = None

# GMM
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X)
gmm_silhouette = silhouette_score(X, gmm_labels)

# 9. Selección del mejor modelo
scores = {
    "KMeans": kmeans_silhouette,
    "KMedoids": kmedoids_silhouette,
    "GMM": gmm_silhouette
}
if dbscan_silhouette is not None:
    scores["DBSCAN"] = dbscan_silhouette
best_model = max(scores, key=scores.get)

# 10. Generación de gráficas y PDF
with PdfPages("graficas_credito.pdf") as pdf:
    # Separador: Mostrando análisis de datos
    plt.figure(figsize=(10, 7))
    plt.axis('off')
    plt.text(0.5, 0.5, "Mostrando análisis de datos", ha='center', va='center', fontsize=24)
    pdf.savefig()
    plt.close()

    # Panorama general de los datos
    plt.figure(figsize=(10, 7))
    plt.axis('off')
    text = f"""Panorama General de los Datos

Cantidad de observaciones: {df.shape[0]}
Cantidad de columnas: {df.shape[1]}

Tipos de datos:
{df.dtypes}

Cantidad de nulos por columna:
{df.isnull().sum()}

Cantidad de duplicados: {df.duplicated().sum()}

Cantidad de valores únicos por columna:
{df.nunique()}
"""
    plt.text(0, 1, text, fontsize=12, va='top', family='monospace')
    pdf.savefig(); plt.close()

    # Estadísticas básicas
    plt.figure(figsize=(10, 7))
    plt.axis('off')
    plt.text(0, 1, "Estadísticas Básicas\n\n" + str(df.describe()), fontsize=10, va='top', family='monospace')
    pdf.savefig(); plt.close()

    # Histogramas individuales con rangos correctos
    for col in df.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(8, 4))
        min_val, max_val = df[col].min(), df[col].max()
        if col == "uso_de_credito":
            bins = np.arange(0, 1.1, 0.1)
            sns.histplot(df[(df[col] >= 0) & (df[col] <= 1)][col], bins=bins, kde=True)
            plt.xticks(bins)
            plt.xlim(0, 1)
            plt.ylim(0, 5000)  # Ajusta este valor según tu dataset para que sea legible
        elif col == "radio_deuda":
            bins = np.arange(0, 1.1, 0.1)
            sns.histplot(df[df[col] <= 1][col], bins=bins, kde=True)
            plt.xticks(bins)
            plt.ylim(0, 20000)
        elif col == "ingreso_mensual":
            step = 5000
            bins = np.arange(0, 50000 + step, step)
            sns.histplot(df[df[col] <= 50000][col], bins=bins, kde=True)
            plt.xticks(bins, rotation=45)
            plt.xlim(0, 50000)
            plt.ylim(0, 20000)
        elif col == "num_otros_prestamos":
            bins = np.arange(0, 6, 1)
            sns.histplot(df[df[col] <= 5][col], bins=bins, kde=False)
            plt.xticks(bins)
            plt.xlim(0, 5)
            plt.ylim(0, 8000)
        elif col == "cuentas_abiertas":
            bins = np.arange(0, 31, 1)
            sns.histplot(df[df[col] <= 30][col], bins=bins, kde=False)
            plt.xticks(bins)
            plt.xlim(0, 30)
            plt.ylim(0, 8000)
        else:
            sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'Histograma de {col} (rango: {min_val:.2f} a {max_val:.2f})')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')
        pdf.savefig()
        plt.close()

    # Boxplots individuales por variable numérica (con límites iguales al histograma)
    for col in df.select_dtypes(include=np.number).columns:
        plt.figure(figsize=(8, 4))
        if col == "uso_de_credito":
            sns.boxplot(x=df[(df[col] >= 0) & (df[col] <= 1)][col])
            plt.xlim(0, 1)
        elif col == "radio_deuda":
            sns.boxplot(x=df[df[col] <= 1][col])
            plt.xlim(0, 1)
        elif col == "ingreso_mensual":
            sns.boxplot(x=df[df[col] <= 50000][col])
            plt.xlim(0, 50000)
        elif col == "num_otros_prestamos":
            sns.boxplot(x=df[df[col] <= 5][col])
            plt.xlim(0, 5)
        elif col == "cuentas_abiertas":
            sns.boxplot(x=df[df[col] <= 30][col])
            plt.xlim(0, 30)
        else:
            sns.boxplot(x=df[col])
        plt.title(f'Boxplot de {col}')
        plt.xlabel(col)
        pdf.savefig()
        plt.close()

    # Heatmap de correlación
    plt.figure(figsize=(12, 10))
    corr = df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap de correlación entre variables numéricas')
    pdf.savefig(); plt.close()

    # Gráficas de barras para variables categóricas vs numéricas
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

    # Separador: Manejo de modelos
    plt.figure(figsize=(10, 7))
    plt.axis('off')
    plt.text(0.5, 0.5, "Manejo de modelos", ha='center', va='center', fontsize=24)
    pdf.savefig()
    plt.close()

    # Método del codo
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

    # PCA
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

    # t-SNE
    plt.figure(figsize=(8, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_tsne, cmap='viridis', alpha=0.6)
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

    # Resultados de modelos (todos juntos en una sola hoja)
    plt.figure(figsize=(10, 7))
    plt.axis('off')
    resultados = (
        f"Regresión Lineal Simple (edad -> ingreso_mensual): R2 = {simple_score:.3f}\n"
        f"Regresión Lineal Múltiple (todas -> ingreso_mensual): R2 = {multi_score:.3f}\n"
        f"Regresión Logística: {logreg_score if logreg_score is not None else 'No aplica'}\n"
        f"LDA: {lda_score:.3f}\n"
        f"KNN: {knn_score:.3f}\n"
        f"KMeans Silhouette: {kmeans_silhouette:.3f}\n"
        f"K-Medoids Silhouette: {kmedoids_silhouette:.3f}\n"
        f"DBSCAN Silhouette: {dbscan_silhouette if dbscan_silhouette is not None else 'No aplica'}\n"
        f"GMM Silhouette: {gmm_silhouette:.3f}\n"
        f"Mejor modelo según silhouette: {best_model} (score: {scores[best_model]:.3f})"
    )
    plt.text(0.01, 0.5, resultados, fontsize=12, va='center')
    pdf.savefig()
    plt.close()

    # Conclusiones finales en el PDF
    plt.figure(figsize=(10, 7))
    plt.axis('off')
    conclusiones = (
        f"Conclusiones:\n\n"
        f"El modelo con mejor desempeño de acuerdo al silhouette score fue: {best_model} (score: {scores[best_model]:.3f}).\n\n"
        "Se recomienda analizar más a fondo las variables que más influyen en la segmentación y considerar la recolección\n de más datos si es posible. para mejorar la prediccion"
        "\n Además, los resultados muestran el desempeño de todos los modelos supervisados y no supervisados aplicados al dataset."
    )
    plt.text(0.01, 0.5, conclusiones, fontsize=14, va='center')
    pdf.savefig()
    plt.close()

# 11. Conclusiones
print("\nConclusiones:")
print(f"El modelo con mejor desempeño de acuerdo al silhouette score fue: {best_model}.")
print("Se recomienda analizar más a fondo las variables que más influyen en la segmentación y considerar la recolección de más datos si es posible.")

# 12. Fin
print("\nAnálisis completo. Gráficas guardadas en archivo PDF: graficas_credito.pdf")