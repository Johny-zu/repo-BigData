import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. Cargar datos desde PySpark (lectura directa de Hadoop)
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("BigDataCredito").getOrCreate()
df_spark = spark.read.csv("Credit_Risk_Benchmark_Dataset.csv", header=True, inferSchema=True)
df = df_spark.toPandas()

# Renombrar columnas a español
df = df.rename(columns={
    'rev_util':'uso_de_credito',
    'age':'edad',
    'debt_ratio':'radio_deuda',
    'real_estate':'num_otros_prestamos',
    'dependents':'num_dependientes',
    'late_30_59':'atraso_30_59',
    'late_60_89':'atraso_60_89',
    'late_90':'atraso_90',
    'monthly_inc':'ingreso_mensual',
    'open_credit':'cuentas_abiertas'
})

# Crear columna 'nivel_riesgo' según el atraso_90, atraso_60_89 y atraso_30_59
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

# 2. Panorama general de los datos
print("Forma del dataset:", df.shape)
print("Tipos de datos:\n", df.dtypes)
print("Nulos por columna:\n", df.isnull().sum())
print("Duplicados:", df.duplicated().sum())
print("Valores únicos por columna:\n", df.nunique())

# 3. Análisis exploratorio de datos
print("\nEstadísticas básicas:\n", df.describe(include='all'))

# Histograma de todas las columnas numéricas
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.savefig("histogramas.png")
plt.close()

# Boxplot de todas las columnas numéricas
df.plot(kind='box', subplots=True, layout=(2, int(np.ceil(len(df.select_dtypes(include=np.number).columns)/2))), figsize=(15, 8))
plt.tight_layout()
plt.savefig("boxplots.png")
plt.close()

# 4. Heatmap de correlación
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap de correlación')
plt.savefig("heatmap_correlacion.png")
plt.close()

# 5. Gráficas de barras entre dos variables (elige dos columnas relevantes)
for col in df.select_dtypes(include='object').columns:
    if df[col].nunique() < 10:
        for num_col in df.select_dtypes(include=np.number).columns:
            plt.figure()
            sns.barplot(x=col, y=num_col, data=df)
            plt.title(f'Relación entre {col} y {num_col}')
            plt.savefig(f"barras_{col}_{num_col}.png")
            plt.close()

# 6. Preparación de datos
if 'ID' in df.columns:
    df = df.drop(['ID'], axis=1)
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category').cat.codes
df = df.dropna()

# 7. Escalamiento
scaler = StandardScaler()
X = scaler.fit_transform(df)

# 8. Reducción de dimensionalidad: PCA y t-SNE
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# 9. Método del codo para KMeans
inertia = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
plt.plot(range(2, 10), inertia, marker='o')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')
plt.title('Método del codo')
plt.savefig("metodo_codo.png")
plt.close()

# 10. Modelos de Machine Learning

# Regresión Lineal (usando la última columna como target)
if df.shape[1] > 1:
    X_lr = X[:, :-1]
    y_lr = X[:, -1]
    lr = LinearRegression().fit(X_lr, y_lr)
    print("R2 Regresión Lineal:", lr.score(X_lr, y_lr))

# Regresión Logística (si el target es binario)
if len(np.unique(y_lr)) == 2:
    logreg = LogisticRegression().fit(X_lr, y_lr)
    print("Accuracy Regresión Logística:", logreg.score(X_lr, y_lr))

# Análisis Discriminante Lineal
lda = LinearDiscriminantAnalysis()
lda.fit(X_lr, y_lr)
print("Accuracy LDA:", lda.score(X_lr, y_lr))

# K-Vecinos
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

# 11. Selección del mejor modelo (comparar silhouette scores)
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

# 12. Conclusiones y recomendaciones
print("\nConclusiones:")
print(f"El modelo con mejor desempeño de acuerdo al silhouette score fue: {best_model}.")
print("Se recomienda analizar más a fondo las variables que más influyen en la segmentación y considerar la recolección de más datos si es posible.")

# 13. Fin del script
print("\nAnálisis completo. Gráficas guardadas como archivos PNG en el directorio actual.")