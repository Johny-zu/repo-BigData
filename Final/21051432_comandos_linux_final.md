# Comandos que se usaron para el proyecto final y dar asi todo el entorno de la MV y ademas lo que ocupe

## Instale dependencias
    sudo yum install zlib
    sudo yum install zlib-devel
    sudo yum install libjpeg-devel
    sudo yum install gcc

## Instale las librerias necesarias.
    sudo pip3 install matplotlib
    sudo pip3 install seaborn
    sudo pip3 install StandardScaler
    sudo pip3 install cdist
    sudo pip3 install PCA
    sudo pip3 install Cython
    sudo pip3 install KMeans
    sudo pip3 install setuptools_scm
    sudo pip3 install scikit-learn
    sudo pip3 install scikit-learn-extra

## Verificamos como andamos en el python
python3 --version
pip3 --version

## Movimiento de mi dataset a la MV
scp "C:\Users\angel\Downloads\Escuela\Octavo semestre\Big data\programas\Programas_peiton\Final\Credit_Risk_Benchmark_Dataset.csv" u21051432@192.168.1.3:/home/u21051432/

## Activar un entorno visual creado antreriormente
source ~/venv_wordcloud/bin/activate

## Creamos el .py para ejecutar los modelos
vi modelos_finales.py

## hacemos correr el codigo
python3 modelos_finales.py