import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
#removes the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)

# Cargar los datos y poner la ruta que corresponde
column_names = ["userid", "itemid","rate","timestamp"]

calificaciones = pd.read_csv('Peliculas/u3.base', sep='\t', header=None, names=column_names)
usuarios = pd.read_csv('Peliculas/u.user', sep='|')
peliculas = pd.read_csv('Peliculas/u.item', sep='|', encoding='latin-1')

