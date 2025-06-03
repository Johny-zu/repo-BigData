# Importamos librerías necesarias
import pandas as pd
import panel as pn
import hvplot.pandas

# Configuración de Panel
pn.extension()

# Lectura del dataset
df = pd.read_csv('Credit Risk Benchmark Dataset.csv')

# Funciones de filtrado y visualización
def filtrar_df(rango_deuda, nivel_riesgo, num_dependientes, rango_edad):
    return df[
        (df['radio_deuda'] >= rango_deuda[0]) & 
        (df['radio_deuda'] <= rango_deuda[1]) & 
        (df['nivel_riesgo'].isin(nivel_riesgo)) &
        (df['num_dependientes'] == num_dependientes) &
        (df['edad'] >= rango_edad[0]) &
        (df['edad'] <= rango_edad[1])
    ]

@pn.depends(...)
def indicadores(rango_deuda, nivel_riesgo, num_dependientes, rango_edad):
    dff = filtrar_df(rango_deuda, nivel_riesgo, num_dependientes, rango_edad)
    total = len(dff)
    ingreso_promedio = dff['ingreso_mensual'].mean()
    return pn.Column(
        pn.pane.Markdown(f"### Total de registros\n# {total}"),
        pn.pane.Markdown(f"### Ingreso mensual promedio\n# {ingreso_promedio:,.2f}")
    )

# Definición de gráficas
@pn.depends(...)
def grafica_barras_edad_ingreso(rango_deuda, nivel_riesgo, num_dependientes, grupos_edad):
    dff = filtrar_df(rango_deuda, nivel_riesgo, num_dependientes, grupos_edad)
    return dff.hvplot.bar(...)

# Layout del dashboard
dashboard = pn.Column(
    pn.pane.Markdown("## Dashboard de Riesgo Crediticio"),
    pn.Row(
        pn.Column(...),  # Filtros
        pn.Column(...),  # Gráficas
    ),
    pn.Row(
        pn.Column(...),  # Tabla de datos
    )
)

# Servir el dashboard
dashboard.servable()