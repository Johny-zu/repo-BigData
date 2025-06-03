def filtrar_df(df, rango, riesgos, num_dep, rango_edad_val):
    max_dependientes = int(df['num_dependientes'].max())
    if num_dep < 0 or num_dep > max_dependientes:
        return None
    return df[
        (df['radio_deuda'] >= rango[0]) & 
        (df['radio_deuda'] <= rango[1]) & 
        (df['nivel_riesgo'].isin(riesgos)) &
        (df['num_dependientes'] == num_dep) &
        (df['edad'] >= rango_edad_val[0]) &
        (df['edad'] <= rango_edad_val[1])
    ]

def clasificar_riesgo(row):
    if row['atraso_90'] > 0:
        return 'Alto'
    elif row['atraso_60_89'] > 0:
        return 'Medio'
    elif row['atraso_30_59'] > 0:
        return 'Bajo'
    else:
        return 'Sin riesgo'