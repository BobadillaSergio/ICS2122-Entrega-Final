import pandas as pd
from enum import Enum, auto
from typing import Dict, Tuple, Optional, List
from load_params.dominios import PriorityFlag, Profile, Pay, DayType, LaneType
import pickle
from scipy.stats import gaussian_kde
from paths import PATH_TIEMPO_SERVICIO
from collections import defaultdict
import csv

def cargar_tiempos_entre_llegadas(ruta_excel: str):
    # Leer el CSV con pandas
    #Se puede Optimizar
    df = pd.read_csv(ruta_excel)
    data_dict = {
        DayType.NORMAL: {},
        DayType.OFERTA: {},
        DayType.DOMINGO: {}
    }
    # Construir el diccionario final
    for _, row in df.iterrows():
        if row["tipo_dia"] == "normal":
            day_type = DayType.NORMAL
        elif row["tipo_dia"] == "oferta":
            day_type = DayType.OFERTA
        elif row["tipo_dia"] == "domingo":
            day_type = DayType.DOMINGO

        data_dict[day_type][int(row['hora'])] = row['scale']


    # # Mostrar resultado
    # import pprint
    # pprint.pprint(data_dict)
    return data_dict





def cargar_paciencia_desde_excel(ruta_excel: str) -> Dict[Tuple[Profile, PriorityFlag], float]:
    ###Se puede optimizar
    df = pd.read_excel(ruta_excel)
    # Asume columnas: 'profile', 'priority', 'mean_patience_sec'
    mapping = {}
    for _, row in df.iterrows():
        profile = row['perfil']
        if profile == "weekly_planner":
            profile = Profile.WEEKLY
        elif profile == "regular":
            profile = Profile.REGULAR
        elif profile == "family_cart":
            profile = Profile.FAMILY
        elif profile == "deal_hunter":
            profile = Profile.DEAL
        elif profile == "express_basket":
            profile = Profile.EXPRESS
        elif profile == "self_checkout_fan":
            profile = Profile.SELF

        if row['prioridad'] == "VERDADERO" or row['prioridad']:
            prio = PriorityFlag.P
        elif row['prioridad'] == "FALSO" or not row['prioridad']:
            prio = PriorityFlag.NP
        else:
            prio = PriorityFlag[row['prioridad']]
        mean = float(row['scale'])
        mapping[(profile, prio)] = mean
    return mapping


#tiempos servicio
def cargar_servicio_desde_pickle(ruta: str):
    with open(ruta, 'rb') as f:
        modelos = pickle.load(f) 
    dic_servicio = {'express__cash': modelos['express__cash'],
                    'self_checkout__card': modelos['self_checkout__card'],
                    'regular__cash': modelos['regular__cash'],
                    'priority__cash': modelos['priority__cash'],
                    'express__card': modelos['express__card'],
                    'regular__card': modelos['regular__card'],
                    'priority__card': modelos['priority__card']}
    return dic_servicio

# with open(PATH_TIEMPO_SERVICIO, 'rb') as f:
#     modelos = pickle.load(f) 
# dic_servicio = {'express__cash': modelos['express__cash'],
#                 'self_checkout': modelos['self_checkout__card'],
#                 'regular__cash': modelos['regular__cash'],
#                 'priority__cash': modelos['priority__cash'],
#                 'express__card': modelos['express__card'],
#                 'regular__card': modelos['regular__card'],
#                 'priority__card': modelos['priority__card']}
# def cargar_servicio_desde_pickle(ruta: str):
#     return dic_servicio


import os
# Obtener el directorio actual del archivo
current_dir = os.path.dirname(os.path.abspath(__file__))
profit_path = os.path.join(current_dir, 'profit.pkl')

with open(profit_path, 'rb') as g:
    distribuciones_profit = pickle.load(g)

perfiles = ['regular', 'express_basket', 'family_cart', 'self_checkout_fan', 'deal_hunter', 'weekly_planner']
dias_config = {
    'normal': [1, 2, 4],
    'oferta': [3, 5, 6], 
    'domingo': [7]
}
horas = range(8, 22)  # 8 a 21

diccionario_profit = {}

# Generar todas las combinaciones
for perfil in perfiles:
    for tipo_dia, dias in dias_config.items():
        for hora in horas: 
            key = f"{perfil}_{tipo_dia}_{hora}"
            diccionario_profit[key] = distribuciones_profit[key]

def cargar_profit_desde_pickle():
    return diccionario_profit



# Precalcular los diccionarios una sola vez
paths = {
    DayType.NORMAL: os.path.join(current_dir, "probabilidades_perfiles_normal(in).csv"),
    DayType.OFERTA: os.path.join(current_dir, "probabilidades_perfiles_oferta(in).csv"),
    DayType.DOMINGO: os.path.join(current_dir, "probabilidades_perfiles_finde(in).csv")
}
# Cargar y transformar los CSV a diccionarios anidados {hora: {tipo_cliente: prob}}
prob_clientes = {}
for day, path in paths.items():
    df = pd.read_csv(path)
    horas_dict = {}
    for h, df_hora in df.groupby("arrival_hour"):
        horas_dict[int(h)] = df_hora.drop(columns=["arrival_hour"]).iloc[0].to_dict()
    prob_clientes[day] = horas_dict
def cargar_tipo_cliente_hora_dia(dia, hora):
    """Obtiene directamente el diccionario de probabilidades para ese día y hora."""
    try:
        return prob_clientes[dia][hora]
    except KeyError:
        raise ValueError(f"No se encontró la hora {hora} para el día {dia}")


def cargar_prob_items(ruta_excel:str):
    df = pd.read_excel(ruta_excel)
    # Asume columnas: 'items' y 'pmf'
    return dict(zip(df['items'], df['pmf']))

def cargar_prob_eleccion_caja(path:str = None) -> Dict:
    if path is None:
        path = os.path.join(current_dir, "probabilidad_eleccion_caja.csv")
    estructura = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            profile = row["profile"]
            priority = row["priority"]
            payment = row["payment_method"]
            lane = row["lane_type"]
            prob = float(row["probability"])

            estructura[profile][priority][payment][lane] = prob

    return estructura

def cargar_min_queue(path:str= None) -> Dict[Tuple[str, str, str], int]:
    if path is None:
        path = os.path.join(current_dir, "cola_minima_balked(in).csv")
    """
    Carga el CSV con cola mínima por perfil y devuelve un diccionario:
    {(profile, priority, payment): cola_minima_int}
    """
    resultado = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            perfil_compuesto = row.get("profile")
            cola_str = row.get("cola")

            if perfil_compuesto and cola_str:
                partes = perfil_compuesto.split("__")
                if len(partes) != 3:
                    print(f"⚠️ Perfil mal formado: {perfil_compuesto}")
                    continue

                perfil, prioridad, pago = partes
                try:
                    resultado[(perfil, prioridad, pago)] = int(cola_str)
                except ValueError:
                    print(f"⚠️ Valor inválido de cola para perfil '{perfil_compuesto}': {cola_str}")
    return resultado
