import os

# Obtener el directorio raíz del proyecto
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LOAD_PARAMS_DIR = os.path.join(PROJECT_ROOT, "load_params")

PATH_EXP_PACIENCIA = os.path.join(LOAD_PARAMS_DIR, "Ajustes_exponencial_paciencia.xlsx")
PATH_ENTRE_LLEGADAS = os.path.join(LOAD_PARAMS_DIR, "tiempo_llegada_entre_clientes.csv")
PATH_TIEMPO_SERVICIO = os.path.join(LOAD_PARAMS_DIR, "t_servicio.pkl")
PATH_CLIENTES_NORMAL = os.path.join(LOAD_PARAMS_DIR, "probabilidades_perfiles_normal(in).csv")
PATH_CLIENTES_OFERTA = os.path.join(LOAD_PARAMS_DIR, "probabilidades_perfiles_oferta(in).csv")
PATH_CLIENTES_DOMINGO = os.path.join(LOAD_PARAMS_DIR, "probabilidades_perfiles_finde(in).csv")