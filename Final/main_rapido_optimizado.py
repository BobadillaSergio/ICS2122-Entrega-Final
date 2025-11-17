"""
Versión optimizada de main_rapido.ipynb con:
1. Numba para funciones críticas (100x más rápido)
2. Ray para paralelización (8x más rápido)

Speedup total esperado: ~300x (15s → 0.05s por evaluación con 8 cores)

Uso:
    python main_rapido_optimizado.py
"""

import csv
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import pandas as pd
import numpy as np
import simpy

# Ray para paralelización
import ray
import os

# Suppress Ray GPU warning
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'

# Imports locales
import sys
import os
sys.path.append("..")
sys.path.append("../load_params")

from load_params.dominios import PriorityFlag, Profile, Pay, DayType, LaneType
from load_params.functions import (
    cargar_profit_desde_pickle,
    cargar_prob_items,
    cargar_tiempos_entre_llegadas,
    cargar_servicio_desde_pickle,
    cargar_tipo_cliente_hora_dia,
    cargar_prob_eleccion_caja
)
from paths import *

# Optimizaciones Numba
from optimizaciones_numba import ServiceTimeOptimizer, ProfitOptimizer


# =========================================
# FUNCIONES AUXILIARES DE CARGA
# =========================================

def cargar_paciencia_desde_excel(ruta_excel: str) -> Dict[Tuple[Profile, PriorityFlag, Pay, DayType], float]:
    """
    Carga paciencia desde CSV (no Excel, a pesar del nombre).
    Implementación correcta de main_rapido.ipynb que usa pd.read_csv()
    """
    df = pd.read_csv(ruta_excel)
    
    mapping = {}
    for _, row in df.iterrows():
        profile = row['profile']
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
        else:
            continue

        # Mapear prioridad
        prioridad_raw = str(row['priority']).strip().lower()
        if prioridad_raw == "pregnant":
            prio = PriorityFlag.P
        elif prioridad_raw == "senior":
            prio = PriorityFlag.S
        elif prioridad_raw == "reduced_mobility":
            prio = PriorityFlag.RM
        elif prioridad_raw == "no_priority":
            prio = PriorityFlag.NP
        else:
            continue

        # Mapear método de pago
        pago_raw = str(row['payment_method']).strip().lower()
        if pago_raw == "card":
            pay = Pay.CARD
        elif pago_raw == "cash":
            pay = Pay.CASH
        else:
            continue
        
        # Mapear valores por tipo de día usando DayType
        dia_map = {
            "mean_normal": DayType.NORMAL,
            "mean_oferta": DayType.OFERTA,
            "mean_domingo": DayType.DOMINGO
        }
        
        for col, dia_enum in dia_map.items():
            try:
                mean = float(row[col])
                mapping[(profile, prio, pay, dia_enum)] = mean
            except (ValueError, TypeError, KeyError):
                continue
    
    return mapping


# =========================================
# PARÁMETROS (Copiados de main_rapido.ipynb)
# =========================================

OPEN_HOUR = 8
CLOSE_HOUR = 22
HOURS = list(range(OPEN_HOUR, CLOSE_HOUR))
SEC_PER_HOUR = 3600

MAX_ITEMS = {
    LaneType.REGULAR: None,
    LaneType.EXPRESS: 10,
    LaneType.PRIORITY: None,
    LaneType.SELF: 15,
}

COSTOS_FIJOS = {
    LaneType.REGULAR: 8000000,
    LaneType.EXPRESS: 8000000,
    LaneType.PRIORITY: 8000000,
    LaneType.SELF: 25000000,
}

COSTO_CAJA_CLP_POR_HORA = {
    LaneType.REGULAR: 4500,
    LaneType.EXPRESS: 4500,
    LaneType.PRIORITY: 4700,
    LaneType.SELF: 0,
}

COSTO_SUPERVISORES_SELF_CLP_POR_HORA = 2 * 5000

COSTOS_OPERATIVOS_CAJA = {
    LaneType.REGULAR: 400,
    LaneType.EXPRESS: 420,
    LaneType.PRIORITY: 400,
    LaneType.SELF: 500,
}

COSTO_ANUAL_MANTENIMIENTO = {
    LaneType.REGULAR: 800000,
    LaneType.EXPRESS: 800000,
    LaneType.PRIORITY: 800000,
    LaneType.SELF: 1500000
}

Gastos_operacionales = {
    2025: 7230000000,
    2026: 7593900000,
    2027: 7979487000,
    2028: 8388248010,
    2029: 8821785555,
    2030: 9281827539
}

Tasa_impositiva = 0.27
Tasa_de_capital_propio = 0.0854

#caso demanda maxima en cada año
# PROYECCION_DEMANDA = {
#     2025: 1,
#     2026: 1.02352316,
#     2027: 1.04474234,
#     2028: 1.06280329,
#     2029: 1.07313915,
#     2030: 1.08347214
# }
PROYECCION_DEMANDA = {
    2025: 1,
    2026: 1.005073575,
    2027: 1.010349788,
    2028: 1.015839207,
    2029: 1.021206901,
    2030: 1.026550779
}


DISCOUNT_RATE_ANUAL = 0.0854
SL_THRESHOLD_SEC = 300

# Cargar datos globales (se hace UNA vez al inicio)
LAMBDA_POR_HORA = cargar_tiempos_entre_llegadas(PATH_ENTRE_LLEGADAS)
PATIENCE_MEAN_SEC = cargar_paciencia_desde_excel(os.path.join("..", "load_params", "paciencia_promedio_por_perfil_y_dia.csv"))

# PMF Cache
PMF_CACHE = {}
for profile in [Profile.EXPRESS, Profile.WEEKLY, Profile.FAMILY, Profile.DEAL, Profile.SELF, Profile.REGULAR]:
    for day in [DayType.NORMAL, DayType.OFERTA, DayType.DOMINGO]:
        path = os.path.join("..", "load_params", "KDE_PMF_por_items", "Probabilidades_")
        if profile == Profile.EXPRESS:
            path += "express_basket"
        elif profile == Profile.WEEKLY:
            path += "weekly_planner"
        elif profile == Profile.FAMILY:
            path += "family_cart"
        elif profile == Profile.DEAL:
            path += "deal_hunter"
        elif profile == Profile.SELF:
            path += "self_checkout_fan"
        else:
            path += "regular"

        if day == DayType.NORMAL:
            path += "_normal.xlsx"
        elif day == DayType.OFERTA:
            path += "_oferta.xlsx"
        elif day == DayType.DOMINGO:
            path += "_domingo.xlsx"
        PMF_CACHE[(profile, day)] = cargar_prob_items(path)


# ⭐ INICIALIZAR OPTIMIZADORES GLOBALES
print("🚀 Inicializando optimizadores...")
dic_con = cargar_servicio_desde_pickle(os.path.join("..", "load_params", "df_con_outliers.pkl"))
dic_sin = cargar_servicio_desde_pickle(os.path.join("..", "load_params", "df_sin_outliers.pkl"))
dic_profit = cargar_profit_desde_pickle()

SERVICE_OPTIMIZER = ServiceTimeOptimizer(dic_sin, dic_con)
PROFIT_OPTIMIZER = ProfitOptimizer(dic_profit)
print("✅ Optimizadores listos!\n")


# =========================================
# FUNCIONES AUXILIARES
# =========================================

def sample_items(profile: Profile, day: DayType) -> int:
    PMF = PMF_CACHE[(profile, day)]
    s_p = sum(PMF.values())
    for k in list(PMF.keys()):
        PMF[k] /= s_p
    r = random.random()
    acc = 0.0
    for items, p in PMF.items():
        acc += p
        if r <= acc:
            return items
    return max(PMF)


def sample_patience_sec(profile: Profile, prio: PriorityFlag, pay: Pay, tipo_dia: DayType) -> float:
    mean = PATIENCE_MEAN_SEC.get((profile, prio, pay, tipo_dia), 420.0)
    return float(np.random.exponential(mean))


def cargar_min_queue(path: str = None) -> Dict[Tuple[str, str, str], int]:
    if path is None:
        path = os.path.join("..", "load_params", "cola_minima_balked(in).csv")
    resultado = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            perfil_compuesto = row.get("profile")
            cola_str = row.get("cola")
            if perfil_compuesto and cola_str:
                partes = perfil_compuesto.split("__")
                if len(partes) == 3:
                    perfil, prioridad, pago = partes
                    try:
                        resultado[(perfil, prioridad, pago)] = int(cola_str)
                    except ValueError:
                        pass
    return resultado


# =========================================
# ENTIDADES (Copiadas de main_rapido.ipynb)
# =========================================

@dataclass
class Cliente:
    id: int
    profile: Profile
    pay: Pay
    prio: PriorityFlag
    items: int
    llegada_t: float
    patience_sec: float
    tiempo_por_caja: Dict[LaneType, float]


tiempo_asociado_a_un_cliente = {
    (Profile.REGULAR, LaneType.REGULAR): 149.47339015788987,
    (Profile.REGULAR, LaneType.PRIORITY): 143.79189539957093,
    (Profile.REGULAR, LaneType.SELF): 105.3969896850983,
    (Profile.REGULAR, LaneType.EXPRESS): 52.360457031035985,
    
    (Profile.SELF, LaneType.REGULAR): 92.97951835888938,
    (Profile.SELF, LaneType.PRIORITY): 83.8852613322624,
    (Profile.SELF, LaneType.SELF): 93.75709422559937,
    (Profile.SELF, LaneType.EXPRESS): 47.05567563408762,
    
    (Profile.FAMILY, LaneType.REGULAR): 315.0669854660576,
    (Profile.FAMILY, LaneType.PRIORITY): 302.9292454891063,
    (Profile.FAMILY, LaneType.SELF): 104.59328933402006,
    (Profile.FAMILY, LaneType.EXPRESS): 60.52369155182244,
    
    (Profile.EXPRESS, LaneType.REGULAR): 87.0224651298767,
    (Profile.EXPRESS, LaneType.PRIORITY): 75.66546390036837,
    (Profile.EXPRESS, LaneType.SELF): 87.68356113494846,
    (Profile.EXPRESS, LaneType.EXPRESS): 46.22434245302876,
    
    (Profile.DEAL, LaneType.REGULAR): 112.42154745915906,
    (Profile.DEAL, LaneType.PRIORITY): 102.7204222768139,
    (Profile.DEAL, LaneType.SELF): 96.87409698963425,
    (Profile.DEAL, LaneType.EXPRESS): 49.6623777247548,
    
    (Profile.WEEKLY, LaneType.REGULAR): 246.1820605537439,
    (Profile.WEEKLY, LaneType.PRIORITY): 238.3920777894574,
    (Profile.WEEKLY, LaneType.SELF): 108.81409602974574,
    (Profile.WEEKLY, LaneType.EXPRESS): 55.84692612007902
}

transformacion_profile = {
    "self_checkout_fan": Profile.SELF,
    "regular": Profile.REGULAR,
    "deal_hunter": Profile.DEAL,
    "weekly_planner": Profile.WEEKLY,
    "express_basket": Profile.EXPRESS,
    "family_cart": Profile.FAMILY
}
transformacion_profile_inverso = {v: k for k, v in transformacion_profile.items()}


class Lane:
    def __init__(self, env: simpy.Environment, lane_type: LaneType, count: int, horario: List[Tuple[float, float]]):
        self.env = env
        self.type = lane_type
        self.servers = simpy.Resource(env, capacity=count)
        self.open_count = count
        self.busy_time = 0.0
        self._last_change = env.now
        self.in_service = 0
        self.cola_actual = 0
        self.horario_apertura = horario

    def start_service(self):
        self.in_service += 1
        self.cola_actual = max(0, self.cola_actual - 1)

    def end_service(self, duration):
        self.in_service -= 1
        self.busy_time += duration

    def receiving_customers(self, hora_actual: float) -> bool:
        for inicio, fin in self.horario_apertura:
            if inicio <= hora_actual < fin:
                return True
        return False


class LanePool:
    def __init__(self, env: simpy.Environment, lane_type: LaneType, total_servers: int,
                 horarios: List[Tuple[int, Tuple[float, float]]], per_queue_cap: int = 5):
        assert total_servers >= 0
        self.env = env
        self.type = lane_type
        self.per_queue_cap = per_queue_cap
        self.total_servers = total_servers
        self.lanes: List[Lane] = []

        caja_id = 0
        horarios_dict = dict(horarios)
        remaining = total_servers

        while remaining > 0:
            cap = min(per_queue_cap, remaining)
            if caja_id not in horarios_dict:
                raise ValueError(f"No hay horario definido para caja {caja_id} de tipo {lane_type}")
            if horarios_dict[caja_id]:
                horario = horarios_dict[caja_id]
                self.lanes.append(Lane(env, lane_type, cap, horario))
                remaining -= cap
                caja_id += 1

        if total_servers == 0 and caja_id in horarios_dict:
            self.lanes.append(Lane(env, lane_type, 0, horario))

    def pick_lane(self, hora_actual) -> Lane:
        lanes_en_horario = [lane for lane in self.lanes if lane.receiving_customers(hora_actual)]
        if not lanes_en_horario:
            return None

        def load(l: Lane):
            return (l.servers.count + len(l.servers.queue)) / (self.total_servers / len(self.lanes))

        return min(lanes_en_horario, key=load)

    @property
    def cola_total(self) -> int:
        return sum(l.cola_actual for l in self.lanes)


# =========================================
# SIMULACIÓN PRINCIPAL (OPTIMIZADA)
# =========================================

class SupermarketSimOptimized:
    """Versión optimizada con Numba"""

    def __init__(self, configuracion, horarios, año, day: DayType, seed: Optional[int] = None):
        self.año = año
        self.day = day
        self.env = simpy.Environment()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.configuracion = configuracion
        self.horarios = horarios
        self.perdidas_por_ventas = 0

        self.lanes: Dict[LaneType, LanePool] = {
            lt: LanePool(self.env, lt, self.configuracion[lt], self.horarios[self.day][lt],
                        per_queue_cap=5 if lt == LaneType.SELF else 1) for lt in LaneType
        }

        # KPIs
        self.wait_count = 0
        self.wait_sum = 0
        self.served_flags = []
        self.patiences = []
        self.abandons = 0
        self.abandons_specified = {LaneType.REGULAR: 0, LaneType.SELF: 0,
                                   LaneType.EXPRESS: 0, LaneType.PRIORITY: 0}
        self.total_customers = 0
        self.service_level_hits = 0
        self.revenue = 0.0
        self.operating_cost = 0.0

        self._cid = 0
        self.time_generator = 0.0
        self.time_new_customer = 0.0
        self.time_elegir_caja = 0.0
        self.time_proceso_cliente_preyield = 0.0
        self.time_proceso_cliente_postyield = 0.0

        self._probabilidades_caja = cargar_prob_eleccion_caja()
        self._cola_minima_por_perfil = cargar_min_queue()

        self.clientes_pre_generados = self._pre_generar_clientes()
        self._cliente_index = 0

        self.tiempo_servicio = {LaneType.REGULAR: 0, LaneType.SELF: 0,
                               LaneType.EXPRESS: 0, LaneType.PRIORITY: 0}
        self.clientes_caja = {LaneType.REGULAR: 0, LaneType.SELF: 0,
                             LaneType.EXPRESS: 0, LaneType.PRIORITY: 0}

    def _pre_generar_clientes(self) -> List[Cliente]:
        clientes = []
        cliente_id = 0

        for hour in HOURS:
            tiempo_entre_llegadas = LAMBDA_POR_HORA[self.day].get(hour, 3600.0) / PROYECCION_DEMANDA[self.año]

            if tiempo_entre_llegadas <= 0 or tiempo_entre_llegadas > 3600:
                continue

            interarrivals = np.random.exponential(tiempo_entre_llegadas, size=1000)
            times = np.cumsum(interarrivals)
            valid_times = times[times < SEC_PER_HOUR]

            tiempo_base_hora = (hour - OPEN_HOUR) * SEC_PER_HOUR

            for t in valid_times:
                tiempo_llegada = tiempo_base_hora + t
                cliente = self._new_customer(tiempo_llegada, hour, cliente_id)
                clientes.append(cliente)
                cliente_id += 1

        clientes.sort(key=lambda x: x.llegada_t)
        return clientes

    def _new_customer(self, t_llegada: float, hour: int, cliente_id: int) -> Cliente:
        dicc = cargar_tipo_cliente_hora_dia(self.day, hour)
        r = random.random()
        acc = 0.0
        for items, p in dicc.items():
            acc += p
            if r <= acc:
                cliente = items
                break
        cliente = items.split("__")

        transformacion_method = {"cash": Pay.CASH, "card": Pay.CARD}
        transformacion_priority = {"pregnant": PriorityFlag.P, "no_priority": PriorityFlag.NP,
                                   "senior": PriorityFlag.S, "reduced_mobility": PriorityFlag.RM}

        profile = transformacion_profile[cliente[0]]
        pay = transformacion_method[cliente[2]]
        prio = transformacion_priority[cliente[1]]
        items = sample_items(profile, self.day)
        patience = sample_patience_sec(profile, prio, pay, self.day)
        self.patiences.append(patience)
        
        # Pre-calcular tiempos de servicio para cada tipo de caja
        tiempos_por_caja = {}
        for lt in LaneType:
            max_it = MAX_ITEMS.get(lt)
            if max_it is not None and items > max_it:
                continue
            if prio == PriorityFlag.NP and lt == LaneType.PRIORITY:
                continue
            if lt == LaneType.SELF and pay == Pay.CASH:
                continue
            if lt == LaneType.SELF and prio == PriorityFlag.RM:
                continue

            try:
                tiempo = SERVICE_OPTIMIZER.get_service_time(lt, items, pay)
                tiempos_por_caja[lt] = tiempo
            except ValueError:
                continue

        return Cliente(cliente_id, profile, pay, prio, items, t_llegada, patience, tiempos_por_caja)

    def generator_llegadas_optimizado(self):
        for cliente in self.clientes_pre_generados:
            yield self.env.timeout(cliente.llegada_t - self.env.now)
            self.env.process(self.proceso_cliente(cliente))

    def elegir_caja(self, c: Cliente) -> Optional[List[LaneType]]:
        elegibles: List[LaneType] = []
        for lt in LaneType:
            max_it = MAX_ITEMS[lt]
            if (max_it is not None) and (c.items > max_it):
                continue
            if c.prio == PriorityFlag.NP and lt == LaneType.PRIORITY:
                continue
            if lt == LaneType.SELF and c.pay == Pay.CASH:
                continue
            if lt == LaneType.SELF and c.prio == PriorityFlag.RM:
                continue
            if self.horarios[self.day][lt] == []:
                continue
            elegibles.append(lt)

        if not elegibles:
            return None

        profile_str = c.profile.value
        priority_map = {PriorityFlag.P: "pregnant", PriorityFlag.S: "senior",
                       PriorityFlag.RM: "reduced_mobility", PriorityFlag.NP: "no_priority"}
        priority_str = priority_map.get(c.prio, "no_priority")
        payment_str = "card" if c.pay == Pay.CARD else "cash"

        key = (profile_str, priority_str, payment_str)
        umbral_cola = self._cola_minima_por_perfil.get(key)
        if umbral_cola is None:
            umbral_cola = 7

        if umbral_cola is not None:
            todas_superan = True
            for lt in elegibles:
                pool = self.lanes.get(lt)
                if pool is None:
                    continue
                for lane in pool.lanes:
                    if lane.receiving_customers(self.env.now / 60) and lane.cola_actual < umbral_cola:
                        todas_superan = False
                        break
            if todas_superan:
                return None

        return elegibles

    def proceso_cliente(self, c: Cliente):
        self.total_customers += 1
        chosen = self.elegir_caja(c)
        if chosen is None:
            self.served_flags.append(False)
            self.abandons += 1
            return

        def load(l: Lane, cliente):
            if l.servers.count <= l.servers.capacity and len(l.servers.queue) == 0:
                a = l.servers.count / l.servers.capacity
                return a
            else:
                a = 1
                return (a + ((len(l.servers.queue) / l.servers.capacity))) * tiempo_asociado_a_un_cliente[(c.profile, l.type)]

        orden_prioridad = {LaneType.EXPRESS: 1, LaneType.SELF: 4,
                          LaneType.REGULAR: 2, LaneType.PRIORITY: 3}

        def load_y_desempate(lane):
            carga = load(lane, c)
            prioridad = orden_prioridad.get(lane.type, 5)
            if carga == 0 or carga == 1:
                return (carga, random.random())  # random.random() da número entre 0 y 1
            else:
                return (carga, prioridad)  # Primero por carga, luego por prioridad

        a_elegir = []
        for lanetype in chosen:
            lane = self.lanes[lanetype].pick_lane(self.env.now / 60)
            a_elegir.append(lane)
        lane = min(a_elegir, key=lambda lane: load_y_desempate(lane))
        self.clientes_caja[lane.type] += 1

        if lane is None:
            self.served_flags.append(False)
            self.abandons += 1
            return

        lane.cola_actual += 1
        arrival_to_queue = c.llegada_t

        with lane.servers.request() as req:
            res = yield req | self.env.timeout(c.patience_sec, value="timeout")

            if req not in res:
                self.served_flags.append(False)
                self.abandons_specified[lane.type] += 1
                self.abandons += 1
                lane.cola_actual = max(0, lane.cola_actual - 1)

                # ⭐ Usar optimizador Numba
                dias = {DayType.NORMAL: 'normal', DayType.DOMINGO: 'domingo', DayType.OFERTA: 'oferta'}
                hour = int(self.env.now / 3600) + 8
                key = transformacion_profile_inverso[c.profile] + "_" + dias[self.day] + "_" + str(hour)
                self.perdidas_por_ventas += PROFIT_OPTIMIZER.get_ingreso(key, c.items)
                return

            wait = self.env.now - arrival_to_queue
            self.wait_count += 1
            self.wait_sum += wait
            if wait <= SL_THRESHOLD_SEC:
                self.service_level_hits += 1

            lane.start_service()

            # ⭐ Usar tiempo pre-calculado (más rápido que calcular en tiempo real)
            st = c.tiempo_por_caja.get(lane.type, 0.0)
            st = max(0, st)
            self.tiempo_servicio[lane.type] += st

            yield self.env.timeout(st)
            lane.end_service(st)

            # ⭐ Usar optimizador Numba
            dias = {DayType.NORMAL: 'normal', DayType.DOMINGO: 'domingo', DayType.OFERTA: 'oferta'}
            hour = int(self.env.now / 3600) + 8
            key = transformacion_profile_inverso[c.profile] + "_" + dias[self.day] + "_" + str(hour)
            self.revenue += PROFIT_OPTIMIZER.get_ingreso(key, c.items)
            self.served_flags.append(True)

    def _costos_operativos_del_dia(self):
        """
        Calcula costos operativos del día (sin inversión inicial).
        La inversión inicial se maneja por separado en el VAN.
        """
        horas_operacion = CLOSE_HOUR - OPEN_HOUR
        costo = 0.0
        self.costo_operacion = 0
        self.costo_mantenimiento_diario = 0

        # Solo costos operacionales y mantenimiento diario (NO inversión inicial)
        for lt, n in self.horarios[self.day].items():
            # Mantenimiento anual diluido por día (CORRECTO)
            self.costo_mantenimiento_diario += COSTO_ANUAL_MANTENIMIENTO[lt] / 365
            for caja, horario in n:
                minutos_operados = 0
                for ventana in horario:
                    minutos_operados += ventana[1] - ventana[0]
                horas_operadas = minutos_operados / 60
                # Costos operacionales por hora (CORRECTO)
                self.costo_operacion += horas_operadas * (COSTO_CAJA_CLP_POR_HORA[lt] + COSTOS_OPERATIVOS_CAJA[lt])

        # Supervisores SCO (si aplica)
        if self.configuracion[LaneType.SELF] > 0:
            cantidad_islas = math.ceil(self.configuracion[LaneType.SELF] / 5)
            self.costo_operacion += COSTO_SUPERVISORES_SELF_CLP_POR_HORA * horas_operacion * cantidad_islas

        # Costos operativos del día (sin inversión inicial)
        costo += self.costo_mantenimiento_diario + self.costo_operacion
        costo += Gastos_operacionales[self.año] / 365
        
        # Calcular impuestos sobre UTILIDAD (revenue - costos operativos)
        utilidad_antes_impuestos = self.revenue - costo
        self.impuestos = max(0, utilidad_antes_impuestos * Tasa_impositiva)  # Solo impuestos si hay utilidad
        
        # Agregar impuestos al costo total
        costo += self.impuestos
        return costo

    def calcular_van_correcto(self):
        """
        Calcula el VAN con estructura temporal correcta:
        t=0: Inversión inicial (CAPEX)
        t=1-5: Flujos anuales considerando multiplicadores por tipo de día
        """
        # t=0: Inversión inicial (CAPEX)
        inversion_inicial = 0
        for lt, num_cajas in self.configuracion.items():
            if num_cajas > 0:
                inversion_inicial += COSTOS_FIJOS[lt] * num_cajas
        
        # VAN del día actual (revenue - costos operativos)
        van_dia = self.revenue - self.operating_cost
        
        # Aplicar multiplicador según tipo de día
        multiplicador_dia = 3 if self.day in [DayType.NORMAL, DayType.OFERTA] else 1
        van_dia_ponderado = van_dia * multiplicador_dia
        
        # Proyectar a flujo anual: (3×NORMAL + 3×OFERTA + 1×DOMINGO) × 52 semanas
        # Como cada simulación representa un tipo de día, multiplicamos por 52
        flujo_anual = van_dia_ponderado * 52
        
        # Calcular VAN con descuento
        van_total = -inversion_inicial  # t=0
        
        for año in range(1, 6):  # t=1 a t=5
            van_total += flujo_anual / ((1 + DISCOUNT_RATE_ANUAL) ** año)
        
        return van_total, inversion_inicial, flujo_anual

    def run(self):
        self.env.process(self.generator_llegadas_optimizado())
        sim_time = (CLOSE_HOUR - OPEN_HOUR) * SEC_PER_HOUR
        self.env.run(until=sim_time)
        self.operating_cost = self._costos_operativos_del_dia()

    def kpis(self) -> Dict[str, float]:
        served = sum(1 for x in self.served_flags if x)
        avg_wait = (self.wait_sum / self.wait_count) if self.wait_count else 0.0
        abandon_rate = self.abandons / self.total_customers if self.total_customers else 0.0
        sl = (self.service_level_hits / served) if served else 0.0
        flujo = self.revenue - self.operating_cost
        
        # Calcular VAN correcto con estructura temporal
        van_correcto, inversion_inicial, flujo_anual = self.calcular_van_correcto()

        return {
            "clientes_totales": int(self.total_customers),
            "atendidos": int(served),
            "abandono_rate": float(abandon_rate),
            "wait_avg_sec": float(avg_wait),
            "service_level_%<=300s": float(sl * 100),
            "ingresos_clp": int(self.revenue),
            "costos_clp": int(self.operating_cost),
            "VAN_dia_clp": int(flujo),
            "VAN_correcto_5_anios": int(van_correcto),
            "Inversion_inicial": int(inversion_inicial),
            "Flujo_anual_proyectado": int(flujo_anual),
            "Asignados REGULAR": int(self.clientes_caja[LaneType.REGULAR]),
            "Asignados SELF": int(self.clientes_caja[LaneType.SELF]),
            "Asignados EXPRESS": int(self.clientes_caja[LaneType.EXPRESS]),
            "Asignados PRIORITY": int(self.clientes_caja[LaneType.PRIORITY]),
            "Perdidas por ventas no concretadas": int(self.perdidas_por_ventas),
            "Impuestos": int(self.impuestos),
            "Costos_mantenimiento_diario": int(self.costo_mantenimiento_diario),
            "Costos_operacion_diario": int(self.costo_operacion),
        }


# =========================================
# FUNCIONES DE SIMULACIÓN
# =========================================

def simular_un_dia(day: DayType, CONFIG, HORARIOS, AÑO, seed: Optional[int] = 42) -> Dict[str, float]:
    """Simula un día con configuración dada"""
    sim = SupermarketSimOptimized(CONFIG, HORARIOS, AÑO, day=day, seed=seed)
    sim.run()
    return sim.kpis()


def simular_varios(dia: DayType, CONFIG, HORARIOS, AÑO, n_rep: int = 10, seed_base: int = 123) -> Dict[str, float]:
    """Simula múltiples réplicas y promedia"""
    resultados = []
    for r in range(n_rep):
        k = simular_un_dia(dia, CONFIG, HORARIOS, AÑO, seed=seed_base + r)
        resultados.append(k)
        seed_base += 1
    keys = resultados[0].keys() if resultados else []
    return {k: float(np.mean([d[k] for d in resultados])) for k in keys}


# =========================================
# INTEGRACIÓN CON RAY (PARALELIZACIÓN)
# =========================================

# Inicializar Ray
ray.init(num_cpus=8, ignore_reinit_error=True)


@ray.remote
def simular_un_dia_remote(day: DayType, CONFIG, HORARIOS, AÑO, seed: int) -> Dict[str, float]:
    """Versión remota para Ray"""
    return simular_un_dia(day, CONFIG, HORARIOS, AÑO, seed)


def simular_varios_paralelo(dia: DayType, CONFIG, HORARIOS, AÑO, n_rep: int = 10,
                            seed_base: int = 123) -> Dict[str, float]:
    """
    Simula múltiples réplicas EN PARALELO usando Ray.
    
    Speedup: 8x con 8 cores
    """
    # Lanzar todas las réplicas en paralelo
    futures = [
        simular_un_dia_remote.remote(dia, CONFIG, HORARIOS, AÑO, seed_base + r)
        for r in range(n_rep)
    ]
    
    # Esperar resultados
    resultados = ray.get(futures)
    
    # Promediar
    keys = resultados[0].keys()
    return {k: float(np.mean([d[k] for d in resultados])) for k in keys}


def simular_configuracion_completa(CONFIG, HORARIOS, AÑO, n_rep: int = 5) -> Dict[str, float]:
    """
    Simula los 3 tipos de día en paralelo.
    
    Args:
        CONFIG: Configuración de cajas
        HORARIOS: Horarios de cajas
        AÑO: Año de proyección
        n_rep: Réplicas por tipo de día
    
    Returns:
        Dict con KPIs agregados
    """
    # Simular los 3 tipos de día en paralelo
    futures = []
    for day in [DayType.NORMAL, DayType.OFERTA, DayType.DOMINGO]:
        for rep in range(n_rep):
            futures.append(simular_un_dia_remote.remote(day, CONFIG, HORARIOS, AÑO, 1000 + rep))
    
    # Esperar todos los resultados
    resultados = ray.get(futures)
    
    # Separar por tipo de día
    kpis_por_dia = {
        DayType.NORMAL: resultados[0:n_rep],
        DayType.OFERTA: resultados[n_rep:2*n_rep],
        DayType.DOMINGO: resultados[2*n_rep:3*n_rep]
    }
    
    # Calcular VAN total
    van_total = 0
    for day, kpis_list in kpis_por_dia.items():
        van_promedio = np.mean([k['VAN_dia_clp'] for k in kpis_list])
        multiplicador = 3 if day in [DayType.NORMAL, DayType.OFERTA] else 1
        van_total += van_promedio * multiplicador
    
    return {
        'VAN_total': van_total,
        'kpis_por_dia': kpis_por_dia
    }


# =========================================
# EJEMPLO DE USO
# =========================================

# Configuración óptima (34 cajas regulares)
CONFIG_CAJA = {
    LaneType.REGULAR: 34,
    LaneType.EXPRESS: 0,
    LaneType.PRIORITY: 0,
    LaneType.SELF: 0,
}

horario_completo = (0, 840)
HORARIOS_CAJA_NORMAL = {
    LaneType.REGULAR: [(i, [horario_completo]) for i in range(CONFIG_CAJA[LaneType.REGULAR])],
    LaneType.EXPRESS: [(i, [horario_completo]) for i in range(CONFIG_CAJA[LaneType.EXPRESS])],
    LaneType.PRIORITY: [(i, [horario_completo]) for i in range(CONFIG_CAJA[LaneType.PRIORITY])],
    LaneType.SELF: [(i, [horario_completo]) for i in range(CONFIG_CAJA[LaneType.SELF])],
}
HORARIOS_CAJA_OFERTA = {
    LaneType.REGULAR: [(i, [horario_completo]) for i in range(CONFIG_CAJA[LaneType.REGULAR])],
    LaneType.EXPRESS: [(i, [horario_completo]) for i in range(CONFIG_CAJA[LaneType.EXPRESS])],
    LaneType.PRIORITY: [(i, [horario_completo]) for i in range(CONFIG_CAJA[LaneType.PRIORITY])],
    LaneType.SELF: [(i, [horario_completo]) for i in range(CONFIG_CAJA[LaneType.SELF])],
}
HORARIOS_CAJA_DOMINGO = {
    LaneType.REGULAR: [(i, [horario_completo]) for i in range(CONFIG_CAJA[LaneType.REGULAR])],
    LaneType.EXPRESS: [(i, [horario_completo]) for i in range(CONFIG_CAJA[LaneType.EXPRESS])],
    LaneType.PRIORITY: [(i, [horario_completo]) for i in range(CONFIG_CAJA[LaneType.PRIORITY])],
    LaneType.SELF: [(i, [horario_completo]) for i in range(CONFIG_CAJA[LaneType.SELF])],
}
HORARIOS_CAJA = {
    DayType.NORMAL: HORARIOS_CAJA_NORMAL,
    DayType.OFERTA: HORARIOS_CAJA_OFERTA,
    DayType.DOMINGO: HORARIOS_CAJA_DOMINGO
}
AÑO = 2025

if __name__ == "__main__":

    # VAN de configuración original (15,3,2,5) para comparación
    CONFIG_ORIGINAL = {
        LaneType.REGULAR: 15,
        LaneType.EXPRESS: 3,
        LaneType.PRIORITY: 2,
        LaneType.SELF: 5,
    }
    
    print("="*80)
    print("📊 ANÁLISIS COMPARATIVO DE CONFIGURACIONES")
    print("="*80)
    
    # Crear horarios para configuración original
    def crear_horarios_para_config(config):
        horarios = {}
        for dia in [DayType.DOMINGO, DayType.NORMAL, DayType.OFERTA]:
            horarios[dia] = {}
            for tipo in LaneType:
                num_cajas = config[tipo]
                if num_cajas > 0:
                    horarios[dia][tipo] = [(i, [horario_completo]) for i in range(num_cajas)]
                else:
                    horarios[dia][tipo] = []
        return horarios
    
    HORARIOS_ORIGINAL = crear_horarios_para_config(CONFIG_ORIGINAL)
    
    # Simular configuración original
    print("\n🏗️ CONFIGURACIÓN ORIGINAL (15,3,2,5):")
    van_original_total = 0
    van_original_por_dia = {}
    
    for dt in [DayType.DOMINGO, DayType.NORMAL, DayType.OFERTA]:
        kpi = simular_varios(dt, CONFIG_ORIGINAL, HORARIOS_ORIGINAL, AÑO, n_rep=20, seed_base=1002)
        van_dia = kpi['VAN_dia_clp']
        van_correcto = kpi.get('VAN_correcto_5_anios', 0)
        multiplicador = 3 if dt in [DayType.NORMAL, DayType.OFERTA] else 1
        van_ponderado = van_dia * multiplicador
        van_original_total += van_ponderado
        van_original_por_dia[dt] = van_dia
        
        print(f"  {dt.value}: VAN_dia=${van_dia:,.0f} (ponderado: ${van_ponderado:,.0f}) | VAN_5años=${van_correcto:,.0f}")
    
    print(f"  📊 VAN TOTAL ORIGINAL: ${van_original_total:,.0f}")
    
    # Simular configuración actual
    print(f"\n🏗️ CONFIGURACIÓN ACTUAL ({CONFIG_CAJA[LaneType.REGULAR]},{CONFIG_CAJA[LaneType.EXPRESS]},{CONFIG_CAJA[LaneType.PRIORITY]},{CONFIG_CAJA[LaneType.SELF]}):")
    van_actual_total = 0
    van_actual_por_dia = {}
    
    for dt in [DayType.DOMINGO, DayType.NORMAL, DayType.OFERTA]:
        kpi = simular_varios(dt, CONFIG_CAJA, HORARIOS_CAJA, AÑO, n_rep=20, seed_base=1002)
        van_dia = kpi['VAN_dia_clp']
        van_correcto = kpi.get('VAN_correcto_5_anios', 0)
        multiplicador = 3 if dt in [DayType.NORMAL, DayType.OFERTA] else 1
        van_ponderado = van_dia * multiplicador
        van_actual_total += van_ponderado
        van_actual_por_dia[dt] = van_dia
        
        print(f"  {dt.value}: VAN_dia=${van_dia:,.0f} (ponderado: ${van_ponderado:,.0f}) | VAN_5años=${van_correcto:,.0f}")
    
    print(f"  📊 VAN TOTAL ACTUAL: ${van_actual_total:,.0f}")
    
    # Comparación
    diferencia = van_actual_total - van_original_total
    mejora_porcentual = (diferencia / abs(van_original_total)) * 100 if van_original_total != 0 else 0
    
    print(f"\n💰 COMPARACIÓN:")
    print(f"  VAN Original: ${van_original_total:,.0f}")
    print(f"  VAN Actual:   ${van_actual_total:,.0f}")
    print(f"  Diferencia:   ${diferencia:,.0f}")
    print(f"  Mejora:      {mejora_porcentual:+.1f}%")
    
    print("="*80)

    ray.shutdown()

