"""
Heurística de 3 Etapas para Optimización de Cajas de Supermercado
================================================================

Implementa Simulated Annealing con 3 fases jerárquicas:
1. Estratégica: Decidir qué cajas construir y cuántas
2. Táctica: Decidir qué cajas activar cada año
3. Operacional: Decidir horarios de apertura/cierre

Integra con la simulación optimizada (main_rapido_optimizado.py) para evaluación rápida.
"""

import sys
import os
import time
import random
import math
import copy
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Agregar paths
sys.path.append("..")
sys.path.append("../load_params")
sys.path.append("../Heurística")

# Imports locales
from load_params.dominios import LaneType, DayType
from load_params.functions import (
    cargar_profit_desde_pickle,
    cargar_prob_items,
    cargar_tiempos_entre_llegadas,
    cargar_servicio_desde_pickle,
    cargar_tipo_cliente_hora_dia,
    cargar_prob_eleccion_caja
)

# Importar simulación optimizada
from main_rapido_optimizado import (
    simular_un_dia, simular_varios, simular_configuracion_completa,
    CONFIG_CAJA, HORARIOS_CAJA, AÑO
)

# Importar estructuras de datos y operadores
from estructuras_datos import (
    ConfigEstrategica, ConfigTactica, ConfigOperacional, SolucionCompleta,
    crear_config_actual, ajustar_tactica_a_estrategica, ajustar_horarios_a_capacidad
)
from operadores_vecindad import generar_vecino


# ============================================
# ESTRUCTURAS DE DATOS ADAPTADAS
# ============================================

@dataclass
class ConfiguracionInicial:
    """Configuración inicial del supermercado"""
    config_caja: Dict[LaneType, int]
    horarios_caja: Dict[DayType, Dict[LaneType, List[Tuple[int, List[Tuple[float, float]]]]]]
    año: int
    
    def __post_init__(self):
        # Validar que la configuración sea factible
        total_cajas = sum(self.config_caja.values())
        if total_cajas > 40:
            raise ValueError(f"Total de cajas ({total_cajas}) excede el límite de 40")
        
        # Validar que self-checkout sea múltiplo de 5
        if self.config_caja[LaneType.SELF] % 5 != 0:
            raise ValueError(f"Self-checkout ({self.config_caja[LaneType.SELF]}) debe ser múltiplo de 5")


@dataclass
class ResultadoOptimizacion:
    """Resultado de la optimización"""
    config_optima: SolucionCompleta
    van_optimo: float
    van_inicial: float
    mejora_porcentual: float
    tiempo_ejecucion: float
    historial: List[Dict]
    kpis_finales: Dict


# ============================================
# PARÁMETROS DEL ALGORITMO
# ============================================

PARAMETROS_SA = {
    "estrategica": {
        "T_inicial": 1000,
        "alpha": 0.95,
        "T_min": 10,
        "prob_greedy": 0.60,
        "iter_max": 1
    },
    "tactica": {
        "T_inicial": 500,
        "alpha": 0.95,
        "T_min": 5,
        "prob_greedy": 0.90,
        "iter_max": 1
    },
    "operacional": {
        "T_inicial": 200,
        "alpha": 0.97,
        "T_min": 1,
        "prob_greedy": 0.80,
        "iter_max": 1500
    }
}

# Parámetros del algoritmo pendular
MAX_CICLOS = 5
TOL_CONVERGENCIA = 0.02
N_REPLICAS_EVALUACION = 20


# ============================================
# FUNCIONES DE CONVERSIÓN
# ============================================

def config_inicial_a_estrategica(config_inicial: ConfiguracionInicial) -> ConfigEstrategica:
    """Convierte configuración inicial a ConfigEstrategica"""
    return ConfigEstrategica(cajas_por_tipo=config_inicial.config_caja.copy())


def config_inicial_a_tactica(config_inicial: ConfiguracionInicial) -> ConfigTactica:
    """Convierte configuración inicial a ConfigTactica (todas las cajas activas desde año 0)"""
    cajas_por_anio = {}
    for anio in range(5):
        cajas_por_anio[anio] = config_inicial.config_caja.copy()
    
    return ConfigTactica(
        cajas_por_anio=cajas_por_anio,
        config_estrategica=config_inicial_a_estrategica(config_inicial)
    )


def config_inicial_a_operacional(config_inicial: ConfiguracionInicial) -> ConfigOperacional:
    """Convierte configuración inicial a ConfigOperacional"""
    horarios = {}
    
    for dia in DayType:
        horarios[dia] = {}
        for hora in range(8, 22):
            horarios[dia][hora] = {}
            for tipo in LaneType:
                # Convertir horarios de la configuración inicial
                cajas_abiertas = []
                if tipo in config_inicial.horarios_caja[dia]:
                    for caja_id, horario_caja in config_inicial.horarios_caja[dia][tipo]:
                        # Verificar si la caja está abierta en esta hora
                        hora_actual = (hora - 8) * 60
                        for inicio, fin in horario_caja:
                            if inicio <= hora_actual < fin:
                                cajas_abiertas.append(caja_id)
                                break
                horarios[dia][hora][tipo] = cajas_abiertas
    
    return ConfigOperacional(
        horarios=horarios,
        config_tactica=config_inicial_a_tactica(config_inicial),
        anio_actual=0
    )


def solucion_completa_a_config_simulacion(solucion: SolucionCompleta, año: int = 0) -> Tuple[Dict, Dict]:
    """Convierte SolucionCompleta a formato de simulación"""
    # Configuración de cajas para el año específico
    detallada = solucion.operacional
    config_caja = solucion.tactica.cajas_por_anio[año]
    
    # Transformar horarios al formato requerido
    horarios = {}
    for dia, horas_dict in detallada.horarios.items():
        horarios[dia] = {}
        
        # Primero, recolectar todos los horarios por tipo de línea y caja
        horarios_por_tipo_caja = {}
        
        for hora, tipos_dict in horas_dict.items():
            # Convertir hora a minutos desde medianoche
            hora_en_minutos = (hora - 8) * 60
            
            for tipo_lane, caja_ids in tipos_dict.items():
                if tipo_lane not in horarios_por_tipo_caja:
                    horarios_por_tipo_caja[tipo_lane] = {}
                
                for caja_id in caja_ids:
                    if caja_id not in horarios_por_tipo_caja[tipo_lane]:
                        horarios_por_tipo_caja[tipo_lane][caja_id] = []
                    horarios_por_tipo_caja[tipo_lane][caja_id].append(hora_en_minutos)
        
        # Ahora convertir las listas de minutos a intervalos de apertura/cierre
        for tipo_lane, cajas_dict in horarios_por_tipo_caja.items():
            horarios[dia][tipo_lane] = []
            
            for caja_id, minutos_list in cajas_dict.items():
                # Ordenar los minutos y agrupar en intervalos consecutivos
                minutos_list.sort()
                intervalos = []
                
                if minutos_list:
                    inicio = minutos_list[0]
                    fin = minutos_list[0] + 60
                    
                    for i in range(1, len(minutos_list)):
                        if minutos_list[i] == minutos_list[i-1] + 60:  # Diferencia de 60 minutos (1 hora)
                            fin = minutos_list[i] + 60
                        else:
                            intervalos.append((inicio, fin))
                            inicio = minutos_list[i]
                            fin = minutos_list[i] + 60
                    
                    intervalos.append((inicio, fin))
                
                horarios[dia][tipo_lane].append((caja_id, intervalos))
    
    return config_caja, horarios


# ============================================
# FUNCIONES DE EVALUACIÓN
# ============================================

def evaluar_configuracion_simulacion(solucion: SolucionCompleta, año: int = 0, n_rep: int = 20) -> Dict:
    """
    Evalúa una configuración usando la simulación optimizada.
    
    Args:
        solucion: SolucionCompleta a evaluar
        año: Año de evaluación (0-4)
        n_rep: Número de réplicas
    
    Returns:
        Dict con KPIs de la simulación
    """
    config_caja, horarios = solucion_completa_a_config_simulacion(solucion, año)
    
    # Simular los 3 tipos de día
    van_total = 0
    kpis_por_dia = {}
    
    for dia in [DayType.NORMAL, DayType.OFERTA, DayType.DOMINGO]:
        kpi = simular_varios(dia, config_caja, horarios, 2025 + año, n_rep=n_rep)
        kpis_por_dia[dia] = kpi
        
        # Multiplicador según tipo de día
        multiplicador = 3 if dia in [DayType.NORMAL, DayType.OFERTA] else 1
        # Usar VAN correcto de 5 años en lugar de VAN del día
        van_correcto = kpi.get('VAN_correcto_5_anios', kpi['VAN_dia_clp'])
        van_total += van_correcto * multiplicador
    
    return {
        'VAN_total': van_total,
        'kpis_por_dia': kpis_por_dia
    }


def evaluar_fase_simulacion(config, fase: str, año: int = 0, n_rep: int = 10) -> float:
    """
    Evalúa una configuración de una fase específica usando simulación.
    
    Args:
        config: ConfigEstrategica | ConfigTactica | ConfigOperacional
        fase: "estrategica" | "tactica" | "operacional"
        año: Año de evaluación
        n_rep: Número de réplicas
    
    Returns:
        VAN (negativo para minimizar en SA)
    """
    if fase == "estrategica":
        # Crear solución completa con configuración estratégica
        tactica = ConfigTactica(
            cajas_por_anio={a: config.cajas_por_tipo.copy() for a in range(5)},
            config_estrategica=config
        )
        operacional = ConfigOperacional(
            horarios={},  # Se llenará automáticamente
            config_tactica=tactica,
            anio_actual=año
        )
        for dia in DayType: #partimos con todas las cajas abiertas
            operacional.horarios[dia] = {}
            for hora in range(8, 22):
                operacional.horarios[dia][hora] = {}
                for tipo in LaneType:
                    num_cajas = tactica.cajas_por_anio[año][tipo]
                    operacional.horarios[dia][hora][tipo] = list(range(num_cajas))
        solucion = SolucionCompleta(estrategica=config, tactica=tactica, operacional=operacional)
    
    elif fase == "tactica":
        # Crear solución completa con configuración táctica
        operacional = ConfigOperacional(
            horarios={},  # Se llenará automáticamente
            config_tactica=config,
            anio_actual=año
        )
        # Inicializar horarios: todas las cajas abiertas 8:00-22:00
        for dia in DayType:
            operacional.horarios[dia] = {}
            for hora in range(8, 22):
                operacional.horarios[dia][hora] = {}
                for tipo in LaneType:
                    num_cajas = config.cajas_por_anio[año][tipo]
                    operacional.horarios[dia][hora][tipo] = list(range(num_cajas))
        solucion = SolucionCompleta(
            estrategica=config.config_estrategica,
            tactica=config,
            operacional=operacional
        )
    
    elif fase == "operacional":
        # Crear solución completa con configuración operacional
        solucion = SolucionCompleta(
            estrategica=config.config_tactica.config_estrategica,
            tactica=config.config_tactica,
            operacional=config
        )
    
    else:
        raise ValueError(f"Fase desconocida: {fase}")
    
    # Evaluar con simulación
    resultado = evaluar_configuracion_simulacion(solucion, año, n_rep)
    return -resultado['VAN_total']  # Negativo para minimizar


# ============================================
# ALGORITMO SA POR FASE
# ============================================

def SA_fase_simulacion(config_inicial, fase: str, año: int = 0, verbose: bool = True) -> Tuple:
    """
    Simulated Annealing para una fase específica usando simulación.
    
    Args:
        config_inicial: Configuración inicial de la fase
        fase: "estrategica" | "tactica" | "operacional"
        año: Año de evaluación
        verbose: Si imprimir progreso
    
    Returns:
        (mejor_config, mejor_costo, historial)
    """
    params = PARAMETROS_SA[fase]
    
    # Estado inicial
    S_actual = config_inicial.copy()
    S_mejor = config_inicial.copy()
    
    costo_actual = evaluar_fase_simulacion(S_actual, fase, año, n_rep=10)  # Pocas réplicas para SA
    costo_mejor = costo_actual
    
    # Temperatura
    T = params["T_inicial"]
    T_min = params["T_min"]
    alpha = params["alpha"]
    
    # Historial
    historial = {
        "costos": [costo_actual],
        "temperaturas": [T],
        "aceptaciones": 0,
        "rechazos": 0
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SA - FASE {fase.upper()}")
        print(f"{'='*60}")
        print(f"Config inicial: costo={costo_actual:.2e}")
        print(f"Parámetros: T_ini={T}, alpha={alpha}, iter_max={params['iter_max']}")
    
    t_inicio = time.time()
    
    for iteracion in range(1, params["iter_max"] + 1):
        # Generar vecino
        if random.random() < params["prob_greedy"]:
            # Local: genera lista de vecinos y elige el mejor
            vecinos = generar_vecino(S_actual, fase, "local")
            
            if not vecinos:
                continue
            
            # Evaluar todos los vecinos y elegir el mejor
            mejor_vecino = None
            mejor_costo_vecino = float('inf')
            print("ACTUAL", costo_actual)
            
            for vecino in vecinos:
                print(vecino)
                costo_v = evaluar_fase_simulacion(vecino, fase, año, n_rep=10)  # Muy pocas réplicas
                print("Vecino", costo_v)
                if costo_v < mejor_costo_vecino:
                    mejor_costo_vecino = costo_v
                    mejor_vecino = vecino
            
            S_vecino = mejor_vecino
            costo_vecino = mejor_costo_vecino
        else:
            # Global: genera 1 vecino con perturbación grande
            S_vecino = generar_vecino(S_actual, fase, "global")

            costo_vecino = evaluar_fase_simulacion(S_vecino, fase, año, n_rep=5)
        
        # Criterio de Metropolis
        delta = costo_vecino - costo_actual
        
        if delta < 0:
            # Mejor solución: aceptar siempre
            S_actual = S_vecino
            costo_actual = costo_vecino
            historial["aceptaciones"] += 1
            
            if costo_actual < costo_mejor:
                S_mejor = S_actual.copy()
                costo_mejor = costo_actual
                
                if verbose:
                    print(f"  Iter {iteracion:4d}: ✓ Nuevo mejor! costo={costo_mejor:.2e}, T={T:.1f}")
                    print(S_actual)
            elif verbose and iteracion % 10 == 0:
                print(f"  Iter {iteracion:4d}: ✓ Mejora local costo={costo_actual:.2e}, T={T:.1f}")
        
        elif random.random() < math.exp(-delta / T):
            # Solución peor pero aceptada probabilísticamente
            S_actual = S_vecino
            costo_actual = costo_vecino
            historial["aceptaciones"] += 1
            if verbose and iteracion % 10 == 0:
                print(f"  Iter {iteracion:4d}: ~ Aceptado peor costo={costo_actual:.2e}, T={T:.1f}")
        else:
            historial["rechazos"] += 1
            if verbose and iteracion % 10 == 0:
                print(f"  Iter {iteracion:4d}: ✗ Rechazado costo={costo_vecino:.2e}, T={T:.1f}")
        
        # Enfriamiento
        T = max(T * alpha, T_min)
        
        # Guardar historial
        historial["costos"].append(costo_actual)
        historial["temperaturas"].append(T)
        
        # Imprimir progreso cada 50 iteraciones
        if verbose and iteracion % 50 == 0:
            tasa_aceptacion = historial["aceptaciones"] / iteracion
            print(f"  Iter {iteracion:4d}: costo_actual={costo_actual:.2e}, "
                  f"mejor={costo_mejor:.2e}, T={T:.1f}, "
                  f"accept_rate={tasa_aceptacion:.2f}")
    
    t_fin = time.time()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SA - FASE {fase.upper()} COMPLETADO")
        print(f"{'='*60}")
        print(f"Mejor costo: {costo_mejor:.2e}")
        if params['iter_max'] > 0: #evitar division by zero
            porcentaje = historial['aceptaciones'] / params['iter_max'] * 100
            print(f"Aceptaciones: {historial['aceptaciones']} ({porcentaje:.1f}%)")
        else:
            print(f"Aceptaciones: {historial['aceptaciones']} (sin iteraciones - iter_max=0)")
        print(f"Tiempo: {t_fin - t_inicio:.1f}s")
    
    return S_mejor, costo_mejor, historial


# ============================================
# ALGORITMO SA PENDULAR PRINCIPAL
# ============================================

def SA_Pendular_Simulacion(config_inicial: ConfiguracionInicial,
                          max_ciclos: int = MAX_CICLOS,
                          tol_convergencia: float = TOL_CONVERGENCIA,
                          verbose: bool = True) -> ResultadoOptimizacion:
    """
    Algoritmo SA Pendular con 3 fases jerárquicas usando simulación.
    
    Args:
        config_inicial: ConfiguracionInicial del supermercado
        max_ciclos: número máximo de ciclos pendulares
        tol_convergencia: mejora mínima requerida para continuar
        verbose: si imprimir progreso detallado
    
    Returns:
        ResultadoOptimizacion con la mejor solución encontrada
    """
    if verbose:
        print("\n" + "="*80)
        print("ALGORITMO SA PENDULAR - 3 FASES CON SIMULACIÓN")
        print("="*80)
        print(f"Configuración inicial:")
        for tipo, num in config_inicial.config_caja.items():
            print(f"  {tipo.value}: {num}")
        print(f"\nParámetros:")
        print(f"  Max ciclos: {max_ciclos}")
        print(f"  Tolerancia convergencia: {tol_convergencia*100}%")
    
    # Convertir configuración inicial a estructuras de datos
    estrategica_actual = config_inicial_a_estrategica(config_inicial)
    tactica_actual = config_inicial_a_tactica(config_inicial)
    operacional_actual = config_inicial_a_operacional(config_inicial)
    
    # Evaluar configuración inicial
    solucion_inicial = SolucionCompleta(
        estrategica=estrategica_actual,
        tactica=tactica_actual,
        operacional=operacional_actual
    )
    
    resultado_inicial = evaluar_configuracion_simulacion(solucion_inicial, año=0, n_rep=N_REPLICAS_EVALUACION)
    VAN_inicial = resultado_inicial['VAN_total']
    
    if verbose:
        print(f"\nVAN configuración inicial: ${VAN_inicial:,.0f}")
    
    # Estado del algoritmo
    VAN_previo = VAN_inicial
    mejor_solucion = solucion_inicial.copy()
    mejor_VAN = VAN_inicial
    
    historial_ciclos = []
    
    t_inicio_total = time.time()
    
    # ========== CICLO PENDULAR ==========
    for ciclo in range(1, max_ciclos + 1):
        if verbose:
            print(f"\n{'#'*80}")
            print(f"CICLO PENDULAR {ciclo}/{max_ciclos}")
            print(f"{'#'*80}")
        
        t_inicio_ciclo = time.time()
        
        # --- FASE 1: ESTRATÉGICA ---
        if verbose:
            print(f"\n>>> FASE 1: ESTRATÉGICA")
            print(f"    Objetivo: Decidir qué cajas construir")
            print(f"    Configuración inicial: {dict(estrategica_actual.cajas_por_tipo)}")
        
        estrategica_nueva, costo_estrategico, hist_estrategico = SA_fase_simulacion(
            config_inicial=estrategica_actual,
            fase="estrategica",
            año=0,
            verbose=verbose
        )
        
        if verbose:
            print(f"    ✅ FASE 1 COMPLETADA")
            print(f"    Configuración resultante: {dict(estrategica_nueva.cajas_por_tipo)}")
        
        # --- FASE 2: TÁCTICA ---
        if verbose:
            print(f"\n>>> FASE 2: TÁCTICA")
            print(f"    Objetivo: Decidir cuándo activar cajas por año")
            print(f"    Configuración inicial: {dict(tactica_actual.cajas_por_anio[0])}")
        
        # Actualizar restricción superior
        tactica_actual.config_estrategica = estrategica_nueva
        tactica_actual = ajustar_tactica_a_estrategica(tactica_actual)
        
        tactica_nueva, costo_tactico, hist_tactico = SA_fase_simulacion(
            config_inicial=tactica_actual,
            fase="tactica",
            año=0,
            verbose=verbose
        )
        
        if verbose:
            print(f"    ✅ FASE 2 COMPLETADA")
            print(f"    Configuración resultante:")
            for anio in range(5):
                print(f"      Año {anio}: {dict(tactica_nueva.cajas_por_anio[anio])}")
        
        # --- FASE 3: OPERACIONAL ---
        if verbose:
            print(f"\n>>> FASE 3: OPERACIONAL")
            print(f"    Objetivo: Decidir horarios de apertura por hora")
        
        # Actualizar restricción superior
        operacional_actual.config_tactica = tactica_nueva
        operacional_actual = ajustar_horarios_a_capacidad(operacional_actual)
        
        if verbose:
            # Calcular cajas disponibles según configuración táctica
            cajas_disponibles = sum(tactica_nueva.cajas_por_anio[0][tipo] for tipo in LaneType)
            print(f"    Cajas disponibles (según táctica): {cajas_disponibles}")
            
            # Calcular cajas abiertas promedio antes de optimización
            total_cajas_antes = sum(len(operacional_actual.horarios[DayType.NORMAL][hora][tipo]) 
                                  for hora in range(8, 22) for tipo in LaneType)
            print(f"    Cajas abiertas promedio (antes): {total_cajas_antes/14:.1f}")
        
        operacional_nueva, costo_operacional, hist_operacional = SA_fase_simulacion(
            config_inicial=operacional_actual,
            fase="operacional",
            año=0,
            verbose=verbose
        )
        
        if verbose:
            print(f"    ✅ FASE 3 COMPLETADA")
            total_cajas_despues = sum(len(operacional_nueva.horarios[DayType.NORMAL][hora][tipo]) 
                                    for hora in range(8, 22) for tipo in LaneType)
            
            print(f"    Cajas abiertas promedio (después): {total_cajas_despues/14:.1f}")
            # PONER PRINTS que muestre los horarios, y cuantas cajas hay abiertas de cada tipo, seria una matriz, filas de horarios, y cada columna seria un tipo de caja.
            try:
                tipos = [t for t in LaneType]
                encabezado = "Hora  " + "  ".join(f"{t.value:>8}" for t in tipos)
                for dia in [DayType.NORMAL, DayType.OFERTA, DayType.DOMINGO]:
                    print(f"\n    Horarios - {dia.value}:")
                    print(f"    {encabezado}")
                    for hora in range(8, 22):
                        cuentas = [len(operacional_nueva.horarios[dia][hora].get(tipo, [])) 
                                   for tipo in tipos]
                        fila = f"    {hora:02d}: " + "  ".join(f"{c:8d}" for c in cuentas)
                        print(fila)
            except Exception as e:
                # No interrumpir ejecución por un print
                print(f"    (Advertencia) No se pudo desplegar matriz de horarios: {e}")
        
        # --- EVALUACIÓN GLOBAL COMPLETA ---
        if verbose:
            print(f"\n>>> EVALUACIÓN GLOBAL COMPLETA")
        
        solucion_ciclo = SolucionCompleta(
            estrategica=estrategica_nueva,
            tactica=tactica_nueva,
            operacional=operacional_nueva
        )
        
        # Evaluar con simulación completa
        resultado_ciclo = evaluar_configuracion_simulacion(solucion_ciclo, año=0, n_rep=N_REPLICAS_EVALUACION)
        VAN_ciclo = resultado_ciclo['VAN_total']
        solucion_ciclo.VAN = VAN_ciclo
        solucion_ciclo.kpis = resultado_ciclo
        
        # Calcular mejora
        if VAN_previo == 0:
            mejora = float('inf')
        else:
            mejora = (VAN_ciclo - VAN_previo) / abs(VAN_previo)
        
        # Actualizar mejor solución global
        if VAN_ciclo > mejor_VAN:
            mejor_solucion = solucion_ciclo.copy()
            mejor_VAN = VAN_ciclo
        
        # Guardar historial
        t_fin_ciclo = time.time()
        historial_ciclos.append({
            "ciclo": ciclo,
            "VAN": VAN_ciclo,
            "mejora": mejora,
            "tiempo_s": t_fin_ciclo - t_inicio_ciclo,
            "config": solucion_ciclo.copy()
        })
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"RESUMEN CICLO {ciclo}")
            print(f"{'='*80}")
            print(f"VAN ciclo: ${VAN_ciclo:,.0f}")
            print(f"Mejora: {mejora*100:.2f}% (umbral: {tol_convergencia*100}%)")
            print(f"Mejor VAN global: ${mejor_VAN:,.0f}")
            print(f"Tiempo ciclo: {t_fin_ciclo - t_inicio_ciclo:.1f}s")
            
            # Mostrar configuración del ciclo
            print(f"\n📋 CONFIGURACIÓN CICLO {ciclo}:")
            print(f"  Estratégica: {dict(estrategica_nueva.cajas_por_tipo)}")
            print(f"  Táctica (año 0): {dict(tactica_nueva.cajas_por_anio[0])}")
            
            # Mostrar resumen de horarios operacionales
            total_cajas_abiertas = {}
            for dia in DayType:
                for hora in range(8, 22):
                    for tipo in LaneType:
                        cajas_abiertas = len(operacional_nueva.horarios[dia][hora][tipo])
                        if tipo not in total_cajas_abiertas:
                            total_cajas_abiertas[tipo] = 0
                        total_cajas_abiertas[tipo] += cajas_abiertas
            
            print(f"  Operacional (promedio cajas/hora):")
            for tipo, total in total_cajas_abiertas.items():
                promedio = total / (3 * 14)  # 3 días × 14 horas
                print(f"    {tipo.value}: {promedio:.1f} cajas/hora promedio")
        
        # --- CRITERIO DE CONVERGENCIA ---
        if mejora < tol_convergencia and ciclo > 1:
            if verbose:
                print(f"\n✓ CONVERGENCIA ALCANZADA (mejora < {tol_convergencia*100}%)")
            break
        
        # Actualizar para próximo ciclo
        estrategica_actual = estrategica_nueva
        tactica_actual = tactica_nueva
        operacional_actual = operacional_nueva
        VAN_previo = VAN_ciclo
    
    t_fin_total = time.time()
    
    # --- RESULTADO FINAL ---
    mejora_porcentual = (mejor_VAN - VAN_inicial) / abs(VAN_inicial) * 100
    
    if verbose:
        print(f"\n{'='*80}")
        print("ALGORITMO SA PENDULAR COMPLETADO")
        print(f"{'='*80}")
        print(f"Ciclos ejecutados: {len(historial_ciclos)}/{max_ciclos}")
        print(f"Tiempo total: {t_fin_total - t_inicio_total:.1f}s ({(t_fin_total - t_inicio_total)/60:.1f} min)")
        print(f"\nVAN inicial:  ${VAN_inicial:>15,.0f}")
        print(f"VAN óptimo:   ${mejor_VAN:>15,.0f}")
        print(f"Mejora:       {mejora_porcentual:>14.1f}%")
        
        # Tabla de evolución
        print(f"\n{'='*80}")
        print("EVOLUCIÓN POR CICLO")
        print(f"{'='*80}")
        print(f"{'Ciclo':<10} {'VAN (MM CLP)':<20} {'Mejora (%)':<15} {'Tiempo (s)':<15}")
        print("-"*80)
        for h in historial_ciclos:
            print(f"{h['ciclo']:<10} ${h['VAN']/1e6:>15,.2f}  {h['mejora']*100:>12.2f}  {h['tiempo_s']:>12.1f}")
    
    return ResultadoOptimizacion(
        config_optima=mejor_solucion,
        van_optimo=mejor_VAN,
        van_inicial=VAN_inicial,
        mejora_porcentual=mejora_porcentual,
        tiempo_ejecucion=t_fin_total - t_inicio_total,
        historial=historial_ciclos,
        kpis_finales=mejor_solucion.kpis
    )


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    """Función principal del algoritmo"""
    print("🚀 Iniciando Heurística de 3 Etapas para Optimización de Cajas")
    print("="*80)
    
    # Configuración inicial (baseline actual)
    config_inicial = ConfiguracionInicial(
        config_caja={
            LaneType.REGULAR: 15,
            LaneType.EXPRESS: 3,
            LaneType.PRIORITY: 2,
            LaneType.SELF: 5
        },
        horarios_caja=HORARIOS_CAJA,
        año=2025
    )
    
    print(f"Configuración inicial:")
    for tipo, num in config_inicial.config_caja.items():
        print(f"  {tipo.value}: {num} cajas")
    
    # Ejecutar optimización
    resultado = SA_Pendular_Simulacion(
        config_inicial=config_inicial,
        max_ciclos=3,  # Reducido para testing
        tol_convergencia=0.02,
        verbose=True
    )
    
    # Mostrar resultados finales
    print(f"\n{'='*80}")
    print("RESULTADOS FINALES")
    print(f"{'='*80}")
    print(f"VAN inicial:  ${resultado.van_inicial:>15,.0f}")
    print(f"VAN óptimo:   ${resultado.van_optimo:>15,.0f}")
    print(f"Mejora:       {resultado.mejora_porcentual:>14.1f}%")
    print(f"Tiempo total: {resultado.tiempo_ejecucion:>14.1f}s")
    
    print(f"\nConfiguración óptima:")
    for tipo, num in resultado.config_optima.estrategica.cajas_por_tipo.items():
        print(f"  {tipo.value}: {num} cajas")
    
    return resultado


if __name__ == "__main__":
    resultado = main()
