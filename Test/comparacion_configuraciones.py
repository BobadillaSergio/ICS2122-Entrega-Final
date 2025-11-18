#!/usr/bin/env python3
"""
🔍 COMPARACIÓN DIRECTA DE CONFIGURACIONES
=========================================
Compara VAN de 5 años entre configuración original (15,3,2,5) vs óptima (34,0,0,0)
"""

import sys
import os
import warnings
from typing import Dict

# Ocultar warnings de Ray
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

# Agregar paths
sys.path.append("..")
sys.path.append("../load_params")

from load_params.dominios import LaneType, DayType
from main_rapido_optimizado import simular_varios, HORARIOS_CAJA, AÑO

def crear_horarios_para_config(config):
    """Crea horarios para una configuración específica"""
    horarios = {}
    horario_completo = (0, 840)  # 8 AM a 22 PM
    
    for dia in [DayType.DOMINGO, DayType.NORMAL, DayType.OFERTA]:
        horarios[dia] = {}
        for tipo in LaneType:
            num_cajas = config[tipo]
            if num_cajas > 0:
                horarios[dia][tipo] = [(i, [horario_completo]) for i in range(num_cajas)]
            else:
                horarios[dia][tipo] = []
    return horarios

def comparar_configuraciones():
    """Compara las dos configuraciones directamente"""
    print("🔍 COMPARACIÓN DIRECTA DE CONFIGURACIONES")
    print("="*60)
    print("Objetivo: Comparar VAN de 5 años entre configuraciones")
    print("="*60)
    
    # Configuraciones a comparar
    config_original = {
        LaneType.REGULAR: 15,
        LaneType.EXPRESS: 3,
        LaneType.PRIORITY: 2,
        LaneType.SELF: 5,
    }
    
    config_optima = {
        LaneType.REGULAR: 34,
        LaneType.EXPRESS: 0,
        LaneType.PRIORITY: 0,
        LaneType.SELF: 0,
    }
    
    print(f"\n📋 CONFIGURACIONES A COMPARAR:")
    print(f"  Original (15,3,2,5): {dict(config_original)}")
    print(f"  Óptima (34,0,0,0):   {dict(config_optima)}")
    
    # Crear horarios para cada configuración
    horarios_original = crear_horarios_para_config(config_original)
    horarios_optima = crear_horarios_para_config(config_optima)
    
    # Simular cada configuración
    configuraciones = [
        ("ORIGINAL", config_original, horarios_original),
        ("ÓPTIMA", config_optima, horarios_optima)
    ]
    
    resultados = {}
    
    for nombre, config, horarios in configuraciones:
        print(f"\n🧪 SIMULANDO CONFIGURACIÓN {nombre}:")
        print("-" * 40)
        
        van_total = 0
        van_por_dia = {}
        
        for dt in [DayType.DOMINGO, DayType.NORMAL, DayType.OFERTA]:
            kpi = simular_varios(dt, config, horarios, AÑO, n_rep=3, seed_base=1002)
            van_dia = kpi['VAN_dia_clp']
            van_5años = kpi.get('VAN_correcto_5_anios', 0)
            
            # Multiplicador según tipo de día
            multiplicador = 3 if dt in [DayType.NORMAL, DayType.OFERTA] else 1
            van_ponderado = van_5años * multiplicador
            van_total += van_ponderado
            van_por_dia[dt] = van_5años
            
            print(f"  {dt.value}:")
            print(f"    VAN_dia: ${van_dia:,.0f}")
            print(f"    VAN_5años: ${van_5años:,.0f}")
            print(f"    Multiplicador: {multiplicador}")
            print(f"    VAN_ponderado: ${van_ponderado:,.0f}")
        
        print(f"  📊 VAN TOTAL {nombre}: ${van_total:,.0f}")
        
        resultados[nombre] = {
            'van_total': van_total,
            'van_por_dia': van_por_dia,
            'config': config
        }
    
    # Comparación final
    print(f"\n💰 COMPARACIÓN FINAL:")
    print("="*60)
    
    van_original = resultados['ORIGINAL']['van_total']
    van_optima = resultados['ÓPTIMA']['van_total']
    diferencia = van_optima - van_original
    mejora_porcentual = (diferencia / abs(van_original)) * 100 if van_original != 0 else 0
    
    print(f"  VAN Original: ${van_original:,.0f}")
    print(f"  VAN Óptima:   ${van_optima:,.0f}")
    print(f"  Diferencia:    ${diferencia:,.0f}")
    print(f"  Mejora:       {mejora_porcentual:+.1f}%")
    
    # Análisis por tipo de día
    print(f"\n📊 ANÁLISIS POR TIPO DE DÍA:")
    print("-" * 40)
    
    for dt in [DayType.DOMINGO, DayType.NORMAL, DayType.OFERTA]:
        van_orig = resultados['ORIGINAL']['van_por_dia'][dt]
        van_opt = resultados['ÓPTIMA']['van_por_dia'][dt]
        diff = van_opt - van_orig
        
        print(f"  {dt.value}:")
        print(f"    Original: ${van_orig:,.0f}")
        print(f"    Óptima:   ${van_opt:,.0f}")
        print(f"    Diferencia: ${diff:,.0f}")
    
    return resultados

if __name__ == "__main__":
    try:
        resultados = comparar_configuraciones()
        print(f"\n🎉 Comparación completada!")
    except Exception as e:
        print(f"\n💥 Error en comparación: {e}")
        import traceback
        traceback.print_exc()
