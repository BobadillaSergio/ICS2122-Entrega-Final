#!/usr/bin/env python3
"""
🚀 HEURÍSTICA OPTIMIZADA PARA 2 HORAS
=====================================

Versión optimizada para ejecutar en 2 horas con 3 ciclos completos.
Parámetros reducidos pero manteniendo calidad de optimización.
"""

import sys
import os
import time
import warnings
from typing import Dict

# Ocultar warnings de Ray
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"

# Agregar paths
sys.path.append("..")
sys.path.append("../load_params")
sys.path.append("../Heurística")

# Imports locales
from load_params.dominios import LaneType, DayType
from heuristica_3_etapas import (
    ConfiguracionInicial, SA_Pendular_Simulacion
)

# Importar simulación optimizada
from main_rapido_optimizado import HORARIOS_CAJA
import json
import os

def ejecutar_heuristica_2_horas():
    """Ejecuta la heurística optimizada para 2 horas"""
    print("🚀 HEURÍSTICA OPTIMIZADA PARA 2 HORAS")
    print("="*60)
    print("Versión optimizada para ejecución en 2 horas")
    print("3 ciclos completos con parámetros reducidos")
    print("="*60)
    
    # Configuración inicial (baseline óptima)
    config_inicial = ConfiguracionInicial(
        config_caja={
            LaneType.REGULAR: 33,
            LaneType.EXPRESS: 0,
            LaneType.PRIORITY: 0,
            LaneType.SELF: 0
        },
        horarios_caja=HORARIOS_CAJA,
        año=2025
    )
    
    print(f"\n📋 CONFIGURACIÓN INICIAL:")
    for tipo, num in config_inicial.config_caja.items():
        print(f"  {tipo.value}: {num} cajas")
    
    print(f"\n🎯 CONFIGURANDO PARÁMETROS PARA 2 HORAS...")
    print("  - 3 ciclos completos de optimización")
    print("  - Parámetros reducidos para velocidad")
    print("  - Manteniendo calidad de optimización")
    
    # Reducir parámetros para ejecución en 2 horas
    import heuristica_3_etapas
    heuristica_3_etapas.PARAMETROS_SA["estrategica"]["iter_max"] = 0  # Muy reducido
    heuristica_3_etapas.PARAMETROS_SA["tactica"]["iter_max"] = 0  # CAMBIARAIURGUYSUBYSUB
    heuristica_3_etapas.PARAMETROS_SA["operacional"]["iter_max"] = 10 # Reducido
    heuristica_3_etapas.N_REPLICAS_EVALUACION = 3  # Muy reducido
    
    print(f"  - Iteraciones estratégica: 30 (muy reducido)")
    print(f"  - Iteraciones táctica: 30 (muy reducido)")
    print(f"  - Iteraciones operacional: 50 (reducido)")
    print(f"  - Réplicas por evaluación: 2 (muy reducido)")
    
    # Ejecutar optimización
    print(f"\n🚀 INICIANDO OPTIMIZACIÓN...")
    t_inicio = time.time()
    
    resultado = SA_Pendular_Simulacion(
        config_inicial=config_inicial,
        max_ciclos=3,  # 3 ciclos completos
        tol_convergencia=0.01,  # 1% de mejora mínima
        verbose=True
    )
    
    t_fin = time.time()
    tiempo_total = t_fin - t_inicio
    
    # Mostrar resultados finales
    print(f"\n{'='*80}")
    print("🎉 OPTIMIZACIÓN COMPLETADA")
    print(f"{'='*80}")
    
    print(f"⏱️  Tiempo total: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
    print(f"🔄 Ciclos ejecutados: {len(resultado.historial)}")
    
    print(f"\n💰 RESULTADOS FINANCIEROS:")
    print(f"  VAN inicial:  ${resultado.van_inicial:>15,.0f}")
    print(f"  VAN óptimo:   ${resultado.van_optimo:>15,.0f}")
    print(f"  Mejora:       {resultado.mejora_porcentual:>14.1f}%")
    
    if resultado.mejora_porcentual > 0:
        print(f"  💵 Ahorro anual: ${(resultado.van_optimo - resultado.van_inicial):,.0f}")
    
    print(f"\n🏗️  CONFIGURACIÓN ÓPTIMA ENCONTRADA:")
    for tipo, num in resultado.config_optima.estrategica.cajas_por_tipo.items():
        cambio = num - config_inicial.config_caja[tipo]
        if cambio > 0:
            print(f"  {tipo.value}: {num} cajas (+{cambio})")
        elif cambio < 0:
            print(f"  {tipo.value}: {num} cajas ({cambio})")
        else:
            print(f"  {tipo.value}: {num} cajas (sin cambio)")
    
    # Guardar resultados
    print(f"\n💾 GUARDANDO RESULTADOS...")
    guardar_resultados(resultado)
    
    return resultado

def guardar_resultados(resultado):
    """Guarda los resultados en archivo JSON"""
    try:
        # Convertir resultado a formato serializable
        datos = {
            "van_inicial": resultado.van_inicial,
            "van_optimo": resultado.van_optimo,
            "mejora_porcentual": resultado.mejora_porcentual,
            "config_optima": {
                "estrategica": {
                    tipo.value: num for tipo, num in resultado.config_optima.estrategica.cajas_por_tipo.items()
                },
                "tactica": {
                    "años": resultado.config_optima.tactica.años,
                    "cajas_por_anio": {
                        str(año): {
                            tipo.value: num for tipo, num in cajas.items()
                        } for año, cajas in resultado.config_optima.tactica.cajas_por_anio.items()
                    }
                },
                "operacional": {
                    "años": resultado.config_optima.operacional.años,
                    "horarios_por_anio": {}
                }
            },
            "historial": [
                {
                    "ciclo": h["ciclo"],
                    "van": h["VAN"],
                    "mejora": h["mejora"],
                    "tiempo_s": h["tiempo_s"]
                } for h in resultado.historial
            ]
        }
        
        # Guardar en archivo JSON
        with open("resultados_heuristica_2h.json", "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        
        print(f"  ✅ Resultados guardados en: resultados_heuristica_2h.json")
        
    except Exception as e:
        print(f"  ❌ Error guardando resultados: {e}")

def main():
    """Función principal"""
    print("🚀 INICIANDO HEURÍSTICA OPTIMIZADA PARA 2 HORAS")
    print("="*60)
    print("Objetivo: Encontrar la mejor configuración en 2 horas")
    print("="*60)
    
    try:
        # Ejecutar heurística optimizada
        resultado = ejecutar_heuristica_2_horas()
        
        print(f"\n{'='*60}")
        print("✅ HEURÍSTICA COMPLETADA EN 2 HORAS")
        print(f"{'='*60}")
        
        if resultado.mejora_porcentual > 0:
            print(f"🎉 ¡Se encontró una mejora del {resultado.mejora_porcentual:.1f}%!")
        else:
            print(f"ℹ️  La configuración inicial ya es óptima.")
        
        print(f"\n💡 Resultados guardados en: resultados_heuristica_2h.json")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 ¡Heurística completada exitosamente!")
    else:
        print("\n💥 La heurística falló. Revisar errores.")
