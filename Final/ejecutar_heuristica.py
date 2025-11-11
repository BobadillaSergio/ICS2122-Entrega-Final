"""
Script principal para ejecutar la heurística de 3 etapas
========================================================

Ejecuta la heurística completa y encuentra la mejor configuración de cajas.
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

def cargar_configuracion_optima_anterior():
    """Carga la configuración óptima de una ejecución anterior"""
    if not os.path.exists('resultados_heuristica.json'):
        return None
    
    try:
        with open('resultados_heuristica.json', 'r', encoding='utf-8') as f:
            datos = json.load(f)
        
        if 'config_optima' not in datos:
            return None
        
        config_optima = datos['config_optima']
        
        # Reconstruir configuración inicial desde la óptima
        config_caja = {}
        for tipo_str, num in config_optima['estrategica'].items():
            tipo_enum = LaneType(tipo_str)
            config_caja[tipo_enum] = num
        
        # Crear horarios optimizados desde la configuración operacional
        horarios_optimos = {}
        for dia_str, horarios_dia in config_optima['operacional']['horarios'].items():
            dia_enum = DayType(dia_str)
            horarios_optimos[dia_enum] = {}
            
            for tipo_enum in LaneType:
                horarios_optimos[dia_enum][tipo_enum] = []
                
                for hora_str, cajas_abiertas in horarios_dia.items():
                    hora = int(hora_str)
                    if tipo_enum.value in cajas_abiertas:
                        for caja_id in cajas_abiertas[tipo_enum.value]:
                            # Crear horario para esta caja (abierta toda la hora)
                            if caja_id < len(horarios_optimos[dia_enum][tipo_enum]):
                                # Actualizar horario existente
                                horarios_optimos[dia_enum][tipo_enum][caja_id] = (hora*60, (hora+1)*60)
                            else:
                                # Agregar nueva caja
                                horarios_optimos[dia_enum][tipo_enum].append((hora*60, (hora+1)*60))
        
        return ConfiguracionInicial(
            config_caja=config_caja,
            horarios_caja=horarios_optimos,
            año=2025
        )
        
    except Exception as e:
        print(f"⚠️  Error cargando configuración anterior: {e}")
        return None

def ejecutar_heuristica_completa():
    """Ejecuta la heurística completa para encontrar la mejor configuración"""
    print("🚀 EJECUTANDO HEURÍSTICA DE 3 ETAPAS")
    print("="*80)
    print("Objetivo: Encontrar la mejor configuración de cajas para maximizar el VAN")
    print("="*80)
    
    # Intentar cargar configuración óptima anterior
    config_inicial = cargar_configuracion_optima_anterior()
    
    if config_inicial is not None:
        print("🔄 Usando configuración óptima anterior como punto de partida")
        print("📋 CONFIGURACIÓN INICIAL (ÓPTIMA ANTERIOR):")
        for tipo, num in config_inicial.config_caja.items():
            print(f"  {tipo.value}: {num} cajas")
    else:
        print("🆕 Usando configuración baseline estándar")
        # Configuración inicial (baseline actual)
        config_inicial = ConfiguracionInicial(
            config_caja={
                LaneType.REGULAR: 34,
                LaneType.EXPRESS: 0,
                LaneType.PRIORITY: 0,
                LaneType.SELF: 0
            },
            horarios_caja=HORARIOS_CAJA,
            año=2025
        )
        
        print("📋 CONFIGURACIÓN INICIAL (BASELINE):")
        for tipo, num in config_inicial.config_caja.items():
            print(f"  {tipo.value}: {num} cajas")
    
    print(f"\n🎯 INICIANDO OPTIMIZACIÓN...")
    print("  - Etapa 1: Estratégica (decidir qué cajas construir)")
    print("  - Etapa 2: Táctica (decidir cuándo activar cajas)")
    print("  - Etapa 3: Operacional (decidir horarios)")
    print("  - Usando simulación optimizada (94x más rápida)")
    print("  - Parámetros reducidos para ejecución en 2 horas")
    
    # Reducir parámetros para ejecución en 2 horas
    import heuristica_3_etapas
    heuristica_3_etapas.PARAMETROS_SA["estrategica"]["iter_max"] = 50  # Reducido de 200
    heuristica_3_etapas.PARAMETROS_SA["tactica"]["iter_max"] = 50     # Reducido de 200
    heuristica_3_etapas.PARAMETROS_SA["operacional"]["iter_max"] = 100 # Reducido de 300
    heuristica_3_etapas.N_REPLICAS_EVALUACION = 3  # Reducido de 5
    
    print(f"  - Iteraciones estratégica: 50 (reducido)")
    print(f"  - Iteraciones táctica: 50 (reducido)")
    print(f"  - Iteraciones operacional: 100 (reducido)")
    print(f"  - Réplicas por evaluación: 3 (reducido)")
    
    # Ejecutar optimización
    t_inicio = time.time()
    
    resultado = SA_Pendular_Simulacion(
        config_inicial=config_inicial,
        max_ciclos=3,  # 3 ciclos completos
        tol_convergencia=0.01,  # 1% de mejora mínima (más estricto)
        verbose=True
    )
    
    t_fin = time.time()
    
    # Mostrar resultados finales
    print(f"\n{'='*80}")
    print("🎉 OPTIMIZACIÓN COMPLETADA")
    print(f"{'='*80}")
    
    print(f"⏱️  Tiempo total: {t_fin - t_inicio:.1f}s ({(t_fin - t_inicio)/60:.1f} min)")
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
    
    # Análisis de cambios
    print(f"\n📊 ANÁLISIS DE CAMBIOS:")
    total_cajas_inicial = sum(config_inicial.config_caja.values())
    total_cajas_optimo = sum(resultado.config_optima.estrategica.cajas_por_tipo.values())
    
    print(f"  Total cajas inicial: {total_cajas_inicial}")
    print(f"  Total cajas óptimo: {total_cajas_optimo}")
    print(f"  Diferencia: {total_cajas_optimo - total_cajas_inicial:+d}")
    
    # KPIs por ciclo
    if len(resultado.historial) > 1:
        print(f"\n📈 EVOLUCIÓN POR CICLO:")
        print(f"{'Ciclo':<8} {'VAN (MM)':<12} {'Mejora (%)':<12} {'Tiempo (s)':<12}")
        print("-" * 50)
        for h in resultado.historial:
            print(f"{h['ciclo']:<8} ${h['VAN']/1e6:>10,.2f}  {h['mejora']*100:>10.2f}  {h['tiempo_s']:>10.1f}")
    
    # Guardar resultados
    guardar_resultados(resultado)
    
    return resultado


def guardar_resultados(resultado):
    """Guarda los resultados en un archivo JSON con configuración completa"""
    print(f"\n💾 GUARDANDO RESULTADOS...")
    
    import json
    
    # Función auxiliar para serializar configuraciones complejas
    def serializar_config_estrategica(config):
        return {k.value: v for k, v in config.cajas_por_tipo.items()}
    
    def serializar_config_tactica(config):
        return {
            'cajas_por_anio': {
                str(anio): {k.value: v for k, v in cajas.items()}
                for anio, cajas in config.cajas_por_anio.items()
            }
        }
    
    def serializar_config_operacional(config):
        return {
            'horarios': {
                dia.value: {
                    str(hora): {k.value: v for k, v in tipos.items()}
                    for hora, tipos in horarios.items()
                }
                for dia, horarios in config.horarios.items()
            },
            'anio_actual': config.anio_actual
        }
    
    resultados = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config_inicial': {k.value: v for k, v in {
            LaneType.REGULAR: 34,
            LaneType.EXPRESS: 0,
            LaneType.PRIORITY: 0,
            LaneType.SELF: 0
        }.items()},
        'config_optima': {
            'estrategica': serializar_config_estrategica(resultado.config_optima.estrategica),
            'tactica': serializar_config_tactica(resultado.config_optima.tactica),
            'operacional': serializar_config_operacional(resultado.config_optima.operacional)
        },
        'van_inicial': resultado.van_inicial,
        'van_optimo': resultado.van_optimo,
        'mejora_porcentual': resultado.mejora_porcentual,
        'tiempo_ejecucion': resultado.tiempo_ejecucion,
        'ciclos_ejecutados': len(resultado.historial),
        'historial': [
            {
                'ciclo': h['ciclo'],
                'VAN': h['VAN'],
                'mejora': h['mejora'],
                'tiempo_s': h['tiempo_s'],
                'config_ciclo': {
                    'estrategica': serializar_config_estrategica(h['config'].estrategica),
                    'tactica': serializar_config_tactica(h['config'].tactica),
                    'operacional': serializar_config_operacional(h['config'].operacional)
                }
            } for h in resultado.historial
        ]
    }
    
    # Guardar en archivo JSON
    with open('resultados_heuristica.json', 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Resultados guardados en 'resultados_heuristica.json'")
    print(f"📋 Configuración óptima completa guardada (estratégica, táctica, operacional)")
    print(f"🔄 Historial de {len(resultado.historial)} ciclos guardado")


def main():
    """Función principal"""
    print("🎯 HEURÍSTICA DE 3 ETAPAS PARA OPTIMIZACIÓN DE CAJAS")
    print("="*80)
    print("Este script ejecuta la heurística completa para encontrar")
    print("la mejor configuración de cajas que maximice el VAN.")
    print("="*80)
    
    try:
        # Ejecutar heurística completa
        resultado = ejecutar_heuristica_completa()
        
        print(f"\n{'='*80}")
        print("✅ HEURÍSTICA COMPLETADA EXITOSAMENTE")
        print(f"{'='*80}")
        
        if resultado.mejora_porcentual > 0:
            print(f"🎉 ¡Se encontró una mejora del {resultado.mejora_porcentual:.1f}%!")
            print(f"💰 Ahorro anual estimado: ${(resultado.van_optimo - resultado.van_inicial):,.0f}")
        else:
            print(f"ℹ️  La configuración inicial ya es óptima o muy cercana.")
        
        print(f"\n📁 Revisa 'resultados_heuristica.json' para más detalles.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR EN LA HEURÍSTICA: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 ¡Heurística ejecutada exitosamente!")
        print("📊 Revisa los resultados en 'resultados_heuristica.json'")
    else:
        print("\n💥 La heurística falló. Revisar errores.")
