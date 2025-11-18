"""
Script para probar la configuración óptima encontrada
==================================================

Configuración óptima:
- regular: 34 cajas (+19)
- express: 0 cajas (-3) 
- priority: 0 cajas (-2)
- self_checkout: 0 cajas (-5)
"""

import os
import sys
import time
from typing import Dict

# Agregar paths absolutos para evitar problemas con directorios con espacios/acentos
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "load_params"))
sys.path.append(os.path.join(PROJECT_ROOT, "Heurística"))

FINAL_DIR = os.path.join(PROJECT_ROOT, "Final")
ORIGINAL_CWD = os.getcwd()
os.chdir(FINAL_DIR)
sys.path.append(FINAL_DIR)

# Imports locales
from load_params.dominios import LaneType, DayType
from main_rapido_optimizado import AÑO, simular_varios

os.chdir(ORIGINAL_CWD)

# Configuración estratégica/táctica óptima reportada
# 34 cajas totales, pero solo 33 operativas durante los primeros 3 años (2025-2027)
# Para este test verificamos el primer año operativo (2025) con 33 cajas activas
TOTAL_CAJAS_REGULARES_ACTIVAS = 33

# Horarios optimizados (número de cajas regulares abiertas por hora)
# Horas expresadas en formato 24h (8=08:00, 21=21:00). Cada bloque dura 60 minutos.
TABLA_HORAS_REGULARES = {
    DayType.NORMAL: {
        8: 8, 9: 8, 10: 33, 11: 33, 12: 33, 13: 33, 14: 33,
        15: 31, 16: 33, 17: 33, 18: 33, 19: 33, 20: 29, 21: 8,
    },
    DayType.OFERTA: {
        8: 3, 9: 6, 10: 33, 11: 33, 12: 33, 13: 33, 14: 33,
        15: 33, 16: 33, 17: 33, 18: 33, 19: 33, 20: 11, 21: 7,
    },
    DayType.DOMINGO: {
        8: 8, 9: 8, 10: 33, 11: 33, 12: 33, 13: 33, 14: 33,
        15: 33, 16: 33, 17: 33, 18: 33, 19: 33, 20: 33, 21: 4,
    },
}


def minutos_desde_apertura(hora: int) -> int:
    """Convierte una hora (8-22) a minutos desde apertura (08:00)."""
    return (hora - 8) * 60


def construir_horarios_optimos(total_cajas: int) -> Dict[DayType, Dict[LaneType, list]]:
    """
    Convierte la tabla de cajas abiertas por hora en el formato esperado por la simulación:
    HORARIOS[DayType][LaneType] = [(caja_id, [(inicio_min, fin_min), ...]), ...]
    """
    horarios = {}

    for dia, tabla in TABLA_HORAS_REGULARES.items():
        horarios[dia] = {lt: [] for lt in LaneType}

        # Preparar intervalos por caja
        intervalos_por_caja = {i: [] for i in range(total_cajas)}
        inicio_activo = {i: None for i in range(total_cajas)}

        for hora in range(8, 22):  # 08:00 a 22:00
            cajas_abiertas = tabla.get(hora, 0)
            cajas_activa_set = set(range(min(cajas_abiertas, total_cajas)))

            inicio = minutos_desde_apertura(hora)
            fin = inicio + 60

            for caja_id in range(total_cajas):
                if caja_id in cajas_activa_set:
                    if inicio_activo[caja_id] is None:
                        inicio_activo[caja_id] = inicio
                else:
                    if inicio_activo[caja_id] is not None:
                        intervalos_por_caja[caja_id].append((inicio_activo[caja_id], inicio))
                        inicio_activo[caja_id] = None

        # Cerrar intervalos que siguen activos a las 22:00
        fin_dia = minutos_desde_apertura(22)
        for caja_id in range(total_cajas):
            if inicio_activo[caja_id] is not None:
                intervalos_por_caja[caja_id].append((inicio_activo[caja_id], fin_dia))
                inicio_activo[caja_id] = None

        # Crear estructura final
        horarios[dia][LaneType.REGULAR] = [
            (caja_id, intervalos_por_caja[caja_id]) for caja_id in range(total_cajas)
        ]

        # Otros tipos sin operación en esta configuración
        horarios[dia][LaneType.EXPRESS] = []
        horarios[dia][LaneType.PRIORITY] = []
        horarios[dia][LaneType.SELF] = []

    return horarios

HORARIOS_OPTIMOS = construir_horarios_optimos(TOTAL_CAJAS_REGULARES_ACTIVAS)


def porcentaje_atendidos(kpi: Dict[str, float]) -> float:
    total = kpi.get("clientes_totales", 0) or 0
    atendidos = kpi.get("atendidos", 0) or 0
    if total == 0:
        return 0.0
    return (atendidos / total) * 100


def probar_configuracion_optima(n_rep: int = 100):
    """Prueba la configuración óptima encontrada"""
    print("🚀 PROBANDO CONFIGURACIÓN ÓPTIMA")
    print("="*60)
    
    # Configuración óptima (primeros 3 años 33 cajas, últimos 2 años 34)
    CONFIG_OPTIMA = {
        LaneType.REGULAR: TOTAL_CAJAS_REGULARES_ACTIVAS,
        LaneType.EXPRESS: 0,
        LaneType.PRIORITY: 0,
        LaneType.SELF: 0
    }
    
    print("📋 CONFIGURACIÓN ÓPTIMA:")
    for tipo, num in CONFIG_OPTIMA.items():
        print(f"  {tipo.value}: {num} cajas")
    print("  Nota: 33 cajas activas años 2025-2027, 34 cajas activas años 2028-2029")
    
    print(f"\n🧪 EJECUTANDO SIMULACIÓN...")
    print("  - 3 tipos de día (NORMAL, OFERTA, DOMINGO)")
    print(f"  - {n_rep} réplicas por tipo de día (promedio)")
    print("  - Evaluación completa de VAN")
    
    # Ejecutar simulación
    t_inicio = time.time()
    
    resultados_por_dia = {}
    VAN_total = 0
    clientes_totales = 0
    clientes_atendidos = 0
    
    for dia in [DayType.NORMAL, DayType.OFERTA, DayType.DOMINGO]:
        print(f"\n📅 SIMULANDO {dia.value.upper()}...")
        os.chdir(FINAL_DIR)
        try:
            resultado = simular_varios(
                dia=dia,
                CONFIG=CONFIG_OPTIMA,
                HORARIOS=HORARIOS_OPTIMOS,
                AÑO=AÑO,
                n_rep=n_rep,
                seed_base=42
            )
        finally:
            os.chdir(ORIGINAL_CWD)
    
        resultados_por_dia[dia] = resultado
        clientes_totales += resultado["clientes_totales"]
        clientes_atendidos += resultado["atendidos"]
        
        # Usar VAN_correcto_5_anios (incluye inversión inicial y 5 años proyectados)
        # Este es el VAN que usa la heurística para optimizar
        van_correcto = resultado.get('VAN_correcto_5_anios', resultado['VAN_dia_clp'])
        
        # Calcular VAN ponderado
        multiplicador = 3 if dia in [DayType.NORMAL, DayType.OFERTA] else 1
        VAN_ponderado = van_correcto * multiplicador
        VAN_total += VAN_ponderado
        
        print(f"  VAN día ({dia.value}): ${resultado['VAN_dia_clp']:>15,.0f}")
        print(f"  VAN 5 años: ${van_correcto:>15,.0f}")
        print(f"  Inversión inicial: ${resultado.get('Inversion_inicial', 0):>15,.0f}")
        print(f"  Ingresos: ${resultado['ingresos_clp']:>15,.0f}")
        print(f"  Costos: ${resultado['costos_clp']:>15,.0f}")
        print(f"  Clientes: {resultado['clientes_totales']}")
        print(f"  Atendidos: {resultado['atendidos']}")
        print(f"  % Clientes atendidos: {porcentaje_atendidos(resultado):.2f}%")
        print(f"  Abandono: {resultado['abandono_rate']*100:.1f}%")
        print(f"  VAN ponderado (5 años): ${VAN_ponderado:>15,.0f}")
    
    t_fin = time.time()
    
    # Mostrar resultados finales
    print(f"\n{'='*60}")
    print("💰 RESULTADOS FINALES")
    print(f"{'='*60}")
    
    print(f"⏱️  Tiempo total: {t_fin - t_inicio:.1f}s")
    print(f"💰 VAN TOTAL: ${VAN_total:>15,.0f}")
    
    print(f"\n📊 RESUMEN POR TIPO DE DÍA:")
    for dia, resultado in resultados_por_dia.items():
        multiplicador = 3 if dia in [DayType.NORMAL, DayType.OFERTA] else 1
        van_correcto = resultado.get('VAN_correcto_5_anios', resultado['VAN_dia_clp'])
        VAN_ponderado = van_correcto * multiplicador
        print(f"  {dia.value}: ${VAN_ponderado:>15,.0f} (×{multiplicador}) - VAN 5 años: ${van_correcto:>15,.0f}")
    
    print(f"\n🏗️  CONFIGURACIÓN UTILIZADA:")
    for tipo, num in CONFIG_OPTIMA.items():
        print(f"  {tipo.value}: {num} cajas")
    
    print(f"\n📈 KPIs PROMEDIO:")
    ingresos_promedio = sum(r['ingresos_clp'] for r in resultados_por_dia.values()) / len(resultados_por_dia)
    costos_promedio = sum(r['costos_clp'] for r in resultados_por_dia.values()) / len(resultados_por_dia)
    abandono_promedio = sum(r['abandono_rate'] for r in resultados_por_dia.values()) / len(resultados_por_dia)
    
    print(f"  Ingresos promedio: ${ingresos_promedio:>15,.0f}")
    print(f"  Costos promedio: ${costos_promedio:>15,.0f}")
    print(f"  Abandono promedio: {abandono_promedio*100:>14.1f}%")
    
    porcentaje_global = (clientes_atendidos / clientes_totales) * 100 if clientes_totales else 0.0
    print(f"\n👥 Porcentaje global de clientes atendidos: {porcentaje_global:.2f}%")
    
    return {
        'config_optima': CONFIG_OPTIMA,
        'VAN_total': VAN_total,
        'resultados_por_dia': resultados_por_dia,
        'tiempo_ejecucion': t_fin - t_inicio,
        'porcentaje_atendidos_total': porcentaje_global,
    }

def main():
    """Función principal"""
    try:
        resultado = probar_configuracion_optima()
        
        print(f"\n{'='*60}")
        print("✅ SIMULACIÓN COMPLETADA")
        print(f"{'='*60}")
        
        if resultado['VAN_total'] > 0:
            print(f"🎉 ¡VAN POSITIVO! ${resultado['VAN_total']:,.0f}")
        else:
            print(f"⚠️  VAN NEGATIVO: ${resultado['VAN_total']:,.0f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
