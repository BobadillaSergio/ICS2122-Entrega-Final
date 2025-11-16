"""
Operadores de vecindad para generar soluciones vecinas en cada fase.
Incluye operadores locales (greedy) y globales (perturbaciones grandes).
"""
import random
from typing import List
import sys
sys.path.append("..")
from load_params.dominios import LaneType, DayType
from estructuras_datos import ConfigEstrategica, ConfigTactica, ConfigOperacional

from Final.main_rapido_optimizado import LAMBDA_POR_HORA
# ============================================
# OPERADORES ESTRATÉGICOS
# ============================================

def incrementar_tipo(config: ConfigEstrategica, tipo: LaneType, cantidad: int = 1) -> ConfigEstrategica:
    """Aumenta el número de cajas de un tipo específico."""
    nueva_config = config.copy()
    if tipo == LaneType.SELF:
        cantidad = 5  # Self siempre en múltiplos de 5
    nueva_config.cajas_por_tipo[tipo] += cantidad
    return nueva_config


def decrementar_tipo(config: ConfigEstrategica, tipo: LaneType, cantidad: int = 1) -> ConfigEstrategica:
    """Disminuye el número de cajas de un tipo específico."""
    nueva_config = config.copy()
    if tipo == LaneType.SELF:
        cantidad = 5
    nueva_config.cajas_por_tipo[tipo] = max(0, nueva_config.cajas_por_tipo[tipo] - cantidad)
    return nueva_config


def swap_tipos(config: ConfigEstrategica, tipo1: LaneType, tipo2: LaneType) -> ConfigEstrategica:
    """Intercambia 1 caja entre dos tipos: -1 de tipo1, +1 de tipo2."""
    nueva_config = config.copy()
    
    if tipo1 == LaneType.SELF or tipo2 == LaneType.SELF:
        cantidad = 5
    else:
        cantidad = 1
    
    nueva_config.cajas_por_tipo[tipo1] = max(0, nueva_config.cajas_por_tipo[tipo1] - cantidad)
    nueva_config.cajas_por_tipo[tipo2] += cantidad
    return nueva_config


def generar_vecinos_estrategicos_locales(config: ConfigEstrategica, max_vecinos: int = 20) -> List[ConfigEstrategica]:
    """
    Genera vecinos locales (pequeñas modificaciones).
    Estrategia: genera todos los vecinos posibles y filtra por factibilidad.
    """
    vecinos = []
    
    for tipo in LaneType:
        # Incrementar
        v1 = incrementar_tipo(config, tipo)
        if v1.es_factible():
            vecinos.append(v1)
        
        # Decrementar
        v2 = decrementar_tipo(config, tipo)
        if v2.es_factible() and v2.total_cajas > 0:
            vecinos.append(v2)
    
    # Swaps entre tipos
    tipos = list(LaneType)
    for i, tipo1 in enumerate(tipos):
        for tipo2 in tipos[i+1:]:
            v3 = swap_tipos(config, tipo1, tipo2)
            if v3.es_factible():
                vecinos.append(v3)
            
            v4 = swap_tipos(config, tipo2, tipo1)
            if v4.es_factible():
                vecinos.append(v4)
    
    # Limitar número de vecinos si es muy grande
    if len(vecinos) > max_vecinos:
        vecinos = random.sample(vecinos, max_vecinos)
    
    return vecinos


def generar_vecino_estrategico_global(config: ConfigEstrategica) -> ConfigEstrategica:
    """
    Genera un vecino global (perturbación grande).
    Estrategia: modificar múltiples tipos simultáneamente.
    """
    nueva_config = config.copy()
    
    # Cambiar aleatoriamente 2-3 tipos
    tipos_a_modificar = random.sample(list(LaneType), k=random.randint(2, 3))
    
    for tipo in tipos_a_modificar:
        cambio = random.choice([-5, -1, 1, 5]) if tipo == LaneType.SELF else random.choice([-2, -1, 1, 2])
        nueva_config.cajas_por_tipo[tipo] = max(0, nueva_config.cajas_por_tipo[tipo] + cambio)
    
    # Si no es factible, intentar reparar
    max_intentos = 10
    for _ in range(max_intentos):
        if nueva_config.es_factible():
            return nueva_config
        
        # Reparación: reducir aleatoriamente hasta ser factible
        tipo_aleatorio = random.choice(list(LaneType))
        decremento = 5 if tipo_aleatorio == LaneType.SELF else 1
        nueva_config.cajas_por_tipo[tipo_aleatorio] = max(0, nueva_config.cajas_por_tipo[tipo_aleatorio] - decremento)
    
    # Si después de 10 intentos no es factible, retornar la original
    return config.copy()


# ============================================
# OPERADORES TÁCTICOS
# ============================================

def activar_temprano(config: ConfigTactica, tipo: LaneType, anio: int, cantidad: int = 1) -> ConfigTactica:
    """Activa más cajas de un tipo en un año específico (y propagarlo a años futuros)."""
    nueva_config = config.copy()
    
    if tipo == LaneType.SELF:
        cantidad = 5
    
    for a in range(anio, 5):
        nueva_config.cajas_por_anio[a][tipo] += cantidad
    
    return nueva_config


def activar_tarde(config: ConfigTactica, tipo: LaneType, anio: int, cantidad: int = 1) -> ConfigTactica:
    """Retrasa la activación de cajas (reduce en años tempranos)."""
    nueva_config = config.copy()
    
    if tipo == LaneType.SELF:
        cantidad = 5
    
    for a in range(anio + 1):
        nueva_config.cajas_por_anio[a][tipo] = max(0, nueva_config.cajas_por_anio[a][tipo] - cantidad)
    
    return nueva_config


def swap_anios(config: ConfigTactica, anio1: int, anio2: int) -> ConfigTactica:
    """Intercambia la configuración completa entre dos años."""
    nueva_config = config.copy()
    nueva_config.cajas_por_anio[anio1], nueva_config.cajas_por_anio[anio2] = \
        nueva_config.cajas_por_anio[anio2], nueva_config.cajas_por_anio[anio1]
    return nueva_config


def generar_vecinos_tacticos_locales(config: ConfigTactica, max_vecinos: int = 30) -> List[ConfigTactica]:
    """Genera vecinos locales (pequeñas modificaciones en asignación anual)."""
    vecinos = []
    
    for anio in range(5):
        for tipo in LaneType:
            # Activar temprano
            v1 = activar_temprano(config, tipo, anio)
            if v1.es_factible():
                vecinos.append(v1)
            
            # Activar tarde
            v2 = activar_tarde(config, tipo, anio)
            if v2.es_factible():
                vecinos.append(v2)
    
    # Swaps entre años
    for a1 in range(5):
        for a2 in range(a1 + 1, 5):
            v3 = swap_anios(config, a1, a2)
            if v3.es_factible():
                vecinos.append(v3)
    
    if len(vecinos) > max_vecinos:
        vecinos = random.sample(vecinos, max_vecinos)
    
    return vecinos


def generar_vecino_tactico_global(config: ConfigTactica) -> ConfigTactica:
    """
    Genera un vecino global (cambio drástico en estrategia temporal).
    Estrategias: rampa acelerada, rampa gradual, step function.
    """
    nueva_config = config.copy()
    estrategia = random.choice(["rampa_acelerada", "rampa_gradual", "step_function"])
    
    if estrategia == "rampa_acelerada":
        # Activar muchas cajas en años 0-1
        for tipo in LaneType:
            max_cajas = nueva_config.config_estrategica.cajas_por_tipo[tipo]
            proporcion_anio_0 = 0.7
            nueva_config.cajas_por_anio[0][tipo] = int(max_cajas * proporcion_anio_0)
            nueva_config.cajas_por_anio[1][tipo] = max_cajas
            for a in range(2, 5):
                nueva_config.cajas_por_anio[a][tipo] = max_cajas
    
    elif estrategia == "rampa_gradual":
        # Distribución uniforme en 5 años
        for tipo in LaneType:
            max_cajas = nueva_config.config_estrategica.cajas_por_tipo[tipo]
            for a in range(5):
                nueva_config.cajas_por_anio[a][tipo] = int(max_cajas * (a + 1) / 5)
    
    else:  # step_function
        # 50% en año 0, 100% en año 2
        for tipo in LaneType:
            max_cajas = nueva_config.config_estrategica.cajas_por_tipo[tipo]
            nueva_config.cajas_por_anio[0][tipo] = max_cajas // 2
            nueva_config.cajas_por_anio[1][tipo] = max_cajas // 2
            for a in range(2, 5):
                nueva_config.cajas_por_anio[a][tipo] = max_cajas
    
    if not nueva_config.es_factible():
        return config.copy()
    
    return nueva_config


# ============================================
# OPERADORES OPERACIONALES
# ============================================

def abrir_caja(config: ConfigOperacional, dia: DayType, hora: int, tipo: LaneType) -> ConfigOperacional:
    """Abre 1 caja adicional en un momento específico."""
    nueva_config = config.copy()
    
    cajas_disponibles = nueva_config.config_tactica.cajas_por_anio[nueva_config.anio_actual][tipo]
    cajas_abiertas = set(nueva_config.horarios[dia][hora][tipo])
    
    # Encontrar una caja que no esté abierta
    for id_caja in range(cajas_disponibles):
        if id_caja not in cajas_abiertas:
            nueva_config.horarios[dia][hora][tipo].append(id_caja)
            break
    
    return nueva_config


def cerrar_caja(config: ConfigOperacional, dia: DayType, hora: int, tipo: LaneType) -> ConfigOperacional:
    """Cierra 1 caja en un momento específico."""
    nueva_config = config.copy()
    
    if len(nueva_config.horarios[dia][hora][tipo]) > 0:
        nueva_config.horarios[dia][hora][tipo].pop()
    
    return nueva_config


def swap_horas_adyacentes(config: ConfigOperacional, dia: DayType, hora1: int) -> ConfigOperacional:
    """Intercambia la configuración entre dos horas adyacentes."""
    nueva_config = config.copy()
    hora2 = hora1 + 1
    
    if hora2 < 22:
        nueva_config.horarios[dia][hora1], nueva_config.horarios[dia][hora2] = \
            nueva_config.horarios[dia][hora2], nueva_config.horarios[dia][hora1]
    
    return nueva_config

def generar_vecinos_operacionales_locales(config: ConfigOperacional, max_vecinos: int = 10) -> List[ConfigOperacional]:
    """
    Genera vecinos locales estratégicos:
    - max_vecinos con ABRIR cajas en horarios de mayor carga
    - max_vecinos con CERRAR cajas en horarios de menor carga
    """
    vecinos = []
    
    # Calcular estado de carga para todos los (dia, hora, tipo)
    estados_carga = []
    
    for dia in DayType:
        for hora in range(8, 22):
            for tipo in LaneType:
                # Calcular carga actual (evitar división por cero)
                carriles_activos = sum(len(config.horarios[dia][hora][l]) for l in LaneType)
                if carriles_activos > 0:
                    carga = 1 / (carriles_activos * LAMBDA_POR_HORA[dia][hora])
                else:
                    carga = float('inf')  # Máxima prioridad si no hay carriles
                
                estados_carga.append({
                    'dia': dia,
                    'hora': hora,
                    'tipo': tipo,
                    'carga': carga,
                    'carriles_actuales': len(config.horarios[dia][hora][tipo])
                })
    
    # Ordenar por carga (mayor a menor)
    estados_carga.sort(key=lambda x: x['carga'], reverse=True)
    

    # 1. VECINOS CON ABRIR CAJAS (en horarios de MAYOR carga)
    vecinos_abrir = []
    for estado in estados_carga:
        if len(vecinos_abrir) >= max_vecinos / 2:
            break
            
        dia = estado['dia']
        hora = estado['hora']
        tipo = estado['tipo']
        if len(config.horarios[dia][hora][tipo]) < config.config_tactica.config_estrategica.cajas_por_tipo[tipo]:
            vecino = abrir_caja(config, dia, hora, tipo)
            if vecino.es_factible():
                vecinos_abrir.append(vecino)
                print(f"🟢 ABRIR caja en {dia} {hora}h {tipo.name} - Carga: {estado['carga']:.2f}")

    # 2. VECINOS CON CERRAR CAJAS (en horarios de MENOR carga)
    vecinos_cerrar = []
    for estado in reversed(estados_carga):  # Recorrer de menor a mayor carga
        if len(vecinos_cerrar) >= max_vecinos:
            break
            
        dia = estado['dia']
        hora = estado['hora']
        tipo = estado['tipo']

        def recursividad_cierre(vec, cerradas, dia, hora, tipo):
            if random.random() > 0.25:
                cerradas += 1
                vec = cerrar_caja(vec, dia, hora, tipo)
                return recursividad_cierre(vec, cerradas, dia, hora, tipo)
            return vec, cerradas

        
        # Verificar si podemos cerrar cajas de este tipo (debe haber al menos 1 abierta)
        if estado['carriles_actuales'] > 0:
            vecino = cerrar_caja(config, dia, hora, tipo)
            vecino, cerradas = recursividad_cierre(vecino, 1, dia, hora, tipo)

            if vecino.es_factible():
                vecinos_cerrar.append(vecino)
                print(f"🔴 CERRAR {cerradas} caja en {dia} {hora}h {tipo.name} - Carga: {estado['carga']:.2f}")
    
    # Combinar ambos tipos de vecinos
    vecinos = vecinos_cerrar
    return vecinos


def generar_vecino_operacional_global(config: ConfigOperacional) -> ConfigOperacional:
    """
    Genera un vecino global (cambios drásticos en horarios).
    Estrategias: boost peak hours, reduce valley hours, replicar patrón.
    """
    nueva_config = config.copy()
    estrategia = random.choice(["boost_peak", "reduce_valley", "replicar_patron"])
    
    if estrategia == "boost_peak":
        # Abrir todas las cajas disponibles en horas pico (12-13, 18-20)
        horas_pico = [12, 13, 18, 19, 20]
        for dia in DayType:
            for hora in horas_pico:
                for tipo in LaneType:
                    max_cajas = nueva_config.config_tactica.cajas_por_anio[nueva_config.anio_actual][tipo]
                    nueva_config.horarios[dia][hora][tipo] = list(range(max_cajas))
    
    elif estrategia == "reduce_valley":
        # Cerrar la mitad de las cajas en horas valle (8-10, 21)
        horas_valle = [8, 9, 21]
        for dia in DayType:
            for hora in horas_valle:
                for tipo in LaneType:
                    cajas_abiertas = len(nueva_config.horarios[dia][hora][tipo])
                    if cajas_abiertas > 2:
                        nueva_config.horarios[dia][hora][tipo] = nueva_config.horarios[dia][hora][tipo][:cajas_abiertas // 2]
    
    else:  # replicar_patron
        # Copiar horarios de domingo a días normales (o viceversa)
        dia_origen = random.choice(list(DayType))
        dia_destino = random.choice([d for d in DayType if d != dia_origen])
        nueva_config.horarios[dia_destino] = nueva_config.horarios[dia_origen].copy()
    
    if not nueva_config.es_factible():
        return config.copy()
    
    return nueva_config


# ============================================
# FUNCIÓN GENÉRICA DE GENERACIÓN DE VECINOS
# ============================================

def generar_vecino(config, fase: str, tipo_vecino: str = "local"):
    """
    Función genérica para generar vecinos según la fase.
    
    Args:
        config: ConfigEstrategica | ConfigTactica | ConfigOperacional
        fase: "estrategica" | "tactica" | "operacional"
        tipo_vecino: "local" | "global"
    
    Returns:
        Configuración vecina (o lista de vecinos si tipo_vecino="local")
    """
    if fase == "estrategica":
        if tipo_vecino == "local":
            return generar_vecinos_estrategicos_locales(config)
        else:
            return generar_vecino_estrategico_global(config)
    
    elif fase == "tactica":
        if tipo_vecino == "local":
            return generar_vecinos_tacticos_locales(config)
        else:
            return generar_vecino_tactico_global(config)
    
    elif fase == "operacional":
        if tipo_vecino == "local":
            return generar_vecinos_operacionales_locales(config)
        else:
            return generar_vecino_operacional_global(config)
    
    else:
        raise ValueError(f"Fase desconocida: {fase}")


if __name__ == "__main__":
    # Test: generar vecinos estratégicos
    from estructuras_datos import crear_config_actual
    
    config_actual = crear_config_actual()
    print("Configuración actual:")
    print(config_actual.estrategica)
    
    print("\n--- Vecinos estratégicos locales ---")
    vecinos_local = generar_vecinos_estrategicos_locales(config_actual.estrategica)
    print(f"Generados {len(vecinos_local)} vecinos factibles")
    for i, v in enumerate(vecinos_local[:5]):  # Mostrar solo los primeros 5
        print(f"{i+1}. {v}")
    
    print("\n--- Vecino estratégico global ---")
    vecino_global = generar_vecino_estrategico_global(config_actual.estrategica)
    print(vecino_global)
    
    print("\n--- Vecinos tácticos locales ---")
    vecinos_tacticos = generar_vecinos_tacticos_locales(config_actual.tactica)
    print(f"Generados {len(vecinos_tacticos)} vecinos factibles")
    
    print("\n--- Vecino táctico global ---")
    vecino_tactico_global = generar_vecino_tactico_global(config_actual.tactica)
    print(vecino_tactico_global)

