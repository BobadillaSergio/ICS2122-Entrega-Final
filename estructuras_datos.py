"""
Estructuras de datos para las 3 fases del SA Pendular
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum
import copy

# Importar desde tu código existente
import sys
sys.path.append("..")
from load_params.dominios import LaneType, DayType


@dataclass
class ConfigEstrategica:
    """
    Fase Estratégica: Define cuántas cajas de cada tipo construir en total.
    
    Restricciones:
    - sum(cajas_por_tipo.values()) <= 40
    - cajas_por_tipo[LaneType.SELF] % 5 == 0 (múltiplos de 5)
    - Cada tipo <= MAX_POR_ISLA (default: 5 para SELF, ilimitado para otros)
    """
    cajas_por_tipo: Dict[LaneType, int] = field(default_factory=dict)
    
    def __post_init__(self):
        # Inicializar con 0 si no se proporciona
        for lt in LaneType:
            if lt not in self.cajas_por_tipo:
                self.cajas_por_tipo[lt] = 0
    
    @property
    def total_cajas(self) -> int:
        """Total de cajas construidas"""
        return sum(self.cajas_por_tipo.values())
    
    @property
    def num_islas_self(self) -> int:
        """Número de islas de self-checkout (cada una tiene 5 cajas)"""
        return self.cajas_por_tipo[LaneType.SELF] // 5 if self.cajas_por_tipo[LaneType.SELF] > 0 else 0
    
    def es_factible(self, max_total: int = 40, max_por_isla: int = 5) -> bool:
        """Verifica si la configuración es factible"""
        # Total no supera 40
        if self.total_cajas > max_total:
            return False
        
        # Self-checkout en múltiplos de 5
        if self.cajas_por_tipo[LaneType.SELF] % 5 != 0:
            return False
        
        # Todos los valores son no negativos
        if any(n < 0 for n in self.cajas_por_tipo.values()):
            return False
        
        # ✨ RESTRICCIONES DE NEGOCIO: Al menos algunas cajas regulares
        # Justificación: Clientes con muchos ítems (>10) no pueden usar express/self
        if self.cajas_por_tipo[LaneType.REGULAR] < 5:
            return False
        
        # Al menos 2 cajas abiertas en total
        if self.total_cajas < 2:
            return False
        
        return True
    
    def copy(self):
        """Copia profunda de la configuración"""
        return ConfigEstrategica(
            cajas_por_tipo=copy.deepcopy(self.cajas_por_tipo)
        )
    
    def __repr__(self):
        return (f"ConfigEstrategica(total={self.total_cajas}, "
                f"regular={self.cajas_por_tipo[LaneType.REGULAR]}, "
                f"express={self.cajas_por_tipo[LaneType.EXPRESS]}, "
                f"priority={self.cajas_por_tipo[LaneType.PRIORITY]}, "
                f"self={self.cajas_por_tipo[LaneType.SELF]})")


@dataclass
class ConfigTactica:
    """
    Fase Táctica: Define cuántas cajas de cada tipo están activas cada año.
    
    Estructura:
    - cajas_por_anio[año][LaneType] = número de cajas activas ese año
    
    Restricciones:
    - cajas_por_anio[año][tipo] <= config_estrategica.cajas_por_tipo[tipo]
    - cajas_por_anio[año][tipo] <= cajas_por_anio[año+1][tipo] (monotonía, opcional)
    """
    cajas_por_anio: Dict[int, Dict[LaneType, int]] = field(default_factory=dict)
    config_estrategica: ConfigEstrategica = None  # Restricción superior
    
    def __post_init__(self):
        # Inicializar 5 años si no se proporciona
        if not self.cajas_por_anio:
            for anio in range(5):
                self.cajas_por_anio[anio] = {lt: 0 for lt in LaneType}
    
    def es_factible(self, permitir_decrecimiento: bool = False) -> bool:
        """Verifica si la configuración es factible"""
        if self.config_estrategica is None:
            return True  # No hay restricción superior
        
        for anio in range(5):
            for lt in LaneType:
                # No superar capacidad estratégica
                if self.cajas_por_anio[anio][lt] > self.config_estrategica.cajas_por_tipo[lt]:
                    return False
                
                # No negativo
                if self.cajas_por_anio[anio][lt] < 0:
                    return False
                
                # Monotonía (opcional): no se "desconstruyen" cajas
                if not permitir_decrecimiento and anio < 4:
                    if self.cajas_por_anio[anio][lt] > self.cajas_por_anio[anio + 1][lt]:
                        return False
                
                # Self-checkout en múltiplos de 5
                if lt == LaneType.SELF and self.cajas_por_anio[anio][lt] % 5 != 0:
                    return False
        
        return True
    
    def copy(self):
        """Copia profunda de la configuración"""
        return ConfigTactica(
            cajas_por_anio=copy.deepcopy(self.cajas_por_anio),
            config_estrategica=self.config_estrategica.copy() if self.config_estrategica else None
        )
    
    def __repr__(self):
        repr_str = "ConfigTactica:\n"
        for anio in range(5):
            total = sum(self.cajas_por_anio[anio].values())
            repr_str += f"  Año {anio+1}: total={total}, "
            repr_str += ", ".join([f"{lt.value}={self.cajas_por_anio[anio][lt]}" for lt in LaneType])
            repr_str += "\n"
        return repr_str


@dataclass
class ConfigOperacional:
    """
    Fase Operacional: Define qué cajas abrir en cada hora de cada tipo de día.
    
    Estructura:
    - horarios[DayType][hora][LaneType] = [id_caja_1, id_caja_2, ...]
    
    Ejemplo:
    horarios[DayType.NORMAL][9][LaneType.REGULAR] = [0, 1, 5, 7, 10]
    → En días normales a las 9:00, abrir cajas regulares 0,1,5,7,10
    
    Restricciones:
    - len(horarios[dia][hora][tipo]) <= config_tactica.cajas_por_anio[año_actual][tipo]
    - sum(len(horarios[dia][hora][tipo]) for tipo in LaneType) >= 2 (mínimo 2 cajas abiertas)
    - Si len(horarios[dia][hora][LaneType.SELF]) > 0, agregar costo de 2 supervisores
    """
    horarios: Dict[DayType, Dict[int, Dict[LaneType, List[int]]]] = field(default_factory=dict)
    config_tactica: ConfigTactica = None  # Restricción superior
    anio_actual: int = 0  # Año de operación (0-4)
    horarios_por_anio: Dict[int, Dict[DayType, Dict[int, Dict[LaneType, List[int]]]]] = field(default_factory=dict)
    
    def __post_init__(self):
        # Inicializar estructura si no se proporciona
        if not self.horarios:
            self.horarios = self._estructura_vacia()
        # Guardar horarios iniciales (pueden estar vacíos)
        self.guardar_horarios_actual()
    
    def _estructura_vacia(self) -> Dict[DayType, Dict[int, Dict[LaneType, List[int]]]]:
        """Crea una estructura vacía de horarios"""
        estructura = {}
        for dia in DayType:
            estructura[dia] = {}
            for hora in range(8, 22):  # 8:00 - 22:00
                estructura[dia][hora] = {lt: [] for lt in LaneType}
        return estructura
    
    def activar_anio(self, anio: int):
        """Activa un año específico, cargando sus horarios si existen"""
        self.anio_actual = anio
        if anio in self.horarios_por_anio:
            self.horarios = copy.deepcopy(self.horarios_por_anio[anio])
        else:
            # Si no hay horarios para este año, inicializar vacío
            self.horarios = self._estructura_vacia()
    
    def guardar_horarios_actual(self):
        """Guarda los horarios actuales para el año activo"""
        self.horarios_por_anio[self.anio_actual] = copy.deepcopy(self.horarios)
    
    def obtener_horarios_para_anio(self, anio: int) -> Dict[DayType, Dict[int, Dict[LaneType, List[int]]]]:
        """Obtiene los horarios para un año específico"""
        if anio in self.horarios_por_anio:
            return copy.deepcopy(self.horarios_por_anio[anio])
        elif self.horarios and self.anio_actual == anio:
            # Si estamos en ese año, devolver horarios actuales
            return copy.deepcopy(self.horarios)
        else:
            # Si no hay horarios para ese año, devolver estructura vacía
            return self._estructura_vacia()
    
    def es_factible(self, min_cajas_abiertas: int = 2) -> bool:
        """Verifica si la configuración es factible"""
        for dia in DayType:
            for hora in range(8, 22):
                # Al menos 2 cajas abiertas en todo momento
                total_abiertas = sum(len(self.horarios[dia][hora][lt]) for lt in LaneType)
                if total_abiertas < min_cajas_abiertas:
                    return False
                
                # No superar capacidad táctica del año actual
                if self.config_tactica is not None:
                    for lt in LaneType:
                        max_disp = self.config_tactica.cajas_por_anio[self.anio_actual][lt]
                        if len(self.horarios[dia][hora][lt]) > max_disp:
                            return False
                        
                        # IDs de cajas válidos (0 a max_disp-1)
                        for id_caja in self.horarios[dia][hora][lt]:
                            if id_caja < 0 or id_caja >= max_disp:
                                return False
        
        return True
    
    def cajas_abiertas(self, dia: DayType, hora: int, tipo: LaneType) -> int:
        """Número de cajas abiertas en un momento específico"""
        return len(self.horarios[dia][hora][tipo])
    
    def total_cajas_abiertas(self, dia: DayType, hora: int) -> int:
        """Total de cajas abiertas (todos los tipos) en un momento"""
        return sum(len(self.horarios[dia][hora][lt]) for lt in LaneType)
    
    def copy(self):
        """Copia profunda de la configuración"""
        return ConfigOperacional(
            horarios=copy.deepcopy(self.horarios),
            config_tactica=self.config_tactica.copy() if self.config_tactica else None,
            anio_actual=self.anio_actual,
            horarios_por_anio=copy.deepcopy(self.horarios_por_anio)
        )
    
    def __repr__(self):
        repr_str = f"ConfigOperacional (Año {self.anio_actual+1}):\n"
        for dia in DayType:
            repr_str += f"  {dia.value}:\n"
            # Mostrar solo una muestra de horas (8, 12, 18, 21)
            for hora in range(8,22):
                total = self.total_cajas_abiertas(dia, hora)
                repr_str += f"    {hora:02d}:00 - total={total:2d} | "
                repr_str += ", ".join([f"{lt.value}={len(self.horarios[dia][hora][lt])}" for lt in LaneType])
                repr_str += "\n"
        return repr_str


@dataclass
class SolucionCompleta:
    """
    Solución completa del problema (3 fases integradas)
    """
    estrategica: ConfigEstrategica
    tactica: ConfigTactica
    operacional: ConfigOperacional
    
    VAN: float = None  # Valor Actual Neto (objetivo a maximizar)
    kpis: Dict[str, float] = field(default_factory=dict)  # Otros KPIs
    
    def es_factible(self) -> bool:
        """Verifica factibilidad global"""
        return (self.estrategica.es_factible() and 
                self.tactica.es_factible() and 
                self.operacional.es_factible())
    
    def copy(self):
        """Copia profunda de la solución"""
        return SolucionCompleta(
            estrategica=self.estrategica.copy(),
            tactica=self.tactica.copy(),
            operacional=self.operacional.copy(),
            VAN=self.VAN,
            kpis=copy.deepcopy(self.kpis)
        )
    
    def __repr__(self):
        repr_str = "=" * 60 + "\n"
        repr_str += "SOLUCIÓN COMPLETA\n"
        repr_str += "=" * 60 + "\n"
        repr_str += f"VAN: ${self.VAN:,.0f} CLP\n" if self.VAN else "VAN: No evaluado\n"
        repr_str += "\n" + str(self.estrategica) + "\n"
        repr_str += "\n" + str(self.tactica) + "\n"
        repr_str += "\n" + str(self.operacional) + "\n"
        if self.kpis:
            repr_str += "\nKPIs:\n"
            for k, v in self.kpis.items():
                repr_str += f"  {k}: {v:.2f}\n"
        repr_str += "=" * 60 + "\n"
        return repr_str


def ajustar_tactica_a_estrategica(tactica: ConfigTactica) -> ConfigTactica:
    """
    Ajusta la configuración táctica para que sea consistente con la estratégica.
    Limita las cajas por año a las disponibles en la configuración estratégica.
    """
    nueva_tactica = tactica.copy()
    
    if nueva_tactica.config_estrategica is None:
        return nueva_tactica
    
    for anio in range(5):
        for tipo in LaneType:
            max_disponibles = nueva_tactica.config_estrategica.cajas_por_tipo[tipo]
            # print(f"maximas disponibles para año {anio}, tipo {tipo}: {max_disponibles}")
            cajas_actuales = nueva_tactica.cajas_por_anio[anio][tipo]
            # print(f"cajas actuales para año {anio}, tipo {tipo}: {cajas_actuales}")
            
            # Limitar a las disponibles estratégicamente
            nueva_tactica.cajas_por_anio[anio][tipo] = max_disponibles #antes min(cajas_actuales, max_disponibles)
    
    return nueva_tactica


def ajustar_horarios_a_capacidad(operacional: ConfigOperacional) -> ConfigOperacional:
    """
    Ajusta los horarios operacionales para que sean consistentes con la capacidad táctica.
    Si una caja no existe en la configuración táctica, se cierra.
    """
    nuevo_operacional = operacional.copy()
    
    for dia in DayType:
        for hora in range(8, 22):
            for tipo in LaneType:
                # Capacidad disponible según táctica
                max_disponibles = nuevo_operacional.config_tactica.cajas_por_anio[nuevo_operacional.anio_actual][tipo]
                cajas_abiertas = nuevo_operacional.horarios[dia][hora][tipo]
                
                # Filtrar solo las cajas que existen (ID < max_disponibles)
                cajas_validas = [id_caja for id_caja in cajas_abiertas if id_caja < max_disponibles]
                
                # Si tenemos más cajas de las que deberíamos, limitar
                if len(cajas_validas) > max_disponibles:
                    cajas_validas = cajas_validas[:max_disponibles]
                
                nuevo_operacional.horarios[dia][hora][tipo] = cajas_validas
    
    # Guardar horarios ajustados
    nuevo_operacional.guardar_horarios_actual()
    
    return nuevo_operacional


def crear_config_actual() -> SolucionCompleta:
    """
    Crea la configuración actual del supermercado (punto de partida).
    
    MODIFICADO: 40 cajas todas abiertas todo el día (20 regular, 10 express, 5 priority, 5 self)
    """
    # Estratégica: capacidad total de 40 cajas
    estrategica = ConfigEstrategica(
        cajas_por_tipo={
            LaneType.REGULAR: 20,
            LaneType.EXPRESS: 10,
            LaneType.PRIORITY: 5,
            LaneType.SELF: 5
        }
    )
    
    # Táctica: todas las 40 cajas activas desde año 1
    tactica = ConfigTactica(
        cajas_por_anio={
            0: {LaneType.REGULAR: 20, LaneType.EXPRESS: 10, LaneType.PRIORITY: 5, LaneType.SELF: 5},
            1: {LaneType.REGULAR: 20, LaneType.EXPRESS: 10, LaneType.PRIORITY: 5, LaneType.SELF: 5},
            2: {LaneType.REGULAR: 20, LaneType.EXPRESS: 10, LaneType.PRIORITY: 5, LaneType.SELF: 5},
            3: {LaneType.REGULAR: 20, LaneType.EXPRESS: 10, LaneType.PRIORITY: 5, LaneType.SELF: 5},
            4: {LaneType.REGULAR: 20, LaneType.EXPRESS: 10, LaneType.PRIORITY: 5, LaneType.SELF: 5}
        },
        config_estrategica=estrategica
    )
    
    # Operacional: todas las 40 cajas abiertas todo el tiempo (8:00-22:00)
    operacional = ConfigOperacional(
        horarios={},
        config_tactica=tactica,
        anio_actual=0
    )
    
    # Inicializar horarios: todas las cajas abiertas 8:00-22:00
    for dia in DayType:
        operacional.horarios[dia] = {}
        for hora in range(8, 22):
            operacional.horarios[dia][hora] = {
                LaneType.REGULAR: list(range(20)),  # [0,1,2,...,19]
                LaneType.EXPRESS: list(range(10)),  # [0,1,2,...,9]
                LaneType.PRIORITY: list(range(5)),  # [0,1,2,3,4]
                LaneType.SELF: list(range(5))       # [0,1,2,3,4]
            }
    
    return SolucionCompleta(
        estrategica=estrategica,
        tactica=tactica,
        operacional=operacional
    )


if __name__ == "__main__":
    # Test: crear configuración actual
    config_actual = crear_config_actual()
    print(config_actual)
    print(f"\n¿Es factible? {config_actual.es_factible()}")
    
    # Test: crear una configuración no factible
    config_invalida = ConfigEstrategica(
        cajas_por_tipo={
            LaneType.REGULAR: 30,
            LaneType.EXPRESS: 5,
            LaneType.PRIORITY: 5,
            LaneType.SELF: 7  # No es múltiplo de 5!
        }
    )
    print(f"\nConfig inválida: {config_invalida}")
    print(f"¿Es factible? {config_invalida.es_factible()}")

