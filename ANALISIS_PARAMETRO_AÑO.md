# Análisis del Parámetro `año` en `evaluar_configuracion_simulacion` y `simular_varios`

## Problema Identificado

Existe una inconsistencia en cómo se interpreta el parámetro `año` en las funciones de evaluación y simulación.

## Flujo Actual

### 1. `evaluar_configuracion_simulacion(solucion, año=0, n_rep=20)`
- **Parámetro `año`**: 0-4 (donde 0 = 2025, 1 = 2026, etc.)
- **Línea 241**: Obtiene configuración para ese año: `solucion.tactica.cajas_por_anio[año]`
- **Línea 248**: Llama a `simular_varios(..., 2025 + año, ...)`
  - Si `año=0` → simula año 2025
  - Si `año=1` → simula año 2026

### 2. `simular_varios(dia, CONFIG, HORARIOS, AÑO, ...)`
- **Parámetro `AÑO`**: Año real (2025, 2026, 2027, etc.)
- Pasa `AÑO` a `SupermarketSimOptimized`

### 3. `SupermarketSimOptimized.__init__(..., año, ...)`
- **`self.año`**: Año real (2025, 2026, etc.)
- **Uso en simulación**:
  - **Línea 503**: `PROYECCION_DEMANDA[self.año]` - Ajusta demanda según el año
  - **Línea 736**: `Gastos_operacionales[self.año]` - Usa costos operacionales del año

### 4. `calcular_van_correcto()`
- **Línea 772-773**: Siempre calcula VAN para 5 años desde t=0:
  ```python
  for año in range(1, 6):  # t=1 a t=5
      van_total += flujo_anual / ((1 + DISCOUNT_RATE_ANUAL) ** año)
  ```
- **Problema**: Usa `flujo_anual` calculado del año simulado, pero siempre proyecta desde t=0 (2025)

## Inconsistencia

**Escenario 1: `año=0` (2025)**
- Simula: 2025
- Calcula VAN: 2025-2029 (5 años) ✓ **CORRECTO**

**Escenario 2: `año=1` (2026)**
- Simula: 2026 (con demanda y costos de 2026)
- Calcula VAN: 2025-2029 (5 años) ✗ **INCORRECTO**
- **Debería calcular**: 2026-2030 (5 años desde 2026)

## Interpretación Correcta

El parámetro `año` debería ser **el año de inicio** para el cálculo del VAN:
- `año=0` → VAN para 2025-2029 (5 años)
- `año=1` → VAN para 2026-2030 (5 años)
- `año=2` → VAN para 2027-2031 (5 años)

## Problema Adicional

Incluso si se corrige el año de inicio, `calcular_van_correcto()` usa un **flujo anual constante** para los 5 años, cuando debería:
1. Variar el flujo por año (demanda y costos cambian)
2. O al menos usar el flujo del año simulado como base y proyectar con crecimiento

## Recomendación

1. **Opción A (Simple)**: `año` es siempre el año de inicio (2025), y siempre se calcula VAN para 2025-2029
   - Cambiar `evaluar_configuracion_simulacion` para siempre usar `año=0` o eliminar el parámetro

2. **Opción B (Correcta)**: `año` es el año de inicio, y `calcular_van_correcto` debe:
   - Calcular VAN para 5 años desde `self.año` (no siempre desde 2025)
   - Variar flujos por año usando `PROYECCION_DEMANDA` y `Gastos_operacionales`

3. **Opción C (Híbrida)**: `año` solo afecta la simulación (demanda/costos), pero VAN siempre se calcula para 2025-2029
   - Documentar claramente este comportamiento

## Uso en el Código

### Llamadas a `evaluar_configuracion_simulacion`:
- **Línea 520**: `año=0` (evaluación inicial)
- **Línea 644**: `año=0` (evaluación de ciclo)
- **Línea 329**: `año` pasado desde `evaluar_fase_simulacion`

### Llamadas a `SA_fase_simulacion`:
- **Línea 553**: `año=4` (FASE ESTRATÉGICA - evalúa año 2029)
- **Línea 575**: `año=0` (FASE TÁCTICA - evalúa año 2025)
- **Línea 607**: `año=0` (FASE OPERACIONAL - evalúa año 2025)

### Inconsistencia Detectada:
- **FASE ESTRATÉGICA** usa `año=4` (2029), pero `calcular_van_correcto()` siempre calcula VAN desde 2025
- Esto significa que cuando se evalúa con `año=4`, se simula 2029 pero se calcula VAN para 2025-2029 (incorrecto)

## Código Actual Relevante

```python
# heuristica_3_etapas.py línea 248
kpi = simular_varios(dia, config_caja, horarios, 2025 + año, n_rep=n_rep)

# main_rapido_optimizado.py línea 772-773
for año in range(1, 6):  # t=1 a t=5 - SIEMPRE desde 2025
    van_total += flujo_anual / ((1 + DISCOUNT_RATE_ANUAL) ** año)
```

## Conclusión

El parámetro `año` tiene **dos interpretaciones conflictivas**:

1. **En `evaluar_configuracion_simulacion`**: `año` (0-4) se usa para:
   - Seleccionar configuración: `solucion.tactica.cajas_por_anio[año]`
   - Simular ese año: `2025 + año`
   - Afectar demanda y costos del año simulado

2. **En `calcular_van_correcto`**: Siempre calcula VAN para 5 años desde 2025 (t=0), ignorando el año simulado

**Problema**: Si `año=4` (2029), se simula 2029 pero se calcula VAN para 2025-2029, lo cual no tiene sentido.

**Solución recomendada**: 
- Si `año` es el año de inicio → `calcular_van_correcto` debe calcular desde `self.año` (no siempre desde 2025)
- Si `año` solo afecta simulación → Documentar que VAN siempre es para 2025-2029, y cambiar FASE ESTRATÉGICA para usar `año=0`

