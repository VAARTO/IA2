from decimal import Decimal, ROUND_HALF_EVEN # [Bloque 0: Importación] - Carga tipos numéricos de precisión fija sin sesgo binario
import numpy as np                            # [Bloque 0: Importación] - Carga motor de álgebra matricial vectorizada en C

# ==========================================
# BLOQUE 1: MODELADO VECTORIAL CONTINUO (MISIONES 1, 2 Y 3)
# ==========================================
v_real_vec = np.array([2.0, 0.01, 25.0, 0.05, 10000.0, 2.0], dtype=np.float64)  # [Bloque 1: Entradas] - Vector columna con magnitudes de referencia | Dimensión: R^6
e_abs_vec = np.array([0.02, 0.02, 0.1, 0.1, 5.0, 5.0], dtype=np.float64)         # [Bloque 1: Entradas] - Vector columna de errores absolutos | Dimensión: R^6

def calcular_error_relativo_vectorizado(v_r: np.ndarray, e_a: np.ndarray) -> np.ndarray: # [Bloque 1: Definición] - Función pura de evaluación vectorizada
    epsilon = 1e-15                                                                       # [Bloque 1: Regularización] - Umbral épsilon contra singularidad | Fórmula: \varepsilon = 10^{-15}
    v_r_seguro = np.where(np.abs(v_r) < epsilon, epsilon, v_r)                            # [Bloque 1: Acondicionamiento] - Filtro condicional para evitar división por cero | Fórmula: v_s = \max(|v_r|, \varepsilon)
    e_p = (np.abs(e_a) / np.abs(v_r_seguro)) * 100.0                                     # [Bloque 1: Cálculo] - Mapeo vectorial de error porcentual | Fórmula: E_p = (|E_a| / |V_real|) * 100
    return e_p                                                                           # [Bloque 1: Salida] - Retorna el vector de errores porcentuales computado en C | Dimensión: R^6

metricas_continuas = calcular_error_relativo_vectorizado(v_real_vec, e_abs_vec)          # [Bloque 1: Ejecución] - Evaluación simultánea mediante registros SIMD

# ==========================================
# BLOQUE 2: MODELADO FINTECH DE ALTA INTEGRIDAD (MISIÓN 4)
# ==========================================
nomina_real = Decimal('5000000.00')   # [Bloque 2: Entradas] - Valor real nómina expresado en precisión arbitraria base-10 | V_real = 5000000.00
micropago_real = Decimal('0.10')      # [Bloque 2: Entradas] - Valor real micropago expresado en precisión arbitraria base-10 | V_real = 0.10
error_constante = Decimal('0.05')     # [Bloque 2: Entradas] - Desviación escalar por truncación de módulo | E_a = 0.05

def calcular_error_financiero(v_r: Decimal, e_a: Decimal) -> Decimal:              # [Bloque 2: Definición] - Algoritmo estricto para prevención de sesgo contable
    if v_r == Decimal('0.0'):                                                      # [Bloque 2: Validación] - Comprobación de división indefinida
        raise ZeroDivisionError("Imposible calcular error relativo sobre balance cero") # [Bloque 2: Control] - Manejo de excepción determinística
    error_porcentual = (e_a.copy_abs() / v_r.copy_abs()) * Decimal('100.0')        # [Bloque 2: Cálculo] - Ratio financiero estricto | Fórmula: E_p = (|E_a| / |V_real|) * 100
    return error_porcentual.quantize(Decimal('0.000001'), rounding=ROUND_HALF_EVEN)# [Bloque 2: Redondeo] - Acotamiento mediante redondeo bancario estocástico Gaussiano

ep_nomina = calcular_error_financiero(nomina_real, error_constante)                 # [Bloque 2: Ejecución] - Computa métrica en nómina empresarial (0.000001%)
ep_micropago = calcular_error_financiero(micropago_real, error_constante)           # [Bloque 2: Ejecución] - Computa métrica en micropago digital (50.000000%)

# ==========================================
# BLOQUE 3: AUDITORÍA DE SALIDAS Y REPORTE TÉCNICO
# ==========================================
etiquetas = [                                                                      # [Bloque 3: Estructura] - Vector de etiquetas descriptivas de pruebas
    "Misión 1 - API Logs", "Misión 1 - API Pagos",                                 # [Bloque 3: Metadatos] - Identificadores de microservicios
    "Misión 2 - Temp. Sensor", "Misión 2 - Servomotor",                            # [Bloque 3: Metadatos] - Identificadores de robótica
    "Misión 3 - Troncal Red", "Misión 3 - Edge Device"                             # [Bloque 3: Metadatos] - Identificadores de telecomunicaciones
]                                                                                  # [Bloque 3: Estructura] - Fin del vector descriptivo

for idx in range(len(etiquetas)):                                                  # [Bloque 3: Bucle] - Itera los resultados continuos computados
    print(f"[{etiquetas[idx]}] -> E_p: {metricas_continuas[idx]:.4f}%")             # [Bloque 3: Impresión] - Despliegue de resultados con cuatro cifras decimales

print(f"[Misión 4 - Nómina FinTech] -> E_p: {ep_nomina}%")                         # [Bloque 3: Impresión] - Emisión de resultado de nómina sin pérdida de precisión
print(f"[Misión 4 - Micropagos App] -> E_p: {ep_micropago}%")                       # [Bloque 3: Impresión] - Emisión de resultado de micropago