import numpy as np  # [Bloque 0: Importación] - Carga NumPy para manipulación de tensores y álgebra matricial continua en memoria C

print("--- TALLER 1: TRANSFORMACIONES AFINES ---")  # [Bloque 0: Salida] - Muestra en la terminal el encabezado de la sección 1

# 1. MATRIZ SOBREEXPUESTA
A = np.random.randint(200, 255, (5, 5))  # [Bloque 1: Generación] - Muestreo pseudoaleatorio discreto en rango semiabierto [200, 255) | Fórmula: A_{i,j} \sim \mathcal{U}\{200, 254\}
print("Matriz Original:\n", A)          # [Bloque 1: Salida] - Vuelca en la terminal la matriz de 5x5 simulando una captura con exceso de luz

# 2. OPERACIONES LINEALES
alpha = 0.5   # [Bloque 2: Parámetros] - Factor escalar de ganancia para comprimir contraste al 50% | Fórmula: \alpha = 0.5
beta = -50.0  # [Bloque 2: Parámetros] - Escalar de sesgo para atenuar luminosidad global en 50 unidades | Fórmula: \beta = -50.0

A_nueva = (alpha * A) + beta  # [Bloque 2: Transformación Afín] - Mapeo afín punto a punto en el tensor bidimensional | Fórmula: A'_{i,j} = \alpha \cdot A_{i,j} + \beta

# 3. ACOTAMIENTO (CLIPPING)
A_nueva = np.clip(A_nueva, 0, 255)      # [Bloque 3: Regularización] - Función de saturación para truncar cotas fuera de rango | Fórmula: f(x) = \min(\max(x, 0), 255)
A_nueva = A_nueva.astype(np.uint8)      # [Bloque 3: Cuantización] - Convierte coma flotante a enteros sin signo de 8 bits (rango estándar 0-255)
print("\nMatriz Procesada:\n", A_nueva)  # [Bloque 3: Salida] - Imprime la matriz ajustada con rango dinámico normalizado

print("\n--- LABORATORIO FINAL: KERNEL Y PRODUCTO HADAMARD ---")  # [Bloque 4: Salida] - Imprime cabecera de la sección de convolución

# 1. CREACIÓN DE MATRICES
I = np.array([[100, 100, 100], [100, 200, 100], [100, 100, 100]])  # [Bloque 5: Inicialización] - Declara la matriz de vecindad local de la imagen (I \in \mathbb{R}^{3 \times 3})
K = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])                # [Bloque 5: Inicialización] - Declara el kernel Laplaciano de realce de bordes (K \in \mathbb{R}^{3 \times 3})

# 2. SIMULADOR DE CONVOLUCIÓN
producto_hadamard = I * K  # [Bloque 6: Producto de Hadamard] - Multiplicación elemento a elemento de matrices del mismo orden | Fórmula: (I \circ K)_{i,j} = I_{i,j} \cdot K_{i,j}

# 3. RESULTADO DEL PÍXEL
pixel_central = np.sum(producto_hadamard)  # [Bloque 7: Reducción Tensorial] - Sumatoria de todos los elementos del producto de Schur | Fórmula: S = \sum_{i=1}^{3} \sum_{j=1}^{3} (I \circ K)_{i,j}
print("Valor del píxel central calculado:", pixel_central)  # [Bloque 7: Salida] - Despliega el escalar obtenido tras la operación de convolución discreta