# ==============================================================================
# TALLER 9: ALGORITMO KNN - K VECINOS MÁS CERCANOS (SESIÓN 9)
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

print("--- SESIÓN 9: MODELO MATEMÁTICO K-NN (K-NEAREST NEIGHBORS) ---")

# 1. MATRIZ DE DATOS DE ENTRENAMIENTO (Coordenadas Bidimensionales)
# Se definen datos adimensionales. Ej: [Ingresos (x1000), Edad] para clasificar si "Compró (1)" o "No Compró (0)"
X_entrenamiento = np.array([
    [10, 20], [15, 25], [12, 22], [11, 21], [13, 24], # Clase 0: No Compra
    [50, 60], [55, 65], [52, 62], [58, 68], [54, 63]  # Clase 1: Compra
])

# Etiquetas de clase correspondientes (0 y 1)
y_etiquetas = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# 2. INICIALIZACIÓN DEL MOTOR MATEMÁTICO
# Definiendo K=3. El algoritmo evaluará los 3 vecinos más próximos usando Distancia Euclidiana por defecto.
knn = KNeighborsClassifier(n_neighbors=3)

# 3. ENTRENAMIENTO DEL MODELO (Ajuste a la data / FIT)
knn.fit(X_entrenamiento, y_etiquetas)
print("✅ Modelo K-NN ajustado y entrenado.")

# 4. DATOS NUEVOS Y PREDICCIÓN MEDIANTE DISTANCIAS
# Puntos ciegos que el algoritmo debe clasificar midiendo la distancia a la matriz de entrenamiento
X_nuevo = np.array([
    [14, 23],  # Debería ser clasificado como 0
    [53, 61],  # Debería ser clasificado como 1
    [30, 40]   # Punto en el medio para probar la votación de fronteras
])

predicciones = knn.predict(X_nuevo)

print("\nResultados de predicción por votación de K-Vecinos:")
for i, punto in enumerate(X_nuevo):
    clase = "Compra (1)" if predicciones[i] == 1 else "No compra (0)"
    print(f"Punto de prueba {punto} -> Clasificado como: {clase}")

# 5. RENDERIZADO GRÁFICO DEL ESPACIO DE DECISIÓN
plt.figure(figsize=(8, 6))

# Ploteo de los datos de entrenamiento
plt.scatter(X_entrenamiento[:, 0], X_entrenamiento[:, 1], c=y_etiquetas, cmap=plt.cm.Paired, s=100, edgecolors='k', label='Datos Entrenamiento')

# Ploteo de los puntos de prueba (Nuevas predicciones) marcados con una 'X'
for i, punto in enumerate(X_nuevo):
    plt.scatter(punto[0], punto[1], color='green' if predicciones[i] == 1 else 'blue', marker='x', s=200, linewidths=3, label='Nuevos Puntos' if i == 0 else "")

plt.title("Clasificación K-NN (K=3) - Distancia Euclidiana")
plt.xlabel("Característica X (Ej: Edad)")
plt.ylabel("Característica Y (Ej: Ingresos)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()