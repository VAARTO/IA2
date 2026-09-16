# ==============================================================================
# TALLER 10: ALGORITMO SVM - GEOMETRÍA DEL HIPERPLANO (MARGEN RÍGIDO)
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

print("--- SESIÓN 10: MODELO MATEMÁTICO SVM (SUPPORT VECTOR MACHINE) ---")

# 1. MATRIZ DE DATOS DE ENTRENAMIENTO (Coordenadas Bidimensionales X, Y)
# Estos puntos representan un conjunto de datos linealmente separable.
X_entrenamiento = np.array([
    [1, 2], [2, 3], [3, 3], [2, 1], [3, 2], # Clase 0 (Azul)
    [6, 5], [7, 7], [8, 6], [7, 5], [9, 7]  # Clase 1 (Rojo)
])

# Etiquetas de clase (Clasificación binaria: 0 y 1)
y_etiquetas = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# 2. INICIALIZACIÓN DEL MOTOR SVM
# Se utiliza un Kernel 'linear' para trazar un hiperplano recto.
# C=1000 se usa para emular un "Hard Margin" (Margen Rígido), penalizando severamente cualquier intrusión en la zona de separación.
motor_svm = svm.SVC(kernel='linear', C=1000)

# 3. ENTRENAMIENTO DEL MODELO (Ajuste al hiperplano / FIT)
motor_svm.fit(X_entrenamiento, y_etiquetas)
print("✅ Modelo SVM ajustado y vectores de soporte calculados.")

# 4. EXTRACCIÓN DE PARÁMETROS GEOMÉTRICOS
# Ecuación del hiperplano: w*x + b = 0
vector_w = motor_svm.coef_[0]          # Vector de pesos (w) ortogonal al hiperplano
sesgo_b = motor_svm.intercept_[0]      # Intercepto (b)
vectores_soporte = motor_svm.support_vectors_ # Puntos críticos que definen el margen

print(f"Vector de pesos (w): {vector_w}")
print(f"Sesgo (b): {sesgo_b}")
print(f"Vectores de soporte: \n{vectores_soporte}")

# 5. RENDERIZADO GRÁFICO DE LA GEOMETRÍA DEL SVM
plt.figure(figsize=(8, 6))

# Ploteo de los datos de entrenamiento
plt.scatter(X_entrenamiento[:, 0], X_entrenamiento[:, 1], c=y_etiquetas, cmap=plt.cm.Paired, s=50, edgecolors='k')

# Ploteo del Hiperplano y los Márgenes
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Crear una cuadrícula para evaluar el modelo
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# Calcular la función de decisión para cada punto de la cuadrícula
Z = motor_svm.decision_function(xy).reshape(XX.shape)

# Trazar el hiperplano (Z=0) y los márgenes de soporte (Z=-1, Z=1)
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# Resaltar los vectores de soporte con círculos rojos
ax.scatter(vectores_soporte[:, 0], vectores_soporte[:, 1], s=150, linewidth=2, facecolors='none', edgecolors='r', label='Vectores de Soporte')

plt.title("Geometría de Separación - SVM Lineal (Margen Rígido)")
plt.xlabel("Característica X")
plt.ylabel("Característica Y")
plt.legend()
plt.show()