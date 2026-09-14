# ==============================================================================
# TALLER: EXTRACCIÓN DE CARACTERÍSTICAS Y CONTORNOS (SESIÓN 6)
# ==============================================================================

import os                        # [Bloque 0: Entorno] - Módulo OS para validar rutas de archivos físicos y prevenir excepciones I/O.
import cv2                       # [Bloque 0: Motor Visión C++] - Ejecuta algoritmos topológicos (Suzuki, 1985) a nivel de hardware.
import numpy as np               # [Bloque 0: Motor Matemático] - Estructuras tensoriales en RAM contigua (arrays multidimensionales).
import matplotlib.pyplot as plt  # [Bloque 0: Renderizado] - Motor gráfico para superposición de vectores sobre el tensor original.

print("--- INICIANDO MOTOR DE EXTRACCIÓN DE CONTORNOS (SESIÓN 6) ---") # [Bloque 0: Telemetría] - Arranque de ejecución.

nombre_imagen = '1. AFICHE CREADO.png' # [Bloque 1: I/O Config] - String con el path estricto. (Asegúrate de tener la imagen).

# [Bloque 1: Blindaje UTF-8 y Carga] - Lectura de bytes puros para evitar colapsos por tildes en el backend C++ de Windows.
if not os.path.exists(nombre_imagen):  # [Bloque 1: Validación] - Comprueba descriptor de archivo a nivel de sistema operativo.
    print(f"ERROR CRÍTICO: El archivo '{nombre_imagen}' no se encuentra.") # [Bloque 1: Excepciones] - Frenado seguro.
else:
    # [Bloque 2: Decodificación RAM] - np.fromfile transfiere O(N) bytes a RAM. cv2.imdecode los traduce a tensor RGB.
    array_bytes = np.fromfile(nombre_imagen, dtype=np.uint8) # [Bloque 2: Memoria] - Array lineal 1D de bytes (0-255).
    imagen_color = cv2.imdecode(array_bytes, cv2.IMREAD_COLOR) # [Bloque 2: Tensor] - Reconstrucción a matriz 3D (BGR).
    
    # [Bloque 3: Mapeo Espectral] - BGR a Escala de Grises (Y = 0.299R + 0.587G + 0.114B).
    imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY) # [Bloque 3: Procesamiento] - Matriz 2D de intensidades fotópicas.

    # [Bloque 4: Umbralización Bimodal] - Binarización (Thresholding) para preparar la matriz morfológica (0 y 255).
    _, binarizada = cv2.threshold(imagen_gris, 127, 255, cv2.THRESH_BINARY) # [Bloque 4: Lógica Binaria] - Máscara de alto contraste.

    # [Bloque 5: Gradientes (Opcional pero recomendado para afinar)] - Filtro Canny para derivar bordes nítidos.
    bordes = cv2.Canny(binarizada, 100, 200) # [Bloque 5: Derivadas Espaciales] - Resalta transiciones abruptas G = √(Gx² + Gy²).

    # [Bloque 6: Topología Matemática] - Algoritmo de Suzuki. Encuentra los límites externos de los objetos detectados.
    contornos, jerarquia = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # [Bloque 6: Extracción] - Búsqueda en C++.
    
    # [Bloque 7: Geometría Analítica] - Copiamos el tensor original para dibujar sin alterar la memoria base.
    imagen_resultados = imagen_color.copy() # [Bloque 7: Clonación RAM] - Evita la mutación no deseada de las matrices.
    
    # [Bloque 8: Renderizado Vectorial] - Traza los polígonos sobre la matriz RGB (Color verde, grosor 2px).
    cv2.drawContours(imagen_resultados, contornos, -1, (0, 255, 0), 2) # [Bloque 8: Rasterización C++] - Pasa de vectores a píxeles.
    
    # [Bloque 9: Corrección Visual Matplotlib] - Matplotlib usa RGB, OpenCV usa BGR. Se debe permutar el canal 0 y 2.
    imagen_resultados_rgb = cv2.cvtColor(imagen_resultados, cv2.COLOR_BGR2RGB) # [Bloque 9: Espectro] - Corrección visual de display.
    
    # [Bloque 10: Interfaz Gráfica (Plot)] - Creación del panel de comparación estructurado.
    plt.figure(figsize=(12, 6)) # [Bloque 10: Ventana] - Dimensión del canvas en memoria.
    
    plt.subplot(1, 2, 1) # [Bloque 10: Grilla Izquierda] - Panel 1.
    plt.imshow(bordes, cmap='gray') # [Bloque 10: Mapeo] - Matriz booleana de bordes.
    plt.title('Máscara de Bordes (Canny)') # [Bloque 10: Metadatos] - Título analítico.
    plt.axis('off') # [Bloque 10: Limpieza] - Oculta ejes cartesianos.
    
    plt.subplot(1, 2, 2) # [Bloque 10: Grilla Derecha] - Panel 2.
    plt.imshow(imagen_resultados_rgb) # [Bloque 10: Mapeo] - Tensor procesado y superpuesto.
    plt.title(f'Características: {len(contornos)} Contornos Hallados') # [Bloque 10: Telemetría] - Muestra el conteo exacto detectado.
    plt.axis('off') # [Bloque 10: Limpieza] - Oculta ejes cartesianos.
    
    plt.tight_layout() # [Bloque 10: Optimización UI] - Ajusta márgenes paramétricos.
    plt.show() # [Bloque 10: Ejecución Hilo] - Levanta ventana asíncrona y bloquea el script hasta su cierre.