# ==============================================================================
# TALLER: CONVOLUCIÓN 2D Y FILTRADO ESPACIAL CON RUIDO SINTÉTICO (SESIÓN 4)
# ==============================================================================

import os                        # [Bloque 0] - Módulo OS para validar descriptores de archivos en disco
import cv2                       # [Bloque 0] - OpenCV C++ Core para el procesamiento digital de imágenes
import numpy as np               # [Bloque 0] - NumPy para álgebra lineal y manipulación veloz de matrices en RAM
import matplotlib.pyplot as plt  # [Bloque 0] - Pyplot de Matplotlib para renderizado gráfico y GUI

# 1. CONFIGURACIÓN Y VALIDACIÓN DEL RECURSO
nombre_imagen = '1. AFICHE CREADO.png' # [Bloque 1] - Nombre del recurso gráfico de entrada en el disco duro

if not os.path.exists(nombre_imagen):
    print(f"ERROR: No se encontró la imagen '{nombre_imagen}'.") # [Bloque 1] - Frena el pipeline si no se encuentra el recurso
else:
    # 2. CARGA DE LA MATRIZ TENSORIAL A LA RAM
    img_original = cv2.imread(nombre_imagen)                   # [Bloque 2] - Decodifica el archivo binario a tensor 3D BGR (H, W, 3)
    print("Imagen original cargada exitosamente.")               # [Bloque 2] - Mensaje de telemetría de carga de imagen a memoria

    # 3. INYECCIÓN DE RUIDO SINTÉTICO (SAL Y PIMIENTA)
    img_ruidosa = img_original.copy()                          # [Bloque 3] - Clona el tensor base en RAM para añadirle ruido sin alterar la fuente
    proporcion_ruido = 0.15                                    # [Bloque 3] - Proporción total de píxeles alterados con ruido (15% de la imagen)
    
    alto_img, ancho_img, _ = img_original.shape                # [Bloque 3] - Extrae resolución espacial (Alto, Ancho)
    num_puntos = int(proporcion_ruido * alto_img * ancho_img)  # [Bloque 3] - Cantidad total de píxeles a corromper con ruido
    
    filas_rnd = np.random.randint(0, alto_img, num_puntos, dtype=np.int32)        # [Bloque 3] - Índices de filas aleatorios
    columnas_rnd = np.random.randint(0, ancho_img, num_puntos, dtype=np.int32)    # [Bloque 3] - Índices de columnas aleatorios
    
    mitad = num_puntos // 2                                    # [Bloque 3] - Divide a la mitad los puntos para inyectar sal y pimienta
    
    img_ruidosa[filas_rnd[:mitad], columnas_rnd[:mitad]] = [255, 255, 255] # [Bloque 3] - Asigna color blanco (sal) en coordenadas BGR
    img_ruidosa[filas_rnd[mitad:], columnas_rnd[mitad:]] = [0, 0, 0]       # [Bloque 3] - Asigna color negro (pimienta) en coordenadas BGR
    print("Ruido sintético de Sal y Pimienta inyectado correctamente.")   # [Bloque 3] - Confirma inyección correcta del ruido impulsivo

    # 4. APLICACIÓN DE FILTRADO ESPACIAL (KERNEL 7x7)
    img_media = cv2.blur(img_ruidosa, (7, 7))                  # [Bloque 4] - Filtro de Media con kernel 7x7 para promediar la vecindad lineal
    img_gauss = cv2.GaussianBlur(img_ruidosa, (7, 7), 0)       # [Bloque 4] - Filtro Gaussiano de 7x7 con ponderación normal
    img_mediana = cv2.medianBlur(img_ruidosa, 7)               # [Bloque 4] - Filtro de Mediana de 7x7 para tomar el valor estadístico intermedio
    print("Filtros aplicados. Desplegando dashboard comparativo...")     # [Bloque 4] - Confirma aplicación de los 3 filtros en RAM

    # 5. RENDERIZADO GRÁFICO EN PANTALLA (DASHBOARD COMPARATIVO)
    plt.figure(figsize=(16, 5))                                # [Bloque 5] - Crea un lienzo gráfico de 16x5 pulgadas de tamaño
    
    plt.subplot(1, 4, 1)                                       # [Bloque 5] - Selecciona panel 1 (Original con ruido)
    plt.imshow(cv2.cvtColor(img_ruidosa, cv2.COLOR_BGR2RGB))   # [Bloque 5] - Convierte imagen ruidosa de BGR a RGB para despliegue
    plt.title("1. Original con Ruido (15%)")                   # [Bloque 5] - Asigna título descriptivo al panel 1
    plt.axis('off')                                            # [Bloque 5] - Oculta los ejes numéricos cartesianos

    plt.subplot(1, 4, 2)                                       # [Bloque 5] - Selecciona el segundo panel (Filtro de Media)
    plt.imshow(cv2.cvtColor(img_media, cv2.COLOR_BGR2RGB))     # [Bloque 5] - Convierte imagen filtrada por Media de BGR a RGB
    plt.title("2. Filtro de Media (7x7)")                      # [Bloque 5] - Asigna título explicativo al panel 2
    plt.axis('off')                                            # [Bloque 5] - Desactiva ejes cartesianos

    plt.subplot(1, 4, 3)                                       # [Bloque 5] - Selecciona el tercer panel (Filtro Gaussiano)
    plt.imshow(cv2.cvtColor(img_gauss, cv2.COLOR_BGR2RGB))     # [Bloque 5] - Convierte imagen procesada por Gaussiano de BGR a RGB
    plt.title("3. Filtro Gaussiano (7x7)")                     # [Bloque 5] - Coloca el título descriptivo en el panel 3
    plt.axis('off')                                            # [Bloque 5] - Oculta los ejes numéricos

    plt.subplot(1, 4, 4)                                       # [Bloque 5] - Selecciona el cuarto panel (Filtro de Mediana)
    plt.imshow(cv2.cvtColor(img_mediana, cv2.COLOR_BGR2RGB))   # [Bloque 5] - Convierte imagen procesada por Mediana de BGR a RGB
    plt.title("4. Filtro de Mediana (7x7)")                    # [Bloque 5] - Coloca el título explicativo en panel 4
    plt.axis('off')                                            # [Bloque 5] - Deshabilita ejes de coordenadas cartesianas

    plt.tight_layout()                                         # [Bloque 5] - Ajusta automáticamente el espacio y márgenes entre paneles
    plt.show()                                                 # [Bloque 5] - Bloquea el hilo principal y despliega la interfaz GUI