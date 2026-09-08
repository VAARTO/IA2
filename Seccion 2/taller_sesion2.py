import numpy as np               # [Bloque 0: Importación] - Carga NumPy para cálculo numérico vectorial y definición de tensores de píxeles
import cv2                       # [Bloque 0: Importación] - Carga OpenCV para decodificación binaria de archivos gráficos y cómputo de histogramas
import matplotlib.pyplot as plt  # [Bloque 0: Importación] - Carga pyplot para el trazado de gráficos y renderizado visual en ventana
import os                        # [Bloque 0: Importación] - Carga utilidades de sistema de archivos para comprobación de existencia en disco

print("--- TALLER 2: TRANSFORMACIÓN DE ESPACIOS ---")  # [Bloque 0: Salida] - Imprime cabecera inicial en consola

# 1. RETO MATEMÁTICO: ESCALA DE GRISES MANUAL
pixel_amarillo = np.array([0, 255, 255])  # [Bloque 1: Vectorización] - Vector 1D en espacio BGR con componentes Azul=0, Verde=255, Rojo=255 | \vec{p} \in \mathbb{R}^3

B = pixel_amarillo[0]  # [Bloque 1: Extracción] - Indexa la componente escalar correspondiente al canal Azul (B)
G = pixel_amarillo[1]  # [Bloque 1: Extracción] - Indexa la componente escalar correspondiente al canal Verde (G)
R = pixel_amarillo[2]  # [Bloque 1: Extracción] - Indexa la componente escalar correspondiente al canal Rojo (R)

# Aplica la fórmula estandarizada de luminancia CIE/ITU-R BT.601 por ponderación fotópica de conos oculares
Y_calculado = (0.114 * B) + (0.587 * G) + (0.299 * R)  # [Bloque 2: Producto Escalar] - Pondera la intensidad lumínica según la sensibilidad ocular | Fórmula: Y = 0.114B + 0.587G + 0.299R
print(f"1. Valor manual del amarillo en escala de grises: {Y_calculado}")  # [Bloque 2: Salida] - Muestra el nivel de gris calculado para el píxel analizado

print("\n--- LABORATORIO 2: ANÁLISIS ESTADÍSTICO (HISTOGRAMA) ---")  # [Bloque 3: Salida] - Imprime cabecera de la etapa de análisis espectral

nombre_imagen = '1. AFICHE CREADO.png'  # [Bloque 3: Asignación] - Almacena la cadena literal con el nombre exacto del recurso en disco

# 2. VALIDACIÓN DE ARCHIVO
if not os.path.exists(nombre_imagen):  # [Bloque 4: Verificación I/O] - Comprueba si el descriptor del archivo físico está presente en el directorio
    print(f"ERROR: No se encontró '{nombre_imagen}'.")  # [Bloque 4: Control de Error] - Emite notificación en consola en caso de ruta fallida
else:  # [Bloque 4: Bifurcación] - Flujo que se ejecuta si la imagen existe físicamente
    
    # 3. LECTURA DE LA IMAGEN
    imagen_color = cv2.imread(nombre_imagen)  # [Bloque 5: Decodificación] - Lee los bytes del archivo PNG y los transforma a un tensor 3D de NumPy (H, W, 3) en orden BGR
    print("2. Imagen cargada correctamente. Procesando canales...")  # [Bloque 5: Salida] - Emite mensaje de confirmación de carga

    # 4. EXTRACCIÓN DE CANALES (SLICING DE TENSORES)
    canal_b = imagen_color[:, :, 0]  # [Bloque 6: Slicing] - Extrae la matriz bidimensional del canal Azul mediante corte de tensor en eje 2 (índice 0)
    canal_g = imagen_color[:, :, 1]  # [Bloque 6: Slicing] - Extrae la matriz bidimensional del canal Verde mediante corte de tensor en eje 2 (índice 1)
    canal_r = imagen_color[:, :, 2]  # [Bloque 6: Slicing] - Extrae la matriz bidimensional del canal Rojo mediante corte de tensor en eje 2 (índice 2)

    # 5. CÁLCULO ESTADÍSTICO
    # cv2.calcHist evalúa la masa de probabilidad discreta acumulando frecuencias por nivel de intensidad [0 a 255]
    hist_b = cv2.calcHist([canal_b], [0], None, [256], [0, 256])  # [Bloque 7: Conteo Estadístico] - Histograma canal Azul | Fórmula: H_B(k) = \sum_{x,y} \delta(I_B(x,y) - k)
    hist_g = cv2.calcHist([canal_g], [0], None, [256], [0, 256])  # [Bloque 7: Conteo Estadístico] - Histograma canal Verde | Fórmula: H_G(k) = \sum_{x,y} \delta(I_G(x,y) - k)
    hist_r = cv2.calcHist([canal_r], [0], None, [256], [0, 256])  # [Bloque 7: Conteo Estadístico] - Histograma canal Rojo | Fórmula: H_R(k) = \sum_{x,y} \delta(I_R(x,y) - k)

    print("3. Histogramas calculados. Abriendo gráfica...")  # [Bloque 7: Salida] - Emite aviso previo a la generación de la ventana gráfica
    
    # 6. RENDERIZADO VISUAL DE LA GRÁFICA
    plt.figure(figsize=(10, 5))  # [Bloque 8: Creación de Figura] - Inicializa el lienzo gráfico con proporción de 10x5 pulgadas
    
    plt.plot(hist_b, color='blue', label='Canal Azul')   # [Bloque 8: Mapeo de Curva] - Traza los puntos de frecuencia para la banda cromática azul
    plt.plot(hist_g, color='green', label='Canal Verde') # [Bloque 8: Mapeo de Curva] - Traza los puntos de frecuencia para la banda cromática verde
    plt.plot(hist_r, color='red', label='Canal Rojo')    # [Bloque 8: Mapeo de Curva] - Traza los puntos de frecuencia para la banda cromática roja
    
    plt.title("Distribución de Intensidades (Histograma RGB)")  # [Bloque 9: Etiquetado] - Establece el título representativo en la cabecera del gráfico
    plt.xlabel("Valor del Pixel (0 oscuro - 255 claro)")        # [Bloque 9: Etiquetado] - Asigna la descripción del eje horizontal (abscisas k \in [0, 255])
    plt.ylabel("Frecuencia (Cantidad de píxeles)")              # [Bloque 9: Etiquetado] - Asigna la descripción del eje vertical (ordenadas con conteo absoluto)
    plt.legend()                                                # [Bloque 9: Anotación] - Despliega el cuadro con los identificadores de cada canal
    
    plt.show()  # [Bloque 10: Despacho GUI] - Abre la ventana interactiva en el sistema operativo y suspende el hilo hasta su cierre