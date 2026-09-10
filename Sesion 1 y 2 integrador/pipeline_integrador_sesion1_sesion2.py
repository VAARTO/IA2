# ==============================================================================
# PIPELINE INTEGRADOR: VISIÓN POR COMPUTADORA Y ÁLGEBRA LINEAL (TALLERES 1 Y 2)
# ==============================================================================

import numpy as np               # [Bloque 0: Runtime] - Inicializa el álgebra matricial C-API ndarray | Gestión contigua en RAM
import cv2                       # [Bloque 0: Motor Visión] - Inicializa OpenCV C++ core para operaciones I/O y filtrado espacial
import matplotlib.pyplot as plt  # [Bloque 0: Visualización] - Motor de renderizado multipanel para despliegue de resultados
import os                        # [Bloque 0: Sistema] - Interfaz con el sistema operativo para verificación de archivos

print("--- INICIANDO PIPELINE DE INTEGRACIÓN (TALLER 1 + TALLER 2) ---")  # [Bloque 0: Telemetría] - Cabecera de inicio

nombre_imagen = '1. AFICHE CREADO.png'  # [Bloque 0: Configuración] - Descriptor de cadena literal de la imagen objetivo

# 1. VERIFICACIÓN DE ARCHIVO
if not os.path.exists(nombre_imagen):  # [Bloque 1: Validación I/O] - Llamada al descriptor Win32/POSIX para validar presencia en disco
    print(f"ERROR: No se encontró la imagen '{nombre_imagen}'.")  # [Bloque 1: Manejo de Excepción] - Emisión de advertencia en caso de fallo
else:  # [Bloque 1: Bifurcación] - Continuación de ejecución si la imagen existe
    # 2. CARGA DE IMAGEN (TALLER 2)
    # OpenCV decodifica la imagen en un tensor 3D de NumPy con formato entrelazado BGR
    img_bgr = cv2.imread(nombre_imagen)  # [Bloque 2: Decodificación Binaria] - Carga estructura cv::Mat en RAM | Dimensión: (H, W, 3)
    print("1. Imagen original cargada.")  # [Bloque 2: Telemetría] - Notificación de tensor cargado

    # 3. TRANSFORMACIÓN ESPACIAL A ESCALA DE GRISES PONDERADA (TALLER 2)
    # Extracción de vistas matriciales mediante strides (sin duplicación en memoria)
    B = img_bgr[:, :, 0]  # [Bloque 3: Slicing Tensorial] - Aísla canal Azul (\lambda \approx 450 nm) con offset 0, salto de 3 bytes
    G = img_bgr[:, :, 1]  # [Bloque 3: Slicing Tensorial] - Aísla canal Verde (\lambda \approx 540 nm) con offset 1, salto de 3 bytes
    R = img_bgr[:, :, 2]  # [Bloque 3: Slicing Tensorial] - Aísla canal Rojo (\lambda \approx 700 nm) con offset 2, salto de 3 bytes
    
    # Aplicación de la ecuación de luminancia según norma CIE / ITU-R BT.601
    img_gris = (0.114 * B) + (0.587 * G) + (0.299 * R)  # [Bloque 3: Proyección Fotópica] - Combinación lineal ponderada | Fórmula: Y = 0.114B + 0.587G + 0.299R
    
    # Acotamiento estricto contra desbordamiento y conversión de tipos
    img_gris = np.clip(img_gris, 0, 255).astype(np.uint8)  # [Bloque 3: Regularización y Cuantización] - Proyección convexa y cast a 8 bits | Fórmula: sat(x) = \min(\max(x, 0), 255)
    print("2. Imagen transformada a Escala de Grises Ponderada.")  # [Bloque 3: Telemetría] - Notificación de luminancia procesada

    # 4. TRANSFORMACIÓN AFÍN: BRILLO Y CONTRASTE (TALLER 1)
    alpha = 0.5   # [Bloque 4: Parametrización] - Factor de ganancia de contraste (\alpha \in \mathbb{R}^+) | Fórmula: \alpha = 0.5 (Compresión al 50%)
    beta = -50.0  # [Bloque 4: Parametrización] - Escalar de traslación del brillo (\beta \in \mathbb{R}) | Fórmula: \beta = -50.0 (Desplazamiento negativo)
    
    # Mapeo afín elemento a elemento sobre el campo escalar de intensidades
    img_oscura = (alpha * img_gris) + beta  # [Bloque 4: Mapeo Afín] - Evaluación lineal continua vectorizada | Fórmula: g(x) = \alpha \cdot x + \beta
    img_oscura = np.clip(img_oscura, 0, 255).astype(np.uint8)  # [Bloque 4: Saturación] - Proyección de rango cerrado a entero de 8 bits | Fórmula: sat(x) = \min(\max(x, 0), 255)
    print("3. Imagen procesada mediante Transformación Afín (Reducción de brillo).")  # [Bloque 4: Telemetría] - Notificación de atenuación

    # 5. DETECCIÓN DE TEXTURAS MEDIANTE CONVOLUCIÓN (TALLER 1)
    # Operador diferencial Laplaciano discreto isotrópico de 4 vecinos
    Kernel = np.array([       # [Bloque 5: Inicialización Kernel] - Instanciación de matriz Laplaciana 3x3 | K \in \mathbb{R}^{3 \times 3}, \sum K = 0
        [ 0, -1,  0],         # [Bloque 5: Fila 0] - Vecino superior ortogonal
        [-1,  4, -1],         # [Bloque 5: Fila 1] - Vecinos laterales y nodo central (\nabla^2 f \approx 4f - \sum f_v)
        [ 0, -1,  0]          # [Bloque 5: Fila 2] - Vecino inferior ortogonal
    ])                        # [Bloque 5: Cierre Kernel] - Fin de declaración diferencial
    
    # Convolución discreta 2D evaluada sobre el núcleo C++ de OpenCV
    img_bordes = cv2.filter2D(img_gris, cv2.CV_8U, Kernel)  # [Bloque 5: Filtrado Espacial] - Convolución discreta | Fórmula: (I * K)(x,y) = \sum_u \sum_v I(x-u, y-v) K(u,v)
    print("4. Convolución de Kernel aplicada (Detección de bordes).")  # [Bloque 5: Telemetría] - Notificación de convolución finalizada

    # 6. ANÁLISIS ESTADÍSTICO RESULTANTE (TALLER 2)
    # Acumulación probabilística de frecuencias discretas en memoria C++
    hist_oscuro = cv2.calcHist([img_oscura], [0], None, [256], [0, 256])  # [Bloque 6: Histograma] - Conteo discreto single-pass | Fórmula: H(k) = \sum_{x,y} \delta(img\_oscura(x,y) - k)

    print("5. Procesamiento finalizado. Desplegando resultados visuales...")  # [Bloque 6: Telemetría] - Fin de cómputo algorítmico

    # ==============================================================================
    # RENDERIZADO DEL DASHBOARD VISUAL
    # ==============================================================================
    # Instanciación de lienzo de Matplotlib en malla ortogonal 2x2
    plt.figure(figsize=(14, 10))  # [Bloque 7: Canvas] - Creación de ventana con escala de 14x10 pulgadas a 100 DPI
    plt.suptitle("PIPELINE DE INTEGRACIÓN: ÁLGEBRA LINEAL Y VISIÓN ARTIFICIAL", fontsize=16, fontweight='bold')  # [Bloque 7: Anotación] - Título maestro

    # Cuadro 1: Imagen Original BGR convertida a espacio RGB
    plt.subplot(2, 2, 1)  # [Bloque 8: Subplot 1] - Localización de cuadrante superior izquierdo (Fila 1, Columna 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))  # [Bloque 8: Render] - Conversión de canales BGR a RGB para correspondencia fotográfica
    plt.title("1. Imagen Original (Extracción de Tensores)")  # [Bloque 8: Título] - Etiqueta descriptiva
    plt.axis('off')  # [Bloque 8: Estilo] - Desactiva visualización de ejes coordenados

    # Cuadro 2: Escala de Grises Ponderada
    plt.subplot(2, 2, 2)  # [Bloque 9: Subplot 2] - Localización de cuadrante superior derecho (Fila 1, Columna 2)
    plt.imshow(img_gris, cmap='gray')  # [Bloque 9: Render] - Mapeo matricial a escala de grises fotópica
    plt.title("2. Escala de Grises (Matemática Ponderada)")  # [Bloque 9: Título] - Etiqueta descriptiva
    plt.axis('off')  # [Bloque 9: Estilo] - Supresión de ejes numéricos

    # Cuadro 3: Transformación Afín (Atenuación)
    plt.subplot(2, 2, 3)  # [Bloque 10: Subplot 3] - Localización de cuadrante inferior izquierdo (Fila 2, Columna 1)
    plt.imshow(img_oscura, cmap='gray', vmin=0, vmax=255)  # [Bloque 10: Render] - Visualización fijando límites absolutos [0, 255]
    plt.title("3. Transformación Afín (Brillo -50, Contraste 50%)")  # [Bloque 10: Título] - Etiqueta descriptiva
    plt.axis('off')  # [Bloque 10: Estilo] - Supresión de bordes cartesianos

    # Cuadro 4: Convolución de Bordes
    plt.subplot(2, 2, 4)  # [Bloque 11: Subplot 4] - Localización de cuadrante inferior derecho (Fila 2, Columna 2)
    plt.imshow(img_bordes, cmap='gray')  # [Bloque 11: Render] - Despliegue de mapa de gradientes Laplaciano
    plt.title("4. Detección de Bordes (Producto Hadamard con Kernel)")  # [Bloque 11: Título] - Etiqueta descriptiva
    plt.axis('off')  # [Bloque 11: Estilo] - Supresión de coordenadas

    plt.tight_layout()  # [Bloque 12: Geometría] - Ajuste automático de márgenes para prevenir solapamiento
    plt.show()  # [Bloque 12: Event Loop] - Despliegue interactivo en Windows y suspensión de ejecución hasta cierre