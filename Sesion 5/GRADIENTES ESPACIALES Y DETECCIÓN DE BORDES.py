# ==============================================================================
# TALLER: GRADIENTES ESPACIALES Y DETECCIÓN DE BORDES (SESIÓN 5)
# ==============================================================================

import os                        # [Bloque 0: Entorno] - Módulo OS para validar descriptores de archivos físicos en el sistema.
import cv2                       # [Bloque 0: Motor Visión C++] - Ejecuta convoluciones y cálculo diferencial a nivel de hardware.
import numpy as np               # [Bloque 0: Motor Matemático] - Estructuras tensoriales para magnitudes de gradientes en RAM.
import matplotlib.pyplot as plt  # [Bloque 0: Renderizado] - Motor gráfico para visualizar las variaciones de alta frecuencia.

print("--- INICIANDO MOTOR DE GRADIENTES Y BORDES (SESIÓN 5) ---") # [Bloque 0: Telemetría] - Mensaje de arranque del sistema.

nombre_imagen = '1. AFICHE CREADO.png' # [Bloque 1: I/O Config] - String con la ruta estricta del recurso gráfico.

# 1. VERIFICACIÓN DE ARCHIVO (BLINDAJE)
if not os.path.exists(nombre_imagen):  # [Bloque 1: Validación] - Llama a Win32 API para prevenir excepciones NoneType.
    print(f"ERROR CRÍTICO: No se encontró la imagen '{nombre_imagen}'.")
else:
    # 2. CARGA Y PROYECCIÓN FOTÓPICA
    img_bgr = cv2.imread(nombre_imagen)                   # [Bloque 2: Carga RAM] - Extrae matriz 3D BGR.
    img_gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)  # [Bloque 2: Grises] - Reduce dimensionalidad para derivación espacial 2D.
    print("1. Tensor original cargado y proyectado a escala de grises.")

    # 3. CÁLCULO DE GRADIENTES: OPERADOR SOBEL
    # BLINDAJE: Se usa cv2.CV_64F (float64) para no perder las pendientes negativas (bordes oscuros).
    grad_x_sobel = cv2.Sobel(img_gris, cv2.CV_64F, 1, 0, ksize=3) # [Bloque 3: Derivada X] - Detecta bordes verticales (∂f/∂x).
    grad_y_sobel = cv2.Sobel(img_gris, cv2.CV_64F, 0, 1, ksize=3) # [Bloque 3: Derivada Y] - Detecta bordes horizontales (∂f/∂y).
    
    # Rectificación: Convierte los números flotantes signados en valores absolutos de 8 bits.
    abs_grad_x_s = cv2.convertScaleAbs(grad_x_sobel)              # [Bloque 3: Norma X] - |G_x|
    abs_grad_y_s = cv2.convertScaleAbs(grad_y_sobel)              # [Bloque 3: Norma Y] - |G_y|
    sobel_mag = cv2.addWeighted(abs_grad_x_s, 0.5, abs_grad_y_s, 0.5, 0) # [Bloque 3: Fusión Sobel] - Magnitud aproximada |G| ≈ 0.5|G_x| + 0.5|G_y|.
    print("2. Derivadas de Sobel calculadas (Suavizado central).")

    # 4. CÁLCULO DE GRADIENTES: OPERADOR PREWITT
    # Kernels manuales ya que OpenCV no tiene función directa de Prewitt (pesos uniformes sin centro realzado).
    kernelx_prewitt = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32) # [Bloque 4: Kernel X Prewitt]
    kernely_prewitt = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32) # [Bloque 4: Kernel Y Prewitt]
    
    grad_x_prewitt = cv2.filter2D(img_gris, cv2.CV_64F, kernelx_prewitt)               # [Bloque 4: Filtro X] - Convolución matricial ∂x.
    grad_y_prewitt = cv2.filter2D(img_gris, cv2.CV_64F, kernely_prewitt)               # [Bloque 4: Filtro Y] - Convolución matricial ∂y.
    prewitt_mag = cv2.convertScaleAbs(grad_x_prewitt) + cv2.convertScaleAbs(grad_y_prewitt) # [Bloque 4: Fusión Prewitt] - Magnitud total.
    print("3. Convoluciones de Prewitt calculadas (Vecindad uniforme).")

    # 5. CÁLCULO DE GRADIENTES: OPERADOR ROBERTS
    # Kernels cruzados de 2x2 para detectar alta frecuencia (muy sensibles al ruido).
    kernelx_roberts = np.array([[1, 0], [0, -1]], dtype=np.float32) # [Bloque 5: Kernel X Roberts]
    kernely_roberts = np.array([[0, 1], [-1, 0]], dtype=np.float32) # [Bloque 5: Kernel Y Roberts]
    
    grad_x_roberts = cv2.filter2D(img_gris, cv2.CV_64F, kernelx_roberts)               # [Bloque 5: Filtro X]
    grad_y_roberts = cv2.filter2D(img_gris, cv2.CV_64F, kernely_roberts)               # [Bloque 5: Filtro Y]
    roberts_mag = cv2.convertScaleAbs(grad_x_roberts) + cv2.convertScaleAbs(grad_y_roberts) # [Bloque 5: Fusión Roberts]
    print("4. Convoluciones de Roberts cruzadas calculadas.")

    # 6. RENDERIZADO GRÁFICO (DASHBOARD COMPARATIVO)
    plt.figure(figsize=(14, 10))                                       # [Bloque 6: Canvas] - Crea lienzo gráfico.
    plt.suptitle("SESIÓN 5: GRADIENTES ESPACIALES Y BORDES", fontsize=16, fontweight='bold') # [Bloque 6: Etiqueta maestra]

    plt.subplot(2, 2, 1); plt.imshow(img_gris, cmap='gray'); plt.title("1. Escala de Grises Original"); plt.axis('off')
    plt.subplot(2, 2, 2); plt.imshow(sobel_mag, cmap='gray'); plt.title("2. Gradiente Sobel (Con Suavizado)"); plt.axis('off')
    plt.subplot(2, 2, 3); plt.imshow(prewitt_mag, cmap='gray'); plt.title("3. Gradiente Prewitt (Peso Uniforme)"); plt.axis('off')
    plt.subplot(2, 2, 4); plt.imshow(roberts_mag, cmap='gray'); plt.title("4. Gradiente Roberts (Cruz 2x2)"); plt.axis('off')

    plt.tight_layout() # [Bloque 7: Ajuste] - Acomoda padding.
    plt.show()         # [Bloque 7: GUI] - Despliega ventana en Windows.