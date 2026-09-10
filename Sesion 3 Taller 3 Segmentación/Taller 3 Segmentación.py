# ==============================================================================
# PIPELINE MAESTRO: ÁLGEBRA LINEAL, VISIÓN Y SEGMENTACIÓN (TALLERES 1, 2 Y 3)
# ==============================================================================

import os                        # [Bloque 0: Sistema] - Interfaz Win32 para resolver rutas físicas absolutas.
import cv2                       # [Bloque 0: Visión C++] - Decodificador binario, convolución espacial y segmentación.
import numpy as np               # [Bloque 0: Matemáticas] - Álgebra de tensores en memoria continua C y lectura cruda de bytes.
import matplotlib.pyplot as plt  # [Bloque 0: Renderizado] - Mapeo de tensores a interfaz gráfica de usuario.
from typing import Tuple         # [Bloque 0: Tipado] - Declaración de variables para estándar de ingeniería.

print("--- INICIANDO PIPELINE MAESTRO (TALLERES 1, 2 Y 3) ---")

class MotorSegmentacion:
    """Motor matemático de binarización y filtrado espacial (Taller 3)."""
    
    def __init__(self, kernel_size: Tuple[int, int] = (3, 3)): # [Bloque 1: Inicialización] - Constructor del Kernel Morfológico.
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size) # [Bloque 1: Elemento Estructurante] - Matriz binaria impar.

    def aplicar_otsu(self, imagen: np.ndarray) -> Tuple[float, np.ndarray]:  # [Bloque 2: Binarización Otsu]
        # Fórmula: σ²_b(t) = ω_0(t)ω_1(t)[μ_0(t) - μ_1(t)]²
        umbral, img_binaria = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # [Bloque 2: Maximización Varianza]
        return umbral, img_binaria

    def limpiar_ruido(self, img_binaria: np.ndarray) -> np.ndarray:          # [Bloque 3: Morfología Matemática]
        # Apertura: A ∘ K = (A ⊖ K) ⊕ K (Elimina ruido exterior)
        img_apertura = cv2.morphologyEx(img_binaria, cv2.MORPH_OPEN, self.kernel)
        # Cierre: A • K = (A ⊕ K) ⊖ K (Rellena agujeros interiores)
        img_final = cv2.morphologyEx(img_apertura, cv2.MORPH_CLOSE, self.kernel)
        return img_final

# ==============================================================================
# AUTO-LOCALIZACIÓN Y LECTURA (BLINDAJE CONTRA TILDES EN WINDOWS)
# ==============================================================================
directorio_actual = os.path.dirname(os.path.abspath(__file__)) 
nombre_imagen = os.path.join(directorio_actual, '1. AFICHE CREADO.png') 

if not os.path.exists(nombre_imagen):
    print(f"ERROR CRÍTICO: No se encontró la imagen en:\n{directorio_actual}")
else:
    # [BLINDAJE UTF-8]: NumPy lee los bytes puros para evitar que el C++ de OpenCV colapse por la "ó" de "Segmentación"
    stream_bytes = np.fromfile(nombre_imagen, dtype=np.uint8) # [Bloque 4: I/O Seguro] - Lee el disco físico con Python nativo.
    img_bgr = cv2.imdecode(stream_bytes, cv2.IMREAD_COLOR)    # [Bloque 4: Decodificador] - Transforma bytes puros a tensor BGR.

    if img_bgr is None: # Validación definitiva post-decodificación
        print("ERROR CRÍTICO: El archivo existe pero OpenCV no pudo decodificar la matriz.")
    else:
        print("1. Tensor original cargado (Protección de tildes activa).")

        # --- 3. CONVERSIÓN FOTÓPICA (TALLER 2) ---
        B, G, R = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2] # [Bloque 5: Slicing] - Aislamiento espacial O(1).
        img_gris = (0.114 * B) + (0.587 * G) + (0.299 * R)             # [Bloque 5: Luminancia] - Producto interno fotópico.
        img_gris = np.clip(img_gris, 0, 255).astype(np.uint8)          # [Bloque 5: Acotamiento] - Proyección a 8 bits.

        # --- 4. TRANSFORMACIÓN AFÍN (TALLER 1) ---
        alpha, beta = 0.5, -50.0                                       # [Bloque 6: Parámetros] - Compresión 50% y traslación -50.
        img_oscura = (alpha * img_gris.astype(np.float64)) + beta      # [Bloque 6: Álgebra 64-bits] - Evita desbordamiento (wrapping).
        img_oscura = np.clip(img_oscura, 0, 255).astype(np.uint8)      # [Bloque 6: Cuantización] - Saturación estricta de hardware.

        # --- 5. DETECCIÓN LAPLACIANA (TALLER 1) ---
        Kernel_Laplace = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])   # [Bloque 7: Filtro] - Matriz Diferencial Σ K_ij = 0.
        img_bordes_16 = cv2.filter2D(img_gris, cv2.CV_16S, Kernel_Laplace) # [Bloque 7: Convolución Signada] - Captura gradientes negativos.
        img_bordes = cv2.convertScaleAbs(img_bordes_16)                    # [Bloque 7: Valor Absoluto] - Recupera energía de bordes.

        # --- 6. SEGMENTACIÓN ESTADÍSTICA (TALLER 3) ---
        motor = MotorSegmentacion(kernel_size=(3, 3))                      # [Bloque 8: Instanciación Taller 3]
        umbral_optimo, img_otsu = motor.aplicar_otsu(img_gris)             # [Bloque 8: Ejecución Otsu] - Binariza la imagen.
        img_limpia = motor.limpiar_ruido(img_otsu)                         # [Bloque 8: Ejecución Morfología] - Purifica el ruido.
        print(f"2. Procesamiento Exitoso. Umbral Otsu automático: {umbral_optimo}")

        # ==============================================================================
        # DASHBOARD VISUAL DE LOS 3 TALLERES
        # ==============================================================================
        plt.figure(figsize=(16, 10))                                       # [Bloque 9: Canvas UI] - Instanciación de objeto Figure | Reserva cuadrícula física de 16x10 pulgadas en memoria.
        plt.suptitle("PIPELINE UNIFICADO: VISIÓN, ÁLGEBRA Y SEGMENTACIÓN", fontsize=16, fontweight='bold') # [Bloque 9: Metadatos UI] - Configuración de título maestro y estilo tipográfico de la interfaz.

        plt.subplot(2, 3, 1)                                               # [Bloque 10: Subplot 1] - Localización de cuadrante superior izquierdo | Configuración de grilla de 2 filas y 3 columnas, índice 1.
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))               # [Bloque 10: Render RGB] - Inversión del tensor cromático nativo C++ de OpenCV (BGR) a convención de Matplotlib (RGB).
        plt.title("1. Tensor RGB Original")                                # [Bloque 10: Etiqueta] - Inserción de descriptor numérico del componente.
        plt.axis('off')                                                    # [Bloque 10: Limpieza Geométrica] - Supresión de los ejes de coordenadas y reglas (abscisas y ordenadas).

        plt.subplot(2, 3, 2)                                               # [Bloque 11: Subplot 2] - Localización de cuadrante superior central | Configuración de grilla 2x3, índice 2.
        plt.imshow(img_gris, cmap='gray', vmin=0, vmax=255)                # [Bloque 11: Render Fotópico] - Mapeo escalar usando mapa de color 'gray' | Bloqueo de intensidad estricta para evitar auto-normalización.
        plt.title("2. Luminancia Ponderada")                               # [Bloque 11: Etiqueta] - Inserción de descriptor analítico.
        plt.axis('off')                                                    # [Bloque 11: Limpieza Geométrica] - Oculta cotas de referencia.

        plt.subplot(2, 3, 3)                                               # [Bloque 12: Subplot 3] - Localización de cuadrante superior derecho | Configuración de grilla 2x3, índice 3.
        plt.imshow(img_oscura, cmap='gray', vmin=0, vmax=255)              # [Bloque 12: Render Afín] - Mapeo de tensor atenuado | Parámetros vmin/vmax evitan el estiramiento automático (auto-stretching) del histograma interno.
        plt.title("3. Mapeo Afín")                                         # [Bloque 12: Etiqueta] - Inserción de descriptor de transformación.
        plt.axis('off')                                                    # [Bloque 12: Limpieza Geométrica] - Retira el contenedor numérico perimetral.

        plt.subplot(2, 3, 4)                                               # [Bloque 13: Subplot 4] - Localización de cuadrante inferior izquierdo | Configuración de grilla 2x3, índice 4.
        plt.imshow(img_bordes, cmap='gray')                                # [Bloque 13: Render Diferencial] - Despliegue de derivada espacial | Contraste guiado por magnitud absoluta de bordes.
        plt.title("4. Laplaciano")                                         # [Bloque 13: Etiqueta] - Inserción de descriptor diferencial.
        plt.axis('off')                                                    # [Bloque 13: Limpieza Geométrica] - Desactiva escalas numéricas para limpieza visual.

        plt.subplot(2, 3, 5)                                               # [Bloque 14: Subplot 5] - Localización de cuadrante inferior central | Configuración de grilla 2x3, índice 5.
        plt.imshow(img_otsu, cmap='gray')                                  # [Bloque 14: Render Binario] - Despliegue de matriz discreta [0, 255] generada por la frontera de Otsu.
        plt.title(f"5. Otsu (Umbral={umbral_optimo})")                     # [Bloque 14: Etiqueta Dinámica] - Inserción del escalar 't' extraído directamente del clasificador de varianza.
        plt.axis('off')                                                    # [Bloque 14: Limpieza Geométrica] - Oculta ejes analíticos de Matplotlib.

        plt.subplot(2, 3, 6)                                               # [Bloque 15: Subplot 6] - Localización de cuadrante inferior derecho | Configuración de grilla 2x3, índice 6.
        plt.imshow(img_limpia, cmap='gray')                                # [Bloque 15: Render Morfológico] - Visualización del filtro espacial (Cierre y Apertura encadenados).
        plt.title("6. Morfología (Apertura + Cierre)")                     # [Bloque 15: Etiqueta] - Inserción de descriptor topológico.
        plt.axis('off')                                                    # [Bloque 15: Limpieza Geométrica] - Remueve marcos estructurales.

        plt.tight_layout()                                                 # [Bloque 16: Padding Inteligente] - C-Core Engine: Recalcula y optimiza colisiones de cuadros para preservar distancias entre títulos y gráficos.
        plt.show()                                                         # [Bloque 16: Despacho OS] - Interrumpe la ejecución del backend y fuerza el renderizado asíncrono en la ventana del sistema operativo (Win32).