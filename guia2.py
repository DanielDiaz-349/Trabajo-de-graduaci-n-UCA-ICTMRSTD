# guia2.py
# -*- coding: utf-8 -*-
"""
Guía 2: Señales y sistemas (versión Streamlit)
Estructura: Objetivos, Introducción teórica, Materiales y equipo,
Ejemplos (1–4), Dinámicas (1–3) y Conclusiones.
"""

import os
import json
import datetime
import requests
import base64
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from github_uploader import upload_bytes_to_github_results

# Disponibilidad de ReportLab (PDF)
try:
    import reportlab  # noqa: F401
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False



BASE_DIR = Path(__file__).resolve().parent
LOGO_UCA_PATH = str(BASE_DIR / "assets" / "logo_uca.png")

TEMA_TG = (
    "Introducción a la caracterización y tratamiento matemático del ruido "
    "en sistemas de telecomunicaciones digitales"
)




def export_results_pdf_txt(filename_base, student_info, dyn_id, key, answers, score):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{filename_base}_{ts}"
    txt_path = os.path.join(RESULTS_DIR, base + ".txt")

    # --- TXT (igual que antes, solo cuidando UTF-8) ---
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Resultados Dinámica\n")
        f.write(f"Dinámica: {dyn_id}\n")
        f.write(f"Fecha: {datetime.datetime.now().isoformat()}\n\n")
        f.write("Alumno:\n")
        f.write(f"  Nombre completo: {student_info.get('name')}\n")
        f.write(f"  Carné: {student_info.get('id')}\n")
        f.write(f"  Fecha de nacimiento: {student_info.get('dob')}\n\n")
        f.write("Clave utilizada (parámetros / respuestas correctas):\n")
        f.write(json.dumps(key, indent=2, ensure_ascii=False))
        f.write("\n\nRespuestas del alumno:\n")
        f.write(json.dumps(answers, indent=2, ensure_ascii=False))
        f.write(f"\n\nNota (oculta al alumno): {score}\n")

    # --- PDF con logo UCA + tema ---
    pdf_path = None
    if REPORTLAB_AVAILABLE:
        pdf_path = os.path.join(RESULTS_DIR, base + ".pdf")
        c = rcanvas.Canvas(pdf_path, pagesize=letter)
        width, height = letter
        left = 40
        top = height - 40
        line_h = 14

        # 1) Dibujar marca de agua (logo UCA) en el centro
        if os.path.exists(LOGO_UCA_PATH):
            from reportlab.lib.utils import ImageReader
            logo = ImageReader(LOGO_UCA_PATH)
            iw, ih = logo.getSize()
            aspect = ih / float(iw)

            logo_width = width * 0.6
            logo_height = logo_width * aspect
            x = (width - logo_width) / 2.0
            y = (height - logo_height) / 2.0

            c.saveState()
            # Intentar bajar opacidad (no todas las versiones lo soportan)
            try:
                c.setFillAlpha(0.2)   # ≈ 20% opacidad
            except Exception:
                pass
            c.drawImage(
                logo,
                x,
                y,
                width=logo_width,
                height=logo_height,
                mask="auto"
            )
            c.restoreState()

        # 2) Texto principal (igual que antes, por encima del logo)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, top, f"Resultados Dinámica {dyn_id}")
        c.setFont("Helvetica", 10)
        y = top - 2 * line_h
        c.drawString(left, y, f"Fecha: {datetime.datetime.now().isoformat()}")

        y -= 1.5 * line_h
        c.drawString(left, y, "Alumno:")
        y -= line_h
        c.drawString(left + 10, y, f"Nombre completo: {student_info.get('name')}")
        y -= line_h
        c.drawString(left + 10, y, f"Carné: {student_info.get('id')}")
        y -= line_h
        c.drawString(left + 10, y, f"Fecha de nacimiento: {student_info.get('dob')}")

        y -= 1.5 * line_h
        c.drawString(left, y, "Clave utilizada (parámetros / respuestas correctas):")
        y -= line_h
        c.setFont("Helvetica", 9)
        for k, v in key.items():
            if y < 80:
                c.showPage()
                y = top
                c.setFont("Helvetica", 9)
            c.drawString(left + 10, y, f"{k}: {v}")
            y -= line_h

        y -= line_h
        c.setFont("Helvetica", 10)
        c.drawString(left, y, "Respuestas del alumno:")
        y -= line_h
        c.setFont("Helvetica", 9)
        for q, a in answers.items():
            if y < 80:
                c.showPage()
                y = top
                c.setFont("Helvetica", 9)
            c.drawString(left + 10, y, f"{q}: {a}")
            y -= line_h

        # 3) Nota (oculta) y tema al pie
        y -= line_h
        c.setFont("Helvetica-Bold", 10)
        c.drawString(left, y, f"Nota (oculta al alumno): {score}")

        # Tema del trabajo de graduación centrado al pie
        c.setFont("Helvetica-Oblique", 9)
        c.drawCentredString(width / 2.0, 30, TEMA_TG)

        c.save()

    return txt_path, pdf_path

def export_results_pdf_guia2(filename_base, student_info, resultados):
    """
    Genera un solo PDF con el resumen de TODAS las dinámicas de la Guía 2.
    No genera TXT, solo PDF.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{filename_base}_{ts}"
    pdf_path = os.path.join(RESULTS_DIR, base + ".pdf")

    if not REPORTLAB_AVAILABLE:
        return pdf_path  # no se puede generar, devolvemos ruta prevista

    c = rcanvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    left = 40
    top = height - 40
    line_h = 14

    # Marca de agua con logo UCA (igual estilo a lo que ya tenías)
    if os.path.exists(LOGO_UCA_PATH):
        from reportlab.lib.utils import ImageReader
        logo = ImageReader(LOGO_UCA_PATH)
        iw, ih = logo.getSize()
        aspect = ih / float(iw)
        logo_width = width * 0.6
        logo_height = logo_width * aspect
        x = (width - logo_width) / 2.0
        y = (height - logo_height) / 2.0

        c.saveState()
        try:
            c.setFillAlpha(0.2)
        except Exception:
            pass
        c.drawImage(logo, x, y, width=logo_width, height=logo_height, mask="auto")
        c.restoreState()

    # Encabezado
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, top, "Resultados Guía 2 – Dinámicas")
    c.setFont("Helvetica", 10)
    y = top - 2 * line_h
    c.drawString(left, y, f"Fecha: {datetime.datetime.now().isoformat()}")

    # Datos del alumno
    y -= 1.5 * line_h
    c.drawString(left, y, "Alumno:")
    y -= line_h
    c.drawString(left + 10, y, f"Nombre completo: {student_info.get('name')}")
    y -= line_h
    c.drawString(left + 10, y, f"Carné: {student_info.get('id')}")
    y -= line_h
    c.drawString(left + 10, y, f"Fecha de nacimiento: {student_info.get('dob')}")

    # Resultados por dinámica
    total_score = 0.0
    for res in resultados:
        dyn_id = res["dyn_id"]
        score = res["score"]
        answers = res["answers"]
        correct = res["correct"]
        key = res["key"]
        total_score += score

        y -= 2 * line_h
        if y < 120:
            c.showPage()
            y = top
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left, y, f"Dinámica {dyn_id}")
        y -= line_h
        c.setFont("Helvetica", 10)
        c.drawString(left, y, f"Nota dinámica (oculta): {score}")
        y -= 1.5 * line_h

        c.setFont("Helvetica", 9)
        c.drawString(left, y, "Parámetros / clave:")
        y -= line_h
        for k, v in key.items():
            if y < 80:
                c.showPage()
                y = top
                c.setFont("Helvetica", 9)
            c.drawString(left + 10, y, f"{k}: {v}")
            y -= line_h

        y -= line_h
        c.drawString(left, y, "Respuestas correctas:")
        y -= line_h
        for q, v in correct.items():
            if y < 80:
                c.showPage()
                y = top
                c.setFont("Helvetica", 9)
            c.drawString(left + 10, y, f"{q}: {v}")
            y -= line_h

        y -= line_h
        c.drawString(left, y, "Respuestas del alumno:")
        y -= line_h
        for q, v in answers.items():
            if y < 80:
                c.showPage()
                y = top
                c.setFont("Helvetica", 9)
            c.drawString(left + 10, y, f"{q}: {v}")
            y -= line_h

    # Nota global (ej: promedio)
    promedio = total_score / max(len(resultados), 1)
    y -= 2 * line_h
    if y < 80:
        c.showPage()
        y = top
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left, y, f"Nota global de la guía (oculta): {promedio:.2f}")

    # Tema del TG en el pie
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(width / 2.0, 30, TEMA_TG)

    c.save()
    return pdf_path

# =========================================================
# Textos estáticos
# =========================================================

MATERIALES_COMUNES = """
Para desarrollar las actividades de esta guía interactiva se recomienda contar con:

- Una computadora personal con sistema operativo actualizado (Windows, Linux o macOS).
- Python instalado (versión 3.8 o superior recomendada).
- Un entorno de desarrollo como Visual Studio Code o PyCharm.
- Las siguientes bibliotecas de Python:
  - `numpy` para el manejo de arreglos y operaciones numéricas.
  - `matplotlib` para la generación de gráficas.
  - `streamlit` para la interfaz interactiva de la guía.
  -  `scipy` para operaciones adicionales de filtrado, convolución y análisis en frecuencia.
"""

OBJETIVOS2_TEXT = r"""

### Objetivos

**Objetivo general**

Analizar y comprender el comportamiento de señales y sistemas mediante la simulación, enfatizando el proceso de muestreo, la representación en el dominio de la frecuencia y la respuesta de sistemas lineales invariantes en el tiempo (LTI), a través de ejemplos interactivos y dinámicas que permitan al estudiante visualizar, manipular y evaluar los conceptos fundamentales de señales y sistemas.

**Objetivos específicos**

- Interpretar visualmente la diferencia entre señales continuas y discretas mediante el proceso de muestreo, analizando cómo varía la representación temporal de una señal al cambiar la frecuencia de muestreo.

- Identificar y explicar el fenómeno de aliasing a través del análisis espectral con FFT, evaluando cómo la selección de la frecuencia de muestreo \(fs\) afecta la reconstrucción y el contenido en frecuencia de la señal.

- Aplicar la convolución para determinar la salida de un sistema lineal invariante en el tiempo (LTI) y comprender cómo la respuesta al impulso define completamente el comportamiento del sistema.

- Relacionar la respuesta en frecuencia de un sistema LTI con su efecto sobre el espectro de la señal, comparando el filtrado en tiempo y en frecuencia para distintos tipos de sistemas.
"""

INTRO2_TEXT = r"""
#### Introducción Teórica

El análisis de señales y sistemas constituye una herramienta esencial para comprender cómo la información es representada, transformada y procesada en los sistemas modernos de telecomunicaciones y procesamiento digital. Todo sistema real, desde un canal de comunicación hasta un filtro pasa bajas, opera sobre señales que contienen información, y la forma en que estas señales se comportan depende tanto de su naturaleza temporal y espectral como de las características del sistema que las procesa. Por ello, esta guía tiene como propósito conectar los conceptos fundamentales de señales y sistemas con simulaciones prácticas que permitan visualizar de forma directa los fenómenos más importantes: el muestreo, el análisis en frecuencia, la convolución y la respuesta en frecuencia de sistemas LTI.

###### Señales en tiempo continuo y tiempo discreto

Una señal puede representarse matemáticamente como una función de una variable independiente. En tiempo continuo se denota en la **ecuación (1)**:

$$
x(t), \quad -\infty < t < \infty \tag{1}
$$

mientras que en tiempo discreto se representa en la **ecuación (2)**:

$$
x[n], \quad n \in \mathbb{Z} \tag{2}
$$

La señal continua describe la evolución de un fenómeno físico sin interrupciones, mientras que la señal discreta contiene valores definidos únicamente en instantes específicos.

###### Muestreo de señales y la conversión de x(t) a x[n]

El muestreo es el proceso mediante el cual una señal de tiempo continuo x(t) se convierte en una secuencia de tiempo discreto x[n], tomando muestras separadas por un intervalo constante. Matemáticamente se representa en la **ecuación (3)**:

$$
x[n] = x(nT_s) \tag{3}
$$

donde el parámetro Ts es el período de muestreo, y se relaciona con la frecuencia de muestreo fs mediante la **ecuación (4)**:

$$
T_s = \frac{1}{f_s} \tag{4}
$$

En la ecuación (4), fs es la frecuencia de muestreo.

Para que una señal de banda limitada pueda ser representada sin pérdida de información, debe cumplirse el criterio de Nyquist, que se define en la **ecuación (5)**:

$$
f_s \ge 2 f_{\max} \tag{5}
$$

donde fmax es la frecuencia máxima presente en la señal.
  
Cuando esta condición no se cumple, ocurre el fenómeno de aliasing. El aliasing se produce cuando existen componentes de frecuencias “falsas” que no forman parte de la señal original y que se crearon debido al error de representar frecuencias mayores a la frecuencia de Nyquist.  

La guía aborda este fenómeno mediante análisis en frecuencia con FFT, permitiendo al estudiante visualizar cómo las componentes espectrales se distorsionan cuando fs es insuficiente.

###### Análisis en frecuencia y la Transformada de Fourier

Toda señal puede analizarse tanto en el dominio del tiempo como en el dominio de la frecuencia. Una herramienta matemática fundamental muy útil en análisis y procesamiento de señales es la transformada de Fourier, esta herramienta permite representar una señal en términos de sus componentes de frecuencia. 

Mediante esta transformada, una señal definida en el dominio del tiempo puede expresarse como una superposición de exponenciales complejas, lo que facilita identificar su contenido espectral y estudiar cómo diferentes sistemas afectan sus componentes sinusoidales. En comunicaciones y procesamiento digital, esta representación es útil ya que revela propiedades esenciales como el ancho de banda, la distribución de energía en frecuencia y la interacción con filtros o canales de transmisión. 

La Transformada de Fourier de tiempo continuo se define en la **ecuación (6)**:

$$
X(f) = \int_{-\infty}^{\infty} x(t)\, e^{-j 2\pi f t}\, dt \tag{6}
$$

y su transformada inversa en la **ecuación (7)**:

$$
x(t) = \int_{-\infty}^{\infty} X(f)\, e^{j 2\pi f t}\, df \tag{7}
$$

Para señales de tiempo discreto se emplea la Transformada Discreta de Fourier (DFT), definida en la **ecuación (8)**:

$$
X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j \frac{2\pi}{N} k n}, \quad k = 0, 1, \dots, N-1 \tag{8}
$$

La DFT se calcula eficientemente mediante la FFT (Fast Fourier Transform) . 
 
La FFT permite computar el espectro de una señal muestreada de manera digital, mostrando sus componentes senoidales y revelando fenómenos como el aliasing. La FFT es un algoritmo que permite visualizar de manera digital espectro de una señal

###### Sistemas Lineales Invariantes en el Tiempo (LTI)

Muchos sistemas físicos pueden modelarse como lineales e invariantes en el tiempo (LTI).

Un sistema es lineal si cumple el principio de superposición mostrado en la **ecuación (9)**:

$$
\mathcal{S}\{a_1 x_1(t) + a_2 x_2(t)\}
= a_1\, \mathcal{S}\{x_1(t)\} + a_2\, \mathcal{S}\{x_2(t)\} \tag{9}
$$

Es invariante en el tiempo si un desplazamiento en la entrada produce el mismo desplazamiento en la salida, como se expresa en la **ecuación (10)**:

$$
x(t - t_0) \xrightarrow{\ \mathcal{}\ } y(t - t_0)
\quad \text{si} \quad
x(t) \xrightarrow{\ \mathcal{}\ } y(t) \tag{10}
$$

Todo sistema LTI se caracteriza completamente por su respuesta al impulso h(t).  
La salida ante cualquier entrada x(t) está dada por la convolución mostrada en la **ecuación (11)**:

$$
y(t) = (x * h)(t)
= \int_{-\infty}^{\infty} x(\tau)\, h(t - \tau)\, d\tau \tag{11}
$$

En tiempo discreto, la convolución se expresa como en la **ecuación (12)**:

$$
y[n] = \sum_{k=-\infty}^{\infty} x[k]\, h[n - k] \tag{12}
$$

###### Respuesta en frecuencia y filtrado

La Transformada de Fourier transforma la convolución en una multiplicación en el dominio de la frecuencia, mostrada en la **ecuación (13)**:

$$
Y(f) = X(f)\, H(f) \tag{13}
$$

Aquí, H(f) es la función de transferencia o respuesta en frecuencia, la cual determina cómo el sistema LTI atenúa o amplifica cada componente de frecuencia.

Por ejemplo:

- Un filtro pasa bajas mantiene las bajas frecuencias y atenúa las altas.
- Un filtro pasa altas hace lo contrario.
- Un promediador suaviza la señal reduciendo variaciones rápidas.

**Filtros digitales**

Un filtro digital es un sistema discreto que procesa una señal mediante operaciones matemáticas sobre sus muestras para modificar su contenido espectral o temporal según un propósito específico, como atenuar ruido o resaltar ciertas frecuencias. Dentro de ellos, un filtro FIR (Finite Impulse Response) es un tipo de filtro cuya respuesta al impulso es finita y se implementa como una suma ponderada de un número limitado de muestras pasadas de la entrad

En un filtro FIR, el parámetro 𝑀 representa el número de coeficientes menos uno, es decir, el orden del filtro. Un filtro de orden M tenga 
M+1 coeficientes en su respuesta al impulso. Estos coeficientes definen completamente el comportamiento del filtro y determinan cuántas muestras pasadas de la entrada se utilizan para generar cada muestra de la salida




"""

CONCLUSIONES2_TEXT = """
### Conclusiones

- El análisis de señales y sistemas constituye la base conceptual del tratamiento moderno de la información. Comprender cómo se representan y clasifican las señales permite interpretar correctamente fenómenos físicos y diseñar herramientas de procesamiento adecuadas a las necesidades de telecomunicaciones, control y electrónica.

- El proceso de muestreo es un paso fundamental en la conversión de señales analógicas a digitales. A lo largo de la guía se evidenció cómo la elección adecuada de la frecuencia de muestreo, en consonancia con el criterio de Nyquist, garantiza representaciones discretas fieles y evita el aliasing que deteriora irreversiblemente la información.

- El estudio del dominio de la frecuencia mediante la DFT y la FFT permitió visualizar de forma directa las componentes espectrales de una señal, herramienta indispensable para comprender modulaciones, filtrado, ruido y métodos de detección digital.

- Los sistemas LTI, descritos mediante su respuesta al impulso y su respuesta en frecuencia, ofrecieron un marco robusto para analizar cómo un filtro o un canal modifica la señal. La relación entre convolución en el tiempo y multiplicación en frecuencia mostró dos perspectivas complementarias para estudiar la acción de un mismo sistema.

- Finalmente, las simulaciones en Python brindaron una representación numérica y gráfica clara de fenómenos como muestreo, aliasing, convolución y respuesta en frecuencia, reforzando el aprendizaje y preparando al estudiante para abordar sistemas más complejos en etapas posteriores.
"""


# =========================================================
# Utilidades internas
# =========================================================


# =========================
# PDF (en memoria) para envío a GitHub - Guía 2
# =========================
def _g2_safe_str(x):
    return "" if x is None else str(x)

def _g2_sanitize_filename(s: str) -> str:
    s = re.sub(r"\s+", "_", (s or "").strip())
    # Solo caracteres seguros para nombre de archivo
    s = re.sub(r"[^A-Za-z0-9_\-]", "", s)
    return s[:80] if len(s) > 80 else s

def _g2_ensure_unicode_font():
    """Registra una fuente Unicode (DejaVuSans) si está disponible, y devuelve el nombre de fuente."""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        # Evitar registrar dos veces
        if "DejaVuSans" in pdfmetrics.getRegisteredFontNames():
            return "DejaVuSans"
        # Rutas típicas (Linux / Streamlit Cloud)
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
        for p in candidates:
            if os.path.exists(p):
                pdfmetrics.registerFont(TTFont("DejaVuSans", p))
                return "DejaVuSans"
    except Exception:
        pass
    return "Helvetica"

def export_results_pdf_guia2_bytes(student_info: dict, resultados: list, nota_global: float, logo_path: str = None):
    """Genera un PDF en memoria (bytes) con los resultados de Guía 2."""
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab no está disponible. Agrega 'reportlab' a requirements.txt")

    import datetime
    from io import BytesIO
    from reportlab.pdfgen import canvas as rcanvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader

    base_font = _g2_ensure_unicode_font()

    # Nombre de archivo (en repo) con timestamp para evitar colisiones
    registro = _g2_sanitize_filename(_g2_safe_str(student_info.get("id") or student_info.get("registro") or ""))
    nombre = _g2_sanitize_filename(_g2_safe_str(student_info.get("name") or student_info.get("nombre") or ""))
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_filename = f"guia2_{registro}_{nombre}_{ts}.pdf"

    buf = BytesIO()
    c = rcanvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Encabezado
    y = height - 50
    c.setFont(base_font, 16)
    c.drawString(50, y, "Guía 2 - Resultados de dinámicas")
    y -= 22

    # Logo (opcional)
    if logo_path:
        try:
            img = ImageReader(logo_path)
            c.drawImage(img, width - 140, height - 85, width=80, height=80, mask="auto")
        except Exception:
            pass

    c.setFont(base_font, 11)
    c.drawString(50, y, f"Nombre: {_g2_safe_str(student_info.get('name') or student_info.get('nombre') or '')}")
    y -= 16
    c.drawString(50, y, f"Registro: {_g2_safe_str(student_info.get('id') or student_info.get('registro') or '')}")
    y -= 16
    dob = _g2_safe_str(student_info.get("dob") or "")
    if dob:
        c.drawString(50, y, f"Fecha de nacimiento: {dob}")
        y -= 16
    c.drawString(50, y, f"Fecha: {ts.replace('_', ' ')}")
    y -= 22

    # Nota global
    c.setFont(base_font, 12)
    c.drawString(50, y, f"Nota global: {nota_global}/10")
    y -= 18

    # Contenido por dinámica
    c.setFont(base_font, 11)
    for res in resultados:
        if y < 120:
            c.showPage()
            y = height - 60
            c.setFont(base_font, 11)

        titulo = _g2_safe_str(res.get("titulo", "Dinámica"))
        correctas = _g2_safe_str(res.get("correctas", ""))
        total = _g2_safe_str(res.get("total", ""))
        nota = _g2_safe_str(res.get("nota", ""))

        c.setFont(base_font, 12)
        c.drawString(50, y, titulo)
        y -= 16
        c.setFont(base_font, 11)
        c.drawString(60, y, f"Correctas: {correctas}/{total}    Nota: {nota}/10")
        y -= 14

        # Parámetros / clave
        key = res.get("key") or {}
        if key:
            c.drawString(60, y, "Parámetros:")
            y -= 14
            for k, v in key.items():
                if y < 90:
                    c.showPage()
                    y = height - 60
                    c.setFont(base_font, 11)
                c.drawString(75, y, f"- {k}: {_g2_safe_str(v)}")
                y -= 12

        # Respuestas
        answers = res.get("answers") or {}
        if answers:
            if y < 110:
                c.showPage()
                y = height - 60
                c.setFont(base_font, 11)
            c.drawString(60, y, "Respuestas del estudiante:")
            y -= 14
            for k, v in answers.items():
                if y < 90:
                    c.showPage()
                    y = height - 60
                    c.setFont(base_font, 11)
                c.drawString(75, y, f"- {k}: {_g2_safe_str(v)}")
                y -= 12

        y -= 10

    c.showPage()
    c.save()

    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes, pdf_filename

def _nyquist_info(f1, f2, fs):
    f_max = max(abs(f1), abs(f2))
    if f_max <= 0:
        return f_max, 0.0, "La señal es constante (sin componentes senoidales)."
    f_nyq = 2 * f_max
    ratio = fs / f_max
    if fs >= f_nyq:
        msg = (
            f"- Frecuencia máxima de la señal: {f_max:.2f} Hz\n"
            f"- Frecuencia de muestreo: f_s = {fs:.2f} Hz\n"
            f"- Criterio de Nyquist: f_s ≥ 2·f_max = {f_nyq:.2f} Hz\n\n"
            "En esta configuración **sí se cumple** el criterio de Nyquist. \n"
            "La señal muestreada puede representar correctamente la forma de la señal continua."
        )
    else:
        msg = (
            f"- Frecuencia máxima de la señal: {f_max:.2f} Hz\n"
            f"- Frecuencia de muestreo: f_s = {fs:.2f} Hz\n"
            f"- Criterio de Nyquist: f_s ≥ 2·f_max = {f_nyq:.2f} Hz\n\n"
            "En esta configuración **no se cumple** el criterio de Nyquist. "
            "Se producirá **aliasing**: las componentes de alta frecuencia se pliegan y la señal discreta ya no representa fielmente a la señal original."
        )
    return f_max, f_nyq, msg


def _render_student_registration(prefix_key: str) -> bool:
    """
    Muestra formulario de registro (nombre, carné, fecha de nacimiento)
    y guarda la info en st.session_state['student_info'].
    Devuelve True si la dinámica puede continuar (datos válidos).
    """
    st.markdown("### Registro de estudiante")

    # Recuperar valores previos si existen
    info = st.session_state.get("student_info", {"name": "", "id": "", "dob": ""})

    with st.form(f"{prefix_key}_registro"):
        name = st.text_input("Nombre completo", value=info.get("name", ""))
        carnet = st.text_input("Carné", value=info.get("id", ""))
        dob = st.text_input("Fecha de nacimiento (YYYY-MM-DD)", value=info.get("dob", ""))
        iniciar = st.form_submit_button("Iniciar dinámica")

    if iniciar:
        if not name or not carnet or not dob:
            st.warning("Por favor complete nombre, carné y fecha de nacimiento antes de continuar.")
            return False
        st.session_state["student_info"] = {"name": name, "id": carnet, "dob": dob}
        st.success("Datos registrados. Puede continuar con la dinámica.")
        st.session_state[f"{prefix_key}_started"] = True

    return st.session_state.get(f"{prefix_key}_started", False)


# =========================================================
# Ejemplo 1 – Muestreo
# =========================================================

def render_ejemplo1():
    st.subheader("Ejemplo 1 - Muestreo de una señal continua")

    with st.expander("Descripción y pasos a seguir", expanded=True):
        st.markdown(
            "Se genera una señal continua formada por la suma de dos senoidales y se "
            "muestra cómo se convierte en una señal discreta al muestrearla con una frecuencia fs.\n\n"
            "**Pasos sugeridos**\n"
            "1. Definir la amplitud **A1** y la frecuencia **A1** de la primer señal senoidal\n"
            "2. Definir la amplotud **A2** y la frecuencia **A2** de la segunda señal senoidal\n"
            "3. Definir la frecuencia de muestreo **fs**\n"
            "4. De manera opcional se puede modificar el tiempo de simulación **T**\n"
            "5. **Generar la señal y muestrear**"
        )

        col1, col2 = st.columns(2)
        with col1:
            A1 = st.number_input("Amplitud A₁", value=1.0, step=0.1)
            f1 = st.number_input("Frecuencia f₁ (Hz)", value=100.0, step=10.0)
            A2 = st.number_input("Amplitud A₂", value=0.7, step=0.1)
            f2 = st.number_input("Frecuencia f₂ (Hz)", value=300.0, step=10.0)
        with col2:
            fs = st.number_input("Frecuencia de muestreo fₛ (Hz)", value=2000.0, step=100.0)
            T = st.number_input("Duración total T (s)", value=0.06, step=0.005, format="%.4f")

        if st.button("Generar señal y muestrear", key="ej2_ej1"):
            # Señal "continua": muestreo muy fino para simular continuidad
            f_max = max(f1, f2)
            fs_cont = max(100 * f_max, 10_000) if f_max > 0 else 10_000
            t_cont = np.arange(0, T, 1.0 / fs_cont)
            x_cont = A1 * np.sin(2 * np.pi * f1 * t_cont) + A2 * np.sin(2 * np.pi * f2 * t_cont)

            # Señal muestreada
            t_disc = np.arange(0, T, 1.0 / fs)
            x_disc = A1 * np.sin(2 * np.pi * f1 * t_disc) + A2 * np.sin(2 * np.pi * f2 * t_disc)

            # Gráficas
            fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            axs[0].plot(t_cont, x_cont)
            axs[0].set_ylabel("x(t)")
            axs[0].set_title("Señal de tiempo continuo")
            axs[0].grid(True, linestyle=":")

            axs[1].stem(t_disc, x_disc)
            axs[1].set_xlabel("Tiempo (s)")
            axs[1].set_ylabel("x[n]")
            axs[1].set_title("Señal muestreada")
            axs[1].grid(True, linestyle=":")

            fig.tight_layout(pad=2.0)
            st.pyplot(fig)

            # Explicación dinámica
            f_max, f_nyq, nyq_msg = _nyquist_info(f1, f2, fs)
            st.markdown("#### Explicación de la simulación y preguntas")
            st.markdown(
                "La señal continua se construye como la suma de dos sinusoides. "
                "Al muestrearla, solo se conservan muestras cada 1/fₛ segundos. "
                "La capacidad de reconstruir la señal original depende de la relación entre fₛ y la frecuencia máxima presente."
            )
            st.markdown(nyq_msg)

            # Preguntas y respuestas (conceptuales)
            st.markdown("##### Preguntas y respuestas: ")
            st.markdown("**1. ¿Qué ocurre si reducimos demasiado la frecuencia de muestreo fₛ?**")
            st.markdown("**R:** La señal discreta comienza a perder detalle y puede aparecer aliasing, es decir, componentes de alta frecuencia se reflejan como frecuencias más bajas.")

            st.markdown("**2. Si fmax = 300 Hz, ¿cuál es el valor mínimo de fₛ que respeta el criterio de Nyquist?**")
            st.markdown("**R:** fₛ mínima = 2·f_max = 600 Hz.")

            st.markdown("**3. ¿Qué ventaja tiene representar la señal tanto en continuo como en discreto en el mismo eje de tiempo?**")
            st.markdown("**R:** Permite comparar visualmente qué tanta información de la forma de onda original se conserva luego del muestreo.")

# =========================================================
# Ejemplo 2 – Aliasing y FFT
# =========================================================

# =========================================================
# Ejemplo 2 – Aliasing y FFT (versión sin "modo de muestreo")
# =========================================================

def render_ejemplo2():
    st.subheader("Ejemplo 2 - Aliasing y análisis en frecuencia (FFT)")

    with st.expander("Descripción y pasos a seguir", expanded=True):
        st.markdown(
            "En este ejemplo se construye una señal como suma de dos senoidales continuas y luego se muestrea a una "
            "frecuencia fs elegida por el usuario. A partir de la señal discreta se calcula la FFT y se muestra "
            "su espectro en el intervalo [-fs/2, fs/2], de forma que se pueda apreciar el efecto del aliasing cuando "
            "no se cumple el criterio de Nyquist.\n\n"
            "**Pasos sugeridos**\n"
            "1. Define las amplitudes **A1**, **A2** y las frecuencias **f1**, **f2** de las dos senoidales.\n"
            "2. Elige una frecuencia de muestreo **fs**.\n"
            "3. (Opcional) Ajusta la duración total **T** de la simulación.\n"
            "4. Pulsa **Analizar en frecuencia**.\n"
            "5. Observa la señal discreta en el tiempo y su espectro centrado en [-fs/2, fs/2].\n"
            "6. Verifica la retroalimentación"
        )

        col1, col2 = st.columns(2)
        with col1:
            A1 = st.number_input("Amplitud A₁", value=1.0, step=0.1, key="g2_ej2_A1")
            f1 = st.number_input("Frecuencia f₁ (Hz)", value=100.0, step=10.0, key="g2_ej2_f1")
            A2 = st.number_input("Amplitud A₂", value=0.7, step=0.1, key="g2_ej2_A2")
            f2 = st.number_input("Frecuencia f₂ (Hz)", value=300.0, step=10.0, key="g2_ej2_f2")
        with col2:
            fs = st.number_input("Frecuencia de muestreo fₛ (Hz)", value=200.0, step=100.0, key="g2_ej2_fs")
            T = st.number_input("Duración total T (s)", value=0.08, step=0.005, format="%.4f", key="g2_ej2_T")

    if st.button("Analizar en frecuencia", key="g2_ej2_btn"):
        # --- Señal discreta con fs elegido por el usuario ---
        t_disc = np.arange(0, T, 1.0 / fs)
        x_disc = A1 * np.sin(2 * np.pi * f1 * t_disc) + A2 * np.sin(2 * np.pi * f2 * t_disc)

        # --- FFT discreta y centrada en [-fs/2, fs/2] ---
        N = len(x_disc)
        X = np.fft.fft(x_disc)
        freqs = np.fft.fftfreq(N, d=1.0 / fs)      # frecuencias en Hz, positivas y negativas
        X_shift = np.fft.fftshift(X)
        freqs_shift = np.fft.fftshift(freqs)
        X_mag_shift = np.abs(X_shift) / N

        # --- Gráfica: señal discreta y espectros (banda base + réplicas) ---
        fig, axs = plt.subplots(3, 1, figsize=(7, 8))

        # Señal discreta en el tiempo
        axs[0].stem(t_disc, x_disc)  # sin use_line_collection
        axs[0].set_xlabel("Tiempo (s)")
        axs[0].set_ylabel("x[n]")
        axs[0].set_title("Señal discreta en el tiempo")
        axs[0].grid(True, linestyle=":")

        # Espectro centrado en [-fs/2, fs/2] (banda base)
        axs[1].stem(freqs_shift, X_mag_shift)
        axs[1].set_xlim(-fs / 2, fs / 2)
        axs[1].set_xlabel("Frecuencia (Hz)")
        axs[1].set_ylabel("|X(f)|")
        axs[1].set_title("Espectro de magnitud centrado en [-fₛ/2, fₛ/2]")
        axs[1].grid(True, linestyle=":")

        # Réplicas espectrales alrededor de k·fs (k = -2…2)
        k_max = 2
        freqs_rep = np.concatenate([freqs_shift + k * fs for k in range(-k_max, k_max + 1)])
        mags_rep = np.tile(X_mag_shift, 2 * k_max + 1)

        axs[2].stem(freqs_rep, mags_rep)
        axs[2].set_xlim(-(k_max + 0.5) * fs, (k_max + 0.5) * fs)
        axs[2].set_xlabel("Frecuencia (Hz)")
        axs[2].set_ylabel("|X(f)|")
        axs[2].set_title("Réplicas espectrales alrededor de k·fₛ (k = -2…2)")
        axs[2].grid(True, linestyle=":")

        fig.tight_layout(pad=2.0)
        st.pyplot(fig)

        # --- Análisis de Nyquist / aliasing ---
        f_max, f_nyq, nyq_msg = _nyquist_info(f1, f2, fs)

        st.markdown("##### Explicación de la simulación y preguntas")

        if fs < 2 * f_max:
            st.markdown(
                "La frecuencia de muestreo seleccionada es **insuficiente** respecto a la frecuencia máxima de la señal. "
                "En el espectro centrado en [-fs/2, fs/2] las componentes de alta frecuencia se han plegado hacia "
                "la banda base, dando lugar a aliasing: aparecen picos en posiciones que no coinciden con f1 y f2 originales."
            )
        else:
            st.markdown(
                "La frecuencia de muestreo seleccionada es **suficiente** para cumplir el criterio de Nyquist. "
                "En el espectro centrado en [-fs/2, fs/2] las componentes correspondientes a f1 y f2 "
                "aparecen en las posiciones esperadas y no hay plegamiento evidente."
            )

        st.markdown(nyq_msg)

        st.markdown("##### Recordatorio:")
        st.markdown(
            "- Toda señal real tiene espectro simétrico: componentes positivas y negativas.\n"
            "- La FFT de la señal muestreada representa una copia del espectro en la **banda base** [-fs/2, fs/2].\n"
            "- Si fs es insuficiente (no cumple Nyquist), las componentes de alta frecuencia se pliegan dentro de esa banda base.\n"
            "- Esas componentes plegadas se interpretan como frecuencias más bajas: esto es el **aliasing**.\n"
            "- El espectro de una señal muestreada idealmente se replica periódicamente en frecuencia cada fs, y esas réplicas aparecen alrededor de kfs para todos los enteros 𝑘"
        )

        # Preguntas y respuestas
        st.markdown("##### Preguntas y respuestas")

        st.markdown("**1. ¿Qué representan los picos en el espectro centrado en [-fₛ/2, fₛ/2]?**")
        st.markdown(
            "**R:** Representan las componentes senoidales que ve el sistema discreto. "
            "Si hay aliasing, estas componentes no coinciden necesariamente con las frecuencias originales f1, f2."
        )

        st.markdown("**2. ¿Cómo puedes saber, solo viendo el espectro centrado, si hubo aliasing?**")
        st.markdown(
            "**R:** Comparando fs con la frecuencia máxima presente en la señal y verificando si fs < 2 fmax. "
            "Si esta condición se no se cumple, los picos observados en la banda base corresponden a frecuencias plegadas."
        )

        st.markdown("**3. ¿Por qué es tan importante elegir correctamente fs antes de muestrear?**")
        st.markdown(
            "**R:** Porque si fs es demasiado baja, el aliasing hace que diferentes señales continuas produzcan la misma "
            "secuencia discreta, perdiendo información de forma irreversible."
        )

        st.markdown("**4. ¿Qué indican los picos en el espectro |X(f)| de la FFT discreta?**")
        st.markdown(
            "**R:** Indican la presencia de componentes senoidales a las frecuencias correspondientes. "
            "Su altura se relaciona con la amplitud de cada componente en la señal muestreada."
        )

        st.markdown("**5. ¿Por qué no es posible corregir el aliasing solo procesando la señal muestreada?**")
        st.markdown(
            "**R:** Porque la información ya se perdió durante el muestreo. Diferentes señales continuas pueden producir "
            "la misma secuencia discreta cuando hay aliasing, por lo que no es posible reconstruir de forma única la señal original."
        )

# =========================================================
# Ejemplo 3 – LTI en tiempo
# =========================================================

def render_ejemplo3():
    st.subheader("Ejemplo 3 - Sistema LTI en el dominio del tiempo")

    with st.expander("Descripción y pasos a seguir", expanded=True):
        st.markdown(
            "Se muestra cómo un sistema LTI puede describirse por su respuesta al impulso h[n] y cómo "
            "la salida y[n] se obtiene mediante la convolución entre x[n] y h[n].\n\n"
            "**Pasos sugeridos**\n"
            "1. Elegir el tipo de **señal de entrada**\n"
            "2. Elegir la longitud del filtro **M**\n"
            "3. Elegir el **sistema LTI**\n"
            "4. **Aplicar el sistema LTI**"
        )

        tipo_entrada = st.selectbox(
            "Tipo de señal de entrada x[n]",
            ["Pulso rectangular", "Suma de sinusoidales discretas"]
        )
        M = st.number_input("Longitud del filtro M (número de coeficientes)", min_value=2, max_value=64, value=20, step=1)
        tipo_filtro = st.selectbox("Tipo de sistema h[n]", ["Filtro pasa bajas", "Suavizado exponencial"])

        if st.button("Aplicar sistema LTI", key="ej3_btn"):
            n = np.arange(0, 64)

            if tipo_entrada == "Pulso rectangular":
                x = np.zeros_like(n, dtype=float)
                x[10:20] = 1.0
            else:
                x = np.sin(2 * np.pi * 0.05 * n) + 0.6 * np.sin(2 * np.pi * 0.15 * n)

            if tipo_filtro == "Filtro pasa bajas":
                h = np.ones(M) / M
            else:
                alpha = 0.4
                h = alpha ** np.arange(M)

            y = np.convolve(x, h)

            n_h = np.arange(0, len(h))
            n_y = np.arange(0, len(y))

            fig, axs = plt.subplots(3, 1, figsize=(7, 6), sharex=False)
            axs[0].stem(n, x)
            axs[0].set_xlabel("n")
            axs[0].set_ylabel("x[n]")
            axs[0].set_title("Entrada x[n]")
            axs[0].grid(True, linestyle=":")

            axs[1].stem(n_h, h)
            axs[1].set_xlabel("n")
            axs[1].set_ylabel("h[n]")
            axs[1].set_title("Respuesta al impulso del sistema")
            axs[1].grid(True, linestyle=":")

            axs[2].stem(n_y, y)
            axs[2].set_xlabel("n")
            axs[2].set_ylabel("y[n]")
            axs[2].set_title("Salida y[n] = x[n] * h[n]")
            axs[2].grid(True, linestyle=":")

            fig.tight_layout(pad=2.0)
            st.pyplot(fig)

            # Explicación dinámica (mejor conectada con las gráficas)
            st.markdown("##### Explicación de la simulación")

            st.markdown(
                "En un sistema LTI, toda la información del sistema está contenida en su respuesta al impulso $h[n]$. "
                "La salida se calcula con la convolución discreta:\n\n"
                "$$y[n] = \\sum_{k=-\\infty}^{\\infty} x[k]\\,h[n-k]$$\n\n"
                "Esto puede interpretarse así: para cada $n$, el sistema toma un fragmento de la entrada $x[k]$ y lo combina con los "
                "pesos del filtro $h[n]$ , es decir, una suma ponderada."
            )

            if tipo_filtro == "Filtro pasa bajas":
                st.markdown(
                    f"**Caso: Filtro  pasa bajas FIR.** \n\n" 
                    "Aquí $h[n]=\\frac{{1}}{{M}}$ para $n=0,1,\\dots,M-1$.\n\n"
                    "- Cada muestra de salida es el promedio de las últimas $M$ muestras de la entrada.\n"
                    "- Por eso, los cambios bruscos (bordes en la salida) se suavizan, esos bordes requieren componentes de alta frecuencia.\n"
                    "- Al aumentar $M$, el suavizado es mayor y elimina variaciones rápidas, pero la salida pierde detalle temporal "
                    "y aparece un retardo efectivo mayor."
                )
            else:
                st.markdown(
                    f"**Caso: Suavizado exponencial o respuesta decreciente**.\n\n" 
                    " Aquí $h[n]=\\alpha^n$ para $n=0,1,\\dots,M-1$.\n\n"
                    "- La salida es una suma ponderada donde las muestras más recientes tienen más peso.\n"
                    "- Esto introduce memoria: el sistema “arrastra” información pasada, lo cual suaviza la señal.\n"
                    "- Si $\\alpha$ es más grande (cerca de 1), la memoria es más larga; si es más pequeña, el sistema responde más rápido."
                )

            st.markdown(
                " **Cómo interpretar las gráficas:**\n"
                "- 1) **x(n) (entrada):** Señal de entrada que ingresa al sistema LTI.\n"
                "- 2) **h(n) (respuesta al impulso del sistema LTI):** Son los pesos que el sistema usa.\n"
                "- 3) **y(n) (salida):** resulta de aplicar esos pesos a la entrada mediante la convolución.\n\n"
                "En términos simples: $y[n]$ se obtiene como el resultado de “deslizar” $h[n]$ sobre $x[n]$ y calcular una suma ponderada en cada desplazamiento."
            )



            # Preguntas y respuestas
            st.markdown("##### Preguntas y respuestas")
            st.markdown("**1. ¿Por qué un filtro promediador se considera un sistema pasa bajas?**")
            st.markdown("**R:** Porque suaviza la señal y atenúa las variaciones rápidas (componentes de alta frecuencia), dejando pasar principalmente las variaciones lentas.")

            st.markdown("**2. ¿Qué interpretación física tiene la convolución en este contexto?**")
            st.markdown("**R:** Cada muestra de y[n] es el resultado de sumar copias desplazadas de h[n] ponderadas por los valores de x[n]; el sistema 'promedia' o 'dispersa' la energía de la señal en el tiempo.")

            st.markdown("**3. ¿Cómo afectaría aumentar el valor de M en el filtro pasabajas?**")
            st.markdown("**R:** El filtro se vuelve más suave: la salida cambia más lentamente y se reducen aún más las componentes de alta frecuencia, pero se pierde detalle temporal.")


# =========================================================
# Ejemplo 4 – LTI en frecuencia
# =========================================================

def render_ejemplo4():
    st.subheader("Ejemplo 4 - Sistema LTI en el dominio de la frecuencia ")

    with st.expander("Descripción y pasos a seguir", expanded=True):
        st.markdown(
            "Se ilustra la relación entre convolución en el tiempo y multiplicación en frecuencia: "
            "un filtro LTI modifica el espectro de la señal de entrada según su respuesta en frecuencia H(f).\n\n"
            "**Pasos sugeridos**\n"
            "1. Elegir el tipo de **sistema LTI** \n"
            "2. Elegir la frecuencia de muestreo **fs**\n"
            "3. De manera opcional se puede elegir la duración de la simulación **T** \n"
            "4. **Aplicar el filtro en frecuencia**\n"
        )

        tipo_filtro = st.selectbox(
            "Tipo de filtro",
            ["Pasa bajas", "Pasa altas", "Suavizado exponencial"]
        )

        fs = st.number_input("Frecuencia de muestreo fₛ (Hz)", value=2000.0, step=100.0, key="ej4_fs")
        T = st.number_input("Duración total T (s)", value=0.05, step=0.005, format="%.4f", key="ej4_T")

        if st.button("Aplicar filtro en frecuencia", key="ej4_btn"):
            # Señal con varias sinusoides
            t = np.arange(0, T, 1.0 / fs)
            x = (
                    np.sin(2 * np.pi * 100 * t) +
                    0.7 * np.sin(2 * np.pi * 400 * t) +
                    0.5 * np.sin(2 * np.pi * 800 * t)
            )

            # Definir h[n]
            N = len(t)
            M = 33
            if tipo_filtro == "Pasa bajas":
                h = np.ones(M) / M
            elif tipo_filtro == "Pasa altas":
                h = np.zeros(M)
                h[0] = 1.0
                h[1] = -1.0
            else:
                alpha = 0.3
                h = alpha ** np.arange(M)

            # FFT de x y h (zero-padding a N)
            H = np.fft.fft(h, n=N)
            X = np.fft.fft(x)
            Y = X * H

            freqs = np.fft.fftfreq(N, d=1.0 / fs)
            idx_pos = freqs >= 0
            fpos = freqs[idx_pos]
            Xmag = np.abs(X[idx_pos]) / N
            Hmag = np.abs(H[idx_pos])
            Ymag = np.abs(Y[idx_pos]) / N

            fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=False)
            axs[0].stem(fpos, Xmag)
            axs[0].set_xlabel("Frecuencia (Hz)")
            axs[0].set_ylabel("|X(f)|")
            axs[0].set_title("Espectro de entrada")

            axs[1].stem(fpos, Hmag)
            axs[1].set_xlabel("Frecuencia (Hz)")
            axs[1].set_ylabel("|H(f)|")
            axs[1].set_title("Respuesta en frecuencia del filtro")

            axs[2].stem(fpos, Ymag)
            axs[2].set_xlabel("Frecuencia (Hz)")
            axs[2].set_ylabel("|Y(f)|")
            axs[2].set_title("Espectro de salida")

            for ax in axs:
                ax.grid(True, linestyle=":")

            fig.tight_layout(pad=2.0)
            st.pyplot(fig)

            # Explicación dinámica
            st.markdown("##### Explicación de la simulación")

            st.markdown(
                "Este ejemplo muestra la misma idea del Ejemplo 3, pero vista desde el dominio de la frecuencia. "
                "En un sistema LTI, la salida se obtiene como:\n\n"
                "$$Y(f)=X(f)\\,H(f)$$\n\n"
                "Es decir, cada componente frecuencial de la entrada se multiplica por la ganancia del filtro en esa frecuencia."
            )

            st.markdown(
                "**Cómo interpretar las gráficas:**\n"
                "1) **|X(f)| (entrada):** aquí aparecen picos en las frecuencias que componen la señal.\n"
                "2) **|H(f)| (filtro):** indica cuánto deja pasar o atenúa el sistema en cada frecuencia.\n"
                "3) **|Y(f)| (salida):** es el resultado de “escalar” cada pico de la entrada según el valor de |H(f)| en esa misma frecuencia.\n"
            )

            if tipo_filtro.startswith("Pasa bajas"):
                st.markdown(
                    "**Caso: Filtro pasa bajas.**\n"
                    "- |H(f)| es grande en bajas frecuencias y decrece hacia frecuencias altas.\n"

                    "- Es una señal más suave porque se eliminan variaciones rápidas."
                )
            elif tipo_filtro.startswith("Pasa altas"):
                st.markdown(
                    "**Caso: Filtro pasa altas.**\n"
                    "- |H(f)| es pequeño cerca de 0 Hz y aumenta hacia frecuencias más altas.\n"

                    "- En el tiempo, el sistema resalta cambios rápidos (bordes o variaciones bruscas)."
                )
            else:
                st.markdown(
                    "**Caso: Suavizado exponencial.**\n"
                    "- La respuesta al impulso decreciente genera un comportamiento tipo **pasa bajas gradual**.\n"
                    "- En |H(f)| la transición no es abrupta, por eso en |Y(f)| las componentes altas se atenúan de forma progresiva.\n"
                    "- En el tiempo, esto se interpreta como un promedio ponderado donde las muestras recientes pesan más."
                )

            st.markdown(
                "En otras palabras, en el tiempo se trabaja con convolución ($y[n]=x[n]*h[n]$) y en frecuencia con multiplicación "
                "($Y(f)=X(f)\\,H(f)$). Son dos formas equivalentes de describir la misma relación entrada–sistema–salida."
            )

            # Preguntas y respuestas
            st.markdown("##### Preguntas y respuestas")
            st.markdown(
                "**1. ¿Qué sucede con las componentes de alta frecuencia cuando aplicamos un filtro pasa bajas?**")
            st.markdown("**R:** Se atenúan, reduciendo su contribución en la señal de salida.")

            st.markdown("**2. ¿Cómo se observa un filtro pasa altas en la gráfica de |H(f)|?**")
            st.markdown("**R:** Presenta magnitud pequeña en bajas frecuencias y mayor magnitud en frecuencias altas.")

            st.markdown("**3. ¿Por qué decimos que Y(f) = X(f)·H(f) es equivalente a y[n] = x[n] * h[n]?**")
            st.markdown(
                "**R:** Porque la Transformada de Fourier convierte la convolución en el tiempo en una multiplicación en frecuencia; ambas representan la misma relación entrada–sistema–salida desde dos perspectivas distintas.")


# =========================================================
# Dinámica 1 – Muestreo
# =========================================================

def render_dinamica1():
    st.subheader("Dinámica 1 – Muestreo correcto e incorrecto (aliasing)")

    # Registro
    started = _render_student_registration("g2_dyn1")
    if not started:
        st.info("Complete el registro y pulse **Iniciar dinámica** para comenzar.")
        return

    st.markdown(
        "En esta dinámica se presentan dos casos de muestreo de la misma señal: "
        "uno que cumple el criterio de Nyquist y otro que no. El objetivo es identificar "
        "visualmente el aliasing y razonar sobre la elección de fₛ."
    )

    f_sig = 500.0
    fs_bueno = 4000.0
    fs_malo = 600.0

    T = 0.02
    t_b = np.arange(0, T, 1.0 / fs_bueno)
    x_b = np.sin(2 * np.pi * f_sig * t_b)

    t_m = np.arange(0, T, 1.0 / fs_malo)
    x_m = np.sin(2 * np.pi * f_sig * t_m)

    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=False)
    axs[0].stem(t_b, x_b)
    axs[0].set_title(f"Caso A: muestreo con fₛ = {fs_bueno:.0f} Hz (bueno)")
    axs[0].set_ylabel("x_A[n]")
    axs[0].grid(True, linestyle=":")

    axs[1].stem(t_m, x_m)
    axs[1].set_title(f"Caso B: muestreo con fₛ = {fs_malo:.0f} Hz (posible aliasing)")
    axs[1].set_xlabel("Tiempo (s)")
    axs[1].set_ylabel("x_B[n]")
    axs[1].grid(True, linestyle=":")

    fig.tight_layout(pad=2.0)
    st.pyplot(fig)

    st.markdown("### Preguntas")

    with st.form("g2_dyn1_respuestas"):
        q1 = st.radio(
            "1) ¿En cuál de los casos se cumple mejor el criterio de Nyquist?",
            ["Seleccione una opción", "Caso A", "Caso B"],
            index=0,
            key="g2_dyn1_q1"
        )
        q2 = st.radio(
            "2) Verdadero o falso: “El aliasing puede corregirse aumentando solo la duración T manteniendo fₛ fija.”",
            ["Seleccione una opción", "Verdadero", "Falso"],
            index=0,
            key="g2_dyn1_q2"
        )
        q3 = st.radio(
            "3) Si la frecuencia más alta de la señal es 3 kHz, ¿cuál de estas opciones evita aliasing?",
            ["Seleccione una opción", "fₛ = 4 kHz", "fₛ = 5 kHz", "fₛ = 8 kHz"],
            index=0,
            key="g2_dyn1_q3"
        )
        enviar = st.form_submit_button("Guardar respuesta")

    if enviar:
        correct_answers = {
            "q1": "Caso A",
            "q2": "Falso",
            "q3": "fₛ = 8 kHz",
        }
        answers = {"q1": q1, "q2": q2, "q3": q3}

        correct = 0
        if q1 == correct_answers["q1"]:
            correct += 1
        if q2 == correct_answers["q2"]:
            correct += 1
        if q3 == correct_answers["q3"]:
            correct += 1

        mapping = {3: 10.0, 2: 8.0, 1: 6.0, 0: 0.0}
        score = mapping.get(correct, 0.0)

        student_info = st.session_state.get("student_info", {})
        key = {
            "descripcion": "Guía 2 - Dinámica 1 - Muestreo correcto e incorrecto",
            "frecuencia_senal_Hz": f_sig,
            "fs_caso_A_Hz": fs_bueno,
            "fs_caso_B_Hz": fs_malo,
        }

        #  En vez de generar PDF aquí, solo guardamos en session_state
        st.session_state["g2_dyn1_result"] = {
            "dyn_id": 1,
            "score": score,
            "answers": answers,
            "correct": correct_answers,
            "key": key,
        }

        st.success("Respuestas guardadas para la Dinámica 1. Continúa con las demás dinámicas.")


# =========================================================
# Dinámica 2 – Convolución
# =========================================================

def render_dinamica2():
    st.subheader("Dinámica 2 – Relación entrada–sistema–salida (convolución)")

    # Registro
    started = _render_student_registration("g2_dyn2")
    if not started:
        st.info("Complete el registro y pulse **Iniciar dinámica** para comenzar.")
        return

    st.markdown(
        "En esta dinámica se presenta una señal de entrada x[n] y un sistema LTI simple h[n]. "
        "El objetivo es predecir cualitativamente cómo será la salida y[n] antes de verla."
    )

    n = np.arange(0, 40)
    x = np.zeros_like(n, dtype=float)
    x[10:15] = 1.0  # pequeño pulso
    h = np.ones(5) / 5.0  # filtro promediador
    y = np.convolve(x, h)

    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=False)
    axs[0].stem(n, x)
    axs[0].set_title("Entrada x[n] (pulso)")
    axs[0].set_ylabel("x[n]")
    axs[0].grid(True, linestyle=":")

    n_h = np.arange(0, len(h))
    axs[1].stem(n_h, h)
    axs[1].set_title("Respuesta al impulso h[n] (promediador)")
    axs[1].set_xlabel("n")
    axs[1].set_ylabel("h[n]")
    axs[1].grid(True, linestyle=":")

    fig.tight_layout(pad=2.0)
    st.pyplot(fig)

    st.markdown("### Preguntas")

    with st.form("g2_dyn2_respuestas"):
        q1 = st.radio(
            "1) La salida y[n] será:",
            ["Seleccione una opción", "Más suave que x[n].", "Más ruidosa.", "Con impulsos más agudos."],
            index=0,
            key="g2_dyn2_q1"
        )
        q2 = st.radio(
            "2) ¿Cuál de las siguientes opciones describe mejor lo que hace el sistema?",
            [
                "Seleccione una opción",
                "Un amplificador puro.",
                "Un filtro suavizador (pasa bajas).",
                "Un generador de ruido."
            ],
            index=0,
            key="g2_dyn2_q2"
        )
        q3 = st.radio(
            "3) ¿Cuál expresión corresponde a la salida de un sistema LTI?",
            [
                "Seleccione una opción",
                "y[n] = x[n] + h[n]",
                "y[n] = x[n] · h[n]",
                "y[n] = Σ_k x[k]·h[n−k]"
            ],
            index=0,
            key="g2_dyn2_q3"
        )
        enviar = st.form_submit_button("Guardar respuesta")

    if enviar:
        correct_answers = {
            "q1": "Más suave que x[n].",
            "q2": "Un filtro suavizador (pasa bajas).",
            "q3": "y[n] = Σ_k x[k]·h[n−k]",
        }
        answers = {"q1": q1, "q2": q2, "q3": q3}

        correct = 0
        if q1 == correct_answers["q1"]:
            correct += 1
        if q2 == correct_answers["q2"]:
            correct += 1
        if q3 == correct_answers["q3"]:
            correct += 1

        mapping = {3: 10.0, 2: 8.0, 1: 6.0, 0: 0.0}
        score = mapping.get(correct, 0.0)

        student_info = st.session_state.get("student_info", {})
        key = {
            "descripcion": "Guía 2 - Dinámica 2 - Convolución y salida de un filtro promediador",
            "tipo_entrada": "pulso rectangular entre n=10 y n=14",
            "tipo_sistema": "filtro promediador de longitud 5",
        }

        # Guardar resultados en session_state (no PDF aquí)
        st.session_state["g2_dyn2_result"] = {
            "dyn_id": 2,
            "score": score,
            "answers": answers,
            "correct": correct_answers,
            "key": key,
        }

        # (Opcional) Mostrar la salida como ya la tenías:
        n_y = np.arange(0, len(y))
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 3))
        ax2.stem(n_y, y)
        ax2.set_title("Salida y[n] = x[n] * h[n]")
        ax2.set_xlabel("n")
        ax2.set_ylabel("y[n]")
        ax2.grid(True, linestyle=":")
        fig2.tight_layout(pad=2.0)
        st.pyplot(fig2)

        st.success("Respuestas guardadas para la Dinámica 2. Continúa con las demás dinámicas.")


# =========================================================
# Dinámica 3 – Respuesta en frecuencia
# =========================================================

def render_dinamica3():
    st.subheader("Dinámica 3 – Interpretación de |H(f)| y filtrado")

    # Registro
    started = _render_student_registration("g2_dyn3")
    if not started:
        st.info("Complete el registro y pulse **Iniciar dinámica** para comenzar.")
        return

    st.markdown(
        "En esta dinámica se presenta la respuesta en frecuencia de un filtro sencillo y una señal de entrada "
        "con varias componentes espectrales. El objetivo es razonar qué partes del espectro se atenúan más."
    )

    fs = 2000.0
    T = 0.05
    t = np.arange(0, T, 1.0 / fs)
    x = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 400 * t) + np.sin(2 * np.pi * 800 * t)

    N = len(t)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    idx_pos = freqs >= 0
    fpos = freqs[idx_pos]
    Xmag = np.abs(X[idx_pos]) / N

    # Diseñar dos filtros sencillos
    M = 33
    h_lp = np.ones(M) / M
    h_hp = np.zeros(M)
    h_hp[0] = 1.0
    h_hp[1] = -1.0

    H_lp = np.fft.fft(h_lp, n=N)
    H_hp = np.fft.fft(h_hp, n=N)

    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axs[0].stem(fpos, Xmag)
    axs[0].set_ylabel("|X(f)|")
    axs[0].set_title("Espectro de entrada")

    axs[1].stem(fpos, np.abs(H_lp[idx_pos]))
    axs[1].set_ylabel("|H_lp(f)|")
    axs[1].set_title("Filtro pasa bajas (referencia)")

    axs[2].stem(fpos, np.abs(H_hp[idx_pos]))
    axs[2].set_xlabel("Frecuencia (Hz)")
    axs[2].set_ylabel("|H_hp(f)|")
    axs[2].set_title("Filtro pasa altas (referencia)")

    for ax in axs:
        ax.grid(True, linestyle=":")

    fig.tight_layout(pad=2.0)
    st.pyplot(fig)

    st.markdown("### Preguntas")

    with st.form("g2_dyn3_respuestas"):
        q1 = st.radio(
            "1) Si aplicamos el filtro pasa bajas, ¿qué parte del espectro de la señal se atenúa más?",
            [
                "Seleccione una opción",
                "Las componentes de baja frecuencia.",
                "Las componentes de alta frecuencia.",
                "Todas las componentes se atenúan por igual."
            ],
            index=0,
            key="g2_dyn3_q1"
        )
        q2 = st.radio(
            "2) ¿Qué gráfico de salida |Y(f)| correspondería a un filtro pasa bajas ideal?",
            [
                "Seleccione una opción",
                "Uno donde se conservan las bajas frecuencias y se reducen las altas.",
                "Uno donde se conservan las altas frecuencias y se reducen las bajas.",
                "Uno donde solo quedan componentes en una frecuencia intermedia."
            ],
            index=0,
            key="g2_dyn3_q2"
        )
        q3 = st.radio(
            "3) ¿Qué tipo de filtro sería más apropiado para eliminar ruido de alta frecuencia superpuesto a una señal de baja frecuencia?",
            [
                "Seleccione una opción",
                "Un filtro pasa bajas.",
                "Un filtro pasa altas.",
                "Un filtro que amplifique todas las frecuencias."
            ],
            index=0,
            key="g2_dyn3_q3"
        )
        enviar = st.form_submit_button("Guardar respuesta")

    if enviar:
        correct_answers = {
            "q1": "Las componentes de alta frecuencia.",
            "q2": "Uno donde se conservan las bajas frecuencias y se reducen las altas.",
            "q3": "Un filtro pasa bajas.",
        }
        answers = {"q1": q1, "q2": q2, "q3": q3}

        correct = 0
        if q1 == correct_answers["q1"]:
            correct += 1
        if q2 == correct_answers["q2"]:
            correct += 1
        if q3 == correct_answers["q3"]:
            correct += 1

        mapping = {3: 10.0, 2: 8.0, 1: 6.0, 0: 0.0}
        score = mapping.get(correct, 0.0)

        student_info = st.session_state.get("student_info", {})
        key = {
            "descripcion": "Guía 2 - Dinámica 3 - Interpretación de respuesta en frecuencia",
            "tipo_filtro_lp": "promediador de longitud 33 (pasa bajas)",
            "tipo_filtro_hp": "diferencia [1, -1] (pasa altas)",
        }

        # Guardar resultados en session_state
        st.session_state["g2_dyn3_result"] = {
            "dyn_id": 3,
            "score": score,
            "answers": answers,
            "correct": correct_answers,
            "key": key,
        }

        st.success(
            "Respuestas guardadas para la Dinámica 3. Ve al tab enviar respuestas.")



# =========================================================
# Dinámicas integradas – Guía 2 (un solo registro + un solo envío)
# =========================================================

def _g2_student_ready() -> bool:
    info = st.session_state.get("student_info", {})
    return bool(info) and all(str(info.get(k, "")).strip() for k in ("name", "id", "dob"))


def _g2_student_form():
    """Formulario único de estudiante para todas las dinámicas de la Guía 2."""
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {"name": "", "id": "", "dob": ""}

    info = st.session_state["student_info"]

    st.subheader("Datos del estudiante")

    with st.form("g2_form_student"):
        name = st.text_input("Nombre completo", value=info.get("name", ""))
        sid = st.text_input("Carné", value=info.get("id", ""))
        dob = st.text_input("Fecha de nacimiento (YYYY-MM-DD)", value=info.get("dob", ""))
        ok = st.form_submit_button("Guardar datos")

    if ok:
        if not name.strip() or not sid.strip() or not dob.strip():
            st.error("Completa nombre, carné y fecha de nacimiento.")
        else:
            st.session_state["student_info"] = {
                "name": name.strip(),
                "id": sid.strip(),
                "dob": dob.strip(),
            }
            st.success("Datos guardados correctamente.")

    if not _g2_student_ready():
        st.markdown(
            """
            <div style="
                background-color: #fff3cd;
                color: #000000;
                padding: 12px;
                border-radius: 6px;
                border: 1px solid #ffeeba;
                font-weight: 500;
            ">
                ⚠️ Ingresa tus datos para habilitar las dinámicas.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()


def render_dinamica1_integrada():
    st.markdown("### Dinámica 1 – Muestreo correcto e incorrecto (aliasing)")

    st.markdown(
        "Se presentan dos casos de muestreo de la **misma** señal: "
        "uno que cumple el criterio de Nyquist y otro que no. "
        "El objetivo es identificar visualmente el aliasing y razonar sobre la elección de $f_s$."
    )

    # Señal de referencia
    f_sig = 3000.0
    T = 0.005  # ventana corta para ver bien muestras
    fs_bueno = 4000.0
    fs_malo = 600.0

    t_cont = np.arange(0, T, 1.0 / 200000)  # "continuo" para referencia visual
    x_cont = np.sin(2 * np.pi * f_sig * t_cont)

    t_b = np.arange(0, T, 1.0 / fs_bueno)
    x_b = np.sin(2 * np.pi * f_sig * t_b)

    t_m = np.arange(0, T, 1.0 / fs_malo)
    x_m = np.sin(2 * np.pi * f_sig * t_m)

    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    axs[0].plot(t_cont, x_cont, linewidth=1.0)
    axs[0].stem(t_b, x_b)
    axs[0].set_title(f"Caso A: muestreo con fₛ = {fs_bueno:.0f} Hz (bueno)")
    axs[0].set_ylabel("x_A[n]")
    axs[0].grid(True, linestyle=":")

    axs[1].plot(t_cont, x_cont, linewidth=1.0)
    axs[1].stem(t_m, x_m)
    axs[1].set_title(f"Caso B: muestreo con fₛ = {fs_malo:.0f} Hz (posible aliasing)")
    axs[1].set_xlabel("Tiempo (s)")
    axs[1].set_ylabel("x_B[n]")
    axs[1].grid(True, linestyle=":")

    fig.tight_layout(pad=2.0)
    st.pyplot(fig)

    st.markdown("#### Preguntas")
    st.radio(
        "1) ¿Cuál caso cumple el criterio de Nyquist y evita aliasing?",
        ["Seleccione una opción", "Caso A", "Caso B"],
        index=0,
        key="g2_dyn1_q1",
    )
    st.radio(
        "2) Verdadero o falso: “El aliasing puede corregirse aumentando solo la duración T manteniendo fₛ fija.”",
        ["Seleccione una opción", "Verdadero", "Falso"],
        index=0,
        key="g2_dyn1_q2",
    )
    st.radio(
        "3) Si la frecuencia más alta de la señal es 3 kHz, ¿cuál de estas opciones evita aliasing?",
        ["Seleccione una opción", "fₛ = 4 kHz", "fₛ = 5 kHz", "fₛ = 8 kHz"],
        index=0,
        key="g2_dyn1_q3",
    )

    a = [st.session_state.get("g2_dyn1_q1"), st.session_state.get("g2_dyn1_q2"), st.session_state.get("g2_dyn1_q3")]
    if all(v and v != "Seleccione una opción" for v in a):
        st.success("Dinámica 1 lista ✅")
    else:
        st.info("Selecciona una opción en cada pregunta para completar la Dinámica 1.")


def render_dinamica2_integrada():
    st.markdown("### Dinámica 2 – Convolución y sistema LTI (interpretación)")

    st.markdown(
        "Se muestra una señal de entrada $x[n]$ y la respuesta al impulso $h[n]$ de un sistema LTI. "
        "Con esa información, se puede predecir cualitativamente la salida $y[n]$ sin necesidad de graficarla: "
        "recordá que en sistemas LTI se cumple $y[n] = x[n] * h[n]$."
    )

    # Señal de entrada y sistema (misma lógica que la versión original)
    n = np.arange(0, 40)
    x = np.zeros_like(n, dtype=float)
    x[10:15] = 1.0  # pulso rectangular

    M = 5
    h = np.ones(M) / M  # promediador (pasa bajas)

    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=False)
    axs[0].stem(n, x)
    axs[0].set_title("Entrada x[n] (pulso)")
    axs[0].set_ylabel("x[n]")
    axs[0].grid(True, linestyle=":")

    n_h = np.arange(0, len(h))
    axs[1].stem(n_h, h)
    axs[1].set_title("Respuesta al impulso h[n] (promediador)")
    axs[1].set_xlabel("n")
    axs[1].set_ylabel("h[n]")
    axs[1].grid(True, linestyle=":")

    fig.tight_layout(pad=2.0)
    st.pyplot(fig)

    st.markdown("#### Preguntas")
    st.radio(
        "1) La salida y[n] será:",
        ["Seleccione una opción", "Más suave que x[n].", "Más ruidosa.", "Con impulsos más agudos."],
        index=0,
        key="g2_dyn2_q1",
    )
    st.radio(
        "2) ¿Cuál de las siguientes opciones describe mejor lo que hace el sistema?",
        ["Seleccione una opción", "Un amplificador puro.", "Un filtro suavizador (pasa bajas).", "Un generador de ruido."],
        index=0,
        key="g2_dyn2_q2",
    )
    st.radio(
        "3) ¿Cuál expresión corresponde a la convolución discreta?",
        ["Seleccione una opción", "y[n] = x[n] + h[n]", "y[n] = x[n] · h[n]", "y[n] = Σ_k x[k]·h[n−k]"],
        index=0,
        key="g2_dyn2_q3",
    )

    a = [st.session_state.get("g2_dyn2_q1"), st.session_state.get("g2_dyn2_q2"), st.session_state.get("g2_dyn2_q3")]
    if all(v and v != "Seleccione una opción" for v in a):
        st.success("Dinámica 2 lista ✅")
    else:
        st.info("Selecciona una opción en cada pregunta para completar la Dinámica 2.")


def render_dinamica3_integrada():
    st.markdown("### Dinámica 3 – Respuesta en frecuencia y filtrado (interpretación)")

    st.markdown(
        "Se presenta el espectro de una señal de entrada y dos respuestas en frecuencia de referencia "
        "(pasa bajas y pasa altas). La idea es identificar qué componentes se atenúan o se conservan "
        "según el tipo de filtro."
    )

    # Señal con varias sinusoides (misma lógica que la versión original)
    fs = 2000.0
    T = 0.05
    t = np.arange(0, T, 1.0 / fs)
    x = (
        np.sin(2 * np.pi * 100 * t) +
        np.sin(2 * np.pi * 400 * t) +
        np.sin(2 * np.pi * 800 * t)
    )

    N = len(t)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    idx_pos = freqs >= 0
    fpos = freqs[idx_pos]
    Xmag = np.abs(X[idx_pos]) / N

    M = 33
    h_lp = np.ones(M) / M             # pasa bajas (promediador)
    h_hp = np.array([1.0, -1.0])      # pasa altas (diferencia)

    H_lp = np.fft.fft(h_lp, n=N)
    H_hp = np.fft.fft(h_hp, n=N)

    fig, axs = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    axs[0].stem(fpos, Xmag)
    axs[0].set_ylabel("|X(f)|")
    axs[0].set_title("Espectro de entrada")

    axs[1].stem(fpos, np.abs(H_lp[idx_pos]))
    axs[1].set_ylabel("|H_lp(f)|")
    axs[1].set_title("Filtro pasa bajas (referencia)")

    axs[2].stem(fpos, np.abs(H_hp[idx_pos]))
    axs[2].set_xlabel("Frecuencia (Hz)")
    axs[2].set_ylabel("|H_hp(f)|")
    axs[2].set_title("Filtro pasa altas (referencia)")

    for ax in axs:
        ax.grid(True, linestyle=":")

    fig.tight_layout(pad=2.0)
    st.pyplot(fig)

    st.markdown("#### Preguntas")
    st.radio(
        "1) Si aplicamos el filtro pasa bajas, ¿qué parte del espectro de la señal se atenúa más?",
        [
            "Seleccione una opción",
            "Las componentes de baja frecuencia.",
            "Las componentes de alta frecuencia.",
            "Todas las componentes se atenúan por igual.",
        ],
        index=0,
        key="g2_dyn3_q1",
    )
    st.radio(
        "2) ¿Qué gráfico de salida |Y(f)| correspondería a un filtro pasa bajas ideal?",
        [
            "Seleccione una opción",
            "Uno donde se conservan las bajas frecuencias y se reducen las altas.",
            "Uno donde se conservan las altas frecuencias y se reducen las bajas.",
            "Uno donde solo quedan componentes en una frecuencia intermedia.",
        ],
        index=0,
        key="g2_dyn3_q2",
    )
    st.radio(
        "3) ¿Qué tipo de filtro sería más apropiado para eliminar ruido de alta frecuencia superpuesto a una señal de baja frecuencia?",
        [
            "Seleccione una opción",
            "Un filtro pasa bajas.",
            "Un filtro pasa altas.",
            "Un filtro que amplifique todas las frecuencias.",
        ],
        index=0,
        key="g2_dyn3_q3",
    )

    a = [st.session_state.get("g2_dyn3_q1"), st.session_state.get("g2_dyn3_q2"), st.session_state.get("g2_dyn3_q3")]
    if all(v and v != "Seleccione una opción" for v in a):
        st.success("Dinámica 3 lista ✅")
    else:
        st.info("Selecciona una opción en cada pregunta para completar la Dinámica 3.")



def render_dinamicas_guia2():
    st.markdown("## Dinámicas – Guía 2")

    _g2_student_form()
    student_info = st.session_state.get("student_info", {})

    st.markdown("---")

    with st.expander("Dinámica 1 — Muestreo y aliasing", expanded=True):
        render_dinamica1_integrada()

    with st.expander("Dinámica 2 — Convolución (entrada–sistema–salida)", expanded=True):
        render_dinamica2_integrada()

    with st.expander("Dinámica 3 — Interpretación de respuesta en frecuencia", expanded=True):
        render_dinamica3_integrada()

    st.markdown("---")

    # -------- ENVÍO FINAL --------
    disabled = (not (d1_done and d2_done and d3_done)) or st.session_state.get("g2_submitted", False)

    if st.session_state.get("g2_submitted", False):
        st.info("Ya enviaste estas respuestas ✅")

    if st.button("Enviar respuestas (subir a GitHub)", disabled=disabled, key="g2_send_github"):
        # Datos del estudiante (desde el formulario común)
        nombre = (student_info.get("name", "") or "").strip()
        registro = (student_info.get("id", "") or "").strip()
        dob = (student_info.get("dob", "") or "").strip()

        if not nombre or not registro:
            st.warning("Completa tus datos (nombre y registro) antes de enviar.")
            return

        if not REPORTLAB_AVAILABLE:
            st.error(
                "No se puede generar el PDF porque 'reportlab' no está disponible. "
                "Agrega 'reportlab' a requirements.txt."
            )
            return

        # Preparar resultados (incluye nota en el PDF, pero NO se muestra al alumno)
        nota_global = round((float(res1["nota"]) + float(res2["nota"]) + float(res3["nota"])) / 3.0, 2)
        resultados = [res1, res2, res3]

        pdf_bytes, pdf_filename = export_results_pdf_guia2_bytes(
            student_info={"name": nombre, "id": registro, "dob": dob},
            resultados=resultados,
            nota_global=nota_global,
            logo_path=LOGO_UCA_PATH if (LOGO_UCA_PATH and os.path.exists(LOGO_UCA_PATH)) else None,
        )

        repo_path = f"guia2/{pdf_filename}"
        commit_msg = f"Guía 2 - {registro} - {nombre}".strip()

        ok, info = upload_bytes_to_github_results(
            content_bytes=pdf_bytes,
            repo_path=repo_path,
            commit_message=commit_msg,
        )

        if ok:
            st.session_state["g2_submitted"] = True
            st.success("¡Listo! Respuestas enviadas y PDF subido al repositorio.")
            if isinstance(info, dict) and info.get("html_url"):
                st.link_button("Ver archivo en GitHub", info["html_url"])
            st.write("Ruta en el repositorio:", repo_path)
            st.info("Consulta tu nota con el catedrático o instructor encargado.")
        else:
            st.error(f"No se pudo subir el PDF: {info}")

# Render principal Guía 2
# =========================================================

def render_guia2():
    st.title("Guía 2: Fundamentos de señales y sistemas")

    tabs = st.tabs([
        "Objetivos",
        "Introducción teórica",
        "Materiales y equipo",
        "Ejemplos",
        "Dinámicas",
        "Conclusiones",
    ])

    with tabs[0]:
        st.markdown(OBJETIVOS2_TEXT)

    with tabs[1]:
        st.markdown(INTRO2_TEXT)

    with tabs[2]:
        st.subheader("Materiales y equipo")
        st.markdown(MATERIALES_COMUNES)

    with tabs[3]:
        st.markdown("En esta sección se presentan cuatro ejemplos interactivos.")
        sub_tabs = st.tabs(["Ejemplo 1", "Ejemplo 2", "Ejemplo 3", "Ejemplo 4"])
        with sub_tabs[0]:
            render_ejemplo1()
        with sub_tabs[1]:
            render_ejemplo2()
        with sub_tabs[2]:
            render_ejemplo3()
        with sub_tabs[3]:
            render_ejemplo4()

    with tabs[4]:
        render_dinamicas_guia2()

    with tabs[5]:
        st.markdown(CONCLUSIONES2_TEXT)
