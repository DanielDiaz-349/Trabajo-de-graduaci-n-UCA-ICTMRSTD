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
import re
import requests
import base64
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def _get_plot_theme():
    ui_theme = st.session_state.get("ui_theme")
    if ui_theme:
        theme_name = ui_theme.lower()
    else:
        base_theme = (st.get_option("theme.base") or "light").lower()
        theme_name = "obscuro" if base_theme == "dark" else "blanco"

    if theme_name == "obscuro":
        return {
            "paper_bgcolor": "#0f1113",
            "plot_bgcolor": "#2b2f36",
            "font_color": "#ffffff",
            "grid_color": "#444444",
            "axis_color": "#ffffff",
            "hover_bg": "#2b2f36",
            "hover_font": "#ffffff",
        }
    if theme_name == "rosa":
        return {
            "paper_bgcolor": "#fff6fb",
            "plot_bgcolor": "#ffffff",
            "font_color": "#330033",
            "grid_color": "#f0c6dc",
            "axis_color": "#330033",
            "hover_bg": "#ffd6eb",
            "hover_font": "#330033",
        }
    return {
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font_color": "#000000",
        "grid_color": "#d9d9d9",
        "axis_color": "#000000",
        "hover_bg": "#ffffff",
        "hover_font": "#000000",
    }


def _apply_plot_theme(fig, theme, font_size=12):
    fig.update_layout(
        paper_bgcolor=theme["paper_bgcolor"],
        plot_bgcolor=theme["plot_bgcolor"],
        font=dict(color=theme["font_color"], size=font_size),
        hoverlabel=dict(bgcolor=theme["hover_bg"], font=dict(color=theme["hover_font"])),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=theme["grid_color"],
        zerolinecolor=theme["axis_color"],
        linecolor=theme["axis_color"],
        ticks="outside",
        tickcolor=theme["axis_color"],
        tickfont=dict(color=theme["font_color"]),
        title_font=dict(color=theme["font_color"]),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=theme["grid_color"],
        zerolinecolor=theme["axis_color"],
        linecolor=theme["axis_color"],
        ticks="outside",
        tickcolor=theme["axis_color"],
        tickfont=dict(color=theme["font_color"]),
        title_font=dict(color=theme["font_color"]),
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

- Dispositivo con acceso a internet.
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

        # Respuestas correctas (si se proveen)
        correct_answers = res.get("correct_answers") or {}
        if correct_answers:
            if y < 110:
                c.showPage()
                y = height - 60
                c.setFont(base_font, 11)
            c.drawString(60, y, "Respuestas correctas:")
            y -= 14
            for k, v in correct_answers.items():
                if y < 90:
                    c.showPage()
                    y = height - 60
                    c.setFont(base_font, 11)
                c.drawString(75, y, f"- {k}: {_g2_safe_str(v)}")
                y -= 12

        # Respuestas del estudiante (marcando correctas/incorrectas)
        answers = res.get("answers") or {}
        if answers:
            if y < 110:
                c.showPage()
                y = height - 60
                c.setFont(base_font, 11)
            c.drawString(60, y, "Respuestas del estudiante (correctas/incorrectas):")
            y -= 14
            for k, v in answers.items():
                if y < 90:
                    c.showPage()
                    y = height - 60
                    c.setFont(base_font, 11)
                expected = correct_answers.get(k)
                if expected is None:
                    status = "•"
                    detail = f"{_g2_safe_str(v)}"
                else:
                    is_correct = _g2_safe_str(v) == _g2_safe_str(expected)
                    status = "✓" if is_correct else "✗"
                    detail = f"{_g2_safe_str(v)}"
                    if not is_correct:
                        detail = f"{detail} (Correcta: {_g2_safe_str(expected)})"
                c.drawString(75, y, f"- {k}: {status} {detail}")
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

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.14,
            subplot_titles=("Señal de tiempo continuo", "Señal muestreada"),
        )
        fig.add_trace(
            go.Scatter(x=t_cont, y=x_cont, mode="lines", name="x(t)"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=t_disc, y=x_disc, mode="markers", name="x[n]"),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="x(t)", row=1, col=1)
        fig.update_yaxes(title_text="x[n]", row=2, col=1)
        fig.update_xaxes(title_text="Tiempo (s)", row=2, col=1)
        fig.update_layout(
            height=600,
            margin=dict(l=40, r=20, t=80, b=60),
            hovermode="x unified",
            showlegend=False,
        )
        plot_theme = _get_plot_theme()
        _apply_plot_theme(fig, plot_theme, font_size=12)
        fig.update_annotations(font=dict(color=plot_theme["font_color"], size=13))
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # Explicación dinámica
        f_max, f_nyq, nyq_msg = _nyquist_info(f1, f2, fs)
        with st.expander("Explicación de la simulación y preguntas", expanded=True):
            st.markdown(
                "La señal continua se construye como la suma de dos sinusoides. "
                "Al muestrearla, solo se conservan muestras cada 1/fₛ segundos. "
                "La capacidad de reconstruir la señal original depende de la relación entre fₛ y la frecuencia máxima presente."
            )
            st.markdown(nyq_msg)

            # Preguntas y respuestas (conceptuales)
            st.markdown("##### Preguntas y respuestas: ")
            st.markdown("**1. ¿Qué ocurre si reducimos demasiado la frecuencia de muestreo fₛ?**")
            st.markdown(
                "**R:** La señal discreta comienza a perder detalle y puede aparecer aliasing, es decir, componentes de alta "
                "frecuencia se reflejan como frecuencias más bajas."
            )

            st.markdown(
                "**2. Si fmax = 300 Hz, ¿cuál es el valor mínimo de fₛ que respeta el criterio de Nyquist?**"
            )
            st.markdown("**R:** fₛ mínima = 2·f_max = 600 Hz.")

            st.markdown(
                "**3. ¿Qué ventaja tiene representar la señal tanto en continuo como en discreto en el mismo eje de tiempo?**"
            )
            st.markdown(
                "**R:** Permite comparar visualmente qué tanta información de la forma de onda original se conserva luego del muestreo."
            )

# =========================================================
# Ejemplo 2 – Aliasing y FFT
# =========================================================

# =========================================================
# Ejemplo 2 – Aliasing y FFT (versión sin "modo de muestreo")
# =========================================================

def _sample_sinusoid(amplitude, frequency, t_samples, fs, phase=0.0):
    nyquist_freq = fs / 2.0
    if np.isclose(abs(frequency), nyquist_freq, rtol=0.0, atol=1e-9):
        phase += np.pi / 2
    return amplitude * np.sin(2 * np.pi * frequency * t_samples + phase)


def _build_stem_lines(x_values, y_values):
    x_lines = []
    y_lines = []
    for x_val, y_val in zip(x_values, y_values):
        x_lines.extend([x_val, x_val, None])
        y_lines.extend([0, y_val, None])
    return x_lines, y_lines


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
        x_disc = _sample_sinusoid(A1, f1, t_disc, fs) + _sample_sinusoid(A2, f2, t_disc, fs)

        # --- FFT discreta y centrada en [-fs/2, fs/2] ---
        N = len(x_disc)
        X = np.fft.fft(x_disc)
        freqs = np.fft.fftfreq(N, d=1.0 / fs)      # frecuencias en Hz, positivas y negativas
        X_shift = np.fft.fftshift(X)
        freqs_shift = np.fft.fftshift(freqs)
        X_mag_shift = np.abs(X_shift) / N

        # --- Gráfica: señal discreta y espectros (banda base + réplicas) ---
        fig = make_subplots(
            rows=3,
            cols=1,
            vertical_spacing=0.18,
            subplot_titles=(
                "Señal discreta en el tiempo",
                "Espectro de magnitud centrado en [-fₛ/2, fₛ/2]",
                "Réplicas espectrales alrededor de k·fₛ (k = -2…2)",
            ),
        )

        # Señal discreta en el tiempo
        fig.add_trace(
            go.Scatter(x=t_disc, y=x_disc, mode="markers", name="x[n]"),
            row=1,
            col=1,
        )
        fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1)
        fig.update_yaxes(title_text="x[n]", row=1, col=1)

        # Espectro centrado en [-fs/2, fs/2] (banda base)
        base_color = "#1f77b4"
        base_x_lines, base_y_lines = _build_stem_lines(freqs_shift, X_mag_shift)
        fig.add_trace(
            go.Scatter(
                x=base_x_lines,
                y=base_y_lines,
                mode="lines",
                line=dict(color=base_color, width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=freqs_shift,
                y=X_mag_shift,
                mode="markers",
                name="|X(f)|",
                marker=dict(color=base_color, size=6),
            ),
            row=2,
            col=1,
        )
        fig.update_xaxes(range=[-fs / 2, fs / 2], title_text="Frecuencia (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="|X(f)|", row=2, col=1)

        # Réplicas espectrales alrededor de k·fs (k = -2…2)
        k_max = 2
        freqs_rep = np.concatenate([freqs_shift + k * fs for k in range(-k_max, k_max + 1)])
        mags_rep = np.tile(X_mag_shift, 2 * k_max + 1)

        replica_color = "#ff7f0e"
        rep_x_lines, rep_y_lines = _build_stem_lines(freqs_rep, mags_rep)
        fig.add_trace(
            go.Scatter(
                x=rep_x_lines,
                y=rep_y_lines,
                mode="lines",
                line=dict(color=replica_color, width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=freqs_rep,
                y=mags_rep,
                mode="markers",
                name="Réplicas",
                marker=dict(color=replica_color, size=6),
            ),
            row=3,
            col=1,
        )
        fig.update_xaxes(
            range=[-(k_max + 0.5) * fs, (k_max + 0.5) * fs],
            title_text="Frecuencia (Hz)",
            row=3,
            col=1,
        )
        fig.update_yaxes(title_text="|X(f)|", row=3, col=1)

        fig.update_layout(
            height=750,
            margin=dict(l=40, r=20, t=90, b=60),
            hovermode="x unified",
            showlegend=False,
        )
        plot_theme = _get_plot_theme()
        _apply_plot_theme(fig, plot_theme, font_size=12)
        fig.update_annotations(font=dict(color=plot_theme["font_color"], size=13), yshift=12)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # --- Análisis de Nyquist / aliasing ---
        f_max, f_nyq, nyq_msg = _nyquist_info(f1, f2, fs)

        with st.expander("Explicación de la simulación y preguntas", expanded=True):
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
            st.markdown(
                "Si una componente cae **exactamente** en $f_s/2$, una senoide con fase 0 se muestrea en todos ceros. "
                "Para evitar esta degeneración y representar correctamente el espectro en Nyquist, se aplica un "
                "desfase de $\\pi/2$ a esa componente."
            )

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

        n_h = np.arange(0, M)
        if tipo_filtro == "Filtro pasa bajas":
            fc = 0.2
            h = 2 * fc * np.sinc(2 * fc * (n_h - (M - 1) / 2))
            h *= np.hamming(M)
        else:
            alpha = 0.4
            h = (1 - alpha) * (alpha ** n_h)

        h = h / np.sum(h)
        y = np.convolve(x, h, mode="full")

        n_y = np.arange(0, len(y))

        fig = make_subplots(
            rows=3,
            cols=1,
            vertical_spacing=0.18,
            subplot_titles=("Entrada x[n]", "Respuesta al impulso del sistema", "Salida y[n] = x[n] * h[n]"),
        )
        fig.add_trace(go.Scatter(x=n, y=x, mode="markers", name="x[n]"), row=1, col=1)
        fig.add_trace(go.Scatter(x=n_h, y=h, mode="markers", name="h[n]"), row=2, col=1)
        fig.add_trace(go.Scatter(x=n_y, y=y, mode="markers", name="y[n]"), row=3, col=1)

        fig.update_xaxes(title_text="n", row=1, col=1)
        fig.update_xaxes(title_text="n", row=2, col=1)
        fig.update_xaxes(title_text="n", row=3, col=1)
        fig.update_yaxes(title_text="x[n]", row=1, col=1)
        fig.update_yaxes(title_text="h[n]", row=2, col=1)
        fig.update_yaxes(title_text="y[n]", row=3, col=1)

        fig.update_layout(
            height=700,
            margin=dict(l=40, r=20, t=90, b=60),
            hovermode="x unified",
            showlegend=False,
        )
        plot_theme = _get_plot_theme()
        _apply_plot_theme(fig, plot_theme, font_size=12)
        fig.update_annotations(font=dict(color=plot_theme["font_color"], size=13), yshift=12)
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # Explicación dinámica (mejor conectada con las gráficas)
        with st.expander("Explicación de la simulación y preguntas", expanded=True):
            st.markdown(
                "En un sistema LTI, toda la información del sistema está contenida en su respuesta al impulso $h[n]$. "
                "La salida se calcula con la convolución discreta:\n\n"
                "$$y[n] = \\sum_{k=-\\infty}^{\\infty} x[k]\\,h[n-k]$$\n\n"
                "Esto puede interpretarse así: para cada $n$, el sistema toma un fragmento de la entrada $x[k]$ y lo combina con los "
                "pesos del filtro $h[n]$ , es decir, una suma ponderada."
            )

            if tipo_filtro == "Filtro pasa bajas":
                st.markdown(
                    "**Caso: Filtro pasa bajas FIR (sinc con ventana).**\n\n"
                    "Aquí $h[n]$ se construye con una sinc ideal truncada y suavizada con una ventana. "
                    "Este tipo de respuesta al impulso es típica de un pasa bajas discreto.\n\n"
                    "- Cada muestra de salida es una suma ponderada de varias muestras de la entrada.\n"
                    "- Los cambios bruscos (bordes) se suavizan porque se atenúan las componentes de alta frecuencia.\n"
                    "- Al aumentar $M$, el suavizado es mayor y la transición es más lenta, pero aumenta el retardo efectivo."
                )
            else:
                st.markdown(
                    "**Caso: Suavizado exponencial o respuesta decreciente**.\n\n"
                    " Aquí $h[n]=(1-\\alpha)\\alpha^n$ para $n=0,1,\\dots,M-1$ (normalizado para conservar ganancia DC).\n\n"
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
            st.markdown(
                "**R:** Porque suaviza la señal y atenúa las variaciones rápidas (componentes de alta frecuencia), "
                "dejando pasar principalmente las variaciones lentas."
            )

            st.markdown("**2. ¿Qué interpretación física tiene la convolución en este contexto?**")
            st.markdown(
                "**R:** Cada muestra de y[n] es el resultado de sumar copias desplazadas de h[n] ponderadas por los valores de x[n]; "
                "el sistema 'promedia' o 'dispersa' la energía de la señal en el tiempo."
            )

            st.markdown("**3. ¿Cómo afectaría aumentar el valor de M en el filtro pasabajas?**")
            st.markdown(
                "**R:** El filtro se vuelve más suave: la salida cambia más lentamente y se reducen aún más las componentes de alta "
                "frecuencia, pero se pierde detalle temporal."
            )


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
            np.sin(2 * np.pi * 100 * t)
            + 0.7 * np.sin(2 * np.pi * 400 * t)
            + 0.5 * np.sin(2 * np.pi * 800 * t)
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
            h = (1 - alpha) * (alpha ** np.arange(M))

        # Señal de salida en el tiempo (convolución lineal)
        y_time = np.convolve(x, h, mode="full")
        t_out = np.arange(0, len(y_time)) / fs

        # FFT de x y h (zero-padding para coincidir con la convolución lineal)
        fft_len = N + M - 1
        X = np.fft.fft(x, n=fft_len)
        H = np.fft.fft(h, n=fft_len)
        Y = X * H

        freqs = np.fft.fftfreq(fft_len, d=1.0 / fs)
        idx_pos = freqs >= 0
        fpos = freqs[idx_pos]
        Xmag = np.abs(X[idx_pos]) / fft_len
        Hmag = np.abs(H[idx_pos])
        Ymag = np.abs(Y[idx_pos]) / fft_len

        fig = make_subplots(
            rows=5,
            cols=1,
            vertical_spacing=0.1,
            subplot_titles=(
                "Entrada en el tiempo",
                "Espectro de entrada",
                "Respuesta en frecuencia del filtro",
                "Espectro de salida",
                "Salida en el tiempo",
            ),
        )
        spectrum_colors = {
            "input": "#1f77b4",
            "filter": "#ff7f0e",
            "output": "#2ca02c",
        }
        time_colors = {
            "input": "#9467bd",
            "output": "#d62728",
        }
        fig.add_trace(
            go.Scatter(
                x=t,
                y=x,
                mode="lines",
                name="x(t)",
                line=dict(color=time_colors["input"], width=2),
            ),
            row=1,
            col=1,
        )
        x_lines, y_lines = _build_stem_lines(fpos, Xmag)
        fig.add_trace(
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line=dict(color=spectrum_colors["input"], width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fpos,
                y=Xmag,
                mode="markers",
                name="|X(f)|",
                marker=dict(color=spectrum_colors["input"], size=6),
            ),
            row=2,
            col=1,
        )
        h_lines_x, h_lines_y = _build_stem_lines(fpos, Hmag)
        fig.add_trace(
            go.Scatter(
                x=h_lines_x,
                y=h_lines_y,
                mode="lines",
                line=dict(color=spectrum_colors["filter"], width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fpos,
                y=Hmag,
                mode="markers",
                name="|H(f)|",
                marker=dict(color=spectrum_colors["filter"], size=6),
            ),
            row=3,
            col=1,
        )
        y_lines_x, y_lines_y = _build_stem_lines(fpos, Ymag)
        fig.add_trace(
            go.Scatter(
                x=y_lines_x,
                y=y_lines_y,
                mode="lines",
                line=dict(color=spectrum_colors["output"], width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=fpos,
                y=Ymag,
                mode="markers",
                name="|Y(f)|",
                marker=dict(color=spectrum_colors["output"], size=6),
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_out,
                y=y_time,
                mode="lines",
                name="y(t)",
                line=dict(color=time_colors["output"], width=2),
            ),
            row=5,
            col=1,
        )

        fig.update_xaxes(title_text="Tiempo (s)", row=1, col=1, title_standoff=18)
        fig.update_xaxes(title_text="Frecuencia (Hz)", row=2, col=1, title_standoff=18)
        fig.update_xaxes(title_text="Frecuencia (Hz)", row=3, col=1, title_standoff=18)
        fig.update_xaxes(title_text="Frecuencia (Hz)", row=4, col=1, title_standoff=18)
        fig.update_xaxes(title_text="Tiempo (s)", row=5, col=1, title_standoff=18)
        fig.update_yaxes(title_text="Amplitud", row=1, col=1)
        fig.update_yaxes(title_text="|X(f)|", row=2, col=1)
        fig.update_yaxes(title_text="|H(f)|", row=3, col=1)
        fig.update_yaxes(title_text="|Y(f)|", row=4, col=1)
        fig.update_yaxes(title_text="Amplitud", row=5, col=1)

        fig.update_layout(
            height=1050,
            margin=dict(l=40, r=20, t=110, b=80),
            hovermode="x unified",
            showlegend=False,
        )
        plot_theme = _get_plot_theme()
        _apply_plot_theme(fig, plot_theme, font_size=12)
        fig.update_annotations(font=dict(color=plot_theme["font_color"], size=13))
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # Explicación dinámica
        with st.expander("Explicación de la simulación y preguntas", expanded=True):
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
                "**R:** Porque la Transformada de Fourier convierte la convolución en el tiempo en una multiplicación en frecuencia; "
                "ambas representan la misma relación entrada–sistema–salida desde dos perspectivas distintas."
            )


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
        "En esta dinámica se observa una señal continua (suma de sinusoides) y dos muestreos: "
        "uno exactamente a la frecuencia de Nyquist y otro por debajo de ella. El objetivo es "
        "identificar el aliasing y relacionarlo con la elección de $f_s$."
    )

    f1 = 400.0
    f2 = 900.0
    f_max = max(f1, f2)
    fs_nyquist = 2.0 * f_max
    fs_bajo = 1200.0

    T = 0.012
    fs_cont = 50000.0
    t_cont = np.arange(0, T, 1.0 / fs_cont)
    x_cont = np.sin(2 * np.pi * f1 * t_cont) + 0.6 * np.sin(2 * np.pi * f2 * t_cont)

    t_n = np.arange(0, T, 1.0 / fs_nyquist)
    x_n = np.sin(2 * np.pi * f1 * t_n) + 0.6 * np.sin(2 * np.pi * f2 * t_n)

    t_b = np.arange(0, T, 1.0 / fs_bajo)
    x_b = np.sin(2 * np.pi * f1 * t_b) + 0.6 * np.sin(2 * np.pi * f2 * t_b)

    fig = make_subplots(
        rows=3,
        cols=1,
        vertical_spacing=0.12,
        subplot_titles=(
            "Gráfica 1: Señal original (continua)",
            f"Gráfica 2: Muestreo a Nyquist (fₛ = {fs_nyquist:.0f} Hz)",
            f"Gráfica 3: Muestreo por debajo de Nyquist (fₛ = {fs_bajo:.0f} Hz)",
        ),
    )
    fig.add_trace(
        go.Scatter(x=t_cont, y=x_cont, mode="lines", name="x(t)"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_n, y=x_n, mode="markers", name="x_N[n]"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_b, y=x_b, mode="markers", name="x_B[n]"),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="x(t)", row=1, col=1)
    fig.update_yaxes(title_text="x_N[n]", row=2, col=1)
    fig.update_yaxes(title_text="x_B[n]", row=3, col=1)
    fig.update_xaxes(title_text="Tiempo (s)", row=3, col=1)
    fig.update_layout(
        height=720,
        margin=dict(l=40, r=20, t=80, b=60),
        hovermode="x unified",
        showlegend=False,
    )
    plot_theme = _get_plot_theme()
    _apply_plot_theme(fig, plot_theme, font_size=12)
    fig.update_annotations(font=dict(color=plot_theme["font_color"], size=13))
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown(
        f"**Frecuencia máxima en la Gráfica 1:** $f_{{\\max}} = {f_max:.0f}\\ \\text{{Hz}}$ "
        "(componente de mayor frecuencia de la señal original)."
    )

    st.markdown("### Preguntas")

    with st.form("g2_dyn1_respuestas"):
        q1 = st.radio(
            "1) ¿Qué gráfica corresponde al muestreo exactamente a la frecuencia de Nyquist?",
            ["Seleccione una opción", "Gráfica 2", "Gráfica 3"],
            index=0,
            key="g2_dyn1_q1"
        )
        q2 = st.radio(
            "2) ¿En cuál gráfica se observa aliasing por muestreo insuficiente?",
            ["Seleccione una opción", "Gráfica 2", "Gráfica 3"],
            index=0,
            key="g2_dyn1_q2"
        )
        q3 = st.radio(
            f"3) Si la frecuencia más alta es {f_max:.0f} Hz, ¿cuál es la fₛ mínima para evitar aliasing?",
            ["Seleccione una opción", "1200 Hz", f"{fs_nyquist:.0f} Hz", "2400 Hz"],
            index=0,
            key="g2_dyn1_q3"
        )
        enviar = st.form_submit_button("Guardar respuesta")

    if enviar:
        correct_answers = {
            "q1": "Gráfica 2",
            "q2": "Gráfica 3",
            "q3": f"{fs_nyquist:.0f} Hz",
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
            "frecuencias_senal_Hz": [f1, f2],
            "fs_nyquist_Hz": fs_nyquist,
            "fs_bajo_Hz": fs_bajo,
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
# Dinámica 2 – Aliasing y análisis en frecuencia
# =========================================================

def render_dinamica2():
    st.subheader("Dinámica 2 – Aliasing en el muestreo y análisis en frecuencia")

    # Registro
    started = _render_student_registration("g2_dyn2")
    if not started:
        st.info("Complete el registro y pulse **Iniciar dinámica** para comenzar.")
        return

    st.markdown(
        "Se muestra el espectro de una señal muestreada alrededor de múltiplos de $k\\,f_s$ "
        "(con $k=-2,\\dots,2$). La primera gráfica ilustra aliasing (réplicas superpuestas) "
        "y la segunda cumple Nyquist, evitando superposición."
    )

    A1 = 1.0
    f1 = 200.0
    A2 = 0.8
    f2 = 650.0
    f_max = max(f1, f2)
    fs_alias = 800.0
    fs_nyquist = 2000.0
    k_min, k_max = -2, 2

    def _replicated_spectrum(freqs, amps, fs, k_min, k_max):
        replica_freqs = []
        replica_amps = []
        for k in range(k_min, k_max + 1):
            for freq, amp in zip(freqs, amps):
                replica_freqs.extend([freq + k * fs, -freq + k * fs])
                replica_amps.extend([amp / 2.0, amp / 2.0])
        return replica_freqs, replica_amps

    def _plot_replicas(fig, freqs, amps, fs, row, col):
        replica_freqs, replica_amps = _replicated_spectrum(freqs, amps, fs, k_min, k_max)
        x_lines, y_lines = _build_stem_lines(replica_freqs, replica_amps)
        fig.add_trace(
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line=dict(color="#1f77b4", width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=replica_freqs,
                y=replica_amps,
                mode="markers",
                marker=dict(color="#1f77b4", size=6),
                name="|X_s(f)|",
            ),
            row=row,
            col=col,
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.18,
        subplot_titles=(
            f"Gráfica 1: Espectro alrededor de k·fₛ (fₛ = {fs_alias:.0f} Hz, aliasing)",
            f"Gráfica 2: Espectro alrededor de k·fₛ (fₛ = {fs_nyquist:.0f} Hz, sin aliasing)",
        ),
    )

    _plot_replicas(fig, [f1, f2], [A1, A2], fs_alias, row=1, col=1)
    _plot_replicas(fig, [f1, f2], [A1, A2], fs_nyquist, row=2, col=1)

    fig.update_yaxes(title_text="|X_s(f)|", row=1, col=1)
    fig.update_yaxes(title_text="|X_s(f)|", row=2, col=1)
    fig.update_xaxes(
        title_text="Frecuencia (Hz)",
        row=1,
        col=1,
        range=[(k_min - 0.5) * fs_alias, (k_max + 0.5) * fs_alias],
    )
    fig.update_xaxes(
        title_text="Frecuencia (Hz)",
        row=2,
        col=1,
        range=[(k_min - 0.5) * fs_nyquist, (k_max + 0.5) * fs_nyquist],
    )
    fig.update_layout(
        height=560,
        margin=dict(l=40, r=20, t=80, b=60),
        hovermode="x unified",
        showlegend=False,
    )
    plot_theme = _get_plot_theme()
    _apply_plot_theme(fig, plot_theme, font_size=12)
    fig.update_annotations(font=dict(color=plot_theme["font_color"], size=13))
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("### Preguntas")

    with st.form("g2_dyn2_respuestas"):
        q1 = st.radio(
            "1) ¿En cuál gráfica se observa aliasing por superposición de réplicas?",
            ["Seleccione una opción", "Gráfica 1", "Gráfica 2"],
            index=0,
            key="g2_dyn2_q1"
        )
        q2 = st.radio(
            "2) ¿Qué condición se cumple en la Gráfica 2 para evitar aliasing?",
            ["Seleccione una opción", "fₛ ≥ 2·f_max", "fₛ = f_max", "fₛ ≤ f_max/2"],
            index=0,
            key="g2_dyn2_q2"
        )
        q3 = st.radio(
            f"3) Si f_max = {f_max:.0f} Hz, ¿cuál es la fₛ mínima para evitar aliasing?",
            ["Seleccione una opción", "1000 Hz", f"{2 * f_max:.0f} Hz", "1600 Hz"],
            index=0,
            key="g2_dyn2_q3"
        )
        enviar = st.form_submit_button("Guardar respuesta")

    if enviar:
        correct_answers = {
            "q1": "Gráfica 1",
            "q2": "fₛ ≥ 2·f_max",
            "q3": f"{2 * f_max:.0f} Hz",
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
            "descripcion": "Guía 2 - Dinámica 2 - Aliasing en muestreo y análisis en frecuencia",
            "A1": A1,
            "f1_Hz": f1,
            "A2": A2,
            "f2_Hz": f2,
            "fs_alias_Hz": fs_alias,
            "fs_nyquist_Hz": fs_nyquist,
            "k_range": [k_min, k_max],
        }

        # Guardar resultados en session_state (no PDF aquí)
        st.session_state["g2_dyn2_result"] = {
            "dyn_id": 2,
            "score": score,
            "answers": answers,
            "correct": correct_answers,
            "key": key,
        }

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

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Espectro de entrada",
            "Filtro pasa bajas (referencia)",
            "Filtro pasa altas (referencia)",
        ),
    )
    fig.add_trace(
        go.Scatter(x=fpos, y=Xmag, mode="markers", name="|X(f)|"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=fpos, y=np.abs(H_lp[idx_pos]), mode="markers", name="|H_lp(f)|"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=fpos, y=np.abs(H_hp[idx_pos]), mode="markers", name="|H_hp(f)|"),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="|X(f)|", row=1, col=1)
    fig.update_yaxes(title_text="|H_lp(f)|", row=2, col=1)
    fig.update_yaxes(title_text="|H_hp(f)|", row=3, col=1)
    fig.update_xaxes(title_text="Frecuencia (Hz)", row=3, col=1)
    fig.update_layout(
        height=680,
        margin=dict(l=40, r=20, t=90, b=60),
        hovermode="x unified",
        showlegend=False,
    )
    plot_theme = _get_plot_theme()
    _apply_plot_theme(fig, plot_theme, font_size=12)
    fig.update_annotations(font=dict(color=plot_theme["font_color"], size=13))
    st.plotly_chart(fig, use_container_width=True, theme=None)

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
        "Se presenta la señal original (continua) y dos muestreos: "
        "uno justo a la frecuencia de Nyquist y otro por debajo. "
        "El objetivo es identificar visualmente el aliasing y razonar sobre la elección de $f_s$."
    )

    # Señal de referencia
    f1 = 400.0
    f2 = 900.0
    f_max = max(f1, f2)
    T = 0.012
    fs_nyquist = 2.0 * f_max
    fs_bajo = 1200.0
    fs_cont = 50000.0

    t_cont = np.arange(0, T, 1.0 / fs_cont)
    x_cont = np.sin(2 * np.pi * f1 * t_cont) + 0.6 * np.sin(2 * np.pi * f2 * t_cont)

    t_n = np.arange(0, T, 1.0 / fs_nyquist)
    x_n = np.sin(2 * np.pi * f1 * t_n) + 0.6 * np.sin(2 * np.pi * f2 * t_n)

    t_b = np.arange(0, T, 1.0 / fs_bajo)
    x_b = np.sin(2 * np.pi * f1 * t_b) + 0.6 * np.sin(2 * np.pi * f2 * t_b)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Gráfica 1: Señal original (continua)",
            f"Gráfica 2: Muestreo a Nyquist (fₛ = {fs_nyquist:.0f} Hz)",
            f"Gráfica 3: Muestreo por debajo de Nyquist (fₛ = {fs_bajo:.0f} Hz)",
        ),
    )
    fig.add_trace(
        go.Scatter(x=t_cont, y=x_cont, mode="lines", name="x(t)"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_n, y=x_n, mode="markers", name="x_N[n]"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t_b, y=x_b, mode="markers", name="x_B[n]"),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="x(t)", row=1, col=1)
    fig.update_yaxes(title_text="x_N[n]", row=2, col=1)
    fig.update_yaxes(title_text="x_B[n]", row=3, col=1)
    fig.update_xaxes(title_text="Tiempo (s)", row=3, col=1)
    fig.update_layout(
        height=720,
        margin=dict(l=40, r=20, t=80, b=60),
        hovermode="x unified",
        showlegend=False,
    )
    plot_theme = _get_plot_theme()
    _apply_plot_theme(fig, plot_theme, font_size=12)
    fig.update_annotations(font=dict(color=plot_theme["font_color"], size=13))
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown(
        f"**Frecuencia máxima en la Gráfica 1:** $f_{{\\max}} = {f_max:.0f}\\ \\text{{Hz}}$ "
        "(componente de mayor frecuencia de la señal original)."
    )

    st.markdown("#### Preguntas")
    st.radio(
        "1) ¿Qué gráfica corresponde al muestreo exactamente a la frecuencia de Nyquist?",
        ["Seleccione una opción", "Gráfica 2", "Gráfica 3"],
        index=0,
        key="g2_dyn1_q1",
    )
    st.radio(
        "2) ¿En cuál gráfica se observa aliasing por muestreo insuficiente?",
        ["Seleccione una opción", "Gráfica 2", "Gráfica 3"],
        index=0,
        key="g2_dyn1_q2",
    )
    st.radio(
        f"3) Si la frecuencia más alta es {f_max:.0f} Hz, ¿cuál es la fₛ mínima para evitar aliasing?",
        ["Seleccione una opción", "1200 Hz", f"{fs_nyquist:.0f} Hz", "2400 Hz"],
        index=0,
        key="g2_dyn1_q3",
    )

    a = [st.session_state.get("g2_dyn1_q1"), st.session_state.get("g2_dyn1_q2"), st.session_state.get("g2_dyn1_q3")]
    if all(v and v != "Seleccione una opción" for v in a):
        st.success("Dinámica 1 lista ✅")
    else:
        st.info("Selecciona una opción en cada pregunta para completar la Dinámica 1.")


def render_dinamica2_integrada():
    st.markdown("### Dinámica 2 – Aliasing en el muestreo y análisis en frecuencia")

    st.markdown(
        "Se muestra el espectro de una señal muestreada alrededor de múltiplos de $k\\,f_s$. "
        "La primera gráfica presenta aliasing por superposición de réplicas; la segunda cumple "
        "Nyquist y evita esa superposición."
    )

    A1 = 1.0
    f1 = 200.0
    A2 = 0.8
    f2 = 650.0
    f_max = max(f1, f2)
    fs_alias = 800.0
    fs_nyquist = 2000.0
    k_min, k_max = -2, 2

    def _replicated_spectrum(freqs, amps, fs, k_min, k_max):
        replica_freqs = []
        replica_amps = []
        for k in range(k_min, k_max + 1):
            for freq, amp in zip(freqs, amps):
                replica_freqs.extend([freq + k * fs, -freq + k * fs])
                replica_amps.extend([amp / 2.0, amp / 2.0])
        return replica_freqs, replica_amps

    def _plot_replicas(fig, freqs, amps, fs, row, col):
        replica_freqs, replica_amps = _replicated_spectrum(freqs, amps, fs, k_min, k_max)
        x_lines, y_lines = _build_stem_lines(replica_freqs, replica_amps)
        fig.add_trace(
            go.Scatter(
                x=x_lines,
                y=y_lines,
                mode="lines",
                line=dict(color="#1f77b4", width=1),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=replica_freqs,
                y=replica_amps,
                mode="markers",
                marker=dict(color="#1f77b4", size=6),
                name="|X_s(f)|",
            ),
            row=row,
            col=col,
        )

    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.18,
        subplot_titles=(
            f"Gráfica 1: Espectro alrededor de k·fₛ (fₛ = {fs_alias:.0f} Hz, aliasing)",
            f"Gráfica 2: Espectro alrededor de k·fₛ (fₛ = {fs_nyquist:.0f} Hz, sin aliasing)",
        ),
    )

    _plot_replicas(fig, [f1, f2], [A1, A2], fs_alias, row=1, col=1)
    _plot_replicas(fig, [f1, f2], [A1, A2], fs_nyquist, row=2, col=1)

    fig.update_yaxes(title_text="|X_s(f)|", row=1, col=1)
    fig.update_yaxes(title_text="|X_s(f)|", row=2, col=1)
    fig.update_xaxes(
        title_text="Frecuencia (Hz)",
        row=1,
        col=1,
        range=[(k_min - 0.5) * fs_alias, (k_max + 0.5) * fs_alias],
    )
    fig.update_xaxes(
        title_text="Frecuencia (Hz)",
        row=2,
        col=1,
        range=[(k_min - 0.5) * fs_nyquist, (k_max + 0.5) * fs_nyquist],
    )
    fig.update_layout(
        height=560,
        margin=dict(l=40, r=20, t=80, b=60),
        hovermode="x unified",
        showlegend=False,
    )
    plot_theme = _get_plot_theme()
    _apply_plot_theme(fig, plot_theme, font_size=12)
    fig.update_annotations(font=dict(color=plot_theme["font_color"], size=13))
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("#### Preguntas")
    st.radio(
        "1) ¿En cuál gráfica se observa aliasing por superposición de réplicas?",
        ["Seleccione una opción", "Gráfica 1", "Gráfica 2"],
        index=0,
        key="g2_dyn2_q1",
    )
    st.radio(
        "2) ¿Qué condición se cumple en la Gráfica 2 para evitar aliasing?",
        ["Seleccione una opción", "fₛ ≥ 2·f_max", "fₛ = f_max", "fₛ ≤ f_max/2"],
        index=0,
        key="g2_dyn2_q2",
    )
    st.radio(
        f"3) Si f_max = {f_max:.0f} Hz, ¿cuál es la fₛ mínima para evitar aliasing?",
        ["Seleccione una opción", "1000 Hz", f"{2 * f_max:.0f} Hz", "1600 Hz"],
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

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=(
            "Espectro de entrada",
            "Filtro pasa bajas (referencia)",
            "Filtro pasa altas (referencia)",
        ),
    )
    fig.add_trace(
        go.Scatter(x=fpos, y=Xmag, mode="markers", name="|X(f)|"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=fpos, y=np.abs(H_lp[idx_pos]), mode="markers", name="|H_lp(f)|"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=fpos, y=np.abs(H_hp[idx_pos]), mode="markers", name="|H_hp(f)|"),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="|X(f)|", row=1, col=1)
    fig.update_yaxes(title_text="|H_lp(f)|", row=2, col=1)
    fig.update_yaxes(title_text="|H_hp(f)|", row=3, col=1)
    fig.update_xaxes(title_text="Frecuencia (Hz)", row=3, col=1)
    fig.update_layout(
        height=680,
        margin=dict(l=40, r=20, t=90, b=60),
        hovermode="x unified",
        showlegend=False,
    )
    plot_theme = _get_plot_theme()
    _apply_plot_theme(fig, plot_theme, font_size=12)
    fig.update_annotations(font=dict(color=plot_theme["font_color"], size=13))
    st.plotly_chart(fig, use_container_width=True, theme=None)

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

    with st.expander("Dinámica 2 — Aliasing y análisis en frecuencia", expanded=True):
        render_dinamica2_integrada()

    with st.expander("Dinámica 3 — Interpretación de respuesta en frecuencia", expanded=True):
        render_dinamica3_integrada()

        st.markdown("---")

    # -------- EVALUACIÓN Y ENVÍO FINAL --------
    # Respuestas seleccionadas (se guardan automáticamente por los keys de Streamlit)
    # ---------- TOMAR RESPUESTAS DESDE SESSION_STATE (mismas keys que usan las radios) ----------
    d1_ans = {
        "q1": st.session_state.get("g2_dyn1_q1", "Seleccione una opción"),
        "q2": st.session_state.get("g2_dyn1_q2", "Seleccione una opción"),
        "q3": st.session_state.get("g2_dyn1_q3", "Seleccione una opción"),
    }

    d2_ans = {
        "q1": st.session_state.get("g2_dyn2_q1", "Seleccione una opción"),
        "q2": st.session_state.get("g2_dyn2_q2", "Seleccione una opción"),
        "q3": st.session_state.get("g2_dyn2_q3", "Seleccione una opción"),
    }

    d3_ans = {
        "q1": st.session_state.get("g2_dyn3_q1", "Seleccione una opción"),
        "q2": st.session_state.get("g2_dyn3_q2", "Seleccione una opción"),
        "q3": st.session_state.get("g2_dyn3_q3", "Seleccione una opción"),
    }


    def _done(ans_dict: dict) -> bool:
        return all((v is not None) and (str(v).strip() != "") and (v != "Seleccione una opción") for v in ans_dict.values())

    d1_done = _done(d1_ans)
    d2_done = _done(d2_ans)
    d3_done = _done(d3_ans)

    # Claves (respuestas correctas) — ajustadas a las preguntas de cada dinámica
    d1_f1 = 400.0
    d1_f2 = 900.0
    d1_f_max = max(d1_f1, d1_f2)
    d1_fs_nyquist = 2.0 * d1_f_max
    d1_fs_bajo = 1200.0
    d1_t = 0.012

    d2_a1 = 1.0
    d2_f1 = 200.0
    d2_a2 = 0.8
    d2_f2 = 650.0
    d2_f_max = max(d2_f1, d2_f2)
    d2_fs_alias = 800.0
    d2_fs_nyquist = 2000.0
    d2_k_range = [-2, 2]

    d3_fs = 2000.0
    d3_m = 33
    d3_hp = [1.0, -1.0]

    d1_corr = {
        "q1": "Gráfica 2",
        "q2": "Gráfica 3",
        "q3": f"{d1_fs_nyquist:.0f} Hz",
    }
    d2_corr = {
        "q1": "Gráfica 1",
        "q2": "fₛ ≥ 2·f_max",
        "q3": f"{2 * d2_f_max:.0f} Hz",
    }
    d3_corr = {
        "q1": "Las componentes de alta frecuencia.",
        "q2": "Uno donde se conservan las bajas frecuencias y se reducen las altas.",
        "q3": "Un filtro pasa bajas.",
    }

    def _count_correct(ans: dict, corr: dict) -> int:
        return sum(1 for k, v in corr.items() if ans.get(k) == v)

    c1 = _count_correct(d1_ans, d1_corr)
    c2 = _count_correct(d2_ans, d2_corr)
    c3 = _count_correct(d3_ans, d3_corr)

    # Notas por dinámica (escala 0–10)
    score_map_3 = {3: 10.0, 2: 8.0, 1: 6.0, 0: 0.0}
    score1 = score_map_3.get(c1, 0.0)
    score2 = score_map_3.get(c2, 0.0)
    score3 = score_map_3.get(c3, 0.0)

    nota_global = round((score1 + score2 + score3) / 3.0, 2)

    # Resumen para PDF
    # (Los parámetros de cada dinámica se fijan acá para que queden documentados)
    d1_questions = {
        "1) ¿Qué gráfica corresponde al muestreo exactamente a la frecuencia de Nyquist?": d1_ans["q1"],
        "2) ¿En cuál gráfica se observa aliasing por muestreo insuficiente?": d1_ans["q2"],
        f"3) Si la frecuencia más alta es {d1_f_max:.0f} Hz, ¿cuál es la fₛ mínima para evitar aliasing?": d1_ans["q3"],
    }
    d1_correct_text = {
        "1) ¿Qué gráfica corresponde al muestreo exactamente a la frecuencia de Nyquist?": d1_corr["q1"],
        "2) ¿En cuál gráfica se observa aliasing por muestreo insuficiente?": d1_corr["q2"],
        f"3) Si la frecuencia más alta es {d1_f_max:.0f} Hz, ¿cuál es la fₛ mínima para evitar aliasing?": d1_corr["q3"],
    }

    d2_questions = {
        "1) ¿En cuál gráfica se observa aliasing por superposición de réplicas?": d2_ans["q1"],
        "2) ¿Qué condición se cumple en la Gráfica 2 para evitar aliasing?": d2_ans["q2"],
        f"3) Si f_max = {d2_f_max:.0f} Hz, ¿cuál es la fₛ mínima para evitar aliasing?": d2_ans["q3"],
    }
    d2_correct_text = {
        "1) ¿En cuál gráfica se observa aliasing por superposición de réplicas?": d2_corr["q1"],
        "2) ¿Qué condición se cumple en la Gráfica 2 para evitar aliasing?": d2_corr["q2"],
        f"3) Si f_max = {d2_f_max:.0f} Hz, ¿cuál es la fₛ mínima para evitar aliasing?": d2_corr["q3"],
    }

    d3_questions = {
        "1) Si aplicamos el filtro pasa bajas, ¿qué parte del espectro de la señal se atenúa más?": d3_ans["q1"],
        "2) ¿Qué gráfico de salida |Y(f)| correspondería a un filtro pasa bajas ideal?": d3_ans["q2"],
        "3) ¿Qué tipo de filtro sería más apropiado para eliminar ruido de alta frecuencia superpuesto a una señal de baja frecuencia?": d3_ans["q3"],
    }
    d3_correct_text = {
        "1) Si aplicamos el filtro pasa bajas, ¿qué parte del espectro de la señal se atenúa más?": d3_corr["q1"],
        "2) ¿Qué gráfico de salida |Y(f)| correspondería a un filtro pasa bajas ideal?": d3_corr["q2"],
        "3) ¿Qué tipo de filtro sería más apropiado para eliminar ruido de alta frecuencia superpuesto a una señal de baja frecuencia?": d3_corr["q3"],
    }

    res1 = {
        "titulo": "Dinámica 1 — Muestreo y aliasing",
        "correctas": c1,
        "total": 3,
        "nota": score1,
        "key": {
            "f1 (Hz)": d1_f1,
            "f2 (Hz)": d1_f2,
            "f_max (Hz)": d1_f_max,
            "f_s Nyquist (Hz)": d1_fs_nyquist,
            "f_s bajo (Hz)": d1_fs_bajo,
            "T (s)": d1_t,
        },
        "correct_answers": d1_correct_text,
        "answers": d1_questions,
    }

    res2 = {
        "titulo": "Dinámica 2 — Aliasing y análisis en frecuencia",
        "correctas": c2,
        "total": 3,
        "nota": score2,
        "key": {
            "A1": d2_a1,
            "f1 (Hz)": d2_f1,
            "A2": d2_a2,
            "f2 (Hz)": d2_f2,
            "f_max (Hz)": d2_f_max,
            "f_s alias (Hz)": d2_fs_alias,
            "f_s Nyquist (Hz)": d2_fs_nyquist,
            "k range": d2_k_range,
        },
        "correct_answers": d2_correct_text,
        "answers": d2_questions,
    }

    res3 = {
        "titulo": "Dinámica 3 — Filtrado en frecuencia",
        "correctas": c3,
        "total": 3,
        "nota": score3,
        "key": {
            "f_s (Hz)": d3_fs,
            "M (coeficientes)": d3_m,
            "Filtro pasa altas (h[n])": d3_hp,
        },
        "correct_answers": d3_correct_text,
        "answers": d3_questions,
    }

    resultados = [res1, res2, res3]

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

    # Conclusiones se renderiza antes que Dinámicas para que esté siempre
    # disponible (la dinámica usa st.stop() al faltar datos del estudiante).
    with tabs[5]:
        st.markdown(CONCLUSIONES2_TEXT)

    with tabs[4]:
        render_dinamicas_guia2()
