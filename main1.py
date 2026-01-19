# main1.py
# -*- coding: utf-8 -*-
"""
Guía interactiva 1 en Streamlit: Ruido en sistemas de telecomunicación
Requisitos: python3, streamlit, numpy, matplotlib, plotly.
Opcional: scipy, reportlab.
"""

# =========================
# IMPORTS (no usar st antes de set_page_config)
# =========================
from pathlib import Path
import os
import math
import json
import random
import datetime
import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
from io import BytesIO
import datetime

# PDF (en memoria) para subir resultados sin escribir archivos locales
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rcanvas
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import guia2
import guia3
import guia4
import guia5

from github_uploader import upload_bytes_to_github_results


# =========================
# TEMA DE GRÁFICAS
# =========================
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


# =========================
# CONSTANTES / PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
LOGO_UCA_PATH = str(BASE_DIR / "assets" / "logo_uca.png")

TEMA_TG = (
    "Introducción a la caracterización y tratamiento matemático del ruido "
    "en sistemas de telecomunicaciones digitales"
)


# =========================
# STREAMLIT CONFIG (DEBE IR ANTES DE CUALQUIER st.*)
# =========================
st.set_page_config(
    page_title="Introducción a la caracterización y tratamiento matemático del ruido",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# OPTIONAL LIBS
# =========================
try:
    from scipy.signal import butter, lfilter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False



# =========================
# RESULTS DIR


# =========================
# SESSION STATE INIT (GUÍA 1)
# =========================
if "guia1_dinamicas" not in st.session_state:
    st.session_state.guia1_dinamicas = {
        "student": {"name": "", "id": "", "dob": ""},
        "dyn1": {"key": None, "answers": {}, "completed": False, "sim": None},
        "dyn2": {"key": None, "answers": {}, "completed": False, "sim": None},
        "submitted": False,
    }

# Para compatibilidad con tu código viejo (si algo lo usa)
if "student_info" not in st.session_state:
    st.session_state.student_info = {"name": "", "id": "", "dob": ""}


# =========================
# TEXTOS ESTÁTICOS
# =========================
OBJETIVOS_TEXT = """**Objetivo general**

Analizar diferentes fuentes de ruido y los efectos que producen sobre señales en sistemas de telecomunicaciones digitales, y proporcionar herramientas prácticas para que el estudiante pueda cuantificar y visualizar la degradación de la información en diferentes tipos de canales de transmisión.

**Objetivos específicos**

- Introducir al estudiante al modelo general del ruido térmico (AWGN) y la distorsión por intermodulación sobre señales digitales y espectros.
- Comparar características y limitaciones de distintos canales (par trenzado, cable coaxial, guía de onda, fibra óptica, espacio libre) y su efecto sobre la propagación y atenuación de la señal.
- Introducir al estudiante a los conceptos de indicadores de desempeño en sistemas de telecomunicación
- Proporcionar al estudiante herramientas de aprendizaje y retroalimentación
"""

INTRO_FULL_TEXT = r"""

En todo sistema de telecomunicaciones existe un elemento inevitable: la incertidumbre. Dicha incertidumbre proviene tanto de la naturaleza aleatoria de la información como de la presencia permanente de perturbaciones indeseadas agregadas a una señal, denominadas ruido. Por ello, el análisis y diseño de sistemas modernos de comunicación exige un enfoque basado en probabilidad y procesos aleatorios. A partir de la década de 1940, se adoptaron formalmente métodos probabilísticos para optimizar el desempeño de sistemas de comunicación.

**Modelo general de un sistema de comunicación**

Un sistema de comunicación típico está compuesto por transmisor, canal y receptor, junto con los transductores de entrada y salida. El transductor convierte magnitudes físicas en señales eléctricas, el transmisor modula y adapta la señal al canal, el canal transporta la señal con pérdidas e interferencias, y el receptor intenta recuperar la información original compensando distorsiones generadas a lo largo del trayecto.

La calidad con la que la señal llega al receptor depende de dos factores principales:

- Las características del canal.
- Las perturbaciones añadidas a la señal, entre ellas el ruido y la distorsión por no linealidad.

**Canales de transmisión**

Existen muchos tipos de canales de transmisión, entre los más comunes se encuentran: canales de propagación de ondas electromagnéticas, canales de propagación de ondas electromagnéticas guiados y enlaces ópticos. 

Canales de propagación de ondas electromagnéticas

El principio básico involucrado es el acoplamiento de la energía electromagnética con un medio de propagación, el cual puede ser el espacio libre o la atmósfera, mediante un elemento de radiación conocido como antena. 

Canales guiados y enlaces opticos

Incluyen par trenzado, cables coaxiales, guías de onda metálicas y fibras ópticas. En estos medios la señal permanece confinada físicamente y la propagación depende de parámetros eléctricos del material como permitividad y permeabilidad. Estos canales se emplean donde se requiere alta capacidad, baja atenuación o inmunidad al ruido externo.

**Naturaleza y clasificación del ruido**

El ruido se define como una señal aleatoria que se adhiere a la señal original e introduce incertidumbre en la detección. Puede ser correlacionado (producto de la no linealidad y dependiente de la señal) o no correlacionado (presente incluso sin señal).

Ruido no correlacionado

Incluye las fuentes externas (atmosféricas, extraterrestres e industriales) y las internas generadas por dispositivos electrónicos (ruido de disparo, de tiempo de tránsito y ruido térmico, este ultimo es considerado el más importante por su carácter aditivo y su presencia en todas las frecuencias).

Ruido correlacionado

Resulta de la no linealidad de los dispositivos e incluye:

- Distorsión armónica, donde aparecen armónicas de la señal fundamental.
- Intermodulación, donde múltiples señales interactúan dentro de un dispositivo no lineal generando nuevas frecuencias.

La intermodulación es especialmente crítica en sistemas multicanal, pues produce términos que pueden caer dentro del ancho de banda útil e interferir directamente con la señal deseada.

**Distorsión por intermodulación**

La distorsión por intermodulación ocurre cuando dos o más señales de diferentes frecuencias atraviesan un dispositivo no lineal, como un amplificador o mezclador, produciendo componentes de frecuencia adicionales que son sumas y diferencias de las frecuencias originales y sus múltiplos armónicos. Estas nuevas frecuencias llamadas productos de intermodulación no estaban presentes en la entrada y pueden caer dentro del ancho de banda útil, interfiriendo con la señal original o con canales adyacentes.

Se introduce el termino coeficiente cubico o k3 que es un parámetro que aparece en el modelo polinómico utilizado para describir la no linealidad de un amplificador o cualquier dispositivo activo. Es fundamental para entender la intermodulación de tercer orden (IM3), que es la forma más problemática de distorsión en sistemas de RF y comunicaciones.

Cuando dos señales de entrada de frecuencias f1 y f2 pasan por un dispositivo no lineal, aparecen nuevas frecuencias dadas por la **ecuación (1)**

$$
f_{IM} = m f_1 \pm n f_2 \tag{1}
$$



Donde:

m y n son enteros positivos
la suma o resta da lugar a distintas combinaciones
los terminos m+n=k se llaman productos de intermodulación de orden k

Los productos de tecer orden son los más problematicos porque sus frecuencias son muy cercanas a las señales originales, lo que los hace difíciles de eliminar con filtros. 

Por ejemplo, un amplificador es un dispositivo no lineal de tercer orden, los productos de intermodulación son: 

 2f1-f2, 2f2-f1, 3f1, 3f2, 2f1+f2}, 2f2+f1.


**Adición de ruido a una señal**

En un sistema real, el canal no solo atenúa y distorsiona la señal sino que introduce ruido. La señal recibida puede representarse en la **ecuación (2)**:

$$
r(t) = s(t) + n(t) \tag{2}
$$



donde:

- s(t) es la señal transmitida posiblemente distorsionada
- n(t) corresponde al ruido agregado por fuentes internas y externas.

**Ruido blanco aditivo gaussiano**

Para análisis y diseño, una de las aproximaciones más utilizadas es el modelo AWGN (Additive White Gaussian Noise), que representa un tipo de ruido:

- Aditivo: se suma linealmente a la señal.
- Blanco: tiene potencia constante en todas las frecuencias.
- Gaussiano: su amplitud sigue una distribución normal.

El AWGN modela adecuadamente:

- Ruido térmico en componentes electrónicos.
- Ruido de fondo en canales de radio.
- El comportamiento agregado de muchas fuentes pequeñas e independientes.

Aunque simplificado, este modelo es ampliamente aceptado en el diseño teórico de moduladores y detectores digitales y analógicos por su precisión estadística y su facilidad matemática.

**Relación Señal-Ruido (SNR)**

El desempeño de un sistema en presencia de ruido suele medirse con la relación señal-ruido (SNR) definido en la **ecuación (3)**:

$$
\mathrm{SNR} = \frac{P_s}{P_n} \tag{3}
$$


donde Ps es la potencia de la señal útil y Pn es la potencia del ruido.
Se expresa en decibelos en la **ecuación (4)**: 

$$
\mathrm{SNR}_{\mathrm{dB}} = 10\log_{10}\!\left(\frac{P_s}{P_n}\right) \tag{4}
$$

Un SNR alto implica que el receptor puede distinguir adecuadamente la señal del ruido, reduciendo errores en detección. Por el contrario, un SNR bajo aumenta la probabilidad de error y limita el ancho de banda útil del canal.

El SNR está directamente relacionado con:

- La distancia de transmisión.
- El tipo de canal.
- La potencia transmitida.
- Las propiedades del ruido generado en el sistema.

**Tasa de error de bit (BER)**

Es la razón entre el número de bits erróneos y el número total de bits recibidos o visto de otra forma, la probabilidad de que un bit recibido sea incorrecto


BER= número de bits erróneos / total de bits recibidos 


**La calidad de la comunicación depende del equilibrio entre:**

- El canal.
- Las fuentes de ruido.
- Las métricas de desempeño.
- La capacidad del receptor para filtrar y detectar señales distorsionadas.

"""

MATERIALES_TEXT = """
Para desarrollar las actividades de esta guía interactiva se recomienda contar con:

- Dispositivo con acceso a internet
"""

CONCLUSIONES_TEXT = """ - El estudio de los sistemas de telecomunicaciones demuestra que la calidad de transmisión depende fundamentalmente de la interacción entre el canal, las fuentes de ruido y los efectos derivados de la no linealidad de los dispositivos. La guía permitió analizar y simular cómo el ruido AWGN, la atenuación del canal y la intermodulación alteran la forma de onda original y afectan directamente la capacidad del receptor para recuperar la información enviada, destacando la importancia del SNR como parámetro clave en la detección confiable de señales digitales.

- A través de los ejemplos prácticos incluidos, el estudiante pudo visualizar de forma gráfica y cuantitativa tanto el impacto del ruido aditivo como la generación de productos de intermodulación en sistemas multiseñal. La comparación entre distintos canales guiados e inalámbricos evidenció que cada medio introduce degradaciones particulares, por lo que el diseño de sistemas modernos exige considerar modelos precisos de ruido, características físicas del canal y técnicas de mitigación orientadas a preservar la integridad de la información transmitida.
"""


# =========================
# UI HELPERS
# =========================
def add_uca_logo_to_ui():
    """Logo UCA en esquina inferior izquierda."""
    if not os.path.exists(LOGO_UCA_PATH):
        return
    with open(LOGO_UCA_PATH, "rb") as f:
        img_bytes = f.read()
    encoded = base64.b64encode(img_bytes).decode("utf-8")

    html = f"""
    <style>
    .uca-logo-fixed {{
        position: fixed;
        right: 10px;
        top: 40px;
        width: 70px;
        opacity: 0.4;
        z-index: 100;
        pointer-events: none;
    }}
    </style>
    <img class="uca-logo-fixed" src="data:image/png;base64,{encoded}">
    """
    st.markdown(html, unsafe_allow_html=True)


def apply_theme(theme_name: str):
    """
    Tema visual. OJO: CSS sin llaves extra (tu versión tenía '}}' y podía romper estilos).
    """
    t = theme_name.lower()
    st.session_state["ui_theme"] = t

    if t == "obscuro":
        bg = "#0f1113"
        fg = "#ffffff"
        panel = "#2b2f36"
        button_bg = "#2b2f36"
        button_fg = "#ffffff"
        fig_face = "#ffffff"
        ax_face = "#ffffff"
        ax_text = "#000000"
        input_bg = "#1f2227"
        input_fg = "#ffffff"
        select_bg = "#2b2f36"
        select_fg = "#ffffff"
        icon_bg = "#2b2f36"
        icon_fg = "#ffffff"
    elif t == "rosa":
        bg = "#fff6fb"
        fg = "#330033"
        panel = "#ffe6f0"
        button_bg = "#ffd6eb"
        button_fg = "#330033"
        fig_face = "#ffffff"
        ax_face = "#ffffff"
        ax_text = "#330033"
        input_bg = "#ffffff"
        input_fg = "#000000"
        select_bg = "#ffffff"
        select_fg = "#000000"
        icon_bg = "#f0f0f0"
        icon_fg = "#000000"
    else:
        bg = "#ffffff"
        fg = "#000000"
        panel = "#f9f9f9"
        button_bg = "#e0e0e0"
        button_fg = "#000000"
        fig_face = "#ffffff"
        ax_face = "#ffffff"
        ax_text = "#000000"
        input_bg = "#ffffff"
        input_fg = "#000000"
        select_bg = "#ffffff"
        select_fg = "#000000"
        icon_bg = "#f0f0f0"
        icon_fg = "#000000"

    css = f"""
    <style>
    .stApp {{
        background-color: {bg};
        color: {fg};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {panel};
        color: {fg};
    }}
    section[data-testid="stSidebar"] * {{
        color: {fg};
    }}

    .stMarkdown, .stText, .stSelectbox, .stNumberInput, .stTextInput, .stSlider, .stTabs, .stExpander {{
        color: {fg};
    }}

    .stTabs [role="tab"], .stTabs [role="tab"] * {{
        color: {fg} !important;
    }}

    /* Expanders */
    div[data-testid="stExpander"] > details {{
        background-color: {panel};
        border-radius: 8px;
        border: 1px solid rgba(0,0,0,0.15);
        padding: 0.25rem 0.75rem;
    }}
    div[data-testid="stExpander"] > details > summary {{
        background-color: {panel} !important;
        color: {fg} !important;
        border-radius: 8px !important;
        padding: 0.25rem 0.5rem !important;
    }}
    div[data-testid="stExpander"] > details > summary * {{
        color: {fg} !important;
    }}

    label, .stRadio label, .stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label {{
        color: {fg} !important;
    }}

    div[role="radiogroup"], div[role="radiogroup"] * {{
        color: {fg} !important;
    }}

    div[data-baseweb="radio"] svg, div[data-baseweb="checkbox"] svg {{
        fill: {fg} !important;
        stroke: {fg} !important;
    }}

    /* Botones */
    button:not([kind="icon"]) {{
        background-color: {button_bg} !important;
        color: {button_fg} !important;
        font-size: 0.8rem;
        padding: 0.35rem 0.75rem;
        border-radius: 4px;
        border: 1px solid #999999;
    }}
    button:not([kind="icon"]):hover {{
        filter: brightness(0.95);
    }}

    input[type="text"],
    input[type="number"],
    textarea {{
        background-color: {input_bg} !important;
        color: {input_fg} !important;
        border-radius: 4px !important;
        border: 1px solid #999999 !important;
    }}

    div[data-baseweb="select"] > div {{
        background-color: {select_bg} !important;
        color: {select_fg} !important;
        border-radius: 4px !important;
        border: 1px solid #999999 !important;
    }}
    div[data-baseweb="select"] span {{
        color: {select_fg} !important;
    }}
    div[data-baseweb="select"] div[role="listbox"],
    div[data-baseweb="select"] ul,
    div[data-baseweb="select"] li {{
        background-color: {select_bg} !important;
        color: {select_fg} !important;
    }}

    .stNumberInput button {{
        background-color: {icon_bg} !important;
        color: {icon_fg} !important;
        border-radius: 3px !important;
        border: 1px solid #999999 !important;
    }}

    button[kind="icon"],
    button[aria-label*="full"],
    button[title*="full"] {{
        background-color: {icon_bg} !important;
        color: {icon_fg} !important;
        border-radius: 3px !important;
        border: 1px solid #999999 !important;
    }}

    /* labels visibles */
    div[data-testid="stWidgetLabel"] > label,
    div[data-testid="stWidgetLabel"] > label p,
    label,
    label p {{
        color: {fg} !important;
        opacity: 1 !important;
    }}
    [data-baseweb="form-control-label"],
    [data-baseweb="form-control-label"] * {{
        color: {fg} !important;
        opacity: 1 !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    mpl.rcParams["figure.facecolor"] = fig_face
    mpl.rcParams["axes.facecolor"] = ax_face
    mpl.rcParams["axes.edgecolor"] = ax_text
    mpl.rcParams["axes.labelcolor"] = ax_text
    mpl.rcParams["xtick.color"] = ax_text
    mpl.rcParams["ytick.color"] = ax_text
    mpl.rcParams["text.color"] = ax_text
    mpl.rcParams["axes.titlecolor"] = ax_text


# =========================
# UTILIDADES NUMÉRICAS
# =========================
def generar_tren_nrz(bits, fs, Tb, level0=0.0, level1=1.0):
    samples_per_bit = int(max(1, round(Tb * fs)))
    t = np.arange(0, len(bits) * Tb, 1.0 / fs)
    if t.size == 0:
        t = np.array([0.0])
    tx = np.zeros_like(t)
    for i, b in enumerate(bits):
        s = i * samples_per_bit
        tx[s:s + samples_per_bit] = level1 if b == 1 else level0
    return t, tx


def generar_ruido_awgn(signal, SNR_dB):
    sigp = np.mean(signal ** 2) if signal.size > 0 else 1e-12
    SNR_lin = 10 ** (SNR_dB / 10.0)
    noise_p = sigp / SNR_lin if SNR_lin > 0 else sigp
    noise = np.sqrt(noise_p) * np.random.randn(*signal.shape)
    return noise


def aplicar_filtro_butter(tx, fs, cutoff_hz):
    if not SCIPY_AVAILABLE or cutoff_hz <= 0:
        return tx
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    if normal_cutoff >= 1.0:
        return tx
    b, a = butter(4, normal_cutoff, btype="low", analog=False)
    y = lfilter(b, a, tx)
    return y


def calcular_BER(bits_tx, decisions):
    L = min(len(bits_tx), len(decisions))
    if L == 0:
        return 0, 0, 0.0
    errors = np.sum(bits_tx[:L] != decisions[:L])
    ber = errors / float(L)
    return errors, L, ber


def regenerador_muestreo(received, bits_len, fs, Tb, delay_samples, threshold):
    samples_per_bit = int(max(1, round(Tb * fs)))
    sampling_times = (np.arange(bits_len) + 0.5) * Tb
    sampling_idxs = (sampling_times * fs).astype(int) + delay_samples
    sampling_idxs = sampling_idxs[sampling_idxs < received.size]
    decisions = np.array([1 if received[idx] >= threshold else 0 for idx in sampling_idxs])
    return decisions


# =========================
# CLAVES ALEATORIAS DINÁMICAS
# =========================
def generate_dyn1_key():
    snr = round(random.uniform(-5.0, 40.0), 2)
    delay = round(random.uniform(0.0, 0.5), 2)
    if snr < 5.0:
        cat = "Baja"
        ber_cat = "Alta"
    elif snr < 12.0:
        cat = "Media"
        ber_cat = "Moderada"
    else:
        cat = "Alta"
        ber_cat = "Baja"
    return {"snr": snr, "delay": delay, "q1": cat, "q2": ber_cat, "q3": "Disminuye", "q4": "Sí"}


def generate_dyn2_key():
    f1 = random.randint(700, 1400)
    f2 = random.randint(700, 1400)
    while abs(f2 - f1) < 50:
        f2 = random.randint(700, 1400)
    A1 = round(random.uniform(0.6, 1.8), 2)
    A2 = round(random.uniform(0.6, 1.8), 2)
    k3 = round(random.uniform(0.05, 0.2), 3)
    im3_1 = 2 * f1 - f2
    im3_2 = 2 * f2 - f1
    in_band = any(850 <= v <= 950 for v in (im3_1, im3_2))
    return {
        "f1": f1,
        "f2": f2,
        "A1": A1,
        "A2": A2,
        "k3": k3,
        "im3_1": im3_1,
        "im3_2": im3_2,
        "q1": "Intermodulación",
        "q2": "Aumentan",
        "q3": "Sí" if in_band else "No",
    }


# =========================
# EXPORT PDF (GUÍA 1)

def _sanitize_filename(text: str) -> str:
    import re
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^a-z0-9_\-\.]", "", t)
    return t or "archivo"


def _safe_str(x) -> str:
    return "" if x is None else str(x).strip()


def _ensure_unicode_font() -> str:
    """Intenta usar una fuente TTF (para tildes/ñ) si está disponible."""
    if not REPORTLAB_AVAILABLE:
        return "Helvetica"
    try:
        import os
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont("DejaVuSans", font_path))
            return "DejaVuSans"
    except Exception:
        pass
    return "Helvetica"


def export_results_pdf_guia1_bytes(student_info: dict, resultados: list, logo_path: str = None):
    """Genera un PDF en memoria (bytes) con los resultados de Guía 1."""
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab no está disponible. Agrega 'reportlab' a requirements.txt")

    base_font = _ensure_unicode_font()

    # Nombre de archivo (en repo) con timestamp para evitar colisiones
    registro = _sanitize_filename(_safe_str(student_info.get("registro", "")))
    nombre = _sanitize_filename(_safe_str(student_info.get("nombre", "")))
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_filename = f"guia1_{registro}_{nombre}_{ts}.pdf"

    # ---- Calcular nota global (promedio) a partir de resultados ----
    notas = []
    for it in resultados or []:
        n = it.get("nota", None)
        try:
            if n is not None:
                notas.append(float(n))
        except Exception:
            pass
    nota_global = (sum(notas) / len(notas)) if notas else None

    buf = BytesIO()
    c = rcanvas.Canvas(buf, pagesize=letter)
    width, height = letter

    # Helper: nueva página
    y = height - 50
    def new_page():
        nonlocal y
        c.showPage()
        y = height - 50
        c.setFont(base_font, 11)

    # Helper: imprimir key/value con salto de página
    def draw_kv(x, k, v):
        nonlocal y
        if y < 90:
            new_page()
        c.drawString(x, y, f"- {_safe_str(k)}: {_safe_str(v)}")
        y -= 14

    # Encabezado
    c.setFont(base_font, 16)
    c.drawString(50, y, "Guía 1 - Resultados de dinámicas")
    y -= 22

    # Logo (opcional)
    if logo_path:
        try:
            img = ImageReader(logo_path)
            c.drawImage(img, width - 140, height - 85, width=80, height=80, mask="auto")
        except Exception:
            pass

    # Datos del estudiante
    c.setFont(base_font, 11)
    c.drawString(50, y, f"Nombre: {_safe_str(student_info.get('nombre', ''))}")
    y -= 16
    c.drawString(50, y, f"Registro: {_safe_str(student_info.get('registro', ''))}")
    y -= 16

    dob = student_info.get("dob", "") or student_info.get("fecha_nacimiento", "")
    if dob:
        c.drawString(50, y, f"Fecha de nacimiento: {_safe_str(dob)}")
        y -= 16

    c.drawString(50, y, f"Fecha: {ts.replace('_', ' ')}")
    y -= 18

    # Nota global
    if nota_global is not None:
        c.setFont(base_font, 12)
        c.drawString(50, y, f"Nota global (promedio): {nota_global:.2f} / 10")
        y -= 18
        c.setFont(base_font, 11)

    y -= 6

    # Cuerpo: resultados
    for item in (resultados or []):
        if y < 120:
            new_page()

        titulo = _safe_str(item.get("titulo", "Dinámica"))
        ok = item.get("correctas", 0)
        total = item.get("total", 0)
        nota = item.get("nota", None)

        # Título + aciertos + nota si existe
        c.setFont(base_font, 13)
        line = f"{titulo}  ({ok}/{total})"
        if nota is not None:
            try:
                line += f"   Nota: {float(nota):.2f}/10"
            except Exception:
                line += f"   Nota: {_safe_str(nota)}"
        c.drawString(50, y, line)
        y -= 18

        # Parámetros / clave
        c.setFont(base_font, 10)
        c.drawString(60, y, "Parámetros / clave:")
        y -= 14
        for k, v in (item.get("key", {}) or {}).items():
            draw_kv(70, k, v)

        y -= 6
        if y < 100:
            new_page()

        # Respuestas del estudiante
        c.drawString(60, y, "Respuestas del estudiante:")
        y -= 14
        for k, v in (item.get("answers", {}) or {}).items():
            draw_kv(70, k, v)

        y -= 10

    # Pie (tema)
    try:
        c.setFont(base_font, 9)
        c.drawCentredString(width / 2.0, 25, _safe_str(TEMA_TG))
    except Exception:
        pass

    c.save()
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes, pdf_filename
# =========================
def render_ejemplo1():
    """
    Ejemplo 1:
    - Gráfica 1: señal digital NRZ.
    - Gráfica 2: ruido AWGN.
    - Gráfica 3: señal + ruido (con posible retardo).
    Parámetros a modificar:
      - Número de bits
      - Retardo debido al canal (fracción de T)
      - Nivel lógico 0
      - Nivel lógico 1
      - Período T del bit (s)
    """



    # Estado de zoom para el Ejemplo 1
    if "ej1_zoom" not in st.session_state:
        st.session_state.ej1_zoom = 1.0  # 1.0 = ver toda la señal

    st.markdown("### Ejemplo 1 - Señal digital y ruido AWGN")

    if "ej1_state" not in st.session_state:
        st.session_state.ej1_state = {
            "bits": np.array([]),
            "t": np.array([]),
            "tx": np.array([]),
            "noise": np.array([]),
            "rx": np.array([]),
            "fs": 2000,
            "Tb": 1.0,
        }
    state = st.session_state.ej1_state

    with st.expander("Descripcion y pasos", expanded=True):
        st.markdown(
            "**Pasos sugeridos**\n"
            "1. Selecciona el **número de bits** y los **niveles lógicos** de la señal.\n"
            "2. Ajusta la **SNR** y el **retardo T** en el canal\n"
            "3. Ajusta el **periodo T2** para cambiar el periodo de la señal de entrada.\n"
            "4. Pulsa **Generar señal** para crear el tren de pulsos.\n"
            "5. Pulsa **Generar ruido** para crear el ruido AWGN\n"
            "6. Pulsa **Combinar** para sumar señal y ruido y verifica la retroalimentación. Tambien puedes hacer zoom y verificar partes especificas de la señal\n"
            "7. Finalmente, pulsa **Calcular BER** para estimar la fracción de bits erróneos y verifica la retroalimentación."
        )

    col1, col2 = st.columns(2)

    with col1:
        nbits = st.number_input("Número de bits", min_value=1, max_value=50000, value=50, step=100)
        lvl0 = st.number_input("Nivel lógico 0", value=0.0)
        lvl1 = st.number_input("Nivel lógico 1", value=1.0)

        snr = st.slider("SNR (dB)", min_value=-5.0, max_value=40.0, value=12.0, step=0.5)
        delay_frac = st.slider("Retardo del canal T (s)", min_value=0.0, max_value=0.9, value=0.25, step=0.01)
        T = st.number_input("Período T2 de bit (s)", min_value=1e-4, max_value=1.0, value=1e-2, format="%.4f")

        b1, b2, b3, b4 = st.columns(4)
        gen_signal_clicked = b1.button("Generar señal")
        gen_noise_clicked = b2.button("Generar ruido")
        combine_clicked = b3.button("Combinar")
        ber_clicked = b4.button("Calcular BER")

    # Lógica de botones
    if gen_signal_clicked:
        bits = np.random.randint(0, 2, size=int(nbits))
        fs = 2000
        Tb = T
        t, tx = generar_tren_nrz(bits, fs, Tb, level0=lvl0, level1=lvl1)
        state["bits"] = bits
        state["t"] = t
        state["tx"] = tx
        state["noise"] = np.zeros_like(tx)
        state["rx"] = np.zeros_like(tx)
        state["fs"] = fs
        state["Tb"] = Tb
        st.info("Señal generada correctamente.")

    if gen_noise_clicked:
        if state["tx"].size == 0:
            st.warning("Primero genera la señal.")
        else:
            noise = generar_ruido_awgn(state["tx"], snr)
            state["noise"] = noise
            st.info("Ruido AWGN generado.")

    if combine_clicked:
        if state["tx"].size == 0:
            st.warning("Primero genera la señal.")
        else:
            # Si no hay ruido previo, se genera con la SNR actual
            if state["noise"].size == 0 or not np.any(state["noise"]):
                noise = generar_ruido_awgn(state["tx"], snr)
                state["noise"] = noise
            else:
                noise = state["noise"]

            rx = state["tx"] + noise
            delay_samples = int(round(delay_frac * state["Tb"] * state["fs"]))
            if delay_samples > 0:
                rx_del = np.concatenate((np.zeros(delay_samples), rx))[:rx.size]
            else:
                rx_del = rx
            state["rx"] = rx_del
            st.success("Señal + ruido combinados correctamente.")

            with st.expander("**Explicación de la simulación y preguntas**", expanded=True):
                st.markdown(
                    "Al sumar la señal digital con el ruido AWGN se obtiene una forma de onda en la que los niveles lógicos "
                    "dejaron de ser perfectamente planos y aparecen fluctuaciones aleatorias a su alrededor . "
                    "El retardo T aplicado  desplaza la señal recibida en el tiempo, emulando el efecto de un "
                    "canal con tiempo de propagación no nulo.\n"
                )
                st.markdown(
                    "**Preguntas y respuestas:**\n\n"
                    "1. **¿Qué ocurre con la forma de la señal cuando la SNR disminuye?**  \n"
                    "   **R:** La señal útil se ve más inmersa en el ruido, los niveles lógicos se vuelven menos distinguibles "
                    "y aumenta la probabilidad de confundir un 0 con un 1.\n\n"
                    "2. **¿Qué representa el retardo aplicado a la señal recibida?**  \n"
                    "   **R:** Representa el efecto del canal sobre el tiempo de llegada de la señal, asociado a la propagación "
                    "en el medio, dispersión o retardos introducidos por diferentes dispositivos o filtros.\n\n"
                    "3. **¿Por qué es razonable modelar el ruido como AWGN en este tipo de simulaciones?**  \n"
                    "   **R:** Porque el ruido térmico y muchas perturbaciones pequeñas e independientes pueden modelarse con "
                    "una distribución gaussiana y espectro aproximadamente plano en el ancho de banda de interés."
                )

    if ber_clicked:
        if state["bits"].size == 0:
            st.warning("Genera primero la señal.")
        else:
            # Se regenera ruido y señal+ruido con la SNR y retardo actuales
            noise = generar_ruido_awgn(state["tx"], snr)
            state["noise"] = noise
            rx = state["tx"] + noise
            delay_samples = int(round(delay_frac * state["Tb"] * state["fs"]))
            if delay_samples > 0:
                rx_del = np.concatenate((np.zeros(delay_samples), rx))[:rx.size]
            else:
                rx_del = rx
            state["rx"] = rx_del

            thr = (lvl0 + lvl1) / 2.0
            decisions = regenerador_muestreo(
                state["rx"],
                len(state["bits"]),
                state["fs"],
                state["Tb"],
                delay_samples,
                thr
            )
            errors, L, ber = calcular_BER(state["bits"], decisions)
            st.success(f"Bits comparados: {L} | Errores: {errors} | BER = {ber:.2e}")

            with st.expander("Explicación de la simulación y preguntas (BER)", expanded=True):
                st.markdown(
                    "La BER (Bit Error Rate) se calcula comparando bit a bit la secuencia transmitida con la secuencia "
                    "detectada en el receptor. Un valor de BER cercano a cero indica que la mayoría de los bits se recuperan "
                    "correctamente, un valor alto implica un sistema fuertemente degradado por el ruido.\n"
                )
                st.markdown(
                    "**Preguntas y respuestas:**\n\n"
                    "1. **¿Qué indica una BER pequeña (por ejemplo 10⁻⁶)?**  \n"
                    "   **R:** Indica que solo una fracción pequeña de los bits se detecta de forma errónea, dependiendo del sistema "
                    " y las regulaciones, puede o no representar un desempeño adecuado para servicios digitales.\n\n"
                    "2. **¿Cómo afecta el SNR a la BER en un sistema digital?**  \n"
                    "   **R:** A mayor SNR, la señal domina sobre el ruido y disminuye la probabilidad de que las muestras crucen "
                    "el umbral por error, reduciendo la BER.\n\n"

                )

    # Gráficas
    state = st.session_state.ej1_state
    with col2:
        t = state["t"]
        tx = state["tx"]
        noise = state["noise"]
        rx = state["rx"]
        plot_theme = _get_plot_theme()
        blue = "blue"

        def _build_ej1_figure(title, x_data, y_data, show_rangeslider=False):
            fig = go.Figure()
            if x_data.size > 0 and y_data.size > 0:
                fig.add_trace(
                    go.Scatter(x=x_data, y=y_data, mode="lines", line=dict(color=blue))
                )
            fig.update_layout(
                title=title,
                height=240,
                margin=dict(l=40, r=20, t=50, b=40),
                hovermode="x",
                showlegend=False,
            )
            fig.update_xaxes(title_text="Tiempo (s)", rangeslider_visible=show_rangeslider)
            fig.update_yaxes(title_text="Amplitud")
            _apply_plot_theme(fig, plot_theme, font_size=12)
            fig.update_layout(title_font=dict(color=plot_theme["font_color"], size=13))
            return fig

        fig_signal = _build_ej1_figure("Señal original", t, tx)
        fig_noise = _build_ej1_figure("Ruido AWGN", t, noise)
        fig_rx = _build_ej1_figure("Señal + ruido", t, rx, show_rangeslider=True)

        st.plotly_chart(fig_signal, use_container_width=True, theme=None)
        st.plotly_chart(fig_noise, use_container_width=True, theme=None)
        st.plotly_chart(fig_rx, use_container_width=True, theme=None)


def render_ejemplo2():
    """
    Ejemplo 2 — Distorsión por intermodulación (IMD) de tercer orden.
    """
    st.markdown("### Ejemplo 2 - Distorsión por intermodulación")

    # --- Descripción y pasos ---
    with st.expander("Descripción y pasos", expanded=True):
        st.markdown(
            "En este ejemplo se simula un dispositivo no lineal de **tercer orden** que recibe "
            "la suma de dos tonos senoidales de frecuencias f1 y f2. \n\n"
            
            "En particular, para una no linealidad cúbica aparecen productos de tercer orden:\n"
            "- 2f1 - f2\n"
            "- 2f2 - f1\n"
            "- 3f1, 3f2\n"
            "- 2f1 + f2, 2f2 + f1\n\n"
            "**Pasos sugeridos:**\n"
            "1. Ajusta las frecuencias **f1** y **f2** y sus amplitudes **A1** y **A2**.\n"
            "2. Ajusta el coeficiente de no linealidad **k3**.\n"
            "3. Pulsa **Generar y aplicar no linealidad**.\n"
            "4. Observa el espectro antes y después de la no linealidad y la forma de onda en el tiempo de un canal adyacente afectado por la intermodulacion.\n"
            "5. Verificar la retroalimentación de la simulación"

        )

    # --- Parámetros ---
    col1, col2 = st.columns([1, 1])

    with col1:
        f1 = st.number_input("Frecuencia f₁ (Hz)", value=900.0, step=10.0, min_value=1.0, key="g1_ej2_f1")
        f2 = st.number_input("Frecuencia f₂ (Hz)", value=1100.0, step=10.0, min_value=1.0, key="g1_ej2_f2")
        A1 = st.number_input("Amplitud A₁", value=1.0, step=0.1, key="g1_ej2_A1")
        A2 = st.number_input("Amplitud A₂", value=1.0, step=0.1, key="g1_ej2_A2")
        k3 = st.slider(
            "Coeficiente de no linealidad k₃",
            min_value=0.0,
            max_value=0.3,
            value=0.05,
            step=0.01,
            key="g1_ej2_k3"
        )
        run = st.button("Generar y aplicar no linealidad", key="g1_ej2_run")


    if run:
        # --- Señales en el tiempo ---
        fs = 32000  # frecuencia de muestreo
        T = 0.03    # duración de la simulación (s)
        t = np.arange(0, T, 1.0 / fs)

        # Componentes de entrada
        x1 = A1 * np.cos(2 * np.pi * f1 * t)
        x2 = A2 * np.cos(2 * np.pi * f2 * t)
        x_in = x1 + x2

        # Dispositivo no lineal cúbico: y = x + k3 x^3
        x_out = x_in + k3 * (x_in ** 3)

        # --- Espectros ---
        N = len(t)
        freq = np.fft.rfftfreq(N, 1.0 / fs)
        X_in = np.abs(np.fft.rfft(x_in)) / N
        X_out = np.abs(np.fft.rfft(x_out)) / N

        fmax_plot = max(f1, f2) * 4.0  # rango básico suficiente para ver IM3 en la mayoría de casos

        # --- Frecuencias de IMD de tercer orden ---
        imd_freqs = {
            "2f₁−f₂": 2 * f1 - f2,
            "2f₂−f₁": 2 * f2 - f1,
            "3f₁": 3 * f1,
            "3f₂": 3 * f2,
            "2f₁+f₂": 2 * f1 + f2,
            "2f₂+f₁": 2 * f2 + f1,
        }

        def _label_y(value, y_min, y_max):
            proposed = value * 1.6
            return min(max(proposed, y_min * 1.2), y_max * 0.9)

        mask = freq <= fmax_plot
        freq_plot = freq[mask]
        X_in_plot = X_in[mask] + 1e-12
        X_out_plot = X_out[mask] + 1e-12

        # 1) Espectro ANTES de la no linealidad (Plotly interactivo)
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=freq_plot,
                y=X_in_plot,
                mode="lines",
                name="Entrada",
                line=dict(color="blue"),
            )
        )
        y_min_in = float(np.min(X_in_plot))
        y_max_in = float(np.max(X_in_plot))

        plot_theme = _get_plot_theme()
        for f_c, label in [(f1, "f₁"), (f2, "f₂")]:
            if 0 < f_c < freq_plot[-1]:
                idx = np.argmin(np.abs(freq_plot - f_c))
                amp = X_in_plot[idx]
                fig1.add_annotation(
                    x=freq_plot[idx],
                    y=_label_y(amp, y_min_in, y_max_in),
                    text=label,
                    showarrow=False,
                    font=dict(size=11, color=plot_theme["font_color"]),
                    textangle=90,
                )

        fig1.update_layout(
            title="Espectro antes de la no linealidad",
            xaxis_title="Frecuencia (Hz)",
            yaxis_title="Magnitud (u.a.)",
            yaxis_type="log",
            height=320,
            margin=dict(l=70, r=20, t=50, b=40),
            hovermode="x unified",
        )
        fig1.update_yaxes(title_standoff=12)
        _apply_plot_theme(fig1, plot_theme, font_size=12)
        fig1.update_xaxes(range=[0, fmax_plot])
        st.plotly_chart(fig1, use_container_width=True, theme=None)

        # 2) Espectro DESPUÉS de la no linealidad (Plotly interactivo)
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=freq_plot,
                y=X_out_plot,
                mode="lines",
                name="Salida",
                line=dict(color="blue"),
            )
        )
        y_min_out = float(np.min(X_out_plot))
        y_max_out = float(np.max(X_out_plot))

        for f_c, label in [(f1, "f₁"), (f2, "f₂")]:
            if 0 < f_c < freq_plot[-1]:
                idx = np.argmin(np.abs(freq_plot - f_c))
                amp = X_out_plot[idx]
                fig2.add_annotation(
                    x=freq_plot[idx],
                    y=_label_y(amp, y_min_out, y_max_out),
                    text=label,
                    showarrow=False,
                    font=dict(size=11, color=plot_theme["font_color"]),
                    textangle=90,
                )

        for label, f_imd in imd_freqs.items():
            if 0 < f_imd < freq_plot[-1]:
                idx = np.argmin(np.abs(freq_plot - f_imd))
                amp = X_out_plot[idx]
                fig2.add_annotation(
                    x=freq_plot[idx],
                    y=_label_y(amp, y_min_out, y_max_out),
                    text=label,
                    showarrow=False,
                    font=dict(size=11, color=plot_theme["font_color"]),
                    textangle=90,
                )

        fig2.update_layout(
            title="Espectro después de la no linealidad",
            xaxis_title="Frecuencia (Hz)",
            yaxis_title="Magnitud (u.a.)",
            yaxis_type="log",
            height=320,
            margin=dict(l=70, r=20, t=50, b=40),
            hovermode="x unified",
        )
        fig2.update_yaxes(title_standoff=12)
        _apply_plot_theme(fig2, plot_theme, font_size=12)
        fig2.update_xaxes(range=[0, fmax_plot])
        st.plotly_chart(fig2, use_container_width=True, theme=None)

        # 3) Canal afectado por intermodulación (antes y después)
        # Elegimos como "frecuencia de canal" uno de los productos IMD (por ejemplo 2f₁−f₂)
        f_canal = imd_freqs["2f₁−f₂"]
        if f_canal <= 0 or f_canal >= fs / 2:
            # Si cae fuera de banda visible, usamos una frecuencia intermedia
            f_canal = 0.5 * (f1 + f2)

        A_canal = 1.0
        canal_limpio = A_canal * np.cos(2 * np.pi * f_canal * t)

        # Interferencia aproximada debida a todos los productos IMD dentro de banda
        interferencia = np.zeros_like(t)
        for f_imd in imd_freqs.values():
            if 0 < f_imd < fs / 2:
                interferencia += 0.3 * np.cos(2 * np.pi * f_imd * t)

        canal_afectado = canal_limpio + interferencia

        # Ventana corta (por ejemplo 5 ms) para apreciar la distorsión
        N_win = int(0.005 * fs)
        t_win = t[:N_win]
        y_clean_win = canal_limpio[:N_win]
        y_dist_win = canal_afectado[:N_win]

        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=t_win,
                y=y_clean_win,
                mode="lines",
                name="Canal adyacente antes de la intermodulación",
                line=dict(color="blue"),
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=t_win,
                y=y_dist_win,
                mode="lines",
                name="Canal adyacente después de la intermodulación",
                line=dict(color="orange"),
            )
        )
        fig3.update_layout(
            title="Señal de un canal afectado por productos de intermodulación",
            xaxis_title="Tiempo (s)",
            yaxis_title="Amplitud",
            height=320,
            margin=dict(l=40, r=20, t=50, b=40),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=0.88,
                xanchor="right",
                x=1,
                font=dict(color=plot_theme["font_color"]),
            ),
        )
        _apply_plot_theme(fig3, plot_theme, font_size=12)
        st.plotly_chart(fig3, use_container_width=True, theme=None)

        # --- Explicación y preguntas ---
        with st.expander("**Explicación de la simulación y preguntas**", expanded=True):
            st.markdown(
                "En la entrada, el espectro muestra solo dos líneas espectrales principales en f1 y f2. "
                "Tras pasar por el dispositivo no lineal cúbico, aparecen nuevos componentes de frecuencia "
                "asociados a productos de intermodulación de tercer orden.\n"
            )

            st.markdown(
                "**Preguntas y respuestas:**\n\n"
                "1. **¿Por qué aparecen nuevas líneas espectrales además de f1 y f2?**  \n"
                "   **R:** Porque la no linealidad mezcla las señales de entrada, generando sumas y diferencias "
                "de frecuencias (productos de intermodulación) además de armónicos.\n\n"
                "2. **¿Por qué los productos 2f1 - f2 y 2f2 - f1 suelen ser críticos en sistemas multicanal?**  \n"
                "   **R:** Porque pueden caer muy cerca de las portadoras útiles o dentro de canales adyacentes, "
                "provocando interferencia entre servicios que comparten el mismo medio de transmisión.\n\n"
                "3. **¿Qué representa la distorsión observada en la tercera gráfica?**  \n"
                "   **R:** Representa un canal que inicialmente transportaba una señal sinusoidal limpia. "
                "Cuando productos de intermodulación caen en esa banda, se suman a la señal útil y "
                "modifican su forma en el tiempo, introduciendo distorsión y degradando la calidad.\n\n"
                "4. **¿Qué efecto tiene aumentar k3 sobre los productos de intermodulación?**  \n"
                "**R**: A medida que aumenta k3, la contribución cúbica es mayor y los productos de intermodulación incrementan su amplitud, lo que empeora la calidad de la señal y la coexistencia de múltiples canales.\n\n"
                "5. **¿Por qué las magnitudes de los productos de intermodulación son menores a comparación de las magnitudes originales?**  \n"
                "**R**: La menor magnitud de los productos de intermodulación se debe a que estos surgen de términos no lineales de orden superior en el modelo del dispositivo. En particular, los productos IM3 están ponderados por el coeficiente de no linealidad y por potencias de las amplitudes de las señales de entrada, lo que hace que su energía sea significativamente menor que la de las componentes fundamentales. Este comportamiento es consistente con el funcionamiento real de amplificadores en la región cuasi-lineal, donde la intermodulación existe pero se mantiene limitada."
            )



def channel_attenuation_curve(channel, freqs_hz, distance_m=1000.0):
    """
    Modelo simplificado de atenuación (dB) en función de la frecuencia (Hz)
    para diferentes tipos de canal, a una cierta distancia en metros.
    """
    freqs_hz = np.asarray(freqs_hz)
    dist_km = distance_m / 1000.0
    f_GHz = freqs_hz / 1e9
    f_MHz = freqs_hz / 1e6

    if channel == "Fibra óptica":
        # Atenuación típica muy baja, casi constante con la frecuencia
        # ~0.2 dB/km + término muy pequeño dependiente de f
        att_db_per_km = 0.2 + 0.02 * f_GHz
        total_db = att_db_per_km * dist_km

    elif channel == "Coaxial":
        # Modelo simplificado: pérdidas aumentan con raíz y línea en frecuencia
        # típico de cables coaxiales de RF
        att_db_per_km = 2.0 * np.sqrt(f_MHz) + 0.02 * f_MHz
        total_db = att_db_per_km * dist_km

    elif channel == "Guía de onda":
        # Guía de onda metálica: buena para microondas, pérdidas moderadas
        att_db_per_km = 0.5 + 1.0 * f_GHz
        total_db = att_db_per_km * dist_km

    elif channel == "Par trenzado (UTP)":
        # Modelo aproximado para UTP categoría telecom:
        # pérdidas por 100 m ~ 1.8*sqrt(f_MHz) + 0.01*f_MHz  (dB/100m)
        # lo convertimos a dB/km multiplicando por 10
        att_db_per_100m = 1.8 * np.sqrt(np.clip(f_MHz, 1e-3, None)) + 0.01 * f_MHz
        att_db_per_km = att_db_per_100m * 10.0
        total_db = att_db_per_km * dist_km

    else:  # "Espacio libre"
        # Free-Space Path Loss (FSPL) en dB
        c = 3e8
        d = max(distance_m, 1.0)
        total_db = 20 * np.log10(4 * np.pi * d * freqs_hz / c + 1e-12)

    return total_db

def describe_channel(chan: str) -> str:
    if chan == "Par trenzado (UTP)":
        return "la atenuación aumenta con la frecuencia debido a pérdidas resistivas y pérdidas dieléctricas , además, se incrementa con la distancia, lo que limita el alcance útil del enlace. Por ello, a frecuencias más altas y tramos más largos, el UTP presenta mayor debilitamiento de la señal y requiere categorías superiores o técnicas de compensación"

    if chan == "Fibra óptica":
        return "presenta muy baja atenuación por km y gran inmunidad al ruido externo, ideal para enlaces de alta capacidad."
    if chan == "Coaxial":
        return "tiene una atenuación moderada que crece con la frecuencia y es susceptible a pérdidas por conductor y dieléctrico."
    if chan == "Guía de onda":
        return "opera típicamente en microondas, con bajas pérdidas pero limitada a ciertos rangos de frecuencia y modos de propagación."
    if chan == "Línea de transmisión":
        return "modela enlaces sobre pares metálicos, con pérdidas significativas y sensibilidad al ruido e interferencia."
    if chan == "Espacio libre":
        return "representa la propagación de ondas electromagnéticas sin guía física, con pérdidas que crecen con la distancia y la frecuencia de operación."
    return ""

def render_ejemplo3():
    """
    Ejemplo 3:
    - Comparar atenuación vs frecuencia para dos canales distintos.
    """
    st.markdown("### Ejemplo 3 - Comparación de canales de transmisión")

    with st.expander("Descripción y pasos", expanded=True):
        st.markdown(
            "En este ejemplo se comparan canales de propagación de ondas electromagnéticas guiadas y enlaces ópticos "
            "en función de la frecuencia y la distancia.\n\n"
            "**Pasos sugeridos**\n"
            "1. Selecciona **Canal A** y **Canal B**.\n"
            "2. Define la **distancia** del enlace en metros.\n"
            "3. Especifica el rango de **frecuencias** a analizar (en MHz).\n"
            "4. Pulsa **Simular comparación** para visualizar las curvas de atenuación.\n"
            "5. Verficar la retroalimentación"
        )

    col1, col2 = st.columns([1, 2])

    # Lista de canales (cambiamos "Línea de transmisión" por "Par trenzado (UTP)")
    canales = ["Fibra óptica", "Coaxial", "Guía de onda", "Par trenzado (UTP)", "Espacio libre"]

    with col1:
        chanA = st.selectbox("Canal A", canales, index=0)
        chanB = st.selectbox("Canal B", canales, index=1)
        dist_m = st.number_input("Distancia (m)", min_value=1.0, value=1000.0, step=100.0)

        # AHORA EN MHz
        fstart_MHz = st.number_input("Frecuencia inicio (MHz)", value=1.0)
        fend_MHz = st.number_input("Frecuencia fin (MHz)", value=1000.0)
        npts = st.number_input("Número de puntos", min_value=10, max_value=2000, value=200, step=10)
        run = st.button("Simular comparación")

    if run:
        try:
            if fstart_MHz <= 0 or fend_MHz <= 0 or fend_MHz <= fstart_MHz or npts <= 2 or dist_m <= 0:
                raise ValueError
        except Exception:
            st.warning("Verifica distancia (>0), rango de frecuencia válido y número de puntos (>2).")
            return

        # Frecuencia en MHz para el eje x (lo que verá el estudiante)
        freqs_MHz = np.logspace(math.log10(fstart_MHz), math.log10(fend_MHz), int(npts))
        # Convertimos a Hz para los modelos de atenuación (cálculo interno)
        freqs_Hz = freqs_MHz * 1e6

        yA = channel_attenuation_curve(chanA, freqs_Hz, distance_m=dist_m)
        yB = channel_attenuation_curve(chanB, freqs_Hz, distance_m=dist_m)

        with col2:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=freqs_MHz,
                    y=yA,
                    mode="lines",
                    name=chanA,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=freqs_MHz,
                    y=yB,
                    mode="lines",
                    name=chanB,
                )
            )
            plot_theme = _get_plot_theme()
            fig.update_layout(
                title="Comparación de canales",
                xaxis_title="Frecuencia (MHz)",
                yaxis_title=f"Atenuación (dB) (distancia = {dist_m} m)",
                xaxis_type="log",
                height=420,
                margin=dict(l=70, r=20, t=50, b=40),
                hovermode="x unified",
            )
            fig.update_yaxes(title_standoff=12)
            _apply_plot_theme(fig, plot_theme, font_size=12)
            st.plotly_chart(fig, use_container_width=True, theme=None)

            wavelength_nm = (3e8 / freqs_Hz) * 1e9
            order = np.argsort(wavelength_nm)
            atten_A_db_m = yA / dist_m
            atten_B_db_m = yB / dist_m

            fig_wavelength = go.Figure()
            fig_wavelength.add_trace(
                go.Scatter(
                    x=wavelength_nm[order],
                    y=atten_A_db_m[order],
                    mode="lines",
                    name=chanA,
                )
            )
            fig_wavelength.add_trace(
                go.Scatter(
                    x=wavelength_nm[order],
                    y=atten_B_db_m[order],
                    mode="lines",
                    name=chanB,
                )
            )
            fig_wavelength.update_layout(
                title="Atenuación por longitud de onda",
                xaxis_title="Longitud de onda (nm)",
                yaxis_title="Atenuación (dB/m)",
                xaxis_type="log",
                height=420,
                margin=dict(l=70, r=20, t=50, b=40),
                hovermode="x unified",
            )
            fig_wavelength.update_yaxes(title_standoff=12)
            _apply_plot_theme(fig_wavelength, plot_theme, font_size=12)
            st.plotly_chart(fig_wavelength, use_container_width=True, theme=None)

        descA = describe_channel(chanA)
        descB = describe_channel(chanB)

        with st.expander("**Explicación de la simulación y preguntas**", expanded=True):
            st.markdown(
                f"Para la distancia seleccionada ({dist_m:.0f} m), la curva de {chanA} y la de {chanB} muestran cómo la "
                "atenuación crece con la frecuencia. En canales guiados, las pérdidas dependen del material y de los "
                "mecanismos de disipación; en espacio libre, la pérdida de trayectoria se incrementa con la distancia y "
                "la frecuencia de operación.\n\n"
                f"- **{chanA}**: {descA}\n"
                f"- **{chanB}**: {descB}\n"
            )
            st.markdown(
                "**Preguntas y respuestas:**\n\n"
                "1. **¿Cuál de los canales presenta menor atenuación para la misma distancia en la mayor parte del rango de frecuencias?**  \n"
                "   **R:** Depende de la selección, pero típicamente la fibra óptica presenta menor atenuación que los medios metálicos para enlaces de larga distancia.\n\n"
                "2. **¿Cómo afecta duplicar la distancia en un enlace de espacio libre a la potencia recibida?**  \n"
                "   **R:** Idealmente, la potencia recibida se reduce en aproximadamente 6 dB (modelo de propagación 1/d²).\n\n"
                "3. **¿Por qué es importante conocer la atenuación en función de la frecuencia al diseñar un sistema?**  \n"
                "   **R:** Porque determina qué bandas de frecuencia son viables, cuánta potencia de transmisión se requiere y qué márgenes de diseño deben considerarse.\n\n"
                "4. **¿Qué ventaja ofrecen las fibras ópticas frente a cables coaxiales o par trenzado en enlaces de larga distancia?**  \n"
                "   **R:** Las fibras ópticas ofrecen menor atenuación, mayor ancho de banda y menor sensibilidad al ruido electromagnético externo."
            )

# =========================
# DINÁMICAS INTEGRADAS (GUÍA 1)
# =========================
def _score_dyn1(key, answers):
    correct_answers = {"q1": key["q1"], "q2": key["q2"], "q3": key["q3"], "q4": key["q4"]}
    correct = sum(answers[k] == correct_answers[k] for k in correct_answers)
    mapping = {4: 10.0, 3: 8.0, 2: 6.0, 1: 4.0, 0: 0.0}
    return mapping.get(correct, 0.0), correct_answers


def _score_dyn2(key2, answers):
    correct_answers = {"q1": key2["q1"], "q2": key2["q2"], "q3": key2["q3"]}
    correct = sum(answers[k] == correct_answers[k] for k in correct_answers)
    mapping = {3: 10.0, 2: 8.0, 1: 6.0, 0: 0.0}
    return mapping.get(correct, 0.0), correct_answers


def render_dinamicas_guia1():
    st.markdown("## Dinámicas – Guía 1")

    state = st.session_state.guia1_dinamicas

    # Sincronizar con student_info por compatibilidad
    if not any(state["student"].values()) and isinstance(st.session_state.student_info, dict):
        si = st.session_state.student_info
        state["student"] = {"name": si.get("name", ""), "id": si.get("id", ""), "dob": si.get("dob", "")}

    # -------- REGISTRO --------
    st.subheader("Datos del estudiante")
    with st.form("form_student_guia1"):
        name = st.text_input("Nombre completo", value=state["student"]["name"])
        sid = st.text_input("Carné", value=state["student"]["id"])
        dob = st.text_input("Fecha de nacimiento (YYYY-MM-DD)", value=state["student"]["dob"])
        ok = st.form_submit_button("Guardar datos")

    if ok:
        if not name or not sid or not dob:
            st.error("Completa todos los campos para continuar.")
        else:
            state["student"] = {"name": name, "id": sid, "dob": dob}
            st.session_state.student_info = {"name": name, "id": sid, "dob": dob}
            st.success("Datos guardados correctamente.")

    if not all(state["student"].values()):
        st.warning("Ingresa tus datos para habilitar las dinámicas.")
        return

    st.markdown("---")

    # -------- DINÁMICA 1 --------
    with st.expander("Dinámica 1 — AWGN, SNR y BER", expanded=True):
        if state["dyn1"]["key"] is None:
            state["dyn1"]["key"] = generate_dyn1_key()
        key1 = state["dyn1"]["key"]

        st.markdown(f"**Caso:** SNR = {key1['snr']} dB | Retardo = {key1['delay']}·T")

        if state["dyn1"]["sim"] is None or state["dyn1"]["sim"].get("snr") != key1["snr"]:
            bits = np.random.randint(0, 2, size=200)
            fs = 2000
            Tb = 0.01
            t, tx = generar_tren_nrz(bits, fs, Tb, level0=0.0, level1=1.0)
            noise = generar_ruido_awgn(tx, key1["snr"])
            rx = tx + noise
            delay_samples = int(round(key1["delay"] * Tb * fs))
            if delay_samples > 0:
                rx = np.concatenate((np.zeros(delay_samples), rx))[:rx.size]
            state["dyn1"]["sim"] = {"snr": key1["snr"], "t": t, "tx": tx, "noise": noise, "rx": rx}

        sim1 = state["dyn1"]["sim"]
        t = sim1["t"]; tx = sim1["tx"]; noise = sim1["noise"]; rx = sim1["rx"]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.16,
                            subplot_titles=("Señal transmitida (NRZ)", "Ruido AWGN", "Señal + ruido"))
        color = "blue"
        fig.add_trace(go.Scattergl(x=t, y=tx, mode="lines", line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scattergl(x=t, y=noise, mode="lines", line=dict(color=color)), row=2, col=1)
        fig.add_trace(go.Scattergl(x=t, y=rx, mode="lines", line=dict(color=color)), row=3, col=1)

        fig.update_xaxes(title_text="Tiempo (s)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitud", row=1, col=1)
        fig.update_yaxes(title_text="Amplitud", row=2, col=1)
        fig.update_yaxes(title_text="Amplitud", row=3, col=1)

        plot_theme = _get_plot_theme()
        fig.update_layout(
            height=750,
            margin=dict(l=40, r=20, t=90, b=60),
            hovermode="x unified",
            showlegend=False,
        )
        _apply_plot_theme(fig, plot_theme, font_size=12)
        fig.update_xaxes(rangeslider_visible=True, row=3, col=1)

        st.plotly_chart(fig, use_container_width=True, theme=None)

        q1 = st.radio("Clasificación del SNR:", ["Baja", "Media", "Alta"], index=None, key="g1_dyn1_q1")
        q2 = st.radio("Comportamiento de la BER:", ["Alta", "Moderada", "Baja"], index=None, key="g1_dyn1_q2")
        q3 = st.radio("¿Qué ocurre con la BER al aumentar el SNR?", ["Aumenta", "Disminuye"], index=None, key="g1_dyn1_q3")
        q4 = st.radio("¿El ruido AWGN es aditivo?", ["Sí", "No"], index=None, key="g1_dyn1_q4")

        state["dyn1"]["answers"] = {"q1": q1, "q2": q2, "q3": q3, "q4": q4}
        state["dyn1"]["completed"] = all(v is not None for v in state["dyn1"]["answers"].values())
        if state["dyn1"]["completed"]:
            st.success("Dinámica 1 lista.")
        else:
            st.info("Completa todas las preguntas.")


    st.markdown("---")

    # -------- DINÁMICA 2 --------
    with st.expander("Dinámica 2 — Intermodulación", expanded=True):
        if state["dyn2"]["key"] is None:
            state["dyn2"]["key"] = generate_dyn2_key()
        key2 = state["dyn2"]["key"]

        st.markdown(
            f"**Caso:** f1={key2['f1']} Hz, f2={key2['f2']} Hz | "
            f"A1={key2['A1']}, A2={key2['A2']}, k3={key2['k3']}"
        )

        tag = (key2["f1"], key2["f2"], key2["A1"], key2["A2"], key2["k3"])
        if state["dyn2"]["sim"] is None or state["dyn2"]["sim"].get("tag") != tag:
            fs2 = 16000
            T = 0.05
            t2 = np.arange(0, T, 1.0 / fs2)
            x = key2["A1"] * np.cos(2*np.pi*key2["f1"]*t2) + key2["A2"] * np.cos(2*np.pi*key2["f2"]*t2)
            y = x + key2["k3"] * (x**3)
            freq = np.fft.rfftfreq(len(t2), 1.0 / fs2)
            X_in = np.abs(np.fft.rfft(x)) / len(t2)
            X_out = np.abs(np.fft.rfft(y)) / len(t2)
            state["dyn2"]["sim"] = {"tag": tag, "freq": freq, "X_in": X_in, "X_out": X_out}

        sim2 = state["dyn2"]["sim"]
        freq = sim2["freq"]; X_in = sim2["X_in"]; X_out = sim2["X_out"]
        fmax_plot = max(key2["f1"], key2["f2"]) * 4.0

        fig2 = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            subplot_titles=("Espectro antes de la no linealidad", "Espectro después de la no linealidad"),
        )
        fig2.add_trace(go.Scattergl(x=freq, y=X_in + 1e-12, mode="lines"), row=1, col=1)
        fig2.add_trace(go.Scattergl(x=freq, y=X_out + 1e-12, mode="lines"), row=2, col=1)
        fig2.update_xaxes(range=[0, fmax_plot], row=1, col=1)
        fig2.update_xaxes(range=[0, fmax_plot], title_text="Frecuencia (Hz)", row=2, col=1)
        fig2.update_yaxes(type="log", title_text="Magnitud (u.a.)", row=1, col=1)
        fig2.update_yaxes(type="log", title_text="Magnitud (u.a.)", row=2, col=1)
        fig2.update_layout(
            height=650,
            margin=dict(l=40, r=20, t=90, b=60),
            hovermode="x unified",
            showlegend=False,
        )
        plot_theme = _get_plot_theme()
        _apply_plot_theme(fig2, plot_theme, font_size=12)

        labels = [(key2["f1"], "f1"), (key2["f2"], "f2")]
        imd_freqs = {
            "2f1-f2": 2 * key2["f1"] - key2["f2"],
            "2f2-f1": 2 * key2["f2"] - key2["f1"],
            "3f1": 3 * key2["f1"],
            "3f2": 3 * key2["f2"],
            "2f1+f2": 2 * key2["f1"] + key2["f2"],
            "2f2+f1": 2 * key2["f2"] + key2["f1"],
        }
        label_font = dict(size=12, color=plot_theme["font_color"])
        for f_c, lab in labels:
            if 0 < f_c < fmax_plot:
                idx = np.argmin(np.abs(freq - f_c))
                amp = X_out[idx] + 1e-12
                fig2.add_annotation(
                    x=f_c,
                    y=amp * 1.5,
                    text=lab,
                    showarrow=False,
                    textangle=90,
                    xref="x2",
                    yref="y2",
                    font=label_font,
                )

        for lab, f_imd in imd_freqs.items():
            if 0 < f_imd < fmax_plot:
                idx = np.argmin(np.abs(freq - f_imd))
                amp = X_out[idx] + 1e-12
                fig2.add_annotation(
                    x=f_imd,
                    y=amp * 1.5,
                    text=lab,
                    showarrow=False,
                    textangle=90,
                    xref="x2",
                    yref="y2",
                    font=label_font,
                )

        st.plotly_chart(fig2, use_container_width=True, theme=None)

        q1 = st.radio("Tipo de distorsión:", ["Armónica", "Intermodulación"], index=None, key="g1_dyn2_q1")
        q2 = st.radio("¿Qué ocurre al aumentar k3?", ["Disminuyen", "Aumentan"], index=None, key="g1_dyn2_q2")
        q3 = st.radio("¿Los productos IM3 pueden caer en banda?", ["Sí", "No"], index=None, key="g1_dyn2_q3")

        state["dyn2"]["answers"] = {"q1": q1, "q2": q2, "q3": q3}
        state["dyn2"]["completed"] = all(v is not None for v in state["dyn2"]["answers"].values())
        if state["dyn2"]["completed"]:
            st.success("Dinámica 2 lista.")
        else:
            st.info("Completa todas las preguntas.")

    st.markdown("---")
    # -------- ENVÍO FINAL --------
    disabled = (not (state["dyn1"]["completed"] and state["dyn2"]["completed"])) or state.get("submitted", False)

    if state.get("submitted", False):
        st.info("Ya enviaste estas respuestas ✅")

    if st.button("Enviar respuestas (subir a GitHub)", disabled=disabled, key="g1_send_github"):
        # Datos del estudiante
        nombre = (state.get("student", {}).get("name", "") or "").strip()
        registro = (state.get("student", {}).get("id", "") or "").strip()
        dob = (state.get("student", {}).get("dob", "") or "").strip()

        # Claves y respuestas
        dyn1_key = state["dyn1"]["key"]
        dyn2_key = state["dyn2"]["key"]
        ans1 = state["dyn1"]["answers"]
        ans2 = state["dyn2"]["answers"]

        correct1 = {"q1": dyn1_key["q1"], "q2": dyn1_key["q2"], "q3": dyn1_key["q3"], "q4": dyn1_key["q4"]}
        correct2 = {"q1": dyn2_key["q1"], "q2": dyn2_key["q2"], "q3": dyn2_key["q3"]}

        def _count_correct(ans, corr):
            return sum(1 for k, v in corr.items() if ans.get(k) == v)

        c1 = _count_correct(ans1, correct1)
        c2 = _count_correct(ans2, correct2)

        score1 = {4: 10.0, 3: 8.0, 2: 6.0, 1: 4.0, 0: 0.0}.get(c1, 0.0)
        score2 = {3: 10.0, 2: 8.0, 1: 6.0, 0: 0.0}.get(c2, 0.0)
        nota_global = round((score1 + score2) / 2.0, 2)
        state["nota_global"] = nota_global
        

        # Lista que alimenta el PDF (cada dinámica incluye su nota)
        resultados = [
            {
                "titulo": "Dinámica 1 — AWGN, SNR y BER",
                "correctas": c1,
                "total": 4,
                "nota": score1,
                "key": {"SNR (dB)": dyn1_key["snr"], "Retardo (T)": dyn1_key["delay"]},
                "answers": ans1,
            },
            {
                "titulo": "Dinámica 2 — Intermodulación",
                "correctas": c2,
                "total": 3,
                "nota": score2,
                "key": {
                    "f1 (Hz)": dyn2_key["f1"],
                    "f2 (Hz)": dyn2_key["f2"],
                    "A1": dyn2_key["A1"],
                    "A2": dyn2_key["A2"],
                    "k3": dyn2_key["k3"],
                    "IM3_1 (Hz)": dyn2_key["im3_1"],
                    "IM3_2 (Hz)": dyn2_key["im3_2"],
                },
                "answers": ans2,
            },
        ]

        # Exportar como PDF (en memoria) y subir a GitHub
        if not REPORTLAB_AVAILABLE:
            st.error(
                "No se puede generar el PDF porque 'reportlab' no está disponible. "
                "Agrega 'reportlab' a requirements.txt."
            )
        else:
            # El PDF incluye nota por dinámica y nota global (promedio)
            pdf_bytes, pdf_filename = export_results_pdf_guia1_bytes(
                student_info={"nombre": nombre, "registro": registro, "dob": dob, "nota_global": nota_global},
                resultados=resultados,
                logo_path=LOGO_UCA_PATH if LOGO_UCA_PATH else None,
            )

            repo_path = f"guia1/{pdf_filename}"
            commit_msg = f"Guía 1 - {registro} - {nombre}".strip()

            ok, info = upload_bytes_to_github_results(
                content_bytes=pdf_bytes,
                repo_path=repo_path,
                commit_message=commit_msg,
            )

            if ok:
                state["submitted"] = True
                state["nota_dyn1"] = score1
                state["nota_dyn2"] = score2
                state["nota_global"] = nota_global

                
                if isinstance(info, dict) and info.get("html_url"):
                    st.link_button("Ver archivo en GitHub", info["html_url"])
                st.write("Verifica tu nota con el catedratico o el instructor encargado")
            else:
                st.error(f"No se pudo subir el PDF: {info}")

# =========================
# GUÍA 1 (TABS)
# =========================
def render_guia1():
    st.title("Guía 1: Introducción al ruido en sistemas de telecomunicaciones")

    tabs = st.tabs(["Objetivos", "Introducción", "Materiales", "Ejemplos", "Dinámicas", "Conclusiones"])

    with tabs[0]:
        st.subheader("Objetivos")
        st.write(OBJETIVOS_TEXT)

    with tabs[1]:
        st.subheader("Introducción teórica")
        st.markdown(INTRO_FULL_TEXT)

    with tabs[2]:
        st.subheader("Materiales y equipo")
        st.write(MATERIALES_TEXT)

    with tabs[3]:
        st.subheader("Ejemplos interactivos")
        sub_tabs = st.tabs(["Ejemplo 1", "Ejemplo 2", "Ejemplo 3"])
        with sub_tabs[0]:
            render_ejemplo1()
        with sub_tabs[1]:
            render_ejemplo2()
        with sub_tabs[2]:
            render_ejemplo3()

    with tabs[4]:
        render_dinamicas_guia1()

    with tabs[5]:
        st.subheader("Conclusiones")
        st.write(CONCLUSIONES_TEXT)


# =========================
# MAIN
# =========================
def main():
    add_uca_logo_to_ui()

    st.sidebar.title("Menú principal")
    tema = st.sidebar.selectbox("Tema de la interfaz", ["Blanco", "Obscuro", "Rosa"], index=0)
    apply_theme(tema)

    guia = st.sidebar.selectbox(
        "Selecciona una guía",
        [
            "Guía 1: Introducción al ruido",
            "Guía 2: Fundamentos de señales y sistemas",
            "Guía 3: Fundamentos de probabilidad",
            "Guía 4: Procesos estocásticos y el ruido",
            "Guía 5: Fundamentos de transmisión digital en presencia de ruido",
        ],
    )

    if guia.startswith("Guía 1"):
        render_guia1()
    elif guia.startswith("Guía 2"):
        guia2.render_guia2()
    elif guia.startswith("Guía 3"):
        guia3.render_guia3()
    elif guia.startswith("Guía 4"):
        guia4.render_guia4()
    elif guia.startswith("Guía 5"):
        guia5.render_guia5()


if __name__ == "__main__":
    main()
