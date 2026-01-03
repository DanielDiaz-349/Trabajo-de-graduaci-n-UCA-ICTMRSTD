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

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import guia2
import guia3
import guia4
import guia5

from github_uploader import upload_file_to_github_results


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

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rcanvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# =========================
# RESULTS DIR
# =========================
RESULTS_DIR = "resultados_dinamicas"
os.makedirs(RESULTS_DIR, exist_ok=True)


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

**Naturaleza y clasificación del ruido**
El ruido se define como una señal aleatoria que se adhiere a la señal original e introduce incertidumbre en la detección. Puede ser correlacionado o no correlacionado.

**Ruido blanco aditivo gaussiano (AWGN)**
Es un modelo ampliamente utilizado: aditivo, blanco y gaussiano.

**Relación Señal-Ruido (SNR)**
\[
\mathrm{SNR}=\frac{P_s}{P_n}, \quad
\mathrm{SNR}_{\mathrm{dB}}=10\log_{10}\left(\frac{P_s}{P_n}\right)
\]

**BER**
Es la razón entre bits erróneos y bits totales recibidos.
"""

MATERIALES_TEXT = """
Para desarrollar las actividades de esta guía interactiva se recomienda contar con:

- Una computadora personal con sistema operativo actualizado (Windows, Linux o macOS).
- Python instalado (versión 3.8 o superior recomendada).
- Un entorno de desarrollo como Visual Studio Code o PyCharm.
- Bibliotecas:
  - numpy
  - matplotlib
  - streamlit
  - scipy (opcional)
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
        left: 10px;
        bottom: 10px;
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
# =========================
def export_results_pdf_guia1(filename_base, student_info, resultados):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{filename_base}_{ts}"
    pdf_path = os.path.join(RESULTS_DIR, base + ".pdf")

    if not REPORTLAB_AVAILABLE:
        return pdf_path

    c = rcanvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    left = 40
    top = height - 40
    line_h = 14

    # Marca de agua
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

    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, top, "Resultados Guía 1 – Dinámicas")
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

    promedio = total_score / max(len(resultados), 1)
    y -= 2 * line_h
    if y < 80:
        c.showPage()
        y = top
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left, y, f"Nota global de la guía (oculta): {promedio:.2f}")

    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(width / 2.0, 30, TEMA_TG)

    c.save()
    return pdf_path


# =========================
# EJEMPLOS (1–3)
# =========================
def render_ejemplo1():
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
            "1. Selecciona el número de bits y niveles lógicos.\n"
            "2. Ajusta SNR y retardo.\n"
            "3. Ajusta el período del bit.\n"
            "4. Genera señal, ruido, combina y luego calcula BER."
        )

    col1, col2 = st.columns(2)
    with col1:
        nbits = st.number_input("Número de bits", min_value=1, max_value=50000, value=50, step=50)
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
            if state["noise"].size == 0 or not np.any(state["noise"]):
                state["noise"] = generar_ruido_awgn(state["tx"], snr)
            rx = state["tx"] + state["noise"]
            delay_samples = int(round(delay_frac * state["Tb"] * state["fs"]))
            if delay_samples > 0:
                rx = np.concatenate((np.zeros(delay_samples), rx))[:rx.size]
            state["rx"] = rx
            st.success("Señal + ruido combinados correctamente.")

    if ber_clicked:
        if state["bits"].size == 0:
            st.warning("Genera primero la señal.")
        else:
            state["noise"] = generar_ruido_awgn(state["tx"], snr)
            rx = state["tx"] + state["noise"]
            delay_samples = int(round(delay_frac * state["Tb"] * state["fs"]))
            if delay_samples > 0:
                rx = np.concatenate((np.zeros(delay_samples), rx))[:rx.size]
            state["rx"] = rx

            thr = (lvl0 + lvl1) / 2.0
            decisions = regenerador_muestreo(state["rx"], len(state["bits"]), state["fs"], state["Tb"], delay_samples, thr)
            errors, L, ber = calcular_BER(state["bits"], decisions)
            st.success(f"Bits comparados: {L} | Errores: {errors} | BER = {ber:.2e}")

    with col2:
        t = state["t"]
        tx = state["tx"]
        noise = state["noise"]
        rx = state["rx"]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.16,
                            subplot_titles=("Señal original", "Ruido AWGN", "Señal + ruido"))

        blue = "blue"
        if t.size and tx.size:
            fig.add_trace(go.Scatter(x=t, y=tx, mode="lines", line=dict(color=blue)), row=1, col=1)
        if t.size and noise.size:
            fig.add_trace(go.Scatter(x=t, y=noise, mode="lines", line=dict(color=blue)), row=2, col=1)
        if t.size and rx.size:
            fig.add_trace(go.Scatter(x=t, y=rx, mode="lines", line=dict(color=blue)), row=3, col=1)

        fig.update_xaxes(title_text="Tiempo (s)", row=3, col=1)
        fig.update_yaxes(title_text="Amplitud", row=1, col=1)
        fig.update_yaxes(title_text="Amplitud", row=2, col=1)
        fig.update_yaxes(title_text="Amplitud", row=3, col=1)

        fig.update_layout(
            height=750,
            margin=dict(l=40, r=20, t=90, b=60),
            hovermode="x unified",
            showlegend=False,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black", size=12),
            hoverlabel=dict(bgcolor="white", font=dict(color="black")),
        )

        fig.update_xaxes(showgrid=True, gridcolor="lightgray", zerolinecolor="black",
                         linecolor="black", ticks="outside", tickcolor="black",
                         tickfont=dict(color="black"), title_font=dict(color="black"))

        fig.update_yaxes(showgrid=True, gridcolor="lightgray", zerolinecolor="black",
                         linecolor="black", ticks="outside", tickcolor="black",
                         tickfont=dict(color="black"), title_font=dict(color="black"))

        fig.update_annotations(font=dict(color="black", size=13))
        fig.update_xaxes(rangeslider_visible=True, row=3, col=1)

        st.plotly_chart(fig, use_container_width=True, theme=None)


def render_ejemplo2():
    st.markdown("### Ejemplo 2 - Distorsión por intermodulación")

    with st.expander("Descripción y pasos", expanded=True):
        st.markdown(
            "Se simula una no linealidad cúbica: y = x + k3 x^3.\n"
            "Aparecen productos IM3: 2f1-f2, 2f2-f1, 3f1, 3f2, 2f1+f2, 2f2+f1."
        )

    col1, _ = st.columns([1, 1])
    with col1:
        f1 = st.number_input("Frecuencia f1 (Hz)", value=900.0, step=10.0, min_value=1.0, key="g1_ej2_f1")
        f2 = st.number_input("Frecuencia f2 (Hz)", value=1100.0, step=10.0, min_value=1.0, key="g1_ej2_f2")
        A1 = st.number_input("Amplitud A1", value=1.0, step=0.1, key="g1_ej2_A1")
        A2 = st.number_input("Amplitud A2", value=1.0, step=0.1, key="g1_ej2_A2")
        k3 = st.slider("Coeficiente k3", 0.0, 0.3, 0.05, 0.01, key="g1_ej2_k3")
        run = st.button("Generar y aplicar no linealidad", key="g1_ej2_run")

    if not run:
        return

    fs = 32000
    T = 0.03
    t = np.arange(0, T, 1.0 / fs)

    x_in = A1 * np.cos(2 * np.pi * f1 * t) + A2 * np.cos(2 * np.pi * f2 * t)
    x_out = x_in + k3 * (x_in ** 3)

    N = len(t)
    freq = np.fft.rfftfreq(N, 1.0 / fs)
    X_in = np.abs(np.fft.rfft(x_in)) / N
    X_out = np.abs(np.fft.rfft(x_out)) / N

    fmax_plot = max(f1, f2) * 4.0

    fig1, ax1 = plt.subplots(figsize=(7, 3))
    ax1.semilogy(freq, X_in + 1e-12)
    ax1.set_xlim(0, fmax_plot)
    ax1.set_xlabel("Frecuencia (Hz)")
    ax1.set_ylabel("Magnitud (u.a.)")
    ax1.set_title("Espectro antes de la no linealidad")
    ax1.grid(True, linestyle=":")
    fig1.tight_layout(pad=2.0)
    st.pyplot(fig1)

    imd_freqs = {
        "2f1-f2": 2 * f1 - f2,
        "2f2-f1": 2 * f2 - f1,
        "3f1": 3 * f1,
        "3f2": 3 * f2,
        "2f1+f2": 2 * f1 + f2,
        "2f2+f1": 2 * f2 + f1,
    }

    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.semilogy(freq, X_out + 1e-12)
    ax2.set_xlim(0, fmax_plot)
    ax2.set_xlabel("Frecuencia (Hz)")
    ax2.set_ylabel("Magnitud (u.a.)")
    ax2.set_title("Espectro después de la no linealidad")
    ax2.grid(True, linestyle=":")

    for f_c, label in [(f1, "f1"), (f2, "f2")]:
        if 0 < f_c < freq[-1]:
            idx = np.argmin(np.abs(freq - f_c))
            amp = X_out[idx] + 1e-12
            ax2.text(freq[idx], amp * 1.3, label, ha="center", va="bottom", fontsize=7, rotation=90, color="black")

    for label, f_imd in imd_freqs.items():
        if 0 < f_imd < freq[-1]:
            idx = np.argmin(np.abs(freq - f_imd))
            amp = X_out[idx] + 1e-12
            ax2.text(freq[idx], amp * 1.3, label, ha="center", va="bottom", fontsize=7, rotation=90, color="black")

    fig2.tight_layout(pad=2.0)
    st.pyplot(fig2)


def channel_attenuation_curve(channel, freqs_hz, distance_m=1000.0):
    freqs_hz = np.asarray(freqs_hz)
    dist_km = distance_m / 1000.0
    f_GHz = freqs_hz / 1e9
    f_MHz = freqs_hz / 1e6

    if channel == "Fibra óptica":
        att_db_per_km = 0.2 + 0.02 * f_GHz
        total_db = att_db_per_km * dist_km
    elif channel == "Coaxial":
        att_db_per_km = 2.0 * np.sqrt(f_MHz) + 0.02 * f_MHz
        total_db = att_db_per_km * dist_km
    elif channel == "Guía de onda":
        att_db_per_km = 0.5 + 1.0 * f_GHz
        total_db = att_db_per_km * dist_km
    elif channel == "Par trenzado (UTP)":
        att_db_per_100m = 1.8 * np.sqrt(np.clip(f_MHz, 1e-3, None)) + 0.01 * f_MHz
        att_db_per_km = att_db_per_100m * 10.0
        total_db = att_db_per_km * dist_km
    else:
        c = 3e8
        d = max(distance_m, 1.0)
        total_db = 20 * np.log10(4 * np.pi * d * freqs_hz / c + 1e-12)

    return total_db


def describe_channel(chan: str) -> str:
    if chan == "Par trenzado (UTP)":
        return ("la atenuación aumenta con la frecuencia debido a pérdidas resistivas y dieléctricas; "
                "además crece con la distancia, lo que limita el alcance útil del enlace.")
    if chan == "Fibra óptica":
        return "presenta muy baja atenuación por km y gran inmunidad al ruido externo."
    if chan == "Coaxial":
        return "tiene una atenuación moderada que crece con la frecuencia."
    if chan == "Guía de onda":
        return "opera típicamente en microondas, con bajas pérdidas en su banda útil."
    if chan == "Espacio libre":
        return "la pérdida de trayectoria crece con la distancia y la frecuencia."
    return ""


def render_ejemplo3():
    st.markdown("### Ejemplo 3 - Comparación de canales de transmisión")

    with st.expander("Descripción y pasos", expanded=True):
        st.markdown(
            "Comparación de atenuación vs frecuencia para distintos canales.\n"
            "1) Selecciona Canal A y B\n"
            "2) Elige distancia\n"
            "3) Define rango de frecuencias (MHz)\n"
            "4) Simula"
        )

    col1, col2 = st.columns([1, 2])
    canales = ["Fibra óptica", "Coaxial", "Guía de onda", "Par trenzado (UTP)", "Espacio libre"]

    with col1:
        chanA = st.selectbox("Canal A", canales, index=0)
        chanB = st.selectbox("Canal B", canales, index=1)
        dist_m = st.number_input("Distancia (m)", min_value=1.0, value=1000.0, step=100.0)
        fstart_MHz = st.number_input("Frecuencia inicio (MHz)", value=1.0)
        fend_MHz = st.number_input("Frecuencia fin (MHz)", value=1000.0)
        npts = st.number_input("Número de puntos", min_value=10, max_value=2000, value=200, step=10)
        run = st.button("Simular comparación")

    if not run:
        return

    if fstart_MHz <= 0 or fend_MHz <= 0 or fend_MHz <= fstart_MHz or npts <= 2 or dist_m <= 0:
        st.warning("Verifica distancia (>0), rango de frecuencia válido y número de puntos (>2).")
        return

    freqs_MHz = np.logspace(math.log10(fstart_MHz), math.log10(fend_MHz), int(npts))
    freqs_Hz = freqs_MHz * 1e6

    yA = channel_attenuation_curve(chanA, freqs_Hz, distance_m=dist_m)
    yB = channel_attenuation_curve(chanB, freqs_Hz, distance_m=dist_m)

    with col2:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogx(freqs_MHz, yA, label=chanA)
        ax.semilogx(freqs_MHz, yB, label=chanB)
        ax.set_xlabel("Frecuencia (MHz)")
        ax.set_ylabel(f"Atenuación (dB) (distancia = {dist_m} m)")
        ax.set_title("Comparación de canales")
        ax.legend()
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        st.pyplot(fig)

        descA = describe_channel(chanA)
        descB = describe_channel(chanB)
        with st.expander("Explicación", expanded=True):
            st.markdown(f"- **{chanA}**: {descA}\n- **{chanB}**: {descB}")


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

        fig.update_layout(
            height=750, margin=dict(l=40, r=20, t=90, b=60),
            hovermode="x unified", showlegend=False,
            paper_bgcolor="white", plot_bgcolor="white",
            font=dict(color="black"),
            hoverlabel=dict(bgcolor="white", font=dict(color="black")),
        )
        fig.update_xaxes(showgrid=True, gridcolor="lightgray", linecolor="black",
                         tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_yaxes(showgrid=True, gridcolor="lightgray", linecolor="black",
                         tickfont=dict(color="black"), title_font=dict(color="black"))
        fig.update_xaxes(rangeslider_visible=True, row=3, col=1)

        st.plotly_chart(fig, use_container_width=True, theme=None)

        q1 = st.radio("Clasificación del SNR:", ["Baja", "Media", "Alta"], index=None, key="g1_dyn1_q1")
        q2 = st.radio("Comportamiento de la BER:", ["Alta", "Moderada", "Baja"], index=None, key="g1_dyn1_q2")
        q3 = st.radio("¿Qué ocurre con la BER al aumentar el SNR?", ["Aumenta", "Disminuye"], index=None, key="g1_dyn1_q3")
        q4 = st.radio("¿El ruido AWGN es aditivo?", ["Sí", "No"], index=None, key="g1_dyn1_q4")

        state["dyn1"]["answers"] = {"q1": q1, "q2": q2, "q3": q3, "q4": q4}
        state["dyn1"]["completed"] = all(v is not None for v in state["dyn1"]["answers"].values())
        st.success("Dinámica 1 lista.") if state["dyn1"]["completed"] else st.info("Completa todas las preguntas.")

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

        fig2, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        a1.semilogy(freq, X_in + 1e-12)
        a1.set_xlim(0, fmax_plot)
        a1.set_ylabel("Magnitud (u.a.)")
        a1.set_title("Espectro antes de la no linealidad")
        a1.grid(True, which="both", linestyle=":", alpha=0.5)

        a2.semilogy(freq, X_out + 1e-12)
        a2.set_xlim(0, fmax_plot)
        a2.set_xlabel("Frecuencia (Hz)")
        a2.set_ylabel("Magnitud (u.a.)")
        a2.set_title("Espectro después de la no linealidad")
        a2.grid(True, which="both", linestyle=":", alpha=0.5)

        labels = [(key2["f1"], "f1"), (key2["f2"], "f2")]
        imd_freqs = {
            "2f1-f2": 2 * key2["f1"] - key2["f2"],
            "2f2-f1": 2 * key2["f2"] - key2["f1"],
            "3f1": 3 * key2["f1"],
            "3f2": 3 * key2["f2"],
            "2f1+f2": 2 * key2["f1"] + key2["f2"],
            "2f2+f1": 2 * key2["f2"] + key2["f1"],
        }
        for f_c, lab in labels:
            if 0 < f_c < fmax_plot:
                idx = np.argmin(np.abs(freq - f_c))
                amp = X_out[idx] + 1e-12
                a2.text(f_c, amp * 1.5, lab, ha="center", va="bottom", fontsize=8, rotation=90, color="black")

        for lab, f_imd in imd_freqs.items():
            if 0 < f_imd < fmax_plot:
                idx = np.argmin(np.abs(freq - f_imd))
                amp = X_out[idx] + 1e-12
                a2.text(f_imd, amp * 1.5, lab, ha="center", va="bottom", fontsize=8, rotation=90, color="black")

        fig2.tight_layout(pad=3.0)
        st.pyplot(fig2)

        q1 = st.radio("Tipo de distorsión:", ["Armónica", "Intermodulación"], index=None, key="g1_dyn2_q1")
        q2 = st.radio("¿Qué ocurre al aumentar k3?", ["Disminuyen", "Aumentan"], index=None, key="g1_dyn2_q2")
        q3 = st.radio("¿Los productos IM3 pueden caer en banda?", ["Sí", "No"], index=None, key="g1_dyn2_q3")

        state["dyn2"]["answers"] = {"q1": q1, "q2": q2, "q3": q3}
        state["dyn2"]["completed"] = all(v is not None for v in state["dyn2"]["answers"].values())
        st.success("Dinámica 2 lista.") if state["dyn2"]["completed"] else st.info("Completa todas las preguntas.")

    st.markdown("---")

    # -------- ENVÍO FINAL --------
    disabled = not (state["dyn1"]["completed"] and state["dyn2"]["completed"])
    if st.button("Enviar respuestas (generar PDF)", key="g1_send_pdf", disabled=disabled):
        if not REPORTLAB_AVAILABLE:
            st.error("No se puede generar el PDF porque ReportLab no está instalado.")
            st.stop()

        ans1 = state["dyn1"]["answers"]
        ans2 = state["dyn2"]["answers"]
        score1, corr1 = _score_dyn1(state["dyn1"]["key"], ans1)
        score2, corr2 = _score_dyn2(state["dyn2"]["key"], ans2)

        res1 = {
            "dyn_id": 1,
            "score": score1,
            "answers": ans1,
            "correct": corr1,
            "key": {
                "descripcion": "Guía 1 - Dinámica 1 - Ruido AWGN y BER",
                "snr_dB": state["dyn1"]["key"]["snr"],
                "delay_Tb": state["dyn1"]["key"]["delay"],
            },
        }
        res2 = {
            "dyn_id": 2,
            "score": score2,
            "answers": ans2,
            "correct": corr2,
            "key": {
                "descripcion": "Guía 1 - Dinámica 2 - Intermodulación IM3",
                "f1_Hz": state["dyn2"]["key"]["f1"],
                "f2_Hz": state["dyn2"]["key"]["f2"],
                "A1": state["dyn2"]["key"]["A1"],
                "A2": state["dyn2"]["key"]["A2"],
                "k3": state["dyn2"]["key"]["k3"],
            },
        }

        resultados = [res1, res2]

        pdf_path = export_results_pdf_guia1(
            filename_base=f"guia1_{state['student'].get('id', 'sin_id')}",
            student_info=state["student"],
            resultados=resultados,
        )

        if not pdf_path or not os.path.exists(pdf_path):
            st.error(f"No se pudo generar el PDF en disco. Ruta esperada:\n{pdf_path}")
            st.stop()

        with open(pdf_path, "rb") as f:
            st.download_button(
                "Descargar PDF",
                data=f.read(),
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
                key="g1_download_pdf",
            )

        nombre_pdf_repo = os.path.basename(pdf_path)
        ruta_repo = f"guia1/{nombre_pdf_repo}"
        ok, info = upload_file_to_github_results(pdf_path, ruta_repo)

        if ok:
            st.success("PDF generado y enviado correctamente al repositorio de RESULTADOS.")
            if isinstance(info, str) and info.startswith("http"):
                st.link_button("Ver archivo en GitHub", info)
            st.write("Ruta en el repositorio:", ruta_repo)
        else:
            st.error(f"El PDF se generó, pero falló el envío a GitHub: {info}")


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
