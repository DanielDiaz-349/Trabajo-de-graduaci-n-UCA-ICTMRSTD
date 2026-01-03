# main1.py
# -*- coding: utf-8 -*-
"""
Guía interactiva 1 en Streamlit: Ruido en sistemas de telecomunicación
Requisitos: python3, streamlit, numpy, matplotlib. Opcional: scipy, reportlab.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import guia2  # <-- Guía 2 en módulo separado (guia2.py)
import guia3  # <-- Guía 3 en módulo separado (guia3.py)
import guia4
import guia5
import os
import math
import json
import random
import datetime
import base64
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st


def add_uca_logo_to_ui():
    """
    Coloca el logo de la UCA en la esquina inferior izquierda
    con opacidad ~0.4 en toda la interfaz de Streamlit.
    """
    logo_path = LOGO_UCA_PATH
    if not os.path.exists(logo_path):
        return

    with open(logo_path, "rb") as f:
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

def upload_file_to_github(local_path, path_in_repo):
    """
    Sube un archivo local a un repositorio de GitHub usando la API.
    path_in_repo: ruta dentro del repo, por ejemplo 'guia1/archivo.pdf'
    """
    token = os.getenv("GITHUB_TOKEN")
    user = os.getenv("GITHUB_USER")
    repo = os.getenv("GITHUB_REPO")

    if not token or not user or not repo:
        print("Faltan variables de entorno GITHUB_TOKEN, GITHUB_USER o GITHUB_REPO.")
        return False

    if not os.path.exists(local_path):
        print("El archivo local no existe:", local_path)
        return False

    with open(local_path, "rb") as f:
        content = f.read()
    b64_content = base64.b64encode(content).decode("utf-8")

    url = f"https://api.github.com/repos/{user}/{repo}/contents/{path_in_repo}"
    headers = {"Authorization": f"token {token}"}
    data = {
        "message": f"Subiendo {os.path.basename(local_path)} desde Streamlit (Guía 1)",
        "content": b64_content
    }

    r = requests.put(url, headers=headers, json=data)
    if r.status_code in (200, 201):
        print("Subida correcta a GitHub:", path_in_repo)
        return True
    else:
        print("Error al subir a GitHub:", r.status_code, r.text)
        return False


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOGO_UCA_PATH = str(BASE_DIR / "assets" / "logo_uca.png")

TEMA_TG = (
    "Introducción a la caracterización y tratamiento matemático del ruido "
    "en sistemas de telecomunicaciones digitales"
)

st.set_page_config(
    page_title="Introddución a la caracterización y tratamiento matematico del ruido",
    layout="wide",                 # <- aquí activás el modo ancho
    initial_sidebar_state="expanded"

)

# >>> NUEVO: Estado centralizado para dinámicas Guía 1
if "guia1_dinamicas" not in st.session_state:
    st.session_state.guia1_dinamicas = {
        "student": {
            "name": "",
            "id": "",
            "dob": ""
        },
        "dyn1": {
            "key": None,
            "answers": {},
            "completed": False
        },
        "dyn2": {
            "key": None,
            "answers": {},
            "completed": False
        },
        "submitted": False
    }


# Optional
try:
    from scipy.signal import butter, lfilter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Optional PDF
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rcanvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# Carpeta para resultados
RESULTS_DIR = "resultados_dinamicas"
os.makedirs(RESULTS_DIR, exist_ok=True)
GENERATE_PDF = True

# ---------- TEXTOS ESTÁTICOS (OBJETIVOS, INTRO, CONCLUSIONES) ----------

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

MATERIALES_TEXT= """
Para desarrollar las actividades de esta guía interactiva se recomienda contar con:

- Una computadora personal con sistema operativo actualizado (Windows, Linux o macOS).
- Python instalado (versión 3.8 o superior recomendada).
- Un entorno de desarrollo como Visual Studio Code o PyCharm.
- Las siguientes bibliotecas de Python:
  - `numpy` para el manejo de arreglos y operaciones numéricas.
  - `matplotlib` para la generación de gráficas.
  - `streamlit` para la interfaz interactiva de la guía.
  - `scipy` para operaciones adicionales de filtrado, convolución y análisis en frecuencia.
"""

CONCLUSIONES_TEXT = """ - El estudio de los sistemas de telecomunicaciones demuestra que la calidad de transmisión depende fundamentalmente de la interacción entre el canal, las fuentes de ruido y los efectos derivados de la no linealidad de los dispositivos. La guía permitió analizar y simular cómo el ruido AWGN, la atenuación del canal y la intermodulación alteran la forma de onda original y afectan directamente la capacidad del receptor para recuperar la información enviada, destacando la importancia del SNR como parámetro clave en la detección confiable de señales digitales.

- A través de los ejemplos prácticos incluidos, el estudiante pudo visualizar de forma gráfica y cuantitativa tanto el impacto del ruido aditivo como la generación de productos de intermodulación en sistemas multiseñal. La comparación entre distintos canales guiados e inalámbricos evidenció que cada medio introduce degradaciones particulares, por lo que el diseño de sistemas modernos exige considerar modelos precisos de ruido, características físicas del canal y técnicas de mitigación orientadas a preservar la integridad de la información transmitida.
"""

# ---------- UTILIDADES NUMÉRICAS ----------

def generar_tren_nrz(bits, fs, Tb, level0=0.0, level1=1.0):
    samples_per_bit = int(max(1, round(Tb * fs)))
    t = np.arange(0, len(bits) * Tb, 1.0/fs)
    if t.size == 0:
        t = np.array([0.0])
    tx = np.zeros_like(t)
    for i, b in enumerate(bits):
        s = i * samples_per_bit
        tx[s:s+samples_per_bit] = level1 if b == 1 else level0
    return t, tx

def generar_ruido_awgn(signal, SNR_dB):
    sigp = np.mean(signal**2) if signal.size > 0 else 1e-12
    SNR_lin = 10**(SNR_dB/10.0)
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
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
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
    samples_per_bit = int(max(1, round(Tb*fs)))
    sampling_times = (np.arange(bits_len) + 0.5) * Tb
    sampling_idxs = (sampling_times * fs).astype(int) + delay_samples
    sampling_idxs = sampling_idxs[sampling_idxs < received.size]
    decisions = np.array([1 if received[idx] >= threshold else 0 for idx in sampling_idxs])
    return decisions

# ---------- CLAVES ALEATORIAS PARA DINÁMICAS ----------

def generate_dyn1_key():
    snr = round(random.uniform(-5.0, 40.0), 2)
    delay = round(random.uniform(0.0, 0.5), 2)
    if snr < 5.0:
        cat = "Baja"; ber_cat = "Alta"
    elif snr < 12.0:
        cat = "Media"; ber_cat = "Moderada"
    else:
        cat = "Alta"; ber_cat = "Baja"
    return {'snr': snr, 'delay': delay, 'q1': cat, 'q2': ber_cat, 'q3': 'Disminuye', 'q4': 'Sí'}

def generate_dyn2_key():
    f1 = random.randint(700, 1400)
    f2 = random.randint(700, 1400)
    while abs(f2 - f1) < 50:
        f2 = random.randint(700, 1400)
    A1 = round(random.uniform(0.6, 1.8), 2)
    A2 = round(random.uniform(0.6, 1.8), 2)
    k3 = round(random.uniform(0.05, 0.2), 3)
    im3_1 = 2*f1 - f2
    im3_2 = 2*f2 - f1
    in_band = any(850 <= v <= 950 for v in (im3_1, im3_2))
    return {
        'f1': f1, 'f2': f2, 'A1': A1, 'A2': A2, 'k3': k3,
        'im3_1': im3_1, 'im3_2': im3_2,
        'q1': 'Intermodulación',
        'q2': 'Aumentan',
        'q3': 'Sí' if in_band else 'No'
    }

# ---------- EXPORTAR RESULTADOS DINÁMICAS ----------

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

def export_results_pdf_guia1(filename_base, student_info, resultados):
    """
    Genera un solo PDF con el resumen de TODAS las dinámicas de la Guía 1.
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

    # Marca de agua con logo UCA (igual estilo que Guía 2)
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
            c.setFillAlpha(0.2)   # ~20% opacidad
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

    # Encabezado
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, top, "Resultados Guía 1 – Dinámicas")
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

    # Nota global (promedio de dinámicas)
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


# ---------- TEMA DE LA INTERFAZ (BLANCO / OBSCURO / ROSA) ----------

def apply_theme(theme_name: str):
    """
    Aplica tema visual (fondo + colores de gráficos) según Blanco / Obscuro / Rosa.
    Ajusta:
    - Texto general y de sidebar.
    - Botones (incluido 'Iniciar dinámica').
    - Inputs de texto/número.
    - Selects (tema, guía, ejemplos).
    - Menús desplegables.
    - Botones + / - de parámetros.
    - Icono de 'ampliar imagen'.
    - Colores de radios (Dinámica 1 / Dinámica 2).
    - Fondos de gráficas en tema Obscuro (blancas).
    """
    t = theme_name.lower()

    if t == "obscuro":
        bg = "#0f1113"
        fg = "#ffffff"      # texto general
        panel = "#2b2f36"
        button_bg = "#2b2f36"
        button_fg = "#ffffff"

        # Gráficas con fondo blanco y texto negro
        fig_face = "#ffffff"
        ax_face = "#ffffff"
        ax_text = "#000000"

        # Inputs / selects / iconos en oscuro
        input_bg = "#1f2227"
        input_fg = "#ffffff"
        select_bg = "#2b2f36"
        select_fg = "#ffffff"
        icon_bg = "#2b2f36"
        icon_fg = "#ffffff"
    elif t == "rosa":
        bg = "#fff6fb"
        fg = "#330033"      # texto general
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
    else:  # blanco
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
    /* Fondo general */
    .stApp {{
        background-color: {bg};
        color: {fg};
        
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {panel};
        color: {fg};
    }}
    section[data-testid="stSidebar"] * {{
        color: {fg};
    }}

    /* Texto general de markdown, radios, selects, sliders, etc. */
    .stMarkdown, .stText, .stSelectbox, .stNumberInput, .stTextInput, .stSlider, .stTabs, .stExpander {{
        color: {fg};
    }}

    /* Tabs (Objetivos, Introducción, Materiales, Ejemplos, Dinámicas, Conclusiones) */
    .stTabs [role="tab"], .stTabs [role="tab"] * {{
        color: {fg} !important;
    }}

    /* Expanders: solo el contenedor (no tocar todos los hijos: rompe Plotly/BaseWeb) */
div[data-testid="stExpander"] > details {{
    background-color: {panel};
    border-radius: 8px;
    border: 1px solid rgba(0,0,0,0.15);
    padding: 0.25rem 0.75rem;
}}
div[data-testid="stExpander"] > details > summary {{
    background-color: {panel} !important;   /* <-- esto quita el negro */
    color: {fg} !important;
    border-radius: 8px !important;
    padding: 0.25rem 0.5rem !important;
}}

div[data-testid="stExpander"] > details > summary * {{
    color: {fg} !important;
}}

div[data-testid="stExpander"] > details > summary:hover {{
    filter: brightness(1.05);
}}

}}

    /* Etiquetas de inputs, radios, selects (Nombre completo, Carné, etc.) */
    label, .stRadio label, .stSelectbox label, .stTextInput label, .stNumberInput label, .stSlider label {{
        color: {fg} !important;
    }}

    /* ==== RADIOS (incluye 'Dinámica 1', 'Dinámica 2') ==== */
    div[role="radiogroup"], div[role="radiogroup"] * {{
        color: {fg} !important;
    }}

/* Radios/checkboxes (BaseWeb): contorno y marca visibles */
div[data-baseweb="radio"] svg, div[data-baseweb="checkbox"] svg {{
    fill: {fg} !important;
    stroke: {fg} !important;
}}

    /* ==== BOTONES GENERALES (incluye 'Iniciar dinámica') ==== */
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

    /* ==== CAMPOS DE TEXTO Y NÚMEROS ==== */
    input[type="text"],
    input[type="number"],
    textarea {{
        background-color: {input_bg} !important;
        color: {input_fg} !important;
        border-radius: 4px !important;
        border: 1px solid #999999 !important;
    }}

    /* ==== SELECTS (ej: Tema de la interfaz, Selecciona una guía, Ejemplo 1/2/3) ==== */
    /* Caja visible del selectbox */
    div[data-baseweb="select"] > div {{
        background-color: {select_bg} !important;
        color: {select_fg} !important;
        border-radius: 4px !important;
        border: 1px solid #999999 !important;
    }}

    /* Texto dentro del valor seleccionado del select */
    div[data-baseweb="select"] span {{
        color: {select_fg} !important;
    }}

    /* Menú desplegable del select (lista de opciones) */
    div[data-baseweb="select"] div[role="listbox"],
    div[data-baseweb="select"] ul,
    div[data-baseweb="select"] li {{
        background-color: {select_bg} !important;
        color: {select_fg} !important;
    }}

    /* ==== BOTONES + / - EN NUMBER_INPUT ==== */
    .stNumberInput button {{
        background-color: {icon_bg} !important;
        color: {icon_fg} !important;
        border-radius: 3px !important;
        border: 1px solid #999999 !important;
    }}

    /* ==== ICONOS (incluye 'ampliar imagen' / fullscreen) ==== */
    button[kind="icon"],
    button[aria-label*="full"],
    button[title*="full"] {{
        background-color: {icon_bg} !important;
        color: {icon_fg} !important;
        border-radius: 3px !important;
        border: 1px solid #999999 !important;
    }}
        /* ===== FIX: labels visibles en todos los temas ===== */
    div[data-testid="stWidgetLabel"] > label,
    div[data-testid="stWidgetLabel"] > label p,
    label,
    label p {{
        color: {fg} !important;
        opacity: 1 !important;
    }}

    div[data-testid="stWidgetLabel"] * {{
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

    # Ajuste de matplotlib para que las gráficas sigan el tema
    mpl.rcParams["figure.facecolor"] = fig_face
    mpl.rcParams["axes.facecolor"] = ax_face
    mpl.rcParams["axes.edgecolor"] = ax_text
    mpl.rcParams["axes.labelcolor"] = ax_text
    mpl.rcParams["xtick.color"] = ax_text
    mpl.rcParams["ytick.color"] = ax_text
    mpl.rcParams["text.color"] = ax_text
    mpl.rcParams["axes.titlecolor"] = ax_text

# ---------- EJEMPLOS: FUNCIONES DE RENDER EN STREAMLIT ----------

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

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.16,  # <-- MÁS ESPACIO ENTRE GRÁFICAS
            subplot_titles=(
                "Señal original",
                "Ruido AWGN",
                "Señal + ruido"
            ),
        )

        # Colores azul uniforme
        blue = "blue"

        if t.size > 0 and tx.size > 0:
            fig.add_trace(
                go.Scatter(x=t, y=tx, mode="lines", line=dict(color=blue)),
                row=1, col=1
            )

        if t.size > 0 and noise.size > 0:
            fig.add_trace(
                go.Scatter(x=t, y=noise, mode="lines", line=dict(color=blue)),
                row=2, col=1
            )

        if t.size > 0 and rx.size > 0:
            fig.add_trace(
                go.Scatter(x=t, y=rx, mode="lines", line=dict(color=blue)),
                row=3, col=1
            )

        # Etiquetas de ejes

        fig.update_xaxes(title_text="Tiempo (s)", row=3, col=1)

        fig.update_yaxes(title_text="Amplitud", row=1, col=1)
        fig.update_yaxes(title_text="Amplitud", row=2, col=1)
        fig.update_yaxes(title_text="Amplitud", row=3, col=1)

        # Layout mejorado
        fig.update_layout(
            height=750,  # <-- MÁS ALTO
            margin=dict(l=40, r=20, t=90, b=60),  # <-- EVITA TRASLAPE
            hovermode="x unified",
            showlegend=False,
            # ---- FORZAR FONDO BLANCO SIEMPRE ----
            paper_bgcolor="white",
            plot_bgcolor="white",
            # ---- TEXTO SIEMPRE LEGIBLE ----
            font=dict(
                color="black",
                size=12
            ),

            # ---- TITULO HOVER ----
            hoverlabel=dict(
                bgcolor="white",
                font=dict(color="black")
            ),

        )
        # ---- FORZAR ESTILO DE EJES (independiente del theme) ----
        fig.update_xaxes(
            showgrid=True,
            gridcolor="lightgray",
            zerolinecolor="black",
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            tickfont=dict(color="black"),
            title_font=dict(color="black"),
        )

        fig.update_yaxes(
            showgrid=True,
            gridcolor="lightgray",
            zerolinecolor="black",
            linecolor="black",
            ticks="outside",
            tickcolor="black",
            tickfont=dict(color="black"),
            title_font=dict(color="black"),
        )
        # ---- FORZAR COLOR DE TÍTULOS DE SUBGRÁFICAS ----
        fig.update_annotations(font=dict(color="black", size=13))

        fig.update_xaxes(rangeslider_visible=True, row=3, col=1)

        st.plotly_chart(fig, use_container_width=True, theme=None)


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

        # 1) Espectro ANTES de la no linealidad
        fig1, ax1 = plt.subplots(figsize=(7, 3))
        ax1.semilogy(freq, X_in + 1e-12)
        ax1.set_xlim(0, fmax_plot)
        ax1.set_xlabel("Frecuencia (Hz)")
        ax1.set_ylabel("Magnitud (u.a.)")
        ax1.set_title("Espectro antes de la no linealidad")
        ax1.grid(True, linestyle=":")
        fig1.tight_layout(pad=2.0)
        st.pyplot(fig1)

        # --- Frecuencias de IMD de tercer orden ---
        imd_freqs = {
            "2f₁−f₂": 2 * f1 - f2,
            "2f₂−f₁": 2 * f2 - f1,
            "3f₁": 3 * f1,
            "3f₂": 3 * f2,
            "2f₁+f₂": 2 * f1 + f2,
            "2f₂+f₁": 2 * f2 + f1,
        }

        # 2) Espectro DESPUÉS de la no linealidad con etiquetas (sin puntos rojos/naranjas)
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        ax2.semilogy(freq, X_out + 1e-12)
        ax2.set_xlim(0, fmax_plot)
        ax2.set_xlabel("Frecuencia (Hz)")
        ax2.set_ylabel("Magnitud (u.a.)")
        ax2.set_title("Espectro después de la no linealidad ")
        ax2.grid(True, linestyle=":")

        # Etiquetar portadoras originales f1 y f2 (solo texto)
        for f_c, label in [(f1, "f₁"), (f2, "f₂")]:
            if 0 < f_c < freq[-1]:
                idx = np.argmin(np.abs(freq - f_c))
                amp = X_out[idx] + 1e-12
                ax2.text(
                    freq[idx],
                    amp * 1.3,   # un poco arriba de la línea, sin irse al título
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="black"
                )

        # Etiquetar productos de IMD de tercer orden (solo texto, sin puntos)
        for label, f_imd in imd_freqs.items():
            if 0 < f_imd < freq[-1]:
                idx = np.argmin(np.abs(freq - f_imd))
                amp = X_out[idx] + 1e-12
                ax2.text(
                    freq[idx],
                    amp * 1.3,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                    color="black"
                )

        fig2.tight_layout(pad=2.0)
        st.pyplot(fig2)

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

        fig3, ax3 = plt.subplots(figsize=(7, 3))
        ax3.plot(t_win, y_clean_win, label="Canal adyacente antes de la intermodulación")
        ax3.plot(t_win, y_dist_win, label="Canal adyacente después de la intermodulación", alpha=0.8)
        ax3.set_xlabel("Tiempo (s)")
        ax3.set_ylabel("Amplitud")
        ax3.set_title("Señal de un canal afectado por productos de intermodulación")
        ax3.grid(True, linestyle=":")
        ax3.legend(loc="upper right", fontsize=8)
        fig3.tight_layout(pad=2.0)
        st.pyplot(fig3)

        # --- Explicación y preguntas ---
        st.markdown("#### Explicación de la simulación y preguntas")

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

# ---------- DINÁMICAS EN STREAMLIT ----------

def render_dinamica1():
    """
    Dinámica 1: relacionada con el Ejemplo 1 (AWGN, SNR, retardo).
    """
    st.markdown("### Dinámica 1 — SNR, AWGN y BER en una señal digital")

    if "student_info" not in st.session_state:
        st.session_state.student_info = {"name": "", "id": "", "dob": ""}

    info = st.session_state.student_info

    # --- Registro del estudiante ---
    with st.form("form_dyn1_registro"):
        st.write("Datos del estudiante")
        name = st.text_input("Nombre completo", value=info["name"])
        carnet = st.text_input("Carné", value=info["id"])
        dob = st.text_input("Fecha de nacimiento (YYYY-MM-DD)", value=info["dob"])
        iniciar = st.form_submit_button("Iniciar dinámica")

    if iniciar:
        if not name or not carnet or not dob:
            st.warning("Completa nombre, carné y fecha de nacimiento.")
        else:
            st.session_state.student_info = {"name": name, "id": carnet, "dob": dob}
            st.session_state.dyn1_key = generate_dyn1_key()
            st.success("Caso generado. Desplázate hacia abajo para responder.")

    key = st.session_state.get("dyn1_key", None)
    if not key:
        st.info("Completa el formulario y pulsa 'Iniciar dinámica' para generar el caso.")
        return

    # --- Texto del caso generado ---
    st.markdown(f"**Caso generado:** SNR = {key['snr']} dB, retardo = {key['delay']} · T")

    # --- Generación de señal, ruido y señal+ruido ---
    bits = np.random.randint(0, 2, size=200)
    fs = 2000
    Tb = 1.0
    t, tx = generar_tren_nrz(bits, fs, Tb, level0=0.0, level1=1.0)
    tx_filtered = tx.copy()
    noise = generar_ruido_awgn(tx_filtered, key["snr"])
    rx = tx_filtered + noise

    # --- Gráficas interactivas con Plotly (zoom, pan, hover) ---
    st.write("Visualiza la señal transmitida, el ruido y la señal combinada. "
             "Puedes hacer **zoom con el mouse**, desplazarte y ver los valores al pasar el cursor.")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.16,
        subplot_titles=(
            "Señal transmitida (NRZ)",
            "Ruido AWGN",
            "Señal transmitida + ruido"
        ),
    )

    color = "blue"

    # Señal transmitida
    fig.add_trace(
        go.Scatter(x=t, y=tx_filtered, mode="lines", line=dict(color=color)),
        row=1, col=1
    )

    # Ruido
    fig.add_trace(
        go.Scatter(x=t, y=noise, mode="lines", line=dict(color=color)),
        row=2, col=1
    )

    # Señal + ruido
    fig.add_trace(
        go.Scatter(x=t, y=rx, mode="lines", line=dict(color=color)),
        row=3, col=1
    )

    # Ejes
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
        font=dict(color="black"),
        hoverlabel=dict(
            bgcolor="white",
            font=dict(color="black")
        ),

    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="lightgray",
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="lightgray",
        linecolor="black",
        tickfont=dict(color="black"),
        title_font=dict(color="black"),
    )

    # Rango deslizante en el eje de tiempo (parte inferior)
    fig.update_xaxes(rangeslider_visible=True, row=3, col=1)

    st.plotly_chart(fig, use_container_width=True, theme=None)

    # --- Preguntas de la dinámica ---
    st.write(
        "Responde las siguientes preguntas interpretando la gráfica anterior "
        "y usando los conceptos de SNR, AWGN y BER."
    )

    with st.form("form_dyn1_resp"):
        q1 = st.radio(
            "1) Observando la señal combinada, la SNR del caso se percibe como:",
            ["Alta", "Media", "Baja"],
            index=None
        )
        q2 = st.radio(
            "2) Si la señal muestra cruces frecuentes del umbral ideal entre 0 y 1, la BER esperada es:",
            ["Baja", "Moderada", "Alta"],
            index=None
        )
        q3 = st.radio(
            "3) Al aumentar la SNR manteniendo todo lo demás constante, la BER:",
            ["Aumenta", "Disminuye", "Se mantiene"],
            index=None
        )
        q4 = st.radio(
            "4) ¿El ruido AWGN se considera de naturaleza gaussiana?",
            ["Sí", "No"],
            index=None
        )

        enviar = st.form_submit_button("Guardar respuestas")

    if enviar:
        answers = {
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "q4": q4,
        }

        if None in answers.values():
            st.warning("Responde todas las preguntas antes de enviar.")
            return

        correct_answers = {
            "q1": key["q1"],
            "q2": key["q2"],
            "q3": key["q3"],
            "q4": key["q4"],
        }

        correct = sum(answers[k] == correct_answers[k] for k in answers.keys())
        mapping = {4: 10.0, 3: 8.0, 2: 6.0, 1: 4.0, 0: 0.0}
        score = mapping.get(correct, 0.0)

        st.session_state["g1_dyn1_result"] = {
            "dyn_id": 1,
            "score": score,
            "answers": answers,
            "correct": correct_answers,
            "key": {
                "descripcion": "Guía 1 - Dinámica 1 - Ruido AWGN y BER",
                "snr_dB": key["snr"],
                "delay_Tb": key["delay"],
            },
        }
        # Estado centralizado (Dinámicas – Guía 1)
        try:
            st.session_state.guia1_dinamicas["dyn1"]["key"] = key
            st.session_state.guia1_dinamicas["dyn1"]["answers"] = answers
            st.session_state.guia1_dinamicas["dyn1"]["completed"] = True
        except Exception:
            pass

        st.success(
            "Respuestas guardadas para la Dinámica 1. "
            "Ve al tab **Enviar respuestas** cuando completes todas las dinámicas de la guía."
        )


def render_dinamica2():
    """
    Dinámica 2: relacionada con el Ejemplo 2 (intermodulación).
    """
    st.markdown("### Dinámica 2 — Identificación de productos de intermodulación")

    # --- Registro del estudiante (compartido con las demás dinámicas) ---
    if "student_info" not in st.session_state:
        st.session_state.student_info = {"name": "", "id": "", "dob": ""}

    info = st.session_state.student_info

    with st.form("form_dyn2_registro"):
        st.write("Datos del estudiante")
        name = st.text_input("Nombre completo", value=info["name"], key="name_dyn2")
        carnet = st.text_input("Carné", value=info["id"], key="id_dyn2")
        dob = st.text_input("Fecha de nacimiento (YYYY-MM-DD)", value=info["dob"], key="dob_dyn2")
        iniciar = st.form_submit_button("Iniciar dinámica")

    if iniciar:
        if not name or not carnet or not dob:
            st.warning("Completa nombre, carné y fecha de nacimiento.")
        else:
            st.session_state.student_info = {"name": name, "id": carnet, "dob": dob}
            st.session_state.dyn2_key = generate_dyn2_key()
            st.success("Caso generado. Desplázate hacia abajo para responder.")

    key = st.session_state.get("dyn2_key", None)
    if not key:
        st.info("Completa el formulario y pulsa 'Iniciar dinámica' para generar el caso.")
        return

    st.markdown(
        f"**Caso generado:** f1 = {key['f1']} Hz, f2 = {key['f2']} Hz, "
        f"A1 = {key['A1']}, A2 = {key['A2']}, k3 = {key['k3']}"
    )

    # --- Señales y espectros (MISMA LÓGICA QUE EN EL EJEMPLO 2) ---
    fs = 16000
    T = 0.05
    t = np.arange(0, T, 1.0/fs)

    # Entrada al dispositivo no lineal
    x = key["A1"] * np.cos(2*np.pi*key["f1"]*t) + key["A2"] * np.cos(2*np.pi*key["f2"]*t)
    # Salida con no linealidad cúbica
    y = x + key["k3"] * (x**3)

    # Espectros
    freq = np.fft.rfftfreq(len(t), 1.0/fs)
    X_in = np.abs(np.fft.rfft(x)) / len(t)
    X_out = np.abs(np.fft.rfft(y)) / len(t)

    # Límite de frecuencia para visualizar (similar al Ejemplo 2)
    fmax_plot = max(key["f1"], key["f2"]) * 4.0

    # Figura con dos subgráficos: antes y después
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # 1) Espectro ANTES de la no linealidad
    a1.semilogy(freq, X_in + 1e-12)
    a1.set_xlim(0, fmax_plot)
    a1.set_ylabel("Magnitud (u.a.)")
    a1.set_title("Espectro antes de la no linealidad")
    a1.grid(True, which="both", linestyle=":", alpha=0.5)

    # 2) Espectro DESPUÉS de la no linealidad con etiquetas como en el ejemplo 2
    a2.semilogy(freq, X_out + 1e-12)
    a2.set_xlim(0, fmax_plot)
    a2.set_xlabel("Frecuencia (Hz)")
    a2.set_ylabel("Magnitud (u.a.)")
    a2.set_title("Espectro después de la no linealidad")
    a2.grid(True, which="both", linestyle=":", alpha=0.5)

    # Etiquetas para las portadoras originales f1 y f2
    for f_c, label in [(key["f1"], r"$f_1$"), (key["f2"], r"$f_2$")]:
        if 0 < f_c < fmax_plot:
            idx = np.argmin(np.abs(freq - f_c))
            amp = X_out[idx] + 1e-12
            a2.text(
                f_c,
                amp * 1.5,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="black",
            )

    # Productos de IMD de tercer orden (mismos que en el ejemplo 2)
    imd_freqs = {
        r"$2f_1-f_2$": 2 * key["f1"] - key["f2"],
        r"$2f_2-f_1$": 2 * key["f2"] - key["f1"],
        r"$3f_1$": 3 * key["f1"],
        r"$3f_2$": 3 * key["f2"],
        r"$2f_1+f_2$": 2 * key["f1"] + key["f2"],
        r"$2f_2+f_1$": 2 * key["f2"] + key["f1"],
    }

    for label, f_imd in imd_freqs.items():
        if 0 < f_imd < fmax_plot:
            idx = np.argmin(np.abs(freq - f_imd))
            amp = X_out[idx] + 1e-12
            a2.text(
                f_imd,
                amp * 1.5,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color="black",
            )

    fig.tight_layout(pad=3.0)
    st.pyplot(fig)

    # --- Preguntas (igual que ya tenías) ---
    st.write("Responde las preguntas, identificando la naturaleza de la distorsión.")

    with st.form("form_dyn2_resp"):
        q1 = st.radio(
            "1) ¿Qué tipo de distorsión aparece al mezclar frecuencias en un dispositivo no lineal?",
            ["Intermodulación", "Armónica pura", "Ruido blanco"],
            index=None
        )
        q2 = st.radio(
            "2) ¿Qué ocurre con los productos de intermodulación si aumenta A1 o A2?",
            ["Disminuyen", "Se mantienen", "Aumentan"],
            index=None
        )
        q3 = st.radio(
            "3) ¿Pueden los productos de intermodulación caer dentro del canal útil?",
            ["Sí", "No"],
            index=None
        )

        enviar = st.form_submit_button("Enviar respuestas (finalizar)")

    if enviar:
        answers = {"q1": q1, "q2": q2, "q3": q3}
        if None in answers.values():
            st.warning("Responde todas las preguntas antes de enviar.")
            return

        # Clave correcta viene de generate_dyn2_key()
        correct = 0
        if answers["q1"] == key["q1"]:
            correct += 1
        if answers["q2"] == key["q2"]:
            correct += 1
        if answers["q3"] == key["q3"]:
            correct += 1

        mapping = {3: 10.0, 2: 8.0, 1: 6.0, 0: 0.0}
        score = mapping.get(correct, 0.0)

        # Guardar todo en session_state (la Guía 1 ya usa un PDF global en "Enviar respuestas")
        st.session_state["g1_dyn2_result"] = {
            "dyn_id": 2,
            "score": score,
            "answers": answers,
            "correct": {
                "q1": key["q1"],
                "q2": key["q2"],
                "q3": key["q3"],
            },
            "key": {
                "descripcion": "Guía 1 - Dinámica 2 - Productos de intermodulación de tercer orden",
                "f1_Hz": key["f1"],
                "f2_Hz": key["f2"],
                "A1": key["A1"],
                "A2": key["A2"],
                "k3": key["k3"],
            },
        }
        # Estado centralizado (Dinámicas – Guía 1)
        try:
            st.session_state.guia1_dinamicas["dyn2"]["key"] = key
            st.session_state.guia1_dinamicas["dyn2"]["answers"] = answers
            st.session_state.guia1_dinamicas["dyn2"]["completed"] = True
        except Exception:
            pass

        st.success(
            "Respuestas guardadas para la Dinámica 2. Ve al tab **Enviar respuestas** cuando completes todas las dinámicas de la guía."
        )
# >>> NUEVO: Dinámicas integradas Guía 1
def render_dinamicas_guia1():
    st.markdown("## Dinámicas – Guía 1")

    state = st.session_state.guia1_dinamicas

    # -------- REGISTRO DEL ESTUDIANTE --------
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
            st.success("Datos guardados correctamente.")

    if not all(state["student"].values()):
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
            unsafe_allow_html=True
        )
        return

    st.markdown("---")

    # -------- DINÁMICA 1 --------
    with st.expander("Dinámica 1 — AWGN, SNR y BER", expanded=True):
        # Caso fijo en la sesión (para que no cambie a cada rerun)
        if state["dyn1"]["key"] is None:
            state["dyn1"]["key"] = generate_dyn1_key()
        key = state["dyn1"]["key"]

        st.markdown(f"**Caso:** SNR = {key['snr']} dB | Retardo = {key['delay']}·T")

        # --- Señal, ruido y señal+ruido (ligera para Plotly) ---
        bits = np.random.randint(0, 2, size=200)
        fs = 2000
        Tb = 0.01  # 10 ms por bit (fluido)

        t, tx = generar_tren_nrz(bits, fs, Tb, level0=0.0, level1=1.0)
        noise = generar_ruido_awgn(tx, key["snr"])
        rx = tx + noise

        delay_samples = int(round(key["delay"] * Tb * fs))
        if delay_samples > 0:
            rx = np.concatenate((np.zeros(delay_samples), rx))[:rx.size]

        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.16,
            subplot_titles=("Señal transmitida (NRZ)", "Ruido AWGN", "Señal + ruido")
        )

        color = "blue"
        fig.add_trace(go.Scattergl(x=t, y=tx, mode="lines", line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scattergl(x=t, y=noise, mode="lines", line=dict(color=color)), row=2, col=1)
        fig.add_trace(go.Scattergl(x=t, y=rx, mode="lines", line=dict(color=color)), row=3, col=1)

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
            font=dict(color="black"),
            hoverlabel=dict(bgcolor="white", font=dict(color="black")),
        )
        fig.update_xaxes(
            showgrid=True, gridcolor="lightgray", linecolor="black",
            tickfont=dict(color="black"), title_font=dict(color="black")
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="lightgray", linecolor="black",
            tickfont=dict(color="black"), title_font=dict(color="black")
        )
        fig.update_xaxes(rangeslider_visible=True, row=3, col=1)

        st.plotly_chart(fig, use_container_width=True, theme=None)

        # --- Preguntas (se guardan en vivo; envío al final) ---
        q1 = st.radio("Clasificación del SNR:", ["Baja", "Media", "Alta"], index=None, key="g1_dyn1_q1")
        q2 = st.radio("Comportamiento de la BER:", ["Alta", "Moderada", "Baja"], index=None, key="g1_dyn1_q2")
        q3 = st.radio("¿Qué ocurre con la BER al aumentar el SNR?", ["Aumenta", "Disminuye"], index=None, key="g1_dyn1_q3")
        q4 = st.radio("¿El ruido AWGN es aditivo?", ["Sí", "No"], index=None, key="g1_dyn1_q4")

        state["dyn1"]["answers"] = {"q1": q1, "q2": q2, "q3": q3, "q4": q4}
        state["dyn1"]["completed"] = all(v is not None for v in state["dyn1"]["answers"].values())

        if state["dyn1"]["completed"]:
            st.success("Dinámica 1 lista (puedes pasar a la Dinámica 2).")
        else:
            st.info("Selecciona una opción en cada pregunta para completar la Dinámica 1.")

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

        # --- Señales y espectros ---
        fs2 = 16000
        T = 0.05
        t2 = np.arange(0, T, 1.0 / fs2)

        x = key2["A1"] * np.cos(2 * np.pi * key2["f1"] * t2) + key2["A2"] * np.cos(2 * np.pi * key2["f2"] * t2)
        y = x + key2["k3"] * (x ** 3)

        freq = np.fft.rfftfreq(len(t2), 1.0 / fs2)
        X_in = np.abs(np.fft.rfft(x)) / len(t2)
        X_out = np.abs(np.fft.rfft(y)) / len(t2)

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

        # Etiquetas simples (sin mathtext, sin unicode raro)
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

        try:
            fig2.tight_layout(pad=3.0)
        except Exception:
            fig2.subplots_adjust(hspace=0.35)

        st.pyplot(fig2)

        q1 = st.radio("Tipo de distorsión:", ["Armónica", "Intermodulación"], index=None, key="g1_dyn2_q1")
        q2 = st.radio("¿Qué ocurre al aumentar k3?", ["Disminuyen", "Aumentan"], index=None, key="g1_dyn2_q2")
        q3 = st.radio("¿Los productos IM3 pueden caer en banda?", ["Sí", "No"], index=None, key="g1_dyn2_q3")

        state["dyn2"]["answers"] = {"q1": q1, "q2": q2, "q3": q3}
        state["dyn2"]["completed"] = all(v is not None for v in state["dyn2"]["answers"].values())

        if state["dyn2"]["completed"]:
            st.success("Dinámica 2 lista.")
        else:
            st.info("Selecciona una opción en cada pregunta para completar la Dinámica 2.")

    st.markdown("---")

    # -------- ENVÍO FINAL (ÚNICO BOTÓN) --------
    if st.button("Enviar respuestas (generar PDF)"):
        if not state["dyn1"]["completed"] or not state["dyn2"]["completed"]:
            st.error("Debes completar ambas dinámicas antes de enviar.")
            return

        # Resultados Dinámica 1
        key1 = state["dyn1"]["key"]
        ans1 = state["dyn1"]["answers"]
        correct1 = {"q1": key1["q1"], "q2": key1["q2"], "q3": key1["q3"], "q4": key1["q4"]}
        ok1 = sum(ans1.get(k) == v for k, v in correct1.items())
        map1 = {4: 10.0, 3: 8.0, 2: 6.0, 1: 4.0, 0: 0.0}
        score1 = map1.get(ok1, 0.0)

        # Resultados Dinámica 2
        key2 = state["dyn2"]["key"]
        ans2 = state["dyn2"]["answers"]
        correct2 = {"q1": key2["q1"], "q2": key2["q2"], "q3": key2["q3"]}
        ok2 = sum(ans2.get(k) == v for k, v in correct2.items())
        map2 = {3: 10.0, 2: 8.0, 1: 6.0, 0: 0.0}
        score2 = map2.get(ok2, 0.0)

        resultados = [
            {
                "dyn_id": 1,
                "score": score1,
                "answers": ans1,
                "correct": correct1,
                "key": {
                    "descripcion": "Guía 1 - Dinámica 1 - Ruido AWGN y BER",
                    "snr_dB": key1["snr"],
                    "delay_Tb": key1["delay"],
                },
            },
            {
                "dyn_id": 2,
                "score": score2,
                "answers": ans2,
                "correct": correct2,
                "key": {
                    "descripcion": "Guía 1 - Dinámica 2 - Productos de intermodulación de tercer orden",
                    "f1_Hz": key2["f1"],
                    "f2_Hz": key2["f2"],
                    "A1": key2["A1"],
                    "A2": key2["A2"],
                    "k3": key2["k3"],
                },
            },
        ]

        pdf_path = export_results_pdf_guia1(
            filename_base=f"guia1_{state['student'].get('id', 'sin_id')}",
            student_info=state["student"],
            resultados=resultados,
        )

        if not REPORTLAB_AVAILABLE:
            st.error("No se puede generar el PDF porque ReportLab no está instalado en este entorno. "
                     "Instala con: pip install reportlab (y agrégalo a requirements.txt si usarás Streamlit Cloud).")
        elif not pdf_path:
            st.error("No se pudo generar el PDF: la función devolvió una ruta vacía.")
        elif not os.path.exists(pdf_path):
            st.error(f"No se pudo generar el PDF en disco. Ruta esperada:\n{pdf_path}\n"
                     "Revisa permisos de escritura o cambia la carpeta de salida.")
        else:
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "Descargar PDF",
                    data=f.read(),
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                )
            st.success("PDF generado correctamente.")


def render_resumen_dinamicas_guia1():
    st.subheader("Enviar respuestas Guia 1")

    student_info = st.session_state.get("student_info")
    if not student_info:
        st.info("Primero completa el registro en alguna de las dinámicas (nombre, carné, fecha de nacimiento).")
        return

    res1 = st.session_state.get("g1_dyn1_result")
    res2 = st.session_state.get("g1_dyn2_result")

    faltan = []
    if res1 is None:
        faltan.append("Dinámica 1")
    if res2 is None:
        faltan.append("Dinámica 2")

    if faltan:
        st.warning("Aún faltan por completar: " + ", ".join(faltan))
        return

    st.markdown("Todas las dinámicas de la Guía 1 están completadas. Puedes enviar las respuestas para generar el PDF.")

    total_score = res1["score"] + res2["score"]
    promedio = total_score / 2.0

    if st.button("Generar y enviar PDF de la Guía 1"):
        pdf_path = export_results_pdf_guia1(
            filename_base=f"guia1_{student_info.get('id', 'sin_id')}",
            student_info=student_info,
            resultados=[res1, res2],
        )

        # Nombre del archivo en el repositorio: carpeta guia1/
        nombre_pdf_repo = os.path.basename(pdf_path)
        ruta_repo = f"guia1/{nombre_pdf_repo}"

        ok = upload_file_to_github(pdf_path, ruta_repo)

        if ok:
            st.success("PDF generado y enviado correctamente a GitHub.")
            st.write("Ruta local del PDF:", pdf_path)
            st.write("Ruta en el repositorio:", ruta_repo)
        else:
            st.error("El PDF se generó, pero hubo un problema al enviarlo a GitHub. Revisa la consola del servidor.")
            st.write("Ruta local del PDF:", pdf_path)


# ---------- GUÍA 1 COMPLETA EN STREAMLIT ----------

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
        sub_tabs = st.tabs([
            "Ejemplo 1",
            "Ejemplo 2",
            "Ejemplo 3",
        ])
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

# ---------- PUNTO DE ENTRADA GENERAL ----------

def main():
    # Añadir siempre el logo en la interfaz
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
            "Guía 5: Fundamentos de transmisión digital en presencia de ruido "
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
