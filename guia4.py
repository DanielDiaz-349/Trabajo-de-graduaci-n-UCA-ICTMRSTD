# guia4.py
# Guía 4: Procesos estocásticos y el ruido
# Estructura: Objetivos | Introducción | Materiales | Ejemplos | Dinámicas | Conclusiones

from __future__ import annotations

import os
import io
import datetime as _dt
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from github_uploader import upload_file_to_github_results

# SciPy es altamente recomendable para los filtros FIR
try:
    from scipy.signal import firwin, lfilter
    _SCIPY_OK = True
except Exception:
    firwin = None
    lfilter = None
    _SCIPY_OK = False

# PDF (ReportLab)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch
    _REPORTLAB_OK = True
except Exception:
    canvas = None
    letter = None
    inch = None
    _REPORTLAB_OK = False

# -----------------------------
# Textos de la guía (basados en GUIA4.docx)
# -----------------------------

OBJETIVOS_TEXT = """**Objetivo general**

Analizar y comprender el comportamiento del ruido modelado como proceso estocástico, mediante simulaciones interactivas que permitan estimar estadísticos en el dominio del tiempo y de la frecuencia, e interpretar sus efectos en sistemas de telecomunicaciones.

**Objetivos específicos**
- Distinguir entre una realización y un conjunto de realizaciones de un proceso estocástico, identificando el carácter aleatorio del ruido en el tiempo.
- Estimar e interpretar la media y la varianza de ruido Gaussiano (AWGN), y relacionarlas con la forma del histograma (ancho, centrado y dispersión).
- Comparar procesos estacionarios y no estacionarios, identificando evidencias prácticas de no estacionariedad cuando la media y/o la desviación estándar varían con el tiempo (caso lineal).
- Calcular e interpretar la función de autocorrelación como herramienta para describir dependencia temporal del proceso y apoyar la clasificación estacionario/no estacionario.
- Estimar e interpretar la densidad espectral de potencia (PSD), reconociendo el comportamiento de un ruido blanco (PSD aproximadamente plana) y la mejora visual al promediar varias realizaciones.
- Analizar el efecto de un filtro LTI sobre AWGN en frecuencia, verificando que el filtrado moldea la PSD (ruido “coloreado”) y modifica la correlación temporal de la señal resultante.
- Resolver dinámicas evaluativas basadas en casos aleatorios, interpretando una gráfica principal por dinámica y respondiendo preguntas conceptuales que conecten la simulación con los conceptos teóricos.
"""

# Nota: Streamlit renderiza mejor ecuaciones cuando el texto se muestra con st.markdown.
# Este texto incluye un bloque de "Ecuaciones clave" en LaTeX.
INTRO_FULL_TEXT = r"""En un sistema de telecomunicaciones, la señal de información es inherentemente impredecible vista desde el receptor: si su comportamiento completo fuese conocido de antemano, su valor informativo sería muy reducido. A esa incertidumbre se suma un fenómeno inevitable: el ruido. Una parte importante de la degradación del desempeño proviene del ruido aditivo generado dentro del receptor, asociado al movimiento aleatorio de electrones en los conductores. Por ello, para caracterizar y predecir el desempeño (SNR, BER, etc.) se requiere modelar la señal y el ruido mediante herramientas probabilísticas y procesos estocásticos.

El punto de partida es el concepto de proceso estocástico. Un proceso estocástico puede entenderse como una regla que, a cada resultado de un experimento aleatorio, le asigna una función (típicamente del tiempo). Así, el resultado de una realización no es un número, sino una forma de onda. El conjunto de todas las realizaciones conforma el ensemble del proceso, y una realización particular se denomina función de muestra.

Formalmente, un proceso estocástico es una colección de variables aleatorias indexadas por un parámetro (tiempo continuo, tiempo discreto o espacio). De aquí se definen funciones de distribución y densidad de primer orden para X(t). Más importante aún, se definen distribuciones conjuntas para varios instantes, que capturan la dependencia estadística temporal. En sentido estricto, la familia de distribuciones conjuntas caracteriza completamente al proceso, aunque en la práctica no siempre es manejable.

Por esa razón se emplea caracterización parcial mediante estadísticas de primer y segundo orden. La media describe el valor promedio, la varianza cuantifica la dispersión, y la correlación mide la similitud o dependencia entre muestras en distintos instantes. La correlación es especialmente relevante en telecomunicaciones porque conecta el comportamiento temporal con el contenido espectral del ruido y permite estimar potencia de ruido tras filtrado.

Un caso central es el proceso estocástico gaussiano. Se dice que un proceso es gaussiano si cualquier conjunto finito de muestras forma un vector aleatorio conjuntamente gaussiano. Este modelo es atractivo porque muchas fuentes físicas de ruido pueden aproximarse bien como gaussianas (apoyadas en el teorema del límite central). Además, en el caso gaussiano, la media y la correlación determinan toda la estadística de orden finito, y la salida de sistemas LTI excitados por procesos gaussianos permanece gaussiana.

El ruido térmico se modela habitualmente como un proceso gaussiano de media cero. La justificación física es que el movimiento aleatorio de portadores de carga no presenta una dirección preferente; una media distinta de cero implicaría una deriva neta causada por una fuerza externa. En un receptor real, el ruido se propaga por etapas de filtrado, que suelen modelarse como un sistema LTI con función de transferencia H(f) o respuesta al impulso h(t). El modelo base consiste en analizar estadísticamente el ruido a la salida del filtro para obtener parámetros como potencia de ruido o SNR.

Otro concepto clave es la estacionariedad. Un proceso es estacionario si su descripción estadística no cambia con el tiempo; en sentido estricto, todas sus distribuciones conjuntas son invariantes ante desplazamientos temporales. En un proceso estacionario, la media y la varianza son constantes, y la correlación depende únicamente del retardo. En telecomunicaciones, el ruido térmico suele aproximarse como estacionario gaussiano en condiciones normales, lo que simplifica notablemente el análisis.

El análisis en frecuencia completa el marco. Para procesos estacionarios, la potencia promedio se describe mediante la densidad espectral de potencia (DSP). La relación fundamental entre correlación y DSP está dada por el teorema de Wiener–Khinchin: la DSP es la transformada de Fourier de la autocorrelación, y viceversa.

Para un resistor R a temperatura T, el ruido térmico presenta una DSP aproximadamente plana en un rango amplio de frecuencias de interés. En un tratamiento más general, la expresión de la DSP involucra constantes físicas como la constante de Planck (h) y la de Boltzmann (k); sin embargo, en las bandas típicas de telecomunicaciones suele adoptarse la aproximación “blanca” (DSP casi constante). Si además se asume gaussianidad y aditividad, se obtiene el modelo de ruido blanco gaussiano aditivo (AWGN), ampliamente usado como referencia para evaluar desempeño.

Cuando un sistema LTI es excitado por AWGN, la salida conserva la gaussianidad y, si la entrada es estacionaria, la salida también lo es. En el caso ideal de ruido blanco, la DSP de salida resulta S_Y(f)=|H(f)|²·(N₀/2), lo que hace evidente el papel del filtrado sobre la potencia de ruido. En transmisión digital es frecuente usar el modelo equivalente de AWGN complejo en banda base (envolvente compleja), lo que facilita el análisis de modulación, detección y desempeño.

##### Definiciones clave

- **Proceso estocástico:** Colección de variables aleatorias indexadas por un parámetro (tiempo, espacio, etc.). Una realización del proceso es una función de muestra.
- **Función de muestra:** Forma de onda asociada a un resultado particular del experimento aleatorio (una realización del ensemble).
- **Distribución de primer orden:** Describe la estadística de X(t) en un instante t específico.
- **Distribución conjunta:** Describe simultáneamente el proceso en varios instantes e incorpora la dependencia entre ellos.
- **Media m_X(t):** Valor esperado del proceso en cada instante; describe el comportamiento promedio.
- **Varianza σ_X²(t):** Dispersión alrededor de la media; mide la intensidad de las fluctuaciones.
- **Correlación R_X(t1,t2):** Medida de dependencia/similitud entre muestras en dos instantes; contiene información espectral.
- **Proceso gaussiano:** Proceso donde cualquier conjunto finito de muestras es conjuntamente gaussiano.
- **Estacionariedad estricta:** Invariancia temporal de todas las distribuciones conjuntas ante desplazamientos en el tiempo.
- **DSP S_X(f):** Distribución de la potencia promedio del proceso según la frecuencia; para procesos estacionarios se vincula con R_X(τ) por Wiener–Khinchin.
- **AWGN:** Modelo ideal: ruido blanco (DSP plana) + gaussiano (muestras normales) + aditivo (se suma a la señal).
- **Constante de Planck (h):** Constante física usada en expresiones generales del ruido térmico; h ≈ 6.626×10⁻³⁴ J·s.
- **Constante de Boltzmann (k):** Constante física que relaciona temperatura y energía térmica; k ≈ 1.381×10⁻²³ J/K.
- **Ruido complejo en banda base:** Modelo equivalente en envolvente compleja para representar ruido pasa banda alrededor de la portadora en sistemas digitales.

##### Ecuaciones clave

- **Proceso estocástico:** X(t),  t ∈ T   o   {X(t): t ∈ T}
- **Función de muestra (realización):** x(t; ω) = X(t, ω)   para un ω fijo
- **CDF de primer orden:** F_X(x; t) = Pr{ X(t) ≤ x }
- **PDF de primer orden:** f_X(x; t) = ∂F_X(x; t) / ∂x
- **CDF conjunta de segundo orden:** F_X(x₁, x₂; t₁, t₂) = Pr{ X(t₁) ≤ x₁,  X(t₂) ≤ x₂ }
- **PDF conjunta de segundo orden:** f_X(x₁, x₂; t₁, t₂) = ∂²F_X(x₁, x₂; t₁, t₂) / (∂x₁ ∂x₂)
- **Media:** m_X(t) = 𝔼{ X(t) }
- **Varianza:** σ_X²(t) = 𝔼{ [X(t) − m_X(t)]² }
- **Correlación:** R_X(t₁, t₂) = 𝔼{ X(t₁) · X(t₂) }
- **Coeficiente de correlación:** ρ_X(t₁, t₂) = R_X(t₁, t₂) / √( R_X(t₁, t₁) · R_X(t₂, t₂) )
- **Estacionariedad estricta:** f_{X(t₁),…,X(tₙ)}(x₁,…,xₙ) = f_{X(t₁+t₀),…,X(tₙ+t₀)}(x₁,…,xₙ),  ∀t₀
- **Autocorrelación en estacionario:** R_X(τ) = 𝔼{ X(t) · X(t+τ) },   τ = t₂ − t₁
- **Transformada de Fourier en ventana finita:** X_T(f) = ∫_{−T/2}^{T/2} x(t) · exp(−j2π f t) dt
- **Densidad espectral de potencia:** S_X(f) = lim_{T→∞} (1/T) · 𝔼{ |X_T(f)|² }
- **Wiener–Khinchin:** S_X(f) = ∫_{−∞}^{∞} R_X(τ) · exp(−j2π f τ) dτ ;  R_X(τ) = ∫_{−∞}^{∞} S_X(f) · exp(j2π f τ) df
- **DSP del ruido térmico (forma general):** S_v(f) = 2R h f · coth( h f / (2kT) )   (bilateral)
- **Aproximación clásica:** Si h f ≪ kT, entonces S_v(f) ≈ 2kTR (bilateral)  ⇔  4kTR (unilateral)
- **AWGN (bilateral):** S_N(f) = N₀/2   ;   R_N(τ) = (N₀/2) δ(τ)
- **Salida de un filtro LTI:** Y(t) = ∫_{−∞}^{∞} h(λ) · W(t−λ) dλ
- **AWGN a través de LTI:** S_Y(f) = |H(f)|² (N₀/2)   ;   R_Y(τ) = (N₀/2) ∫_{−∞}^{∞} h(u) h(u+τ) du
- **Modelo en banda base:** r̃(t) = s̃(t) + ñ(t),   con  ñ(t) ~ AWGN complejo

#### Ecuaciones destacadas

$$m_X(t)=\mathbb{E}\{X(t)\}$$
$$\sigma_X^2(t)=\mathbb{E}\{(X(t)-m_X(t))^2\}$$
$$R_X(t_1,t_2)=\mathbb{E}\{X(t_1)X(t_2)\}$$
$$S_X(f)=\int_{-\infty}^{\infty} R_X(\tau)\,e^{-j2\pi f\tau}\,d\tau$$
$$Y(f)=X(f)\,H(f),\qquad S_Y(f)=|H(f)|^2 S_X(f)$$
"""

MATERIALES_TEXT = """
- Dispositivo con acceso a internet
"""

CONCLUSIONES_TEXT = """1. El ruido en telecomunicaciones se modela de forma natural como un proceso estocástico, porque su comportamiento no puede describirse con certeza determinista. A través de las simulaciones se verificó que, para el caso AWGN, la distribución de amplitud es aproximadamente gaussiana y que sus estadísticos de primer orden (media y varianza) permiten caracterizar de manera directa la dispersión y el nivel de incertidumbre que introduce en una señal.
2. La estacionariedad es una propiedad clave para el análisis práctico, ya que cuando la media y la varianza no cambian con el tiempo, la caracterización del proceso se simplifica y herramientas como la autocorrelación y la densidad espectral de potencia (PSD) se vuelven especialmente útiles. Además, se observó que al aplicar un sistema LTI a ruido blanco, la PSD deja de ser plana y pasa a estar moldeada por la respuesta en frecuencia del filtro, dando lugar a ruido “coloreado” con mayor correlación temporal, lo cual impacta directamente en el comportamiento del sistema y en la interpretación de sus señales.
"""


# -----------------------------
# Utilidades numéricas
# -----------------------------

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


def _rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def _autocorr_biased(x: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """Autocorrelación sesgada (normalización por N) de x (demean)."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    N = len(x)
    if N == 0:
        return np.array([0]), np.array([0.0])
    max_lag = int(max(1, min(max_lag, N - 1)))
    r_full = np.correlate(x, x, mode="full") / N
    mid = len(r_full) // 2
    lags = np.arange(-max_lag, max_lag + 1)
    r = r_full[mid - max_lag: mid + max_lag + 1]
    return lags, r


def _moving_mean_var(x: np.ndarray, win: int) -> Tuple[np.ndarray, np.ndarray]:
    """Media y varianza móvil con ventana rectangular de tamaño win."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    win = int(max(3, min(win, N)))
    w = np.ones(win) / win
    mean = np.convolve(x, w, mode="same")
    mean2 = np.convolve(x * x, w, mode="same")
    var = np.maximum(mean2 - mean * mean, 0.0)
    return mean, var


def _periodogram_one_sided(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Periodograma unilateral simple."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N == 0:
        return np.array([0.0]), np.array([0.0])
    x = x - np.mean(x)
    X = np.fft.rfft(x)
    Pxx = (np.abs(X) ** 2) / (fs * N)
    f = np.fft.rfftfreq(N, d=1.0 / fs)
    # Ajuste a unilateral (excepto DC y Nyquist si aplica)
    if N > 1:
        if N % 2 == 0:
            Pxx[1:-1] *= 2.0
        else:
            Pxx[1:] *= 2.0
    return f, Pxx


def _avg_periodogram_one_sided(x_mat: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Promedio de periodogramas para matriz (Nr x N)."""
    x_mat = np.asarray(x_mat, dtype=float)
    Nr, N = x_mat.shape
    acc_f = None
    acc = None
    for i in range(Nr):
        f, Pxx = _periodogram_one_sided(x_mat[i], fs)
        if acc is None:
            acc = Pxx
            acc_f = f
        else:
            acc += Pxx
    acc = acc / max(Nr, 1)
    return acc_f, acc


def _design_fir_filter(fs: float, kind: str, M: int, fc: Optional[float] = None,
                       f0: Optional[float] = None, bw: Optional[float] = None) -> np.ndarray:
    """Diseña FIR con firwin. kind: 'lowpass' o 'bandpass'."""
    if not _SCIPY_OK:
        # Fallback muy simple (promediador)
        M = int(max(5, M))
        return np.ones(M) / M

    M = int(max(5, M))
    if kind == "lowpass":
        if fc is None:
            fc = 300.0
        fc = float(np.clip(fc, 1.0, 0.49 * fs))
        h = firwin(numtaps=M, cutoff=fc, fs=fs, pass_zero=True)
        return h

    if kind == "bandpass":
        if f0 is None:
            f0 = 600.0
        if bw is None:
            bw = 200.0
        f0 = float(f0)
        bw = float(max(20.0, bw))
        f1 = max(1.0, f0 - bw / 2)
        f2 = min(0.49 * fs, f0 + bw / 2)
        if f2 <= f1 + 1e-9:
            f1 = max(1.0, f0 * 0.8)
            f2 = min(0.49 * fs, f0 * 1.2)
        h = firwin(numtaps=M, cutoff=[f1, f2], fs=fs, pass_zero=False)
        return h

    # default
    return np.ones(M) / M


def _apply_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    if _SCIPY_OK and lfilter is not None:
        return lfilter(h, [1.0], x)
    # fallback: convolución
    return np.convolve(x, h, mode="same")


# -----------------------------
# Ejemplos
# -----------------------------

def render_ejemplo1():
    import numpy as np
    import plotly.graph_objects as go
    import streamlit as st

    st.subheader("Ejemplo 1 — Realizaciones, histograma y estimación de parámetros ruido AWGN")
    plot_theme = _get_plot_theme()

    with st.expander("Descripción y pasos a seguir", expanded=True):
        st.markdown(
            "Este ejemplo genera realizaciones de un proceso gaussiano (ruido AWGN) y permite comparar:\n\n" 
            "sea x(t) un proceso gaussiano estacionario:\n"
            "- Una realización (muestra en el tiempo).\n"
            "- Un subconjunto de realizaciones para visualizar variabilidad.\n"
            "- Histograma de una realización vs histograma usando todas las realizaciones**.\n"
            "- Estimaciones de **media** y **varianza**.\n\n"
            "**Pasos sugeridos**\n"
            "1. Ajusta **m**, **σ²** y la **duración**.\n"
            "2. Elige el número de realizaciones **Nᵣ**.\n"
            "3. Selecciona el porcentaje de realizaciones a mostrar.\n"
            "4. Pulsa **Simular** para generar el conjunto."
        )

    # Estado para persistir resultados
    if "g4_e1_state" not in st.session_state:
        st.session_state.g4_e1_state = {
            "ready": False,
            "params": {},
            "t": None,
            "x0": None,
            "X": None,
            "centers": None,
            "hist1": None,
            "hist_all": None,
            "mu_hat_1": None,
            "var_hat_1": None,
            "mu_hat_all": None,
            "var_hat_all": None,
            "show_pct": None,
        }
    state = st.session_state.g4_e1_state

    col1, col2 = st.columns(2)

    with col1:
        mu = st.number_input("Media m", value=0.0, step=0.1, key="g4_e1_mu")

        # >>> Cambio: el usuario controla VARIANZA σ²
        var = st.number_input(
            "Varianza σ²",
            min_value=1e-6,
            value=1.0,
            step=0.1,
            key="g4_e1_var"
        )

        T = st.slider("Duración T (s)", min_value=0.5, max_value=5.0, value=1.0, step=0.5, key="g4_e1_T")
        Nr = st.slider("Número de realizaciones Nᵣ", min_value=1, max_value=200, value=20, step=1, key="g4_e1_Nr")
        show_pct = st.slider(
            "Porcentaje de realizaciones a mostrar (%)",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            key="g4_e1_show_pct"
        )

        fs = 2000.0  # fijo internamente
        N = int(fs * T)

        if st.button("Simular", key="g4_e1_btn"):
            st.session_state.g4_e1_seed = int(np.random.randint(0, 2**31 - 1))
            st.session_state.g4_e1_run = True

    if not st.session_state.get("g4_e1_run"):
        state["ready"] = False
        with col2:
            st.info("Ajusta los parámetros y pulsa **Simular** para generar las gráficas.")
    else:
        sigma = float(np.sqrt(var))  # >>> se usa σ internamente
        seed = int(st.session_state.get("g4_e1_seed", 12345))
        rng = np.random.default_rng(seed)

        X = mu + sigma * rng.standard_normal(size=(Nr, N))
        t = np.arange(N) / fs
        x0 = X[0].copy()

        mu_hat_1 = float(np.mean(x0))
        var_hat_1 = float(np.var(x0))
        mu_hat_all = float(np.mean(X))
        var_hat_all = float(np.var(X))

        bins = 50
        lo = mu - 4.5 * sigma
        hi = mu + 4.5 * sigma
        edges = np.linspace(lo, hi, bins + 1)

        hist1, _ = np.histogram(x0, bins=edges, density=True)
        hist_all, _ = np.histogram(X.reshape(-1), bins=edges, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        state.update({
            "ready": True,
            "params": {"mu": mu, "var": var, "sigma": sigma, "T": T, "Nr": Nr, "fs": fs, "seed": seed},
            "t": t,
            "x0": x0,
            "X": X,
            "centers": centers,
            "hist1": hist1,
            "hist_all": hist_all,
            "mu_hat_1": mu_hat_1,
            "var_hat_1": var_hat_1,
            "mu_hat_all": mu_hat_all,
            "var_hat_all": var_hat_all,
            "show_pct": show_pct,
        })

        st.success("Simulación generada. Revisa las gráficas y la interpretación.")

    if not state["ready"]:
        with col2:
            st.info("Ajusta los parámetros y pulsa **Simular** para generar las gráficas.")
    else:
        t = state["t"]
        x0 = state["x0"]
        X = state["X"]
        centers = state["centers"]
        hist1 = state["hist1"]
        hist_all = state["hist_all"]
        show_pct = state["show_pct"]

        axis_title_font = dict(family="Arial Black", size=14, color=plot_theme["font_color"])
        tick_font = dict(family="Arial Black", size=12, color=plot_theme["font_color"])

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=centers, y=hist1,
            mode="lines",
            line=dict(width=2, color="blue"),
            name="Histograma (1 realización)"
        ))

        fig_hist.add_trace(go.Scatter(
            x=centers, y=hist_all,
            mode="lines",
            line=dict(width=2, color="darkorange"),
            name=f"Histograma (todas, Nᵣ={state['params']['Nr']})"
        ))

        fig_hist.update_layout(
            title=dict(
                text="Histograma(s) (densidad)",
                font=dict(family="Arial Black", size=16, color=plot_theme["font_color"])
            ),
            height=330,
            margin=dict(l=55, r=20, t=65, b=55),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.05,
                xanchor="left",
                x=0,
                font=dict(color=plot_theme["font_color"])
            ),
        )
        fig_hist.update_xaxes(title_text="Valor", title_font=axis_title_font, tickfont=tick_font)
        fig_hist.update_yaxes(title_text="Densidad", title_font=axis_title_font, tickfont=tick_font)
        _apply_plot_theme(fig_hist, plot_theme)
        st.plotly_chart(fig_hist, use_container_width=True, theme=None)

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(x=t, y=x0, mode="lines", line=dict(width=2), name="x(t)"))
            fig_time.update_layout(
                title=dict(
                    text="Una realización x(t)",
                    font=dict(family="Arial Black", size=16, color=plot_theme["font_color"])
                ),
                height=330,
                margin=dict(l=55, r=20, t=65, b=55),
                hovermode="x unified",
                showlegend=False,
            )
            fig_time.update_xaxes(
                title_text="Tiempo (s)",
                title_font=axis_title_font,
                tickfont=tick_font,
                showline=True,
                linewidth=2,
                ticks="outside",
                tickwidth=2,
            )
            fig_time.update_yaxes(
                title_text="Amplitud",
                title_font=axis_title_font,
                tickfont=tick_font,
                showline=True,
                linewidth=2,
                ticks="outside",
                tickwidth=2,
            )
            _apply_plot_theme(fig_time, plot_theme)
            st.plotly_chart(fig_time, use_container_width=True, theme=None)

        with row2_col2:
            fig_subset = go.Figure()
            if X is not None:
                subset_count = max(1, int(np.ceil(X.shape[0] * show_pct / 100)))
                for idx in range(subset_count):
                    fig_subset.add_trace(go.Scatter(
                        x=t,
                        y=X[idx],
                        mode="lines",
                        line=dict(width=1),
                        showlegend=False,
                        hoverinfo="skip"
                    ))

            fig_subset.update_layout(
                title=dict(
                    text=f"Subconjunto de realizaciones ({show_pct}% de Nᵣ)",
                    font=dict(family="Arial Black", size=16, color=plot_theme["font_color"])
                ),
                height=330,
                margin=dict(l=55, r=20, t=65, b=55),
                hovermode=False,
            )
            fig_subset.update_xaxes(
                title_text="Tiempo (s)",
                title_font=axis_title_font,
                tickfont=tick_font,
                showline=True,
                linewidth=2,
                ticks="outside",
                tickwidth=2,
            )
            fig_subset.update_yaxes(
                title_text="Amplitud",
                title_font=axis_title_font,
                tickfont=tick_font,
                showline=True,
                linewidth=2,
                ticks="outside",
                tickwidth=2,
            )
            _apply_plot_theme(fig_subset, plot_theme)
            st.plotly_chart(fig_subset, use_container_width=True, theme=None)

        st.markdown(
            f"**Estimación (1 realización):** m = {state['mu_hat_1']:.3f}, σ̂² = {state['var_hat_1']:.3f}\n\n"
            f"**Estimación (todas las realizaciones):** m = {state['mu_hat_all']:.3f}, σ̂² = {state['var_hat_all']:.3f}"
        )

    # --- Explicación + Preguntas ---
    if state["ready"]:
        p = state["params"]
        mu = p["mu"]
        var = p["var"]
        sigma = p["sigma"]
        Nr = p["Nr"]
        T = p["T"]

        with st.expander("Explicación y preguntas", expanded=True):
            st.markdown("##### Explicación de la simulación")
            st.markdown(
                f"- Se generó ruido gaussiano con **μ = {mu:.2f}** y **σ² = {var:.2f}** "
                f"durante **T = {T:.1f} s**, con **Nᵣ = {Nr}** realizaciones.\n"
                "- La gráfica superior muestra una realización: aunque el modelo estadístico sea el mismo, cada realización cambia.\n"
                "- El subconjunto de realizaciones ilustra la variabilidad entre funciones de muestra; al mostrar más realizaciones se aprecia mejor la tendencia global.\n"
                "- El histograma usando todas las realizaciones tiende a verse más “estable” porque incorpora muchas más muestras."
            )

            st.markdown(
                "En este ejemplo el ruido se modela como **blanco, Gaussiano y aditivo** El termino Gaussiano indica que las muestras del ruido siguen una distribución normal, completamente caracterizada por su media y su varianza. El termino blanco hace referencia a que su densidad espectral de potencia es plana y es aditivo ya que el ruido se suma linealmente a la señal de información. "
                "En ese caso, la media m y la varianza σ² describen completamente la distribución de amplitud en cada instante: "
                "m fija el centro de la distribución y σ² controla la dispersión. "
                "De forma más general, para un proceso gaussiano la descripción completa se determina por  "
                "su función de correlación; y cuando además es blanco , la información esencial de segundo orden "
                "se resume en la varianza. "
                "Al aumentar el número de realizaciones, el histograma converge a la distribución teórica por la **ley de los grandes números**: el promedio sobre muchas muestras reduce la variabilidad de la estimación. "
                "Además, la apariencia de “curva normalizada” se vuelve más clara porque el error de muestreo disminuye aproximadamente como 1/√N, lo que hace que la forma gaussiana se perciba cada vez más suave y estable."
            )

            st.markdown("##### Preguntas y respuestas")
            st.markdown(
                "**1. ¿Por qué el histograma usando todas las realizaciones se ve más suave que el de una sola realización?**  \n"
                "**R:** Porque utiliza muchas más muestras; al aumentar la cantidad de datos, la estimación de la distribución se vuelve menos variable.\n\n"
                "**2. Si aumentas σ² manteniendo m constante, ¿qué cambia principalmente en el histograma?**  \n"
                "**R:** El histograma se ensancha (mayor dispersión), ya que aumenta la probabilidad de observar valores alejados del centro.\n\n"
                "**3. ¿Qué representa m en este modelo de ruido?**  \n"
                "**R:** El valor promedio alrededor del cual fluctúa el ruido. Si μ≈0, el ruido oscila alrededor de cero.\n\n"
                "**4. ¿Por qué es importante el hecho de que el ruido AWGN está caracterizado completamente por su media y varianza?**  \n"
                "**R:** Es crucial porque la media y la varianza definen completamente el comportamiento estadístico del proceso, permitiendo modelar, simular y analizar sistemas de comunicación de manera matemáticamente sencilla y predecible conociendo únicamente dos parametros.  "
            )


def render_ejemplo2():
    import numpy as np
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.subheader("Ejemplo 2 — Proceso estacionario vs no estacionario y función de correlación")
    plot_theme = _get_plot_theme()

    with st.expander("Descripción y pasos a seguir", expanded=True):
        st.markdown(
            "Se comparan procesos:\n"
            "- **Estacionario:** media y varianza constantes.\n"
            "- **No estacionario (media):** la media cambia con el tiempo.\n"
            "- **No estacionario (varianza):** la varianza cambia con el tiempo.\n\n"
            "Se visualiza una realización, estimaciones de **media** y **varianza** en el tiempo, "
            "y la **función de autocorrelación**.\n\n"
            "**Pasos sugeridos**\n"
            "1. Elige el tipo de proceso.\n"
            "2. Ajusta parámetros.\n"
            "3. Pulsa **Simular**.\n"
            "4. Interpreta las gráficas y revisa la retroalimentación."
        )

    # -------------------- Estado persistente --------------------
    if "g4_e2_state" not in st.session_state:
        st.session_state.g4_e2_state = {
            "ready": False,
            "params": {},
            "t": None,
            "x": None,
            "m_hat": None,
            "v_hat": None,
            "tau": None,
            "r": None,
        }
    state = st.session_state.g4_e2_state

    col1, col2 = st.columns(2)

    # -------------------- Controles --------------------
    with col1:
        tipo = st.selectbox(
            "Tipo de proceso",
            ["Estacionario", "No estacionario (media)", "No estacionario (varianza)"],
            key="g4_e2_tipo"
        )

        T = st.slider(
            "Duración T (s)",
            min_value=1.0, max_value=10.0, value=9.0, step=1.0,
            key="g4_e2_T"
        )

        mu0 = st.number_input("Media m", value=0.0, step=0.1, key="g4_e2_mu0")

        # >>> Cambio: el usuario controla varianza σ²
        var0 = st.number_input(
            "Varianza σ²",
            min_value=1e-6,
            value=1.0,
            step=0.1,
            key="g4_e2_var0"
        )

        # Parámetros extra SOLO si es no estacionario
        drift = 0.0
        var_slope = 0.0

        if tipo == "No estacionario (media)":
            drift = st.number_input(
                "Razón de cambio de la media a (por segundo)",
                value=0.80,
                step=0.05,
                key="g4_e2_drift"
            )

        if tipo == "No estacionario (varianza)":
            var_slope = st.number_input(
                "Razón de cambio de la varianza b (por segundo)",
                value=7.0,
                step=0.05,
                key="g4_e2_varslope"
            )

        if st.button("Simular", key="g4_e2_btn"):
            st.session_state.g4_e2_seed = int(np.random.randint(0, 2**31 - 1))
            st.session_state.g4_e2_run = True

    # -------------------- Simulación (se re-ejecuta en cada rerun) --------------------
    if not st.session_state.get("g4_e2_run"):
        state["ready"] = False
    else:
        # fs fijo interno (no es parámetro de estudio aquí)
        fs = 250.0
        N = int(fs * T)
        N = max(N, 2)
        t = np.arange(N) / fs

        seed = int(st.session_state.get("g4_e2_seed", 12345))
        rng = _rng_from_seed(seed)

        # Construir μ(t) y σ²(t)
        if tipo == "Estacionario":
            mu_t = mu0 * np.ones_like(t)
            var_t = var0 * np.ones_like(t)

        elif tipo == "No estacionario (media)":
            mu_t = mu0 + drift * t
            var_t = var0 * np.ones_like(t)

        else:  # No estacionario (varianza)
            mu_t = mu0 * np.ones_like(t)
            var_t = np.maximum(var0 + var_slope * t, 1e-6)

        sig_t = np.sqrt(var_t)

        # Realización
        x = mu_t + sig_t * rng.standard_normal(size=N)

        # Estadísticos móviles (ventana fija ~0.4 s)
        win = max(25, int(0.4 * fs))
        win = min(win, N)  # seguridad si T es pequeña

        # Media móvil siempre sobre x(t)
        m_hat, _ = _moving_mean_var(x, win=win)

        # Varianza móvil:
        # Si la no estacionariedad es por MEDIA, calcula varianza sobre el residuo x - mu(t)
        if "No estacionario (media)" in tipo:
            x_res = x - mu_t
            _, v_hat = _moving_mean_var(x_res, win=win)
        else:
            _, v_hat = _moving_mean_var(x, win=win)

        # --- FIX borde: anula los extremos donde la ventana queda "incompleta" ---
        k = max(1, win // 2)
        m_hat[:k] = np.nan
        m_hat[-k:] = np.nan
        v_hat[:k] = np.nan
        v_hat[-k:] = np.nan

        # Autocorrelación (máx ~0.8 s)
        max_lag = int(0.8 * fs)
        max_lag = min(max_lag, N - 1)
        lags, r = _autocorr_biased(x, max_lag=max_lag)
        tau = lags / fs

        state.update({
            "ready": True,
            "params": {
                "tipo": tipo,
                "T": T,
                "fs": fs,
                "mu0": float(mu0),
                "var0": float(var0),
                "sigma0": float(np.sqrt(var0)),
                "drift": float(drift),
                "var_slope": float(var_slope),
                "seed": seed,
                "win_s": win / fs,
                "maxlag_s": max_lag / fs
            },
            "t": t,
            "x": x,
            "m_hat": m_hat,
            "v_hat": v_hat,
            "tau": tau,
            "r": r
        })

        st.success("Simulación generada. Revisa las gráficas y la retroalimentación.")

    # -------------------- Gráficas (solo si ready) --------------------
    if not state["ready"]:
        with col2:
            st.info("Ajusta los parámetros y pulsa **Simular** para generar las gráficas.")
    else:
        p = state["params"]
        t = state["t"]
        x = state["x"]
        m_hat = state["m_hat"]
        v_hat = state["v_hat"]
        tau = state["tau"]
        r = state["r"]

        axis_title_font = dict(family="Arial Black", size=14, color=plot_theme["font_color"])
        tick_font = dict(family="Arial Black", size=12, color=plot_theme["font_color"])

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=t, y=x, mode="lines", line=dict(width=2), name="x(t)"))
        fig1.update_layout(
            title=dict(text="Realización x(t)", font=dict(family="Arial Black", size=16, color=plot_theme["font_color"])),
            height=320,
            margin=dict(l=55, r=20, t=65, b=55),
            hovermode="x unified",
            showlegend=False,
        )
        fig1.update_xaxes(
            title_text="Tiempo (s)",
            title_font=axis_title_font,
            tickfont=tick_font,
            showline=True, linewidth=2, linecolor="black",
            ticks="outside", tickwidth=2,
            rangeslider=dict(visible=True)
        )
        fig1.update_yaxes(
            title_text="Amplitud",
            title_font=axis_title_font,
            tickfont=tick_font,
            showline=True, linewidth=2, linecolor="black",
            ticks="outside", tickwidth=2
        )
        _apply_plot_theme(fig1, plot_theme)
        st.plotly_chart(fig1, use_container_width=True, theme=None)

        row2_col1, row2_col2 = st.columns(2)

        with row2_col1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=t, y=m_hat, mode="lines", line=dict(width=2, color="blue"), name="Media"))
            fig2.add_trace(go.Scatter(x=t, y=v_hat, mode="lines", line=dict(width=2, color="firebrick"), name="Varianza"))
            fig2.update_layout(
                title=dict(
                    text="Media y varianza estimadas en el tiempo",
                    font=dict(family="Arial Black", size=16, color=plot_theme["font_color"])
                ),
                height=320,
                margin=dict(l=55, r=20, t=65, b=55),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="left",
                    x=0,
                    font=dict(color=plot_theme["font_color"])
                ),
            )
            fig2.update_xaxes(
                title_text="Tiempo (s)",
                title_font=axis_title_font,
                tickfont=tick_font,
                showline=True, linewidth=2, linecolor="black",
                ticks="outside", tickwidth=2,
                rangeslider=dict(visible=True)
            )
            fig2.update_yaxes(
                title_text="Valor",
                title_font=axis_title_font,
                tickfont=tick_font,
                showline=True, linewidth=2, linecolor="black",
                ticks="outside", tickwidth=2
            )
            _apply_plot_theme(fig2, plot_theme)
            st.plotly_chart(fig2, use_container_width=True, theme=None)

        with row2_col2:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=tau, y=r, mode="lines", line=dict(width=2), name="R̂(τ)"))
            fig3.update_layout(
                title=dict(
                    text="Autocorrelación estimada",
                    font=dict(family="Arial Black", size=16, color=plot_theme["font_color"])
                ),
                height=320,
                margin=dict(l=55, r=20, t=65, b=55),
                hovermode="x unified",
                showlegend=False,
            )
            fig3.update_xaxes(
                title_text="Retardo τ (s)",
                title_font=axis_title_font,
                tickfont=tick_font,
                showline=True, linewidth=2, linecolor="black",
                ticks="outside", tickwidth=2
            )
            fig3.update_yaxes(
                title_text="R̂(τ)",
                title_font=axis_title_font,
                tickfont=tick_font,
                showline=True, linewidth=2, linecolor="black",
                ticks="outside", tickwidth=2
            )
            _apply_plot_theme(fig3, plot_theme)
            st.plotly_chart(fig3, use_container_width=True, theme=None)

    # -------------------- Explicación + comparación + Q&A --------------------
    if state["ready"]:
        p = state["params"]

        with st.expander("Explicación y preguntas", expanded=True):
            st.markdown("##### Explicación de la simulación")
            if p["tipo"] == "Estacionario":
                st.markdown(
                    f"- Se generó un proceso con m constante = {p['mu0']:.2f}** y **σ² constante = {p['var0']:.2f}**.\n"
                    f"- En la segunda gráfica se muestra como evolucionan en el tiempo la **media** y la **varianza**.\n"
                    "- La correlación presenta un pico dominante en **τ=0** y cae rápidamente si el proceso es cercano a blanco."
                )
            elif p["tipo"] == "No estacionario (media)":
                st.markdown(
                    f"- Se generó un proceso con **media variable**: m(t)=m₀+a·t, con m₀={p['mu0']:.2f} y a={p['drift']:.2f}.\n"
                    f"- La **varianza** permanece aproximadamente constante en **σ²={p['var0']:.2f}**.\n"
                    "- En la segunda gráfica se observa claramente la razón de cambio de la media, lo cual es evidencia de no estacionariedad."
                )
            else:
                st.markdown(
                    f"- Se generó un proceso con **varianza variable**: σ²(t)=σ₀²+b·t, con σ₀²={p['var0']:.2f} y b={p['var_slope']:.2f}.\n"
                    f"- La **media** se mantiene aproximadamente constante en **m={p['mu0']:.2f}**.\n"
                    "- En la segunda gráfica se observa que la varianza móvil crece (la señal se vuelve más dispersa)."
                )

            st.markdown("##### Diferencias clave: estacionario vs no estacionario")
            st.markdown(
                "- **Estacionario:** sus estadísticos (media y correlación) **no cambian con el tiempo**. "
                "Esto facilita el modelado, la estimación y el diseño de receptores.\n"
                "- **No estacionario:** los estadísticos **dependen del tiempo** (m(t), σ²(t), correlación variable). "
                "En la práctica exige herramientas más complejas: segmentación por intervalos, modelos dependientes del tiempo, "
                "y estimación adaptativa.\n"
                "- En telecomunicaciones, el **ruido térmico** se aproxima muy bien como **AWGN estacionario** en bandas y ventanas "
                "de observación típicas, lo que hace el análisis **mucho más simple** y permite obtener métricas como BER/SNR de forma directa."
            )

            st.markdown("##### Preguntas y respuestas")
            st.markdown(
                "**1. ¿Cómo identificas que un proceso es no estacionario en estas gráficas?**  \n"
                "**R:** Porque la media y/o la varianza cambian con el tiempo.\n\n"
                "**2. ¿Por qué un proceso no estacionario es más difícil de tratar que uno estacionario?**  \n"
                "**R:** Porque sus estadísticas dependen del tiempo; no basta una sola media/varianza global y el modelo debe adaptarse a la evolución temporal.\n\n"
                "**3. ¿Por qué modelar el ruido como AWGN estacionario simplifica el diseño del receptor?**  \n"
                "**R:** Porque con estadísticos constantes (y modelo gaussiano), se pueden derivar umbrales/detectores óptimos y predecir desempeño (p.ej., BEP) con mucha menos complejidad.\n\n"
                "**4. ¿Que representa la función de correlación de un proceso estocástico?**  \n"
                "**R:** Es una función estadística de segundo orden que mide el grado de dependencia o similitud entre los valores del proceso en dos instantes distintos.. La manera en que las muestras se relacionan entre si siguen un patrón bien definido y este patrón está muy bien representado por la función de correlación RX(Τ), la cual contiene toda la información necesaria para caracterizar la dependencia estadística entre muestras y, en el caso de un proceso estocástico Gaussiano estacionario, determina por completo su estructura estadística. También como se verá en el proximo ejemplo, un análisis util en telecomunicaciones es el análisis de frecuencia, este análisis esta descrito por la función de densidad espectral del proceso, esta función de densidad se obtiene al realizar la transformada de Fourier de la función de correlación \n\n"
                " Por ejemplo, en el caso de ruido AWGN, el cual es estacionario, se observa que la correlación alcanza su máximo en T=0 y decae conforme aumenta el retardo. Esto tiene todo el sentido ya que se indica que en T=0 una muestra del proceso está perfectamente correlacionada consigo misma "
            )


def render_ejemplo3():
    st.subheader("Ejemplo 3 — Ruido blanco, ruido coloreado, función de correlación y función de densidad espectral")

    # Requiere Plotly
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        _PLOTLY_OK = True
    except Exception:
        _PLOTLY_OK = False

    if not _PLOTLY_OK:
        st.error("Plotly no está disponible. Instala plotly para usar gráficas interactivas.")
        return

    if not _SCIPY_OK:
        st.warning("SciPy no está disponible. Se usará un filtro FIR simple como alternativa (promediador).")

    with st.expander("Descripción y pasos a seguir", expanded=True):
        st.markdown(
            "Se genera un **ruido AWGN** w(t) y se filtra con un sistema LTI para obtener ruido **coloreado** n(t).\n\n"
            "Se visualiza:\n"
            "- Realización en el tiempo (salida).\n"
            "- Función de correlación estimada.\n"
            "- Función de densidad espectral (PSD) estimada y PSD promedio sobre Nr realizaciones.\n"
            "- Histograma del proceso de el carácter **gaussiano**.\n\n"
            "La **densidad espectral de potencia** Sx(f) se obtiene como la Transformada de Fourier de la **función de correlación** Rx(τ) (Teorema de Wiener–Khinchin).\n"
        )
        st.latex(r"S_x(f)=\mathcal{F}\{R_x(\tau)\}")

    col1, col2 = st.columns(2)

    with col1:
        tipo_filtro = st.selectbox("Tipo de filtro", ["Pasa bajas", "Pasa banda"], key="g4_e3_tipo")

        # Pedir MEDIA m y VARIANZA sigma^2
        m = st.number_input("Media m", value=0.0, step=0.1, key="g4_e3_m")
        var = st.number_input("Varianza σ²", min_value=0.001, value=1.0, step=0.1, key="g4_e3_var")

        T = st.slider("Duración T (s)", min_value=0.5, max_value=4.0, value=1.5, step=0.5, key="g4_e3_T")
        Nr = st.slider("Número de realizaciones para PSD promedio Nᵣ", min_value=1, max_value=300, value=30, step=1, key="g4_e3_Nr")

        fs = 2000.0  # fijo interno
        M = st.slider("Orden del filtro M", min_value=11, max_value=101, value=41, step=2, key="g4_e3_M")

        fc = None
        f0 = None
        bw = None
        if tipo_filtro == "Pasa bajas":
            fc = st.slider(
                "Frecuencia de corte fc (Hz)",
                min_value=50, max_value=int(0.45 * fs), value=300, step=25, key="g4_e3_fc"
            )
        else:
            f0 = st.slider(
                "Frecuencia central f0 (Hz)",
                min_value=100, max_value=int(0.45 * fs), value=600, step=25, key="g4_e3_f0"
            )
            bw = st.slider(
                "Ancho de banda BW (Hz)",
                min_value=50, max_value=800, value=250, step=25, key="g4_e3_bw"
            )

        if st.button("Simular", key="g4_e3_btn"):
            st.session_state.g4_e3_seed = int(np.random.randint(0, 2**31 - 1))
            st.session_state.g4_e3_run = True

    # Se re-ejecuta en cada rerun usando los valores actuales de los widgets
    if st.session_state.get("g4_e3_run"):
        seed = int(st.session_state.get("g4_e3_seed", 12345))
        rng = _rng_from_seed(seed)

        sigma = float(np.sqrt(var))
        N = int(fs * T)
        t = np.arange(N) / fs

        # Realizaciones (Nr x N): w(t) = m + sigma * N(0,1)
        W = m + sigma * rng.standard_normal(size=(Nr, N))

        # Diseñar filtro FIR
        if tipo_filtro == "Pasa bajas":
            h = _design_fir_filter(fs=fs, kind="lowpass", M=M, fc=float(fc))
            etiqueta_filtro = f"Pasa bajas (f_c={fc} Hz)"
        else:
            h = _design_fir_filter(fs=fs, kind="bandpass", M=M, f0=float(f0), bw=float(bw))
            etiqueta_filtro = f"Pasa banda (f0={f0} Hz, BW={bw} Hz)"

        # Filtrar: n(t) = w(t) * h(t)
        Nn = np.zeros_like(W)
        for i in range(Nr):
            Nn[i] = _apply_fir(W[i], h)

        # Autocorrelación (una realización)
        lags_w, rw = _autocorr_biased(W[0] - np.mean(W[0]), max_lag=int(0.8 * fs))
        lags_n, rn = _autocorr_biased(Nn[0] - np.mean(Nn[0]), max_lag=int(0.8 * fs))
        tau = lags_w / fs

        # PSD (una y promedio)
        f_w, P_w = _periodogram_one_sided(W[0] - np.mean(W[0]), fs)
        f_n, P_n = _periodogram_one_sided(Nn[0] - np.mean(Nn[0]), fs)

        f_wm, P_wm = _avg_periodogram_one_sided(W - np.mean(W, axis=1, keepdims=True), fs)
        f_nm, P_nm = _avg_periodogram_one_sided(Nn - np.mean(Nn, axis=1, keepdims=True), fs)

        # Guardar resultados para que no se regeneren solos al mover sliders
        st.session_state.g4_e3_data = {
            "t": t,
            "W0": W[0],
            "N0": Nn[0],
            "tau": tau,
            "rw": rw,
            "rn": rn,
            "fw": f_w,
            "Pn": P_n,
            "Pw": P_w,
            "fwm": f_wm,
            "Pwm": P_wm,
            "fnm": f_nm,
            "Pnm": P_nm,
            "etiqueta": etiqueta_filtro,
            "Nr": Nr,
            "m": m,
            "var": var,
        }

    data = st.session_state.get("g4_e3_data", None)
    if data is None:
        st.info("Configura los parámetros y pulsa **Simular** para generar las gráficas.")
        return



    plot_theme = _get_plot_theme()

    # --------- PLOTLY (siempre legible según tema) ---------
    def _plotly_layout(fig, title):
        fig.update_layout(
            title=dict(text=title, font=dict(color=plot_theme["font_color"])),
            height=420,
            margin=dict(l=50, r=20, t=60, b=50),
            hovermode="x unified",
            showlegend=True,
            # ✅ FIX hover negro:
            hoverlabel=dict(
                bgcolor=plot_theme["hover_bg"],
                font=dict(color=plot_theme["hover_font"]),
                bordercolor=plot_theme["axis_color"]
            ),
        )
        _apply_plot_theme(fig, plot_theme)

    c_white = "blue"  # ruido blanco
    c_colored = "orange"  # ruido coloreado

    # 1) Tiempo: entrada vs salida
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=data["t"], y=data["W0"],
        mode="lines",
        name="w(t) (blanco)",
        line=dict(color=c_white)
    ))
    fig1.add_trace(go.Scatter(
        x=data["t"], y=data["N0"],
        mode="lines",
        name="n(t) (coloreado)",
        line=dict(color=c_colored)
    ))
    fig1.update_xaxes(title_text="Tiempo (s)", rangeslider_visible=True)
    fig1.update_yaxes(title_text="Amplitud")
    _plotly_layout(fig1, f"Ruido en el tiempo — {data['etiqueta']}")
    st.plotly_chart(fig1, use_container_width=True, theme=None)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=data["fw"], y=(data["Pw"] + 1e-18),
            mode="lines",
            name="PSD w(t) (1 realización)",
            line=dict(color=c_white)
        ))
        fig3.add_trace(go.Scatter(
            x=data["fw"], y=(data["Pn"] + 1e-18),
            mode="lines",
            name="PSD n(t) (1 realización)",
            line=dict(color=c_colored)
        ))
        fig3.update_xaxes(title_text="Frecuencia (Hz)")
        fig3.update_yaxes(title_text="S(f) (u.a.)", type="log")
        _plotly_layout(fig3, "PSD estimada — 1 realización")
        st.plotly_chart(fig3, use_container_width=True, theme=None)

    with row2_col2:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=data["fwm"], y=(data["Pwm"] + 1e-18),
            mode="lines",
            name=f"PSD w(t) promedio (Nᵣ={data['Nr']})",
            line=dict(color=c_white)
        ))
        fig4.add_trace(go.Scatter(
            x=data["fnm"], y=(data["Pnm"] + 1e-18),
            mode="lines",
            name=f"PSD n(t) promedio (Nᵣ={data['Nr']})",
            line=dict(color=c_colored)
        ))
        fig4.update_xaxes(title_text="Frecuencia (Hz)")
        fig4.update_yaxes(title_text="S(f) (u.a.)", type="log")
        _plotly_layout(fig4, "PSD promedio")
        st.plotly_chart(fig4, use_container_width=True, theme=None)

    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=data["tau"], y=data["rw"],
            mode="lines",
            name="R̂w(τ)",
            line=dict(color=c_white)
        ))
        fig2.add_trace(go.Scatter(
            x=data["tau"], y=data["rn"],
            mode="lines",
            name="R̂n(τ)",
            line=dict(color=c_colored)
        ))
        fig2.update_xaxes(title_text="Retardo τ (s)")
        fig2.update_yaxes(title_text="Función de correlación")
        _plotly_layout(fig2, "Función de correlación estimada")
        st.plotly_chart(fig2, use_container_width=True, theme=None)

    with row3_col2:
        fig5 = go.Figure()
        fig5.add_trace(go.Histogram(
            x=data["W0"], nbinsx=40,
            name="Hist w(t)",
            opacity=0.6,
            marker=dict(color=c_white)
        ))
        fig5.add_trace(go.Histogram(
            x=data["N0"], nbinsx=40,
            name="Hist n(t)",
            opacity=0.6,
            marker=dict(color=c_colored)
        ))
        fig5.update_layout(barmode="overlay")
        fig5.update_xaxes(title_text="Valor")
        fig5.update_yaxes(title_text="Frecuencia")
        _plotly_layout(fig5, "Histograma ")
        st.plotly_chart(fig5, use_container_width=True, theme=None)

    # --------- Explicación + ideas clave ---------
    with st.expander("Explicación y preguntas", expanded=True):
        st.markdown("##### Explicación de la simulación")

        st.latex(r"\textbf{Wiener–Khinchin:}\quad S_x(f)=\mathcal{F}\{R_x(\tau)\}")
        st.markdown(
            "La densidad espectral (PSD) de potencia de un proceso estocástico estacionario describe la forma en que la potencia promedio del proceso se distribuye en función de la frecuencia \n\n "
            "Es posible calcular esta densidad espectral de potencia a través de la transformada de Fourier de la función de correlación del proceso, esta afirmación se conoce como el teorema de Wiener-Khinchin\n\n"
            "El ruido blanco Gaussiano aditivo (AWGN) es un modelo ideal ampliamente utilizado para analizar el desempeño de sistemas de telecomunicaciones. Este ruido se caracteriza por ser un proceso estocástico Gaussiano de media cero, estacionario y con una densidad espectral de potencia constante a lo largo de las frecuencias de interés. \n\n"
            "El ruido blanco Gaussiano aditivo tiene una densidad espectral de potencia constante como se define en la ecuación \n\n"
            "Sn(f)=(N0/2)|H(f)|^{2}\n\n Donde N0 representa la densidad espectral de potencia total del ruido  y H es el filtro del receptor \n\n"
            "Ruido coloreado: Al filtrar el ruido blanco con un sistema LTI no diseñado para ruido blanco, se introduce correlación temporal (R̂n(τ) se “ensancha”) y la PSD deja de ser plana.\n\n"
            "De este ejemplo se puede concluir que:\n\n"
            "-  La función de correlación caracteriza completamente cualquier conjunto de funciones de densidad de las muestras del proceso X(t).\n\n"
            "- El ruido térmico se puede modelar como un proceso estocástico estacionario Gaussiano para el análisis de desempeño en sistemas de telecomunicaciones "


        )

        st.markdown("##### Diferencia clave (blanco vs coloreado)")
        st.markdown(
            "- **Blanco:** muestras casi no correlacionadas : R(τ) cae rápido y Sn(f) es casi plana.\n"
            "- **Coloreado:** el filtro concentra energía en ciertas bandas : Sn(f) tiene forma y R(τ) se extiende en el tiempo.\n"
            "- **Gaussiano y sistema LTI:** si la entrada es gaussiana, la salida sigue siendo gaussiana por linealidad."
        )

        # --------- Preguntas y respuestas ---------
        st.markdown("##### Preguntas y respuestas")
        st.markdown("**1) ¿Por qué el ruido blanco se asocia a una PSD “plana”?**")
        st.markdown("**R:** Porque su potencia se distribuye de forma aproximadamente uniforme en frecuencia , por eso el espectro tiende a verse “horizontal”.")

        st.markdown("**2) ¿Qué indica que el ruido de salida es “coloreado”?**")
        st.markdown("**R:** Que la PSD ya no es plana: el filtro favorece ciertas frecuencias y atenúa otras. Además, la función de correlación se hace más ancha.")

        st.markdown("**3) ¿Qué teorema conecta la función correlación y PSD?**")
        st.markdown("**R:** El teorema de Wiener–Khinchin: la PSD es la Transformada de Fourier de la autocorrelación, \(S_x(f)=\\mathcal{F}\\{R_x(\\tau)\\}\\).")

# -----------------------------
# Dinámicas (casos aleatorios + un solo botón de envío)
# -----------------------------

def _plotly_layout_dyn(fig, title, height=380, showlegend=True, y_log=False, rangeslider=False):
    plot_theme = _get_plot_theme()
    fig.update_layout(
        title=dict(text=title, font=dict(color=plot_theme["font_color"])),
        height=height,
        margin=dict(l=55, r=20, t=60, b=55),
        hovermode="x unified",
        showlegend=showlegend,
        # FIX tooltip negro
        hoverlabel=dict(
            bgcolor=plot_theme["hover_bg"],
            font=dict(color=plot_theme["hover_font"]),
            bordercolor=plot_theme["axis_color"]
        )
    )
    _apply_plot_theme(fig, plot_theme)
    fig.update_xaxes(rangeslider_visible=rangeslider)
    fig.update_yaxes(type="log" if y_log else "linear")
    return fig



def _init_guia4_state():
    if "guia4_dinamicas" not in st.session_state:
        st.session_state.guia4_dinamicas = {
            "student": {"name": "", "id": "", "dob": ""},
            "dyn1": {"seed": None, "key": None},
            "dyn2": {"seed": None, "key": None},
            "dyn3": {"seed": None, "key": None},
        }


def _gen_dyn1_key(seed: int) -> Dict[str, Any]:
    rng = _rng_from_seed(seed)
    mu = float(rng.uniform(-0.5, 0.5))
    sigma = float(rng.uniform(0.2, 1.5))
    T = float(rng.choice([1.0, 1.5, 2.0]))
    fs = 2000.0
    N = int(fs * T)
    x = mu + sigma * rng.standard_normal(size=N)
    mu_hat = float(np.mean(x))
    var_hat = float(np.var(x))

    # Respuestas correctas
    if abs(mu_hat) < 0.15:
        mu_cat = "Cercana a 0"
    elif mu_hat > 0:
        mu_cat = "Positiva"
    else:
        mu_cat = "Negativa"

    return {
        "mu": mu, "sigma": sigma, "T": T, "fs": fs, "N": N,
        "mu_hat": mu_hat, "var_hat": var_hat,
        "correct": {
            "q1": "Gaussiana",
            "q2": mu_cat,
            "q3": "Aumenta",
        },
    }


def _render_dyn1():
    st.markdown("### Dinámica 1 — Ruido gaussiano: histograma y parámetros")

    state = st.session_state.guia4_dinamicas
    if state["dyn1"]["seed"] is None:
        state["dyn1"]["seed"] = int(np.random.randint(0, 2**31 - 1))
        state["dyn1"]["key"] = _gen_dyn1_key(state["dyn1"]["seed"])

    colA, colB = st.columns([1, 1])

    with colA:
        if st.button("Nuevo caso (Dinámica 1)", key="g4_dyn1_new"):
            state["dyn1"]["seed"] = int(np.random.randint(0, 2**31 - 1))
            state["dyn1"]["key"] = _gen_dyn1_key(state["dyn1"]["seed"])
            # limpiar respuestas
            for k in ["g4_dyn1_q1", "g4_dyn1_q2", "g4_dyn1_q3"]:
                if k in st.session_state:
                    del st.session_state[k]

    key = state["dyn1"]["key"]
    rng = _rng_from_seed(state["dyn1"]["seed"])
    x = key["mu"] + key["sigma"] * rng.standard_normal(size=key["N"])
    t = np.arange(key["N"]) / key["fs"]

    with colB:
        st.markdown(
            f"**Caso:** μ≈{key['mu']:.2f}, σ≈{key['sigma']:.2f}, T={key['T']:.1f}s  \n"
            f"Estimación: μ̂={key['mu_hat']:.2f}, σ̂²={key['var_hat']:.2f}"
        )

    # --- Plotly: Realización del ruido (interactiva) ---
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=t, y=x,
        mode="lines",
        name="x(t)",
        line=dict(color="blue")
    ))
    fig1.update_xaxes(title_text="Tiempo (s)", rangeslider_visible=True)
    fig1.update_yaxes(title_text="Amplitud")
    _plotly_layout_dyn(fig1, "Realización del ruido", height=380, showlegend=False, rangeslider=True)
    st.plotly_chart(fig1, use_container_width=True, theme=None)

    # --- Plotly: Histograma (interactivo) ---
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=x,
        nbinsx=40,
        histnorm="probability density",
        name="Histograma",
        marker=dict(color="blue"),
        opacity=0.85
    ))
    fig2.update_xaxes(title_text="Valor")
    fig2.update_yaxes(title_text="Densidad")
    _plotly_layout_dyn(fig2, "Histograma del ruido", height=380, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True, theme=None)

    st.markdown("#### Preguntas")
    q1 = st.radio(
        "1) La forma del histograma sugiere que el ruido sigue una distribución:",
        ["Gaussiana", "Uniforme", "Exponencial"],
        index=None,
        key="g4_dyn1_q1",
    )
    q2 = st.radio(
        "2) Según la estimación μ̂ mostrada, la media es:",
        ["Positiva", "Negativa", "Cercana a 0"],
        index=None,
        key="g4_dyn1_q2",
    )
    q3 = st.radio(
        "3) Si σ aumentara, el histograma típicamente:",
        ["Aumenta", "Disminuye", "Se mantiene"],
        index=None,
        key="g4_dyn1_q3",
    )

    answers = {"q1": q1, "q2": q2, "q3": q3}
    return answers, key["correct"], {"mu": key["mu"], "sigma": key["sigma"], "T": key["T"], "mu_hat": key["mu_hat"], "var_hat": key["var_hat"]}


def _gen_dyn2_key(seed: int) -> Dict[str, Any]:
    rng = _rng_from_seed(seed)
    tipo = rng.choice(["Estacionario", "No estacionario (media)", "No estacionario (varianza)"])
    mu0 = float(rng.uniform(-0.3, 0.3))
    sigma0 = float(rng.uniform(0.4, 1.2))
    T = float(rng.choice([4.0, 6.0, 8.0]))
    drift = float(rng.uniform(-0.8, 0.8))
    varslope = float(rng.uniform(0.2, 1.0))

    if tipo == "Estacionario":
        correct_q1 = "Estacionario"
        correct_q2 = "Ninguna (es estacionario)"
        correct_q3 = "Se mantiene"
    elif "media" in tipo:
        correct_q1 = "No estacionario"
        correct_q2 = "Media"
        correct_q3 = "Se mantiene"
    else:
        correct_q1 = "No estacionario"
        correct_q2 = "Varianza"
        correct_q3 = "Aumenta"

    return {
        "tipo": tipo, "mu0": mu0, "sigma0": sigma0, "T": T,
        "drift": drift, "varslope": varslope,
        "correct": {"q1": correct_q1, "q2": correct_q2, "q3": correct_q3},
    }


def _render_dyn2():
    st.markdown("### Dinámica 2 — Estacionariedad: media/varianza en el tiempo")

    state = st.session_state.guia4_dinamicas
    if state["dyn2"]["seed"] is None:
        state["dyn2"]["seed"] = int(np.random.randint(0, 2**31 - 1))
        state["dyn2"]["key"] = _gen_dyn2_key(state["dyn2"]["seed"])

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Nuevo caso (Dinámica 2)", key="g4_dyn2_new"):
            state["dyn2"]["seed"] = int(np.random.randint(0, 2**31 - 1))
            state["dyn2"]["key"] = _gen_dyn2_key(state["dyn2"]["seed"])
            for k in ["g4_dyn2_q1", "g4_dyn2_q2", "g4_dyn2_q3"]:
                if k in st.session_state:
                    del st.session_state[k]

    key = state["dyn2"]["key"]
    with colB:
        st.markdown(f"**Caso generado:** {key['tipo']}")

    fs = 250.0
    N = int(fs * key["T"])
    t = np.arange(N) / fs
    rng = _rng_from_seed(state["dyn2"]["seed"])

    if key["tipo"] == "Estacionario":
        mu_t = key["mu0"] * np.ones_like(t)
        sig_t = key["sigma0"] * np.ones_like(t)
    elif "media" in key["tipo"]:
        mu_t = key["mu0"] + key["drift"] * t
        sig_t = key["sigma0"] * np.ones_like(t)
    else:
        mu_t = key["mu0"] * np.ones_like(t)
        sig_t = np.maximum(key["sigma0"] + key["varslope"] * t, 0.05)

    x = mu_t + sig_t * rng.standard_normal(size=N)

    win = max(25, int(0.4 * fs))
    win = min(win, N)

    # Media móvil (siempre sobre x)
    m_hat, _ = _moving_mean_var(x, win=win)

    # Varianza móvil:
    # Si el caso es "No estacionario (media)", calcula varianza sobre residuo x - mu_t (para que sea ~constante)
    if "media" in key["tipo"]:
        x_res = x - mu_t
        _, v_hat = _moving_mean_var(x_res, win=win)
    else:
        _, v_hat = _moving_mean_var(x, win=win)

    # --- Plotly: Realización ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=x,
        mode="lines",
        name="x(t)",
        line=dict(color="blue")
    ))
    fig.update_xaxes(title_text="Tiempo (s)", rangeslider_visible=True)
    fig.update_yaxes(title_text="Amplitud")
    _plotly_layout_dyn(fig, "Realización x(t)", height=380, showlegend=False, rangeslider=True)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    # --- Plotly: Media y Varianza ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=t, y=m_hat,
        mode="lines",
        name="Media",
        line=dict(color="blue")
    ))
    fig2.add_trace(go.Scatter(
        x=t, y=v_hat,
        mode="lines",
        name="Varianza",
        line=dict(color="orange")
    ))
    fig2.update_xaxes(title_text="Tiempo (s)", rangeslider_visible=True)
    fig2.update_yaxes(title_text="Valor")
    _plotly_layout_dyn(fig2, "Media y varianza estimadas", height=380, showlegend=True, rangeslider=True)
    st.plotly_chart(fig2, use_container_width=True, theme=None)

    st.markdown("#### Preguntas")
    q1 = st.radio(
        "1) Según las curvas estimadas, el proceso es:",
        ["Estacionario", "No estacionario"],
        index=None,
        key="g4_dyn2_q1",
    )
    q2 = st.radio(
        "2) Si es no estacionario, ¿qué cambia principalmente?",
        ["Media", "Varianza", "Ninguna (es estacionario)"],
        index=None,
        key="g4_dyn2_q2",
    )
    q3 = st.radio(
        "3) La varianza estimada en el tiempo:",
        ["Aumenta", "Disminuye", "Se mantiene"],
        index=None,
        key="g4_dyn2_q3",
    )

    answers = {"q1": q1, "q2": q2, "q3": q3}
    return answers, key["correct"], {
        "tipo": key["tipo"], "mu0": key["mu0"], "sigma0": key["sigma0"],
        "T": key["T"], "drift": key["drift"], "varslope": key["varslope"]
    }


def _gen_dyn3_key(seed: int) -> Dict[str, Any]:
    rng = _rng_from_seed(seed)
    tipo = rng.choice(["Pasa bajas", "Pasa banda"])
    sigma = float(rng.uniform(0.3, 1.2))
    fs = 2000.0
    T = float(rng.choice([1.0, 1.5, 2.0]))
    M = int(rng.choice([21, 33, 51]))
    if tipo == "Pasa bajas":
        fc = float(rng.uniform(150.0, 600.0))
        correct_q2 = "Pasa bajas"
    else:
        f0 = float(rng.uniform(300.0, 900.0))
        bw = float(rng.uniform(100.0, 400.0))
        correct_q2 = "Pasa banda"

    return {
        "tipo": tipo, "sigma": sigma, "fs": fs, "T": T, "M": M,
        "fc": locals().get("fc", None),
        "f0": locals().get("f0", None),
        "bw": locals().get("bw", None),
        "correct": {"q1": "Blanco", "q2": correct_q2},  # q3 se calcula por pico
    }


def _render_dyn3():
    st.markdown("### Dinámica 3 — PSD de ruido filtrado (ruido coloreado)")

    state = st.session_state.guia4_dinamicas
    if state["dyn3"]["seed"] is None:
        state["dyn3"]["seed"] = int(np.random.randint(0, 2**31 - 1))
        state["dyn3"]["key"] = _gen_dyn3_key(state["dyn3"]["seed"])

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Nuevo caso (Dinámica 3)", key="g4_dyn3_new"):
            state["dyn3"]["seed"] = int(np.random.randint(0, 2**31 - 1))
            state["dyn3"]["key"] = _gen_dyn3_key(state["dyn3"]["seed"])
            for k in ["g4_dyn3_q1", "g4_dyn3_q2", "g4_dyn3_q3"]:
                if k in st.session_state:
                    del st.session_state[k]

    key = state["dyn3"]["key"]
    with colB:
        desc = key["tipo"]
        if key["tipo"] == "Pasa bajas":
            desc += f" (f_c={key['fc']} Hz)"
        else:
            desc += f" (f0={key['f0']} Hz, BW={key['bw']} Hz)"
        st.markdown(f"**Caso generado:** {desc}")

    fs = key["fs"]
    N = int(fs * key["T"])
    t = np.arange(N) / fs
    rng = _rng_from_seed(state["dyn3"]["seed"])

    Nr_psd = 40
    W = key["sigma"] * rng.standard_normal(size=(Nr_psd, N))

    if key["tipo"] == "Pasa bajas":
        h = _design_fir_filter(fs=fs, kind="lowpass", M=key["M"], fc=float(key["fc"]))
    else:
        h = _design_fir_filter(fs=fs, kind="bandpass", M=key["M"], f0=float(key["f0"]), bw=float(key["bw"]))

    Nn = np.zeros_like(W)
    for i in range(Nr_psd):
        Nn[i] = _apply_fir(W[i], h)

    # PSD promedio
    f_wm, P_wm = _avg_periodogram_one_sided(W, fs)
    f_nm, P_nm = _avg_periodogram_one_sided(Nn, fs)

    # Pico (excluyendo DC)
    idx_peak = int(np.argmax(P_nm[1:]) + 1) if len(P_nm) > 2 else 0
    f_peak = float(f_nm[idx_peak]) if len(f_nm) > idx_peak else 0.0
    if f_peak < 0.2 * fs:
        band = "Bajas"
    elif f_peak < 0.4 * fs:
        band = "Medias"
    else:
        band = "Altas"

    correct = {"q1": "Blanco", "q2": ("Pasa bajas" if key["tipo"] == "Pasa bajas" else "Pasa banda"), "q3": band}

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f_wm, y=(P_wm + 1e-18),
        mode="lines",
        name="PSD entrada (prom.)",
        line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=f_nm, y=(P_nm + 1e-18),
        mode="lines",
        name="PSD salida (prom.)",
        line=dict(color="orange")
    ))
    fig.update_xaxes(title_text="Frecuencia (Hz)")
    fig.update_yaxes(title_text="S(f) (u.a.)", type="log")
    _plotly_layout_dyn(fig, "PSD promedio: ruido blanco vs ruido filtrado", height=420, showlegend=True, y_log=True)
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("#### Preguntas")
    q1 = st.radio(
        "1) La PSD de la entrada corresponde a ruido:",
        ["Blanco", "Coloreado"],
        index=None,
        key="g4_dyn3_q1",
    )
    q2 = st.radio(
        "2) Según la forma de la PSD de salida, el filtro aplicado es:",
        ["Pasa bajas", "Pasa banda"],
        index=None,
        key="g4_dyn3_q2",
    )
    q3 = st.radio(
        "3) En la salida, la mayor potencia se concentra principalmente en frecuencias:",
        ["Bajas", "Medias", "Altas"],
        index=None,
        key="g4_dyn3_q3",
    )

    answers = {"q1": q1, "q2": q2, "q3": q3}
    meta = {"tipo": key["tipo"], "sigma": key["sigma"], "fs": fs, "T": key["T"], "M": key["M"],
            "fc": key["fc"], "f0": key["f0"], "bw": key["bw"], "f_peak": f_peak}
    return answers, correct, meta


def _score_answers(answers: Dict[str, Any], correct: Dict[str, Any]) -> Tuple[int, float]:
    keys = list(correct.keys())
    n_total = len(keys)
    n_ok = 0
    for k in keys:
        if answers.get(k) == correct.get(k):
            n_ok += 1
    # mapeo simple a escala 0-10
    if n_total == 4:
        mapping = {4: 10.0, 3: 8.0, 2: 6.0, 1: 4.0, 0: 0.0}
    elif n_total == 3:
        mapping = {3: 10.0, 2: 8.0, 1: 5.0, 0: 0.0}
    else:
        mapping = {n_total: 10.0}
    return n_ok, float(mapping.get(n_ok, 0.0))


# -----------------------------
# PDF + GitHub
# -----------------------------

def export_results_pdf_guia4(filename_base: str, student_info: Dict[str, str], resultados: List[Dict[str, Any]]) -> Optional[str]:
    if not _REPORTLAB_OK:
        return None

    out_dir = os.path.join(os.getcwd(), "guia4")
    os.makedirs(out_dir, exist_ok=True)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = os.path.join(out_dir, f"{filename_base}_{ts}.pdf")

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    x0 = 0.8 * inch
    y = height - 0.8 * inch

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x0, y, "Guía 4 — Procesos estocásticos y el ruido")
    y -= 0.35 * inch

    c.setFont("Helvetica", 11)
    c.drawString(x0, y, f"Nombre: {student_info.get('name','')}")
    y -= 0.22 * inch
    c.drawString(x0, y, f"Carné: {student_info.get('id','')}")
    y -= 0.22 * inch
    c.drawString(x0, y, f"Fecha de nacimiento: {student_info.get('dob','')}")
    y -= 0.30 * inch

    c.setFont("Helvetica-Bold", 12)
    c.drawString(x0, y, "Resultados")
    y -= 0.25 * inch

    total = 0.0
    for res in resultados:
        if y < 1.2 * inch:
            c.showPage()
            y = height - 0.8 * inch

        c.setFont("Helvetica-Bold", 11)
        c.drawString(x0, y, f"Dinámica {res.get('dyn_id')}: {res.get('title','')}")
        y -= 0.22 * inch

        c.setFont("Helvetica", 10)
        c.drawString(x0, y, f"Puntaje: {res.get('score',0.0):.1f}/10")
        total += float(res.get("score", 0.0))
        y -= 0.18 * inch

        # Caso (meta)
        meta = res.get("meta", {})
        if meta:
            meta_lines = []
            for k, v in meta.items():
                if v is None:
                    continue
                meta_lines.append(f"- {k}: {v}")
            for line in meta_lines[:10]:
                c.drawString(x0 + 0.15 * inch, y, line)
                y -= 0.16 * inch

        # Respuestas
        answers = res.get("answers", {})
        correct = res.get("correct", {})
        for qk in correct.keys():
            if y < 1.2 * inch:
                c.showPage()
                y = height - 0.8 * inch
            a = answers.get(qk, "")
            cor = correct.get(qk, "")
            c.drawString(x0 + 0.15 * inch, y, f"{qk}: resp='{a}' | correcta='{cor}'")
            y -= 0.16 * inch

        y -= 0.12 * inch

    prom = total / max(len(resultados), 1)
    if y < 1.4 * inch:
        c.showPage()
        y = height - 0.8 * inch

    c.setFont("Helvetica-Bold", 11)
    c.drawString(x0, y, f"Promedio: {prom:.2f}/10")
    c.showPage()
    c.save()
    return pdf_path


def render_dinamicas_guia4():
    _init_guia4_state()
    state = st.session_state.guia4_dinamicas

    st.markdown("## Dinámicas — Guía 4")

    # -------- REGISTRO ÚNICO --------
    st.subheader("Datos del estudiante")
    with st.form("g4_form_student"):
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
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown("---")

    # -------- DINÁMICAS (expanders siempre abiertos) --------
    with st.expander("Dinámica 1 — Ruido gaussiano", expanded=True):
        ans1, cor1, meta1 = _render_dyn1()

    with st.expander("Dinámica 2 — Estacionariedad", expanded=True):
        ans2, cor2, meta2 = _render_dyn2()

    with st.expander("Dinámica 3 — PSD de ruido filtrado", expanded=True):
        ans3, cor3, meta3 = _render_dyn3()

    st.markdown("---")

    # -------- ENVÍO FINAL (único botón) --------
    if st.button("Enviar respuestas y generar PDF", key="g4_send"):
        # Validación
        pending = []
        for i, ans in [(1, ans1), (2, ans2), (3, ans3)]:
            if any(v is None for v in ans.values()):
                pending.append(f"Dinámica {i}")
        if pending:
            st.warning("Responde todas las preguntas antes de enviar. Falta: " + ", ".join(pending))
            return

        ok1, score1 = _score_answers(ans1, cor1)
        ok2, score2 = _score_answers(ans2, cor2)
        ok3, score3 = _score_answers(ans3, cor3)

        resultados = [
            {"dyn_id": 1, "title": "Ruido gaussiano", "answers": ans1, "correct": cor1, "score": score1, "meta": meta1},
            {"dyn_id": 2, "title": "Estacionariedad", "answers": ans2, "correct": cor2, "score": score2, "meta": meta2},
            {"dyn_id": 3, "title": "PSD de ruido filtrado", "answers": ans3, "correct": cor3, "score": score3, "meta": meta3},
        ]

        pdf_path = export_results_pdf_guia4(
            filename_base=f"guia4_{state['student'].get('id','sin_id')}",
            student_info=state["student"],
            resultados=resultados
        )
        if not pdf_path:
            st.error("No se pudo generar el PDF (verifica ReportLab o permisos).")
            return

        st.success("PDF generado correctamente.")
        st.write("Ruta local del PDF:", pdf_path)

        # Subida a GitHub (si hay credenciales)
        nombre_pdf_repo = os.path.basename(pdf_path)
        repo_path = f"guia4/{nombre_pdf_repo}"
        commit_msg = f"Guía 4 - {state['student'].get('id','sin_id')} - {state['student'].get('name','')}".strip()
        ok_up, info = upload_file_to_github_results(
            local_path=pdf_path,
            repo_path=repo_path,
            commit_message=commit_msg,
        )
        if ok_up:
            st.success("PDF enviado a GitHub correctamente.")
            if isinstance(info, dict) and info.get("html_url"):
                st.link_button("Ver archivo en GitHub", info["html_url"])
            st.write("Ruta en el repositorio:", repo_path)
        else:
            err_msg = info.get("error") if isinstance(info, dict) else str(info)
            st.warning("El PDF se generó, pero no se subió a GitHub: " + err_msg)


# -----------------------------
# Render principal de la guía
# -----------------------------

def render_guia4():
    st.title("Guía 4: Procesos estocásticos y el ruido")

    tabs = st.tabs(["Objetivos", "Introducción", "Materiales", "Ejemplos", "Dinámicas", "Conclusiones"])

    with tabs[0]:
        st.subheader("Objetivos")
        st.markdown(OBJETIVOS_TEXT)

    with tabs[1]:
        st.subheader("Introducción teórica")
        st.markdown(INTRO_FULL_TEXT)

    with tabs[2]:
        st.subheader("Materiales y equipo")
        st.markdown(MATERIALES_TEXT)

    with tabs[3]:
        st.subheader("Ejemplos interactivos")
        sub_tabs = st.tabs(["Ejemplo 1", "Ejemplo 2", "Ejemplo 3"])
        with sub_tabs[0]:
            render_ejemplo1()
        with sub_tabs[1]:
            render_ejemplo2()
        with sub_tabs[2]:
            render_ejemplo3()

    # Conclusiones se renderiza antes que Dinámicas para que esté siempre
    # disponible (la dinámica usa st.stop() al faltar datos del estudiante).
    with tabs[5]:
        st.subheader("Conclusiones")
        st.markdown(CONCLUSIONES_TEXT)

    with tabs[4]:
        render_dinamicas_guia4()
