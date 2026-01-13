# -*- coding: utf-8 -*-
"""
Guía 5 Fundamentos de transmisión de datos digitales en presencia de ruido

Notas:
- Diseñado para integrarse al proyecto de guías en Streamlit.
- Gráficas interactivas con Plotly (zoom, pan, hover con valores).
"""
from __future__ import annotations

import os
import math
import json
import re
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from github_uploader import upload_bytes_to_github_results

# ----------------------------
# Texto base (desde GUIA5.docx)
# ----------------------------

OBJETIVOS_MD = r"""**Objetivo general**

Analizar y validar, mediante simulación interactiva, el desempeño de un sistema de transmisión digital en presencia de ruido térmico AWGN, aplicando el concepto de detector óptimo para estimar la probabilidad de error y comparar el rendimiento de los esquemas BPSK y BFSK.

**Objetivos específicos**

- Representar el modelo general de un sistema de comunicación digital , incorporando la adición de ruido AWGN y el proceso de decisión en el receptor mediante un estadístico de decisión y un umbral.
- Interpretar el detector óptimo como un mecanismo de decisión basado en la comparación de una variable de decisión con un umbral, relacionándolo con la separación entre distribuciones condicionadas y la probabilidad de error.
- Simular un enlace BPSK en AWGN y obtener la BER estimada , comparándola con la tendencia teórica , identificando cómo el aumento de reduce los errores.
- Simular un enlace BFSK en AWGN y evaluar su BER estimada , destacando el efecto del esquema de modulación y del proceso de detección sobre el desempeño.
- Contrastar el desempeño entre BPSK y BFSK usando curvas de desempeño.
- Fortalecer la lectura de resultados mediante gráficas interactivas (tiempo, histogramas de la variable de decisión y BER), respondiendo preguntas de verificación conceptual para consolidar el vínculo entre teoría y simulación."""

INTRO_MD = r"""En un sistema de telecomunicación digital, una fuente de datos binarios genera un flujo de bits que se agrupa en un vector de datos transmitidos. Este vector alimenta el modulador de datos, el cual realiza una transformación que asigna la información binaria a una forma de onda transmitida. En señales pasa banda, esta forma de onda se expresa mediante una portadora y su envolvente compleja. Tras propagarse por el canal de transmisión, la señal se atenúa por la pérdida de propagación y se retrasa por el retardo debido a la propagación; además, al sumarse el ruido, la señal recibida en el receptor se modela como una suma entre la salida del canal y un proceso de ruido blanco Gaussiano aditivo (AWGN).

El receptor aplica demodulación para convertir la señal recibida a banda base y así obtener una señal de banda base que pueda procesarse. El objetivo es producir un estimado del vector de datos transmitidos a partir de la observación ruidosa, lo que conduce al problema de detección. En transmisión binaria, la detección se plantea como la elección entre dos hipótesis asociadas a dos señales posibles, x0(t) y x1(t), durante un intervalo Tp. El desempeño se cuantifica por la probabilidad de error de bit (BEP), la cual depende tanto de la energía de las señales como de su similitud, medida mediante el coeficiente de correlación y la distancia euclidiana cuadrática entre señales.

Para evaluar la eficacia de un sistema de transmisión digital se emplean métricas de desempeño: fidelidad, complejidad y eficiencia de ancho de banda. La fidelidad refleja qué tan frecuente ocurren errores en la transmisión en función de la relación señal a ruido (SNR), mientras que la complejidad se asocia a los algoritmos de demodulación y detección, al número de operaciones por símbolo y a la cantidad de señales posibles que deben evaluarse. Por su parte, la eficiencia de ancho de banda expresa cuánta información digital se transmite por unidad de ancho de banda del canal, y evidencia el compromiso entre transmitir más bits y la sensibilidad al ruido.

En la práctica, el desempeño no se evalúa únicamente con la potencia transmitida, sino mediante la razón Eb/N0, donde Eb es la energía promedio recibida por bit y N0 es la densidad espectral de potencia del ruido. Esta razón normaliza el efecto del ruido y permite comparar sistemas distintos independientemente del ancho de banda o de la velocidad de transmisión. Bajo el modelo AWGN, el cálculo de BEP se apoya en funciones asociadas a la variable aleatoria Gaussiana, como la función erf y su función de error complementaria erfc.

El detector óptimo para el caso binario puede implementarse mediante un demodulador con filtro adaptado (matched filter), diseñado a partir de la señal efectiva que contiene la información relevante para distinguir entre hipótesis. El filtro adaptado maximiza la SNR efectiva del detector, y su salida muestreada define un estadístico de decisión que se compara con un umbral. Esta estructura permite expresar el BEP en términos de la distancia euclidiana entre señales y de N0, evidenciando que el diseño de señales busca maximizar la separabilidad (aumentar la distancia) para minimizar la probabilidad de error.

Modulaciones sobre portadora: BFSK y BPSK. En BFSK la información binaria se transmite seleccionando una de dos frecuencias, manteniendo constante la amplitud y la duración del pulso; su desempeño depende de la separación en frecuencia y del coeficiente de correlación entre señales. En BPSK la información binaria se transmite cambiando la fase de una portadora, manteniendo constante su amplitud y energía; cuando la diferencia de fase es π se obtiene un caso de rendimiento de BEP óptimo. Estos resultados permiten comparar fidelidad y eficiencia espectral, y sirven de referencia frente a un límite superior para un canal AWGN establecido por Claude Shannon.

##### Definiciones y notación

**Sistema de telecomunicación digital**: Estructura formada por fuente de datos binarios, modulador de datos, canal de transmisión y receptor (demodulación y sumidero de datos).

**Señal pasa banda / señal de banda base**: En pasa banda se utiliza una portadora; en banda base se trabaja con la señal descendida para procesamiento en el receptor.

**Envolvente compleja:** Representación compleja asociada a la señal transmitida, utilizada para describir la modulación y la señal recibida en banda base.

**AWGN**: Ruido blanco Gaussiano aditivo que se suma a la salida del canal para modelar la señal recibida.

**Fidelidad, complejidad y eficiencia de ancho de banda**: Métricas de desempeño: frecuencia de errores en función de la SNR; dificultad computacional y estructural del receptor; y bits por segundo por Hertz del canal.

**Eb/N0**: Razón entre la energía promedio recibida por bit Eb y la densidad espectral de potencia del ruido N0; métrica que normaliza el efecto del ruido.

**BEP**: Probabilidad de error de bit; se expresa a partir del estadístico de decisión, el umbral y la distribución Gaussiana del ruido.

**Filtro adaptado (matched filter)**: Filtro diseñado a partir de la señal efectiva para maximizar la SNR efectiva del detector; su salida muestreada define el estadístico de decisión.

**Distancia euclidiana cuadrática**: Medida geométrica de separabilidad entre señales continuas; puede expresarse en términos de energía y correlación.

**BFSK y BPSK**: Esquemas de modulación binaria sobre portadora: BFSK selecciona una de dos frecuencias; BPSK cambia la fase de la portadora.

##### Ecuaciones clave

**Modelo del canal con ruido:** Y_c(t) = R_c(t) + W(t)  

**Salida del canal (pasa banda):** R_c(t) = L_p X_c(t-τ_p)  

**Fase adicional por retardo:** φ = -2π f_c τ_p  (Ec. 6.3)

**Señal pasa banda transmitida:** X_c(t) = √2 X_A(t) cos(2π f_c t + X_p(t))  

**Salida del canal en términos de la envolvente compleja:** R_c(t) = Re{ √(2 L_p) X_z(t-τ_p) e^{jφ_p} e^{j2π f_c t} }  

**Envolvente compleja recibida (banda base):** R_z(t) = √(L_p) X_z(t-τ_p) e^{jφ_p}  

**Energía promedio recibida por bit:** E_b = E[E_{R_z}]/K_b = (1/K_b) E_s  

**Eficiencia de ancho de banda:** η_B = W_b / B_T   [bits/s/Hz]  

**Función erf:** erf(z) = (2/√π) ∫_0^z e^{-t^2} dt  

**CDF en términos de erf:** F_X(x) = P(X ≤ x) = 1/2 + (1/2) erf( (x - m_x)/√(2σ_x) )  

**Filtro adaptado en frecuencia:** H(f) = C B_{10}^*(f) e^{-j2π f T_p}  

**Respuesta al impulso del filtro adaptado:** h(t) = b_{10}^*(T_p - t) = x_1^*(T_p - t) - x_0^*(T_p - t)  

**Distancia euclidiana cuadrática:** Δ_E(i,j) = ∫_{-∞}^{∞} |x_i(t) - x_j(t)|^2 dt  

**Distancia en términos de energía y correlación**: Δ_E(i,j) = E_i + E_j - 2√(E_i E_j) Re{ρ_{ij}}  

**SNR efectiva del detector:** η = Δ_E(1,0) / (4 N_0)  

**BEP general (erfc): P_B(E)** = (1/2) erfc( √(Δ_E(1,0)/(4 N_0)) ) 

**Conjunto de señales BFSK:** x_0(t) = √(E_b/T_p) e^{j2π f_d t}, 0≤t≤T_p; 0 otro 
x_1(t) = √(E_b/T_p) e^{-j2π f_d t}, 0≤t≤T_p; 0 otro  (Ec. 6.84)

**BEP para BFSK:** P(E) = (1/2) erfc( √( (E_b/(2N_0)) [1 - sin(4π f_d T_p)/(4π f_d T_p)] ) )  

**Conjunto de señales BPSK:** x_0(t) = √(E_b/T_p), 0≤t≤T_p; 0 otro  
x_1(t) = √(E_b/T_p) e^{jθ}, 0≤t≤T_p; 0 otro  (Ec. 6.97)

**Distancia y BEP para BPSK:** Δ_E(1,0) = 2E_b(1 - cosθ)  
P(E) = (1/2) erfc( √( (E_b/(2N_0)) (1 - cosθ) ) )  
Para θ = π:  P(E) = (1/2) erfc( √(E_b/(2N_0)) )    """

MATERIALES_MD = r"""Para desarrollar las actividades de esta guía interactiva se recomienda contar con:

Una computadora personal con sistema operativo actualizado (Windows, Linux o macOS).

Python instalado (versión 3.8 o superior recomendada).

Un entorno de desarrollo como Visual Studio Code o PyCharm. Las siguientes bibliotecas de Python:

numpy para el manejo de arreglos y operaciones numéricas.

matplotlib para la generación de gráficas.

streamlit para la interfaz interactiva de la guía.

scipy para operaciones adicionales de filtrado, convolución y análisis en frecuencia."""

CONCLUSIONES_MD = r"""- En un sistema de transmisión digital en presencia de AWGN, el desempeño queda fuertemente gobernado por la relación : al aumentar la probabilidad de error disminuye, porque el ruido tiene menor capacidad de desplazar la variable de decisión hacia la región equivocada.
- El detector óptimo (MAP/ML bajo hipótesis equiprobables) se interpreta como una regla de decisión basada en un umbral/estadístico que separa las hipótesis. Esta visión permite conectar directamente la teoría (distribuciones condicionadas y probabilidad de error) con la simulación (BER estimada), haciendo evidente por qué la detección óptima mejora el desempeño.
- Las curvas BER vs permiten comparar de forma clara los esquemas BPSK y BFSK: ambos mejoran al aumentar , pero presentan diferencias de rendimiento debido a cómo se representan las señales y cómo se separan en el receptor bajo ruido, lo que confirma que el esquema de modulación y el método de detección determinan la robustez del enlace."""


# ----------------------------
# Utilidades matemáticas
# ----------------------------

def _erfc(x: np.ndarray | float) -> np.ndarray | float:
    """Complementary error function con fallback sin SciPy."""
    try:
        from scipy.special import erfc  # type: ignore
        return erfc(x)
    except Exception:
        xf = np.asarray(x, dtype=float)
        out = np.vectorize(math.erfc)(xf)
        if np.isscalar(x):
            return float(out)
        return out


def Q(x: np.ndarray | float) -> np.ndarray | float:
    """Q(x) = P(Z>x), Z~N(0,1)."""
    return 0.5 * _erfc(np.asarray(x) / math.sqrt(2.0))


def ber_teorica_bpsk(EbN0_lin: np.ndarray) -> np.ndarray:
    return Q(np.sqrt(2.0 * EbN0_lin))


def rho_bfsk(fd: float, Tb: float) -> float:
    """Coeficiente de correlación ρ para BFSK (sin(4π f_d T_b)/(4π f_d T_b))."""
    x = 4.0 * math.pi * fd * Tb
    if abs(x) < 1e-9:
        return 1.0
    return math.sin(x) / x


def ber_teorica_bfsk(EbN0_lin: np.ndarray, fd: float, Tb: float) -> np.ndarray:
    """BER teórica coherente para BFSK con separación fd (modelo del capítulo)."""
    rho = rho_bfsk(fd, Tb)
    term = max(0.0, 1.0 - rho)
    return 0.5 * _erfc(np.sqrt(EbN0_lin * term / 2.0))


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


# ----------------------------
# Utilidades Plotly (tema legible)
# ----------------------------

def _plotly_layout(fig: go.Figure, title: str, height: int = 380, showlegend: bool = True) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16, color="black")),
        height=height,
        margin=dict(l=55, r=25, t=55, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if showlegend else dict(),
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", font_color="black"),
        font=dict(color="black"),
    )
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True, ticks="outside", tickfont=dict(size=12, color="black"))
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True, ticks="outside", tickfont=dict(size=12, color="black"))
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.15)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.15)")
    return fig


def _line(fig: go.Figure, x, y, name: str, color: str, row: int = 1, col: int = 1, mode: str = "lines"):
    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name, line=dict(color=color, width=2.5)), row=row, col=col)


def _force_plotly_readable(fig, height: int = 420):
    fig.update_layout(
        height=height,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        hoverlabel=dict(bgcolor="white", font_color="black"),
        margin=dict(l=60, r=20, t=95, b=55),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1.0,
            font=dict(color="black", size=12),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.25)",
            borderwidth=1,
        ),
    )

    fig.update_xaxes(
        showline=True, linecolor="black", linewidth=2,
        ticks="outside", tickcolor="black",
        tickfont=dict(color="black"),
        title=dict(font=dict(color="black")),
        gridcolor="rgba(0,0,0,0.15)",
        zeroline=False,
    )
    fig.update_yaxes(
        showline=True, linecolor="black", linewidth=2,
        ticks="outside", tickcolor="black",
        tickfont=dict(color="black"),
        title=dict(font=dict(color="black")),
        gridcolor="rgba(0,0,0,0.15)",
        zeroline=False,
    )

    if hasattr(fig.layout, "annotations") and fig.layout.annotations:
        for ann in fig.layout.annotations:
            if isinstance(ann, dict):
                ann["font"] = dict(color="black")
                ann["x"] = 0.5
                ann["xanchor"] = "center"
            else:
                ann.font = dict(color="black")
                ann.x = 0.5
                ann.xanchor = "center"


# ----------------------------
# Ejemplo 1
# ----------------------------

def render_ejemplo1():
    st.subheader("Ejemplo 1 — Cadena digital: bits → modulación → AWGN → decisión")

    with st.expander("Descripción y pasos a seguir", expanded=True):
        st.markdown(
            "Este ejemplo muestra el flujo completo de un sistema digital binario en presencia de **ruido AWGN**:\n\n"
            "1) **Dato binario enviado** $b[k]$\n"
            "2) **Señal analógica modulada** $s(t)$ (BPSK pasabanda)\n"
            "3) **Señal en el canal** $r(t)=s(t)+w(t)$\n"
            "4) **Dato demodulado** $\\hat{b}[k]$ mediante un detector \n\n"
            "**Pasos sugeridos**\n"
            "1. Ajusta $E_b/N_0$ y el número de bits.\n"
            "2. Pulsa **Simular**.\n"
            "3. Observa cómo al bajar $E_b/N_0$ aparecen errores (bits mal detectados)."
        )

    col1, col2 = st.columns(2)

    with col1:
        EbN0_dB = st.slider(
            "Relación $E_b/N_0$ (dB)",
            min_value=-2.0, max_value=14.0,
            value=2.0, step=1.0, key="g5_e1_ebn0"
        )
        Tb_ms = st.slider(
            "Duración de bit $T_b$ (ms)",
            min_value=0.5, max_value=5.0,
            value=2.0, step=0.5, key="g5_e1_tb_ms"
        )
        Nb = st.slider(
            "Número de bits a visualizar",
            min_value=8, max_value=64,
            value=20, step=1, key="g5_e1_nb"
        )

        fc = st.slider(
            "Frecuencia de portadora $f_c$ (Hz)",
            min_value=300, max_value=6000,
            value=1000, step=100, key="g5_e1_fc"
        )

        if st.button("Simular", key="g5_e1_btn"):
            st.session_state.g5_e1_seed = int(np.random.randint(0, 2**31 - 1))
            st.session_state.g5_e1_run = True

    if not st.session_state.get("g5_e1_run"):
        st.info("Pulsa **Simular** para generar las gráficas del ejemplo 1.")
        return

    seed = st.session_state.get("g5_e1_seed", 12345)
    rng = _rng(seed)

    Eb = 1.0
    Tb = float(Tb_ms) * 1e-3
    EbN0_lin = 10 ** (EbN0_dB / 10.0)
    N0 = Eb / EbN0_lin

    fs = max(20.0 * fc, 80.0 / Tb)
    fs = float(min(fs, 200000.0))
    dt = 1.0 / fs

    Ns = int(max(32, round(fs * Tb)))
    N = Nb * Ns
    t = np.arange(N) * dt

    b = rng.integers(0, 2, size=Nb)
    a = 2 * b - 1
    a_samp = np.repeat(a, Ns)

    A = math.sqrt(2.0 * Eb / Tb)
    carrier = np.cos(2 * np.pi * fc * t)
    s = A * a_samp * carrier

    sigma_w = math.sqrt(N0 * fs / 2.0)
    w = rng.normal(0.0, sigma_w, size=N)
    r = s + w

    yk = np.zeros(Nb, dtype=float)
    for k in range(Nb):
        i0 = k * Ns
        i1 = (k + 1) * Ns
        yk[k] = np.sum(r[i0:i1] * carrier[i0:i1]) * dt

    b_hat = (yk > 0).astype(int)
    b_hat_samp = np.repeat(b_hat, Ns)

    n_err = int(np.sum(b_hat != b))
    ber_hat = n_err / Nb

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(
            "1) Dato binario enviado b[k]",
            "2) Señal analógica modulada s(t) (BPSK pasabanda)",
            "3) Señal en el canal con AWGN: r(t)=s(t)+w(t)",
            "4) Dato demodulado (decisión) ŷ → b̂[k]"
        )
    )

    bit_edges = np.arange(0, Nb + 1) * Tb
    for xedge in bit_edges:
        fig.add_vline(x=float(xedge), line_width=1, line_dash="dot", line_color="rgba(0,0,0,0.25)")

    fig.add_trace(
        go.Scatter(
            x=t, y=np.repeat(b, Ns), mode="lines",
            name="b[k]", line=dict(width=2, color="#1f77b4"), line_shape="hv"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=t, y=s, mode="lines",
            name="s(t)", line=dict(width=2, color="#d62728")
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=t, y=r, mode="lines",
            name="r(t)", line=dict(width=1.5, color="#2ca02c")
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=t, y=b_hat_samp, mode="lines",
            name="b̂[k]", line=dict(width=2, color="#9467bd"), line_shape="hv"
        ),
        row=4, col=1
    )

    fig.update_yaxes(title_text="Bit", row=1, col=1, range=[-0.2, 1.2])
    fig.update_yaxes(title_text="Amplitud", row=2, col=1)
    fig.update_yaxes(title_text="Amplitud", row=3, col=1)
    fig.update_yaxes(title_text="Bit", row=4, col=1, range=[-0.2, 1.2])
    fig.update_xaxes(title_text="Tiempo (s)", row=4, col=1)

    if getattr(fig.layout, "annotations", None):
        for ann in fig.layout.annotations:
            ann["x"] = 0.5
            ann["xanchor"] = "center"

    fig.update_layout(
        template="plotly_white",
        title="Cadena digital: bits → modulación → AWGN → decisión",
        height=900,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black", size=14),
        hoverlabel=dict(bgcolor="white", font_color="black"),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1.0,
            font=dict(color="black", size=13),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="black",
            borderwidth=1,
        ),
        margin=dict(l=70, r=20, t=90, b=60),
    )
    fig.update_xaxes(showline=True, linecolor="black", linewidth=2,
                     gridcolor="rgba(0,0,0,0.15)", tickfont=dict(size=13))
    fig.update_yaxes(showline=True, linecolor="black", linewidth=2,
                     gridcolor="rgba(0,0,0,0.15)", tickfont=dict(size=13))

    # ---- Imagen + gráficas juntas (imagen primero) ----
    # Buscar la imagen en rutas RELATIVAS al repo (compatibles con Streamlit Cloud)
    base_dir = os.path.dirname(__file__)
    candidates = [
        os.path.join(base_dir, "assets", "modelo_binario.png"),
        os.path.join(base_dir, "assets", "Modelo binario.png"),
        os.path.join(base_dir, "Modelo binario.png"),
    ]

    diagram_path = None
    for p in candidates:
        if os.path.exists(p):
            diagram_path = p
            break

    with col2:
        if diagram_path is not None:
            st.image(
                diagram_path,
                caption="Modelo general de un sistema digital en presencia de AWGN",
                use_container_width=True
            )
        else:
            st.warning("No pude cargar la imagen. Guarda 'modelo_binario.png' en la carpeta assets/ del repositorio.")

        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.markdown(
            f"**Parámetros:** $E_b/N_0$ = {EbN0_dB:.1f} dB, $T_b$ = {Tb_ms:.2f} ms, $f_c$ = {fc} Hz  \n"
            f"**Resultado:** errores = {n_err}/{Nb},  $\\widehat{{BER}}$ ≈ {ber_hat:.3f}"
        )

    st.markdown("##### Explicación de la simulación")
        
    st.markdown(
        "- **$E_b$ (energía por bit):** es la energía promedio invertida para transmitir **un bit**. "
        "Se obtiene integrando la energía de la señal en el intervalo de bit: "
        "$$E_b=\\int_0^{T_b} s^2(t)\\,dt.$$ "
        "A mayor $E_b$, el bit llega “más fuerte” al receptor.\n"
        "- **$N_0$ (densidad espectral de potencia del ruido):** mide cuánta potencia de ruido hay por cada Hz (W/Hz) en AWGN. "
        "Para ruido blanco, la PSD es aproximadamente constante y suele expresarse como $N_0/2$ en banda base. "
        "A mayor $N_0$, el canal es “más ruidoso”.\n"
        "- **Relación $E_b/N_0$:** es una SNR normalizada por bit que compara la energía útil contra la intensidad del ruido. "
        "Si $E_b/N_0$ aumenta, disminuye la probabilidad de error (BER); si baja, el ruido domina y aparecen más errores. \n"
        "- Un correlador es un bloque del receptor que mide qué tan parecida es la señal recibida a una señal de referencia que el receptor “espera” recibir. \n"
        )

    st.markdown(
        "- En la gráfica (2), la señal **modulada es analógica** porque es una **onda continua** (portadora) cuya fase cambia según el bit.\n"
        "- En la gráfica (3), el canal agrega **ruido AWGN**: al bajar $E_b/N_0$ el ruido domina y la señal se distorsiona más.\n"
        "- En la gráfica (4), el receptor usa un **correlador** (equivalente al filtro igualado) y decide por el **signo** del estadístico: "
        "si $y_k>0$ decide 1, si $y_k<0$ decide 0.\n"
        "- Por eso, **$E_b/N_0$ sí afecta la decisión**: con menos $E_b/N_0$ aumenta el traslape y aparecen errores."
        )

    st.markdown("##### Preguntas y respuestas")
    st.markdown("**1. ¿Por qué la señal de la gráfica (2) es analógica aunque transmita bits?**")
    st.markdown("**R:** Porque la información binaria se representa mediante un **parámetro continuo** de una onda (fase/amplitud) en el tiempo continuo.")
    st.markdown("**2. ¿Qué efecto tiene disminuir $E_b/N_0$?**")
    st.markdown("**R:** Aumenta la potencia relativa del ruido frente a la energía por bit, haciendo más probable que el estadístico cambie de signo y se detecte mal el bit.")
    st.markdown("**3. ¿Qué bloque del receptor realiza la “decisión” final?**")
    st.markdown("**R:** El muestreador a la salida del correlador, que compara el estadístico con el umbral (aquí 0).")


# ----------------------------
# Ejemplo 2 — BPSK
# ----------------------------

def render_ejemplo2():
    st.subheader("Ejemplo 2 — Desempeño óptimo de BPSK en AWGN: BER vs $E_b/N_0$")

    with st.expander("Descripción y pasos a seguir", expanded=True):
        st.markdown(
            "Se calcula la **BER** de BPSK en AWGN de dos maneras:\n\n"
            "- **Teórica** (función Q)\n"
            "- **Simulada** \n"
        )

    col1, col2 = st.columns(2)

    with col1:
        Nbits = st.slider(
            "Número de bits ",
            min_value=2000, max_value=200000, value=30000, step=2000,
            key="g5_e2_nbits"
        )
        ebn0_min = st.slider(
            "$E_b/N_0$ mínimo (dB)",
            min_value=-2.0, max_value=10.0, value=0.0, step=1.0,
            key="g5_e2_min"
        )
        ebn0_max = st.slider(
            "$E_b/N_0$ máximo (dB)",
            min_value=2.0, max_value=16.0, value=12.0, step=1.0,
            key="g5_e2_max"
        )
        step_db = st.select_slider("Paso (dB)", options=[0.5, 1.0, 2.0], value=1.0, key="g5_e2_step")

        punto_hist = st.slider(
            "Punto para histogramas (dB)",
            min_value=float(ebn0_min), max_value=float(ebn0_max),
            value=6.0, step=float(step_db),
            key="g5_e2_hist_pt"
        )
        ver_hist = st.checkbox("Mostrar histogramas de decisión", value=True, key="g5_e2_show_hist")

        if st.button("Simular", key="g5_e2_btn"):
            st.session_state.g5_e2_seed = int(np.random.randint(0, 2**31 - 1))
            st.session_state.g5_e2_run = True

    if not st.session_state.get("g5_e2_run"):
        st.info("Pulsa **Simular** para generar la curva BER de BPSK.")
        return

    def _force_plotly_readable_local(fig, height=420):
        fig.update_layout(
            template="plotly_white",
            height=height,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
            legend=dict(font=dict(color="black")),
            hoverlabel=dict(bgcolor="white", font_color="black"),
            margin=dict(l=60, r=20, t=70, b=55),
        )
        fig.update_xaxes(
            showline=True,
            linecolor="black",
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=False,
            tickfont=dict(color="black"),
            title=dict(font=dict(color="black")),
        )
        fig.update_yaxes(
            showline=True,
            linecolor="black",
            gridcolor="rgba(0,0,0,0.15)",
            zeroline=False,
            tickfont=dict(color="black"),
            title=dict(font=dict(color="black")),
        )
        if hasattr(fig.layout, "annotations") and fig.layout.annotations:
            for ann in fig.layout.annotations:
                ann["font"] = dict(color="black")
        return fig

    seed = st.session_state.get("g5_e2_seed", 22222)
    rng = _rng(seed)

    Eb = 1.0
    ebn0_dB = np.arange(float(ebn0_min), float(ebn0_max) + 1e-9, float(step_db))
    EbN0_lin = 10 ** (ebn0_dB / 10.0)

    b = rng.integers(0, 2, size=Nbits)
    a = 2*b - 1
    s = a * math.sqrt(Eb)

    ber_sim = np.zeros_like(EbN0_lin, dtype=float)
    for k, g in enumerate(EbN0_lin):
        N0 = Eb / g
        sigma = math.sqrt(N0 / 2.0)
        n = rng.normal(0.0, sigma, size=Nbits)
        r = s + n
        bhat = (r >= 0).astype(int)
        ber_sim[k] = np.mean(bhat != b)

    ber_th = ber_teorica_bpsk(EbN0_lin)

    ber_sim_plot = ber_sim.copy()
    piso_plot = 1.0 / float(Nbits)
    zero_mask = (ber_sim_plot <= 0.0)
    ber_sim_plot[zero_mask] = piso_plot
    n_zeros = int(np.sum(zero_mask))

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ebn0_dB, y=ber_th, mode="lines",
            name="BER teórica", line=dict(color="#1f77b4", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=ebn0_dB, y=ber_sim_plot, mode="lines+markers",
            name="BER simulada", line=dict(color="#ff7f0e", width=2.5),
            marker=dict(size=7)
        ))
        fig.update_yaxes(type="log", title_text="BER (log)")
        fig.update_xaxes(title_text="$E_b/N_0$ (dB)")

        try:
            _plotly_layout(fig, "BPSK en AWGN: BER vs $E_b/N_0$", height=420, showlegend=True)
        except Exception:
            fig.update_layout(title="BPSK en AWGN: BER vs $E_b/N_0$")

        _force_plotly_readable_local(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

        if n_zeros > 0:
            st.info(
                f"Nota: en {n_zeros} punto(s) del barrido no se observaron errores (BER simulada = 0). "
                f"En escala log, 0 no se puede graficar; por eso se usa un piso para la gráfica: 1/Nbits = {piso_plot:.2e}."
            )

        if ver_hist:
            g = 10 ** (float(punto_hist) / 10.0)
            N0 = Eb / g
            sigma = math.sqrt(N0 / 2.0)
            Nh = 20000

            bh = rng.integers(0, 2, size=Nh)
            ah = 2*bh - 1
            rh = ah * math.sqrt(Eb) + rng.normal(0.0, sigma, size=Nh)

            r0 = rh[bh == 0]
            r1 = rh[bh == 1]

            figH = go.Figure()
            figH.add_trace(go.Histogram(x=r0, nbinsx=60, name="r | bit 0", opacity=0.55, marker_color="#1f77b4"))
            figH.add_trace(go.Histogram(x=r1, nbinsx=60, name="r | bit 1", opacity=0.55, marker_color="#ff7f0e"))
            figH.add_vline(x=0.0, line_width=3, line_dash="dash", line_color="black")
            figH.update_xaxes(title_text="Estadístico (muestra)")
            figH.update_yaxes(title_text="Frecuencia relativa")

            try:
                _plotly_layout(figH, f"Histogramas (BPSK) para $E_b/N_0$={float(punto_hist):.1f} dB",
                              height=380, showlegend=True)
            except Exception:
                figH.update_layout(title=f"Histogramas (BPSK) para $E_b/N_0$={float(punto_hist):.1f} dB")

            _force_plotly_readable_local(figH, height=380)
            st.plotly_chart(figH, use_container_width=True)

    st.markdown("##### Explicación de la simulación")

    st.markdown(
        "Como se definió en la guía 1, la **BER** mide la tasa de errores por bits enviados. "
        "Pero, ¿de dónde sale la probabilidad de que el error suceda? A esta función se le conoce "
        "**Probabilidad de error de bit (BEP)**. Para BPSK se define como:\n"
    )

    st.latex(r"P(E)=\frac{1}{2}\,\mathrm{erfc}\!\left(\sqrt{\frac{E_b}{2N_0}}\right)")

    st.markdown(
        "**BPSK (Binary Phase Shift Keying)** es una técnica de modulación digital que representa datos binarios (0s y 1s) "
        "cambiando la fase de una señal portadora en 180 grados; un '1' se codifica con una fase (ej., 0°) y un '0' con la "
        "fase invertida (ej., 180°).\n\n"
        "- Transmite bits usando **dos fases** (0° y 180°): en banda base equivale a mapear el bit a **±√E_b**.\n"
        "- En AWGN y bits equiprobables, el receptor óptimo decide por el **signo** del estadístico (umbral 0).\n"
        "- Al aumentar $E_b/N_0$, las distribuciones condicionadas se separan y la **BER disminuye**.\n"
        "- Para estimar BER muy pequeñas por simulación, se requieren muchos bits (para observar suficientes errores).\n\n"
        "**Aplicaciones típicas:** enlaces satelitales y espaciales, telemetría robusta, sistemas GNSS y escenarios donde se "
        "privilegia robustez sobre eficiencia espectral."
    )

    st.markdown("##### Preguntas y respuestas")
    st.markdown("**1. ¿Por qué graficamos BER en escala logarítmica?**")
    st.markdown("**R:** Porque la BER cae por órdenes de magnitud al aumentar $E_b/N_0$; la escala log permite ver esa caída claramente.")
    st.markdown("**2. ¿Qué representa $E_b/N_0$?**")
    st.markdown("**R:** Es la energía promedio por bit comparada con la densidad espectral de potencia del ruido; resume la “calidad” del enlace ante AWGN.")
    st.markdown("**3. ¿Por qué la curva simulada puede desviarse en BER bajas?**")
    st.markdown("**R:** Porque el conteo de errores es aleatorio: con pocos o cero errores observados, la estimación tiene alta incertidumbre y requiere más bits.")
    st.markdown("**4. ¿Por qué aparece una caída vertical rara en la curva simulada (cuando pasa)?**")
    st.markdown("**R:** Ocurre cuando la BER simulada es 0 (no hubo errores). En escala log, 0 no se puede representar, por eso se usa un piso (≈1/Nbits) solo para graficar.")


# ----------------------------
# Ejemplo 3 — BFSK
# ----------------------------

def render_ejemplo3():
    st.subheader("Ejemplo 3 — BFSK en AWGN: efecto de la separación en frecuencia")

    with st.expander("Descripción y pasos a seguir", expanded=True):
        st.markdown(
            "En **BFSK** (Binary Frequency Shift Keying) cada bit se representa con **una de dos frecuencias**.\n\n"
            "En el receptor , se correlaciona (o filtra igualado) con ambas señales y se decide por la mayor.\n\n"
            "En este ejemplo:\n"
            "- Se calcula BER **teórica** y **simulada**\n"
        )

    col1, col2 = st.columns(2)

    with col1:
        Nbits = st.slider("Número de bits", min_value=2000, max_value=200000,
                          value=40000, step=2000, key="g5_e3_nbits")
        Tb_ms = st.slider("Duración de bit $T_b$ (ms)", min_value=0.5, max_value=6.0,
                          value=2.0, step=0.5, key="g5_e3_tb")
        fd = st.slider("Separación (desviación) $f_d$ (Hz)", min_value=20.0, max_value=1200.0,
                       value=250.0, step=10.0, key="g5_e3_fd")
        ebn0_min = st.slider("$E_b/N_0$ mínimo (dB)", min_value=-2.0, max_value=10.0,
                             value=0.0, step=1.0, key="g5_e3_min")
        ebn0_max = st.slider("$E_b/N_0$ máximo (dB)", min_value=2.0, max_value=16.0,
                             value=12.0, step=1.0, key="g5_e3_max")
        step_db = st.select_slider("Paso (dB)", options=[0.5, 1.0, 2.0], value=1.0, key="g5_e3_step")

        if st.button("Simular", key="g5_e3_btn"):
            st.session_state.g5_e3_seed = int(np.random.randint(0, 2**31 - 1))
            st.session_state.g5_e3_run = True

    if not st.session_state.get("g5_e3_run"):
        st.info("Pulsa **Simular** para generar la curva BER de BFSK.")
        return

    seed = st.session_state.get("g5_e3_seed", 33333)
    rng = _rng(seed)

    Eb = 1.0
    Tb = float(Tb_ms) * 1e-3

    rho = rho_bfsk(float(fd), Tb)
    rho_clip = min(0.999999, max(-0.999999, float(rho)))

    ebn0_dB = np.arange(float(ebn0_min), float(ebn0_max) + 1e-9, float(step_db))
    EbN0_lin = 10 ** (ebn0_dB / 10.0)

    ber_th_bfsk = ber_teorica_bfsk(EbN0_lin, float(fd), Tb)

    u0 = np.array([1.0, 0.0])
    u1 = np.array([rho_clip, math.sqrt(max(0.0, 1.0 - rho_clip**2))])
    s0 = math.sqrt(Eb) * u0
    s1 = math.sqrt(Eb) * u1

    b = rng.integers(0, 2, size=Nbits)
    ber_sim = np.zeros_like(EbN0_lin)

    for k, g in enumerate(EbN0_lin):
        N0 = Eb / g
        sigma = math.sqrt(N0 / 2.0)
        n = rng.normal(0.0, sigma, size=(Nbits, 2))
        s = np.where(b[:, None] == 1, s1, s0)
        r = s + n
        z0 = r @ u0
        z1 = r @ u1
        bhat = (z1 > z0).astype(int)
        ber_sim[k] = np.mean(bhat != b)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ebn0_dB, y=ber_th_bfsk, mode="lines",
            name="BFSK teórica", line=dict(color="#1f77b4", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=ebn0_dB, y=ber_sim, mode="lines+markers",
            name="BFSK simulada", line=dict(color="#ff7f0e", width=2.5),
            marker=dict(size=7)
        ))

        fig.update_yaxes(type="log", title_text="BER (log)")
        fig.update_xaxes(title_text="$E_b/N_0$ (dB)")

        _plotly_layout(fig, "BFSK : BER vs $E_b/N_0$", height=440, showlegend=True)
        _force_plotly_readable(fig, height=440)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Explicación de la simulación")
    st.markdown("Para **BFSK** la **BEP** está dada por la ecuación:")
    st.latex(
        r"P(E)=\frac{1}{2}\,\mathrm{erfc}\!\left(\sqrt{\frac{E_b}{2N_0}\left[1-\frac{\sin\left(4\pi f_d T_b\right)}{4\pi f_d T_b}\right]}\right)"
    )

    st.markdown(
        "**BFSK** es una modulación binaria donde el bit 0 y el bit 1 se transmiten con **dos frecuencias distintas**.\n"
        "- En detección, el receptor calcula dos correlaciones (una por cada frecuencia) y decide por la mayor.\n"
        "- **Comparación con BPSK:** en detección óptima y para el mismo $E_b/N_0$, **BPSK suele ser mejor** (menor BER) porque usa señales **antipodales**.\n"
        "  En cambio, BFSK típicamente requiere **más ancho de banda** y, si no es ortogonal ($\\rho\\neq 0$), puede degradarse más.\n"
    )

    st.markdown("##### Preguntas y respuestas")
    st.markdown("**1. ¿Por qué BPSK suele superar a BFSK en BER para el mismo $E_b/N_0$?**")
    st.markdown("**R:** Porque BPSK usa señales antipodales (máxima separación en espacio de señales), lo que reduce la probabilidad de confusión bajo ruido gaussiano; BFSK depende de qué tan ortogonales sean sus señales (ρ).")


# ----------------------------
# Dinámicas (evaluación)
# ----------------------------

def _ensure_student_info(form_key: str) -> Dict[str, str] | None:
    if "student_info" not in st.session_state:
        st.session_state.student_info = {"name": "", "id": "", "dob": ""}

    info = st.session_state.student_info

    with st.form(form_key):
        st.write("Datos del estudiante")
        name = st.text_input("Nombre completo", value=info["name"])
        carnet = st.text_input("Carné", value=info["id"])
        dob = st.text_input("Fecha de nacimiento (YYYY-MM-DD)", value=info["dob"])
        enviar = st.form_submit_button("Guardar / actualizar datos")

    if enviar:
        if not name or not carnet or not dob:
            st.warning("Completa nombre, carné y fecha de nacimiento.")
            return None
        st.session_state.student_info = {"name": name, "id": carnet, "dob": dob}
        st.success("Datos del estudiante guardados correctamente.")

    return st.session_state.student_info


def _sanitize_filename(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _build_results_payload(student_info: Dict[str, str], results: Dict[str, Any]) -> bytes:
    payload = {
        "guia": 5,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "student": student_info,
        "results": results,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _init_dyn_state():
    if "guia5_dinamicas" not in st.session_state:
        st.session_state.guia5_dinamicas = {
            "dyn1": {"seed": None, "key": None},
            "dyn2": {"seed": None, "key": None},
            "dyn3": {"seed": None, "key": None},
            "submitted": False,
            "last_result": None,
            "uploaded": False,
            "last_upload": None,
        }


def _score(answers: Dict[str, Any], correct: Dict[str, Any]) -> Tuple[int, float]:
    keys = list(correct.keys())
    n_total = len(keys)
    n_ok = 0
    for k in keys:
        if answers.get(k) == correct.get(k):
            n_ok += 1
    mapping = {3: {3: 10.0, 2: 8.0, 1: 5.0, 0: 0.0}}.get(n_total, {n_total: 10.0})
    return n_ok, float(mapping.get(n_ok, 0.0))


def _dyn1_key(seed: int) -> Dict[str, Any]:
    rng = _rng(seed)
    EbN0_dB = float(rng.choice([0.0, 2.0, 4.0, 6.0, 8.0]))
    P1 = float(rng.choice([0.5, 0.6, 0.7]))
    EbN0_lin = 10 ** (EbN0_dB / 10.0)
    Eb = 1.0
    N0 = Eb / EbN0_lin
    muY = 2.0 * Eb
    sigY = math.sqrt(2.0 * Eb * N0)
    gamma = (N0 / 2.0) * math.log((1.0 - P1) / P1)
    correct = {
        "q1": "Positiva",
        "q2": "Hacia la izquierda (favorece bit 1)" if P1 > 0.5 else "No cambia",
        "q3": "Disminuye",
    }
    return {"EbN0_dB": EbN0_dB, "P1": P1, "muY": muY, "sigY": sigY, "gamma": gamma, "correct": correct}


def _render_dyn1():
    st.markdown("### Dinámica 1 — Detector óptimo: histogramas y umbral")

    state = st.session_state.guia5_dinamicas
    if state["dyn1"]["seed"] is None:
        state["dyn1"]["seed"] = int(np.random.randint(0, 2**31 - 1))
        state["dyn1"]["key"] = _dyn1_key(state["dyn1"]["seed"])

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Nuevo caso (Dinámica 1)", key="g5_dyn1_new"):
            state["dyn1"]["seed"] = int(np.random.randint(0, 2**31 - 1))
            state["dyn1"]["key"] = _dyn1_key(state["dyn1"]["seed"])
            for k in ["g5_dyn1_q1", "g5_dyn1_q2", "g5_dyn1_q3"]:
                st.session_state.pop(k, None)

    key = state["dyn1"]["key"]
    with colB:
        st.markdown(f"**Caso:** $E_b/N_0$={key['EbN0_dB']:.0f} dB, $P(1)$={key['P1']:.1f}")

    rng = _rng(state["dyn1"]["seed"])
    Nr = 2500
    Y0 = -key["muY"] + rng.normal(0.0, key["sigY"], size=Nr)
    Y1 = +key["muY"] + rng.normal(0.0, key["sigY"], size=Nr)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=Y0, nbinsx=50, name="Y | bit 0", opacity=0.55, marker_color="#1f77b4"))
    fig.add_trace(go.Histogram(x=Y1, nbinsx=50, name="Y | bit 1", opacity=0.55, marker_color="#ff7f0e"))
    fig.add_vline(x=key["gamma"], line_width=3, line_dash="dash", line_color="black")
    fig.update_xaxes(title_text="Y")
    fig.update_yaxes(title_text="Frecuencia relativa")
    _plotly_layout(fig, "Histogramas y umbral", height=360, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Preguntas")
    q1 = st.radio(
        "1) En este modelo, la media del estadístico bajo bit 1 es:",
        ["Positiva", "Negativa"],
        index=None,
        key="g5_dyn1_q1",
    )
    q2 = st.radio(
        "2) Como $P(1)>0.5$, el umbral MAP se desplaza:",
        ["Hacia la izquierda (favorece bit 1)", "Hacia la derecha (favorece bit 0)", "No cambia"],
        index=None,
        key="g5_dyn1_q2",
    )
    q3 = st.radio(
        "3) Si aumentamos $E_b/N_0$, el traslape entre histogramas típicamente:",
        ["Aumenta", "Disminuye", "Se mantiene"],
        index=None,
        key="g5_dyn1_q3",
    )
    answers = {"q1": q1, "q2": q2, "q3": q3}
    return answers, key["correct"]


def _dyn2_key(seed: int) -> Dict[str, Any]:
    rng = _rng(seed)
    EbN0_dB = float(rng.choice([0.0, 2.0, 4.0, 6.0, 8.0, 10.0]))
    EbN0_lin = 10 ** (EbN0_dB / 10.0)
    ber = float(ber_teorica_bpsk(np.array([EbN0_lin]))[0])
    if ber >= 1e-1:
        orden = "≈10⁻¹"
    elif ber >= 1e-2:
        orden = "≈10⁻²"
    elif ber >= 1e-3:
        orden = "≈10⁻³"
    elif ber >= 1e-4:
        orden = "≈10⁻⁴"
    else:
        orden = "<10⁻⁴"
    correct = {"q1": "Disminuye", "q2": orden, "q3": "Signo del estadístico"}
    return {"EbN0_dB": EbN0_dB, "orden": orden, "correct": correct}


def _render_dyn2():
    st.markdown("### Dinámica 2 — BPSK: lectura de BER vs $E_b/N_0$")

    state = st.session_state.guia5_dinamicas
    if state["dyn2"]["seed"] is None:
        state["dyn2"]["seed"] = int(np.random.randint(0, 2**31 - 1))
        state["dyn2"]["key"] = _dyn2_key(state["dyn2"]["seed"])

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Nuevo caso (Dinámica 2)", key="g5_dyn2_new"):
            state["dyn2"]["seed"] = int(np.random.randint(0, 2**31 - 1))
            state["dyn2"]["key"] = _dyn2_key(state["dyn2"]["seed"])
            for k in ["g5_dyn2_q1", "g5_dyn2_q2", "g5_dyn2_q3"]:
                st.session_state.pop(k, None)

    key = state["dyn2"]["key"]
    with colB:
        st.markdown(f"**Caso:** $E_b/N_0$ = {key['EbN0_dB']:.0f} dB")

    ebn0 = np.arange(0.0, 11.0, 1.0)
    th = ber_teorica_bpsk(10 ** (ebn0 / 10.0))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ebn0, y=th, mode="lines+markers", name="BER teórica", line=dict(color="#1f77b4", width=3)))
    fig.add_vline(x=key["EbN0_dB"], line_width=3, line_dash="dash", line_color="black")
    fig.update_yaxes(type="log", title_text="BER")
    fig.update_xaxes(title_text="$E_b/N_0$ (dB)")
    _plotly_layout(fig, "Curva teórica (BPSK)", height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Preguntas")
    q1 = st.radio("1) Si $E_b/N_0$ aumenta, la BER:", ["Aumenta", "Disminuye", "Se mantiene"], index=None, key="g5_dyn2_q1")
    q2 = st.radio("2) En el punto marcado, el orden aproximado de BER es:", ["≈10⁻¹", "≈10⁻²", "≈10⁻³", "≈10⁻⁴", "<10⁻⁴"], index=None, key="g5_dyn2_q2")
    q3 = st.radio("3) Para BPSK equiprobable, la decisión óptima se basa en el:", ["Signo del estadístico", "Valor absoluto", "Promedio móvil"], index=None, key="g5_dyn2_q3")
    answers = {"q1": q1, "q2": q2, "q3": q3}
    return answers, key["correct"]


def _dyn3_key(seed: int) -> Dict[str, Any]:
    rng = _rng(seed)
    Tb_ms = float(rng.choice([1.0, 2.0, 3.0]))
    Tb = Tb_ms * 1e-3
    fd = float(rng.choice([50.0, 100.0, 200.0, 400.0, 800.0]))
    rho = rho_bfsk(fd, Tb)
    if abs(rho) < 0.15:
        cual = "Casi ortogonales (ρ≈0)"
    elif rho > 0.6:
        cual = "Muy correlacionadas (ρ→1)"
    else:
        cual = "Intermedias"
    correct = {"q1": cual, "q2": "Mejora (BER baja)", "q3": "f_d y T_b"}
    return {"Tb_ms": Tb_ms, "fd": fd, "rho": rho, "correct": correct}


def _render_dyn3():
    st.markdown("### Dinámica 3 — BFSK: correlación y desempeño")

    state = st.session_state.guia5_dinamicas
    if state["dyn3"]["seed"] is None:
        state["dyn3"]["seed"] = int(np.random.randint(0, 2**31 - 1))
        state["dyn3"]["key"] = _dyn3_key(state["dyn3"]["seed"])

    colA, colB = st.columns([1, 1])
    with colA:
        if st.button("Nuevo caso (Dinámica 3)", key="g5_dyn3_new"):
            state["dyn3"]["seed"] = int(np.random.randint(0, 2**31 - 1))
            state["dyn3"]["key"] = _dyn3_key(state["dyn3"]["seed"])
            for k in ["g5_dyn3_q1", "g5_dyn3_q2", "g5_dyn3_q3"]:
                st.session_state.pop(k, None)

    key = state["dyn3"]["key"]
    with colB:
        st.markdown(f"**Caso:** $T_b$={key['Tb_ms']:.0f} ms, $f_d$={key['fd']:.0f} Hz, ρ={key['rho']:.3f}")

    ebn0 = np.arange(0.0, 11.0, 1.0)
    th = ber_teorica_bfsk(10 ** (ebn0 / 10.0), key["fd"], key["Tb_ms"]*1e-3)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ebn0, y=th, mode="lines+markers", name="BER teórica", line=dict(color="#ff7f0e", width=3)))
    fig.update_yaxes(type="log", title_text="BER")
    fig.update_xaxes(title_text="$E_b/N_0$ (dB)")
    _plotly_layout(fig, "Curva teórica (BFSK) para el caso", height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Preguntas")
    q1 = st.radio("1) Con el ρ mostrado, las señales son:", ["Casi ortogonales (ρ≈0)", "Intermedias", "Muy correlacionadas (ρ→1)"], index=None, key="g5_dyn3_q1")
    q2 = st.radio("2) Si ρ disminuye (señales menos correlacionadas), el desempeño:", ["Empeora (BER sube)", "Mejora (BER baja)", "No cambia"], index=None, key="g5_dyn3_q2")
    q3 = st.radio("3) ¿Qué parámetros controlan directamente ρ en este modelo?", ["Solo $E_b/N_0$", "f_d y T_b", "Solo la potencia transmitida"], index=None, key="g5_dyn3_q3")
    answers = {"q1": q1, "q2": q2, "q3": q3}
    return answers, key["correct"]


def render_dinamicas_guia5():
    _init_dyn_state()
    st.markdown("## Actividades dinámicas (evaluación)")
    st.markdown("Completa las 3 dinámicas. Al final, pulsa **Enviar respuestas**.")

    state = st.session_state.guia5_dinamicas

    student_info = _ensure_student_info("g5_form_student")
    if not student_info or not all(student_info.values()):
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
        return

    st.divider()
    ans1, cor1 = _render_dyn1()
    st.divider()
    ans2, cor2 = _render_dyn2()
    st.divider()
    ans3, cor3 = _render_dyn3()
    st.divider()

    if state.get("uploaded"):
        st.info("Ya enviaste estas respuestas ✅")
        last_upload = state.get("last_upload") or {}
        if isinstance(last_upload, dict) and last_upload.get("html_url"):
            st.link_button("Ver archivo en GitHub", last_upload["html_url"])

    if st.button("Enviar respuestas", key="g5_submit", disabled=state.get("uploaded", False)):
        all_answers = [ans1, ans2, ans3]
        missing = 0
        for a in all_answers:
            missing += sum(v is None for v in a.values())
        if missing > 0:
            st.error("Aún faltan respuestas. Completa todas las preguntas antes de enviar.")
            return

        s1 = _score(ans1, cor1)
        s2 = _score(ans2, cor2)
        s3 = _score(ans3, cor3)
        nota = round((s1[1] + s2[1] + s3[1]) / 3.0, 2)

        state["submitted"] = True
        state["last_result"] = {"scores": {"dyn1": s1, "dyn2": s2, "dyn3": s3}, "nota": nota}

        results_payload = {
            "scores": {"dyn1": s1, "dyn2": s2, "dyn3": s3},
            "nota": nota,
            "answers": {"dyn1": ans1, "dyn2": ans2, "dyn3": ans3},
            "correct": {"dyn1": cor1, "dyn2": cor2, "dyn3": cor3},
        }
        content_bytes = _build_results_payload(student_info, results_payload)

        safe_id = _sanitize_filename(student_info.get("id", ""), "sin_id")
        safe_name = _sanitize_filename(student_info.get("name", ""), "sin_nombre")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"guia5_{safe_id}_{safe_name}_{timestamp}.json"
        repo_path = f"guia5/{filename}"
        commit_msg = f"Guía 5 - {safe_id} - {student_info.get('name','')}".strip()

        ok, info = upload_bytes_to_github_results(
            content_bytes=content_bytes,
            repo_path=repo_path,
            commit_message=commit_msg,
        )
        if ok:
            state["uploaded"] = True
            state["last_upload"] = info
            st.success("¡Listo! Respuestas enviadas y archivo subido al repositorio.")
            if isinstance(info, dict) and info.get("html_url"):
                st.link_button("Ver archivo en GitHub", info["html_url"])
            st.write("Ruta en el repositorio:", repo_path)
        else:
            err_msg = info.get("error") if isinstance(info, dict) else str(info)
            st.error(f"No se pudo subir el archivo: {err_msg}")

    if state.get("submitted") and state.get("last_result"):
        res = state["last_result"]
        st.success(f"**Nota final (Guía 5):** {res['nota']}/10")
        st.markdown(
            f"- Dinámica 1: {res['scores']['dyn1'][0]} aciertos → {res['scores']['dyn1'][1]}/10\n"
            f"- Dinámica 2: {res['scores']['dyn2'][0]} aciertos → {res['scores']['dyn2'][1]}/10\n"
            f"- Dinámica 3: {res['scores']['dyn3'][0]} aciertos → {res['scores']['dyn3'][1]}/10"
        )


# ----------------------------
# Render principal de la guía
# ----------------------------

def render_guia5():
    st.title("Guía 5: Fundamentos de transmisión de datos digitales en presencia de ruido")

    tabs = st.tabs(["Objetivos", "Introducción teórica", "Materiales y equipo", "Ejemplos", "Dinámicas", "Conclusiones"])

    with tabs[0]:
        st.markdown(OBJETIVOS_MD)

    with tabs[1]:
        st.markdown(INTRO_MD)

    with tabs[2]:
        st.markdown(MATERIALES_MD)

    with tabs[3]:
        st.markdown("### Ejemplos interactivos")
        sub = st.tabs(["Ejemplo 1", "Ejemplo 2 (BPSK)", "Ejemplo 3 (BFSK)"])
        with sub[0]:
            render_ejemplo1()
        with sub[1]:
            render_ejemplo2()
        with sub[2]:
            render_ejemplo3()

    with tabs[4]:
        render_dinamicas_guia5()

    with tabs[5]:
        st.markdown(CONCLUSIONES_MD)
