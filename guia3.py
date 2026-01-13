# guia3.py
# -*- coding: utf-8 -*-
"""
Guía 3: Fundamentos de probabilidad (versión Streamlit)
Estructura: Objetivos, Introducción teórica, Materiales y equipo,
Ejemplos (1–5), Dinámicas (1–3) y Conclusiones.
"""

import os

import datetime
import importlib.util
import io
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# Constantes generales (compatibles con Guía 1 y 2)
# ---------------------------------------------------------

from pathlib import Path
from github_uploader import upload_bytes_to_github_results

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

# --- Opcional: generación de PDF (igual estilo Guía 1 y 2) ---
REPORTLAB_AVAILABLE = importlib.util.find_spec("reportlab") is not None
if REPORTLAB_AVAILABLE:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas as rcanvas
else:
    letter = None
    rcanvas = None


def export_results_pdf_guia3_bytes(student_info, resultados):
    """
    Genera un solo PDF con el resumen de TODAS las dinámicas de la Guía 3.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre = (student_info.get("name") or "sin_nombre").strip().replace(" ", "_")
    registro = (student_info.get("id") or "sin_id").strip().replace(" ", "_")
    pdf_filename = f"guia3_{registro}_{nombre}_{ts}.pdf"
    buffer = io.BytesIO()

    if not REPORTLAB_AVAILABLE:
        return b"", pdf_filename  # no se puede generar, devolvemos bytes vacíos

    c = rcanvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    left = 40
    top = height - 40
    line_h = 14

    # Marca de agua con logo UCA
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
    c.drawString(left, top, "Resultados Guía 3 – Dinámicas")
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

    # Nota global
    promedio = total_score / max(len(resultados), 1)
    y -= 2 * line_h
    if y < 80:
        c.showPage()
        y = top
    c.setFont("Helvetica-Bold", 10)
    c.drawString(left, y, f"Nota global de la guía (oculta): {promedio:.2f}")

    # Tema del TG
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(width / 2.0, 30, TEMA_TG)

    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes, pdf_filename

# =========================================================
# Textos estáticos
# =========================================================

MATERIALES_COMUNES = """
Para desarrollar las actividades de esta guía interactiva se recomienda contar con:

- Dispositivo con acceso a internet
"""

OBJETIVOS3_TEXT = r"""
### Objetivos

**Objetivo general**

Analizar los fundamentos de probabilidad aplicados al modelado del ruido en sistemas de telecomunicaciones digitales, utilizando simulaciones interactivas que permitan comprender el comportamiento estadístico de variables aleatorias, la construcción de PDF y CDF, la probabilidad condicionada y la aplicación del teorema de Bayes en procesos de detección óptima.

**Objetivos específicos**

- Relacionar la frecuencia relativa y la probabilidad real mediante simulaciones repetitivas, para comprender cómo se modelan experimentalmente los fenómenos aleatorios presentes en el ruido.

- Diferenciar y caracterizar variables aleatorias discretas, continuas y mixtas, visualizando sus funciones de distribución (CDF) y densidad (PDF) para establecer bases sólidas en la descripción matemática de señales aleatorias.

- Modelar el ruido térmico mediante la variable aleatoria Gaussiana, comparando histogramas experimentales con la PDF y CDF teóricas y analizando cómo la media y la varianza determinan el comportamiento del ruido.

- Aplicar la probabilidad condicionada y la independencia para describir la relación entre símbolos transmitidos y valores recibidos en un canal digital afectado por ruido, interpretando probabilidades de error y decisiones bajo incertidumbre.

- Utilizar el teorema de Bayes para implementar un detector MAP (Maximum A Posteriori) en presencia de ruido Gaussiano, interpretando cómo las probabilidades a priori y la PDF del ruido influyen en las decisiones óptimas del receptor.
"""

INTRO3_TEXT = r"""
#### Introducción teórica – Fundamentos de probabilidad y variables aleatorias

Hasta este momento se han estudiado varios tipos de señales, entre ellas las señales determinísticas, es decir, aquellas cuyo valor en cualquier instante \(t\) está completamente determinado, ya sea por su descripción gráfica o analítica. Estas señales, que pueden ser especificadas en cualquier instante \(t\), no pueden transmitir información por sí solas. 

La teoría de la información que es el estudio matemático de la cuantización, almacenamiento y comunicación de la información establece que la información está inherentemente ligada a la incertidumbre. Entre mayor sea la incertidumbre de una señal a recibir, mayor es la información que contiene. Si un mensaje por recibir está completamente especificado, entonces no contiene incertidumbre y no transmite nueva información al receptor. Por lo tanto, las señales que realmente transmiten información deben ser, en algún sentido, impredecibles.

Existe además un tipo de señal que se “adhiere” a las señales que contienen la información que se desea transmitir: la señal de ruido. Esta señal también es impredecible. Tanto las señales de información como las señales de ruido son ejemplos de procesos estocásticos (aleatorios), cuyo papel en los sistemas de comunicación es crucial, ya que determinan la confiabilidad y el desempeño del enlace.

En esta guía se revisan los fundamentos de probabilidad y variables aleatorias necesarios para modelar matemáticamente estas señales aleatorias (información y ruido) y para construir modelos de canal y ruido que luego se utilizarán en el análisis de desempeño de sistemas de telecomunicaciones digitales.


##### 1. Álgebra de conjuntos y operaciones básicas

La teoría de la probabilidad se construye sobre la teoría de conjuntos. Un conjunto es una colección de elementos. Es común representar un conjunto con letras mayúsculas, como A, B, C, etc. y los elementos de este se representan con una letra minúscula como a, b, c, etc. 

El número de elementos de un conjunto se denomina cardinal; este puede ser finito o infinito. También es común analizar relaciones entre conjuntos:
- Pertenencia: Los conjuntos tienen una relación de pertenencia que se denota con el símbolo ∈, por ejemplo, si el elemento b pertenece al conjunto B se expresa matemáticamente como  b ∈ B 
- Inclusión: Se dice que un conjunto A esta incluido en un conjunto B si todos los elementos de A están en B. Si el caso es tal, se expresa de forma abreviada la relación A ⊂ B
- Igualdad: Si la relación entre conjuntos cumple tanto A ⊂ B como B ⊂ A de forma simultánea, se dice que los conjuntos son iguales
- Conjuntos mutuamente excluyentes: Si dos conjuntos no tienen ningún elemento en común se denominan conjuntos mutuamente excluyentes


Las operaciones fundamentales entre conjuntos son:

Sean A y B dos conjuntos.

- Intersección: Denotado por A  ∩ B , es el conjunto formado por los elementos que pertenecen a ambos conjuntos simultáneamente
- Union: Denotado por A ∪ B, es el conjunto formado por los elementos que pertenecen a los dos conjuntos.
- Diferencia: Denotado por C=A-B. Un conjunto diferencia de dos conjuntos A y B es aquel que está formado por los elementos del primer conjunto que no están en el segundo conjunto 
- Complemento: Denotado por Ā=S-A , se denomina conjunto complemento a aquel conjunto diferencia entre el conjunto universal S y el conjunto A

Las operaciones de unión e intersección cumplen las propiedades conmutativa, asociativa y distributiva.


##### 2. Definiciones fundamentales de probabilidad

Antes de introducir la probabilidad, se definen algunos conceptos básicos:

- **Experimento aleatorio**: es aquel en el cual no puede predecirse con certeza el resultado, aunque el conjunto de resultados posibles sí sea conocido.
- **Espacio muestral** \(S\): conjunto de todos los resultados posibles del experimento.
- **Suceso (o evento)**: subconjunto del espacio muestral \(S\) (un conjunto de resultados).

###### 2.1 Definición clásica de probabilidad

La definición clásica se basa en un análisis “a priori” del problema, en lugar de la experimentación. Dados un experimento aleatorio y un suceso \(A\), la probabilidad clásica de \(A\) se define por la **ecuación (1)**:

$$
P(A) = \frac{N_A}{N} \tag{1}
$$

donde:
- P(A) es la probabilidad de que ocurra el suceso A
- NA es el número de casos favorables a A
- N es el número total de casos posibles (equiprobables).

Esta definición es intuitiva, pero solo es válida cuando los resultados son equiprobables y el espacio muestral es finito.

###### 2.2 Frecuencia relativa

En la práctica, muchas veces la probabilidad se estima repitiendo el experimento y contando cuántas veces se presenta un suceso. Si un experimento se repite N veces y el suceso A ocurre en NA de ellas, la frecuencia relativa de A se define en la **ecuación (2)**:

$$
f_A = \frac{N_A}{N} \tag{2}
$$

Cuando N → ∞, la frecuencia relativa fA tiende, bajo condiciones adecuadas, a la probabilidad real P(A). Este enfoque conecta la probabilidad con la observación empírica y la idea de “proporción de veces” que ocurre un suceso.

###### 2.3 Definición axiomática de probabilidad

La definición axiomática de probabilidad establece que la probabilidad es una función que asigna a cada suceso un número real que cumple tres axiomas fundamentales. 

Para poder manejar espacios muestrales más generales (por ejemplo, infinitos o no equiprobables), se utiliza la definición axiomática. Un axioma es una proposición tan clara y general que se acepta sin demostración. En el enfoque axiomático:

Dado un experimento con un espacio muestral S asociado, la probabilidad es una función que asocia a cada suceso del espacio muestral un número real, y que cumple tres axiomas definidos a continuación 

1. 	Para todos los sucesos A en el espacio muestral S se cumple la desigualdad:

$$
0 \leq P(A) \leq 1 
$$

2. 2.	La probabilidad de que todos los sucesos posibles ocurran es la unidad, según la **ecuación (3)**:

$$
P(S) = 1 \tag{3}
$$

3. Si la ocurrencia del suceso A antecede la ocurrencia del suceso B y viceversa, entonces se cumple la ecuación  **ecuación (4)**:

$$
P(A \cup B) = P(A) + P(B) \tag{4}
$$

Estos axiomas se pueden extender a la unión de una colección numerable de sucesos disjuntos. A partir de ellos se construye el resto de la teoría de probabilidad.


##### 3. Espacio de probabilidad 

Un experimento aleatorio es caracterizado por un espacio de probabilidad que consiste en un espacio muestral S, una clase de sucesos denominada F que se pueden extraer del espacio muestral y una medida de probabilidad P, el espacio de probabilidad se denota como  <S,F,P>. En términos más simples, el espacio de probabilidad es un modelo matemático que representa un fenómeno aleatorio.


##### 4. Probabilidad condicionada e independencia


###### 4.1 Probabilidad condicionada

La probabilidad condicionada mide cómo cambia el conocimiento probabilístico de un suceso A cuando se sabe que otro suceso B ya ha ocurrido.

Si definimos la probabilidad de A sabiendo que B ha ocurrido como P(A|B), su definición formal para P(B) > 0 viene dada por la **ecuación (5)**:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)} \tag{5}
$$

Aquí:

- P(A|B) representa la probabilidad condicionada
- P(B) representa la probabilidad a priori
- P(A ∩ B) representa la probabilidad de que ocurran simultáneamente dos eventos, A y B

###### 4.2 Independencia de sucesos

Sea A y B dos sucesos con probabilidades distintas de cero. Se dice que A y B son independientes si el conocimiento de que uno ocurre no altera la probabilidad del otro. Matemáticamente, esto se expresa en la **ecuación (6)**:

$$
A \text{ y } B \text{ son independientes } \Longleftrightarrow P(A \mid B) = P(A) \tag{6}
$$

Usando la ecuación (5), esta condición equivale a la **ecuación (7)**:

$$
P(A \cap B) = P(A)\,P(B) \tag{7}
$$

Es decir, la probabilidad conjunta de A y B es el producto de las probabilidades individuales.


##### 5. Teoremas fundamentales: probabilidad total y Bayes

###### 5.1 Partición del espacio muestral

Una partición del espacio muestral es una colección de subconjuntos que son mutuamente excluyentes y que su unión forma todo el espacio muestral, dividiendo el conjunto de todos los resultados posibles de un experimento en partes que no dejan huecos ni elementos repetidos.


###### 5.2 Teorema de la probabilidad total

Sea A un suceso cualquiera y Bi,...,BN es una partición de S con P(Bi > 0). Entonces, la probabilidad de A puede expresarse como la suma ponderada de las probabilidades condicionadas se define en la **ecuación (8)**

$$
P(A) = \sum_{i=1}^{N} P(A \mid B_i)\,P(B_i) \tag{8}
$$

###### 5.3 Teorema de Bayes

El teorema de Bayes permite invertir probabilidades condicionadas. Supongamos que queremos la probabilidad de Bi dado que ha ocurrido A, es decir, P(Bi|A), y conocemos P(A|B_i). Entonces, la **ecuación (9)** establece:

$$
P(B_i \mid A) = \frac{P(A \mid B_i)\,P(B_i)}{\displaystyle\sum_{k=1}^{N} P(A \mid B_k)\,P(B_k)} \tag{9}
$$

Este teorema es fundamental para sistemas de detección optima en telecomunicaciones digitales, donde a partir de una observación (por ejemplo, una señal recibida) se actualizan las probabilidades de haber transmitido determinado símbolo o mensaje.


##### 6. Variables aleatorias

Una variable aleatoria (VA) es una regla que asigna un valor numérico a cada posible resultado del experimento aleatorio. De forma más formal, una VA \(X\) es una función que mapea cada elemento del espacio muestral S a un número real, como se muestra en la **ecuación (10)**:

$$
X: S \to \mathbb{R} \tag{10}
$$

Las variables aleatorias permiten cuantificar señales aleatorias como el ruido, que se manifiesta como fluctuaciones en voltaje, corriente o amplitud en sistemas de telecomunicaciones.

Las VA pueden ser:

- **Discretas**: toman valores en un conjunto numerable (por ejemplo, el número de errores en un bloque de bits).
- **Continuas**: pueden tomar cualquier valor en un intervalo de la recta real (por ejemplo, el voltaje instantáneo de ruido térmico).
- **Mixtas**: combinan componentes discretas y continuas.

Caracterizar una VA significa describir su comportamiento probabilístico, es decir, cómo se distribuyen los valores que puede tomar.


##### 7. Función de distribución (CDF)

La función de distribución acumulada (CDF, por sus siglas en inglés) de una VA X se denota por F**X**(x) y se define como la probabilidad de que X tome un valor menor o igual que x, tal como se muestra en la **ecuación (11)**:

$$
F_X(x) = P(X \leq x) \tag{11}
$$

La función F**X**(x):

- Es una función real de variable real.
- Es no decreciente.
- Satisface:
  - FX(-∞) = 0
  - FX(∞) = 1

La función de distribución permite hacer cualquier cálculo probabilístico con relación a los valores que puede tomar la variable aleatoria **X**,  por tanto, contiene toda la información probabilística de la VA.


##### 8. Función de densidad (PDF)

Para variables aleatorias continuas, la función de densidad de probabilidad (PDF) f**X**(x) se define a partir de la CDF como la derivada, según la **ecuación (12)**:

$$
f_X(x) = \frac{d}{dx} F_X(x) \tag{12}
$$

La PDF mide, de forma aproximada, cuánta probabilidad por unidad de longitud hay alrededor de un punto x. Para intervalos pequeños Δx:

$$
P(x \leq X \leq x + \Delta x) \approx f_X(x)\,\Delta x
$$

La PDF cumple:

- f**X**(x)≥ 0 para todo x.
- El área bajo una función de densidad de probabilidad es siempre unitaria

En el contexto de ruido, la PDF describe cómo se distribuyen los valores instantáneos del ruido en el tiempo.


##### 9. Variable aleatoria Gaussiana o normal

La VA más importante en sistemas de telecomunicaciones es la variable aleatoria Gaussiana o normal. Proporciona una excelente aproximación a muchos fenómenos físicos; por ejemplo, el ruido (voltaje) generado por la agitación térmica de electrones libres en conductores.

Una VA Gaussiana X con media mx y varianza σ^2 se denota como X(mx,σ^2). Una VA Gaussiana es una VA continua cuya función de densidad está definida para todos los números reales. La PDF de una VA Gaussiana se da en la **ecuación (13)**:

$$
f_X(x) = \frac{1}{\sqrt{2\pi\,\sigma^2}}\,
\exp\!\left(-\,\frac{(x - \mu)^2}{2\,\sigma^2}\right) \tag{13}
$$

Una VA Gaussiana está completamente caracterizada por su media mx y su varianza σ^2. A su PDF se le conoce como “campana de Gauss”. La media desplaza el centro de la campana a lo largo del eje x, mientras que la varianza controla la dispersión de los valores que puede tomar X, una varianza grande implica que X puede tomar valores más alejados de la media; una varianza pequeña concentra la mayor parte de la probabilidad cerca de mx.


##### 10. Transformación de variables aleatorias

En el análisis de desempeño de sistemas de telecomunicaciones, es común que una variable aleatoria (por ejemplo, una señal de entrada \(X\)) se transforme mediante un sistema o dispositivo para producir otra VA \(Y\). Si la relación entre ellas es:

$$
Y = g(X) \tag{15}
$$

entonces, caracterizar \(Y\) implica encontrar su CDF y su PDF a partir de la distribución de \(X\).

Una técnica general para obtener la distribución de \(Y\) consiste en dos pasos:

**Paso 1 , CDF de Y:**  
Se parte de la definición de la **ecuación (16)**:

$$
F_Y(y) = P(Y \leq y) = P(g(X) \leq y) \tag{16}
$$

y se escribe esta probabilidad como una o varias integrales sobre la densidad de X. En general, si Γ es el conjunto de valores de x tales que g(x) < y, entonces se cumple la **ecuación (17)**:

$$
F_Y(y) = \int_{\Gamma(y)} f_X(x)\,dx \tag{17}
$$

**Paso 2, PDF de Y:**  
La PDF de Y se obtiene derivando la CDF, como se indica en la **ecuación (18)**:

$$
f_Y(y) = \frac{d}{dy} F_Y(y) \tag{18}
$$

Cuando los límites de la integral de la ecuación (17) dependen de y, se utiliza la regla de Leibniz para derivar integrales con límites variables. En términos generales, si:

$$
I(y) = \int_{a(y)}^{b(y)} f(x,y)\,dx, \tag{19}
$$

entonces la regla de Leibniz establece la **ecuación (20)**:


$$
\frac{dI(y)}{dy} = f\bigl(b(y),y\bigr)\,b'(y)\;-\;f\bigl(a(y),y\bigr)\,a'(y)\;+\;\int_{a(y)}^{b(y)} \frac{\partial f(x,y)}{\partial y}\,dx \tag{20}
$$

Aplicando esta regla a la CDF de Y, se obtiene la PDF de la VA transformada.
"""

CONCLUSIONES3_TEXT = r"""
### Conclusiones

- La guía permitió vincular los conceptos fundamentales de probabilidad con el modelado del ruido en sistemas de telecomunicaciones digitales, mostrando que los fenómenos aleatorios solo pueden describirse de forma rigurosa a través de variables aleatorias, PDF y CDF.

- Mediante simulaciones de frecuencia relativa, variables discretas, continuas y mixtas, y variables Gaussianas, el estudiante pudo observar cómo las distribuciones de probabilidad resumen el comportamiento de grandes conjuntos de muestras, y cómo la media y la varianza gobiernan la intensidad y dispersión del ruido térmico.

- El análisis de probabilidades condicionadas en un canal digital y la implementación de un detector MAP basados en el teorema de Bayes ilustraron que las decisiones óptimas en el receptor dependen tanto de la estadística del ruido (modelo Gaussiano) como de las probabilidades a priori de los símbolos transmitidos, conectando directamente la teoría de probabilidad con el diseño práctico de sistemas de comunicación digitales.
"""

# =========================================================
# Utilidades comunes
# =========================================================

def _ensure_student_info(form_key: str):
    """
    Asegura que exista st.session_state["student_info"] con las claves
    name, id, dob, y muestra el formulario si hace falta.
    Devuelve el diccionario o None si no se completó.

    form_key: key único para el formulario (por dinámica).
    """
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



# =========================================================
# EJEMPLOS
# =========================================================

def render_ejemplo1():
    """
    Ejemplo 1 — Frecuencia relativa vs probabilidad real.
    """
    st.markdown("### Ejemplo 1 - Frecuencia relativa vs. probabilidad clasica")

    with st.expander("Descripción y pasos", expanded=True):
        st.markdown(
            "En este ejemplo se simula un experimento aleatorio (lanzamiento de moneda o dado) para ilustrar "
            "cómo la **frecuencia relativa** converge hacia la **probabilidad real** al aumentar el número de ensayos.\n\n"
            "**Pasos sugeridos:**\n"
            "1. Elige el tipo de experimento **(moneda o dado).**\n"
            "2. Selecciona el número de lanzamientos **N**.\n"
            "3. Pulsa **Simular** para generar los resultados.\n"
            "4. Observa la gráfica de frecuencia relativa y compárala con la probabilidad teórica.\n"
            "5. Repite el experimento cambiando N para analizar la convergencia estadística."
        )

    col1, col2 = st.columns([1, 2])

    with col1:
        tipo = st.selectbox("Tipo de experimento", ["Moneda (cara)", "Dado (resultado = 6)"])
        N = st.slider("Número de lanzamientos N", min_value=10, max_value=5000, value=200, step=10)
        run = st.button("Simular experimento", key="g3_ej1_run")

    if run:
        if tipo.startswith("Moneda"):
            p_teor = 0.5
            # 1 = cara, 0 = cruz
            muestras = np.random.rand(N) < p_teor
            evento = "cara"
        else:
            p_teor = 1.0 / 6.0
            # 1 = sale 6, 0 = no sale 6
            tiradas = np.random.randint(1, 7, size=N)
            muestras = (tiradas == 6)
            evento = "resultado 6"

        frec_rel = np.cumsum(muestras) / np.arange(1, N + 1)

        with col2:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=np.arange(1, N + 1),
                    y=frec_rel,
                    mode="lines",
                    name="Frecuencia relativa",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[1, N],
                    y=[p_teor, p_teor],
                    mode="lines",
                    name="Probabilidad teórica",
                    line=dict(color="red", dash="dash"),
                )
            )
            fig.update_layout(
                title="Convergencia de frecuencia relativa al modelar el experimento",
                xaxis_title="Número de ensayos",
                yaxis_title="Frecuencia relativa",
                height=320,
                margin=dict(l=40, r=20, t=60, b=40),
                hovermode="x unified",
            )
            plot_theme = _get_plot_theme()
            _apply_plot_theme(fig, plot_theme, font_size=12)
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with st.expander("Explicación de la simulación y preguntas", expanded=True):
            st.markdown(
                f"En este experimento se representa un proceso aleatorio discreto mediante una sucesión de ensayos independientes, como el lanzamiento de una moneda o de un dado. En cada ensayo se evalúa la ocurrencia de un evento de interés A, por ejemplo, la obtención de “cara” en el caso de la moneda.Aunque la probabilidad teórica P(A) es conocida a partir del modelo probabilístico, el resultado de cada ensayo individual no puede predecirse con certeza, reflejando la naturaleza aleatoria del fenómeno.\n\n"
                "A partir de los datos generados se calcula la frecuencia relativa "
                "fN(A)=N(A)/N, que representa una estimación empírica de la probabilidad. "
                "Cuando el número de ensayos es pequeño, esta estimación presenta variaciones significativas "
                "debido a las fluctuaciones aleatorias propias del proceso. Sin embargo, a medida que el número "
                "de ensayos aumenta, la frecuencia relativa tiende a estabilizarse alrededor de la probabilidad "
                "teórica.\n\n"
                "Este comportamiento ilustra el fundamento experimental de la ley de los grandes números "
            )

            st.markdown("#### Preguntas y respuestas")
            st.markdown(
                "- **1. ¿Por qué la frecuencia relativa varía tanto para valores pequeños de N?**  \n"
                "  **R:** Porque con pocos ensayos la muestra disponible es limitada y las fluctuaciones aleatorias dominan; "
                "no hay suficiente información estadística para que el promedio se estabilice.\n\n"
                "- **2. ¿Qué sucede con la frecuencia relativa cuando N es muy grande?**  \n"
                "  **R:** Se aproxima a la probabilidad real P(A), mostrando el principio de la ley de los grandes números.\n\n"
                "- **3. ¿Que es la ley de los numeros grandes?**  \n"
                "**R:** La ley de los grandes números, en probabilidad, establece que los resultados de una prueba en una muestra se acercan al promedio de la población total a medida que aumenta el tamaño de la muestra. Es decir, se vuelve más representativa de la población en su conjunto.\n\n"
                "- **4. ¿Cómo se relaciona este experimento con el modelado del ruido en un canal de comunicación?**  \n"
                "  **R:** El ruido, igual que el resultado de un lanzamiento, es impredecible muestra a muestra, pero su "
                "distribución de probabilidad se puede estimar por frecuencias relativas a partir de muchas "
                "realizaciones."
            )

def render_ejemplo2():
    """
    Ejemplo 2 — PDF y CDF de variables discretas, continuas y mixtas.
    """
    st.markdown("### Ejemplo 2 - PDF y CDF de variables discretas, continuas y mixtas")

    with st.expander("Descripción y pasos", expanded=True):
        st.markdown(
            "Este ejemplo construye y compara la **PDF** y la **CDF** de tres tipos de variables aleatorias:\n"
            "- Una variable **discreta** (resultado de un dado).\n"
            "- Una variable **continua** (uniforme en un intervalo).\n"
            "- Una variable **mixta** (un valor puntual con probabilidad fija y una parte continua).\n\n"
            "**Pasos sugeridos:**\n"
            "1. Selecciona el **número de muestras** a generar.\n"
            "2. Ajusta el **intervalo** de la parte continua.\n"
            "3. Pulsa **Simular** para generar las muestras y graficar PDF y CDF.\n"
            "4. Compara la forma escalonada de la CDF discreta con la CDF continua y mixta."
        )

    col1, col2 = st.columns([1, 2])

    with col1:
        N = st.slider("Número de muestras", min_value=100, max_value=50000, value=5000, step=100)
        a = st.number_input("Límite inferior del intervalo continuo a", value=0.0, format="%.2f")
        b = st.number_input("Límite superior del intervalo continuo b", value=1.0, format="%.2f")
        peso_puntual = st.slider("Probabilidad del valor puntual en VA mixta", 0.0, 0.9, 0.3, 0.05)
        run = st.button("Simular variables", key="g3_ej2_run")

    if run:
        if b <= a:
            st.error("Debe cumplirse b > a para el intervalo continuo.")
            return

        # Discreta: dado (1..6)
        Xd = np.random.randint(1, 7, size=N)

        # Continua: uniforme en [a, b]
        Xc = np.random.uniform(a, b, size=N)

        # Mixta: valor puntual + uniforme
        valor_puntual = a  # por simplicidad
        U = np.random.rand(N)
        Xm = np.empty(N)
        mask_p = U < peso_puntual
        Xm[mask_p] = valor_puntual
        Xm[~mask_p] = np.random.uniform(a, b, size=np.sum(~mask_p))

        def emp_cdf(x):
            xs = np.sort(x)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            return xs, ys

        with col2:
            tabs = st.tabs(["Discreta (dado)", "Continua (uniforme)", "Mixta"])
            plot_theme = _get_plot_theme()

            # Discreta
            with tabs[0]:
                valores, cuentas = np.unique(Xd, return_counts=True)
                xs, ys = emp_cdf(Xd)
                fig1 = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=False,
                    vertical_spacing=0.16,
                    subplot_titles=("PDF empírica", "CDF empírica"),
                )
                fig1.add_trace(
                    go.Bar(x=valores, y=cuentas / N, name="PDF empírica"),
                    row=1,
                    col=1,
                )
                fig1.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        name="CDF empírica",
                        line_shape="hv",
                    ),
                    row=2,
                    col=1,
                )
                fig1.update_xaxes(title_text="Resultado", row=1, col=1)
                fig1.update_yaxes(title_text="Probabilidad", row=1, col=1)
                fig1.update_xaxes(title_text="x", row=2, col=1)
                fig1.update_yaxes(title_text="F(x)", row=2, col=1)
                fig1.update_layout(height=420, margin=dict(l=40, r=20, t=70, b=50))
                _apply_plot_theme(fig1, plot_theme, font_size=12)
                fig1.update_annotations(font=dict(color=plot_theme["font_color"], size=12))
                st.plotly_chart(fig1, use_container_width=True, theme=None)

            # Continua
            with tabs[1]:
                xs, ys = emp_cdf(Xc)
                fig2 = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=False,
                    vertical_spacing=0.16,
                    subplot_titles=("PDF empírica", "CDF empírica"),
                )
                fig2.add_trace(
                    go.Histogram(
                        x=Xc,
                        nbinsx=40,
                        histnorm="probability density",
                        name="PDF empírica",
                        opacity=0.7,
                    ),
                    row=1,
                    col=1,
                )
                fig2.add_trace(
                    go.Scatter(x=xs, y=ys, mode="lines", name="CDF empírica"),
                    row=2,
                    col=1,
                )
                fig2.update_xaxes(title_text="x", row=1, col=1)
                fig2.update_yaxes(title_text="f(x)", row=1, col=1)
                fig2.update_xaxes(title_text="x", row=2, col=1)
                fig2.update_yaxes(title_text="F(x)", row=2, col=1)
                fig2.update_layout(height=420, margin=dict(l=40, r=20, t=70, b=50))
                _apply_plot_theme(fig2, plot_theme, font_size=12)
                fig2.update_annotations(font=dict(color=plot_theme["font_color"], size=12))
                st.plotly_chart(fig2, use_container_width=True, theme=None)

            # Mixta
            with tabs[2]:
                xs, ys = emp_cdf(Xm)
                fig3 = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=False,
                    vertical_spacing=0.16,
                    subplot_titles=("PDF empírica", "CDF empírica"),
                )
                fig3.add_trace(
                    go.Histogram(
                        x=Xm,
                        nbinsx=40,
                        histnorm="probability density",
                        name="PDF empírica",
                        opacity=0.7,
                    ),
                    row=1,
                    col=1,
                )
                fig3.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        name="CDF empírica",
                        line_shape="hv",
                    ),
                    row=2,
                    col=1,
                )
                fig3.update_xaxes(title_text="x", row=1, col=1)
                fig3.update_yaxes(title_text="f(x)", row=1, col=1)
                fig3.update_xaxes(title_text="x", row=2, col=1)
                fig3.update_yaxes(title_text="F(x)", row=2, col=1)
                fig3.update_layout(height=420, margin=dict(l=40, r=20, t=70, b=50))
                _apply_plot_theme(fig3, plot_theme, font_size=12)
                fig3.update_annotations(font=dict(color=plot_theme["font_color"], size=12))
                st.plotly_chart(fig3, use_container_width=True, theme=None)

        with st.expander("Explicación de la simulación y preguntas", expanded=True):
            st.markdown(
                "Caracterizar una variable aleatoria consiste en describir su comportamiento probabilístico mediante funciones que determinan cómo se distribuyen sus valores. La herramienta más general es la función de distribución acumulada (CDF). Cuando la variable es continua, también puede describirse mediante una función de densidad de probabilidad (PDF); y cuando es discreta, mediante una función de delta de diracs que representan una masa de probabilidad.\n\n"
                "**¿Que se observa en las gráficas?**\n\n"
                "Las variables discretas producen una **CDF** escalonada con saltos en los valores posibles y un tren de delta diracs para la **PDF** \n\n"
                "las variables continuas generan una **CDF** suave y una **PDF** que se interpreta como densidad de probabilidad.\n\n "
                "La variable mixta combina ambas características: presenta un salto en el valor puntual con probabilidad asignada y "
                "una parte continua en el resto del intervalo.\n\n"
                "En general, la función de distribución acumulada (CDF) de una variable aleatoria representa la probabilidad acumulada de que dicha variable tome valores menores o iguales a un cierto umbral. Para variables discretas, la CDF se obtiene sumando las probabilidades de todos los valores posibles hasta ese punto. Para variables continuas, la CDF se define como la integral de la densidad de probabilidad. En el caso mixto, la CDF combina incrementos discretos con tramos continuos, describiendo completamente el comportamiento probabilístico de la variable\n\n"
                "Para variables continuas, la PDF describe cómo se distribuye la probabilidad “por unidad de valor” y permite obtener probabilidades integrando sobre intervalos; además, es la derivada de la CDF cuando esta es derivable. Para variables discretas no existe PDF como tal, sino una función de delta de diracs que representan una masa de probabilidad que asigna probabilidad a valores puntuales. En variables mixtas, la parte continua se modela con una PDF y la parte discreta con masas puntuales (saltos en la CDF)."
            )

            st.markdown("#### Preguntas y respuestas")
            st.markdown(
                "- **1. ¿Por qué la CDF de la VA discreta tiene forma de escalera?**  \n"
                "  **R:** Porque la probabilidad se concentra en valores aislados, cada valor posible aporta un salto de magnitud igual a su probabilidad.\n\n"
                "- **2. ¿Qué representa el área bajo la PDF en el caso continuo?**  \n"
                "  **R:** Representa la probabilidad total, por definición, el área bajo la PDF en todo el eje real es igual a 1.\n\n"
                "- **3. ¿Qué rasgo distingue visualmente a una VA mixta en su CDF?**  \n"
                "  **R:** La presencia de un salto finito en un punto específico, superpuesto a una parte continua que crece suavemente."
            )


def render_ejemplo3():
    """
    Ejemplo 3 — PDF y CDF de una VA Gaussiana (ruido térmico).
    Trabaja con varianza σ^2 (entrada del usuario) y fija escala para visualizar ensanchamiento.
    """
    st.markdown("### Ejemplo 3 - VA Gaussiana: PDF y CDF del ruido térmico")

    # Parámetros para fijar escala (ajústalos si lo deseas)
    SIGMA2_MIN = 0.01
    SIGMA2_MAX = 16.0  # varianza máxima mostrada (σ_max = 4)
    SIGMA_MAX = float(np.sqrt(SIGMA2_MAX))
    XSPAN = 4 * SIGMA_MAX  # ventana fija: ±4σ_max alrededor de la media

    # Límite superior fijo de la PDF usando la varianza mínima (pico máximo posible)
    SIGMA_MIN = float(np.sqrt(SIGMA2_MIN))
    PDF_MAX = 1.0 / (np.sqrt(2 * np.pi) * SIGMA_MIN)  # pico de N(mu, sigma_min^2)
    YLIM_PDF = (0.0, 1.05 * PDF_MAX)

    with st.expander("Descripción y pasos", expanded=True):
        st.markdown(r"""
Este ejemplo modela el ruido térmico como una variable aleatoria Gaussiana.

Una VA Gaussiana queda completamente caracterizada por su **media m** y su **varianza σ^2** . 

La **PDF** asociada tiene la forma conocida como **campana de Gauss**, y la **CDF** representa la probabilidad acumulada P(X≤x).

Una variable aleatoria Gaussiana X de media m y varianza σ^2 se denota como:
$$
X \sim \mathcal{N}(m,\sigma^2)
$$

**Pasos sugeridos:**
1. Ajusta la **media m** y la **varianza σ^2** .
2. Define el **número de muestras** N.
3. Pulsa **Simular** para graficar las funciones de distribución y densidad.
4. Observa las graficas y verifica la retroalimentación.
        """)

    col1, col2 = st.columns([1, 2])

    with col1:
        mu = st.number_input("Media m", value=0.0, format="%.2f")
        sigma2 = st.number_input(
            "Varianza σ^2",
            value=5.0,
            format="%.2f",
            min_value=SIGMA2_MIN,
            max_value=SIGMA2_MAX
        )
        N = st.slider("Número de muestras", min_value=500, max_value=100000, value=10000, step=500)
        run = st.button("Simular ruido Gaussiano", key="g3_ej3_run")

    if run:
        sigma = float(np.sqrt(sigma2))

        # Simulación
        X = np.random.normal(loc=mu, scale=sigma, size=N)

        # Malla fija (para que NO cambie el rango con σ)
        xs = np.linspace(mu - XSPAN, mu + XSPAN, 600)

        # PDF teórica en la malla fija
        pdf_teo = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((xs - mu) / sigma) ** 2)

        # CDF empírica
        Xs = np.sort(X)
        cdf_emp = np.arange(1, N + 1) / N

        # CDF teórica en la malla fija
        from scipy.stats import norm
        cdf_teo = norm.cdf(xs, loc=mu, scale=sigma)

        with col2:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.18,
                subplot_titles=("PDF de la VA Gaussiana (ruido térmico)", "CDF de la VA Gaussiana"),
            )
            fig.add_trace(
                go.Histogram(
                    x=X,
                    nbinsx=50,
                    histnorm="probability density",
                    name="Histograma (PDF empírica)",
                    opacity=0.6,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=xs, y=pdf_teo, mode="lines", name="PDF teórica", line=dict(dash="dash")),
                row=1,
                col=1,
            )
            fig.update_xaxes(title_text="x", range=[mu - XSPAN, mu + XSPAN], row=1, col=1)
            fig.update_yaxes(title_text="f_X(x)", range=[0.0, 1.15 * np.max(pdf_teo)], row=1, col=1)

            fig.add_trace(
                go.Scatter(x=Xs, y=cdf_emp, mode="lines", name="CDF empírica"),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=xs, y=cdf_teo, mode="lines", name="CDF teórica", line=dict(color="red", dash="dash")),
                row=2,
                col=1,
            )
            fig.update_xaxes(title_text="x", range=[mu - XSPAN, mu + XSPAN], row=2, col=1)
            fig.update_yaxes(title_text="F_X(x)", range=[0.0, 1.0], row=2, col=1)

            fig.update_layout(height=520, margin=dict(l=40, r=20, t=70, b=50), hovermode="x unified")
            plot_theme = _get_plot_theme()
            _apply_plot_theme(fig, plot_theme, font_size=12)
            fig.update_annotations(font=dict(color=plot_theme["font_color"], size=12))
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with st.expander("Explicación de la simulación y preguntas", expanded=True):
            st.markdown(
                r"Una variable aleatoria Gaussiana se define por dos parámetros: la **media m** y la **varianza "
                r"σ^2**. Con estos dos valores queda determinada toda su distribución: la **PDF** (campana de Gauss) "
                r"y la **CDF**, que acumula probabilidades P(X≤x). "
                r"En la simulación se generan N realizaciones del ruido térmico y el histograma normalizado aproxima la "
                r"**PDF empírica**. Al aumentar N, la forma empírica se estabiliza y se acerca a la **PDF teórica**."
                r"La media representa el valor promedio alrededor del cual se distribuyen los posibles valores de la variable aleatoria, desplaza el centroide de la campana y la varianza es una medida de la dispersión de los valores que alcanza la variable aleatoria, una varianza grande significa que la VA probablemente tomará valores lejanos de la media y una varianza pequeña significa que una gran cantidad de valores de la variable aleatoria estarán cerca de la media. "
            )

            st.markdown("#### Preguntas y respuestas")
            st.markdown(
                "- **1. ¿Qué efecto tiene incrementar la varianza sobre la forma de la campana de Gauss?**  \n"
                "  **R:** La distribución se ensancha y el pico disminuye; las muestras quedan más dispersas alrededor de la media.\n\n"
                "- **2. ¿Qué interpretación tiene el área bajo la PDF entre dos valores \(a\) y \(b\)?**  \n"
                "  **R:** Representa la probabilidad de que el ruido tome valores en el intervalo [a,b], es decir, P(a ≤ X ≤ b).\n\n"
            )


def render_ejemplo4():
    """
    Ejemplo 4 — Probabilidad condicionada y Bayes con un detector (alarma) en presencia de ruido.
    Interpretación telecom: detector de presencia de señal.
    """
    st.markdown("### Ejemplo 4 - Probabilidad condicionada: detector de señal (alarma) con ruido")

    with st.expander("Descripción y pasos", expanded=True):
        st.markdown(
            "Este ejemplo modela un **detector binario** que intenta decidir si hay señal o no hay señal.\n\n"
            "- El estado real (S) representa la realidad del canal: hay señal o no hay señal.\n"
            "- La decisión del detector (R) representa lo que el receptor concluye: detectó o no detectó.\n\n"
            "La idea central es entender dos preguntas distintas:\n"
            "1) **Probabilidad condicionada del detector**: conociendo la realidad, ¿qué tan seguido se equivoca?\n"
            "2) **Probabilidad a posteriori (Bayes)**: conociendo la decisión del detector, ¿qué tan probable es que la realidad haya sido ‘hay señal’?\n\n"
            "**Pasos sugeridos:**\n"
            "1. Ajusta la probabilidad de que realmente exista señal.\n"
            "2. Ajusta la probabilidad de **falsa alarma** y la de **pérdida**.\n"
            "3. Elige cuántos **intentos** simular.\n"
            "4. Pulsa **Simular** y analiza la tabla y la probabilidad a posteriori cuando el detector dice “sí”."
        )

    col1, col2 = st.columns([1, 2])

    with col1:
        p_signal = st.slider(
            "Probabilidad de que realmente haya señal (estado real)",
            min_value=0.0, max_value=1.0, value=0.3, step=0.05,
            key="g3_ej4_p_signal"
        )
        p_false_alarm = st.slider(
            "Probabilidad de falsa alarma (detecta señal cuando NO hay señal)",
            min_value=0.0, max_value=0.5, value=0.05, step=0.01,
            key="g3_ej4_p_fa"
        )
        p_miss = st.slider(
            "Probabilidad de pérdida (NO detecta cuando SÍ hay señal)",
            min_value=0.0, max_value=0.5, value=0.10, step=0.01,
            key="g3_ej4_p_miss"
        )
        N = st.slider(
            "Número de intentos (observaciones)",
            min_value=1000, max_value=100000, value=20000, step=1000,
            key="g3_ej4_N_new"
        )
        run = st.button("Simular detector", key="g3_ej4_run_new")

    if run:
        # Estado real S: 1 si hay señal, 0 si no hay señal
        S = (np.random.rand(N) < p_signal).astype(int)

        # Decisión del detector R:
        # - Si S=0, R=1 con prob_false_alarm
        # - Si S=1, R=0 con prob_miss
        R = np.zeros_like(S)

        idx0 = (S == 0)
        idx1 = (S == 1)

        R[idx0] = (np.random.rand(np.sum(idx0)) < p_false_alarm).astype(int)          # falsas alarmas
        R[idx1] = (np.random.rand(np.sum(idx1)) >= p_miss).astype(int)               # detecta con prob 1 - miss

        # Conteos tipo "matriz de confusión"
        TN = int(np.sum((S == 0) & (R == 0)))  # correcto: no señal y no detecta
        FP = int(np.sum((S == 0) & (R == 1)))  # falsa alarma
        FN = int(np.sum((S == 1) & (R == 0)))  # pérdida / miss
        TP = int(np.sum((S == 1) & (R == 1)))  # correcto: hay señal y detecta

        # Probabilidades condicionadas estimadas
        # (si no hay muestras en el denominador, evitar división por cero)
        pR1_S0_emp = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        pR0_S1_emp = FN / (FN + TP) if (FN + TP) > 0 else 0.0
        pR1_S1_emp = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        pR0_S0_emp = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        # Bayes: probabilidad de que realmente haya señal dado que el detector dijo R=1
        # Teórica:
        num = p_signal * (1.0 - p_miss)
        den = p_signal * (1.0 - p_miss) + (1.0 - p_signal) * p_false_alarm
        pS1_R1_teo = num / den if den > 0 else 0.0

        # Empírica:
        total_R1 = TP + FP
        pS1_R1_emp = TP / total_R1 if total_R1 > 0 else 0.0

        with col2:
            st.markdown("#### Resultados principales")

            st.write(f"Intentos simulados: **{N:,}**")
            st.write(f"Frecuencia de ‘hay señal’ (estado real) ≈ **{np.mean(S == 1):.3f}**")
            st.write(f"Frecuencia de ‘detectó señal’ (decisión) ≈ **{np.mean(R == 1):.3f}**")

            st.markdown("**Conteos**")
            st.write(f"- Correcto sin señal: **{TN:,}**")
            st.write(f"- Falsa alarma : **{FP:,}**")
            st.write(f"- Pérdida : **{FN:,}**")
            st.write(f"- Correcto con señal : **{TP:,}**")

            st.markdown("**Probabilidades condicionadas del detector**")
            st.write(f"- Probabilidad de falsa alarma estimada ≈ **{pR1_S0_emp:.3f}**")
            st.write(f"- Probabilidad de pérdida estimada ≈ **{pR0_S1_emp:.3f}**")
            st.write(f"- Probabilidad de detección estimada ≈ **{pR1_S1_emp:.3f}**")
            st.write(f"- Probabilidad de rechazo correcto estimada ≈ **{pR0_S0_emp:.3f}**")

            st.markdown("**Bayes: credibilidad del ‘sí’ del detector**")
            st.write(f"- Probabilidad a posteriori teórica (hay señal dado que detectó) ≈ **{pS1_R1_teo:.3f}**")
            st.write(f"- Probabilidad a posteriori empírica ≈ **{pS1_R1_emp:.3f}**")

            # Gráfico simple: de todos los R=1, cuántos son TP vs FP
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=["R=1 y había señal (TP)", "R=1 sin señal (FP)"],
                        y=[TP, FP],
                        name="Casos",
                    )
                ]
            )
            fig.update_layout(
                title="Cuando el detector dice 'sí', ¿cuántas veces acierta?",
                xaxis_title="Resultado",
                yaxis_title="Cantidad de casos",
                height=320,
                margin=dict(l=40, r=20, t=60, b=40),
            )
            plot_theme = _get_plot_theme()
            _apply_plot_theme(fig, plot_theme, font_size=12)
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with st.expander("Explicación de la simulación y preguntas", expanded=True):
            st.markdown(
                "En este ejemplo se separan dos preguntas que suelen confundirse. La primera describe el comportamiento del "
                "detector: si en realidad no hay señal, ¿con qué probabilidad se equivoca y declara que sí la hay "
                "(falsa alarma)? y si sí hay señal, ¿con qué probabilidad falla y no la detecta (pérdida)? Estas son "
                "probabilidades condicionadas del tipo P(R|S), porque la condición es “la realidad”. "
                "La segunda pregunta es la más práctica: una vez que el detector decidió “sí”, ¿qué tan confiable es esa "
                "decisión?, es decir P(S|R). Ahí entra Bayes, que combina el desempeño del detector con la frecuencia real "
                "con la que aparece la señal; por eso, si la señal es poco frecuente, un “sí” puede no ser tan convincente "
                "aunque el detector tenga pocos errores."
            )

            st.markdown("#### Preguntas y respuestas")
            st.markdown(
                "- **1. ¿Por qué la probabilidad de ‘hay señal dado que detectó’ puede ser baja aunque el detector sea bueno?**  \n"
                "  **R:** Porque si ‘hay señal’ ocurre muy pocas veces, la mayoría de los ‘sí’ pueden venir de falsas alarmas, "
                "aunque la tasa de falsa alarma sea pequeña.\n\n"
                "- **2. ¿Qué parámetro cambia más la credibilidad del ‘sí’?**  \n"
                "  **R:** La probabilidad de que realmente haya señal (a priori) y la tasa de falsa alarma; ambos afectan fuertemente a Bayes.\n\n"
                "- **3. ¿Cómo se conecta esto con un receptor digital?**  \n"
                "  **R:** ‘Hay señal’ puede interpretarse como ‘se transmitió el bit 1’, ‘detectó señal’ como ‘el receptor decidió 1’; "
                "las falsas alarmas y pérdidas son errores de decisión, y Bayes es la base de detectores óptimos."
            )


def render_ejemplo5():
    """
    Ejemplo 5 — Detector MAP con ruido Gaussiano (BPSK).
    """
    st.markdown("### Ejemplo 5 - Detector MAP con ruido Gaussiano en un sistema BPSK")

    with st.expander("Descripción y pasos", expanded=True):
        st.markdown(r"""
        En este ejemplo se modela un sistema binario con modulación **BPSK**, donde los símbolos son:

        $$
        s_0 = -A, \qquad s_1 = +A
        $$

        El ruido del canal se modela como una variable aleatoria Gaussiana:

        $$
        n \sim \mathcal{N}(0, \sigma^2)
        $$

        El receptor observa:

        $$
        Y = S + n
        $$

        y toma una decisión mediante un **detector MAP** (Máxima Probabilidad A Posteriori).

        ---

        **Pasos sugeridos:**

        1. Ajusta la amplitud \(A\) de los símbolos.
        2. Ajusta la relación señal-ruido mediante el parámetro **SNR en dB**.
        3. Ajusta la probabilidad a priori \(P(S=1)\).
        4. Pulsa **Simular detector MAP** para generar las muestras, el umbral óptimo y el BER resultante.
        5. Observa cómo cambia el rendimiento al modificar SNR y las probabilidades a priori.
        """)

    col1, col2 = st.columns([1, 2])

    with col1:
        A = st.number_input(
            "Amplitud A",
            value=1.0,
            format="%.2f",
            min_value=0.01,
            key="g3_ej5_A",
        )
        SNR_dB = st.slider(
            "SNR (dB)",
            min_value=-5.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            key="g3_ej5_snr",
        )
        p1 = st.slider(
            "Probabilidad a priori P(S=1)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="g3_ej5_p1",
        )
        N = st.slider(
            "Número de bits",
            1000, 100000, 20000, 1000,
            key="g3_ej5_N",
        )
        run = st.button("Simular detector MAP", key="g3_ej5_run")

    if run:
        # SNR = A^2 / sigma^2 => sigma
        SNR_lin = 10 ** (SNR_dB / 10.0)
        sigma2 = A ** 2 / max(SNR_lin, 1e-12)
        sigma = np.sqrt(sigma2)

        # Bits transmitidos
        U = np.random.rand(N)
        S_bits = (U < p1).astype(int)
        S_symbols = np.where(S_bits == 1, A, -A)

        # Ruido Gaussiano
        n = np.random.normal(0.0, sigma, size=N)
        Y = S_symbols + n

        # Umbral MAP (para BPSK con ruido Gaussiano)
        # gamma = (sigma^2 / (2A)) * ln((1-p1)/p1)
        from math import log
        if p1 in (0.0, 1.0):
            gamma = 0.0  # caso degenerado; el detector MAP se colapsa
        else:
            gamma = (sigma2 / (2.0 * A)) * log((1.0 - p1) / p1)

        # Decisión: si Y >= gamma -> 1, si no -> 0
        decisions = (Y >= gamma).astype(int)

        errores = np.sum(decisions != S_bits)
        BER_emp = errores / float(N)

        # BER teórica (para BPSK con decisiones ML, p1=0.5):
        # Pb = Q( sqrt(2 * Eb/N0) ) ~ Q( sqrt(2*SNR_lin) ), aquí aproximamos:
        from math import erf, sqrt
        def Q(x):
            return 0.5 * (1 - erf(x / sqrt(2)))
        BER_teo_aprox = Q(np.sqrt(2 * SNR_lin))

        with col2:
            st.markdown("#### Resultados del detector MAP")
            st.write(f"SNR (lineal) ≈ {SNR_lin:.2f}")
            st.write(f"Varianza del ruido σ² ≈ {sigma2:.4f}")
            st.write(f"Umbral MAP γ ≈ {gamma:.4f}")
            st.write(f"BER empírico ≈ {BER_emp:.4f}")
            st.write(f"BER teórico aproximado ≈ {BER_teo_aprox:.4f}, referencia")

            # Histograma de Y condicionado
            Y0 = Y[S_bits == 0]
            Y1 = Y[S_bits == 1]
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=Y0,
                    nbinsx=40,
                    histnorm="probability density",
                    opacity=0.55,
                    name="Y | S=0",
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=Y1,
                    nbinsx=40,
                    histnorm="probability density",
                    opacity=0.55,
                    name="Y | S=1",
                )
            )
            fig.add_vline(x=gamma, line=dict(color="black", dash="dash"), annotation_text="Umbral MAP")
            fig.update_layout(
                title="Distribuciones de Y condicionadas y umbral MAP",
                xaxis_title="Y",
                yaxis_title="Densidad aproximada",
                barmode="overlay",
                height=340,
                margin=dict(l=40, r=20, t=60, b=40),
            )
            plot_theme = _get_plot_theme()
            _apply_plot_theme(fig, plot_theme, font_size=12)
            st.plotly_chart(fig, use_container_width=True, theme=None)

        with st.expander("Explicación de la simulación y preguntas", expanded=True):
            st.markdown(
                "En BPSK el transmisor representa el bit **0** con el símbolo -A y el bit **1** con +A. "
                "El canal agrega ruido Gaussiano, por eso el receptor no ve una observación "
                "Y que queda “corrida” por el ruido: Y = S + n. Como n es Gaussiano, entonces la distribución "
                "de Y condicionado al símbolo transmitido también es Gaussiana: aparece una campana centrada en -A "
                "para Y|S=0 y otra centrada en +A para Y|S=1. Esas dos PDF condicionadas, f{Y|S}(y|0) y "
                "f{Y|S}(y|1), resumen estadísticamente cómo se ve cada símbolo a la salida del canal.\n\n"
                "El detector **MAP** (máxima probabilidad a posteriori) responde a la pregunta más práctica del receptor: "
                "dado un valor observado (y), **¿cuál símbolo fue más probable que se haya enviado?** Matemáticamente compara "
                "las probabilidades a posteriori P(S=1|Y=y) y P(S=0|Y=y). Como esas cantidades no se calculan ‘a ojo’, "
                "se usan las PDF condicionadas y las probabilidades a priori mediante Bayes: la evidencia del canal viene en "
                "\(f{Y|S}(y|s) y la tendencia de transmisión viene en P(S=s). En otras palabras, MAP combina lo que "
                "miden los datos con lo que ya se sabía antes de observar (Y).\n\n"
                "En este modelo Gaussiano, esa comparación se reduce a un **umbral** \(\gamma\). El umbral es el punto sobre el "
                "eje \(Y\) donde el receptor cambia de decisión: valores mayores que \(\gamma\) favorecen \(S=1\) y valores menores "
                "favorecen \(S=0\). Su significado es muy concreto: es la frontera que separa la región donde ‘es más creíble’ la "
                "hipótesis \(S=1\) de la región donde ‘es más creíble’ \(S=0\).\n\n"
                "Cuando los símbolos son equiprobables, \(P(S=1)=P(S=0)=0.5\), el umbral cae en el centro y el detector se vuelve "
                "de **máxima verosimilitud (ML)**: decide únicamente por cercanía a \(\pm A\), típicamente con \(\gamma\) cerca de 0. "
                "En cambio, si \(P(S=1)\) es mayor que \(P(S=0)\), el umbral se desplaza para **favorecer al símbolo más frecuente**: "
                "acepta \(S=1\) con observaciones menos “contundentes”, porque a priori es más probable que ese símbolo haya sido enviado. "
                "Además, el valor de \(\sigma^2\) también afecta a \(\gamma\): con más ruido las campanas se traslapan más y el detector "
                "necesita ajustar la frontera para minimizar el error promedio.\n\n"
                "El histograma muestra justamente ese traslape: las barras de \(Y|S=0\) y \(Y|S=1\) se enciman por culpa del ruido. "
                "Cada vez que una muestra cae del ‘lado equivocado’ del umbral ocurre un error de decisión, y al contarlos se estima el BER."
            )

            st.markdown("#### Preguntas y respuestas")
            st.markdown(
                "- **1. ¿Qué pasa con el BER cuando aumenta la SNR?**  \n"
                "  **R:** Disminuye, porque la separación entre las distribuciones de \(Y|S=0\) y \(Y|S=1\) es mayor en comparación con la desviación estándar del ruido.\n\n"
                "- **2. ¿Por qué el umbral MAP no siempre es cero?**  \n"
                "  **R:** Porque cuando \(P(S=0) \neq P(S=1)\), es óptimo desplazar el umbral hacia el símbolo menos probable, "
                "de manera que se reduzca la probabilidad de confundir el símbolo más frecuente.\n\n"
                "- **3. ¿Cómo se relaciona este ejemplo con el rendimiento de sistemas BPSK en canales AWGN?**  \n"
                "  **R:** El ejemplo reproduce el modelo clásico de BPSK sobre canal AWGN, donde el BER se expresa en función de la SNR "
                "y se evalúa típicamente mediante la función Q(⋅). que mide la cola derecha de una distribución Gaussiana estándar"
            )


# =========================================================
# DINÁMICAS (integradas: 1 registro + 1 botón de envío)
# =========================================================

def _g3_reset_keys(keys):
    """Elimina claves del session_state para reiniciar radios/selects cuando se regenere un caso."""
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]


def _g3_init_dyn_state():
    if "g3_dyn_state" not in st.session_state:
        st.session_state.g3_dyn_state = {
            "dyn1_case": None,
            "dyn2_case": None,
            "dyn3_case": None,
        }


def _g3_gen_dyn1_case():
    """Dinámica 1 (Ejemplo 1): Frecuencia relativa (Ley de los grandes números)."""
    p = float(np.round(np.random.uniform(0.20, 0.80), 2))
    N = int(np.random.choice([200, 300, 400, 500]))
    rng = np.random.default_rng()
    x = (rng.random(N) < p).astype(int)

    fr = np.cumsum(x) / np.arange(1, N + 1)

    correct = {
        "q1": f"{p:.2f}",
        "q2": "Disminuye",
    }
    return {"p": p, "N": N, "x": x, "fr": fr, "correct": correct}


def render_dinamica1():
    st.markdown("### Dinámica 1 — Frecuencia relativa y convergencia")

    _g3_init_dyn_state()

    # Caso (persistente)
    if st.session_state.g3_dyn_state["dyn1_case"] is None:
        st.session_state.g3_dyn_state["dyn1_case"] = _g3_gen_dyn1_case()

    if st.button("Regenerar caso (Dinámica 1)", key="g3_dyn1_regen"):
        st.session_state.g3_dyn_state["dyn1_case"] = _g3_gen_dyn1_case()
        _g3_reset_keys(["g3_dyn1_q1", "g3_dyn1_q2"])

    case = st.session_state.g3_dyn_state["dyn1_case"]
    p, N, fr = case["p"], case["N"], case["fr"]

    st.markdown(f"**Caso generado:** experimento Bernoulli con probabilidad teórica **p = {p:.2f}**, con **N = {N}** repeticiones.")

    # Gráfica: frecuencia relativa acumulada
    theme = _get_plot_theme()
    x_vals = np.arange(1, N + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=fr,
            mode="lines",
            name="f_R(n)",
            line=dict(color="#2a9d8f", width=2),
            hovertemplate="n=%{x}<br>f_R(n)=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[1, N],
            y=[p, p],
            mode="lines",
            name=f"p={p:.2f}",
            line=dict(color="#e76f51", dash="dash"),
            hovertemplate="p=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Convergencia de la frecuencia relativa",
        xaxis_title="Número de ensayos n",
        yaxis_title="Frecuencia relativa acumulada f_R(n)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(color=theme["font_color"]),
        ),
    )
    _apply_plot_theme(fig, theme)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Responde **2 preguntas** basándote en la gráfica y en la definición de frecuencia relativa.")

    # Preguntas (sin botón de guardar: se guardan en session_state automáticamente)
    opciones_q1 = [f"{p:.2f}", "0.50", f"{1 - p:.2f}"]
    q1 = st.radio(
        "1) Conforme n aumenta, $f_R(n)$ tiende a:",
        opciones_q1,
        index=None,
        key="g3_dyn1_q1",
    )
    q2 = st.radio(
        "2) Si aumentamos N (más ensayos), la variabilidad de $f_R(n)$ alrededor de p:",
        ["Disminuye", "Aumenta", "No cambia"],
        index=None,
        key="g3_dyn1_q2",
    )


def _g3_gen_dyn2_case():
    """Dinámica 2 (Ejemplo 3): Probabilidad condicional en un canal binario simétrico (BSC)."""
    p = float(np.round(np.random.uniform(0.05, 0.30), 2))  # prob. de error (crossover)
    P = np.array([[1 - p, p],
                  [p, 1 - p]], dtype=float)  # filas: X=0/1, cols: Y=0/1

    correct = {
        "q1": f"{p:.2f}",
        "q2": "Aumenta",
    }
    return {"p": p, "P": P, "correct": correct}


def render_dinamica2():
    st.markdown("### Dinámica 2 — Probabilidad condicional en un canal binario")

    _g3_init_dyn_state()

    if st.session_state.g3_dyn_state["dyn2_case"] is None:
        st.session_state.g3_dyn_state["dyn2_case"] = _g3_gen_dyn2_case()

    if st.button("Regenerar caso (Dinámica 2)", key="g3_dyn2_regen"):
        st.session_state.g3_dyn_state["dyn2_case"] = _g3_gen_dyn2_case()
        _g3_reset_keys(["g3_dyn2_q1", "g3_dyn2_q2"])

    case = st.session_state.g3_dyn_state["dyn2_case"]
    p, P = case["p"], case["P"]

    st.markdown(
        f"**Caso generado:** Canal binario simétrico (BSC) con probabilidad de error "
        f"**p = {p:.2f}**."
    )

    # Gráfica: matriz de probabilidades condicionales como "heatmap" interactivo
    theme = _get_plot_theme()
    labels_x = ["Y=0", "Y=1"]
    labels_y = ["X=0", "X=1"]
    text_vals = [[f"{P[i, j]:.2f}" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        data=go.Heatmap(
            z=P,
            x=labels_x,
            y=labels_y,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(color=theme["font_color"]),
            colorscale="Viridis",
            hovertemplate="Entrada %{y}<br>Salida %{x}<br>P=%{z:.2f}<extra></extra>",
            colorbar=dict(title="P(Y|X)", tickfont=dict(color=theme["font_color"])),
        )
    )
    fig.update_layout(
        title="Matriz de probabilidad condicional P(Y|X)",
        xaxis_title="Salida Y",
        yaxis_title="Entrada X",
    )
    _apply_plot_theme(fig, theme)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Responde **2 preguntas** usando el concepto de probabilidad condicional.")

    q1 = st.radio(
        f"1) Si se envía **X = 1**, la probabilidad de recibir **Y = 0** es:",
        [f"{p:.2f}", f"{1 - p:.2f}", "0.50"],
        index=None,
        key="g3_dyn2_q1",
    )
    q2 = st.radio(
        "2) Si la probabilidad de error p aumenta, la BER esperada:",
        ["Aumenta", "Disminuye", "Se mantiene"],
        index=None,
        key="g3_dyn2_q2",
    )


def _g3_gaussian_pdf(x, mu, sigma):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _g3_gen_dyn3_case():
    """Dinámica 3 (Ejemplo 5): Decisión MAP en AWGN para BPSK (dos gaussianas)."""
    A = 1.0
    sigma = float(np.round(np.random.uniform(0.25, 0.75), 2))
    pi1 = float(np.random.choice([0.25, 0.35, 0.65, 0.75]))
    pi0 = float(np.round(1.0 - pi1, 2))

    # Umbral MAP para BPSK con medias ±A, varianza sigma^2
    gamma = float((sigma ** 2 / (2 * A)) * np.log(pi0 / pi1))

    r0 = float(np.round(np.random.uniform(-1.6, 1.6), 2))
    decision = "1" if r0 > gamma else "0"

    correct = {
        "q1": "Negativo" if gamma < 0 else ("Positivo" if gamma > 0 else "Cero"),
        "q2": decision,
    }
    return {
        "A": A,
        "sigma": sigma,
        "pi0": pi0,
        "pi1": pi1,
        "gamma": gamma,
        "r0": r0,
        "correct": correct,
    }


def render_dinamica3():
    st.markdown("### Dinámica 3 — Decisión MAP (BPSK en AWGN)")

    _g3_init_dyn_state()

    if st.session_state.g3_dyn_state["dyn3_case"] is None:
        st.session_state.g3_dyn_state["dyn3_case"] = _g3_gen_dyn3_case()

    if st.button("Regenerar caso (Dinámica 3)", key="g3_dyn3_regen"):
        st.session_state.g3_dyn_state["dyn3_case"] = _g3_gen_dyn3_case()
        _g3_reset_keys(["g3_dyn3_q1", "g3_dyn3_q2"])

    case = st.session_state.g3_dyn_state["dyn3_case"]
    A, sigma, pi0, pi1, gamma, r0 = case["A"], case["sigma"], case["pi0"], case["pi1"], case["gamma"], case["r0"]

    st.markdown(
        f"**Caso generado:** BPSK con medias $\\mu_0=-A$ y $\\mu_1=+A$, "
        f"**A = {A:.1f}**, **σ = {sigma:.2f}**, priors **π₀ = {pi0:.2f}**, **π₁ = {pi1:.2f}**.\n\n"
        f"Umbral MAP: **γ = {gamma:.2f}** (decide 1 si r > γ)."
    )

    # Gráfica de densidades (ponderadas por priors)
    x = np.linspace(-3.0, 3.0, 600)
    p0 = pi0 * _g3_gaussian_pdf(x, -A, sigma)
    p1 = pi1 * _g3_gaussian_pdf(x, +A, sigma)

    theme = _get_plot_theme()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p0,
            mode="lines",
            name=r"π₀ p(r|0)",
            line=dict(color="#3a86ff", width=2),
            hovertemplate="r=%{x:.2f}<br>π₀ p(r|0)=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p1,
            mode="lines",
            name=r"π₁ p(r|1)",
            line=dict(color="#ff006e", width=2),
            hovertemplate="r=%{x:.2f}<br>π₁ p(r|1)=%{y:.4f}<extra></extra>",
        )
    )
    max_y = float(max(p0.max(), p1.max()) * 1.1)
    fig.add_trace(
        go.Scatter(
            x=[gamma, gamma],
            y=[0.0, max_y],
            mode="lines",
            name="Umbral γ",
            line=dict(color="#6c757d", dash="dash"),
            hovertemplate="γ=%{x:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[r0, r0],
            y=[0.0, max_y],
            mode="lines",
            name=f"Muestra r={r0:.2f}",
            line=dict(color="#00b4d8", dash="dot"),
            hovertemplate="r=%{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Criterio MAP: comparación de densidades ponderadas",
        xaxis_title="r",
        yaxis_title="Densidad ponderada",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5,
            font=dict(color=theme["font_color"]),
        ),
    )
    _apply_plot_theme(fig, theme)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Responde **2 preguntas** usando el umbral mostrado en la figura.")

    q1 = st.radio(
        "1) En este caso, el umbral γ es:",
        ["Negativo", "Cero", "Positivo"],
        index=None,
        key="g3_dyn3_q1",
    )
    q2 = st.radio(
        f"2) Si la muestra es r = {r0:.2f}, la decisión MAP es:",
        ["0", "1"],
        index=None,
        key="g3_dyn3_q2",
    )


def render_dinamicas_guia3():
    """
    Dinámicas integradas de la Guía 3:
    - 1 solo registro (estudiante)
    - Dinámicas 1, 2 y 3 (con gráficas)
    - 1 solo botón final para enviar y generar PDF
    """
    st.subheader("Dinámicas")

    student_info = _ensure_student_info("g3_form_registro_unico")
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
            unsafe_allow_html=True
        )
        return

    st.markdown("---")

    with st.expander("Dinámica 1 — Frecuencia relativa (Ejemplo 1)", expanded=True):
        render_dinamica1()

    with st.expander("Dinámica 2 — Probabilidad condicional (Ejemplo 3)", expanded=True):
        render_dinamica2()

    with st.expander("Dinámica 3 — Bayes / MAP (Ejemplo 5)", expanded=True):
        render_dinamica3()

    st.markdown("---")

    # Botón único para enviar y generar PDF
    if st.button("Enviar respuestas y generar PDF", key="g3_btn_enviar"):
        # Validar que haya casos generados y respuestas completas
        _g3_init_dyn_state()

        faltan = []

        # Dyn1
        c1 = st.session_state.g3_dyn_state.get("dyn1_case")
        a1 = {"q1": st.session_state.get("g3_dyn1_q1"), "q2": st.session_state.get("g3_dyn1_q2")}
        if c1 is None or None in a1.values():
            faltan.append("Dinámica 1")

        # Dyn2
        c2 = st.session_state.g3_dyn_state.get("dyn2_case")
        a2 = {"q1": st.session_state.get("g3_dyn2_q1"), "q2": st.session_state.get("g3_dyn2_q2")}
        if c2 is None or None in a2.values():
            faltan.append("Dinámica 2")

        # Dyn3
        c3 = st.session_state.g3_dyn_state.get("dyn3_case")
        a3 = {"q1": st.session_state.get("g3_dyn3_q1"), "q2": st.session_state.get("g3_dyn3_q2")}
        if c3 is None or None in a3.values():
            faltan.append("Dinámica 3")

        if faltan:
            st.error("Aún faltan por completar: " + ", ".join(faltan))
            return

        # Calificar (2 preguntas por dinámica)
        def _score_2q(ans, corr):
            correct_count = sum(ans[k] == corr[k] for k in corr.keys())
            mapping = {2: 10.0, 1: 6.0, 0: 0.0}
            return mapping.get(correct_count, 0.0)

        res1 = {
            "dyn_id": 1,
            "score": _score_2q(a1, c1["correct"]),
            "answers": a1,
            "correct": c1["correct"],
            "key": {
                "descripcion": "Guía 3 - Dinámica 1 - Frecuencia relativa",
                "p_teorica": f"{c1['p']:.2f}",
                "N": c1["N"],
            },
        }
        res2 = {
            "dyn_id": 2,
            "score": _score_2q(a2, c2["correct"]),
            "answers": a2,
            "correct": c2["correct"],
            "key": {
                "descripcion": "Guía 3 - Dinámica 2 - Probabilidad condicional (BSC)",
                "p_error": f"{c2['p']:.2f}",
            },
        }
        res3 = {
            "dyn_id": 3,
            "score": _score_2q(a3, c3["correct"]),
            "answers": a3,
            "correct": c3["correct"],
            "key": {
                "descripcion": "Guía 3 - Dinámica 3 - Decisión MAP (BPSK/AWGN)",
                "sigma": f"{c3['sigma']:.2f}",
                "pi0": f"{c3['pi0']:.2f}",
                "pi1": f"{c3['pi1']:.2f}",
                "gamma": f"{c3['gamma']:.2f}",
            },
        }

        if not REPORTLAB_AVAILABLE:
            st.error(
                "No se puede generar el PDF porque 'reportlab' no está disponible. "
                "Agrega 'reportlab' a requirements.txt."
            )
            return

        pdf_bytes, pdf_filename = export_results_pdf_guia3_bytes(
            student_info=student_info,
            resultados=[res1, res2, res3],
        )

        if not pdf_bytes:
            st.error("No se pudo generar el PDF. Revisa ReportLab o permisos.")
            return

        # Subir a GitHub (si está configurado)
        ruta_repo = f"guia3/{pdf_filename}"
        commit_msg = f"Guía 3 - {student_info.get('id','sin_id')} - {student_info.get('name','')}".strip()
        ok, info = upload_bytes_to_github_results(
            content_bytes=pdf_bytes,
            repo_path=ruta_repo,
            commit_message=commit_msg,
        )

        if ok:
            st.success("PDF generado y enviado correctamente a GitHub.")
            if isinstance(info, dict) and info.get("html_url"):
                st.link_button("Ver archivo en GitHub", info["html_url"])
            st.write("Ruta en el repositorio:", ruta_repo)
            st.info("Consulta tu nota con el catedrático o instructor encargado.")
        else:
            st.error(f"No se pudo subir el PDF: {info}")


def render_guia3():
    st.title("Guía 3: Fundamentos de probabilidad")

    tabs = st.tabs([
        "Objetivos",
        "Introducción teórica",
        "Materiales y equipo",
        "Ejemplos",
        "Dinámicas",
        "Conclusiones",
    ])

    with tabs[0]:
        st.markdown(OBJETIVOS3_TEXT)

    with tabs[1]:
        st.markdown(INTRO3_TEXT)

    with tabs[2]:
        st.subheader("Materiales y equipo")
        st.markdown(MATERIALES_COMUNES)

    with tabs[3]:
        st.markdown("En esta sección se presentan cinco ejemplos interactivos.")
        sub_tabs = st.tabs(["Ejemplo 1", "Ejemplo 2", "Ejemplo 3", "Ejemplo 4", "Ejemplo 5"])
        with sub_tabs[0]:
            render_ejemplo1()
        with sub_tabs[1]:
            render_ejemplo2()
        with sub_tabs[2]:
            render_ejemplo3()
        with sub_tabs[3]:
            render_ejemplo4()
        with sub_tabs[4]:
            render_ejemplo5()

    with tabs[4]:
        render_dinamicas_guia3()

    with tabs[5]:
        st.markdown(CONCLUSIONES3_TEXT)
