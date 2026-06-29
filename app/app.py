import streamlit as st
import pandas as pd
import pickle
import spacy
import re
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ======================================================
# CONFIGURACIÓN
# ======================================================

st.set_page_config(
    page_title="Ecommerce Review Analyzer",
    layout="wide"
)


# ======================================================
# ESTILOS
# ======================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #f5f7fa;
}

[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e5e7eb;
}

.block-container {
    padding-top: 0rem;
    padding-left: 2.5rem;
    padding-right: 2.5rem;
    max-width: 1250px;
}

.header {
    background: linear-gradient(135deg, #0f172a, #172554);
    color: white;
    padding: 2.2rem 2.5rem;
    margin-left: -2.5rem;
    margin-right: -2.5rem;
    margin-bottom: 2rem;
}

.header-title {
    font-family: Georgia, serif;
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.header-subtitle {
    color: #cbd5e1;
    font-size: 1rem;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #111827;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.4rem;
}

.section-subtitle {
    color: #6b7280;
    font-size: 0.92rem;
    margin-bottom: 1rem;
}

.metric-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.4rem;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
}

.metric-label {
    color: #6b7280;
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
}

.metric-value {
    font-family: Georgia, serif;
    font-size: 2.15rem;
    font-weight: 700;
    color: #0f172a;
}

.dark-card {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
    border-radius: 14px;
    padding: 1.8rem;
    margin-bottom: 1.4rem;
}

.dark-card-label {
    color: #cbd5e1;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}

.dark-card-title {
    font-family: Georgia, serif;
    font-size: 2rem;
    font-weight: 700;
}

.dark-card-subtitle {
    color: #e2e8f0;
    font-size: 1rem;
    margin-top: 0.5rem;
}

.sentiment-label {
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}

.sentiment-value {
    font-family: Georgia, serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #0f172a;
}

.sentiment-count {
    color: #6b7280;
    font-size: 0.9rem;
    margin-top: 0.3rem;
}

.progress-bg {
    width: 100%;
    background: #e5e7eb;
    border-radius: 999px;
    height: 7px;
    margin-top: 0.9rem;
}

.progress-fill {
    height: 7px;
    border-radius: 999px;
}

.aspect-row {
    margin-bottom: 0.85rem;
}

.aspect-head {
    display: flex;
    justify-content: space-between;
    font-size: 0.92rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 0.3rem;
}

.card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.35rem;
    box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    margin-bottom: 1.2rem;
}

.info-table {
    background: #f8fafc;
    border-radius: 12px;
    padding: 1rem;
    font-size: 0.92rem;
}

.summary-box {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.5rem;
    line-height: 1.7;
    color: #1f2937;
    margin-top: 1rem;
}

.sidebar-title {
    font-size: 1rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
    color: #111827;
}

.sidebar-text {
    color: #4b5563;
    font-size: 0.92rem;
    line-height: 1.6;
}

</style>
""", unsafe_allow_html=True)


# ======================================================
# CARGA DE MODELOS
# ======================================================

@st.cache_resource
def cargar_modelos():
    with open("models/modelo_sentimiento.pkl", "rb") as archivo:
        modelo = pickle.load(archivo)

    with open("models/vectorizador_tfidf.pkl", "rb") as archivo:
        vectorizador = pickle.load(archivo)

    return modelo, vectorizador


@st.cache_resource
def cargar_spacy():
    return spacy.load("en_core_web_sm")


modelo, vectorizador = cargar_modelos()
nlp = cargar_spacy()


# ======================================================
# FUNCIONES
# ======================================================

palabras_excluir = [
    "amazon", "product", "thing", "time", "day", "month", "year",
    "star", "stars", "tv", "stick", "fire", "br", "show",
    "love", "device", "item", "way", "lot", "bit", "review",
    "people", "something", "anything", "everything"
]


def limpiar_texto_basico(texto):
    texto = str(texto).lower()
    texto = re.sub(r"<.*?>", " ", texto)
    texto = re.sub(r"[^a-zA-Z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    palabras = texto.split()
    palabras = [
        palabra for palabra in palabras
        if palabra not in ENGLISH_STOP_WORDS and len(palabra) > 2
    ]

    return " ".join(palabras)


def extraer_sustantivos(texto):
    doc = nlp(str(texto))
    sustantivos = []

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            palabra = token.lemma_.lower()

            if palabra not in palabras_excluir and len(palabra) > 2:
                sustantivos.append(palabra)

    return sustantivos


def obtener_aspectos(textos, n=5):
    todos_aspectos = []

    for texto in textos.dropna():
        aspectos = extraer_sustantivos(texto)
        todos_aspectos.extend(aspectos)

    contador = Counter(todos_aspectos)

    return contador.most_common(n)


def barra_aspectos(aspectos, color):
    if len(aspectos) == 0:
        st.write("No hay suficientes datos para extraer aspectos.")
        return

    max_valor = aspectos[0][1]

    for aspecto, frecuencia in aspectos:
        porcentaje = frecuencia / max_valor * 100

        st.markdown(f"""
        <div class="aspect-row">
            <div class="aspect-head">
                <span>{aspecto}</span>
                <span>{frecuencia}</span>
            </div>
            <div class="progress-bg">
                <div class="progress-fill" style="width:{porcentaje}%; background:{color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def generar_resumen_ejecutivo(
    producto,
    porcentajes,
    aspectos_positivos,
    aspectos_negativos,
    total_reviews
):
    porcentaje_positivo = porcentajes.get("Positivo", 0)
    porcentaje_negativo = porcentajes.get("Negativo", 0)

    positivos = [aspecto for aspecto, frecuencia in aspectos_positivos[:3]]
    negativos = [aspecto for aspecto, frecuencia in aspectos_negativos[:3]]

    texto_positivos = ", ".join(positivos) if positivos else "aspectos positivos no identificados"
    texto_negativos = ", ".join(negativos) if negativos else "aspectos negativos no identificados"

    if porcentaje_positivo >= 70:
        valoracion = "mayoritariamente positiva"
    elif porcentaje_negativo >= 40:
        valoracion = "con una presencia relevante de opiniones negativas"
    else:
        valoracion = "mixta"

    resumen = (
        f"El producto {producto} presenta una valoración {valoracion}, "
        f"con un {porcentaje_positivo:.1f}% de opiniones positivas y un "
        f"{porcentaje_negativo:.1f}% de opiniones negativas. "
        f"Los clientes destacan principalmente {texto_positivos} como los aspectos mejor valorados. "
        f"Las críticas se concentran en {texto_negativos}, que son los aspectos más mencionados "
        f"en las opiniones negativas. "
        f"Este resumen se basa en el análisis de {total_reviews} reviews."
    )

    return resumen


# ======================================================
# CABECERA
# ======================================================

st.markdown("""
<div class="header">
    <div class="header-title">Ecommerce Review Analyzer</div>
    <div class="header-subtitle">Analiza opiniones de productos con NLP y Machine Learning</div>
</div>
""", unsafe_allow_html=True)


# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:
    st.markdown('<div class="sidebar-title">Datos</div>', unsafe_allow_html=True)

    archivo = st.file_uploader(
        "Sube un archivo CSV con reviews",
        type=["csv"]
    )

    st.markdown('<div class="sidebar-title">Filtros</div>', unsafe_allow_html=True)

    minimo_reviews = st.slider(
        "Mínimo de reviews por producto",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )

    st.markdown('<div class="sidebar-title">Ayuda</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-text">
    El archivo debe contener una columna con el identificador del producto
    y una columna con el texto de la review.
    <br><br>
    Si el archivo no incluye sentimiento, la aplicación lo generará automáticamente
    usando el modelo entrenado.
    </div>
    """, unsafe_allow_html=True)


# ======================================================
# APP PRINCIPAL
# ======================================================

if archivo is None:
    st.markdown("""
    <div class="card">
        <div class="section-title">Inicio</div>
        <div class="section-subtitle">
            Sube un archivo CSV desde el panel lateral para comenzar el análisis.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    df = pd.read_csv(archivo)

    if "parent_asin" not in df.columns:
        st.error("El CSV debe incluir una columna llamada 'parent_asin'.")
        st.stop()

    if "review" in df.columns:
        texto_col = "review"
    elif "text" in df.columns:
        texto_col = "text"
    elif "review_clean" in df.columns:
        texto_col = "review_clean"
    else:
        st.error("El CSV debe incluir una columna de texto: 'review', 'text' o 'review_clean'.")
        st.stop()

    if "review_clean" not in df.columns:
        df["review_clean"] = df[texto_col].apply(limpiar_texto_basico)

    if "sentiment" not in df.columns:
        X_tfidf = vectorizador.transform(df["review_clean"].fillna(""))
        df["sentiment"] = modelo.predict(X_tfidf)

    productos = (
        df["parent_asin"]
        .value_counts()
        .reset_index()
    )

    productos.columns = ["parent_asin", "num_reviews"]
    productos = productos[productos["num_reviews"] >= minimo_reviews]
    productos = productos.sort_values(by="num_reviews", ascending=False)

    if productos.empty:
        st.warning("No hay productos con el mínimo de reviews seleccionado.")
        st.stop()

    # ---------------- RESUMEN GENERAL ----------------

    st.markdown('<div class="section-title">Resumen general</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Reviews analizadas</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Productos únicos</div>
            <div class="metric-value">{df["parent_asin"].nunique():,}</div>
        </div>
        """.replace(",", "."), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- SELECTOR ----------------

    st.markdown('<div class="section-title">Selecciona un producto</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Elige un producto para ver el análisis detallado de sus opiniones.</div>',
        unsafe_allow_html=True
    )

    producto = st.selectbox(
        "Producto",
        productos["parent_asin"],
        label_visibility="collapsed",
        format_func=lambda x: f"{x} ({int(productos.loc[productos['parent_asin'] == x, 'num_reviews'].iloc[0])} reviews)"
    )

    producto_df = df[df["parent_asin"] == producto]

    # ---------------- PRODUCTO SELECCIONADO ----------------

    st.markdown(f"""
    <div class="dark-card">
        <div class="dark-card-label">Producto seleccionado</div>
        <div class="dark-card-title">{producto}</div>
        <div class="dark-card-subtitle">{len(producto_df)} reviews</div>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- SENTIMIENTO ----------------

    conteo = producto_df["sentiment"].value_counts()
    porcentajes = producto_df["sentiment"].value_counts(normalize=True) * 100

    st.markdown('<div class="section-title">Resumen de sentimiento</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Distribución de opiniones clasificadas por sentimiento</div>',
        unsafe_allow_html=True
    )

    col_pos, col_neu, col_neg = st.columns(3)

    with col_pos:
        st.markdown(f"""
        <div class="metric-box">
            <div class="sentiment-label" style="color:#3f6f64;">Positivas</div>
            <div class="sentiment-value">{porcentajes.get("Positivo", 0):.1f}%</div>
            <div class="sentiment-count">{conteo.get("Positivo", 0)} reviews</div>
            <div class="progress-bg">
                <div class="progress-fill" style="width:{porcentajes.get("Positivo", 0)}%; background:#3f6f64;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_neu:
        st.markdown(f"""
        <div class="metric-box">
            <div class="sentiment-label" style="color:#b49b57;">Neutras</div>
            <div class="sentiment-value">{porcentajes.get("Neutro", 0):.1f}%</div>
            <div class="sentiment-count">{conteo.get("Neutro", 0)} reviews</div>
            <div class="progress-bg">
                <div class="progress-fill" style="width:{porcentajes.get("Neutro", 0)}%; background:#b49b57;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_neg:
        st.markdown(f"""
        <div class="metric-box">
            <div class="sentiment-label" style="color:#9b5c5c;">Negativas</div>
            <div class="sentiment-value">{porcentajes.get("Negativo", 0):.1f}%</div>
            <div class="sentiment-count">{conteo.get("Negativo", 0)} reviews</div>
            <div class="progress-bg">
                <div class="progress-fill" style="width:{porcentajes.get("Negativo", 0)}%; background:#9b5c5c;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- ASPECTOS ----------------

    positivas = producto_df[producto_df["sentiment"] == "Positivo"]
    negativas = producto_df[producto_df["sentiment"] == "Negativo"]

    aspectos_positivos = obtener_aspectos(positivas[texto_col], n=5)
    aspectos_negativos = obtener_aspectos(negativas[texto_col], n=5)

    col_aspectos_pos, col_aspectos_neg = st.columns(2)

    with col_aspectos_pos:
        st.markdown("""
        <div class="card">
            <div class="section-title">Aspectos mejor valorados</div>
            <div class="section-subtitle">Temas más mencionados en opiniones positivas</div>
        """, unsafe_allow_html=True)

        barra_aspectos(aspectos_positivos, "#3f6f64")

        st.markdown("</div>", unsafe_allow_html=True)

    with col_aspectos_neg:
        st.markdown("""
        <div class="card">
            <div class="section-title">Aspectos peor valorados</div>
            <div class="section-subtitle">Temas más mencionados en opiniones negativas</div>
        """, unsafe_allow_html=True)

        barra_aspectos(aspectos_negativos, "#9b5c5c")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- DONUT E INFORMACIÓN ----------------

    col_grafico, col_info = st.columns(2)

    with col_grafico:
        st.markdown("""
        <div class="card">
            <div class="section-title">Sentimiento</div>
            <div class="section-subtitle">Distribución porcentual</div>
        """, unsafe_allow_html=True)

        orden = ["Positivo", "Neutro", "Negativo"]
        valores = [conteo.get(x, 0) for x in orden]

        colores = ["#3f6f64", "#b49b57", "#9b5c5c"]

        fig, ax = plt.subplots(figsize=(2.8, 2.8))

        wedges, texts, autotexts = ax.pie(
            valores,
            labels=None,
            autopct=lambda pct: f"{pct:.0f}%" if pct >= 5 else "",
            startangle=90,
            colors=colores,
            pctdistance=0.76,
            wedgeprops={
                "width": 0.34,
                "edgecolor": "white",
                "linewidth": 2
            },
            textprops={
                "fontsize": 8,
                "color": "#111827",
                "fontweight": "bold"
            }
        )

        ax.text(
            0,
            0,
            f"{len(producto_df)}\nreviews",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#0f172a"
        )

        ax.legend(
            wedges,
            orden,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=3,
            frameon=False,
            fontsize=7.5
        )

        ax.set(aspect="equal")
        plt.tight_layout()

        st.pyplot(fig, use_container_width=False)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div class="card" style="min-height: 415px;">
            <div class="section-title">Información del análisis</div>
            <div class="section-subtitle">Configuración utilizada en el procesamiento</div>
            <div class="info-table">
                <b>Modelo utilizado:</b> LinearSVC con TF-IDF<br><br>
                <b>Clasificación:</b> Positivo / Neutro / Negativo<br><br>
                <b>Extracción de aspectos:</b> spaCy<br><br>
                <b>Identificador de producto:</b> parent_asin<br><br>
                <b>Dataset base:</b> Amazon Reviews 2023
            </div>
        </div>
        """, unsafe_allow_html=True)


    # ---------------- RESUMEN EJECUTIVO ----------------

    resumen = generar_resumen_ejecutivo(
        producto,
        porcentajes,
        aspectos_positivos,
        aspectos_negativos,
        len(producto_df)
    )

    st.markdown(f"""
    <div class="summary-box">
        <div class="section-title">Resumen ejecutivo</div>
        <p>{resumen}</p>
    </div>
    """, unsafe_allow_html=True)